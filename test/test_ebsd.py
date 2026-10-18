import warnings

import ase
import numpy as np
import pytest

import abtem
import abtem.ebsd.reciprocity as reciprocity
import abtem.ebsd.reference as reference_module
from abtem.core.axes import EnergyAxis, OrdinalAxis
from abtem.core.energy import energy2wavelength
from abtem.ebsd import (
    EBSD,
    AntialiasLossWarning,
    BackscatterDetector,
    EBSDDetectorPattern,
    EBSDGeometry,
    EBSDPatternImages,
    SparseProjectionWarning,
    SphericalPattern,
    SquareLambertProjection,
    StereographicProjection,
    bin_directions,
    bulk_block,
    bunge_rotation,
    central_origin,
    estimate_repetitions,
    fibonacci_hemisphere,
    pixel_centers,
    rotated_slab,
    validate_projection,
    write_emsoft_master_pattern,
    zone_axis_rotation,
)
from abtem.ebsd.emission import EmissionSlices, backscatter_cross_section
from abtem.ebsd.orientations import (
    _crystal_and_displacements,
    _displaced,
    is_centrosymmetric,
)
from abtem.ebsd.reciprocity import (
    _validate_backscatter_energy,
    _validate_depth_weight,
)
from abtem.ebsd.reference import (
    EBSDReferencePattern,
    fft_friendly_gpts,
    patch_half_angle,
)
from abtem.ebsd.sampling import potential_sampling, scattering_power_lost

# The calculations here run on grids far coarser than a converged pattern
# needs, to be quick, and the constructors warn about that; the tests of the
# warning itself catch it with pytest.warns.
pytestmark = pytest.mark.filterwarnings(
    "ignore::abtem.ebsd.sampling.AntialiasLossWarning"
)


@pytest.fixture(autouse=True)
def eager_by_default():
    """Run eagerly unless a test asks otherwise.

    EBSD.scan and EBSDReferencePattern.build follow abTEM's ``dask.lazy``
    setting, which ships as True. Most tests here want the numbers rather than
    a graph, so pin the setting and let the lazy tests pass ``lazy=True``.
    """
    with abtem.config.set({"dask.lazy": False}):
        yield


PROJECTIONS = [StereographicProjection(), SquareLambertProjection()]
PROJECTION_IDS = [p.name for p in PROJECTIONS]


def random_hemisphere_directions(n, seed=0):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions[:, 2] = np.abs(directions[:, 2])
    return directions


class TestProjections:
    @pytest.mark.parametrize("projection", PROJECTIONS, ids=PROJECTION_IDS)
    def test_round_trip(self, projection):
        directions = random_hemisphere_directions(500)
        assert np.allclose(
            projection.unproject(projection.project(directions)), directions
        )

    @pytest.mark.parametrize("projection", PROJECTIONS, ids=PROJECTION_IDS)
    def test_unproject_gives_unit_northern_vectors(self, projection):
        grid = projection.grid(64)
        assert np.allclose(np.linalg.norm(grid, axis=1), 1.0)
        assert np.all(grid[:, 2] >= -1e-12)

    @pytest.mark.parametrize("projection", PROJECTIONS, ids=PROJECTION_IDS)
    def test_projects_into_the_unit_square(self, projection):
        xy = projection.project(random_hemisphere_directions(500))
        assert np.all(np.abs(xy) <= 1.0 + 1e-12)

    @pytest.mark.parametrize("projection", PROJECTIONS, ids=PROJECTION_IDS)
    def test_pole_maps_to_the_origin(self, projection):
        xy = projection.project(np.array([[0.0, 0.0, 1.0]]))
        assert np.allclose(xy, 0.0)

    def test_stereographic_equator_maps_to_the_unit_circle(self):
        theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        equator = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
        xy = StereographicProjection().project(equator)
        assert np.allclose(np.linalg.norm(xy, axis=1), 1.0)

    def test_lambert_is_equal_area(self):
        # An equal-area projection sends a uniform distribution on the
        # hemisphere to a uniform distribution on the square, so every pixel of
        # a coarse binning should receive the same count to within noise.
        directions = random_hemisphere_directions(400_000, seed=3)
        counts = bin_directions(
            directions, np.ones(len(directions)), gpts=1, projection="lambert"
        )
        assert counts.shape == (1, 1)

        xy = SquareLambertProjection().project(directions)
        counts, _, _ = np.histogram2d(
            xy[:, 0], xy[:, 1], bins=8, range=[[-1, 1], [-1, 1]]
        )
        assert counts.std() / counts.mean() < 0.02

    def test_stereographic_grid_fills_the_inscribed_disk_only(self):
        grid = StereographicProjection().grid(64)
        # pi / 4 of the square's points lie in the inscribed disk
        assert 0.75 < len(grid) / 64**2 < 0.81

    def test_lambert_grid_fills_the_whole_square(self):
        assert len(SquareLambertProjection().grid(64)) == 64**2

    def test_validate_projection(self):
        assert isinstance(validate_projection("lambert"), SquareLambertProjection)
        instance = StereographicProjection()
        assert validate_projection(instance) is instance
        with pytest.raises(ValueError, match="projection must be one of"):
            validate_projection("gnomonic")


class TestBinDirections:
    def test_averages_within_a_pixel(self):
        directions = np.array([[0.0, 0.0, 1.0], [1e-6, 1e-6, 1.0]])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        image = bin_directions(directions, np.array([1.0, 3.0]), gpts=2)
        assert image.sum() == pytest.approx(2.0)

    def test_empty_pixels_are_zero(self):
        directions = np.array([[0.0, 0.0, 1.0]])
        image = bin_directions(directions, np.array([5.0]), gpts=8)
        assert (image != 0).sum() == 1
        assert image.max() == pytest.approx(5.0)

    def test_southern_directions_are_dropped(self):
        directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        image = bin_directions(directions, np.array([2.0, 100.0]), gpts=4)
        assert image.max() == pytest.approx(2.0)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="values has length"):
            bin_directions(np.zeros((3, 3)), np.zeros(2), gpts=4)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
            bin_directions(np.zeros((3, 2)), np.zeros(3), gpts=4)


class TestFibonacciHemisphere:
    def test_unit_vectors_on_the_northern_hemisphere(self):
        directions = fibonacci_hemisphere(500)
        assert directions.shape == (500, 3)
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
        assert np.all(directions[:, 2] > 0.0)

    def test_approximately_equal_area(self):
        # z is uniformly spaced, and equal spacing in z is equal solid angle.
        z = fibonacci_hemisphere(1000)[:, 2]
        assert np.allclose(np.diff(z), np.diff(z)[0])

    def test_rejects_non_positive_n(self):
        with pytest.raises(ValueError, match="n must be at least 1"):
            fibonacci_hemisphere(0)


class TestZoneAxisRotation:
    @pytest.mark.parametrize(
        "zone_axis",
        [(0, 0, 1), (1, 1, 1), (1, 0, 0), (2, 2, 3), (-1, 2, -3)],
    )
    def test_maps_the_zone_axis_onto_z(self, zone_axis):
        rotation = zone_axis_rotation(np.array(zone_axis, dtype=float))
        unit = np.array(zone_axis, dtype=float)
        unit /= np.linalg.norm(unit)
        assert np.allclose(rotation @ unit, [0.0, 0.0, 1.0])

    @pytest.mark.parametrize(
        "zone_axis", [(0, 0, 1), (1, 1, 1), (1, 0, 0), (2, 2, 3), (-1, 2, -3)]
    )
    def test_is_a_proper_rotation(self, zone_axis):
        rotation = zone_axis_rotation(np.array(zone_axis, dtype=float))
        assert np.allclose(rotation @ rotation.T, np.eye(3))
        assert np.linalg.det(rotation) == pytest.approx(1.0)

    @pytest.mark.parametrize("zone_axis", [(1, 1, 1), (2, 2, 3), (-1, 2, -3)])
    def test_matches_ase(self, zone_axis):
        # rotated_slab relies on this agreement: it rotates the atoms with ASE
        # but returns this matrix for transforming directions.
        unit = np.array(zone_axis, dtype=float)
        unit /= np.linalg.norm(unit)

        atoms = ase.Atoms("H", positions=[(0, 0, 0)], cell=np.eye(3))
        atoms.rotate(unit, "z", rotate_cell=True)
        ase_rotation = np.linalg.inv(np.eye(3)) @ atoms.cell[:]

        assert np.allclose(ase_rotation, zone_axis_rotation(unit).T)

    def test_rejects_zero_and_wrong_shape(self):
        with pytest.raises(ValueError, match="must be nonzero"):
            zone_axis_rotation(np.zeros(3))
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            zone_axis_rotation(np.zeros(2))


class TestRotatedSlab:
    @pytest.fixture
    def silicon(self):
        return ase.build.bulk("Si", "diamond", a=5.431)

    def test_cell_matches_the_request(self, silicon):
        cell = (10.0, 12.0, 20.0)
        slab, _ = rotated_slab(silicon, np.array([1.0, 1.0, 1.0]), cell)
        assert np.allclose(slab.cell.lengths(), cell)

    def test_atoms_lie_inside_the_cell(self, silicon):
        cell = (10.0, 10.0, 20.0)
        slab, _ = rotated_slab(silicon, np.array([1.0, 2.0, 3.0]), cell)
        assert len(slab) > 0
        assert np.all(slab.positions >= 0.0)
        assert np.all(slab.positions <= np.array(cell))

    def test_density_is_preserved(self, silicon):
        # A slab this small holds a whole number of atomic planes, so its count
        # depends on where the box falls against them -- along [111] by about
        # 13%, one double layer in eight. Averaged over where the cut is made,
        # it has to come out at the density.
        cell = (12.0, 12.0, 24.0)
        rng = np.random.default_rng(0)
        counts = [
            len(
                rotated_slab(
                    silicon,
                    np.array([1.0, 1.0, 1.0]),
                    cell,
                    origin=rng.random(3) @ silicon.cell.array,
                )[0]
            )
            for _ in range(24)
        ]
        expected = len(silicon) / silicon.get_volume() * float(np.prod(cell))
        assert np.mean(counts) == pytest.approx(expected, rel=0.03)

    def test_rotation_takes_the_zone_axis_to_the_beam_direction(self, silicon):
        zone_axis = np.array([2.0, 2.0, 3.0])
        _, rotation = rotated_slab(silicon, zone_axis, (10.0, 10.0, 10.0))
        unit = zone_axis / np.linalg.norm(zone_axis)
        assert np.allclose(rotation @ unit, [0.0, 0.0, 1.0])

    @pytest.mark.parametrize(
        "atoms",
        [
            ase.build.bulk("Si", "diamond", a=5.431),  # fcc primitive, 60 degrees
            ase.build.bulk("Si", "diamond", a=5.431, cubic=True),
            ase.build.bulk("Mg", "hcp", a=3.21, c=5.21),  # 120 degrees
            ase.Atoms(
                "Cu",
                cell=[[4.0, 0.0, 0.0], [1.5, 3.5, 0.0], [0.8, 1.1, 3.9]],
                pbc=True,
            ),
        ],
        ids=["fcc-primitive", "cubic", "hcp", "triclinic"],
    )
    def test_the_block_contains_the_slab_at_any_orientation(self, atoms):
        # The slab is cut about the centroid of the block, whichever way it is
        # turned, so the block must hold a ball of the slab's diagonal about
        # that point. What bounds the ball is the separation of opposite
        # faces, which a non-orthogonal cell makes shorter than its edges;
        # sizing on the edges once gave silicon slabs with vacuum in them.
        cell = (12.0, 12.0, 30.0)
        block = atoms * estimate_repetitions(atoms, cell)

        lattice = np.asarray(block.cell.array)
        volume = abs(np.linalg.det(lattice))
        heights = np.array(
            [
                volume
                / np.linalg.norm(np.cross(lattice[(i + 1) % 3], lattice[(i + 2) % 3]))
                for i in range(3)
            ]
        )
        fractional = np.linalg.solve(lattice.T, block.positions.mean(axis=0))
        clearance = np.minimum(fractional, 1.0 - fractional) * heights

        assert np.all(clearance >= np.linalg.norm(cell) / 2.0)

    def test_rejects_wrong_cell_shape(self, silicon):
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            rotated_slab(silicon, np.array([0.0, 0.0, 1.0]), (10.0, 10.0))

    def test_a_reused_block_gives_the_same_slab(self, silicon):
        # Building the block is the expensive part, so a reference pattern
        # builds it once and cuts every zone axis out of it. That must give
        # exactly what repeating per call does.
        cell = (10.0, 10.0, 20.0)
        zone_axis = np.array([1.0, 2.0, 3.0])

        block = bulk_block(silicon, cell, (12, 12, 12))
        reused, _ = rotated_slab(block, zone_axis, cell, repetitions=(1, 1, 1))
        rebuilt, _ = rotated_slab(silicon, zone_axis, cell, repetitions=(12, 12, 12))

        assert np.array_equal(reused.positions, rebuilt.positions)
        assert np.array_equal(reused.numbers, rebuilt.numbers)

    def test_a_reused_block_is_not_mutated(self, silicon):
        # rotated_slab rotates and recentres in place, so it has to copy.
        cell = (10.0, 10.0, 20.0)
        block = bulk_block(silicon, cell, (12, 12, 12))
        before = block.positions.copy()

        rotated_slab(block, np.array([1.0, 1.0, 1.0]), cell, repetitions=(1, 1, 1))

        assert np.array_equal(block.positions, before)

    def test_bulk_block_defaults_to_the_estimated_repetitions(self, silicon):
        cell = (10.0, 10.0, 20.0)
        block = bulk_block(silicon, cell)
        expected = silicon * estimate_repetitions(silicon, cell)
        assert len(block) == len(expected)


def empty_fraction(slab, cell, voxel=4.0):
    """Fraction of a slab's voxels holding no atom: vacuum where crystal should be."""
    bins = [max(1, int(round(c / voxel))) for c in cell]
    counts, _ = np.histogramdd(
        slab.positions, bins=bins, range=[(0.0, c) for c in cell]
    )
    return float(np.mean(counts == 0))


class TestSlabCompleteness:
    """Every patch of a reference pattern needs a whole slab of crystal.

    A slab missing atoms in some orientations and not others makes neighbouring
    patches disagree, which shows up as bands broken at the patch boundaries.
    """

    @pytest.fixture
    def silicon(self):
        return ase.build.bulk("Si", "diamond", a=5.431)

    @pytest.mark.parametrize("cell", [(10.0, 10.0, 40.0), (40.0, 40.0, 100.0)])
    def test_no_orientation_leaves_vacuum_in_the_slab(self, silicon, cell):
        # 40 x 40 x 100 is the size the reference patterns were validated at,
        # and the one that sizing the block on edge lengths got wrong: 123 of
        # 200 patches came out with vacuum at their corners and ends.
        block = bulk_block(silicon, cell)
        expected = len(silicon) / silicon.get_volume() * float(np.prod(cell))

        for zone_axis in fibonacci_hemisphere(60):
            slab, _ = rotated_slab(block, zone_axis, cell, repetitions=(1, 1, 1))
            assert empty_fraction(slab, cell) == 0.0, zone_axis
            if cell[0] >= 40.0:
                # large enough that the count is set by the density alone
                # (one (111) double layer of this slab is 3% of it)
                assert len(slab) == pytest.approx(expected, rel=0.04), zone_axis

    def test_an_origin_in_a_repeated_cell_gives_a_whole_slab(self, silicon):
        # A point of the unrepeated cell is the corner of the block; cut about
        # it as given, the slab was 85% empty.
        cell = (10.0, 10.0, 40.0)
        slab, _ = rotated_slab(
            silicon, np.array([1.0, 0.0, 1.0]), cell, origin=np.zeros(3)
        )

        assert empty_fraction(slab, cell) == 0.0
        # and it is still the requested point: the atom at the origin of the
        # cell sits at the centre of the slab
        centre = np.array(cell) / 2.0
        assert np.min(np.linalg.norm(slab.positions - centre, axis=1)) < 1e-6

    def test_central_origin_moves_by_whole_lattice_vectors(self, silicon):
        repetitions = (15, 15, 15)
        point = np.array([0.3, -0.2, 0.1])
        moved = central_origin(silicon, repetitions, point)

        shift = np.linalg.solve(silicon.cell.array.T, moved - point)
        assert np.allclose(shift, np.round(shift))

        block = silicon * repetitions
        fractional = np.linalg.solve(block.cell.array.T, moved)
        assert np.all(np.abs(fractional - 0.5) <= 0.5 / np.array(repetitions) + 1e-9)

    def test_central_origin_leaves_a_single_cell_alone(self, silicon):
        # an atomic model holding one feature is not repeated, and a point of
        # it must stay where it is
        point = 0.4 * silicon.cell.array.sum(axis=0)
        assert np.allclose(central_origin(silicon, (1, 1, 1), point), point)

    def test_the_reference_builder_centres_the_origin(self):
        # EBSDReferencePattern builds its own block, so it has to move the
        # origin itself; an origin at the corner of the block would have left
        # every patch nearly empty, and the yield collapsing towards vacuum.
        def build(origin):
            return EBSDReferencePattern(
                ase.build.bulk("Si", "diamond", a=5.431),
                30e3,
                n_patches=4,
                slab_cell=(6.0, 6.0, 4.0),
                gpts=8,
                direction_gpts=30,
                max_angle=200.0,
                origin=origin,
            ).build(pbar=False, lazy=False)

        centroid = np.asarray(build(None).array).mean()
        corner = np.asarray(build(np.zeros(3)).array).mean()

        assert corner == pytest.approx(centroid, rel=0.25)


class TestBackscatterDetector:
    def test_grid_directions_are_unit_vectors_into_the_crystal(self):
        detector = BackscatterDetector(max_angle=200, gpts=9)
        assert detector.directions.shape == (81, 3)
        assert np.allclose(np.linalg.norm(detector.directions, axis=1), 1.0)
        assert np.all(detector.directions[:, 2] > 0.0)

    def test_grid_is_centred_on_the_beam_direction(self):
        detector = BackscatterDetector(max_angle=200, gpts=9)
        centre = detector.directions.reshape(9, 9, 3)[9 // 2, 9 // 2]
        assert np.allclose(centre, [0.0, 0.0, 1.0])

    def test_grid_half_width_matches_max_angle(self):
        detector = BackscatterDetector(max_angle=200, gpts=64)
        transverse = np.abs(detector.directions[:, :2]).max()
        assert transverse == pytest.approx(np.sin(0.2), rel=0.05)

    def test_base_shape(self):
        assert BackscatterDetector(max_angle=100, gpts=7).base_shape == (7, 7)
        directions = random_hemisphere_directions(5)
        assert BackscatterDetector(directions=directions).base_shape == (5,)

    def test_explicit_directions_are_normalized(self):
        detector = BackscatterDetector(directions=np.array([[0.0, 0.0, 2.0]]))
        assert np.allclose(detector.directions, [[0.0, 0.0, 1.0]])

    def test_transverse_wave_vectors_scale_with_the_wave_number(self):
        from abtem.core.energy import energy2wavelength

        detector = BackscatterDetector(max_angle=100, gpts=5)
        for energy in (30e3, 200e3):
            expected = detector.directions[:, :2] / energy2wavelength(energy)
            assert np.allclose(detector.transverse_wave_vectors(energy), expected)

    def test_reciprocal_sampling_spans_the_grid(self):
        detector = BackscatterDetector(max_angle=100, gpts=16)
        vectors = detector.transverse_wave_vectors(30e3)
        step = np.diff(np.unique(np.round(vectors[:, 0], 9)))
        assert np.allclose(step, detector.reciprocal_sampling(30e3))

    def test_reciprocal_sampling_rejects_explicit_directions(self):
        detector = BackscatterDetector(directions=random_hemisphere_directions(4))
        with pytest.raises(RuntimeError, match="no regular sampling"):
            detector.reciprocal_sampling(30e3)

    def test_is_grid(self):
        assert BackscatterDetector(max_angle=100, gpts=4).is_grid
        assert not BackscatterDetector(
            directions=random_hemisphere_directions(4)
        ).is_grid

    def test_warns_when_the_small_angle_approximation_fails(self):
        with pytest.warns(UserWarning, match="small-angle approximation"):
            BackscatterDetector(max_angle=700, gpts=5)

    def test_rejects_backward_directions(self):
        with pytest.raises(ValueError, match="positive z component"):
            BackscatterDetector(directions=np.array([[0.0, 0.0, -1.0]]))

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({}, "give either"),
            ({"max_angle": 100}, "must be given together"),
            ({"gpts": 4}, "must be given together"),
            (
                {"max_angle": 100, "gpts": 4, "directions": np.zeros((1, 3))},
                "not both",
            ),
            ({"directions": np.zeros((3, 2))}, r"shape \(N, 3\)"),
        ],
    )
    def test_rejects_bad_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BackscatterDetector(**kwargs)


class TestSphericalPattern:
    @pytest.fixture
    def pattern(self):
        directions = fibonacci_hemisphere(500)
        return SphericalPattern(
            1.0 + directions[:, 2], directions, metadata={"energy": 30e3}
        )

    def test_project_shape_and_axes(self, pattern):
        # A Fibonacci lattice is not a projection grid, so the binned image has
        # holes; that is what TestSparseProjection covers, not this.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseProjectionWarning)
            images = pattern.bin(32)
        assert images.array.shape == (32, 32)
        assert [axis.units for axis in images.base_axes_metadata] == ["", ""]
        assert images.metadata["projection"] == "stereographic"

    def test_project_preserves_the_ensemble(self):
        directions = fibonacci_hemisphere(200)
        array = np.stack([np.ones(200), 2 * np.ones(200)])
        pattern = SphericalPattern(
            array, directions, ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseProjectionWarning)
            images = pattern.bin(16)
        assert images.array.shape == (2, 16, 16)
        assert images.array[1].max() == pytest.approx(2.0)

    def test_zarr_round_trip(self, pattern, tmp_path):
        url = str(tmp_path / "pattern.zarr")
        pattern.to_zarr(url)
        restored = SphericalPattern.from_zarr(url)
        assert np.array_equal(restored.array, pattern.array)
        assert np.array_equal(restored.directions, pattern.directions)
        assert restored.metadata == pattern.metadata

    def test_concatenate(self, pattern):
        halves = [
            SphericalPattern(pattern.array[:200], pattern.directions[:200]),
            SphericalPattern(pattern.array[200:], pattern.directions[200:]),
        ]
        joined = SphericalPattern.concatenate(halves)
        assert np.array_equal(joined.array, pattern.array)
        assert np.array_equal(joined.directions, pattern.directions)

    def test_concatenate_rejects_mismatched_ensembles(self, pattern):
        other = SphericalPattern(
            np.stack([pattern.array, pattern.array]),
            pattern.directions,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        )
        with pytest.raises(ValueError, match="mismatched ensemble shapes"):
            SphericalPattern.concatenate([pattern, other])

    def test_concatenate_rejects_empty(self):
        with pytest.raises(ValueError, match="empty sequence"):
            SphericalPattern.concatenate([])

    def test_rejects_mismatched_array_and_directions(self):
        with pytest.raises(ValueError, match="last axis of array"):
            SphericalPattern(np.zeros(4), fibonacci_hemisphere(5))

    def test_rejects_wrong_ensemble_axis_count(self):
        with pytest.raises(ValueError, match="ensemble axes"):
            SphericalPattern(np.zeros((2, 5)), fibonacci_hemisphere(5))


def silicon_slab(thickness=8.0, zone_axis=(0.0, 0.0, 1.0)):
    return rotated_slab(
        ase.build.bulk("Si", "diamond", a=5.431),
        np.array(zone_axis),
        (10.0, 10.0, thickness),
    )[0]


LIGHTINGS = ["uniform", "illumination"]


def lighting(name, atoms):
    """No illumination, or a random but fixed one for `atoms`."""
    if name == "illumination":
        return np.random.default_rng(0).uniform(0.5, 1.5, size=len(atoms))
    return None


def make_ebsd(atoms, detector, sampling=0.15, slice_thickness=1.0, **kwargs):
    """EBSD of `atoms` at 30 keV."""
    potential = abtem.Potential(
        atoms, sampling=sampling, slice_thickness=slice_thickness, projection="finite"
    )
    return EBSD(potential, detector, 30e3, **kwargs)


@pytest.fixture
def featureless(monkeypatch):
    """Atoms that emit, in a specimen that does not scatter.

    The emission is the atoms' own, but the waves travel through a potential of
    zeros sliced the same way, so they stay what they were sent in as.
    """
    prepare = reciprocity._prepare

    def without_scattering(*args, **kwargs):
        potential, emission = prepare(*args, **kwargs)
        vacuum = abtem.PotentialArray(
            np.zeros_like(np.asarray(potential.array)),
            slice_thickness=potential.slice_thickness,
            extent=potential.extent,
        )
        return vacuum, emission

    monkeypatch.setattr(reciprocity, "_prepare", without_scattering)


class TestDepthWeight:
    @pytest.mark.parametrize(
        "weight, expected",
        [
            (None, [0.25] * 4),
            (np.ones(4), [0.25] * 4),
            ([1.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]),
        ],
    )
    def test_normalized_to_sum_to_one(self, weight, expected):
        depths = np.arange(1.0, 5.0)
        assert np.allclose(_validate_depth_weight(weight, depths), expected)

    def test_escape_depth_decays_exponentially(self):
        depths = np.arange(1.0, 5.0)
        weights = _validate_depth_weight(2.0, depths)
        ratios = weights[1:] / weights[:-1]
        assert np.allclose(ratios, np.exp(-0.5))

    def test_callable(self):
        depths = np.arange(1.0, 5.0)
        weights = _validate_depth_weight(lambda z: np.ones_like(z), depths)
        assert np.allclose(weights, 0.25)

    @pytest.mark.parametrize(
        "weight, match",
        [
            (-1.0, "escape depth must be positive"),
            (np.ones(3), "one weight per slice"),
            (np.array([-1.0, 1.0, 1.0, 1.0]), "must not be negative"),
            (np.zeros(4), "must not be all zero"),
        ],
    )
    def test_rejects_bad_weights(self, weight, match):
        with pytest.raises(ValueError, match=match):
            _validate_depth_weight(weight, np.arange(1.0, 5.0))


class TestEmission:
    """Backscatter is generated at the atoms, each by its Z^2."""

    SAMPLING = 0.1

    def emission(self, atoms, **kwargs):
        potential = abtem.Potential(atoms, sampling=self.SAMPLING, slice_thickness=1.0)
        return EmissionSlices(
            atoms,
            potential.slice_thickness,
            potential.gpts,
            potential.extent,
            30e3,
            **kwargs,
        )

    def integral(self, emitted):
        """The emission of a slice integrated over it: per Å², times the area."""
        if emitted is None:
            return 0.0
        return (
            np.asarray(emitted, dtype=np.float64).sum(axis=(-2, -1)) * self.SAMPLING**2
        )

    def test_the_cross_section_is_rutherfords_z_squared(self):
        # The electron scattering factor at q = 2 / wavelength, squared: that
        # far inside the screening radius, the nucleus alone sets it.
        numbers = np.array([6, 14, 29, 79])
        cross_section = backscatter_cross_section(numbers, 30e3)
        assert np.allclose(cross_section, numbers**2, rtol=0.03)

    def test_the_cross_section_is_relative_to_hydrogen(self):
        assert backscatter_cross_section(np.array([1]), 30e3) == pytest.approx([1.0])

    def test_one_cross_section_per_atom_in_order(self):
        cross_section = backscatter_cross_section(np.array([14, 6, 14]), 30e3)
        assert cross_section[0] == cross_section[2] > cross_section[1]

    def test_every_atom_emits_once_in_its_own_slice(self):
        slab = silicon_slab()
        slices = self.emission(slab)
        cross_section = backscatter_cross_section(slab.numbers, 30e3)

        # an emission per unit area, like the potential it is laid out beside
        owner = np.floor(slab.positions[:, 2]).astype(int)
        for index in range(len(slices)):
            assert np.sum(self.integral(slices[index])) == pytest.approx(
                cross_section[owner == index].sum(), rel=1e-4, abs=1e-3
            )

    def test_an_emitter_sits_exactly_at_its_atom(self):
        # Between grid points, where only the phase of its transform shows
        # where it is.
        atoms = ase.Atoms("Si", positions=[(3.37, 6.81, 0.5)], cell=(10.0, 10.0, 1.0))
        emitted = np.asarray(self.emission(atoms)[0][0], dtype=np.float64)
        transform = np.fft.fft2(emitted)
        x = (-np.angle(transform[1, 0]) / (2 * np.pi) * 10.0) % 10.0
        y = (-np.angle(transform[0, 1]) / (2 * np.pi) * 10.0) % 10.0
        assert (x, y) == pytest.approx((3.37, 6.81), abs=1e-4)

    def test_the_illumination_scales_each_atom(self):
        slab = silicon_slab()
        cross_section = backscatter_cross_section(slab.numbers, 30e3)
        illumination = np.random.default_rng(0).uniform(size=(2, len(slab)))
        slices = self.emission(slab, illumination=illumination)
        total = sum(self.integral(slices[i]) for i in range(len(slices)))
        assert np.allclose(total, illumination @ cross_section, rtol=1e-4)

    def test_rejects_an_illumination_of_the_wrong_length(self):
        with pytest.raises(ValueError, match="values per source"):
            self.emission(silicon_slab(), illumination=np.ones(3))


class TestEBSD:
    def test_vacuum_generates_nothing(self):
        # Nothing is there to emit, which must come out as zero, not 0/0.
        vacuum = ase.Atoms(cell=(10.0, 10.0, 8.0), pbc=True)
        patterns = make_ebsd(
            vacuum, BackscatterDetector(max_angle=50, gpts=5), sampling=0.1
        ).build()
        assert np.all(np.isfinite(patterns.array))
        assert np.allclose(patterns.array, 0.0)

    @pytest.mark.parametrize("name", LIGHTINGS)
    def test_a_featureless_specimen_yields_one(self, name, featureless):
        # The definition of the normalization, however the atoms are lit --
        # and in every direction. A plane wave in a direction off the cell's
        # reciprocal grid would jump in phase at the cell edge and diffract off
        # the jump; carried as a periodic envelope it has no edge to jump at.
        slab = silicon_slab()
        patterns = make_ebsd(
            slab,
            BackscatterDetector(max_angle=150, gpts=5),
            illumination=lighting(name, slab),
        ).build()
        assert np.allclose(patterns.array, 1.0, atol=1e-5)

    @pytest.mark.parametrize("name", LIGHTINGS)
    def test_the_first_atoms_see_the_plane_waves_unscattered(self, name):
        # Emission is collected where the multislice puts the atoms: from the
        # waves arriving at their slice, before the step through it. The first
        # atoms the waves meet therefore see them as they were sent in, with
        # unit intensity everywhere, however strongly the specimen scatters.
        # Collected after the step, they saw them already scattered, and the
        # yield depended on the slice thickness.
        slab = silicon_slab()
        detector = BackscatterDetector(max_angle=150, gpts=4)
        occupied = np.unique(np.floor(slab.positions[:, 2]).astype(int))

        def only(index):
            weights = np.zeros(8)
            weights[index] = 1.0
            return make_ebsd(
                slab, detector, illumination=lighting(name, slab), depth_weight=weights
            ).build()

        assert np.allclose(only(occupied[0]).array, 1.0, atol=1e-5)
        # and the deepest, which the waves reach scattered, do not
        assert not np.allclose(only(occupied[-1]).array, 1.0, atol=1e-2)

    def test_the_yield_is_converged_in_the_slicing(self):
        # Along [101] thin slices hold a few atoms each and thick ones many;
        # collected at the wrong plane the yield moved by a quarter between 1
        # and 0.25 Å slices.
        slab, _ = rotated_slab(
            ase.build.bulk("Si", "diamond", a=5.431),
            np.array([1.0, 0.0, 1.0]),
            (10.0, 10.0, 24.0),
        )
        detector = BackscatterDetector(max_angle=50, gpts=6)

        def mean_yield(slice_thickness):
            patterns = make_ebsd(
                slab,
                detector,
                sampling=0.1,
                slice_thickness=slice_thickness,
            ).build()
            return float(np.mean(patterns.array))

        assert mean_yield(1.0) == pytest.approx(mean_yield(0.25), rel=0.03)

    def test_species_emit_by_their_cross_sections(self):
        # The yield is the emission-weighted mean over the atoms, so each
        # species contributes its own pattern in proportion to its total
        # cross-section: silicon about 5.4 times carbon's in SiC.
        slab, _ = rotated_slab(
            ase.build.bulk("SiC", "zincblende", a=4.36),
            np.array([0.0, 0.0, 1.0]),
            (8.0, 8.0, 6.0),
        )
        detector = BackscatterDetector(max_angle=50, gpts=4)
        silicon = (slab.numbers == 14).astype(float)
        carbon = 1.0 - silicon

        both = np.asarray(make_ebsd(slab, detector).build().array)
        per_species = [
            np.asarray(make_ebsd(slab, detector, illumination=lit).build().array)
            for lit in (silicon, carbon)
        ]

        cross_section = backscatter_cross_section(slab.numbers, 30e3)
        weights = np.array([cross_section @ silicon, cross_section @ carbon])
        expected = (weights[0] * per_species[0] + weights[1] * per_species[1]) / (
            weights.sum()
        )
        assert np.allclose(both, expected, rtol=1e-5)
        # and the two species' patterns differ, so the weighting is visible
        assert not np.allclose(per_species[0], per_species[1], rtol=1e-2)

    def test_uniform_depth_weight_matches_the_default(self):
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        default = make_ebsd(atoms, detector).build()
        explicit = make_ebsd(
            atoms, detector, depth_weight=np.ones(int(atoms.cell[2, 2]))
        ).build()
        assert np.allclose(default.array, explicit.array)

    def test_a_long_escape_depth_approaches_uniform_weighting(self):
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        uniform = make_ebsd(atoms, detector).build()
        deep = make_ebsd(atoms, detector, depth_weight=1e6).build()
        assert np.allclose(uniform.array, deep.array, rtol=1e-4)

    def test_a_short_escape_depth_weights_the_surface(self):
        # A shallow escape depth must give a different answer from a uniform
        # one; otherwise the weighting is not reaching the accumulation.
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        uniform = make_ebsd(atoms, detector).build()
        shallow = make_ebsd(atoms, detector, depth_weight=2.0).build()
        assert not np.allclose(uniform.array, shallow.array, rtol=1e-3)

    def test_batching_does_not_change_the_result(self):
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=6)
        whole = make_ebsd(atoms, detector).build(max_batch_directions=10_000)
        split = make_ebsd(atoms, detector).build(max_batch_directions=4)
        assert np.allclose(whole.array, split.array, rtol=1e-5)

    def test_grid_detector_returns_diffraction_patterns(self):
        patterns = make_ebsd(
            silicon_slab(), BackscatterDetector(max_angle=50, gpts=8)
        ).build()
        assert isinstance(patterns, abtem.DiffractionPatterns)
        assert patterns.array.shape == (8, 8)
        assert patterns.metadata["energy"] == 30e3

    def test_explicit_directions_reproduce_the_equivalent_grid(self):
        # The two ways of building a detector describe the same directions, so
        # they must give the same intensities -- only the measurement type and
        # its shape differ.
        atoms = silicon_slab()
        grid = BackscatterDetector(max_angle=50, gpts=4)
        explicit = BackscatterDetector(directions=grid.directions)

        from_grid = make_ebsd(atoms, grid).build()
        from_explicit = make_ebsd(atoms, explicit).build()

        assert isinstance(from_explicit, SphericalPattern)
        assert from_explicit.array.shape == (16,)
        assert np.allclose(from_explicit.directions, grid.directions)
        assert np.allclose(from_grid.array.ravel(), from_explicit.array, rtol=1e-6)

    @pytest.mark.parametrize("lazy", [False, True])
    def test_frozen_phonon_configurations_are_averaged(self, lazy):
        # A potential built on an ensemble carries a configuration axis that
        # the propagation cannot see, so it has to be taken apart before
        # generate_slices, which would otherwise walk the first configuration
        # and silently return it alone. Each configuration's atoms emit from
        # where that configuration put them.
        rng = np.random.default_rng(0)
        trajectory = []
        for _ in range(3):
            configuration = silicon_slab()
            configuration.positions += rng.normal(
                scale=0.2, size=configuration.positions.shape
            )
            trajectory.append(configuration)

        detector = BackscatterDetector(max_angle=50, gpts=4)
        separately = [
            np.asarray(make_ebsd(atoms, detector).build().array) for atoms in trajectory
        ]

        ensemble = make_ebsd(abtem.AtomsEnsemble(trajectory), detector).build(lazy=lazy)
        if lazy:
            ensemble = ensemble.compute(progress_bar=False)

        assert ensemble.array.shape == (4, 4)
        assert np.allclose(ensemble.array, np.mean(separately, axis=0), rtol=1e-5)
        # and it is an average, not the first configuration
        assert not np.allclose(ensemble.array, separately[0], rtol=1e-3)

    def test_lights_every_atom_alike_by_default(self):
        ebsd = make_ebsd(silicon_slab(), BackscatterDetector(max_angle=50, gpts=4))
        assert ebsd.illumination is None
        assert ebsd.build().metadata["source"] == "uniform"

    def test_needs_an_energy(self):
        with pytest.raises(TypeError, match="energy"):
            EBSD(silicon_slab(), BackscatterDetector(max_angle=50, gpts=4))

    def test_rejects_an_illumination_of_the_wrong_shape(self):
        with pytest.raises(ValueError, match=r"shape \(atoms,\) or \(sources, atoms\)"):
            make_ebsd(
                silicon_slab(),
                BackscatterDetector(max_angle=50, gpts=4),
                illumination=np.ones((2, 2, 2)),
            )

    def test_refuses_a_potential_without_its_atoms(self):
        # The atoms are what emits, and a built potential no longer has them.
        potential = abtem.Potential(
            silicon_slab(), sampling=0.15, slice_thickness=1.0, projection="finite"
        )
        with pytest.raises(ValueError, match="does not carry its atoms"):
            EBSD(
                potential.build(lazy=False),
                BackscatterDetector(max_angle=50, gpts=4),
                energy=30e3,
            )

    def test_an_even_illumination_is_the_uniform_source(self):
        # at any brightness: only how the atoms are lit relative to each
        # other matters
        slab = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        uniform = make_ebsd(slab, detector).build()
        even = make_ebsd(slab, detector, illumination=np.full(len(slab), 3.7)).build()
        assert even.metadata["source"] == "illumination"
        assert np.allclose(even.array, uniform.array, rtol=1e-5)

    def test_several_illuminations_add_a_leading_axis(self):
        slab = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        illumination = np.random.default_rng(0).uniform(size=(3, len(slab)))

        together = make_ebsd(slab, detector, illumination=illumination).build()

        assert together.array.shape == (3, 4, 4)
        assert together.ensemble_axes_metadata[0].label == "illumination"
        for i in range(3):
            alone = make_ebsd(slab, detector, illumination=illumination[i]).build()
            assert np.allclose(together.array[i], alone.array, rtol=1e-5)

    def test_rejects_an_illumination_of_the_wrong_length(self):
        with pytest.raises(ValueError, match="one value per atom"):
            make_ebsd(
                silicon_slab(),
                BackscatterDetector(max_angle=50, gpts=4),
                illumination=np.ones(3),
            )

    def test_an_ensemble_kept_separate_warns_that_it_is_averaged(self):
        trajectory = [silicon_slab(), silicon_slab()]
        trajectory[1].positions += 0.2

        with pytest.warns(UserWarning, match="incoherent sum"):
            make_ebsd(
                abtem.AtomsEnsemble(trajectory, ensemble_mean=False),
                BackscatterDetector(max_angle=50, gpts=4),
            ).build()

    def test_yield_is_of_order_one(self):
        # The normalization is relative to a featureless specimen, so a real
        # one should land near unity rather than at an arbitrary scale.
        patterns = make_ebsd(
            silicon_slab(), BackscatterDetector(max_angle=50, gpts=8)
        ).build()
        assert 0.5 < patterns.array.mean() < 2.0

    def test_warns_when_the_sampling_cannot_resolve_the_atoms(self):
        # The antialias aperture cuts the potential's scattering beyond it, and
        # silently: the transmission function is band-limited before it acts.
        with pytest.warns(AntialiasLossWarning, match="scattering power"):
            make_ebsd(
                silicon_slab(), BackscatterDetector(max_angle=50, gpts=4), sampling=0.2
            )

    def test_a_sampling_that_resolves_the_atoms_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", AntialiasLossWarning)
            make_ebsd(
                silicon_slab(),
                BackscatterDetector(max_angle=50, gpts=4),
                sampling=potential_sampling(silicon_slab()),
            )

    def test_steep_waves_are_not_cut_off_by_the_aperture(self):
        # A wave's direction uses up none of the aperture, which limits only
        # the scattering about it. Carried as plane waves on this grid, the
        # waves at 250 and 300 mrad lay outside it, and their yield collapsed
        # to 0.19 and 0.18.
        angles = np.array([0.0, 150.0, 250.0, 300.0]) * 1e-3
        directions = np.column_stack(
            [np.sin(angles), np.zeros_like(angles), np.cos(angles)]
        )
        patterns = make_ebsd(
            silicon_slab(), BackscatterDetector(directions=directions), sampling=0.1
        ).build()
        assert np.all((0.8 < patterns.array) & (patterns.array < 1.5))


class TestEnvelopePropagator:
    """Exact propagation of the waves' periodic envelopes."""

    GPTS, SAMPLING, THICKNESS = (48, 48), (0.1, 0.1), 1.5

    def kernel(self, wave_vectors):
        from abtem.ebsd.reciprocity import _envelope_propagator_array

        return _envelope_propagator_array(
            np.asarray(wave_vectors, dtype=np.float32),
            self.GPTS,
            self.SAMPLING,
            30e3,
            self.THICKNESS,
            np,
        )

    def test_leaves_a_plane_wave_alone(self):
        k0 = 1 / energy2wavelength(30e3)
        kernel = self.kernel([[0.0, 0.0], [0.1 * k0, 0.0], [0.2 * k0, -0.15 * k0]])
        assert np.allclose(kernel[:, 0, 0], 1.0)
        assert np.all(np.abs(kernel) <= 1.0 + 1e-6)

    def test_along_the_axis_is_abtems_propagator(self):
        from abtem.antialias import antialias_aperture
        from abtem.multislice import _fresnel_propagator_array

        expected = _fresnel_propagator_array(
            self.THICKNESS, self.GPTS, self.SAMPLING, 30e3, "cpu", order="exact"
        ) * antialias_aperture(self.GPTS, self.SAMPLING, np)
        assert np.allclose(self.kernel([[0.0, 0.0]])[0], expected, atol=1e-5)

    def test_tilted_is_abtems_propagator_shifted(self):
        # For a tilt k on the grid, the envelope's component q propagates as
        # the wave's component q + k does, relative to the carrier's phase.
        from abtem.antialias import antialias_aperture
        from abtem.multislice import _fresnel_propagator_array

        shift = 3
        k = shift / (self.GPTS[0] * self.SAMPLING[0])
        plain = _fresnel_propagator_array(
            self.THICKNESS, self.GPTS, self.SAMPLING, 30e3, "cpu", order="exact"
        )
        aperture = antialias_aperture(self.GPTS, self.SAMPLING, np)
        kernel = self.kernel([[k, 0.0]])[0]

        rows = np.arange(-8, 9)  # small enough that q + k does not wrap around
        expected = plain[(rows + shift) % self.GPTS[0]] / plain[shift, 0]
        assert np.allclose(
            kernel[rows % self.GPTS[0]],
            expected * aperture[rows % self.GPTS[0]],
            atol=1e-5,
        )


class TestPatchGeometry:
    def test_half_angle_shrinks_as_patches_are_added(self):
        assert patch_half_angle(100) > patch_half_angle(400)
        # solid angle per patch goes as 1 / n, so the radius goes as 1 / sqrt(n)
        assert patch_half_angle(100) / patch_half_angle(400) == pytest.approx(2.0)

    def test_half_angle_covers_the_hemisphere(self):
        # The discs must at least cover the hemisphere: n * (1 - cos(a)) >= 1,
        # since a cap of half-angle a subtends 2*pi*(1 - cos a) of the 2*pi.
        for n_patches in (50, 200, 400, 1000):
            half_angle = patch_half_angle(n_patches) * 1e-3
            assert n_patches * (1.0 - np.cos(half_angle)) > 1.0

    def test_half_angle_rejects_non_positive(self):
        with pytest.raises(ValueError, match="n_patches must be at least 1"):
            patch_half_angle(0)


class TestEBSDReferencePattern:
    @pytest.fixture
    def builder(self):
        return EBSDReferencePattern(
            ase.build.bulk("Si", "diamond", a=5.431),
            energy=30e3,
            n_patches=1,
            slab_cell=(6.0, 6.0, 4.0),
            gpts=8,
            direction_gpts=30,
            max_angle=200.0,
        )

    def test_the_slab_has_to_be_given(self):
        # The right size depends on the crystal and the accuracy wanted, and a
        # default is how an undersized slab goes unnoticed.
        with pytest.raises(TypeError, match="slab_cell"):
            EBSDReferencePattern(
                ase.build.bulk("Si", "diamond", a=5.431),
                energy=30e3,
            )

    @pytest.mark.parametrize("slab_cell", [(10.0, 10.0), (10.0, -10.0, 40.0)])
    def test_rejects_a_bad_slab(self, slab_cell):
        with pytest.raises(ValueError, match="three positive lengths"):
            EBSDReferencePattern(
                ase.build.bulk("Si", "diamond", a=5.431),
                energy=30e3,
                slab_cell=slab_cell,
            )

    def test_the_atoms_are_lit_evenly(self, builder):
        assert builder.energy == 30e3
        assert builder.build(pbar=False, lazy=False).metadata["source"] == "uniform"

    def test_defaults_are_derived_from_the_patch_count(self):
        builder = EBSDReferencePattern(
            ase.build.bulk("Si", "diamond", a=5.431),
            energy=30e3,
            slab_cell=(20.0, 20.0, 100.0),
            n_patches=400,
            gpts=128,
        )
        assert builder.max_angle == pytest.approx(patch_half_angle(400))
        assert builder.sampling == pytest.approx(
            potential_sampling(ase.build.bulk("Si", "diamond", a=5.431))
        )
        assert len(builder.zone_axes) == 400

    def test_warns_about_too_coarse_a_sampling(self):
        with pytest.warns(AntialiasLossWarning, match="scattering power"):
            EBSDReferencePattern(
                ase.build.bulk("Si", "diamond", a=5.431),
                energy=30e3,
                slab_cell=(20.0, 20.0, 100.0),
                n_patches=400,
                sampling=1.0,
            )

    def test_every_direction_is_assigned_exactly_once(self):
        builder = EBSDReferencePattern(
            ase.build.bulk("Si", "diamond", a=5.431),
            energy=30e3,
            slab_cell=(20.0, 20.0, 100.0),
            n_patches=40,
            gpts=16,
        )
        assignment = builder._assign_directions()
        counts = np.bincount(
            np.concatenate(assignment), minlength=len(builder.directions)
        )
        assert np.all(counts == 1)

    def test_assignment_covers_the_whole_direction_grid(self):
        builder = EBSDReferencePattern(
            ase.build.bulk("Si", "diamond", a=5.431),
            energy=30e3,
            slab_cell=(20.0, 20.0, 100.0),
            n_patches=40,
            gpts=16,
        )
        assigned = np.unique(np.concatenate(builder._assign_directions()))
        assert len(assigned) == len(builder.directions)

    def test_compute_returns_directions_within_the_patch(self, builder):
        pattern = builder.build(pbar=False)
        assert isinstance(pattern, SphericalPattern)
        assert 0 < len(pattern) <= len(builder.directions)

        zone_axis = builder.zone_axes[0]
        cosines = pattern.directions @ zone_axis
        assert np.all(cosines > np.cos(builder.max_angle * 1e-3))

    @pytest.mark.parametrize("crystal", ["Si", "GaN"])
    def test_compute_matches_a_direct_calculation(self, crystal):
        # The builder is bookkeeping around EBSD: the electrons leave along a
        # direction d, and by reciprocity it is calculated with a wave coming
        # in along -d, in a slab cut against the patch. For GaN, which has no
        # inversion centre, calculating along +d instead would give the
        # other polar face.
        atoms = (
            ase.build.bulk("Si", "diamond", a=5.431)
            if crystal == "Si"
            else ase.build.bulk("GaN", "wurtzite", a=3.189, c=5.185, u=0.377)
        )
        builder = EBSDReferencePattern(
            atoms,
            energy=30e3,
            n_patches=1,
            slab_cell=(6.0, 6.0, 4.0),
            gpts=8,
            direction_gpts=30,
            max_angle=200.0,
        )
        pattern = builder.build(pbar=False)

        slab, rotation = rotated_slab(
            builder.atoms, -builder.zone_axes[0], builder.slab_cell
        )
        direct = EBSD(
            abtem.Potential(
                slab,
                gpts=builder.potential_gpts,
                slice_thickness=1.0,
                projection="finite",
            ),
            detector=BackscatterDetector(directions=-pattern.directions @ rotation.T),
            energy=30e3,
        ).build()

        assert np.allclose(pattern.array, direct.array, rtol=1e-6)

    def test_compute_records_the_setup_in_the_metadata(self, builder):
        metadata = builder.build(pbar=False).metadata
        assert metadata["n_patches"] == 1
        assert metadata["max_angle"] == pytest.approx(200.0)
        assert metadata["energy"] == 30e3
        assert metadata["projection"] == "lambert"

    def test_compute_projects_to_an_image(self, builder):
        # One patch covers a cap, not the hemisphere, so most of the image is
        # empty by construction.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseProjectionWarning)
            images = builder.build(pbar=False).bin(8)
        assert images.array.shape == (8, 8)
        assert np.all(np.isfinite(images.array))


class TestSparseProjection:
    @staticmethod
    def pattern_sampled_in(projection, gpts=128):
        directions = validate_projection(projection).grid(gpts)
        return SphericalPattern(1.0 + directions[:, 2], directions)

    @pytest.mark.parametrize("projection", ["stereographic", "lambert"])
    def test_matching_projection_does_not_warn(self, projection):
        pattern = self.pattern_sampled_in(projection, gpts=148)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseProjectionWarning)
            pattern.bin(128, projection)

    @pytest.mark.parametrize(
        "sampled, viewed",
        [("stereographic", "lambert"), ("lambert", "stereographic")],
    )
    def test_mismatched_projection_warns(self, sampled, viewed):
        # An even grid in one projection is uneven in the other, so binning
        # through the wrong one leaves a moire of empty pixels.
        pattern = self.pattern_sampled_in(sampled, gpts=148)
        with pytest.warns(SparseProjectionWarning, match="no sampled direction"):
            pattern.bin(128, viewed)

    def test_too_many_pixels_warns(self):
        pattern = self.pattern_sampled_in("lambert", gpts=32)
        with pytest.warns(SparseProjectionWarning):
            pattern.bin(256, "lambert")

    def test_stereographic_corners_do_not_count_as_holes(self):
        # The corners outside the disk are legitimately empty; only pixels
        # inside the projection's domain are holes.
        pattern = self.pattern_sampled_in("stereographic", gpts=148)
        image = pattern.bin(128, "stereographic")
        assert np.count_nonzero(image.array) < 128**2
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseProjectionWarning)
            pattern.bin(128, "stereographic")


class TestPixelCenters:
    def test_spans_the_square_without_touching_the_edges(self):
        centers = pixel_centers(4)
        assert centers.shape == (16, 2)
        assert centers[:, 0].min() == pytest.approx(-0.75)
        assert centers[:, 0].max() == pytest.approx(0.75)

    def test_ordering_matches_the_raveled_image(self):
        gpts = 5
        centers = pixel_centers(gpts).reshape(gpts, gpts, 2)
        # first axis varies x, second varies y, as histogram2d bins them
        assert centers[1, 0, 0] < centers[2, 0, 0]
        assert centers[0, 1, 1] < centers[0, 2, 1]

    def test_bin_directions_counts(self):
        directions = np.array([[0.0, 0.0, 1.0]])
        image, counts = bin_directions(
            directions, np.array([7.0]), gpts=4, return_counts=True
        )
        assert counts.sum() == 1
        assert image[counts > 0] == pytest.approx(7.0)


def emsoft_interpolation(master, xy):
    """EMsoft's LambertgetInterpolation and the sum it feeds, line by line.

    `master` is indexed ``[-npx..npx, -npx..npx]`` shifted to start at zero,
    and `xy` is in the Lambert square scaled to ``[-1, 1]``.
    """
    npx = (master.shape[0] - 1) // 2
    out = np.empty(len(xy))
    for m, (x, y) in enumerate(xy * npx):  # xy = scl * Lambert2DSquareInverse(dc)
        nix = int(npx + x) - npx
        niy = int(npx + y) - npx
        nixp, niyp = nix + 1, niy + 1
        if nixp > npx:
            nixp = nix
        if niyp > npx:
            niyp = niy
        if nix < -npx:
            nix = nixp
        if niy < -npx:
            niy = niyp
        dx, dy = x - nix, y - niy
        dxm, dym = 1.0 - dx, 1.0 - dy
        out[m] = (
            master[nix + npx, niy + npx] * dxm * dym
            + master[nixp + npx, niy + npx] * dx * dym
            + master[nix + npx, niyp + npx] * dxm * dy
            + master[nixp + npx, niyp + npx] * dx * dy
        )
    return out


class TestEBSDDetectorPattern:
    """One orientation's pattern, calculated at the detector's own pixels."""

    @staticmethod
    def builder(**kwargs):
        # A 0.2 rad wide detector and patches of about 60 mrad, so it spans
        # several; a coarse grid and a tiny slab keep it quick.
        defaults = dict(
            slab_cell=(6.0, 6.0, 4.0),
            energy=30e3,
            n_patches=2000,
            sampling=0.25,
        )
        atoms = kwargs.pop("atoms", ase.build.bulk("Si", "diamond", a=5.431))
        return EBSDDetectorPattern(
            atoms,
            EBSDGeometry(shape=(6, 6), detector_distance=15000.0, pixel_size=500.0),
            (30.0, 54.7, 45.0),
            **{**defaults, **kwargs},
        )

    def test_the_pixels_are_the_detector_directions_in_the_crystal(self):
        builder = self.builder()
        expected = builder.geometry.rotated_directions(builder.euler)
        assert np.allclose(builder.directions, expected)

    def test_the_beam_comes_down_the_microscope_axis(self):
        # into the surface at the sample tilt, towards the detector's side,
        # and turned into the crystal frame by the same Euler angles
        builder = self.builder()
        tilt = np.radians(builder.geometry.sample_tilt)
        expected = bunge_rotation(builder.euler) @ [np.sin(tilt), 0.0, -np.cos(tilt)]
        assert np.allclose(builder.beam_direction, expected)

    def test_every_pixel_is_calculated_near_its_own_axis(self):
        builder = self.builder()
        assert len(builder.zone_axes) > 1
        assert builder.max_angle < patch_half_angle(2000)

    def test_a_single_slab_is_along_the_mean_direction(self):
        builder = self.builder(n_patches=None, sampling=0.15)
        mean = builder.directions.reshape(-1, 3).mean(axis=0)
        assert np.allclose(builder.zone_axes, [mean / np.linalg.norm(mean)])

    def test_is_an_image_of_the_detector(self):
        pattern = self.builder().build(lazy=False)
        assert isinstance(pattern, EBSDPatternImages)
        assert pattern.array.shape == (6, 6)
        assert pattern.metadata["source"] == "uniform"
        assert pattern.metadata["n_patches"] == 2000

    def test_each_pixel_is_its_own_calculation(self):
        # The pixels are calculated patch by patch and put back in order;
        # computing each one alone, in its own patch's slab, must agree.
        builder = self.builder()
        image = np.asarray(builder.build(lazy=False).array).ravel()

        directions = builder.directions.reshape(-1, 3)
        axes = np.concatenate([fibonacci_hemisphere(2000), -fibonacci_hemisphere(2000)])
        silicon = ase.build.bulk("Si", "diamond", a=5.431)
        block = bulk_block(silicon, builder.slab_cell)

        for i in [0, 5, 14, 21, 30, 35]:
            # the electrons leave along d; the reciprocity wave comes in along
            # -d, in a slab cut against the pixel's patch
            axis = axes[np.argmax(axes @ directions[i])]
            slab, rotation = rotated_slab(
                block, -axis, builder.slab_cell, repetitions=(1, 1, 1)
            )
            alone = EBSD(
                abtem.Potential(
                    slab,
                    gpts=builder.potential_gpts,
                    slice_thickness=1.0,
                    projection="finite",
                ),
                BackscatterDetector(directions=(rotation @ -directions[i])[None]),
                energy=30e3,
            ).build(lazy=False)
            assert np.asarray(alone.array)[0] == pytest.approx(image[i], rel=1e-5)

    def test_matches_lazily(self):
        builder = self.builder()
        eager = np.asarray(builder.build(lazy=False).array)
        lazy = builder.build(lazy=True)
        assert np.allclose(np.asarray(lazy.compute().array), eager, rtol=1e-6)

    def test_a_wave_source_gives_the_energy(self):
        builder = self.builder(energy=None, source=abtem.PlaneWave(energy=20e3))
        assert builder.energy == 20e3

    def test_refuses_an_energy_that_contradicts_the_wave(self):
        with pytest.raises(ValueError, match="contradicts the source"):
            self.builder(source=abtem.PlaneWave(energy=20e3))

    def test_needs_an_energy(self):
        with pytest.raises(ValueError, match="give the beam energy"):
            self.builder(energy=None)

    def test_rejects_an_unknown_source(self):
        with pytest.raises(ValueError, match="source must be"):
            self.builder(source="beam")

    def test_a_plane_wave_lights_the_atoms_unevenly(self):
        # Sent in along the beam's own direction, it channels, and lights some
        # atoms more than others -- normalized within each of its slices.
        builder = self.builder(source=abtem.PlaneWave(energy=30e3))
        block, origin = builder._block_and_origin()
        illumination, axes, shape = builder._illumination(block, origin, None)

        assert illumination.shape == (1, len(block))
        assert shape == () and axes == []

        # The block is cut for the incident slab and holds more than it; the
        # atoms outside it no patch uses, and they are left at one.
        lit = illumination[0][illumination[0] != 1.0]
        assert len(lit) > 100
        assert lit.mean() == pytest.approx(1.0, rel=1e-6)
        assert lit.std() > 0.05

    def test_a_plane_wave_changes_the_pattern(self):
        uniform = np.asarray(self.builder().build(lazy=False).array)
        builder = self.builder(source=abtem.PlaneWave(energy=30e3))
        lit = builder.build(lazy=False)

        assert lit.metadata["source"] == "PlaneWave"
        assert not np.allclose(np.asarray(lit.array), uniform, rtol=1e-3)
        lazy = builder.build(lazy=True).compute()
        assert np.allclose(np.asarray(lazy.array), np.asarray(lit.array), rtol=1e-6)

    @pytest.mark.parametrize("n_patches", [None, 2000])
    def test_scans_the_probe(self, n_patches):
        # The beam is sent in once for the whole block, so it can be scanned
        # whether the pixels share one slab or not.
        probe = abtem.Probe(semiangle_cutoff=10, energy=30e3)
        builder = self.builder(n_patches=n_patches, sampling=0.15, source=probe)
        positions = [[0.0, 0.0], [1.0, 1.5]]

        together = builder.build(scan=abtem.CustomScan(positions), lazy=False)
        assert together.array.shape == (2, 6, 6)

        alone = builder.build(scan=abtem.CustomScan(positions[1:]), lazy=False)
        assert np.allclose(
            np.asarray(together.array)[1], np.asarray(alone.array)[0], rtol=1e-4
        )
        assert not np.allclose(
            np.asarray(together.array)[0], np.asarray(together.array)[1], rtol=1e-3
        )

    def test_a_scan_keeps_its_shape_and_its_positions(self):
        # A grid scan stays a grid, and positions are reported as given --
        # measured from the origin, not from the corner of the beam's slab.
        probe = abtem.Probe(semiangle_cutoff=30, energy=30e3)
        builder = self.builder(n_patches=None, sampling=0.15, source=probe)

        scan = abtem.GridScan(start=(0, 0), end=(1.5, 1.0), gpts=(2, 3))
        pattern = builder.build(scan=scan, lazy=False)
        assert pattern.array.shape == (2, 3, 6, 6)

        position = np.asarray(scan.get_positions())[1, 1]
        alone = builder.build(scan=abtem.CustomScan([position]), lazy=False)
        assert np.allclose(
            np.asarray(pattern.array)[1, 1], np.asarray(alone.array)[0], rtol=1e-4
        )
        assert np.allclose(alone.ensemble_axes_metadata[0].values, [position])

    @pytest.mark.parametrize("source", ["uniform", "plane wave"])
    def test_only_a_probe_can_be_scanned(self, source):
        source = abtem.PlaneWave(energy=30e3) if source == "plane wave" else source
        with pytest.raises(ValueError, match="only a probe source depends"):
            self.builder(source=source).build(
                scan=abtem.CustomScan([[0.0, 0.0], [1.0, 1.0]]), lazy=False
            )

    def test_warns_when_the_sampling_cannot_resolve_the_atoms(self):
        with pytest.warns(AntialiasLossWarning, match="scattering power"):
            self.builder(n_patches=None, sampling=0.3)

    @pytest.mark.parametrize(
        "euler, hemisphere", [((0.0, 0.0, 0.0), 1.0), ((0.0, 180.0, 0.0), -1.0)]
    )
    def test_each_hemisphere_is_calculated_in_its_own_patches(self, euler, hemisphere):
        # Rather than assuming the crystal centrosymmetric, as a reference
        # pattern does: a detector looking into the crystal's southern
        # hemisphere is calculated there, in slabs along the negated axes.
        builder = EBSDDetectorPattern(
            ase.build.bulk("Si", "diamond", a=5.431),
            EBSDGeometry(shape=(4, 4), detector_distance=15000.0, pixel_size=500.0),
            euler,
            slab_cell=(6.0, 6.0, 4.0),
            energy=30e3,
            sampling=0.25,
        )
        assert np.all(np.sign(builder.directions[..., 2]) == hemisphere)
        assert np.all(np.sign(builder.zone_axes[:, 2]) == hemisphere)

    @pytest.mark.parametrize("euler", [(0.0, 0.0), (1.0, 2.0, 3.0, 4.0)])
    def test_rejects_anything_but_one_orientation(self, euler):
        with pytest.raises(ValueError, match="one orientation"):
            EBSDDetectorPattern(
                ase.build.bulk("Si", "diamond", a=5.431),
                EBSDGeometry(),
                euler,
                slab_cell=(6.0, 6.0, 4.0),
                energy=30e3,
            )


class TestThermalDisplacements:
    """FrozenPhonons in place of Atoms: every slab displaced as its crystal is."""

    SILICON = ase.build.bulk("Si", "diamond", a=5.431)

    def frozen(self, sigmas=0.1):
        return abtem.FrozenPhonons(self.SILICON, num_configs=2, sigmas=sigmas, seed=3)

    def test_displacements_given_atom_by_atom_follow_each_atom(self):
        # Into the block and out into a slab cut at an arbitrary orientation:
        # only the atoms of the second site of the unit cell may move.
        crystal, frozen_phonons = _crystal_and_displacements(self.frozen([0.0, 0.1]))
        slab, _ = rotated_slab(
            bulk_block(crystal, (8.0, 8.0, 6.0)),
            np.array([1.0, 1.0, 1.0]),
            (8.0, 8.0, 6.0),
            repetitions=(1, 1, 1),
        )
        displaced = _displaced(slab, frozen_phonons)

        moved = np.linalg.norm(
            displaced.randomize(slab).positions - slab.positions, axis=1
        )
        assert np.array_equal(moved > 0, slab.arrays["cell_index"] == 1)
        assert displaced.num_configs == 2
        assert displaced.seed == frozen_phonons.seed

    def test_displacements_given_by_element_apply_to_any_slab(self):
        _, frozen_phonons = _crystal_and_displacements(self.frozen(0.08))
        displaced = _displaced(silicon_slab(), frozen_phonons)
        assert isinstance(displaced, abtem.FrozenPhonons)
        assert displaced.num_configs == 2

    @pytest.mark.parametrize("builder", ["reference", "detector"])
    def test_only_frozen_phonons_are_accepted(self, builder):
        ensemble = abtem.AtomsEnsemble([self.SILICON, self.SILICON])
        with pytest.raises(TypeError, match="give Atoms, or FrozenPhonons"):
            if builder == "reference":
                EBSDReferencePattern(ensemble, 30e3, slab_cell=(6.0, 6.0, 4.0))
            else:
                TestEBSDDetectorPattern.builder(atoms=ensemble)

    def reference(self, atoms):
        return EBSDReferencePattern(
            atoms,
            30e3,
            n_patches=1,
            slab_cell=(6.0, 6.0, 4.0),
            gpts=8,
            direction_gpts=30,
            max_angle=200.0,
        )

    def test_the_reference_pattern_averages_the_configurations(self):
        # The one patch by hand: its slab, displaced by the same FrozenPhonons.
        frozen = self.frozen()
        builder = self.reference(frozen)
        pattern = builder.build(pbar=False, lazy=False)
        assert builder.atoms is not frozen.atoms and len(builder.atoms) == 2
        assert pattern.metadata["num_configs"] == 2

        slab, rotation = rotated_slab(
            builder.atoms, -builder.zone_axes[0], builder.slab_cell
        )
        direct = EBSD(
            abtem.Potential(
                abtem.FrozenPhonons(slab, num_configs=2, sigmas=0.1, seed=frozen.seed),
                gpts=builder.potential_gpts,
                slice_thickness=1.0,
                projection="finite",
            ),
            BackscatterDetector(directions=-pattern.directions @ rotation.T),
            energy=30e3,
        ).build()
        assert np.allclose(pattern.array, direct.array, rtol=1e-5)

        static = self.reference(self.SILICON).build(pbar=False, lazy=False)
        assert not np.allclose(pattern.array, static.array, rtol=1e-3)

    def test_the_detector_pattern_averages_the_configurations(self):
        builder = TestEBSDDetectorPattern.builder(atoms=self.frozen())
        thermal = builder.build(lazy=False)
        assert thermal.metadata["num_configs"] == 2

        lazy = builder.build(lazy=True).compute()
        assert np.allclose(np.asarray(lazy.array), np.asarray(thermal.array), rtol=1e-6)

        static = TestEBSDDetectorPattern.builder().build(lazy=False)
        assert not np.allclose(
            np.asarray(thermal.array), np.asarray(static.array), rtol=1e-3
        )

    def test_the_incident_beam_is_averaged_too(self):
        def illumination(atoms):
            builder = TestEBSDDetectorPattern.builder(
                atoms=atoms, source=abtem.PlaneWave(energy=30e3)
            )
            block, origin = builder._block_and_origin()
            return builder._illumination(block, origin, None)[0][0]

        thermal = illumination(self.frozen())
        static = illumination(self.SILICON)

        lit = thermal != 1.0
        assert thermal[lit].mean() == pytest.approx(1.0, rel=1e-6)
        assert not np.allclose(thermal[lit], static[lit], rtol=1e-2)


class TestHemispheres:
    """Which hemisphere a reference covers, and what stands in for the other."""

    SILICON = ase.build.bulk("Si", "diamond", a=5.431)
    GAN = ase.build.bulk("GaN", "wurtzite", a=3.189, c=5.185, u=0.377)

    def reference(self, atoms, hemisphere, **kwargs):
        return EBSDReferencePattern(
            atoms,
            30e3,
            slab_cell=(6.0, 6.0, 4.0),
            n_patches=3,
            gpts=8,
            direction_gpts=13,
            hemisphere=hemisphere,
            **kwargs,
        )

    def test_detects_an_inversion_centre(self):
        assert is_centrosymmetric(self.SILICON)
        assert not is_centrosymmetric(self.GAN)
        assert not self.reference(self.GAN, "north").centrosymmetric

    def test_the_southern_directions_and_patches_lie_south(self):
        south = self.reference(self.SILICON, "south")
        # the equator of a southern grid stays southern: -0.0
        assert np.all(np.signbit(south.directions[:, 2]))
        assert np.all(np.signbit(south.zone_axes[:, 2]))
        # (x, y, -z) below each northern direction, as EMsoft lays them out
        north = self.reference(self.SILICON, "north")
        assert np.allclose(south.directions, north.directions * [1, 1, -1])
        assert np.allclose(south.zone_axes, -north.zone_axes)

    def test_opposite_directions_differ_only_by_inversion(self):
        # The south is the north of the crystal inverted about the slabs'
        # centre. Silicon cut about a bond centre, one of its inversion
        # centres, is its own inversion image: a direction and its opposite
        # are the same calculation, and agree to rounding. GaN has no
        # inversion centre, and its two hemispheres differ.
        bond_centre = np.full(3, 5.431 / 8)
        silicon = self.reference(self.SILICON, "both", origin=bond_centre).build(
            pbar=False, lazy=False
        )
        directions = silicon.northern.directions
        north = silicon.interpolate_directions(directions)
        south = silicon.interpolate_directions(-directions)
        assert np.allclose(north, south, rtol=1e-4)

        gan = self.reference(self.GAN, "both").build(pbar=False, lazy=False)
        north = gan.interpolate_directions(directions)
        south = gan.interpolate_directions(-directions)
        assert (
            np.sqrt(np.mean((north / north.mean() - south / south.mean()) ** 2)) > 1e-3
        )

    @pytest.mark.parametrize("given", [True, False])
    def test_a_given_symmetry_skips_the_search(self, monkeypatch, given):
        def search(atoms):
            raise AssertionError("the symmetry was searched for")

        monkeypatch.setattr(reference_module, "is_centrosymmetric", search)
        reference = self.reference(self.GAN, "north", centrosymmetric=given)
        assert reference.centrosymmetric is given
        pattern = reference.build(pbar=False, lazy=False)
        assert pattern.metadata["centrosymmetric"] is given

    def test_without_a_given_symmetry_it_is_searched_for(self, monkeypatch):
        calls = []

        def search(atoms):
            calls.append(atoms)
            return False

        monkeypatch.setattr(reference_module, "is_centrosymmetric", search)
        reference = self.reference(self.SILICON, "north")
        assert not calls  # only when first needed
        assert reference.centrosymmetric is False
        assert reference.centrosymmetric is False
        assert len(calls) == 1  # and once

    def test_rejects_a_symmetry_that_is_not_a_bool(self):
        with pytest.raises(ValueError, match="centrosymmetric must be"):
            self.reference(self.SILICON, "north", centrosymmetric="yes")

    def test_both_is_north_and_south(self):
        both = self.reference(self.SILICON, "both")
        north = self.reference(self.SILICON, "north")
        assert len(both.directions) == 2 * len(north.directions)
        assert len(both.zone_axes) == 2 * len(north.zone_axes)

    def test_a_pattern_records_its_hemisphere_and_symmetry(self):
        pattern = self.reference(self.GAN, "both").build(pbar=False, lazy=False)
        assert pattern.hemisphere == "both"
        assert pattern.metadata["hemisphere"] == "both"
        assert pattern.metadata["centrosymmetric"] is False

    def test_rejects_an_unknown_hemisphere(self):
        with pytest.raises(ValueError, match="hemisphere must be"):
            self.reference(self.SILICON, "east")

    @staticmethod
    def synthetic(hemisphere, centrosymmetric=None):
        northern = SquareLambertProjection().grid(21)
        north = 1.0 + 0.3 * northern[:, 0]
        metadata = (
            {} if centrosymmetric is None else {"centrosymmetric": centrosymmetric}
        )
        if hemisphere == "north":
            return SphericalPattern(north, northern, metadata=metadata)
        southern = northern * [1, 1, -1]
        south = 2.0 + 0.3 * southern[:, 1]
        return SphericalPattern(
            np.concatenate([north, south]),
            np.concatenate([northern, southern]),
            metadata=metadata,
        )

    def test_the_inversion_image_stands_in_for_a_centrosymmetric_crystal(self):
        pattern = self.synthetic("north", centrosymmetric=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            southern = pattern.southern
        assert southern.hemisphere == "south"
        assert np.allclose(southern.directions, -pattern.directions)
        assert np.allclose(southern.array, pattern.array)

    def test_warns_when_it_stands_in_for_a_polar_crystal(self):
        pattern = self.synthetic("north", centrosymmetric=False)
        with pytest.warns(UserWarning, match="not centrosymmetric"):
            pattern.southern
        with pytest.warns(UserWarning, match="not centrosymmetric"):
            pattern.interpolate_directions(np.array([[0.1, 0.2, -0.97]]))

    def test_a_calculated_southern_hemisphere_is_used(self):
        pattern = self.synthetic("both", centrosymmetric=False)
        southern = SquareLambertProjection().grid(21)[:40] * [1, 1, -1]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            values = pattern.interpolate_directions(southern)
        assert np.allclose(values, 2.0 + 0.3 * southern[:, 1])

    def test_emsoft_gets_the_calculated_southern_hemisphere(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        pattern = self.synthetic("both", centrosymmetric=False)
        pattern.metadata["energy"] = 30e3
        path = tmp_path / "both.h5"
        write_emsoft_master_pattern(
            str(path), pattern, self.GAN, npx=10, space_group=186
        )
        with h5py.File(path, "r") as f:
            group = f["EMData/EBSDmaster"]
            south = np.asarray(group["mLPSH"]).squeeze()
            north = np.asarray(group["mLPNH"]).squeeze()
        expected = np.asarray(
            pattern.interpolate(21, "lambert", hemisphere="south").array
        ).T
        assert np.allclose(south, expected, atol=1e-5)
        # not the inversion image a centrosymmetric fill would have written
        assert not np.allclose(south, north[::-1, ::-1], atol=1e-2)


class TestEnergyInTheBuilders:
    def test_a_reference_pattern_gets_an_energy_axis(self):
        common = dict(
            slab_cell=(6.0, 6.0, 4.0),
            n_patches=1,
            gpts=8,
            direction_gpts=30,
            max_angle=200.0,
        )
        silicon = ase.build.bulk("Si", "diamond", a=5.431)
        pattern = EBSDReferencePattern(
            silicon, 30e3, backscatter_energy=[30e3, 26e3], **common
        ).build(pbar=False, lazy=False)
        assert pattern.array.shape[0] == 2
        assert isinstance(pattern.ensemble_axes_metadata[0], EnergyAxis)
        assert pattern.metadata["backscatter_energies"] == [30e3, 26e3]

        single = EBSDReferencePattern(silicon, 30e3, **common).build(
            pbar=False, lazy=False
        )
        assert np.allclose(pattern.array[0], single.array, rtol=1e-5)
        assert not np.allclose(pattern.array[1], single.array, rtol=1e-3)

    def test_a_detector_pattern_gets_an_energy_axis(self):
        builder = TestEBSDDetectorPattern.builder(backscatter_energy=[30e3, 26e3])
        pattern = builder.build(lazy=False)
        assert pattern.array.shape == (2, 6, 6)
        assert isinstance(pattern.ensemble_axes_metadata[0], EnergyAxis)
        lazy = builder.build(lazy=True).compute()
        assert np.allclose(np.asarray(lazy.array), np.asarray(pattern.array), rtol=1e-6)


class TestDepthResolved:
    @pytest.fixture
    def setup(self):
        slab, _ = rotated_slab(
            ase.build.bulk("Si", "diamond", a=5.431),
            np.array([1.0, 0.0, 1.0]),
            (8.0, 8.0, 8.0),
        )
        return slab, BackscatterDetector(max_angle=50, gpts=4)

    def test_adds_a_depth_axis(self, setup):
        slab, detector = setup
        pattern = make_ebsd(slab, detector, depth_bins=4).build()
        assert pattern.array.shape == (4, 4, 4)
        axis = pattern.ensemble_axes_metadata[0]
        assert axis.label == "depth" and axis.units == "Å"
        assert np.allclose(axis.values, [1.0, 3.0, 5.0, 7.0])
        assert pattern.metadata["depth_bins"] == [0.0, 2.0, 4.0, 6.0, 8.0]
        assert np.array(pattern.metadata["depth_emission"]).shape == (4,)

    def test_the_bins_recombine_into_the_whole_pattern(self, setup):
        slab, detector = setup
        whole = make_ebsd(slab, detector).build()
        binned = make_ebsd(slab, detector, depth_bins=4).build()
        emission = np.array(binned.metadata["depth_emission"])
        recombined = np.tensordot(emission, binned.array, (0, 0)) / emission.sum()
        assert np.allclose(recombined, whole.array, rtol=1e-5)

    def test_any_depth_weighting_can_be_applied_afterwards(self, setup):
        # weights constant within each bin give exactly the run weighted so
        slab, detector = setup
        binned = make_ebsd(slab, detector, depth_bins=4).build()
        emission = np.array(binned.metadata["depth_emission"])
        weights = np.array([1.0, 0.5, 0.25, 0.125])
        recombined = (
            np.tensordot(weights * emission, binned.array, (0, 0))
            / (weights * emission).sum()
        )

        per_slice = np.repeat(weights, 2)  # two 1 Å slices to a bin
        weighted = make_ebsd(slab, detector, depth_weight=per_slice).build()
        assert np.allclose(recombined, weighted.array, rtol=1e-5)

    def test_matches_lazily(self, setup):
        slab, detector = setup
        eager = make_ebsd(slab, detector, depth_bins=4).build(lazy=False)
        lazy = make_ebsd(slab, detector, depth_bins=4).build(lazy=True).compute()
        assert np.allclose(np.asarray(lazy.array), np.asarray(eager.array), rtol=1e-6)
        assert lazy.metadata["depth_emission"] == eager.metadata["depth_emission"]

    def test_energy_comes_before_depth(self, setup):
        slab, detector = setup
        pattern = make_ebsd(
            slab, detector, depth_bins=2, backscatter_energy=[30e3, 26e3]
        ).build()
        assert pattern.array.shape == (2, 2, 4, 4)
        assert [a.label for a in pattern.ensemble_axes_metadata] == ["Energy", "depth"]

    @pytest.mark.parametrize(
        "depth_bins, match",
        [
            (np.array([0.0, 0.2, 8.0]), "at least one slice"),
            ([4.0, 2.0], "increasing"),
            (0, "at least 1"),
        ],
    )
    def test_rejects_bad_bins(self, setup, depth_bins, match):
        slab, detector = setup
        with pytest.raises(ValueError, match=match):
            make_ebsd(slab, detector, depth_bins=depth_bins).build()


class TestEnergyDependentDepthWeights:
    @pytest.fixture
    def setup(self):
        return silicon_slab(), BackscatterDetector(max_angle=50, gpts=4)

    def test_a_callable_of_depth_and_energy(self, setup):
        # deeper for the electrons that lost more
        atoms, detector = setup
        energies = [30e3, 26e3]

        def weight(z, energy):
            return np.exp(-z / (2.0 + (30e3 - energy) / 1e3))

        together = make_ebsd(
            atoms, detector, backscatter_energy=energies, depth_weight=weight
        ).build()
        for i, energy in enumerate(energies):
            alone = make_ebsd(
                atoms,
                detector,
                backscatter_energy=energy,
                depth_weight=lambda z, e=energy: weight(z, e),
            ).build()
            assert np.allclose(together.array[i], alone.array, rtol=1e-5)

    def test_one_row_of_weights_per_energy(self, setup):
        atoms, detector = setup
        rows = np.array([np.linspace(1.0, 0.2, 8), np.ones(8)])
        together = make_ebsd(
            atoms, detector, backscatter_energy=[30e3, 26e3], depth_weight=rows
        ).build()
        second = make_ebsd(atoms, detector, backscatter_energy=26e3).build()
        assert np.allclose(together.array[1], second.array, rtol=1e-5)

    def test_rejects_a_row_count_that_is_not_the_energies(self, setup):
        atoms, detector = setup
        with pytest.raises(ValueError, match="rows but there are"):
            make_ebsd(
                atoms,
                detector,
                backscatter_energy=[30e3, 26e3],
                depth_weight=np.ones((3, 8)),
            ).build()


class TestLambertBilinear:
    """A pattern on a Lambert grid is interpolated the way EMsoft does it."""

    @staticmethod
    def on_grid(values_of, gpts=21):
        directions = SquareLambertProjection().grid(gpts)
        xy = SquareLambertProjection().project(directions)
        return SphericalPattern(values_of(xy), directions), xy

    def test_matches_emsoft(self):
        rng = np.random.default_rng(0)
        gpts = 21
        pattern, xy = self.on_grid(lambda xy: rng.random(len(xy)), gpts)

        # the master pattern EMsoft would hold: node values on the square
        nodes = np.rint((xy + 1) / 2 * (gpts - 1)).astype(int)
        master = np.empty((gpts, gpts))
        master[nodes[:, 0], nodes[:, 1]] = np.asarray(pattern.array)

        targets = rng.normal(size=(300, 3))
        targets[:, 2] = np.abs(targets[:, 2])
        targets /= np.linalg.norm(targets, axis=1, keepdims=True)
        # the edges and corners of the square, where the clamping acts
        targets = np.concatenate(
            [targets, [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.6, 0.8, 0.0]]]
        )

        expected = emsoft_interpolation(
            master, SquareLambertProjection().project(targets)
        )
        assert np.allclose(
            pattern.interpolate_directions(targets), expected, atol=1e-12
        )

    def test_reproduces_a_bilinear_function_exactly(self):
        # Which triangulation does not: a product of the two coordinates is
        # bilinear in each cell, but not linear on either of its triangles.
        pattern, _ = self.on_grid(lambda xy: 2.0 + xy[:, 0] * xy[:, 1])
        rng = np.random.default_rng(1)
        targets = rng.normal(size=(200, 3))
        targets[:, 2] = np.abs(targets[:, 2])
        targets /= np.linalg.norm(targets, axis=1, keepdims=True)

        xy = SquareLambertProjection().project(targets)
        assert np.allclose(
            pattern.interpolate_directions(targets),
            2.0 + xy[:, 0] * xy[:, 1],
            atol=1e-12,
        )

    def test_repeated_nodes_are_averaged(self):
        # What an overlap between patches produces: a direction computed twice.
        directions = SquareLambertProjection().grid(11)
        values = np.ones(len(directions))
        doubled = SphericalPattern(
            np.concatenate([values, np.full(len(directions), 3.0)]),
            np.concatenate([directions, directions]),
        )
        assert np.allclose(doubled.interpolate_directions(directions), 2.0)

    def test_other_samplings_are_triangulated(self):
        # A stereographic grid is not a Lambert one, so it takes the general
        # path -- and still interpolates a linear function exactly.
        from abtem.ebsd.measurements import _lambert_grid

        lambert = SquareLambertProjection()
        directions = StereographicProjection().grid(41)
        assert _lambert_grid(lambert.project(directions)) is None

        # linear in the plane the triangulation works in
        pattern = SphericalPattern(
            1.0 + 0.5 * lambert.project(directions)[:, 0], directions
        )

        # well inside the disk the grid covers, so nothing is filled
        rng = np.random.default_rng(2)
        targets = rng.normal(size=(100, 3))
        targets[:, 2] = np.abs(targets[:, 2]) + 1.5
        targets /= np.linalg.norm(targets, axis=1, keepdims=True)

        values = pattern.interpolate_directions(targets)
        assert np.allclose(
            values, 1.0 + 0.5 * lambert.project(targets)[:, 0], atol=1e-9
        )


class TestInterpolate:
    def test_exact_when_sampled_on_the_same_grid(self):
        # Sampling the pattern on the grid it is asked for makes the
        # interpolation a repackaging: the node values are the computed ones.
        gpts = 61
        directions = SquareLambertProjection().grid(gpts)
        values = 1.0 + directions[:, 2] + 0.3 * directions[:, 0]
        pattern = SphericalPattern(values, directions)

        image = np.asarray(pattern.interpolate(gpts, "lambert").array)

        nodes = np.linspace(-1, 1, gpts)
        x, y = np.meshgrid(nodes, nodes, indexing="ij")
        expected = SquareLambertProjection().unproject(
            np.stack([x.ravel(), y.ravel()], axis=1)
        )
        expected = 1.0 + expected[:, 2] + 0.3 * expected[:, 0]
        assert np.allclose(image.ravel(), expected, atol=1e-10)

    def test_no_nearest_fill_when_sampled_in_the_same_projection(self):
        directions = SquareLambertProjection().grid(101)
        pattern = SphericalPattern(1.0 + directions[:, 2], directions)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseProjectionWarning)
            pattern.interpolate(51, "lambert")

    def test_warns_when_the_grid_reaches_past_the_sampled_directions(self):
        # A stereographic sampling does not reach the corners of the Lambert
        # square, so those nodes have to be filled from the nearest sample.
        directions = StereographicProjection().grid(101)
        pattern = SphericalPattern(1.0 + directions[:, 2], directions)
        with pytest.warns(SparseProjectionWarning, match="outside the sampled"):
            pattern.interpolate(51, "lambert")

    def test_stereographic_corners_are_zero(self):
        directions = StereographicProjection().grid(101)
        pattern = SphericalPattern(1.0 + directions[:, 2], directions)
        image = np.asarray(pattern.interpolate(51, "stereographic").array)
        assert image[0, 0] == 0.0
        assert image[25, 25] > 0.0

    def test_preserves_the_ensemble(self):
        directions = SquareLambertProjection().grid(41)
        array = np.stack([np.ones(len(directions)), 2 * np.ones(len(directions))])
        pattern = SphericalPattern(
            array, directions, ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))]
        )
        image = pattern.interpolate(21, "lambert")
        assert image.array.shape == (2, 21, 21)
        assert np.allclose(image.array[1], 2.0)


class TestEMsoftWriter:
    @pytest.fixture
    def silicon(self):
        return ase.build.bulk("Si", "diamond", a=5.431)

    @pytest.fixture
    def pattern(self):
        directions = SquareLambertProjection().grid(81)
        values = 1.0 + 0.5 * directions[:, 2] ** 2
        return SphericalPattern(values, directions, metadata={"energy": 30e3})

    @pytest.fixture
    def written(self, tmp_path, pattern, silicon):
        h5py = pytest.importorskip("h5py")
        pytest.importorskip("spglib")
        path = str(tmp_path / "master.h5")
        write_emsoft_master_pattern(path, pattern, silicon, npx=20)
        return path, h5py

    def test_array_shapes_follow_the_format(self, written):
        path, h5py = written
        with h5py.File(path) as f:
            assert f["EMData/EBSDmaster/mLPNH"].shape == (1, 1, 41, 41)
            assert f["EMData/EBSDmaster/mLPSH"].shape == (1, 1, 41, 41)
            assert f["EMData/EBSDmaster/masterSPNH"].shape == (1, 41, 41)
            assert f["CrystalData/AtomData"].shape[0] == 5
            assert f["CrystalData/LatticeParameters"].shape == (6,)

    def test_scalars_are_one_element_arrays(self, written):
        # EMsoft writes Fortran rank-1 arrays and readers index them as
        # `dataset[:][0]`, which raises on a true scalar dataspace.
        path, h5py = written
        with h5py.File(path) as f:
            for name in [
                "EMData/EBSDmaster/numset",
                "EMData/EBSDmaster/numEbins",
                "EMheader/EBSDmaster/ProgramName",
                "CrystalData/SpaceGroupNumber",
                "NMLparameters/EBSDMasterNameList/npx",
            ]:
                assert f[name].shape == (1,), name
                f[name][:][0]  # must not raise

    def test_writes_the_conventional_cell_in_nanometres(self, written):
        # ase.build.bulk gives the primitive rhombohedral cell, whose axes
        # would contradict the cubic space group recorded next to them.
        path, h5py = written
        with h5py.File(path) as f:
            lattice = f["CrystalData/LatticeParameters"][()]
            assert f["CrystalData/SpaceGroupNumber"][:][0] == 227
            assert np.allclose(lattice[:3], 0.5431, atol=1e-4)  # nm, not Å
            assert np.allclose(lattice[3:], 90.0)
            assert f["CrystalData/CrystalSystem"][:][0] == 1  # cubic

    def test_southern_hemisphere_is_the_centrosymmetric_image(self, written):
        path, h5py = written
        with h5py.File(path) as f:
            north = f["EMData/EBSDmaster/mLPNH"][0, 0]
            south = f["EMData/EBSDmaster/mLPSH"][0, 0]
        assert np.array_equal(south, north[::-1, ::-1])

    def test_energy_is_recorded_in_kev(self, written):
        path, h5py = written
        with h5py.File(path) as f:
            assert f["EMData/EBSDmaster/EkeVs"][()] == pytest.approx([30.0])

    def test_refuses_an_ensemble(self, tmp_path, silicon):
        pytest.importorskip("h5py")
        directions = SquareLambertProjection().grid(21)
        pattern = SphericalPattern(
            np.ones((2, len(directions))),
            directions,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
            metadata={"energy": 30e3},
        )
        with pytest.raises(ValueError, match="only a single pattern"):
            write_emsoft_master_pattern(
                str(tmp_path / "m.h5"), pattern, silicon, npx=10
            )

    def test_requires_an_energy(self, tmp_path, silicon):
        pytest.importorskip("h5py")
        directions = SquareLambertProjection().grid(21)
        pattern = SphericalPattern(np.ones(len(directions)), directions)
        with pytest.raises(ValueError, match="no energy"):
            write_emsoft_master_pattern(
                str(tmp_path / "m.h5"), pattern, silicon, npx=10
            )

    def test_rejects_an_impossible_space_group(self, tmp_path, pattern, silicon):
        pytest.importorskip("h5py")
        with pytest.raises(ValueError, match="between 1 and 230"):
            write_emsoft_master_pattern(
                str(tmp_path / "m.h5"), pattern, silicon, npx=10, space_group=300
            )


class TestBackscatterEnergy:
    def test_defaults_to_the_beam_energy(self):
        energies, weights = _validate_backscatter_energy(None, 30e3)
        assert energies == pytest.approx([30e3])
        assert weights == pytest.approx([1.0])

    def test_weights_are_normalized(self):
        from abtem.distributions import uniform

        _, weights = _validate_backscatter_energy(uniform(28e3, 30e3, 4), 30e3)
        assert weights.sum() == pytest.approx(1.0)

    def test_rejects_energies_above_the_beam(self):
        # A backscattered electron has lost energy; it cannot have gained any.
        with pytest.raises(ValueError, match="cannot carry more than the beam"):
            _validate_backscatter_energy([30e3, 31e3], 30e3)

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            _validate_backscatter_energy([0.0], 30e3)

    def test_single_energy_matches_the_default(self):
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        default = make_ebsd(atoms, detector).build()
        explicit = make_ebsd(atoms, detector, backscatter_energy=30e3).build()
        assert np.array_equal(default.array, explicit.array)

    def test_adds_a_leading_energy_axis(self):
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        patterns = make_ebsd(
            atoms, detector, backscatter_energy=[30e3, 29e3, 28e3]
        ).build()
        assert patterns.array.shape == (3, 4, 4)
        assert isinstance(patterns.ensemble_axes_metadata[0], EnergyAxis)
        assert patterns.ensemble_axes_metadata[0].values == (30e3, 29e3, 28e3)

    def test_the_first_bin_reproduces_the_single_energy_result(self):
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        single = make_ebsd(atoms, detector, backscatter_energy=29e3).build()
        multi = make_ebsd(atoms, detector, backscatter_energy=[29e3, 27e3]).build()
        assert np.allclose(multi.array[0], single.array, rtol=1e-5)

    def test_a_lower_energy_gives_a_different_pattern(self):
        # The reciprocity waves travel at the backscattered energy, so their
        # wavelength -- and the diffraction -- changes with it.
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=6)
        patterns = make_ebsd(atoms, detector, backscatter_energy=[30e3, 24e3]).build()
        assert not np.allclose(patterns.array[0], patterns.array[1], rtol=1e-3)

    def test_records_the_weights_in_the_metadata(self):
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        patterns = make_ebsd(atoms, detector, backscatter_energy=[30e3, 28e3]).build()
        assert patterns.metadata["backscatter_energies"] == [30e3, 28e3]
        assert patterns.metadata["energy_weights"] == pytest.approx([0.5, 0.5])


class TestDepthTolerance:
    def test_a_short_escape_depth_stops_early(self):
        # Slices past the escape depth carry no weight, so propagating them is
        # wasted; the result must be unchanged to within the discarded weight.
        atoms = silicon_slab(thickness=40.0)
        detector = BackscatterDetector(max_angle=50, gpts=4)
        exact = make_ebsd(
            atoms, detector, depth_weight=4.0, depth_tolerance=0.0
        ).build()
        truncated = make_ebsd(
            atoms, detector, depth_weight=4.0, depth_tolerance=1e-3
        ).build()
        # The bound is on the discarded weight; the error in any one direction
        # can be a small multiple of it, since the dropped slices are not
        # average ones.
        assert np.allclose(exact.array, truncated.array, rtol=1e-2)

    def test_uniform_weighting_is_unaffected(self):
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        a = make_ebsd(atoms, detector, depth_tolerance=0.0).build()
        b = make_ebsd(atoms, detector).build()
        assert np.array_equal(a.array, b.array)


class TestFFTFriendlyGpts:
    def test_never_coarser_than_requested(self):
        for sampling in (0.05, 0.1414, 0.17, 0.3):
            gpts = fft_friendly_gpts((10.0, 10.0), sampling)
            assert all(10.0 / n <= sampling for n in gpts)

    def test_avoids_awkward_sizes(self):
        # 71 is prime, and its transform is slower than one twice the size.
        assert fft_friendly_gpts((10.0, 10.0), 10.0 / 71) == (72, 72)

    def test_only_small_prime_factors(self):
        for sampling in (0.05, 0.1414, 0.17, 0.3):
            for n in fft_friendly_gpts((10.0, 10.0), sampling):
                remainder = n
                for prime in (2, 3, 5, 7, 11):
                    while remainder % prime == 0:
                        remainder //= prime
                assert remainder == 1, n

    def test_handles_unequal_extents(self):
        assert fft_friendly_gpts((10.0, 20.0), 0.2) == (50, 100)

    def test_the_builder_uses_it(self):
        builder = EBSDReferencePattern(
            ase.build.bulk("Si", "diamond", a=5.431),
            energy=30e3,
            n_patches=400,
            slab_cell=(10.0, 10.0, 40.0),
        )
        assert builder.potential_gpts == fft_friendly_gpts(
            (10.0, 10.0), builder.sampling
        )


class TestSamplingEstimators:
    @pytest.fixture
    def silicon(self):
        return ase.build.bulk("Si", "diamond", a=5.431)

    def test_heavier_atoms_need_finer_sampling(self):
        # A heavier nucleus has a more compact potential, so its transform
        # reaches further and the grid has to be finer.
        samplings = [
            potential_sampling(ase.Atoms(s, positions=[(0, 0, 0)], cell=(4, 4, 4)))
            for s in ("C", "Si", "Cu", "Au")
        ]
        assert samplings == sorted(samplings, reverse=True)

    def test_a_tighter_tolerance_asks_for_finer_sampling(self, silicon):
        assert potential_sampling(silicon, tolerance=0.001) < potential_sampling(
            silicon, tolerance=0.01
        )

    def test_matches_the_silicon_convergence_measurement(self, silicon):
        # Measured against a far finer grid, silicon at 30 kV runs about 0.6%
        # off at 0.07 A and 5% off at 0.10 A, so a 1% target should land
        # between them.
        assert 0.05 < potential_sampling(silicon, tolerance=0.01) < 0.10

    @pytest.mark.parametrize("tolerance", [0.0, 1.0, -0.1])
    def test_rejects_an_impossible_tolerance(self, silicon, tolerance):
        with pytest.raises(ValueError, match="tolerance must be between"):
            potential_sampling(silicon, tolerance=tolerance)

    def test_rejects_an_empty_cell(self):
        with pytest.raises(ValueError, match="empty cell"):
            potential_sampling(ase.Atoms(cell=(4, 4, 4)))

    def test_the_builder_defaults_to_resolving_the_atoms(self, silicon):
        builder = EBSDReferencePattern(
            silicon,
            energy=30e3,
            slab_cell=(20.0, 20.0, 100.0),
            n_patches=400,
        )
        assert builder.sampling == pytest.approx(potential_sampling(silicon))

    def test_the_estimate_is_converged_in_its_own_grid(self):
        # The measuring grid has to reach well past the cutoff it is looking
        # for, or the tail it misses shifts the answer. Pin that it is close
        # to the limit rather than still drifting.
        from abtem.ebsd.sampling import _scattering_power

        _scattering_power.cache_clear()
        coarse_grid_estimate = 0.0659  # what a 512-point measuring grid gives
        assert (
            potential_sampling(ase.build.bulk("Si", "diamond", a=5.431))
            < coarse_grid_estimate
        )

    def test_repeated_calls_are_cached(self):
        from abtem.ebsd.sampling import _scattering_power

        silicon = ase.build.bulk("Si", "diamond", a=5.431)
        potential_sampling(silicon)
        before = _scattering_power.cache_info().hits
        potential_sampling(silicon)
        assert _scattering_power.cache_info().hits > before

    def test_the_power_lost_falls_as_the_sampling_refines(self):
        silicon = ase.build.bulk("Si", "diamond", a=5.431)
        lost = [scattering_power_lost(silicon, s) for s in (0.2, 0.1, 0.05)]
        assert lost == sorted(lost, reverse=True)
        # and the recommended sampling loses what it was asked to
        assert scattering_power_lost(
            silicon, potential_sampling(silicon, tolerance=0.01)
        ) == pytest.approx(0.01, abs=0.003)


class TestLazy:
    @pytest.fixture
    def setup(self):
        return silicon_slab(), BackscatterDetector(max_angle=50, gpts=6)

    def test_matches_the_eager_result(self, setup):
        atoms, detector = setup
        eager = make_ebsd(atoms, detector).build(lazy=False)
        lazy = make_ebsd(atoms, detector).build(lazy=True, max_batch_directions=8)
        assert lazy.is_lazy
        assert np.allclose(np.asarray(lazy.compute().array), np.asarray(eager.array))

    def test_matches_with_explicit_directions(self, setup):
        atoms, grid = setup
        detector = BackscatterDetector(directions=grid.directions)
        eager = make_ebsd(atoms, detector).build(lazy=False)
        lazy = make_ebsd(atoms, detector).build(lazy=True, max_batch_directions=8)
        assert isinstance(lazy, SphericalPattern)
        assert lazy.is_lazy
        assert np.allclose(np.asarray(lazy.compute().array), np.asarray(eager.array))

    def test_matches_over_energies(self, setup):
        atoms, detector = setup
        kwargs = dict(backscatter_energy=[30e3, 28e3])
        eager = make_ebsd(atoms, detector, **kwargs).build(lazy=False)
        lazy = make_ebsd(atoms, detector, **kwargs).build(
            lazy=True, max_batch_directions=8
        )
        assert lazy.array.shape == (2, 6, 6)
        assert np.allclose(np.asarray(lazy.compute().array), np.asarray(eager.array))

    def test_the_block_size_does_not_change_the_result(self, setup):
        atoms, detector = setup
        whole = make_ebsd(atoms, detector).build(lazy=True, max_batch_directions=10_000)
        split = make_ebsd(atoms, detector).build(lazy=True, max_batch_directions=4)
        assert np.allclose(
            np.asarray(whole.compute().array), np.asarray(split.compute().array)
        )

    def test_the_diffraction_pattern_base_axes_are_one_chunk(self, setup):
        # abTEM requires the base axes of a measurement to be unchunked.
        atoms, detector = setup
        lazy = make_ebsd(atoms, detector).build(lazy=True, max_batch_directions=4)
        assert lazy.array.chunks[-2:] == ((6,), (6,))

    def test_nothing_runs_until_computed(self, setup, monkeypatch):
        atoms, detector = setup
        calls = []
        propagate = EBSD._propagate_block

        def counted(self, *args, **kwargs):
            calls.append(1)
            return propagate(self, *args, **kwargs)

        monkeypatch.setattr(EBSD, "_propagate_block", counted)
        lazy = make_ebsd(atoms, detector).build(lazy=True, max_batch_directions=8)
        assert not calls
        lazy.compute()
        assert calls

    def test_a_computed_pattern_is_not_lazy(self, setup):
        atoms, grid = setup
        detector = BackscatterDetector(directions=grid.directions)
        lazy = make_ebsd(atoms, detector).build(lazy=True)
        computed = lazy.compute()
        assert not computed.is_lazy
        assert computed.compute() is computed

    def test_is_reproducible(self, setup):
        # The eager and lazy paths differ by a few float32 eps -- a different
        # FFT plan per thread rounds differently -- but a given path repeats
        # exactly.
        atoms, detector = setup
        first = make_ebsd(atoms, detector).build(lazy=True, max_batch_directions=8)
        second = make_ebsd(atoms, detector).build(lazy=True, max_batch_directions=8)
        assert np.array_equal(
            np.asarray(first.compute().array), np.asarray(second.compute().array)
        )

    def test_the_potential_is_built_inside_the_graph(self, setup):
        # Building it while the graph is assembled would do the work serially
        # and hold every patch's potential at once.
        atoms, detector = setup
        potential = abtem.Potential(
            atoms, sampling=0.15, slice_thickness=1.0, projection="finite"
        )
        lazy = EBSD(
            potential,
            energy=30e3,
            detector=detector,
        ).build(lazy=True)
        assert not potential.is_built if hasattr(potential, "is_built") else True
        assert lazy.is_lazy

    def test_projection_refuses_a_lazy_pattern(self, setup):
        atoms, grid = setup
        detector = BackscatterDetector(directions=grid.directions)
        lazy = make_ebsd(atoms, detector).build(lazy=True)
        with pytest.raises(RuntimeError, match="compute\\(\\) the pattern"):
            lazy.bin(8)


class TestReferencePatternLazy:
    @pytest.fixture
    def builder(self):
        return EBSDReferencePattern(
            ase.build.bulk("Si", "diamond", a=5.431),
            energy=30e3,
            n_patches=1,
            slab_cell=(6.0, 6.0, 4.0),
            gpts=8,
            direction_gpts=30,
            max_angle=200.0,
        )

    def test_matches_the_eager_build(self, builder):
        eager = builder.build(pbar=False, lazy=False)
        lazy = builder.build(lazy=True)
        assert lazy.is_lazy
        assert np.allclose(np.asarray(lazy.compute().array), np.asarray(eager.array))
        assert np.array_equal(lazy.directions, eager.directions)


class TestSlabOrigin:
    @pytest.fixture
    def marked(self):
        """A block with one tagged site well away from the centroid."""
        block = ase.build.bulk("Si", "diamond", a=5.431, cubic=True) * (9, 9, 9)
        target = np.array([12.0, 30.0, 20.0])
        atoms = block.copy()
        atoms += ase.Atom("Au", position=target)
        return atoms, target

    @pytest.mark.parametrize("zone_axis", [(0, 0, 1), (1, 1, 1), (2, 2, 3)])
    def test_anchors_a_feature_at_the_slab_centre(self, marked, zone_axis):
        atoms, target = marked
        cell = (14.0, 14.0, 14.0)
        slab, _ = rotated_slab(
            atoms,
            np.array(zone_axis, dtype=float),
            cell,
            repetitions=(1, 1, 1),
            origin=target,
        )
        gold = [a for a in slab if a.symbol == "Au"]
        assert len(gold) == 1
        assert np.allclose(gold[0].position, np.array(cell) / 2, atol=1e-6)

    def test_without_an_origin_the_feature_is_not_tracked(self, marked):
        # The default anchor is the centroid, which an off-centre feature
        # misses; this is what `origin` exists to fix.
        atoms, _ = marked
        slab, _ = rotated_slab(
            atoms, np.array([1.0, 1.0, 1.0]), (14.0, 14.0, 14.0), repetitions=(1, 1, 1)
        )
        assert not [a for a in slab if a.symbol == "Au"]

    def test_the_rotation_maps_any_offset_into_the_slab(self, marked):
        # The returned matrix is the exact transform, so a feature anywhere can
        # be located in the cut.
        atoms, target = marked
        cell = np.array([14.0, 14.0, 14.0])
        offset = np.array([3.0, -2.0, 1.0])

        moved = atoms.copy()
        moved[-1].position = target + offset
        slab, rotation = rotated_slab(
            moved,
            np.array([1.0, 1.0, 1.0]),
            tuple(cell),
            repetitions=(1, 1, 1),
            origin=target,
        )
        gold = [a for a in slab if a.symbol == "Au"][0]
        assert np.allclose(gold.position, rotation @ offset + cell / 2, atol=1e-6)

    def test_the_default_anchor_is_the_centroid(self):
        atoms = ase.build.bulk("Si", "diamond", a=5.431) * (12, 12, 12)
        cell = (10.0, 10.0, 10.0)
        zone_axis = np.array([1.0, 2.0, 3.0])
        default, _ = rotated_slab(atoms, zone_axis, cell, repetitions=(1, 1, 1))
        explicit, _ = rotated_slab(
            atoms,
            zone_axis,
            cell,
            repetitions=(1, 1, 1),
            origin=atoms.positions.mean(axis=0),
        )
        assert len(default) == len(explicit)
        assert np.allclose(default.positions, explicit.positions, atol=1e-9)

    def test_rejects_a_bad_origin(self):
        with pytest.raises(ValueError, match=r"origin must have shape \(3,\)"):
            rotated_slab(
                ase.build.bulk("Si", "diamond", a=5.431),
                np.array([0.0, 0.0, 1.0]),
                (10.0, 10.0, 10.0),
                origin=np.zeros(2),
            )


class TestInterpolateDirections:
    @staticmethod
    def analytic(directions):
        return (
            1.0
            + 0.4 * directions[:, 2] ** 2
            + 0.25 * directions[:, 0] * directions[:, 1]
        )

    @pytest.fixture
    def pattern(self):
        directions = SquareLambertProjection().grid(101)
        return SphericalPattern(self.analytic(directions), directions)

    def test_exact_at_the_sampled_directions(self, pattern):
        got = pattern.interpolate_directions(pattern.directions)
        assert np.allclose(got, self.analytic(pattern.directions), atol=1e-12)

    def test_accurate_between_them(self, pattern):
        rng = np.random.default_rng(0)
        q = rng.normal(size=(2000, 3))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        q[:, 2] = np.abs(q[:, 2])
        error = np.abs(pattern.interpolate_directions(q) - self.analytic(q))
        assert error.mean() < 1e-3

    def test_handles_gnomonic_rays(self, pattern):
        # A flat detector's rays are a gnomonic projection of the sphere, so
        # they land on no square grid; this is what makes them workable.
        n, spacing, distance = 32, 50.0, 15000.0
        axis = (np.arange(n) - n // 2) * spacing
        x, y = np.meshgrid(axis, axis, indexing="ij")
        rays = np.stack([x.ravel(), y.ravel(), np.full(x.size, distance)], axis=1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)

        patch = pattern.interpolate_directions(rays).reshape(n, n)
        assert patch.shape == (n, n)
        assert np.allclose(patch.ravel(), self.analytic(rays), atol=1e-2)

    def test_southern_directions_use_the_mirror(self, pattern):
        northern = np.array([[0.3, 0.4, np.sqrt(1 - 0.25)]])
        assert np.allclose(
            pattern.interpolate_directions(northern),
            pattern.interpolate_directions(-northern),
        )

    def test_preserves_the_ensemble(self):
        directions = SquareLambertProjection().grid(61)
        values = self.analytic(directions)
        pattern = SphericalPattern(
            np.stack([values, 2 * values]),
            directions,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        )
        got = pattern.interpolate_directions(directions[:50])
        assert got.shape == (2, 50)
        assert np.allclose(got[1], 2 * got[0])

    def test_rejects_a_bad_shape(self, pattern):
        with pytest.raises(ValueError, match=r"shape \(M, 3\)"):
            pattern.interpolate_directions(np.zeros((4, 2)))

    def test_refuses_a_lazy_pattern(self):
        atoms = silicon_slab()
        grid = BackscatterDetector(max_angle=50, gpts=6)
        lazy = make_ebsd(atoms, BackscatterDetector(directions=grid.directions)).build(
            lazy=True
        )
        with pytest.raises(RuntimeError, match=r"compute\(\) the pattern"):
            lazy.interpolate_directions(grid.directions)


class TestSamplingProjectionDefault:
    def test_defaults_to_lambert(self):
        builder = EBSDReferencePattern(
            ase.build.bulk("Si", "diamond", a=5.431),
            energy=30e3,
            slab_cell=(20.0, 20.0, 100.0),
            n_patches=400,
        )
        assert builder.projection.name == "lambert"

    def test_lambert_samples_solid_angle_uniformly(self):
        # A stereographic grid puts far more directions per steradian at the
        # equator than at the pole; an equal-area one does not.
        edges = np.cos(np.radians([0, 30, 60, 90]))[::-1]
        spread = {}
        for name in ("lambert", "stereographic"):
            z = validate_projection(name).grid(148)[:, 2]
            counts, _ = np.histogram(z, bins=edges)
            density = counts / (2 * np.pi * np.diff(edges))
            spread[name] = density.max() / density.min()
        assert spread["lambert"] < 1.2
        assert spread["stereographic"] > 2.0

    def test_lambert_sampling_covers_either_view(self):
        # The Lambert square reaches the near-equator azimuths a stereographic
        # disk never does, so it can be interpolated to either projection.
        directions = validate_projection("lambert").grid(121)
        pattern = SphericalPattern(1.0 + directions[:, 2], directions)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseProjectionWarning)
            pattern.interpolate(64, "lambert")
            pattern.interpolate(64, "stereographic")

    def test_show_interpolates_rather_than_bins(self):
        # Binning a Lambert-sampled pattern into a stereographic image empties
        # most of it; show must not do that.
        import matplotlib

        matplotlib.use("Agg")
        directions = validate_projection("lambert").grid(121)
        pattern = SphericalPattern(1.0 + directions[:, 2], directions)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseProjectionWarning)
            pattern.show(gpts=64, display=False)


class TestBungeRotation:
    def test_identity_at_zero(self):
        assert np.allclose(bunge_rotation((0.0, 0.0, 0.0)), np.eye(3))

    @pytest.mark.parametrize(
        "euler", [(0, 0, 0), (30, 54.7, 45), (110, 35, 20), (200, 80, 300)]
    )
    def test_is_a_proper_rotation(self, euler):
        g = bunge_rotation(euler)
        assert np.allclose(g @ g.T, np.eye(3))
        assert np.linalg.det(g) == pytest.approx(1.0)

    @pytest.mark.parametrize("euler", [(30, 54.7, 45), (110, 35, 20)])
    def test_matches_abtem_euler_to_rotation(self, euler):
        # Pins the documented equivalence, which involves both a transpose and
        # an angle order; getting either backwards mirrors every pattern.
        from abtem.atoms import euler_to_rotation

        expected = euler_to_rotation(
            *np.radians(euler), axes="zxz", convention="extrinsic"
        ).T
        assert np.allclose(bunge_rotation(euler), expected)

    def test_degrees_flag(self):
        assert np.allclose(
            bunge_rotation((30, 45, 60)),
            bunge_rotation(np.radians([30, 45, 60]), degrees=False),
        )

    def test_stacks(self):
        stacked = bunge_rotation([[0, 0, 0], [30, 45, 60]])
        assert stacked.shape == (2, 3, 3)
        assert np.allclose(stacked[1], bunge_rotation((30, 45, 60)))

    def test_rejects_a_bad_shape(self):
        with pytest.raises(ValueError, match=r"shape \(3,\) or \(N, 3\)"):
            bunge_rotation(np.zeros(4))


class TestEBSDGeometry:
    # Direction cosines from kikuchipy's _get_direction_cosines_for_fixed_pc,
    # which is validated against EMsoft. Pinned here so the convention cannot
    # drift without kikuchipy being a test dependency.
    KIKUCHIPY = np.array(
        [
            [
                [0.411739949235, -0.150712623042, 0.898752423896],
                [0.411949717925, -0.147416940529, 0.899202800010],
                [0.412154921799, -0.144119437244, 0.899643211638],
                [0.412355553407, -0.140820222859, 0.900073642769],
            ],
            [
                [0.414767614101, -0.150715134529, 0.897358776921],
                [0.414977487428, -0.147419397260, 0.897809114589],
                [0.415182762154, -0.144121839127, 0.898249502920],
                [0.415383430825, -0.140822569810, 0.898679925907],
            ],
            [
                [0.417790738425, -0.150715971719, 0.895955129878],
                [0.418000713757, -0.147420216198, 0.896405423430],
                [0.418206056856, -0.144122639782, 0.896845783126],
                [0.418406760259, -0.140823352153, 0.897276192963],
            ],
        ]
    )

    def test_matches_kikuchipy(self):
        geometry = EBSDGeometry(
            shape=(3, 4),
            detector_distance=15000.0,
            pattern_center=(2.0, -1.0),
            pixel_size=50.0,
            sample_tilt=70.0,
            camera_tilt=5.0,
            azimuthal_angle=8.0,
        )
        assert np.allclose(geometry.directions, self.KIKUCHIPY, atol=1e-11)

    def test_directions_are_unit_vectors(self):
        directions = EBSDGeometry(shape=(20, 30)).directions
        assert directions.shape == (20, 30, 3)
        assert np.allclose(np.linalg.norm(directions, axis=-1), 1.0)

    def test_the_pattern_centre_looks_along_the_tilted_normal(self):
        # With the pattern centre on axis, the middle of the detector looks
        # back along the sample normal tilted by sample_tilt.
        geometry = EBSDGeometry(shape=(101, 101), sample_tilt=70.0, camera_tilt=0.0)
        centre = geometry.directions[50, 50]
        assert centre[0] == pytest.approx(np.cos(np.radians(70.0)), abs=1e-6)
        assert centre[2] == pytest.approx(np.sin(np.radians(70.0)), abs=1e-6)

    def test_detector_to_sample_is_a_rotation(self):
        geometry = EBSDGeometry(sample_tilt=70.0, azimuthal_angle=12.0)
        rotation = geometry.detector_to_sample
        assert np.allclose(rotation @ rotation.T, np.eye(3))
        assert np.linalg.det(rotation) == pytest.approx(1.0)

    def test_moving_the_pattern_centre_shifts_the_rays(self):
        a = EBSDGeometry(shape=(20, 20), pattern_center=(0.0, 0.0)).directions
        b = EBSDGeometry(shape=(20, 20), pattern_center=(3.0, 0.0)).directions
        assert not np.allclose(a, b)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"shape": (0, 4)}, "shape must be positive"),
            ({"detector_distance": -1.0}, "detector_distance must be positive"),
            ({"pixel_size": 0.0}, "pixel_size must be positive"),
        ],
    )
    def test_rejects_bad_geometry(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            EBSDGeometry(**kwargs)


class TestDetectorProjection:
    @pytest.fixture
    def pattern(self):
        directions = SquareLambertProjection().grid(201)
        values = 1.0 + 0.5 * directions[:, 2] ** 2 + 0.35 * directions[:, 0]
        return SphericalPattern(values, directions, metadata={"energy": 30e3})

    def test_shape_and_units(self, pattern):
        geometry = EBSDGeometry(shape=(30, 40), pixel_size=50.0)
        patch = pattern.project(geometry)
        assert isinstance(patch, EBSDPatternImages)
        assert patch.array.shape == (30, 40)
        assert [axis.units for axis in patch.base_axes_metadata] == ["µm", "µm"]
        assert patch.sampling == (50.0, 50.0)

    def test_matches_a_direct_lookup(self, pattern):
        # project is the geometry plus interpolate_directions, nothing else.
        geometry = EBSDGeometry(shape=(12, 16))
        euler = (30.0, 54.7, 45.0)
        direct = pattern.interpolate_directions(
            geometry.rotated_directions(euler).reshape(-1, 3)
        ).reshape(12, 16)
        assert np.allclose(np.asarray(pattern.project(geometry, euler).array), direct)

    def test_orientation_changes_the_patch(self, pattern):
        geometry = EBSDGeometry(shape=(16, 16))
        a = np.asarray(pattern.project(geometry, (0.0, 0.0, 0.0)).array)
        b = np.asarray(pattern.project(geometry, (30.0, 54.7, 45.0)).array)
        assert not np.allclose(a, b)

    def test_several_orientations_at_once(self, pattern):
        geometry = EBSDGeometry(shape=(10, 12))
        euler = [[0.0, 0.0, 0.0], [30.0, 54.7, 45.0], [110.0, 35.0, 20.0]]
        patches = pattern.project(geometry, euler)
        assert patches.array.shape == (3, 10, 12)
        for i, e in enumerate(euler):
            one = pattern.project(geometry, e)
            assert np.allclose(patches.array[i], one.array)

    def test_values_are_a_modulation_about_one(self, pattern):
        patch = np.asarray(pattern.project(EBSDGeometry(shape=(20, 20))).array)
        assert 0.5 < patch.mean() < 2.0

    def test_poisson_noise_applies_to_the_patch(self, pattern):
        # The detector patch is where shot noise belongs; total_dose multiplies
        # every pixel, and these sit near one, so it is the counts per pixel of
        # a featureless specimen.
        patch = pattern.project(EBSDGeometry(shape=(24, 24)))
        noisy = patch.poisson_noise(total_dose=400, seed=1)
        recovered = np.asarray(noisy.array) / 400
        assert np.abs(recovered.mean() - np.asarray(patch.array).mean()) < 0.05
        assert recovered.std() > np.asarray(patch.array).std()
