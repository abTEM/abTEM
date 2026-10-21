import warnings

import ase
import numpy as np
import pytest
from utils import gpu

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
    zone_axis_rotation,
)
from abtem.ebsd.emission import EmissionSlices, backscatter_cross_section
from abtem.ebsd.emsoft import _conventional_cell
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
from abtem.inelastic.phonons import SOURCE_INDEX

# The calculations here run on grids far coarser than a converged pattern
# needs, to be quick, and the constructors warn about that; the tests of the
# warning itself catch it with pytest.warns.
pytestmark = pytest.mark.filterwarnings(
    "ignore::abtem.ebsd.sampling.AntialiasLossWarning"
)

SILICON = ase.build.bulk("Si", "diamond", a=5.431)
GAN = ase.build.bulk("GaN", "wurtzite", a=3.189, c=5.185, u=0.377)
PROJECTIONS = [StereographicProjection(), SquareLambertProjection()]


@pytest.fixture(autouse=True)
def eager_by_default():
    """abTEM ships lazy; most tests here want numbers, so pin it eager."""
    with abtem.config.set({"dask.lazy": False}):
        yield


def random_hemisphere_directions(n, seed=0):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions[:, 2] = np.abs(directions[:, 2])
    return directions


def silicon_slab(thickness=8.0, zone_axis=(0.0, 0.0, 1.0)):
    return rotated_slab(SILICON, np.array(zone_axis), (10.0, 10.0, thickness))[0]


def make_ebsd(atoms, detector, sampling=0.15, slice_thickness=1.0, **kwargs):
    """EBSD of `atoms` at 30 keV."""
    potential = abtem.Potential(
        atoms, sampling=sampling, slice_thickness=slice_thickness, projection="finite"
    )
    return EBSD(potential, detector, 30e3, **kwargs)


def lightings(atoms):
    """Every atom lit alike, and a random but fixed lighting of `atoms`."""
    return {
        "uniform": None,
        "illumination": np.random.default_rng(0).uniform(0.5, 1.5, size=len(atoms)),
    }


def tiny_reference(atoms=SILICON, **kwargs):
    """One patch about the pole: the smallest reference pattern there is."""
    defaults = dict(n_patches=1, slab_cell=(6.0, 6.0, 4.0), gpts=8, direction_gpts=30)
    defaults["max_angle"] = 200.0
    # known, so that only the tests of the symmetry search need spglib
    defaults["centrosymmetric"] = atoms is not GAN
    return EBSDReferencePattern(atoms, 30e3, **{**defaults, **kwargs})


def detector_builder(**kwargs):
    """A 0.2 rad wide detector over several patches of about 60 mrad."""
    defaults = dict(
        slab_cell=(6.0, 6.0, 4.0), energy=30e3, n_patches=2000, sampling=0.25
    )
    atoms = kwargs.pop("atoms", SILICON)
    return EBSDDetectorPattern(
        atoms,
        EBSDGeometry(shape=(6, 6), detector_distance=15000.0, pixel_size=500.0),
        (30.0, 54.7, 45.0),
        **{**defaults, **kwargs},
    )


def direct_patch(builder, directions, atoms=None):
    """A reference pattern's one patch calculated by hand.

    The electrons leave along d; by reciprocity the wave comes in along -d, in
    a slab cut against the patch.
    """
    slab, rotation = rotated_slab(
        builder.atoms, -builder.zone_axes[0], builder.slab_cell
    )
    return EBSD(
        abtem.Potential(
            slab if atoms is None else atoms(slab),
            gpts=builder.potential_gpts,
            slice_thickness=1.0,
            projection="finite",
        ),
        BackscatterDetector(directions=-directions @ rotation.T),
        energy=30e3,
    ).build()


@pytest.fixture(scope="module")
def si_reference():
    return tiny_reference().build(pbar=False, lazy=False)


@pytest.fixture(scope="module")
def uniform():
    """The detector builder's pattern with every atom lit alike."""
    return detector_builder().build(lazy=False)


@pytest.fixture
def featureless(monkeypatch):
    """Atoms that emit, in a specimen that does not scatter.

    The atoms emit as usual, but the waves travel through a potential of zeros
    sliced the same way, so they stay plane waves.
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


# --------------------------------------------------------------------------
# projections.py
# --------------------------------------------------------------------------


class TestProjections:
    def test_both_projections_round_trip_the_hemisphere(self):
        directions = random_hemisphere_directions(500)
        for projection in PROJECTIONS:
            xy = projection.project(directions)
            assert np.allclose(projection.unproject(xy), directions), projection.name
            assert np.all(np.abs(xy) <= 1.0 + 1e-12), projection.name
            assert np.allclose(projection.project([[0.0, 0.0, 1.0]]), 0.0)
            grid = projection.grid(64)
            assert np.allclose(np.linalg.norm(grid, axis=1), 1.0), projection.name
            assert np.all(grid[:, 2] >= -1e-12), projection.name

    def test_stereographic_is_a_disk(self):
        theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        equator = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], 1)
        xy = StereographicProjection().project(equator)
        assert np.allclose(np.linalg.norm(xy, axis=1), 1.0)
        # pi / 4 of the square's points lie in the inscribed disk
        assert 0.75 < len(StereographicProjection().grid(64)) / 64**2 < 0.81

    def test_lambert_is_equal_area(self):
        # A uniform distribution on the hemisphere lands uniformly on the square.
        xy = SquareLambertProjection().project(
            random_hemisphere_directions(400_000, seed=3)
        )
        counts, _, _ = np.histogram2d(
            xy[:, 0], xy[:, 1], bins=8, range=[[-1, 1], [-1, 1]]
        )
        assert counts.std() / counts.mean() < 0.02
        assert len(SquareLambertProjection().grid(64)) == 64**2

        # so its grid samples solid angle evenly, where a stereographic one
        # crowds the equator
        edges = np.cos(np.radians([0, 30, 60, 90]))[::-1]
        spread = {}
        for name in ("lambert", "stereographic"):
            z = validate_projection(name).grid(148)[:, 2]
            counts, _ = np.histogram(z, bins=edges)
            density = counts / (2 * np.pi * np.diff(edges))
            spread[name] = density.max() / density.min()
        assert spread["lambert"] < 1.2 and spread["stereographic"] > 2.0

    def test_validate_projection(self):
        assert isinstance(validate_projection("lambert"), SquareLambertProjection)
        instance = StereographicProjection()
        assert validate_projection(instance) is instance
        with pytest.raises(ValueError, match="projection must be one of"):
            validate_projection("gnomonic")

    def test_pixel_centres_follow_the_binned_image(self):
        centers = pixel_centers(4)
        assert centers.shape == (16, 2)
        assert (centers[:, 0].min(), centers[:, 0].max()) == pytest.approx(
            (-0.75, 0.75)
        )
        # first axis varies x, second varies y, as histogram2d bins them
        centers = pixel_centers(5).reshape(5, 5, 2)
        assert centers[1, 0, 0] < centers[2, 0, 0]
        assert centers[0, 1, 1] < centers[0, 2, 1]

    def test_bin_directions(self):
        pole = np.array([[0.0, 0.0, 1.0]])
        near = np.array([[0.0, 0.0, 1.0], [1e-6, 1e-6, 1.0]])
        near /= np.linalg.norm(near, axis=1, keepdims=True)
        # averaged within a pixel
        assert bin_directions(near, np.array([1.0, 3.0]), gpts=2).sum() == 2.0
        # empty pixels are zero, and the southern hemisphere is dropped
        both = np.concatenate([pole, -pole])
        image, counts = bin_directions(
            both, np.array([5.0, 100.0]), gpts=8, return_counts=True
        )
        assert (image != 0).sum() == 1 and image.max() == pytest.approx(5.0)
        assert counts.sum() == 1

        for directions, values, match in [
            (np.zeros((3, 3)), np.zeros(2), "values has length"),
            (np.zeros((3, 2)), np.zeros(3), r"shape \(N, 3\)"),
        ]:
            with pytest.raises(ValueError, match=match):
                bin_directions(directions, values, gpts=4)

    def test_fibonacci_hemisphere(self):
        directions = fibonacci_hemisphere(1000)
        assert directions.shape == (1000, 3)
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
        assert np.all(directions[:, 2] > 0.0)
        # equal spacing in z is equal solid angle
        assert np.allclose(np.diff(directions[:, 2]), np.diff(directions[:, 2])[0])
        with pytest.raises(ValueError, match="n must be at least 1"):
            fibonacci_hemisphere(0)


# --------------------------------------------------------------------------
# orientations.py
# --------------------------------------------------------------------------


def empty_fraction(slab, cell, voxel=4.0):
    """Fraction of a slab's voxels holding no atom: vacuum where crystal should be."""
    bins = [max(1, int(round(c / voxel))) for c in cell]
    counts, _ = np.histogramdd(
        slab.positions, bins=bins, range=[(0.0, c) for c in cell]
    )
    return float(np.mean(counts == 0))


class TestOrientations:
    def test_zone_axis_rotation(self):
        for zone_axis in [(0, 0, 1), (1, 1, 1), (1, 0, 0), (2, 2, 3), (-1, 2, -3)]:
            unit = np.array(zone_axis, dtype=float) / np.linalg.norm(zone_axis)
            rotation = zone_axis_rotation(unit)
            assert np.allclose(rotation @ unit, [0.0, 0.0, 1.0]), zone_axis
            assert np.allclose(rotation @ rotation.T, np.eye(3)), zone_axis
            assert np.linalg.det(rotation) == pytest.approx(1.0), zone_axis

            # rotated_slab turns the atoms with ASE but returns this matrix
            # for the directions, so the two must agree
            atoms = ase.Atoms("H", positions=[(0, 0, 0)], cell=np.eye(3))
            atoms.rotate(unit, "z", rotate_cell=True)
            assert np.allclose(atoms.cell[:], rotation.T), zone_axis

        with pytest.raises(ValueError, match="must be nonzero"):
            zone_axis_rotation(np.zeros(3))
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            zone_axis_rotation(np.zeros(2))

    def test_a_slab_fills_its_cell_along_its_zone_axis(self):
        cell = (10.0, 12.0, 20.0)
        zone_axis = np.array([1.0, 2.0, 3.0])
        slab, rotation = rotated_slab(SILICON, zone_axis, cell)
        assert np.allclose(slab.cell.lengths(), cell)
        assert len(slab) > 0
        assert np.all((slab.positions >= 0.0) & (slab.positions <= np.array(cell)))
        assert np.allclose(rotation @ zone_axis / np.linalg.norm(zone_axis), [0, 0, 1])

    def test_a_slab_has_the_crystals_density(self):
        # A slab this small holds a whole number of atomic planes, so its count
        # depends on where the box falls against them; averaged over where the
        # cut is made, it has to come out at the density.
        cell = (12.0, 12.0, 24.0)
        rng = np.random.default_rng(0)
        counts = [
            len(
                rotated_slab(
                    SILICON,
                    np.array([1.0, 1.0, 1.0]),
                    cell,
                    origin=rng.random(3) @ SILICON.cell.array,
                )[0]
            )
            for _ in range(24)
        ]
        expected = len(SILICON) / SILICON.get_volume() * float(np.prod(cell))
        assert np.mean(counts) == pytest.approx(expected, rel=0.03)

    def test_the_block_holds_the_slab_at_any_orientation(self):
        # The slab is cut about the centroid of the block, whichever way it is
        # turned, so the block must hold a ball of the slab's diagonal about
        # that point. What bounds the ball is the separation of opposite
        # faces, which a non-orthogonal cell makes shorter than its edges.
        cell = (12.0, 12.0, 30.0)
        for name, atoms in {
            "fcc primitive": SILICON,
            "cubic": ase.build.bulk("Si", "diamond", a=5.431, cubic=True),
            "hcp": ase.build.bulk("Mg", "hcp", a=3.21, c=5.21),
            "triclinic": ase.Atoms(
                "Cu", cell=[[4.0, 0.0, 0.0], [1.5, 3.5, 0.0], [0.8, 1.1, 3.9]]
            ),
        }.items():
            block = atoms * estimate_repetitions(atoms, cell)
            lattice = np.asarray(block.cell.array)
            volume = abs(np.linalg.det(lattice))
            heights = volume / np.linalg.norm(
                np.cross(np.roll(lattice, -1, 0), np.roll(lattice, -2, 0)), axis=1
            )
            fractional = np.linalg.solve(lattice.T, block.positions.mean(axis=0))
            clearance = np.minimum(fractional, 1.0 - fractional) * heights
            assert np.all(clearance >= np.linalg.norm(cell) / 2.0), name

    def test_no_orientation_leaves_vacuum_in_the_slab(self):
        # sizing the block on edge lengths instead of face separations leaves
        # vacuum in the corners of 40 x 40 x 100 Å slabs
        for cell in [(10.0, 10.0, 40.0), (40.0, 40.0, 100.0)]:
            block = bulk_block(SILICON, cell)
            expected = len(SILICON) / SILICON.get_volume() * float(np.prod(cell))
            for zone_axis in fibonacci_hemisphere(60):
                slab, _ = rotated_slab(block, zone_axis, cell, repetitions=(1, 1, 1))
                assert empty_fraction(slab, cell) == 0.0, (cell, zone_axis)
                if cell[0] >= 40.0:
                    # one (111) double layer of this slab is 3% of it
                    assert len(slab) == pytest.approx(expected, rel=0.04), zone_axis

    def test_a_reused_block_gives_the_same_slab_and_is_left_alone(self):
        cell = (10.0, 10.0, 20.0)
        zone_axis = np.array([1.0, 2.0, 3.0])
        block = bulk_block(SILICON, cell, (12, 12, 12))
        before = block.positions.copy()

        reused, _ = rotated_slab(block, zone_axis, cell, repetitions=(1, 1, 1))
        rebuilt, _ = rotated_slab(SILICON, zone_axis, cell, repetitions=(12, 12, 12))

        assert np.array_equal(reused.positions, rebuilt.positions)
        assert np.array_equal(reused.numbers, rebuilt.numbers)
        assert np.array_equal(block.positions, before)
        assert len(bulk_block(SILICON, cell)) == len(
            SILICON * estimate_repetitions(SILICON, cell)
        )

    def test_an_origin_anchors_a_feature_at_the_slab_centre(self):
        block = ase.build.bulk("Si", "diamond", a=5.431, cubic=True) * (9, 9, 9)
        target = np.array([12.0, 30.0, 20.0])
        cell = np.array([14.0, 14.0, 14.0])

        for zone_axis in [(0, 0, 1), (1, 1, 1), (2, 2, 3)]:
            # the feature offset from the origin lands where the rotation says
            offset = np.array([3.0, -2.0, 1.0])
            marked = block + ase.Atom("Au", position=target + offset)
            slab, rotation = rotated_slab(
                marked,
                np.array(zone_axis, dtype=float),
                tuple(cell),
                repetitions=(1, 1, 1),
                origin=target,
            )
            gold = [a.position for a in slab if a.symbol == "Au"]
            assert len(gold) == 1, zone_axis
            assert np.allclose(gold[0], rotation @ offset + cell / 2, atol=1e-6)

        # The default anchor is the centroid, which an off-centre feature
        # misses; that is what `origin` exists to fix.
        marked = block + ase.Atom("Au", position=target)
        slab, _ = rotated_slab(
            marked, np.array([1.0, 1.0, 1.0]), tuple(cell), repetitions=(1, 1, 1)
        )
        assert not [a for a in slab if a.symbol == "Au"]
        explicit, _ = rotated_slab(
            marked,
            np.array([1.0, 1.0, 1.0]),
            tuple(cell),
            repetitions=(1, 1, 1),
            origin=marked.positions.mean(axis=0),
        )
        assert np.allclose(slab.positions, explicit.positions, atol=1e-9)

    def test_an_origin_in_a_repeated_cell_is_moved_to_the_middle(self):
        # a point of the unrepeated cell is at the corner of the block
        cell = (10.0, 10.0, 40.0)
        slab, _ = rotated_slab(
            SILICON, np.array([1.0, 0.0, 1.0]), cell, origin=np.zeros(3)
        )
        assert empty_fraction(slab, cell) == 0.0
        # and it is still the requested point: the atom at the origin of the
        # cell sits at the centre of the slab
        centre = np.array(cell) / 2.0
        assert np.min(np.linalg.norm(slab.positions - centre, axis=1)) < 1e-6

        repetitions = (15, 15, 15)
        point = np.array([0.3, -0.2, 0.1])
        moved = central_origin(SILICON, repetitions, point)
        shift = np.linalg.solve(SILICON.cell.array.T, moved - point)
        assert np.allclose(shift, np.round(shift))
        fractional = np.linalg.solve((SILICON * repetitions).cell.array.T, moved)
        assert np.all(np.abs(fractional - 0.5) <= 0.5 / np.array(repetitions) + 1e-9)

        # a model holding one feature is not repeated, and stays where it is
        point = 0.4 * SILICON.cell.array.sum(axis=0)
        assert np.allclose(central_origin(SILICON, (1, 1, 1), point), point)

    def test_rejects_a_bad_cell_or_origin(self):
        zone_axis = np.array([0.0, 0.0, 1.0])
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            rotated_slab(SILICON, zone_axis, (10.0, 10.0))
        with pytest.raises(ValueError, match=r"origin must have shape \(3,\)"):
            rotated_slab(SILICON, zone_axis, (10.0, 10.0, 10.0), origin=np.zeros(2))

    def test_thermal_displacements_follow_each_atom_into_a_slab(self):
        # Displacements given per site follow each atom into the block and out
        # into a slab cut at an arbitrary orientation: only the atoms of the
        # second site may move.
        frozen = abtem.FrozenPhonons(SILICON, num_configs=2, sigmas=[0.0, 0.1], seed=3)
        crystal, frozen_phonons = _crystal_and_displacements(frozen)
        cell = (8.0, 8.0, 6.0)
        slab, rotation = rotated_slab(
            bulk_block(crystal, cell), np.array([1.0, 1.0, 1.0]), cell, (1, 1, 1)
        )
        displaced = _displaced(slab, rotation, frozen_phonons)
        moved = np.linalg.norm(
            displaced._randomize_transformed(slab).positions - slab.positions, axis=1
        )
        assert np.array_equal(moved > 0, slab.arrays["cell_index"] == 1)
        assert (displaced.num_configs, displaced.seed) == (2, frozen_phonons.seed)

        # Anisotropic displacements and restricted directions are the crystal's,
        # whatever the slab's orientation: displaced only along the crystal's z,
        # or only in its xy-plane, as seen back in the crystal frame.
        for kwargs, kept in (
            (dict(sigmas={"Si": (0.0, 0.0, 0.1)}), [False, False, True]),
            (dict(sigmas=0.1, directions="xy"), [True, True, False]),
        ):
            _, frozen_phonons = _crystal_and_displacements(
                abtem.FrozenPhonons(SILICON, num_configs=2, seed=3, **kwargs)
            )
            displaced = _displaced(slab, rotation, frozen_phonons)
            shifts = displaced._randomize_transformed(slab).positions - slab.positions
            in_crystal = np.abs(shifts @ rotation).max(axis=0)
            assert np.array_equal(in_crystal > 1e-9, kept), kwargs

        # given per element, they apply to any slab
        _, by_element = _crystal_and_displacements(
            abtem.FrozenPhonons(SILICON, num_configs=2, sigmas=0.08, seed=3)
        )
        assert _displaced(silicon_slab(), rotation, by_element).num_configs == 2

        # the configurations of a yield are averaged, whatever was asked
        with pytest.warns(UserWarning, match="ensemble_mean=False is ignored"):
            _crystal_and_displacements(
                abtem.FrozenPhonons(SILICON, 2, sigmas=0.08, ensemble_mean=False)
            )

    def test_detects_an_inversion_centre(self):
        pytest.importorskip("spglib")
        assert is_centrosymmetric(SILICON)
        assert not is_centrosymmetric(GAN)
        # spglib finds no symmetry for atoms on top of each other, and either
        # returns None or, in its newer error mode, raises
        spglib = pytest.importorskip("spglib")
        overlapping = ase.Atoms("Si2", positions=np.zeros((2, 3)), cell=[4.0] * 3)
        for old_error_handling in (True, False):
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(
                    spglib.error, "OLD_ERROR_HANDLING", old_error_handling
                )
                with pytest.raises(ValueError, match="could not determine"):
                    is_centrosymmetric(overlapping)
                with pytest.raises(ValueError, match="could not standardize"):
                    _conventional_cell(overlapping, None)


# --------------------------------------------------------------------------
# detectors.py and geometry.py
# --------------------------------------------------------------------------


class TestDetectorGeometry:
    def test_a_grid_detector(self):
        detector = BackscatterDetector(max_angle=200, gpts=9)
        directions = detector.directions
        assert detector.is_grid and detector.base_shape == (9, 9)
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
        assert np.all(directions[:, 2] > 0.0)
        # centred on the slab's axis, out to max_angle
        assert np.allclose(directions.reshape(9, 9, 3)[4, 4], [0.0, 0.0, 1.0])
        wide = BackscatterDetector(max_angle=200, gpts=64)
        assert np.abs(wide.directions[:, :2]).max() == pytest.approx(
            np.sin(0.2), rel=0.05
        )
        # its wave vectors scale with the wave number, on a regular grid
        for energy in (30e3, 200e3):
            vectors = detector.transverse_wave_vectors(energy)
            assert np.allclose(vectors, directions[:, :2] / energy2wavelength(energy))
            steps = np.diff(np.unique(np.round(vectors[:, 0], 9)))
            assert np.allclose(steps, detector.reciprocal_sampling(energy))

    def test_explicit_directions(self):
        detector = BackscatterDetector(directions=np.array([[0.0, 0.0, 2.0]] * 5))
        assert np.allclose(detector.directions, [0.0, 0.0, 1.0])
        assert not detector.is_grid and detector.base_shape == (5,)
        with pytest.raises(RuntimeError, match="no regular sampling"):
            detector.reciprocal_sampling(30e3)

    def test_rejects_bad_detectors(self):
        with pytest.warns(UserWarning, match="small-angle approximation"):
            BackscatterDetector(max_angle=700, gpts=5)
        for kwargs, match in [
            ({}, "give either"),
            ({"max_angle": 100}, "must be given together"),
            ({"gpts": 4}, "must be given together"),
            ({"max_angle": 100, "gpts": 4, "directions": np.zeros((1, 3))}, "not both"),
            ({"directions": np.zeros((3, 2))}, r"shape \(N, 3\)"),
            ({"directions": np.array([[0.0, 0.0, -1.0]])}, "positive z component"),
        ]:
            with pytest.raises(ValueError, match=match):
                BackscatterDetector(**kwargs)

    def test_bunge_rotation(self):
        from abtem.atoms import euler_to_rotation

        assert np.allclose(bunge_rotation((0.0, 0.0, 0.0)), np.eye(3))
        for euler in [(30, 54.7, 45), (110, 35, 20), (200, 80, 300)]:
            g = bunge_rotation(euler)
            assert np.allclose(g @ g.T, np.eye(3)) and np.linalg.det(g) > 0, euler
            # the documented equivalence involves a transpose and an angle
            # order; getting either backwards mirrors every pattern
            expected = euler_to_rotation(
                *np.radians(euler), axes="zxz", convention="extrinsic"
            ).T
            assert np.allclose(g, expected), euler
            assert np.allclose(g, bunge_rotation(np.radians(euler), degrees=False))

        stacked = bunge_rotation([[0, 0, 0], [30, 45, 60]])
        assert stacked.shape == (2, 3, 3)
        assert np.allclose(stacked[1], bunge_rotation((30, 45, 60)))
        with pytest.raises(ValueError, match=r"shape \(3,\) or \(N, 3\)"):
            bunge_rotation(np.zeros(4))

    def test_detector_directions_match_kikuchipy(self):
        # From kikuchipy's _get_direction_cosines_for_fixed_pc, which is
        # validated against EMsoft; pinned so that the convention cannot drift
        # without kikuchipy being a test dependency.
        expected = np.array(
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
        geometry = EBSDGeometry(
            shape=(3, 4),
            detector_distance=15000.0,
            pattern_center=(2.0, -1.0),
            pixel_size=50.0,
            sample_tilt=70.0,
            camera_tilt=5.0,
            azimuthal_angle=8.0,
        )
        assert np.allclose(geometry.directions, expected, atol=1e-11)

    def test_detector_geometry(self):
        # With the pattern centre on axis, the middle of the detector looks
        # back along the sample normal tilted by sample_tilt.
        geometry = EBSDGeometry(shape=(101, 101), sample_tilt=70.0)
        directions = geometry.directions
        assert directions.shape == (101, 101, 3)
        assert np.allclose(np.linalg.norm(directions, axis=-1), 1.0)
        assert directions[50, 50, 0] == pytest.approx(np.cos(np.radians(70.0)))
        assert directions[50, 50, 2] == pytest.approx(np.sin(np.radians(70.0)))

        rotation = EBSDGeometry(azimuthal_angle=12.0).detector_to_sample
        assert np.allclose(rotation @ rotation.T, np.eye(3))
        assert np.linalg.det(rotation) == pytest.approx(1.0)

        shifted = EBSDGeometry(shape=(101, 101), pattern_center=(3.0, 0.0))
        assert not np.allclose(directions, shifted.directions)

        for kwargs, match in [
            ({"shape": (0, 4)}, "shape must be positive"),
            ({"detector_distance": -1.0}, "detector_distance must be positive"),
            ({"pixel_size": 0.0}, "pixel_size must be positive"),
        ]:
            with pytest.raises(ValueError, match=match):
                EBSDGeometry(**kwargs)


# --------------------------------------------------------------------------
# measurements.py
# --------------------------------------------------------------------------


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


def sampled_in(projection, gpts, values_of=lambda d: 1.0 + d[:, 2], **kwargs):
    directions = validate_projection(projection).grid(gpts)
    return SphericalPattern(values_of(directions), directions, **kwargs)


def hemispheres(hemisphere, centrosymmetric=None):
    """A synthetic pattern with distinguishable northern and southern values."""
    northern = SquareLambertProjection().grid(21)
    north = 1.0 + 0.3 * northern[:, 0]
    metadata = {} if centrosymmetric is None else {"centrosymmetric": centrosymmetric}
    if hemisphere == "north":
        return SphericalPattern(north, northern, metadata=metadata)
    southern = northern * [1, 1, -1]
    return SphericalPattern(
        np.concatenate([north, 2.0 + 0.3 * southern[:, 1]]),
        np.concatenate([northern, southern]),
        metadata=metadata,
    )


class TestSphericalPattern:
    def test_construction_and_concatenation(self, tmp_path):
        pattern = sampled_in("lambert", 23, metadata={"energy": 30e3})
        n = len(pattern)
        halves = [
            SphericalPattern(pattern.array[: n // 2], pattern.directions[: n // 2]),
            SphericalPattern(pattern.array[n // 2 :], pattern.directions[n // 2 :]),
        ]
        joined = SphericalPattern.concatenate(halves)
        assert np.array_equal(joined.array, pattern.array)
        assert np.array_equal(joined.directions, pattern.directions)

        url = str(tmp_path / "pattern.zarr")
        pattern.to_zarr(url)
        restored = SphericalPattern.from_zarr(url)
        assert np.array_equal(restored.array, pattern.array)
        assert np.array_equal(restored.directions, pattern.directions)
        assert restored.metadata == pattern.metadata

        pair = SphericalPattern(
            np.stack([pattern.array] * 2),
            pattern.directions,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        )
        for call, match in [
            (lambda: SphericalPattern.concatenate([pattern, pair]), "mismatched"),
            (lambda: SphericalPattern.concatenate([]), "empty sequence"),
            (lambda: SphericalPattern(np.zeros(4), pattern.directions), "last axis"),
            (lambda: SphericalPattern(np.zeros((2, n)), pattern.directions), "axes"),
        ]:
            with pytest.raises(ValueError, match=match):
                call()

    def test_binning_into_an_image(self):
        directions = fibonacci_hemisphere(200)
        pattern = SphericalPattern(
            np.stack([np.ones(200), 2 * np.ones(200)]),
            directions,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        )
        with warnings.catch_warnings():
            # a Fibonacci lattice is no projection grid, so the image has holes
            warnings.simplefilter("ignore", SparseProjectionWarning)
            images = pattern.bin(16)
        assert images.array.shape == (2, 16, 16)
        assert images.array[1].max() == pytest.approx(2.0)
        assert [axis.units for axis in images.base_axes_metadata] == ["", ""]
        assert images.metadata["projection"] == "stereographic"

    def test_binning_warns_about_holes(self):
        # An even grid in one projection is uneven in the other, so binning
        # through the wrong one leaves a moire of empty pixels; so does asking
        # for more pixels than there are directions.
        for sampled, viewed in [
            ("stereographic", "lambert"),
            ("lambert", "stereographic"),
        ]:
            with pytest.warns(SparseProjectionWarning, match="no sampled direction"):
                sampled_in(sampled, 148).bin(128, viewed)
        with pytest.warns(SparseProjectionWarning):
            sampled_in("lambert", 32).bin(256, "lambert")

        # Through its own projection there are none; the corners outside the
        # stereographic disk do not count.
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseProjectionWarning)
            for projection in ("stereographic", "lambert"):
                image = sampled_in(projection, 148).bin(128, projection)
            assert np.count_nonzero(image.array) == 128**2
            stereographic = sampled_in("stereographic", 148).bin(128, "stereographic")
            assert np.count_nonzero(stereographic.array) < 128**2

    def test_interpolates_a_lambert_grid_as_emsoft_does(self):
        rng = np.random.default_rng(0)
        gpts = 21
        lambert = SquareLambertProjection()
        pattern = sampled_in("lambert", gpts, lambda d: rng.random(len(d)))

        # the master pattern EMsoft would hold: node values on the square
        xy = lambert.project(pattern.directions)
        nodes = np.rint((xy + 1) / 2 * (gpts - 1)).astype(int)
        master = np.empty((gpts, gpts))
        master[nodes[:, 0], nodes[:, 1]] = np.asarray(pattern.array)

        # and the edges and corners of the square, where the clamping acts
        targets = np.concatenate(
            [
                random_hemisphere_directions(300, seed=1),
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.6, 0.8, 0.0]],
            ]
        )
        expected = emsoft_interpolation(master, lambert.project(targets))
        assert np.allclose(pattern.interpolate_directions(targets), expected)

        # Bilinear in the square, which a triangulation is not: a product of
        # the two coordinates is not linear on either triangle of a cell.
        product = sampled_in(
            "lambert", gpts, lambda d: 2.0 + np.prod(lambert.project(d), 1)
        )
        xy = lambert.project(targets)
        assert np.allclose(
            product.interpolate_directions(targets), 2.0 + xy[:, 0] * xy[:, 1]
        )

        # A direction calculated twice, where patches overlap, is averaged.
        directions = lambert.grid(11)
        doubled = SphericalPattern(
            np.concatenate([np.ones(len(directions)), np.full(len(directions), 3.0)]),
            np.concatenate([directions, directions]),
        )
        assert np.allclose(doubled.interpolate_directions(directions), 2.0)

    def test_interpolates_other_samplings_by_triangulation(self):
        from abtem.ebsd.measurements import _lambert_grid

        lambert = SquareLambertProjection()
        directions = StereographicProjection().grid(41)
        assert _lambert_grid(lambert.project(directions)) is None
        # linear in the plane the triangulation works in
        pattern = SphericalPattern(
            1.0 + 0.5 * lambert.project(directions)[:, 0], directions
        )
        # well inside the disk the grid covers, so nothing is filled
        targets = random_hemisphere_directions(100, seed=2)
        targets[:, 2] += 1.5
        targets /= np.linalg.norm(targets, axis=1, keepdims=True)
        assert np.allclose(
            pattern.interpolate_directions(targets),
            1.0 + 0.5 * lambert.project(targets)[:, 0],
            atol=1e-9,
        )

    def test_interpolates_at_any_directions(self):
        def analytic(d):
            return 1.0 + 0.4 * d[:, 2] ** 2 + 0.25 * d[:, 0] * d[:, 1]

        pattern = sampled_in("lambert", 101, analytic)
        assert np.allclose(
            pattern.interpolate_directions(pattern.directions),
            analytic(pattern.directions),
        )
        between = random_hemisphere_directions(2000)
        error = np.abs(pattern.interpolate_directions(between) - analytic(between))
        assert error.mean() < 1e-3

        # A flat detector's rays are a gnomonic projection, on no square grid.
        axis = (np.arange(32) - 16) * 50.0
        x, y = np.meshgrid(axis, axis, indexing="ij")
        rays = np.stack([x.ravel(), y.ravel(), np.full(x.size, 15000.0)], axis=1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        assert np.allclose(
            pattern.interpolate_directions(rays), analytic(rays), atol=1e-2
        )

        pair = SphericalPattern(
            np.stack([pattern.array, 2 * pattern.array]),
            pattern.directions,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        )
        got = pair.interpolate_directions(between[:50])
        assert got.shape == (2, 50) and np.allclose(got[1], 2 * got[0])

        with pytest.raises(ValueError, match=r"shape \(M, 3\)"):
            pattern.interpolate_directions(np.zeros((4, 2)))

    def test_interpolates_onto_a_projection_grid(self):
        # On the grid the pattern was sampled on, the nodes are the values.
        gpts = 61
        lambert = SquareLambertProjection()
        pattern = sampled_in("lambert", gpts, lambda d: 1.0 + d[:, 2] + 0.3 * d[:, 0])
        nodes = np.linspace(-1, 1, gpts)
        x, y = np.meshgrid(nodes, nodes, indexing="ij")
        expected = lambert.unproject(np.stack([x.ravel(), y.ravel()], axis=1))
        image = np.asarray(pattern.interpolate(gpts, "lambert").array)
        assert np.allclose(image.ravel(), 1.0 + expected[:, 2] + 0.3 * expected[:, 0])

        # A Lambert sampling reaches either view; a stereographic one stops
        # short of the Lambert square's corners, which are filled and warned of.
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseProjectionWarning)
            pattern.interpolate(51, "lambert")
            pattern.interpolate(51, "stereographic")
        stereographic = sampled_in("stereographic", 101)
        with pytest.warns(SparseProjectionWarning, match="outside the sampled"):
            stereographic.interpolate(51, "lambert")
        image = np.asarray(stereographic.interpolate(51, "stereographic").array)
        assert image[0, 0] == 0.0 and image[25, 25] > 0.0

        pair = SphericalPattern(
            np.stack([pattern.array, 2 * pattern.array]),
            pattern.directions,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        )
        image = pair.interpolate(21, "lambert")
        assert image.array.shape == (2, 21, 21)
        assert np.allclose(image.array[1], 2 * image.array[0])

    def test_show_interpolates_rather_than_bins(self):
        # Binning a Lambert-sampled pattern into a stereographic image empties
        # most of it; show must not do that.
        import matplotlib

        matplotlib.use("Agg")
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseProjectionWarning)
            sampled_in("lambert", 121).show(gpts=64, display=False)

    def test_projects_onto_a_detector(self):
        pattern = sampled_in(
            "lambert", 201, lambda d: 1.0 + 0.5 * d[:, 2] ** 2 + 0.35 * d[:, 0]
        )
        geometry = EBSDGeometry(shape=(10, 12), pixel_size=50.0)
        euler = [[0.0, 0.0, 0.0], [30.0, 54.7, 45.0], [110.0, 35.0, 20.0]]

        patches = pattern.project(geometry, euler)
        assert isinstance(patches, EBSDPatternImages)
        assert patches.array.shape == (3, 10, 12)
        assert [axis.units for axis in patches.base_axes_metadata] == ["µm", "µm"]
        assert patches.sampling == (50.0, 50.0)
        # it is the geometry plus interpolate_directions, nothing else
        for i, angles in enumerate(euler):
            rays = geometry.rotated_directions(angles).reshape(-1, 3)
            direct = pattern.interpolate_directions(rays).reshape(10, 12)
            assert np.allclose(np.asarray(patches.array)[i], direct), angles
        assert not np.allclose(patches.array[0], patches.array[1])

        # Shot noise belongs on the detector; these values sit near one, so
        # the dose is the counts per pixel of a featureless specimen.
        noisy = np.asarray(patches[1].poisson_noise(total_dose=400, seed=1).array)
        assert noisy.mean() / 400 == pytest.approx(patches.array[1].mean(), abs=0.05)

    def test_stands_in_for_the_hemisphere_not_calculated(self):
        # The inversion image, silently, for a centrosymmetric crystal ...
        pattern = hemispheres("north", centrosymmetric=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            southern = pattern.southern
        assert southern.hemisphere == "south"
        assert np.allclose(southern.directions, -pattern.directions)
        assert np.allclose(southern.array, pattern.array)

        # ... with a warning for a polar one ...
        pattern = hemispheres("north", centrosymmetric=False)
        with pytest.warns(UserWarning, match="not centrosymmetric"):
            pattern.southern
        with pytest.warns(UserWarning, match="not centrosymmetric"):
            pattern.interpolate_directions(np.array([[0.1, 0.2, -0.97]]))

        # ... and the calculated one when there is one.
        pattern = hemispheres("both", centrosymmetric=False)
        southern = SquareLambertProjection().grid(21)[:40] * [1, 1, -1]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            values = pattern.interpolate_directions(southern)
        assert np.allclose(values, 2.0 + 0.3 * southern[:, 1])
        # a northern-only pattern mirrors a southern direction onto its north
        north = sampled_in("lambert", 41)
        d = np.array([[0.3, 0.4, np.sqrt(0.75)]])
        assert np.allclose(
            north.interpolate_directions(d), north.interpolate_directions(-d)
        )

    def test_a_lazy_pattern_has_to_be_computed_first(self):
        grid = BackscatterDetector(max_angle=50, gpts=3)
        detector = BackscatterDetector(directions=grid.directions)
        lazy = make_ebsd(silicon_slab(thickness=2.0), detector).build(lazy=True)
        assert lazy.is_lazy
        for call in (
            lambda: lazy.bin(8),
            lambda: lazy.interpolate_directions(grid.directions),
        ):
            with pytest.raises(RuntimeError, match=r"compute\(\) the pattern"):
                call()
        computed = lazy.compute()
        assert not computed.is_lazy and computed.compute() is computed


# --------------------------------------------------------------------------
# emission.py
# --------------------------------------------------------------------------


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
        total = np.asarray(emitted, dtype=np.float64).sum(axis=(-2, -1))
        return total * self.SAMPLING**2

    def test_the_cross_section_is_rutherfords_z_squared(self):
        # The electron scattering factor at q = 2 / wavelength, squared: that
        # far inside the screening radius, the nucleus alone sets it.
        numbers = np.array([6, 14, 29, 79])
        assert np.allclose(
            backscatter_cross_section(numbers, 30e3), numbers**2, rtol=0.03
        )
        # relative to hydrogen, one per atom, in order
        assert backscatter_cross_section(np.array([1]), 30e3) == pytest.approx([1.0])
        cross_section = backscatter_cross_section(np.array([14, 6, 14]), 30e3)
        assert cross_section[0] == cross_section[2] > cross_section[1]

    def test_every_atom_emits_once_in_its_own_slice(self):
        slab = silicon_slab()
        slices = self.emission(slab)
        cross_section = backscatter_cross_section(slab.numbers, 30e3)
        owner = np.floor(slab.positions[:, 2]).astype(int)
        for index in range(len(slices)):
            assert np.sum(self.integral(slices[index])) == pytest.approx(
                cross_section[owner == index].sum(), rel=1e-4, abs=1e-3
            ), index
        assert np.allclose(
            slices.totals()[:, 0],
            [cross_section[owner == i].sum() for i in range(len(slices))],
        )

    def test_an_emitter_sits_exactly_at_its_atom(self):
        # Between grid points, where only the phase of its transform shows it.
        atoms = ase.Atoms("Si", positions=[(3.37, 6.81, 0.5)], cell=(10.0, 10.0, 1.0))
        transform = np.fft.fft2(np.asarray(self.emission(atoms)[0][0], np.float64))
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
        with pytest.raises(ValueError, match="values per source"):
            self.emission(slab, illumination=np.ones(3))

    def test_follows_the_configured_precision(self):
        slab = silicon_slab()
        for precision, dtype in [("float32", np.float32), ("float64", np.float64)]:
            with abtem.config.set({"precision": precision}):
                assert self.emission(slab)[0].dtype == dtype, precision


# --------------------------------------------------------------------------
# reciprocity.py
# --------------------------------------------------------------------------


class TestEBSD:
    def test_vacuum_generates_nothing(self):
        # Nothing is there to emit, which must come out as zero, not 0/0.
        vacuum = ase.Atoms(cell=(10.0, 10.0, 8.0), pbc=True)
        patterns = make_ebsd(
            vacuum, BackscatterDetector(max_angle=50, gpts=5), sampling=0.1
        ).build()
        assert np.all(np.isfinite(patterns.array))
        assert np.allclose(patterns.array, 0.0)

    def test_a_featureless_specimen_yields_one(self, featureless):
        # the normalization, for any lighting, in every direction (including
        # directions off the cell's reciprocal grid)
        slab = silicon_slab()
        detector = BackscatterDetector(max_angle=150, gpts=5)
        for name, illumination in lightings(slab).items():
            patterns = make_ebsd(slab, detector, illumination=illumination).build()
            assert np.allclose(patterns.array, 1.0, atol=1e-5), name

    def test_the_first_atoms_see_the_waves_unscattered(self):
        # Emission is read from the waves arriving at each slice, before the
        # step through it, so the first atoms see unscattered waves and the
        # deepest scattered ones.
        slab = silicon_slab()
        detector = BackscatterDetector(max_angle=150, gpts=4)
        occupied = np.unique(np.floor(slab.positions[:, 2]).astype(int))
        for name, illumination in lightings(slab).items():
            for index, unscattered in [(occupied[0], True), (occupied[-1], False)]:
                weights = np.zeros(8)
                weights[index] = 1.0
                patterns = make_ebsd(
                    slab, detector, illumination=illumination, depth_weight=weights
                ).build()
                ones = np.allclose(
                    patterns.array, 1.0, atol=1e-5 if unscattered else 1e-2
                )
                assert ones == unscattered, (name, index)

    def test_the_yield_is_converged_in_the_slicing(self):
        # Along [101] thin slices hold a few atoms each and thick ones many;
        # collected at the wrong plane the yield moved by a quarter between 1
        # and 0.25 Å slices.
        slab, _ = rotated_slab(SILICON, np.array([1.0, 0.0, 1.0]), (10.0, 10.0, 24.0))
        detector = BackscatterDetector(max_angle=50, gpts=6)
        mean = [
            float(
                np.mean(
                    make_ebsd(slab, detector, sampling=0.1, slice_thickness=thickness)
                    .build()
                    .array
                )
            )
            for thickness in (1.0, 0.25)
        ]
        assert mean[0] == pytest.approx(mean[1], rel=0.03)

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
        both = np.asarray(make_ebsd(slab, detector).build().array)
        per_species = [
            np.asarray(make_ebsd(slab, detector, illumination=lit).build().array)
            for lit in (silicon, 1.0 - silicon)
        ]
        cross_section = backscatter_cross_section(slab.numbers, 30e3)
        weights = np.array([cross_section @ silicon, cross_section @ (1.0 - silicon)])
        expected = np.tensordot(weights, per_species, (0, 0)) / weights.sum()
        assert np.allclose(both, expected, rtol=1e-5)
        assert not np.allclose(per_species[0], per_species[1], rtol=1e-2)

    def test_illuminations(self):
        slab = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        ebsd = make_ebsd(slab, detector)
        uniform = ebsd.build()
        assert ebsd.illumination is None
        assert uniform.metadata["source"] == "uniform"

        # only how the atoms are lit relative to each other matters
        even = make_ebsd(slab, detector, illumination=np.full(len(slab), 3.7)).build()
        assert even.metadata["source"] == "illumination"
        assert np.allclose(even.array, uniform.array, rtol=1e-5)

        # several add a leading axis
        illumination = np.random.default_rng(0).uniform(size=(3, len(slab)))
        together = make_ebsd(slab, detector, illumination=illumination).build()
        assert together.array.shape == (3, 4, 4)
        assert together.ensemble_axes_metadata[0].label == "illumination"
        for i in range(3):
            alone = make_ebsd(slab, detector, illumination=illumination[i]).build()
            assert np.allclose(together.array[i], alone.array, rtol=1e-5), i

        # The values are for the atoms as given, also when the potential
        # transforms them: the primitive cell is orthogonalized into copies of
        # its two atoms, each lit as the atom it came from.
        lit = make_ebsd(SILICON, detector, illumination=[1.0, 0.2]).build()
        potential = abtem.Potential(SILICON, sampling=0.15)
        source = potential._configuration()[0].arrays[SOURCE_INDEX]
        expected = make_ebsd(
            potential.get_transformed_atoms(),
            detector,
            illumination=np.array([1.0, 0.2])[source],
        ).build()
        assert len(source) == 4 * len(SILICON)
        assert np.allclose(lit.array, expected.array, rtol=1e-5)

    def test_frozen_phonon_configurations_are_averaged(self):
        # A potential built on an ensemble carries a configuration axis the
        # propagation cannot see, which generate_slices would otherwise walk
        # for the first configuration alone. Each configuration's atoms emit
        # from where that configuration put them.
        rng = np.random.default_rng(0)
        trajectory = []
        for _ in range(3):
            configuration = silicon_slab()
            configuration.positions += rng.normal(
                scale=0.2, size=(len(configuration), 3)
            )
            trajectory.append(configuration)
        detector = BackscatterDetector(max_angle=50, gpts=4)
        mean = np.mean(
            [np.asarray(make_ebsd(a, detector).build().array) for a in trajectory], 0
        )
        for lazy in (False, True):
            ensemble = make_ebsd(abtem.AtomsEnsemble(trajectory), detector).build(
                lazy=lazy
            )
            array = np.asarray(ensemble.compute().array if lazy else ensemble.array)
            assert array.shape == (4, 4)
            assert np.allclose(array, mean, rtol=1e-5), lazy

        with pytest.warns(UserWarning, match="incoherent sum"):
            make_ebsd(
                abtem.AtomsEnsemble(trajectory, ensemble_mean=False), detector
            ).build()

    def test_grid_and_explicit_directions_agree(self):
        # Two descriptions of the same directions: only the type differs.
        slab = silicon_slab()
        grid = BackscatterDetector(max_angle=50, gpts=4)
        from_grid = make_ebsd(slab, grid).build()
        from_explicit = make_ebsd(
            slab, BackscatterDetector(directions=grid.directions)
        ).build()

        assert isinstance(from_grid, abtem.DiffractionPatterns)
        assert from_grid.array.shape == (4, 4)
        assert isinstance(from_explicit, SphericalPattern)
        assert np.allclose(from_explicit.directions, grid.directions)
        assert np.allclose(from_grid.array.ravel(), from_explicit.array, rtol=1e-6)

    def test_steep_waves_are_not_cut_off_by_the_aperture(self):
        # a wave's direction uses none of the antialias aperture: at this
        # sampling, 250 and 300 mrad lie outside it
        angles = np.array([0.0, 150.0, 250.0, 300.0]) * 1e-3
        directions = np.column_stack(
            [np.sin(angles), np.zeros_like(angles), np.cos(angles)]
        )
        patterns = make_ebsd(
            silicon_slab(), BackscatterDetector(directions=directions), sampling=0.1
        ).build()
        assert np.all((0.8 < patterns.array) & (patterns.array < 1.5))

    def test_batching_does_not_change_the_result(self):
        slab = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=6)
        whole = np.asarray(
            make_ebsd(slab, detector).build(max_batch_directions=10_000).array
        )
        for lazy in (False, True):
            split = make_ebsd(slab, detector).build(max_batch_directions=4, lazy=lazy)
            if lazy:
                # abTEM requires the base axes of a measurement to be one chunk
                assert split.array.chunks[-2:] == ((6,), (6,))
                split = split.compute()
            assert np.allclose(np.asarray(split.array), whole, rtol=1e-5), lazy

    def test_lazily(self, monkeypatch):
        slab = silicon_slab()
        grid = BackscatterDetector(max_angle=50, gpts=6)
        for detector, kwargs in [
            (grid, {}),
            (BackscatterDetector(directions=grid.directions), {}),
            (grid, {"backscatter_energy": [30e3, 28e3]}),
        ]:
            eager = make_ebsd(slab, detector, **kwargs).build(lazy=False)
            lazy = make_ebsd(slab, detector, **kwargs).build(
                lazy=True, max_batch_directions=8
            )
            assert lazy.is_lazy and lazy.array.shape == eager.array.shape
            assert np.allclose(
                np.asarray(lazy.compute().array), np.asarray(eager.array)
            )

        # eager and lazy differ by rounding, but each path is reproducible
        first, second = (
            make_ebsd(slab, grid).build(lazy=True, max_batch_directions=8).compute()
            for _ in range(2)
        )
        assert np.array_equal(np.asarray(first.array), np.asarray(second.array))

        # and nothing runs until computed
        calls = []
        propagate = EBSD._propagate_block

        def counted(self, *args, **kwargs):
            calls.append(1)
            return propagate(self, *args, **kwargs)

        monkeypatch.setattr(EBSD, "_propagate_block", counted)
        lazy = make_ebsd(slab, grid).build(lazy=True, max_batch_directions=8)
        assert not calls
        lazy.compute()
        assert calls

    def test_follows_the_configured_precision(self):
        slab = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        arrays = {}
        for precision, dtype in [("float32", np.float32), ("float64", np.float64)]:
            with abtem.config.set({"precision": precision}):
                for lazy in (False, True):
                    patterns = make_ebsd(slab, detector).build(lazy=lazy)
                    assert patterns.array.dtype == dtype, (precision, lazy)
                    arrays[precision] = np.asarray(patterns.compute().array)
        assert np.allclose(arrays["float32"], arrays["float64"], rtol=1e-4)

    def test_sampling_warnings(self):
        # The antialias aperture cuts the potential's scattering beyond it, and
        # silently: the transmission function is band-limited before it acts.
        detector = BackscatterDetector(max_angle=50, gpts=4)
        with pytest.warns(AntialiasLossWarning, match="scattering power"):
            make_ebsd(silicon_slab(), detector, sampling=0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", AntialiasLossWarning)
            make_ebsd(silicon_slab(), detector, sampling=potential_sampling(SILICON))

    def test_rejects_bad_arguments(self):
        detector = BackscatterDetector(max_angle=50, gpts=4)
        with pytest.raises(TypeError, match="energy"):
            EBSD(silicon_slab(), detector)
        for illumination, match in [
            (np.ones((2, 2, 2)), r"shape \(atoms,\) or \(sources, atoms\)"),
            (np.ones(3), "one value per atom"),
        ]:
            with pytest.raises(ValueError, match=match):
                make_ebsd(silicon_slab(), detector, illumination=illumination)
        # the atoms are what emits, and a built potential no longer has them
        potential = abtem.Potential(silicon_slab(), sampling=0.15, projection="finite")
        with pytest.raises(ValueError, match="does not carry its atoms"):
            EBSD(potential.build(lazy=False), detector, energy=30e3)


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

    def abtems(self):
        from abtem.antialias import antialias_aperture
        from abtem.multislice import _fresnel_propagator_array

        plain = _fresnel_propagator_array(
            self.THICKNESS, self.GPTS, self.SAMPLING, 30e3, "cpu", order="exact"
        )
        return plain, antialias_aperture(self.GPTS, self.SAMPLING, np)

    def test_along_the_axis_is_abtems_propagator(self):
        plain, aperture = self.abtems()
        assert np.allclose(self.kernel([[0.0, 0.0]])[0], plain * aperture, atol=1e-5)

        # and it leaves the envelope of a plane wave in any direction alone
        k0 = 1 / energy2wavelength(30e3)
        kernel = self.kernel([[0.1 * k0, 0.0], [0.2 * k0, -0.15 * k0]])
        assert np.allclose(kernel[:, 0, 0], 1.0)
        assert np.all(np.abs(kernel) <= 1.0 + 1e-6)

    def test_tilted_is_abtems_propagator_shifted(self):
        # For a tilt k on the grid, the envelope's component q propagates as
        # the wave's component q + k does, relative to the carrier's phase.
        plain, aperture = self.abtems()
        shift = 3
        kernel = self.kernel([[shift / (self.GPTS[0] * self.SAMPLING[0]), 0.0]])[0]
        rows = np.arange(-8, 9)  # small enough that q + k does not wrap around
        expected = plain[(rows + shift) % self.GPTS[0]] / plain[shift, 0]
        assert np.allclose(
            kernel[rows % self.GPTS[0]],
            expected * aperture[rows % self.GPTS[0]],
            atol=1e-5,
        )


class TestDepth:
    @pytest.fixture
    def setup(self):
        slab, _ = rotated_slab(SILICON, np.array([1.0, 0.0, 1.0]), (8.0, 8.0, 8.0))
        return slab, BackscatterDetector(max_angle=50, gpts=4)

    def test_depth_weights(self):
        depths = np.arange(1.0, 5.0)
        for weight, expected in [
            (None, [0.25] * 4),
            (np.ones(4), [0.25] * 4),
            ([1.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]),
            (lambda z: np.ones_like(z), [0.25] * 4),
            (2.0, np.exp(-depths / 2.0) / np.exp(-depths / 2.0).sum()),
        ]:
            assert np.allclose(_validate_depth_weight(weight, depths), expected), weight

        for weight, match in [
            (-1.0, "escape depth must be positive"),
            (np.ones(3), "one weight per slice"),
            (np.array([-1.0, 1.0, 1.0, 1.0]), "must not be negative"),
            (np.zeros(4), "must not be all zero"),
        ]:
            with pytest.raises(ValueError, match=match):
                _validate_depth_weight(weight, depths)

    def test_escape_depth(self, setup):
        slab, detector = setup
        uniform = make_ebsd(slab, detector).build().array
        explicit = make_ebsd(slab, detector, depth_weight=np.ones(8)).build().array
        deep = make_ebsd(slab, detector, depth_weight=1e6).build().array
        shallow = make_ebsd(slab, detector, depth_weight=2.0).build().array
        assert np.allclose(explicit, uniform)
        assert np.allclose(deep, uniform, rtol=1e-4)
        assert not np.allclose(shallow, uniform, rtol=1e-3)

    def test_a_short_escape_depth_stops_early(self):
        # Slices past the escape depth carry no weight, so propagating them is
        # wasted; the result changes by a small multiple of the discarded weight.
        slab = silicon_slab(thickness=40.0)
        detector = BackscatterDetector(max_angle=50, gpts=4)
        exact, truncated = (
            make_ebsd(slab, detector, depth_weight=4.0, depth_tolerance=t).build().array
            for t in (0.0, 1e-3)
        )
        assert np.allclose(exact, truncated, rtol=1e-2)
        # and under uniform weighting there is nothing to stop for
        slab = silicon_slab()
        assert np.array_equal(
            make_ebsd(slab, detector, depth_tolerance=0.0).build().array,
            make_ebsd(slab, detector).build().array,
        )

    def test_depth_bins(self, setup):
        slab, detector = setup
        binned = make_ebsd(slab, detector, depth_bins=4).build()
        assert binned.array.shape == (4, 4, 4)
        axis = binned.ensemble_axes_metadata[0]
        assert axis.label == "depth" and axis.units == "Å"
        assert np.allclose(axis.values, [1.0, 3.0, 5.0, 7.0])
        assert binned.metadata["depth_bins"] == [0.0, 2.0, 4.0, 6.0, 8.0]

        # weighted by their emission, the bins are the whole pattern ...
        emission = np.array(binned.metadata["depth_emission"])
        whole = make_ebsd(slab, detector).build().array
        recombined = np.tensordot(emission, binned.array, (0, 0)) / emission.sum()
        assert np.allclose(recombined, whole, rtol=1e-5)

        # ... and a depth weighting constant within each bin, applied
        # afterwards, equals the run weighted so
        weights = np.array([1.0, 0.5, 0.25, 0.125])
        recombined = np.tensordot(weights * emission, binned.array, (0, 0))
        recombined /= (weights * emission).sum()
        weighted = make_ebsd(slab, detector, depth_weight=np.repeat(weights, 2)).build()
        assert np.allclose(recombined, weighted.array, rtol=1e-5)

        # A callable sees the slices at the depths the bins do, their centres:
        # a cut-off at a bin edge keeps exactly the bins above it.
        shallow = make_ebsd(slab, detector, depth_weight=lambda z: z < 4.0).build()
        recombined = np.tensordot(emission[:2], binned.array[:2], (0, 0))
        assert np.allclose(recombined / emission[:2].sum(), shallow.array, rtol=1e-5)

        lazy = make_ebsd(slab, detector, depth_bins=4).build(lazy=True).compute()
        assert np.allclose(np.asarray(lazy.array), binned.array, rtol=1e-6)
        assert lazy.metadata["depth_emission"] == binned.metadata["depth_emission"]

        for depth_bins, match in [
            (np.array([0.0, 0.2, 8.0]), "at least one slice"),
            ([4.0, 2.0], "increasing"),
            (0, "at least 1"),
        ]:
            with pytest.raises(ValueError, match=match):
                make_ebsd(slab, detector, depth_bins=depth_bins).build()


class TestBackscatterEnergy:
    def test_resolving_the_energies(self):
        from abtem.distributions import uniform

        energies, weights = _validate_backscatter_energy(None, 30e3)
        assert (list(energies), list(weights)) == ([30e3], [1.0])
        _, weights = _validate_backscatter_energy(uniform(28e3, 30e3, 4), 30e3)
        assert weights.sum() == pytest.approx(1.0)
        # a backscattered electron has lost energy; it cannot have gained any
        for energies, match in [
            ([30e3, 31e3], "cannot carry more than the beam"),
            ([0.0], "must be positive"),
        ]:
            with pytest.raises(ValueError, match=match):
                _validate_backscatter_energy(energies, 30e3)

    def test_each_energy_is_its_own_calculation(self):
        slab = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        default = make_ebsd(slab, detector).build()
        assert np.array_equal(
            make_ebsd(slab, detector, backscatter_energy=30e3).build().array,
            default.array,
        )

        several = make_ebsd(slab, detector, backscatter_energy=[30e3, 26e3]).build()
        assert several.array.shape == (2, 4, 4)
        axis = several.ensemble_axes_metadata[0]
        assert isinstance(axis, EnergyAxis) and axis.values == (30e3, 26e3)
        assert np.allclose(several.array[0], default.array, rtol=1e-5)
        alone = make_ebsd(slab, detector, backscatter_energy=26e3).build()
        assert np.allclose(several.array[1], alone.array, rtol=1e-5)
        # the waves travel at the backscattered energy, so the diffraction
        # changes with it
        assert not np.allclose(several.array[0], several.array[1], rtol=1e-3)

        # energy before depth
        binned = make_ebsd(
            slab, detector, depth_bins=2, backscatter_energy=[30e3, 26e3]
        ).build()
        assert [a.label for a in binned.ensemble_axes_metadata] == ["Energy", "depth"]

    def test_energy_metadata_and_calibration(self):
        # `energy` is abTEM's: that of the electrons counted, unset beside an
        # EnergyAxis; `beam_energy` is the beam's. A grid detector's angles
        # are the same at every energy, and read back as such.
        slab = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        step = detector.reciprocal_sampling(30e3) * energy2wavelength(30e3) * 1e3

        single = make_ebsd(slab, detector, backscatter_energy=26e3).build()
        assert (single.metadata["energy"], single.metadata["beam_energy"]) == (
            26e3,
            30e3,
        )

        several = make_ebsd(slab, detector, backscatter_energy=[30e3, 26e3]).build()
        assert "energy" not in several.metadata
        assert several.metadata["beam_energy"] == 30e3
        assert several.metadata["backscatter_energies"] == [30e3, 26e3]
        assert several.metadata["energy_weights"] == pytest.approx([0.5, 0.5])

        for patterns in (single, several):
            assert patterns.angular_sampling == pytest.approx((step, step))

    def test_depth_weights_per_energy(self):
        slab = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        energies = [30e3, 26e3]

        def deeper_for_larger_losses(z, energy):
            return np.exp(-z / (2.0 + (30e3 - energy) / 1e3))

        rows = np.array([np.linspace(1.0, 0.2, 8), np.ones(8)])
        for weight, of_energy in [
            (
                deeper_for_larger_losses,
                lambda i, e: lambda z: deeper_for_larger_losses(z, e),
            ),
            (rows, lambda i, e: rows[i]),
        ]:
            together = make_ebsd(
                slab, detector, backscatter_energy=energies, depth_weight=weight
            ).build()
            for i, energy in enumerate(energies):
                alone = make_ebsd(
                    slab,
                    detector,
                    backscatter_energy=energy,
                    depth_weight=of_energy(i, energy),
                ).build()
                assert np.allclose(together.array[i], alone.array, rtol=1e-5), energy

        with pytest.raises(ValueError, match="rows but there are"):
            make_ebsd(
                slab,
                detector,
                backscatter_energy=energies,
                depth_weight=np.ones((3, 8)),
            ).build()


@pytest.mark.parametrize("device", [gpu])
class TestDevices:
    """On an accelerator (CUDA, or Metal through torch), as on the CPU.

    Kept small: on an accelerator the time goes with the number of operations.
    """

    @staticmethod
    def agree(build, device, lazies=(False,)):
        def host(result):
            array = result.array
            return array.get() if hasattr(array, "get") else np.asarray(array)

        expected = host(build("cpu", False))
        for lazy in lazies:
            result = build(device, lazy)
            got = host(result.compute() if lazy else result)
            assert np.allclose(got, expected, rtol=1e-4, atol=1e-6), lazy

    def test_the_engine(self, device):
        # every axis the engine can add: energy, depth and illumination
        slab = silicon_slab(thickness=4.0)
        illumination = np.random.default_rng(0).uniform(size=(2, len(slab)))

        def build(on, lazy):
            potential = abtem.Potential(
                slab, sampling=0.2, slice_thickness=2.0, projection="finite", device=on
            )
            return EBSD(
                potential,
                BackscatterDetector(max_angle=50, gpts=3),
                30e3,
                illumination=illumination,
                depth_bins=2,
                backscatter_energy=[30e3, 26e3],
                device=on,
            ).build(lazy=lazy)

        self.agree(build, device, lazies=(False, True))

    def test_the_builders(self, device):
        def reference(on, lazy):
            return EBSDReferencePattern(
                SILICON,
                30e3,
                slab_cell=(6.0, 6.0, 4.0),
                n_patches=1,
                gpts=4,
                direction_gpts=7,
                max_angle=200.0,
                hemisphere="both",
                centrosymmetric=True,
                device=on,
            ).build(lazy=lazy, pbar=False)

        def detector(on, lazy):
            # the incident beam on the device too, scanned
            probe = abtem.Probe(semiangle_cutoff=10, energy=30e3)
            return detector_builder(n_patches=None, sampling=0.25, device=on).scan(
                probe, scan=abtem.CustomScan([[0.0, 0.0], [1.0, 1.5]]), lazy=lazy
            )

        self.agree(reference, device)
        self.agree(detector, device)


# --------------------------------------------------------------------------
# reference.py
# --------------------------------------------------------------------------


class TestEBSDReferencePattern:
    def test_patch_geometry(self):
        # solid angle per patch goes as 1 / n, so the radius as 1 / sqrt(n),
        # and the caps must at least cover the hemisphere
        assert patch_half_angle(100) / patch_half_angle(400) == pytest.approx(2.0)
        for n_patches in (50, 200, 400, 1000):
            half_angle = patch_half_angle(n_patches) * 1e-3
            assert n_patches * (1.0 - np.cos(half_angle)) > 1.0, n_patches
        with pytest.raises(ValueError, match="n_patches must be at least 1"):
            patch_half_angle(0)

    def test_fft_friendly_gpts(self):
        for sampling in (0.05, 0.1414, 0.17, 0.3):
            for n in fft_friendly_gpts((10.0, 10.0), sampling):
                assert 10.0 / n <= sampling, sampling  # never coarser
                remainder = n
                for prime in (2, 3, 5, 7, 11):
                    while remainder % prime == 0:
                        remainder //= prime
                assert remainder == 1, n
        # 71 is prime, and its transform is slower than one twice the size
        assert fft_friendly_gpts((10.0, 10.0), 10.0 / 71) == (72, 72)
        assert fft_friendly_gpts((10.0, 20.0), 0.2) == (50, 100)

    def test_defaults(self):
        builder = EBSDReferencePattern(SILICON, 30e3, slab_cell=(10.0, 10.0, 40.0))
        assert builder.energy == 30e3
        assert builder.max_angle == pytest.approx(patch_half_angle(400))
        assert len(builder.zone_axes) == 400
        assert builder.projection.name == "lambert"
        assert builder.hemisphere == "north"
        assert builder.sampling == pytest.approx(potential_sampling(SILICON))
        assert builder.potential_gpts == fft_friendly_gpts(
            (10.0, 10.0), builder.sampling
        )

    def test_rejects_bad_arguments(self):
        # The right slab depends on the crystal and the accuracy wanted, and a
        # default is how an undersized one goes unnoticed.
        with pytest.raises(TypeError, match="slab_cell"):
            EBSDReferencePattern(SILICON, energy=30e3)
        for kwargs, match in [
            ({"slab_cell": (10.0, 10.0)}, "three positive lengths"),
            ({"slab_cell": (10.0, -10.0, 40.0)}, "three positive lengths"),
            ({"hemisphere": "east"}, "hemisphere must be"),
            ({"centrosymmetric": "yes"}, "centrosymmetric must be"),
        ]:
            with pytest.raises(ValueError, match=match):
                EBSDReferencePattern(
                    SILICON, 30e3, **{"slab_cell": (6.0, 6.0, 4.0), **kwargs}
                )
        with pytest.raises(TypeError, match="give Atoms, or FrozenPhonons"):
            EBSDReferencePattern(
                abtem.AtomsEnsemble([SILICON, SILICON]), 30e3, slab_cell=(6.0, 6.0, 4.0)
            )

    def test_every_direction_is_assigned_exactly_once(self):
        builder = EBSDReferencePattern(
            SILICON, 30e3, slab_cell=(20.0, 20.0, 100.0), n_patches=40, gpts=16
        )
        counts = np.bincount(
            np.concatenate(builder._assign_directions()),
            minlength=len(builder.directions),
        )
        assert np.all(counts == 1)

    def test_a_patch_is_a_direct_calculation(self, si_reference):
        # The builder is bookkeeping around EBSD. For GaN, which has no
        # inversion centre, calculating along +d instead would give the other
        # polar face.
        builder = tiny_reference()
        assert isinstance(si_reference, SphericalPattern)
        assert 0 < len(si_reference) <= len(builder.directions)
        cosines = si_reference.directions @ builder.zone_axes[0]
        assert np.all(cosines > np.cos(builder.max_angle * 1e-3))
        assert np.allclose(
            si_reference.array,
            direct_patch(builder, si_reference.directions).array,
            rtol=1e-6,
        )

        gan = tiny_reference(GAN)
        pattern = gan.build(pbar=False, lazy=False)
        assert np.allclose(
            pattern.array, direct_patch(gan, pattern.directions).array, rtol=1e-6
        )

        metadata = si_reference.metadata
        assert (metadata["n_patches"], metadata["max_angle"]) == (1, 200.0)
        assert (metadata["energy"], metadata["beam_energy"]) == (30e3, 30e3)
        assert (metadata["projection"], metadata["source"]) == ("lambert", "uniform")

        lazy = builder.build(lazy=True)
        assert lazy.is_lazy
        assert np.allclose(np.asarray(lazy.compute().array), si_reference.array)
        assert np.array_equal(lazy.directions, si_reference.directions)

    def test_an_origin_is_moved_into_the_block(self, si_reference):
        # The builder makes its own block, so it has to move the origin
        # itself; at the corner of the block every patch would be nearly
        # empty, and the yield would collapse towards vacuum.
        corner = tiny_reference(origin=np.zeros(3), direction_gpts=13)
        mean = np.asarray(corner.build(pbar=False, lazy=False).array).mean()
        assert mean == pytest.approx(np.asarray(si_reference.array).mean(), rel=0.25)

    def test_frozen_phonons_average_the_configurations(self, si_reference):
        frozen = abtem.FrozenPhonons(SILICON, num_configs=2, sigmas=0.1, seed=3)
        builder = tiny_reference(frozen)
        pattern = builder.build(pbar=False, lazy=False)
        assert builder.atoms is not frozen.atoms and len(builder.atoms) == 2
        assert pattern.metadata["num_configs"] == 2

        def displaced(slab):
            return abtem.FrozenPhonons(
                slab, num_configs=2, sigmas=0.1, seed=frozen.seed
            )

        direct = direct_patch(builder, pattern.directions, atoms=displaced)
        assert np.allclose(pattern.array, direct.array, rtol=1e-5)
        assert not np.allclose(pattern.array, si_reference.array, rtol=1e-3)

    def test_several_energies(self, si_reference):
        pattern = tiny_reference(backscatter_energy=[30e3, 26e3]).build(
            pbar=False, lazy=False
        )
        assert pattern.array.shape[0] == 2
        assert isinstance(pattern.ensemble_axes_metadata[0], EnergyAxis)
        assert "energy" not in pattern.metadata
        assert pattern.metadata["beam_energy"] == 30e3
        assert np.allclose(pattern.array[0], si_reference.array, rtol=1e-5)
        assert not np.allclose(pattern.array[1], si_reference.array, rtol=1e-3)

    def test_sampling_warnings(self):
        with pytest.warns(AntialiasLossWarning, match="scattering power"):
            EBSDReferencePattern(
                SILICON, 30e3, slab_cell=(20.0, 20.0, 100.0), sampling=1.0
            )
        with pytest.warns(AntialiasLossWarning, match="scattering power"):
            detector_builder(n_patches=None, sampling=0.3)


class TestHemispheres:
    """Which hemisphere a reference covers, and what stands in for the other."""

    @staticmethod
    def reference(atoms, hemisphere, **kwargs):
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

    def test_the_southern_directions_and_patches(self):
        north = self.reference(SILICON, "north")
        south = self.reference(SILICON, "south")
        both = self.reference(SILICON, "both")
        # the equator of a southern grid stays southern: -0.0
        assert np.all(np.signbit(south.directions[:, 2]))
        assert np.all(np.signbit(south.zone_axes[:, 2]))
        # (x, y, -z) below each northern direction, as EMsoft lays them out,
        # and the patches opposite the northern ones
        assert np.allclose(south.directions, north.directions * [1, 1, -1])
        assert np.allclose(south.zone_axes, -north.zone_axes)
        assert len(both.directions) == 2 * len(north.directions)
        assert len(both.zone_axes) == 2 * len(north.zone_axes)

    def test_opposite_directions_differ_only_by_inversion(self):
        # The south is the north of the crystal inverted about the slabs'
        # centre. Silicon cut about a bond centre, one of its inversion
        # centres, is its own inversion image: a direction and its opposite
        # are the same calculation. GaN has none, and its hemispheres differ.
        bond_centre = np.full(3, 5.431 / 8)
        silicon = self.reference(
            SILICON, "both", origin=bond_centre, centrosymmetric=True
        ).build(pbar=False, lazy=False)
        directions = silicon.northern.directions
        assert np.allclose(
            silicon.interpolate_directions(directions),
            silicon.interpolate_directions(-directions),
            rtol=1e-4,
        )

        gan = self.reference(GAN, "both", centrosymmetric=False).build(
            pbar=False, lazy=False
        )
        assert gan.hemisphere == "both"
        assert gan.metadata["hemisphere"] == "both"
        assert gan.metadata["centrosymmetric"] is False
        north = gan.interpolate_directions(directions)
        south = gan.interpolate_directions(-directions)
        assert (
            np.sqrt(np.mean((north / north.mean() - south / south.mean()) ** 2)) > 1e-3
        )

    def test_the_symmetry_is_searched_for_once_unless_given(self, monkeypatch):
        calls = []

        def search(atoms):
            calls.append(atoms)
            return False

        monkeypatch.setattr(reference_module, "is_centrosymmetric", search)
        reference = self.reference(SILICON, "north")
        assert not calls  # only when first needed
        assert reference.centrosymmetric is False
        assert reference.centrosymmetric is False
        assert len(calls) == 1

        # given, it is never searched for
        for given in (True, False):
            reference = self.reference(GAN, "north", centrosymmetric=given)
            pattern = reference.build(pbar=False, lazy=False)
            assert reference.centrosymmetric is given
            assert pattern.metadata["centrosymmetric"] is given
        assert len(calls) == 1

    def test_without_spglib_the_symmetry_is_unknown(self, monkeypatch):
        # without spglib, or when it cannot find the symmetry, the pattern is
        # still built, and records the symmetry as unknown
        for error, match in (
            (ImportError("no spglib"), "spglib is not installed"),
            (ValueError("no symmetry"), "spglib could not determine"),
        ):

            def search(atoms, error=error):
                raise error

            monkeypatch.setattr(reference_module, "is_centrosymmetric", search)
            with pytest.warns(UserWarning, match=match):
                pattern = self.reference(SILICON, "north").build(pbar=False, lazy=False)
            assert pattern.metadata["centrosymmetric"] is None


# --------------------------------------------------------------------------
# detector_pattern.py
# --------------------------------------------------------------------------


class TestEBSDDetectorPattern:
    """One orientation's pattern, calculated at the detector's own pixels."""

    def test_geometry(self):
        builder = detector_builder()
        assert np.allclose(
            builder.directions, builder.geometry.rotated_directions(builder.euler)
        )
        # the beam comes down the microscope axis, into the surface at the
        # sample tilt, turned into the crystal frame by the same angles
        tilt = np.radians(builder.geometry.sample_tilt)
        beam = bunge_rotation(builder.euler) @ [np.sin(tilt), 0.0, -np.cos(tilt)]
        assert np.allclose(builder.beam_direction, beam)
        # every pixel is calculated near its own axis ...
        assert len(builder.zone_axes) > 1
        assert builder.max_angle < patch_half_angle(2000)
        # ... or all of them in one slab, along their mean direction
        single = detector_builder(n_patches=None, sampling=0.15)
        mean = single.directions.reshape(-1, 3).mean(axis=0)
        assert np.allclose(single.zone_axes, [mean / np.linalg.norm(mean)])

        # Rather than assuming the crystal centrosymmetric, a detector looking
        # into its southern hemisphere is calculated there.
        for euler, hemisphere in [((0.0, 0.0, 0.0), 1.0), ((0.0, 180.0, 0.0), -1.0)]:
            builder = EBSDDetectorPattern(
                SILICON,
                EBSDGeometry(shape=(4, 4), detector_distance=15000.0, pixel_size=500.0),
                euler,
                slab_cell=(6.0, 6.0, 4.0),
                energy=30e3,
                sampling=0.25,
            )
            assert np.all(np.sign(builder.directions[..., 2]) == hemisphere), euler
            assert np.all(np.sign(builder.zone_axes[:, 2]) == hemisphere), euler

    def test_each_pixel_is_its_own_calculation(self, uniform):
        assert isinstance(uniform, EBSDPatternImages)
        assert uniform.array.shape == (6, 6)
        assert (uniform.metadata["source"], uniform.metadata["n_patches"]) == (
            "uniform",
            2000,
        )

        # calculated patch by patch and put back in order; alone, in its own
        # patch's slab, each pixel must agree
        builder = detector_builder()
        image = np.asarray(uniform.array).ravel()
        directions = builder.directions.reshape(-1, 3)
        axes = np.concatenate([fibonacci_hemisphere(2000), -fibonacci_hemisphere(2000)])
        block = bulk_block(SILICON, builder.slab_cell)
        for i in [0, 14, 35]:
            axis = axes[np.argmax(axes @ directions[i])]
            slab, rotation = rotated_slab(block, -axis, builder.slab_cell, (1, 1, 1))
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
            assert np.asarray(alone.array)[0] == pytest.approx(image[i], rel=1e-5), i

    def test_sources_and_energies(self):
        builder = detector_builder(n_patches=None)
        # a source without an energy is sent in at the beam's
        lit = builder.scan(abtem.PlaneWave(), lazy=False)
        assert lit.metadata["source"] == "PlaneWave"
        assert np.array_equal(
            lit.array, builder.scan(abtem.PlaneWave(energy=30e3), lazy=False).array
        )
        for source, match in [
            (abtem.PlaneWave(energy=20e3), "differs from the beam energy"),
            (abtem.Probe(semiangle_cutoff=10, energy=[20e3, 30e3]), "one energy"),
            ("beam", "source must be"),
        ]:
            with pytest.raises(ValueError, match=match):
                builder.scan(source)
        with pytest.raises(ValueError, match="give the energy"):
            detector_builder(energy=None)
        for euler in [(0.0, 0.0), (1.0, 2.0, 3.0, 4.0)]:
            with pytest.raises(ValueError, match="one orientation"):
                EBSDDetectorPattern(
                    SILICON,
                    EBSDGeometry(),
                    euler,
                    slab_cell=(6.0, 6.0, 4.0),
                    energy=30e3,
                )

    def test_a_plane_wave_lights_the_atoms_unevenly(self, uniform):
        # it channels and lights some atoms more than others, normalized
        # within each slice
        builder, source = detector_builder(), abtem.PlaneWave(energy=30e3)
        block, origin = builder._block_and_origin(lit=True)
        illumination, axes, shape = builder._illumination(source, block, origin, None)
        assert illumination.shape == (1, len(block))
        assert shape == () and axes == []
        # the block holds more than the incident slab; the atoms outside it
        # no patch uses, and they are left at one
        lit = illumination[0][illumination[0] != 1.0]
        assert len(lit) > 100
        assert lit.mean() == pytest.approx(1.0, rel=1e-6)
        assert lit.std() > 0.05

        pattern = builder.scan(source, lazy=False)
        assert pattern.metadata["source"] == "PlaneWave"
        assert not np.allclose(
            np.asarray(pattern.array), np.asarray(uniform.array), rtol=1e-3
        )
        lazy = builder.scan(source, lazy=True).compute()
        assert np.allclose(np.asarray(lazy.array), np.asarray(pattern.array), rtol=1e-6)

    def test_scans_a_probe(self):
        # in one slab or several patches
        probe = abtem.Probe(semiangle_cutoff=10, energy=30e3)
        positions = [[0.0, 0.0], [1.0, 1.5]]
        for n_patches in (None, 2000):
            builder = detector_builder(n_patches=n_patches)
            together = np.asarray(
                builder.scan(probe, abtem.CustomScan(positions), lazy=False).array
            )
            assert together.shape == (2, 6, 6)
            alone = builder.scan(probe, abtem.CustomScan(positions[1:]), lazy=False)
            assert np.allclose(together[1], np.asarray(alone.array)[0], rtol=1e-4)
            assert not np.allclose(together[0], together[1], rtol=1e-3)

        # a grid scan keeps its shape, and positions are measured from the origin
        builder = detector_builder(n_patches=None)
        probe = abtem.Probe(semiangle_cutoff=30, energy=30e3)
        scan = abtem.GridScan(start=(0, 0), end=(1.5, 1.0), gpts=(2, 3))
        pattern = builder.scan(probe, scan, lazy=False)
        assert pattern.array.shape == (2, 3, 6, 6)
        position = np.asarray(scan.get_positions())[1, 1]
        alone = builder.scan(probe, abtem.CustomScan([position]), lazy=False)
        assert np.allclose(
            np.asarray(pattern.array)[1, 1], np.asarray(alone.array)[0], rtol=1e-4
        )
        assert np.allclose(alone.ensemble_axes_metadata[0].values, [position])

        # a probe's own ensemble axes come before the scan's
        series = abtem.Probe(semiangle_cutoff=30, defocus=[0.0, 50.0])
        patterns = builder.scan(series, abtem.CustomScan(positions), lazy=True)
        assert patterns.array.shape == (2, 2, 6, 6)
        assert [type(axis).__name__ for axis in patterns.ensemble_axes_metadata] == [
            "ParameterAxis",
            "PositionsAxis",
        ]
        member = abtem.Probe(semiangle_cutoff=30, defocus=50.0)
        alone = builder.scan(member, abtem.CustomScan(positions[:1]), lazy=False)
        assert np.allclose(
            np.asarray(patterns.compute().array)[1, 0],
            np.asarray(alone.array)[0],
            rtol=1e-5,
        )

        with pytest.raises(ValueError, match="only a probe source depends"):
            builder.scan(abtem.PlaneWave(), abtem.CustomScan(positions), lazy=False)

    def test_frozen_phonons_average_the_configurations(self, uniform):
        frozen = abtem.FrozenPhonons(SILICON, num_configs=2, sigmas=0.1, seed=3)
        builder = detector_builder(atoms=frozen)
        thermal = builder.build(lazy=False)
        assert thermal.metadata["num_configs"] == 2
        lazy = builder.build(lazy=True).compute()
        assert np.allclose(np.asarray(lazy.array), np.asarray(thermal.array), rtol=1e-6)
        assert not np.allclose(
            np.asarray(thermal.array), np.asarray(uniform.array), rtol=1e-3
        )

        # and so is the incident beam's lighting
        def illumination(atoms):
            builder = detector_builder(atoms=atoms)
            block, origin = builder._block_and_origin(lit=True)
            source = abtem.PlaneWave(energy=30e3)
            return builder._illumination(source, block, origin, None)[0][0]

        thermal, static = illumination(frozen), illumination(SILICON)
        lit = thermal != 1.0
        assert thermal[lit].mean() == pytest.approx(1.0, rel=1e-6)
        assert not np.allclose(thermal[lit], static[lit], rtol=1e-2)

    def test_several_energies(self):
        builder = detector_builder(backscatter_energy=[30e3, 26e3])
        pattern = builder.build(lazy=False)
        assert pattern.array.shape == (2, 6, 6)
        assert isinstance(pattern.ensemble_axes_metadata[0], EnergyAxis)
        assert "energy" not in pattern.metadata
        assert pattern.metadata["beam_energy"] == 30e3
        lazy = builder.build(lazy=True).compute()
        assert np.allclose(np.asarray(lazy.array), np.asarray(pattern.array), rtol=1e-6)


# --------------------------------------------------------------------------
# emsoft.py
# --------------------------------------------------------------------------


class TestEMsoftWriter:
    @pytest.fixture
    def pattern(self):
        return sampled_in(
            "lambert", 81, lambda d: 1.0 + 0.5 * d[:, 2] ** 2, metadata={"energy": 30e3}
        )

    def write(self, tmp_path, pattern, atoms=SILICON, **kwargs):
        h5py = pytest.importorskip("h5py")
        pytest.importorskip("spglib")
        path = str(tmp_path / "master.h5")
        pattern.to_emsoft(path, atoms, **kwargs)
        return h5py.File(path)

    def test_follows_the_format(self, tmp_path, pattern):
        with self.write(tmp_path, pattern, npx=20) as f:
            master = f["EMData/EBSDmaster"]
            assert master["mLPNH"].shape == master["mLPSH"].shape == (1, 1, 41, 41)
            assert master["masterSPNH"].shape == (1, 41, 41)
            assert f["CrystalData/AtomData"].shape[0] == 5
            assert master["EkeVs"][()] == pytest.approx([30.0])

            # EMsoft writes Fortran rank-1 arrays and readers index them as
            # `dataset[:][0]`, which raises on a true scalar dataspace
            for name in [
                "EMData/EBSDmaster/numset",
                "EMData/EBSDmaster/numEbins",
                "EMheader/EBSDmaster/ProgramName",
                "CrystalData/SpaceGroupNumber",
                "NMLparameters/EBSDMasterNameList/npx",
            ]:
                assert f[name].shape == (1,), name
                f[name][:][0]

            # The conventional cell, in nanometres: ase.build.bulk gives the
            # primitive one, whose axes would contradict the cubic space group.
            lattice = f["CrystalData/LatticeParameters"][()]
            assert lattice.shape == (6,)
            assert f["CrystalData/SpaceGroupNumber"][:][0] == 227
            assert np.allclose(lattice[:3], 0.5431, atol=1e-4)
            assert np.allclose(lattice[3:], 90.0)
            assert f["CrystalData/CrystalSystem"][:][0] == 1

            # a pattern of one hemisphere fills the other with its inversion image
            north, south = master["mLPNH"][0, 0], master["mLPSH"][0, 0]
            assert np.array_equal(south, north[::-1, ::-1])

    def test_writes_the_calculated_grid(self, tmp_path, pattern):
        # By default, a pattern on an odd Lambert grid is written as calculated,
        # also when it holds both hemispheres; the reference pattern's default
        # grid is odd.
        both = hemispheres("both", centrosymmetric=False)
        both.metadata["energy"] = 30e3
        for written, gpts in ((pattern, 81), (both, 21)):
            with self.write(tmp_path, written, overwrite=True) as f:
                north = np.asarray(f["EMData/EBSDmaster/mLPNH"]).squeeze()
            assert north.shape == (gpts, gpts)
            calculated = written.northern
            expected = np.zeros((gpts, gpts))
            xy = SquareLambertProjection().project(calculated.directions)
            nodes = np.rint((xy + 1.0) / 2.0 * (gpts - 1)).astype(int)
            expected[nodes[:, 0], nodes[:, 1]] = np.asarray(calculated.array)
            assert np.allclose(north, expected.T, rtol=1e-6)
        assert tiny_reference(direction_gpts=None, gpts=128)._direction_gpts == 149

    def test_writes_the_pattern_in_the_frame_of_the_written_cell(self, tmp_path):
        # EMsoft builds its frame from the lattice parameters, with a along x and
        # b in the xy-plane. A crystal given in another orientation, with its
        # pattern turned with it, gives the same file.
        def written(atoms, values):
            pattern = SphericalPattern(
                values,
                self.directions,
                metadata={"energy": 30e3, "centrosymmetric": True},
            )
            with self.write(tmp_path, pattern, atoms, overwrite=True) as f:
                master = f["EMData/EBSDmaster"]
                return np.asarray(master["mLPNH"]), np.asarray(master["mLPSH"])

        def even(d):
            return 1.0 + 0.3 * d[:, 0] ** 2 + 0.2 * d[:, 1] * d[:, 2]

        cubic = ase.build.bulk("Si", "diamond", a=5.431, cubic=True)
        expected = written(cubic, even(self.directions))
        # about z, and about x, which mixes the hemispheres
        for axis, angle in (("z", 30.0), ("x", 20.0)):
            rotated = cubic.copy()
            rotated.rotate(angle, axis, rotate_cell=True)
            turn = rotated.cell[:].T @ np.linalg.inv(cubic.cell[:].T)
            for result, reference in zip(
                written(rotated, even(self.directions @ turn)), expected
            ):
                assert np.allclose(result, reference, atol=3e-3), (axis, angle)

    directions = SquareLambertProjection().grid(81)

    def test_writes_a_calculated_southern_hemisphere(self, tmp_path):
        pattern = hemispheres("both", centrosymmetric=False)
        pattern.metadata["energy"] = 30e3
        with self.write(tmp_path, pattern, GAN, npx=10, space_group=186) as f:
            south = np.asarray(f["EMData/EBSDmaster/mLPSH"]).squeeze()
            north = np.asarray(f["EMData/EBSDmaster/mLPNH"]).squeeze()
        expected = np.asarray(
            pattern.interpolate(21, "lambert", hemisphere="south").array
        ).T
        assert np.allclose(south, expected, atol=1e-5)
        assert not np.allclose(south, north[::-1, ::-1], atol=1e-2)

    def test_the_beam_energy_is_the_beams(self, tmp_path, pattern):
        # not that of the backscattered electrons the pattern counts
        pattern.metadata.update({"energy": 26e3, "beam_energy": 30e3})
        with self.write(tmp_path, pattern, npx=10) as f:
            assert f["EMData/EBSDmaster/EkeVs"][()] == pytest.approx([30.0])

    def test_refuses_what_it_cannot_write(self, tmp_path, pattern):
        directions = pattern.directions
        for bad, kwargs, match in [
            (
                SphericalPattern(
                    np.ones((2, len(directions))),
                    directions,
                    ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
                    metadata={"energy": 30e3},
                ),
                {},
                "only a single pattern",
            ),
            (
                SphericalPattern(np.ones(len(directions)), directions),
                {},
                "no beam energy",
            ),
            (pattern, {"space_group": 300}, "between 1 and 230"),
        ]:
            with pytest.raises(ValueError, match=match):
                self.write(tmp_path, bad, npx=10, **kwargs)


# --------------------------------------------------------------------------
# sampling.py
# --------------------------------------------------------------------------


class TestSampling:
    def test_potential_sampling(self):
        # A heavier nucleus has a more compact potential, so its transform
        # reaches further and the grid has to be finer.
        samplings = [
            potential_sampling(ase.Atoms(s, positions=[(0, 0, 0)], cell=(4, 4, 4)))
            for s in ("C", "Si", "Cu", "Au")
        ]
        assert samplings == sorted(samplings, reverse=True)
        assert potential_sampling(SILICON, tolerance=0.001) < potential_sampling(
            SILICON, tolerance=0.01
        )
        # Measured against a far finer grid, silicon at 30 kV runs about 0.6%
        # off at 0.07 Å and 5% off at 0.10 Å, so a 1% target lands between.
        assert 0.05 < potential_sampling(SILICON, tolerance=0.01) < 0.10

        for atoms, tolerance, match in [
            (SILICON, 0.0, "tolerance must be between"),
            (SILICON, 1.0, "tolerance must be between"),
            (ase.Atoms(cell=(4, 4, 4)), 0.01, "empty cell"),
        ]:
            with pytest.raises(ValueError, match=match):
                potential_sampling(atoms, tolerance=tolerance)

    def test_the_estimate_is_converged_and_cached(self):
        # The measuring grid has to reach well past the cutoff it is looking
        # for, or the tail it misses shifts the answer: 0.0659 Å is what a
        # 512-point measuring grid gave.
        from abtem.ebsd.sampling import _scattering_power

        _scattering_power.cache_clear()
        assert potential_sampling(SILICON) < 0.0659
        hits = _scattering_power.cache_info().hits
        potential_sampling(SILICON)
        assert _scattering_power.cache_info().hits > hits

    def test_the_power_lost(self):
        lost = [scattering_power_lost(SILICON, s) for s in (0.2, 0.1, 0.05)]
        assert lost == sorted(lost, reverse=True)
        # and the recommended sampling loses what it was asked to
        assert scattering_power_lost(
            SILICON, potential_sampling(SILICON, tolerance=0.01)
        ) == pytest.approx(0.01, abs=0.003)
