import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.build import graphene
from utils import float64_devices

import abtem
from abtem.core.backend import asnumpy
from abtem.inelastic.phonons import FrozenPhonons
from abtem.potentials.charge_density import (
    ChargeDensityPotential,
    _interpolate_between_cells,
    _interpolate_slice,
)
from abtem.slicing import slice_limits

# These densities are not a valence density of their atoms (random values, no
# electrons, or all of them), so ChargeDensityPotential warns that it cannot
# subtract the core and builds the potential of the density as given, which is
# what these tests check.
ignore_core_count_warning = pytest.mark.filterwarnings(
    "ignore:charge_density integrates to:UserWarning"
)


@pytest.fixture
def carbon_atoms():
    return Atoms("C", positions=[(2.5, 2.5, 2.5)], cell=(5, 5, 5), pbc=True)


@pytest.fixture
def charge_density_3d(carbon_atoms):
    """Random, but normalised to the valence-electron count a conventional frozen
    core implies for a single carbon atom (Z=6, Nc=2 -> 4 valence electrons), so the
    core-subtraction consistency check is satisfied."""
    rho = np.random.RandomState(0).rand(32, 32, 32).astype(np.float64) * 0.1
    voxel_volume = carbon_atoms.cell.volume / rho.size
    return (rho * (4.0 / (rho.sum() * voxel_volume))).astype(np.float32)


def test_build_paths_and_a_4d_density_give_one_potential(
    carbon_atoms, charge_density_3d
):
    """The eager build, the lazy build and the generated slices give one potential
    of one slice per slice thickness, and a 4D density holding one configuration
    gives that of the 3D density."""
    potential = ChargeDensityPotential(carbon_atoms, charge_density_3d, sampling=0.1)
    eager = potential.build(lazy=False).array
    assert eager.shape == (len(potential), *potential.gpts)
    scale = np.abs(eager).max()

    np.testing.assert_array_equal(potential.build().compute().array, eager)
    # the slices one at a time differ from the stack by ~3e-8 of the maximum
    generated = np.concatenate(
        [np.asarray(s.array) for s in potential.generate_slices()]
    )
    np.testing.assert_allclose(generated, eager, rtol=0, atol=1e-6 * scale)

    four_dimensional = ChargeDensityPotential(
        carbon_atoms, charge_density_3d[None], sampling=0.1
    )
    np.testing.assert_array_equal(four_dimensional.build(lazy=False).array, eager)


def test_slices_of_any_thickness_add_up_to_the_projected_potential(
    carbon_atoms, charge_density_3d
):
    """The slices of any thickness, down to the z-sampling of the density, add up
    to the potential of the cell projected as one slice."""
    cell_z = carbon_atoms.cell[2, 2]
    projected = ChargeDensityPotential(
        carbon_atoms, charge_density_3d, sampling=0.1, slice_thickness=cell_z
    ).build(lazy=False).array
    assert projected.shape[0] == 1
    scale = np.abs(projected).max()

    dz = cell_z / charge_density_3d.shape[2]
    for slice_thickness in (0.5, 1.0, 2.0, dz):
        potential = ChargeDensityPotential(
            carbon_atoms,
            charge_density_3d,
            sampling=0.1,
            slice_thickness=slice_thickness,
        )
        array = potential.build(lazy=False).array
        message = f"slice_thickness={slice_thickness}"
        assert array.shape == (len(potential), *potential.gpts), message
        assert sum(potential.slice_thickness) == pytest.approx(cell_z), message
        # float32 sums are <= 4e-5 of the maximum apart; the smallest slice
        # holds 3e-3 of it, so a dropped or doubled slice is 30x the tolerance.
        np.testing.assert_allclose(
            array.sum(axis=0),
            projected[0],
            rtol=0,
            atol=1e-4 * scale,
            err_msg=message,
        )


# A periodic field on a hexagonal cell, sampled on a grid whose size differs along every
# axis. Its exact values are known everywhere, so the interpolation can be checked
# against the field itself and against the invariances of a periodic interpolant.
_HEX_A, _HEX_C = 2.5, 4.0
_HEX_CELL = np.array(
    [[_HEX_A, 0, 0], [-_HEX_A / 2, _HEX_A * np.sqrt(3) / 2, 0], [0, 0, _HEX_C]]
)
_HEX_BOX = np.diag([_HEX_A, _HEX_A * np.sqrt(3), _HEX_C])
_HEX_GPTS = (24, 20, 40)


def _periodic_field(fractional):
    reciprocal = 2 * np.pi * np.linalg.inv(_HEX_CELL).T
    terms = zip(
        [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
        [1.0, 0.7, 0.5, 0.4],
        [0.3, 1.1, 2.0, 0.7],
    )
    positions = fractional @ _HEX_CELL
    return sum(
        amplitude * np.cos(positions @ (np.array(hkl) @ reciprocal) + phase)
        for hkl, amplitude, phase in terms
    )


def _fractional_grid(shape):
    axes = [np.arange(n) / n for n in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


@pytest.fixture(scope="module")
def hexagonal_field():
    return _periodic_field(_fractional_grid(_HEX_GPTS)).reshape(_HEX_GPTS)


def test_interpolation_between_cells_reproduces_a_periodic_field(hexagonal_field):
    """Against the exact field, including the target points next to the cell faces."""
    shape = (25, 43, 40)
    interpolated = _interpolate_between_cells(
        hexagonal_field, shape, _HEX_CELL, _HEX_BOX
    )
    fractional = _fractional_grid(shape) @ _HEX_BOX @ np.linalg.inv(_HEX_CELL)
    exact = _periodic_field(fractional % 1.0).reshape(shape)

    scale = np.abs(hexagonal_field).max()
    assert np.abs(interpolated - exact).max() < 3e-4 * scale


def test_interpolation_between_cells_is_invariant_to_whole_point_rolls(
    hexagonal_field,
):
    shape = (25, 43, 40)
    rolls = (5, 7, 9)
    shift = np.array(rolls) / np.array(_HEX_GPTS) @ _HEX_CELL

    rolled = _interpolate_between_cells(
        np.roll(hexagonal_field, rolls, axis=(0, 1, 2)),
        shape,
        _HEX_CELL,
        _HEX_BOX,
        offset=shift,
    )
    original = _interpolate_between_cells(hexagonal_field, shape, _HEX_CELL, _HEX_BOX)

    scale = np.abs(hexagonal_field).max()
    np.testing.assert_allclose(rolled, original, rtol=0, atol=1e-12 * scale)


def test_interpolation_between_cells_is_invariant_to_tiling(hexagonal_field):
    shape = (25, 43, 40)
    tiled_cell = _HEX_CELL * np.array([[2], [1], [1]])

    tiled = _interpolate_between_cells(
        np.tile(hexagonal_field, (2, 1, 1)), shape, tiled_cell, _HEX_BOX
    )
    original = _interpolate_between_cells(hexagonal_field, shape, _HEX_CELL, _HEX_BOX)

    scale = np.abs(hexagonal_field).max()
    np.testing.assert_allclose(tiled, original, rtol=0, atol=1e-12 * scale)


@ignore_core_count_warning
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize(
    "charge_densities", [1, 3], ids=["shared", "per_configuration"]
)
def test_build_from_frozen_phonons(
    carbon_atoms, charge_density_3d, lazy, charge_densities
):
    """Each configuration equals the potential of its displaced atoms built alone,
    with one charge density for every configuration or one for each."""
    frozen_phonons = FrozenPhonons(
        carbon_atoms, num_configs=3, sigmas=0.1, seed=4, ensemble_mean=False
    )
    densities = [charge_density_3d * (1 + 0.1 * i) for i in range(charge_densities)]
    charge_density = densities[0] if charge_densities == 1 else np.stack(densities)

    potential = ChargeDensityPotential(frozen_phonons, charge_density, sampling=0.2)
    built = potential.build(lazy=lazy)
    if lazy:
        built = built.compute()

    assert built.ensemble_shape == (3,)
    for i, atoms in enumerate(frozen_phonons):
        density = densities[i if charge_densities > 1 else 0]
        expected = ChargeDensityPotential(atoms, density, sampling=0.2)
        expected = expected.build(lazy=False)
        np.testing.assert_allclose(
            built.array[i], expected.array, rtol=0, atol=1e-6 * expected.array.max()
        )


def test_chunks_of_frozen_phonons_hold_every_configuration(
    carbon_atoms, charge_density_3d
):
    """The chunks of an unbuilt potential with frozen phonons hold every
    configuration, each equal to the potential of its displaced atoms built alone."""
    frozen_phonons = FrozenPhonons(
        carbon_atoms, num_configs=3, sigmas=0.1, seed=4, ensemble_mean=False
    )
    potential = ChargeDensityPotential(
        frozen_phonons, charge_density_3d, sampling=0.2, slice_thickness=1.0
    )

    chunks = list(potential.generate_chunked_slices(chunk_size=2))

    assert len(potential) == 5
    assert [chunk.shape[:2] for chunk in chunks] == [(3, 1), (3, 2), (3, 2)]
    for i, atoms in enumerate(frozen_phonons):
        expected = ChargeDensityPotential(
            atoms, charge_density_3d, sampling=0.2, slice_thickness=1.0
        ).build(lazy=False)
        np.testing.assert_allclose(
            np.concatenate([chunk.array[i] for chunk in chunks]),
            expected.array,
            rtol=0,
            atol=1e-6 * expected.array.max(),
        )


@ignore_core_count_warning
@pytest.mark.parametrize("plane", ["xz", "yz"])
def test_anisotropic_sigmas_follow_the_axes_of_the_input_atoms(
    charge_density_3d, plane
):
    """The point charges of frozen phonons in a potential rotated to another plane
    are displaced along the axes of the input atoms."""
    atoms = Atoms("C", positions=[(2.0, 2.5, 3.0)], cell=(5, 6, 7), pbc=True)
    sigmas = (0.05, 0.10, 0.20)
    frozen_phonons = FrozenPhonons(atoms, num_configs=1, sigmas=sigmas, seed=4)
    r = np.random.default_rng(frozen_phonons.seed[0]).normal(size=(1, 3))
    displaced = atoms.copy()
    displaced.positions += np.array(sigmas, dtype=np.float32) * r

    actual = ChargeDensityPotential(
        frozen_phonons, charge_density_3d, sampling=0.2, plane=plane
    ).build(lazy=False)
    expected = ChargeDensityPotential(
        displaced, charge_density_3d, sampling=0.2, plane=plane
    ).build(lazy=False)

    np.testing.assert_allclose(
        actual.array[0], expected.array, rtol=0, atol=1e-5 * expected.array.max()
    )


@ignore_core_count_warning
@pytest.mark.parametrize("num_densities", [2, 4])
def test_a_charge_density_count_other_than_one_or_the_configurations_raises(
    carbon_atoms, charge_density_3d, num_densities
):
    """Three configurations take one charge density or three: two leave one
    configuration without a density, and four leave one density unused."""
    frozen_phonons = FrozenPhonons(carbon_atoms, num_configs=3, sigmas=0.1, seed=4)
    densities = np.stack([charge_density_3d] * num_densities)
    with pytest.raises(ValueError, match="charge densities were given for 3"):
        ChargeDensityPotential(frozen_phonons, densities, sampling=0.2)


@ignore_core_count_warning
def test_several_charge_densities_without_frozen_phonons_raise(
    carbon_atoms, charge_density_3d
):
    densities = np.stack([charge_density_3d] * 2)
    with pytest.raises(ValueError, match="charge densities were given for 1"):
        ChargeDensityPotential(carbon_atoms, densities, sampling=0.2)


# `repetitions` repeats the lattice vectors of the cell, not their components.


def _bn():
    atoms = graphene(formula="BN", a=2.5, vacuum=2.0)
    atoms.pbc = True
    return atoms


def _two_atoms():
    return Atoms(
        "CO",
        positions=[(0.5, 0.6, 0.7), (2.0, 1.5, 3.0)],
        cell=[4.0, 3.0, 5.0],
        pbc=True,
    )


def _charge_density():
    shape = (16, 12, 20)
    x, y, z = np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij")
    return (
        0.3
        + 0.1 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
        + 0.05 * np.cos(4 * np.pi * z)
    )


# The orthogonal cell is the baseline accuracy of the comparison with the tiled
# one-cell potential (1.4e-5 of the maximum); the others are of the same order.
REPETITION_CASES = [
    ("orthogonal x (2, 3, 2)", _two_atoms, (2, 3, 2), (40, 30)),
    ("BN x (2, 1, 1)", _bn, (2, 1, 1), (25, 45)),
    ("BN x (2, 3, 2)", _bn, (2, 3, 2), (25, 45)),
    ("BN x (1, 2, 1)", _bn, (1, 2, 1), (25, 45)),
]


@ignore_core_count_warning
@float64_devices
@pytest.mark.parametrize("case", REPETITION_CASES, ids=lambda c: c[0])
def test_charge_density_potential_repetitions_tile_the_one_cell_potential(case):
    name, make_atoms, repetitions, unit_gpts = case
    atoms, rho = make_atoms(), _charge_density()

    unit = ChargeDensityPotential(atoms, rho, gpts=unit_gpts, slice_thickness=1.0)
    unit_array = asnumpy(unit.build(lazy=False).array)
    repeated = ChargeDensityPotential(
        atoms,
        rho,
        sampling=unit.sampling,
        slice_thickness=1.0,
        repetitions=repetitions,
    )

    # The box of the repeated crystal.
    assert repeated.box == pytest.approx(
        abtem.Potential(atoms * repetitions, sampling=0.1).box
    )

    nx, ny, nz = (round(float(repeated.box[i] / unit.box[i])) for i in range(3))
    tiled = np.tile(unit_array, (nz, nx, ny))
    actual = asnumpy(repeated.build(lazy=False).array)
    assert actual.shape == tiled.shape
    np.testing.assert_allclose(actual, tiled, rtol=0, atol=1e-4 * np.abs(tiled).max())


@ignore_core_count_warning
@float64_devices
def test_charge_density_potential_repetitions_lazy_equals_eager_on_bn():
    atoms, rho = _bn(), _charge_density()
    potential = ChargeDensityPotential(
        atoms, rho, gpts=(50, 135), slice_thickness=1.0, repetitions=(2, 3, 2)
    )
    eager = asnumpy(potential.build(lazy=False).array)
    lazy = asnumpy(potential.build(lazy=True).compute().array)
    np.testing.assert_allclose(lazy, eager, rtol=0, atol=1e-10 * np.abs(eager).max())


@float64_devices
def test_charge_density_potential_with_an_approximate_default_box_reports_it_once():
    # The default box of BN x (3, 1, 1) is reached by a strain of about 1 %. The
    # potential reports it when it is constructed; the Ewald potential it builds
    # from its own box does not repeat it.
    atoms, rho = _bn(), _charge_density()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        potential = ChargeDensityPotential(
            atoms,
            rho,
            sampling=0.2,
            slice_thickness=1.0,
            repetitions=(3, 1, 1),
            non_orthogonal=False,
        )
        eager = asnumpy(potential.build(lazy=False).array)
        lazy = asnumpy(potential.build(lazy=True).compute().array)
    assert len([r for r in records if "abTEM chose" in str(r.message)]) == 1
    np.testing.assert_allclose(lazy, eager, rtol=0, atol=1e-10 * np.abs(eager).max())


def test_interpolated_slices_of_a_repeated_cell_tile_those_of_the_one_cell():
    # Cumulative slice limits of 0.9 A give b - a = 0.9000000000000004 from the
    # sixth slice on, a hair above nine grid steps of 0.1 A: the repeated cell
    # must still sample every slice at nine points, as the one cell does.
    z = np.arange(90) / 90
    x, y = np.meshgrid(np.arange(8) / 8, np.arange(6) / 6, indexing="ij")
    array = (
        1
        + 0.3 * np.cos(2 * np.pi * x)[..., None] * np.sin(2 * np.pi * y)[..., None]
        + 0.5 * np.cos(2 * np.pi * z)
        + 0.2 * np.sin(6 * np.pi * z)
    )
    gpts, sampling = (8, 6), (0.5, 0.5)

    def slices(array, height, num_slices):
        cell = np.diag([4.0, 3.0, height])
        limits = slice_limits((0.9,) * num_slices)
        return np.array(
            [_interpolate_slice(array, cell, gpts, sampling, a, b) for a, b in limits]
        )

    one = slices(array, 9.0, 10)
    repeated = slices(np.tile(array, (1, 1, 3)), 27.0, 30)
    np.testing.assert_allclose(repeated, np.tile(one, (3, 1, 1)), rtol=0, atol=1e-9)


def test_generate_slices_poisson_solve_matches_requested_gpts(
    monkeypatch, carbon_atoms, charge_density_3d
):
    """The internal reciprocal-space Poisson solve (which places the point-charge
    correction, and for VASPPotential the core-density correction too) must be
    built at the requested output `gpts` -- not silently pinned to
    `charge_density`'s own native resolution via `fft_crop`. Before this was
    fixed, `fft_crop`'s target used `charge.shape[:2]` (the *input* array's own
    resolution) instead of `ewald_potential.gpts`, so refining `gpts` never
    actually increased the resolution at which atomic-scale features were
    resolved -- it only amplified `_interpolate_slice`'s downstream resampling
    artifacts (see the VASPPotential vacuum/core-sharpness investigation)."""
    import abtem.potentials.charge_density as cd_mod

    requested_gpts = (77, 77)  # deliberately different from charge_density_3d's
    assert charge_density_3d.shape[:2] != requested_gpts  # native resolution (32, 32)

    pot = ChargeDensityPotential(carbon_atoms, charge_density_3d, gpts=requested_gpts)

    seen_shapes = []
    original_fft_crop = cd_mod.fft_crop

    def spy_fft_crop(array, new_shape, **kwargs):
        seen_shapes.append(new_shape)
        return original_fft_crop(array, new_shape, **kwargs)

    monkeypatch.setattr(cd_mod, "fft_crop", spy_fft_crop)

    next(pot.generate_slices())

    assert seen_shapes, "fft_crop was not called"
    assert seen_shapes[0][:2] == requested_gpts


def test_reference_defaults_to_mean_and_rejects_others(carbon_atoms, charge_density_3d):
    pot = ChargeDensityPotential(carbon_atoms, charge_density_3d, sampling=0.1)
    assert pot.reference == "mean"
    with pytest.raises(ValueError, match="reference must be one of"):
        ChargeDensityPotential(
            carbon_atoms, charge_density_3d, sampling=0.1, reference="minimum"
        )


@float64_devices
def test_slice_min_reference_subtracts_each_slice_minimum(
    carbon_atoms, charge_density_3d, device
):
    """reference="slice_min" is the per-slice minimum subtraction of v1.0.10 and
    earlier: each slice of the "mean" potential minus its own minimum, which
    differs from slice to slice. Up to rounding: 1e-12 of the maximum."""
    slices = {
        reference: asnumpy(
            ChargeDensityPotential(
                carbon_atoms,
                charge_density_3d,
                sampling=0.1,
                slice_thickness=0.5,
                device=device,
                reference=reference,
            )
            .build(lazy=False)
            .array
        )
        for reference in ("mean", "slice_min")
    }
    minima = slices["mean"].min(axis=(1, 2), keepdims=True)
    assert np.ptp(minima) > 0.1  # V A: the constants do differ between slices
    np.testing.assert_allclose(
        slices["slice_min"],
        slices["mean"] - minima,
        rtol=0,
        atol=1e-12 * np.abs(slices["mean"]).max(),
    )


def _skewed_monolayer():
    """Graphene with 4 A of vacuum on either side (a skewed in-plane cell) and its
    -Z electrons, Gaussians of std 0.7 A on a 0.125 A grid."""
    atoms = graphene(vacuum=4.0)
    atoms.pbc = True
    shape = (20, 20, int(round(atoms.cell[2, 2] / 0.125)))
    fractions = np.stack(
        np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij"), axis=-1
    )
    density = np.zeros(shape)
    for position in atoms.get_scaled_positions():
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                d = fractions - position - (i, j, 0)
                d[..., 2] -= np.round(d[..., 2])
                r2 = ((d @ np.array(atoms.cell)) ** 2).sum(-1)
                density += 6 * np.exp(-r2 / (2 * 0.49)) / (2 * np.pi * 0.49) ** 1.5
    return atoms, density


@ignore_core_count_warning
@float64_devices
@pytest.mark.parametrize("reference", ["mean", "vacuum"])
def test_reference_constant_holds_across_build_routes(reference, device):
    """The constant is the same however the potential is built: a skewed cell
    repeated (2, 1, 1), lazy blocks of frozen phonons against the eager build, and
    a range of slices against those of the whole. "mean" averages to zero (within
    2e-4 of the removed constant sum(Z) w^2 / (2 eps0 V), 57 V here, as in
    test_potential_physics; measured 5e-5), "vacuum" has a planar average of
    zero in the slice at z = 0, 4 A from the atoms (1e-10 V). Sampling 0.05 A
    narrows the Ewald width to 1.5 A, which keeps the build cheap."""
    from abtem.core.constants import eps0

    atoms, density = _skewed_monolayer()
    potential = ChargeDensityPotential(
        atoms,
        density,
        sampling=0.05,
        slice_thickness=0.5,
        repetitions=(2, 1, 1),
        device=device,
        reference=reference,
    )
    width = potential._ewald_width()
    removed = np.sum(atoms.numbers) * width**2 / (2 * eps0 * atoms.cell.volume)
    array = asnumpy(potential.build(lazy=False).array) / 0.5  # V
    assert array.shape[-2:] == potential.gpts
    if reference == "mean":
        assert abs(array.mean()) < 2e-4 * removed, (array.mean(), removed)
    else:
        assert abs(array[0].mean()) < 1e-10, array[0].mean()

    part = (
        np.concatenate([asnumpy(s.array) for s in potential.generate_slices(6, 12)])
        / 0.5
    )
    np.testing.assert_allclose(part, array[6:12], rtol=0, atol=1e-12 * array.max())

    frozen_phonons = FrozenPhonons(atoms, num_configs=2, sigmas=0.05, seed=3)
    potential = ChargeDensityPotential(
        frozen_phonons,
        density,
        sampling=0.05,
        slice_thickness=0.5,
        device=device,
        reference=reference,
    )
    width = potential._ewald_width()
    removed = np.sum(atoms.numbers) * width**2 / (2 * eps0 * atoms.cell.volume)
    eager = asnumpy(potential.build(lazy=False).array) / 0.5
    lazy = asnumpy(potential.build(lazy=True).compute().array) / 0.5
    np.testing.assert_allclose(lazy, eager, rtol=0, atol=1e-12 * eager.max())
    if reference == "mean":
        assert np.abs(eager.mean(axis=(1, 2, 3))).max() < 2e-4 * removed
    else:
        assert np.abs(eager[:, 0].mean(axis=(1, 2))).max() < 1e-10


def test_charge_density_potential_on_skew_cell_preserves_cell():
    """ChargeDensityPotential.build on a non-orthogonal in-plane cell must produce a
    PotentialArray on the same skew grid (not silently rectified onto a Cartesian
    rectangle). Before the slicer was generalised, _interpolate_slice built the
    target slice on a diagonal slice_box, so the resulting PotentialArray came back
    with ``grid.is_orthogonal == True`` even on a skew input."""
    import numpy as np
    from ase import Atoms

    a, c = 2.46, 3.35
    atoms = Atoms(
        "C2",
        cell=[
            [a, 0, 0],
            [a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60)), 0],
            [0, 0, c],
        ],
        pbc=True,
        scaled_positions=[(0, 0, 0), (1 / 3, 1 / 3, 0.5)],
    )
    # normalised to this cell's own valence count: 2 C atoms, Z - Nc = 4 each
    density = np.random.RandomState(0).rand(16, 16, 16).astype(np.float64) * 0.1
    density *= 8.0 / (density.sum() * atoms.cell.volume / density.size)

    # sampling=0.2 (13 x 13): the build cost is the skew-grid point-charge
    # correction, which grows with the in-plane pixel count (3.4 s at sampling=0.1)
    pot = ChargeDensityPotential(atoms, density.astype(np.float32), sampling=0.2)
    assert not pot.grid.is_orthogonal
    assert pot.cell is not None

    built = pot.build(lazy=False)
    arr = np.asarray(built.array)
    assert np.all(np.isfinite(arr))
    # the cell metadata must propagate through the slicer to the built array
    assert built.cell is not None, (
        "ChargeDensityPotential.build silently lost the non-orthogonal cell metadata"
    )
    assert not built.grid.is_orthogonal


def test_point_charges_general_cell_conserves_nuclear_charge():
    """The pixel-volume normalisation in _add_point_charges_real_space /
    _add_point_charges_fourier must be the true parallelepiped volume |det(cell)|,
    not the product of diagonal entries. For an ASE-standard cell (a along x, b in
    the xy-plane) the cell matrix is upper-triangular and the two formulas happen
    to coincide. They diverge when the cell is rotated -- e.g. a not along x --
    which abTEM accepts through ChargeDensityPotential. Test with such a cell that
    the total integrated nuclear charge equals sum(atomic_numbers) (a
    method-independent sanity check)."""
    import numpy as np
    from ase import Atoms

    from abtem.potentials.charge_density import _add_point_charges_real_space

    # rotated cell where prod(diag) = 60 but |det| = 55 (a NOT along x)
    cell = [[3.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
    parallelepiped = abs(np.linalg.det(np.array(cell)))
    prod_diag = float(np.prod(np.diag(np.array(cell))))
    # sanity: the two normalisations differ here (so we're testing the real fix)
    assert abs(prod_diag - parallelepiped) / parallelepiped > 0.05

    atoms = Atoms(
        "C2", cell=cell, pbc=True,
        scaled_positions=[(0.1, 0.2, 0.3), (0.6, 0.7, 0.7)],
    )
    expected_total_charge = float(np.sum(atoms.numbers))

    shape = (32, 32, 32)
    array = np.zeros(shape, dtype=np.float32)
    array = _add_point_charges_real_space(array, atoms)
    pixel_volume = parallelepiped / np.prod(shape)
    total_real = float(array.sum()) * pixel_volume
    assert abs(total_real - expected_total_charge) / expected_total_charge < 1e-5, (
        f"nuclear-charge normalisation broken on rotated cell: got {total_real:.4f} "
        f"expected {expected_total_charge:.4f} -- pixel_volume should be |det(cell)|/N, "
        f"not prod(diag(cell))/N"
    )


def test_no_warning_for_valence_like_density(carbon_atoms, charge_density_3d):
    """A smooth, low-amplitude density (as any real valence-only density is) must
    not trigger the all-electron-density warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChargeDensityPotential(carbon_atoms, charge_density_3d, sampling=0.1)


def test_warns_for_implausibly_peaked_density(carbon_atoms):
    """A density with an implausibly large peak value (as a genuinely all-electron
    density, e.g. from summing VASP AECCAR0+AECCAR2, would have near each nucleus)
    must raise the all-electron-density warning."""
    charge_density = np.random.RandomState(0).rand(32, 32, 32).astype(np.float32) * 0.1
    charge_density[16, 16, 16] = 1e5

    # such a density also fails the core-subtraction electron-count check, which
    # warns separately -- capture everything and assert on the one under test
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ChargeDensityPotential(carbon_atoms, charge_density, sampling=0.1)
    assert any("all-electron density" in str(w.message) for w in caught)


def test_no_warning_for_sharply_peaked_valence_density(carbon_atoms):
    """A large peak alone must not trigger the warning. VASP's AECCAR2 is
    valence-only yet keeps the true orbitals' nodal structure near each nucleus, so
    for a heavy element on a fine grid its peak is genuinely large (SrTiO3 reaches
    ~900 e/A^3) while its integrated count stays far below the total atomic
    number."""
    charge_density = np.random.RandomState(0).rand(64, 64, 64).astype(np.float64) * 1e-3
    charge_density[32, 32, 32] = 5e3
    voxel_volume = carbon_atoms.cell.volume / charge_density.size
    charge_density *= 4.0 / (charge_density.sum() * voxel_volume)  # C: Z - Nc = 4
    assert charge_density.max() > 1000.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChargeDensityPotential(
            carbon_atoms, charge_density.astype(np.float32), sampling=0.1
        )


def test_no_warning_for_smooth_density_near_total_atomic_number(carbon_atoms):
    """A high integrated count alone must not trigger the warning either. Hydrogen
    has no core electrons to omit, so its legitimate valence-only density integrates
    to exactly the total atomic number -- the case the two-signal rule protects."""
    atoms = Atoms("H", positions=[(2.5, 2.5, 2.5)], cell=(5, 5, 5), pbc=True)
    charge_density = np.full((32, 32, 32), 1.0, dtype=np.float64)
    charge_density *= 1.0 / (charge_density.sum() * atoms.cell.volume / charge_density.size)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChargeDensityPotential(
            atoms, charge_density.astype(np.float32), sampling=0.1
        )


def test_no_warning_for_lazy_charge_density(carbon_atoms):
    """The all-electron-density check is skipped for a lazy (dask) charge_density,
    even if its content would otherwise trigger the warning, to avoid forcing an
    eager computation of a potentially large array just for this check."""
    da = pytest.importorskip("dask.array")
    charge_density = np.full((32, 32, 32), 1.0, dtype=np.float32)
    lazy_charge_density = da.from_array(charge_density, chunks=-1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChargeDensityPotential(carbon_atoms, lazy_charge_density, sampling=0.1)


def _total_charge(potential, atoms, density):
    """Net charge of the array handed to the Poisson solve, in electrons."""
    from abtem.potentials.charge_density import (
        _add_core_density_correction_fourier,
        add_point_charges_fourier,
    )

    charge = -np.fft.fftn(density)
    charge = add_point_charges_fourier(charge, atoms, 0.05)
    if potential._core_density_correction is not None:
        charge = _add_core_density_correction_fourier(
            charge, atoms, potential._core_density_correction
        )
    voxel_volume = atoms.cell.volume / np.prod(density.shape)
    return float(charge[0, 0, 0].real) * voxel_volume


def test_core_subtraction_makes_the_cell_neutral(carbon_atoms, charge_density_3d):
    """The point that motivates the whole correction: adding each atom's full Z to a
    valence-only density leaves a net +Nc per atom, which the Poisson solve can only
    absorb into a uniform background. A net charge contributes q / (eps0 k^2), so the
    error lands at low spatial frequency -- in the low-order structure factors.

    By default the conventional frozen core is subtracted: 2 electrons for carbon."""
    density = np.asarray(charge_density_3d, dtype=float)
    potential = ChargeDensityPotential(carbon_atoms, charge_density_3d, sampling=0.1)
    assert potential.core_electrons == {"C": 2}

    assert _total_charge(potential, carbon_atoms, density) == pytest.approx(0.0, abs=0.05)

    # constructing the potential tabulates the core form factor (~0.5 s); remove the
    # correction from the same one instead of building a second
    potential._core_density_correction = None
    assert _total_charge(potential, carbon_atoms, density) == pytest.approx(2.0, abs=0.05)


def test_explicit_valence_electrons_override_the_table(carbon_atoms):
    """Semicore pseudopotentials keep shells in the valence that the conventional
    table assigns to the core, so the caller must be able to say so."""
    rho = np.random.RandomState(0).rand(32, 32, 32).astype(np.float64) * 0.1
    rho *= 5.0 / (rho.sum() * carbon_atoms.cell.volume / rho.size)

    potential = ChargeDensityPotential(
        carbon_atoms, rho.astype(np.float32), sampling=0.1, valence_electrons={"C": 5}
    )
    assert potential.core_electrons == {"C": 1}


def test_inconsistent_electron_count_warns_and_skips_the_correction(carbon_atoms):
    """Subtracting the wrong core is worse than subtracting none -- on SrTiO3 the
    conventional table taken on trust scores below making no correction at all. So an
    inconsistency falls back to the old behaviour rather than guessing."""
    rho = np.random.RandomState(0).rand(32, 32, 32).astype(np.float64) * 0.1
    rho *= 5.0 / (rho.sum() * carbon_atoms.cell.volume / rho.size)  # not 4

    with pytest.warns(UserWarning, match="will not be subtracted"):
        potential = ChargeDensityPotential(
            carbon_atoms, rho.astype(np.float32), sampling=0.1
        )
    assert potential.core_electrons == {"C": 0}


def test_lazy_charge_density_skips_the_electron_count_check(carbon_atoms):
    """The check would force an eager computation of a potentially large array."""
    import dask.array as da

    rho = da.random.random((32, 32, 32), chunks=(32, 32, 32)) * 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        potential = ChargeDensityPotential(carbon_atoms, rho, sampling=0.1)
    assert potential.core_electrons == {"C": 2}
