"""The grid of a built potential array follows its data.

The array fixes the number of grid points, as it does for `Waves`: `gpts` cannot be
set, and setting `sampling` rescales the extent. Built waves on another grid are
refused without touching the potential.
"""

import numpy as np
import pytest
from ase import Atoms

import abtem
from abtem.potentials.iam import PotentialArray
from abtem.prism.s_matrix import SMatrix

GPTS = (40, 30)
EXTENT = (4.0, 3.0)
ENERGY = 100e3


@pytest.fixture
def potential_array():
    atoms = Atoms("C", positions=[(1.0, 1.0, 1.0)], cell=(*EXTENT, 2.0), pbc=True)
    return abtem.Potential(atoms, gpts=GPTS, slice_thickness=1.0).build(lazy=False)


@pytest.fixture
def transmission_function(potential_array):
    return potential_array.transmission_function(ENERGY)


def _assert_grid_follows_data(field):
    assert field.gpts == field.array.shape[-2:]
    np.testing.assert_allclose(
        field.extent, np.array(field.gpts) * np.array(field.sampling), rtol=1e-12
    )


@pytest.mark.parametrize("fixture", ["potential_array", "transmission_function"])
def test_gpts_of_a_built_array_cannot_be_set(fixture, request):
    field = request.getfixturevalue(fixture)

    with pytest.raises(RuntimeError, match="gpts cannot be modified"):
        field.gpts = (80, 60)

    assert field.gpts == GPTS
    _assert_grid_follows_data(field)


@pytest.mark.parametrize("fixture", ["potential_array", "transmission_function"])
def test_sampling_of_a_built_array_rescales_the_extent(fixture, request):
    field = request.getfixturevalue(fixture)

    field.sampling = 0.05

    assert field.gpts == GPTS
    assert field.array.shape[-2:] == GPTS
    np.testing.assert_allclose(field.extent, (2.0, 1.5), rtol=1e-12)
    _assert_grid_follows_data(field)


@pytest.mark.parametrize(
    "waves_kwargs",
    [
        dict(extent=EXTENT, gpts=(80, 60)),
        dict(extent=(8.0, 6.0), gpts=GPTS),
    ],
    ids=["other gpts", "other extent"],
)
def test_waves_on_another_grid_are_refused_without_changing_the_potential(
    potential_array, waves_kwargs
):
    waves = abtem.PlaneWave(energy=ENERGY, **waves_kwargs).build(lazy=False)

    with pytest.raises(RuntimeError, match="Inconsistent grid"):
        waves.multislice(potential_array)

    assert potential_array.gpts == GPTS
    np.testing.assert_allclose(potential_array.extent, EXTENT, rtol=1e-12)
    np.testing.assert_allclose(potential_array.project().sampling, (0.1, 0.1))


def test_waves_on_the_grid_of_the_potential_are_not_refused(potential_array):
    waves = abtem.PlaneWave(energy=ENERGY, extent=EXTENT, gpts=GPTS).build(lazy=False)

    exit_waves = waves.multislice(potential_array).compute()

    assert exit_waves.gpts == GPTS


def test_a_potential_array_built_from_an_array_locks_the_grid_to_the_data():
    array = PotentialArray(np.zeros((2, 40, 30)), slice_thickness=1.0, sampling=0.1)

    assert array.gpts == GPTS
    np.testing.assert_allclose(array.extent, EXTENT)
    with pytest.raises(RuntimeError, match="gpts cannot be modified"):
        array.gpts = (80, 60)


def _smatrix_of_built_potential(potential_array, how):
    kwargs = dict(energy=ENERGY, semiangle_cutoff=20, interpolation=2)
    if how == "init":
        return SMatrix(potential=potential_array, **kwargs)
    atoms = Atoms("C", positions=[(1.0, 1.0, 1.0)], cell=(*EXTENT, 2.0), pbc=True)
    smatrix = SMatrix(potential=abtem.Potential(atoms, gpts=(80, 60)), **kwargs)
    smatrix.potential = potential_array
    return smatrix


def _assert_potential_untouched(potential_array):
    assert potential_array.gpts == GPTS
    np.testing.assert_allclose(potential_array.extent, EXTENT, rtol=1e-12)
    np.testing.assert_allclose(potential_array.project().sampling, (0.1, 0.1))


@pytest.mark.parametrize("how", ["init", "potential setter"])
def test_smatrix_sampling_does_not_rescale_its_built_potential(potential_array, how):
    smatrix = _smatrix_of_built_potential(potential_array, how)

    smatrix.sampling = 0.05

    _assert_potential_untouched(potential_array)
    with pytest.raises(RuntimeError, match="Inconsistent grid extent"):
        smatrix.build(lazy=True)
    with pytest.raises(RuntimeError, match="Inconsistent grid extent"):
        smatrix.scan(
            scan=abtem.GridScan((0, 0), EXTENT, gpts=(3, 3)),
            detectors=abtem.AnnularDetector(50, 100),
        ).compute()


@pytest.mark.parametrize("upsample", [False, True])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("change", ["sampling", "extent"])
def test_smatrix_with_a_changed_grid_is_refused_by_every_route(
    potential_array, change, lazy, upsample
):
    smatrix = SMatrix(
        potential=potential_array,
        energy=ENERGY,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=upsample,
    )
    if change == "sampling":
        smatrix.sampling = 0.05
    else:
        smatrix.extent = (2 * EXTENT[0], 2 * EXTENT[1])
    scan = abtem.GridScan((0, 0), EXTENT, gpts=(3, 3))
    detector = abtem.AnnularDetector(50, 100)

    with pytest.raises(RuntimeError, match="Inconsistent grid extent"):
        smatrix.build(lazy=lazy)
    with pytest.raises(RuntimeError, match="Inconsistent grid extent"):
        smatrix.scan(scan=scan, detectors=detector, lazy=lazy)
    with pytest.raises(RuntimeError, match="Inconsistent grid extent"):
        smatrix.reduce(scan=scan, detectors=detector, lazy=lazy)

    _assert_potential_untouched(potential_array)


@pytest.mark.parametrize("how", ["init", "potential setter"])
def test_smatrix_gpts_cannot_be_changed_for_a_built_potential(potential_array, how):
    smatrix = _smatrix_of_built_potential(potential_array, how)

    with pytest.raises(RuntimeError, match="gpts cannot be modified"):
        smatrix.gpts = (80, 60)

    _assert_potential_untouched(potential_array)


@pytest.mark.filterwarnings("ignore:The interpolation factor does not exactly divide")
def test_rounding_the_gpts_of_the_smatrix_of_a_built_potential_leaves_it_untouched():
    atoms = Atoms("C", positions=[(1.0, 1.0, 1.0)], cell=(*EXTENT, 2.0), pbc=True)
    potential_array = abtem.Potential(atoms, gpts=(41, 31)).build(lazy=False)
    smatrix = SMatrix(
        potential=potential_array, energy=ENERGY, semiangle_cutoff=20, interpolation=2
    )

    with pytest.raises(RuntimeError, match="gpts cannot be modified"):
        smatrix.round_gpts_to_interpolation()

    assert potential_array.gpts == potential_array.array.shape[-2:] == (41, 31)


@pytest.mark.filterwarnings("ignore:The interpolation factor does not exactly divide")
def test_the_grid_of_the_smatrix_of_a_builder_follows_the_smatrix():
    atoms = Atoms("C", positions=[(1.0, 1.0, 1.0)], cell=(*EXTENT, 2.0), pbc=True)
    potential = abtem.Potential(atoms, gpts=(41, 31))
    smatrix = SMatrix(
        potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=2
    )

    rounded = smatrix.round_gpts_to_interpolation()

    assert rounded.gpts == potential.gpts == (42, 32)


@pytest.mark.parametrize(
    "other",
    [
        {"gpts": (32, 40), "extent": (10.0, 13.0)},
        {"gpts": (32, 48), "extent": (10.0, 13.0)},
    ],
)
def test_matching_built_waves_to_another_grid_is_refused_without_changing_them(other):
    waves = abtem.PlaneWave(energy=ENERGY, gpts=(32, 40), extent=(10.0, 12.0)).build(
        lazy=False
    )
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20, **other)

    with pytest.raises(RuntimeError, match="Inconsistent grid"):
        waves.match_grid(probe)

    assert waves.gpts == (32, 40)
    assert waves.extent == (10.0, 12.0)


def test_multislice_through_a_built_potential_on_a_grid_within_tolerance_keeps_it(
    potential_array,
):
    extent = potential_array.extent
    sampling = potential_array.sampling
    waves = abtem.PlaneWave(
        energy=ENERGY,
        gpts=GPTS,
        extent=tuple(e * (1 + 5e-6) for e in extent),
    ).build(lazy=False)

    waves.multislice(potential_array)

    assert potential_array.extent == extent
    assert potential_array.sampling == sampling


def test_matching_built_waves_fills_the_grid_of_a_builder_without_one():
    waves = abtem.PlaneWave(energy=ENERGY, gpts=(32, 40), extent=(10.0, 12.0)).build(
        lazy=False
    )
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)

    waves.match_grid(probe)

    assert probe.gpts == (32, 40) and probe.extent == (10.0, 12.0)
