import hypothesis.strategies as st
import numpy as np
import pytest
from ase import Atoms
from hypothesis import given, assume, settings

import strats as abst
from strategies import core as core_st
from abtem import Probe, PlaneWave, SMatrix, Waves
from abtem.core.backend import get_array_module
from abtem.core.energy import EnergyUndefinedError, energy2wavelength
from abtem.core.grid import GridUndefinedError
from test_grid import grid_data, _test_grid_consistent
from utils import ensure_is_tuple, check_array_matches_device, gpu

wave_data = st.fixed_dictionaries(
    {'energy': abst.energy()}
)


@pytest.mark.parametrize("builder", [Probe, PlaneWave, SMatrix])
@given(grid_data=grid_data())
def test_grid_raises(grid_data, builder):
    probe = builder(**grid_data)
    try:
        probe.grid.check_is_defined()
    except GridUndefinedError:
        with pytest.raises(GridUndefinedError):
            probe.build()


@given(grid_data=grid_data(), energy=abst.energy(allow_none=True))
@pytest.mark.parametrize("builder", [Probe, PlaneWave, SMatrix])
def test_energy_raises(grid_data, energy, builder):
    probe = builder(energy=energy, **grid_data)
    assume(energy is None)
    with pytest.raises(EnergyUndefinedError):
        probe.build()


@settings(deadline=None)
@pytest.mark.parametrize("builder", [Probe, PlaneWave, SMatrix])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("lazy", [True, False])
@given(grid_data=grid_data(), energy=abst.energy())
def test_can_build(grid_data, energy, builder, device, lazy):
    probe = builder(**grid_data, energy=energy, device=device)
    wave = probe.build(lazy=lazy)
    check_array_matches_device(wave.array, device)
    assert wave.array.shape[-2:] == wave.gpts == ensure_is_tuple(grid_data['gpts'], 2)
    assert np.all(np.isclose(probe.extent, wave.extent))
    assert np.all(np.isclose(probe.extent, ensure_is_tuple(grid_data['extent'], 2)))
    _test_grid_consistent(wave.extent, wave.gpts, wave.sampling)
    assert np.isclose(wave.energy, energy)


# @settings(deadline=None)
# @given(grid_data=grid_data(), wave_data=wave_data)
# @pytest.mark.parametrize("builder", [Probe, PlaneWave, SMatrix])
# @pytest.mark.parametrize("lazy", [True, False])
# def test_can_compute(grid_data, wave_data, lazy, builder):
#     builder = builder(**grid_data, **wave_data)
#     wave = builder.build(lazy=lazy)
#     assert len(wave.shape) == len(builder.base_axes)
#     check_array_matches_laziness(wave.array, lazy=lazy)
#     wave.compute()
#     check_array_matches_laziness(wave.array, lazy=False)
#     assert wave.array.shape[-2:] == builder.gpts == ensure_is_tuple(grid_data['gpts'], 2)
#     assert np.all(np.isclose(builder.extent, wave.extent))
#     assert np.all(np.isclose(builder.extent, ensure_is_tuple(grid_data['extent'], 2)))
#
#
# @settings(deadline=None)
# @given(atom_data=abst.empty_atoms_data(),
#        wave_data=wave_data,
#        gpts=abst.gpts())
# @pytest.mark.parametrize("builder", [PlaneWave, Probe, SMatrix])
# @pytest.mark.parametrize("lazy", [True, False])
# @pytest.mark.parametrize("device", ['cpu', gpu])
# def test_can_multislice(atom_data, wave_data, gpts, lazy, builder, device):
#     atoms = Atoms(**atom_data)
#     probe = builder(gpts=gpts, **wave_data, device=device)
#     wave = probe.multislice(atoms, lazy=lazy)
#
#     assert wave.shape == probe.build().shape
#     if hasattr(wave, 'reduce'):
#         wave = wave.reduce()
#
#     check_array_matches_laziness(wave.array, lazy=lazy)
#     shape = wave.shape
#     wave = wave.compute()
#     assert wave.shape == shape
#     check_array_matches_laziness(wave.array, lazy=False)


@settings(deadline=None)
@given(grid_data=grid_data(), energy=abst.energy())
@pytest.mark.parametrize("builder", [Probe, PlaneWave, SMatrix])
@pytest.mark.parametrize("lazy", [True, False])
def test_normalized(grid_data, energy, lazy, builder):
    builder = builder(**grid_data, energy=energy)

    if hasattr(builder, '_normalize'):
        builder._normalize = True

    wave = builder.build(lazy=lazy)
    wave = wave.compute()

    if hasattr(wave, 'reduce'):
        wave = wave.reduce()

    # if wave_data['normalize']:
    check_is_normalized(wave.array)


def check_is_normalized(array):
    xp = get_array_module(array)
    assert np.isclose((xp.abs(xp.fft.fft2(array)) ** 2).sum(), 1., atol=1e-6)


@settings(deadline=None)
@given(atom_data=abst.empty_atoms_data(), energy=abst.energy(), gpts=abst.gpts())
@pytest.mark.parametrize("lazy", [True, False])
def test_planewave_empty_multislice(atom_data, energy, gpts, lazy):
    atoms = Atoms(**atom_data)
    plane_wave = PlaneWave(gpts=gpts, energy=energy, extent=10)

    wave = plane_wave.build()

    #print(wave.array.sum().compute())
    wave = plane_wave.multislice(atoms, lazy=lazy).compute()
    #check_is_normalized(wave.array)


def antialias_cutoff_angle(sampling, energy):
    return energy2wavelength(energy) / sampling * 1e3 / 2 * 2 / 3


@settings(deadline=None)
@given(atom_data=abst.empty_atoms_data(),
       energy=abst.energy(),
       sampling=abst.sampling())
@pytest.mark.parametrize("lazy", [True, False])
def test_probe_empty_multislice(atom_data, energy, sampling, lazy):
    atoms = Atoms(**atom_data)
    cutoff = 0.9 * antialias_cutoff_angle(max(ensure_is_tuple(sampling, 2)), energy)
    plane_wave = Probe(semiangle_cutoff=cutoff, sampling=sampling, energy=energy)
    wave = plane_wave.multislice(atoms, lazy=lazy).compute()
    check_is_normalized(wave.array)


@settings(deadline=None)
@given(atom_data=abst.empty_atoms_data(),
       energy=abst.energy(),
       planewave_cutoff=st.floats(min_value=5., max_value=10.),
       sampling=abst.sampling())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
def test_s_matrix_empty_multislice(atom_data, energy, sampling, lazy, planewave_cutoff, device):
    atoms = Atoms(**atom_data)
    max_planewave_cutoff = 0.9 * antialias_cutoff_angle(max(ensure_is_tuple(sampling, 2)), energy)
    planewave_cutoff = min(planewave_cutoff, max_planewave_cutoff)
    plane_wave = SMatrix(atoms, planewave_cutoff=planewave_cutoff, sampling=sampling, device=device,
                         energy=energy)
    wave = plane_wave.build(lazy=lazy).reduce().compute()
    check_is_normalized(wave.array)


@settings(deadline=None)
@given(atoms_data=abst.random_atoms_data(),
       energy=abst.energy(),
       sampling=abst.sampling(min_value=.1))
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("builder", [Probe, PlaneWave])
def test_scatter(atoms_data, builder, energy, sampling, lazy):
    atoms = Atoms(**atoms_data)
    builder = builder(sampling=sampling, energy=energy)
    wave = builder.multislice(atoms, lazy=lazy).compute()
    xp = get_array_module(wave.array)
    assert (xp.abs(xp.fft.fft2(wave.array)[..., 0, 0]) ** 2).sum() < 1.


@given(gpts=core_st.gpts(),
       energy=core_st.energy(),
       sampling=core_st.sampling(min_value=.1))
def test_downsample(gpts, sampling, energy):
    gpts = ensure_is_tuple(gpts, 2)
    waves = Waves(np.ones(gpts, dtype=np.complex64), sampling=sampling, energy=energy)

    amplitude = np.abs(waves.array).sum()

    downsampled_waves = waves.downsample(waves.full_cutoff_angles[0] / 2, normalization='values')

    assert downsampled_waves

    assert np.allclose(downsampled_waves.array, 1)

    downsampled_waves = waves.downsample(waves.full_cutoff_angles[0] / 2, normalization='amplitude')

    assert np.allclose(np.abs(downsampled_waves.array).sum(), amplitude)


@settings(deadline=None)
@given(atoms_data=abst.random_atoms_data(),
       energy=abst.energy(),
       sampling=abst.sampling(min_value=.1))
@pytest.mark.parametrize("lazy", [True, False])
def test_build_then_multislice(atoms_data, energy, sampling, lazy):
    atoms = Atoms(**atoms_data)
    plane_wave = PlaneWave(sampling=sampling, energy=energy)
    wave = plane_wave.multislice(atoms, lazy=lazy).compute()
    xp = get_array_module(wave.array)

    assert (xp.abs(xp.fft.fft2(wave.array)[..., 0, 0]) ** 2).sum() < 1.
