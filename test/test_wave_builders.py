import hypothesis.strategies as st
import numpy as np
import pytest
from ase import Atoms
from hypothesis import given, assume, settings

import strats as abst
from abtem import Probe, PlaneWave
from abtem.core.backend import get_array_module
from abtem.core.energy import EnergyUndefinedError, energy2wavelength
from abtem.core.grid import GridUndefinedError
from test_grid import grid_data, _test_grid_consistent
from utils import ensure_is_tuple, check_array_matches_device, check_array_matches_laziness

probe_data = st.fixed_dictionaries(
    {'energy': abst.energy(),
     'device': abst.device
     }
)

plane_wave_data = st.fixed_dictionaries(
    {'energy': abst.energy(),
     'device': abst.device,
     'normalize': st.booleans()}
)


@pytest.mark.parametrize("builder", [Probe, PlaneWave])
@given(grid_data=grid_data())
def test_grid_raises(grid_data, builder):
    probe = builder(**grid_data)
    try:
        probe.grid.check_is_defined()
    except GridUndefinedError:
        with pytest.raises(GridUndefinedError):
            probe.build()


@given(grid_data=grid_data(), energy=abst.energy(allow_none=True))
@pytest.mark.parametrize("builder", [Probe, PlaneWave])
def test_energy_raises(grid_data, energy, builder):
    probe = builder(energy=energy, **grid_data)
    assume(energy is None)
    with pytest.raises(EnergyUndefinedError):
        probe.build()


@settings(deadline=None)
@pytest.mark.parametrize("builder", [Probe, PlaneWave])
@given(grid_data=grid_data(), probe_data=probe_data)
def test_can_build(grid_data, probe_data, builder):
    probe = builder(**grid_data, **probe_data)
    wave = probe.build()
    check_array_matches_device(wave.array, probe_data['device'])
    assert wave.array.shape[-2:] == wave.gpts == ensure_is_tuple(grid_data['gpts'], 2)
    assert probe.extent == wave.extent == ensure_is_tuple(grid_data['extent'], 2)
    _test_grid_consistent(wave.extent, wave.gpts, wave.sampling)
    assert np.isclose(wave.energy, probe_data['energy'])


@settings(deadline=None)
@given(grid_data=grid_data(), probe_data=probe_data, lazy=st.booleans())
@pytest.mark.parametrize("builder", [Probe, PlaneWave])
def test_build_is_lazy(grid_data, probe_data, lazy, builder):
    builder = builder(**grid_data, **probe_data)
    wave = builder.build(lazy=lazy)
    assert len(wave.shape) == 2
    check_array_matches_laziness(wave.array, lazy=lazy)
    wave.compute()
    check_array_matches_laziness(wave.array, lazy=False)
    assert wave.array.shape[-2:] == builder.gpts == ensure_is_tuple(grid_data['gpts'], 2)
    assert wave.extent == builder.extent == ensure_is_tuple(grid_data['extent'], 2)


@settings(deadline=None)
@given(atom_data=abst.empty_atoms_data(),
       probe_data=probe_data,
       gpts=abst.gpts(),
       lazy=st.booleans())
@pytest.mark.parametrize("builder", [PlaneWave, Probe])
def test_multislice_is_lazy(atom_data, probe_data, gpts, lazy, builder):
    atoms = Atoms(**atom_data)
    probe = builder(gpts=gpts, **probe_data)
    wave = probe.multislice(atoms, lazy=lazy)
    assert wave.shape == probe.build().shape
    check_array_matches_laziness(wave.array, lazy=lazy)
    wave.compute()
    assert wave.shape == probe.build().shape
    check_array_matches_laziness(wave.array, lazy=False)


@settings(deadline=None)
@given(grid_data=grid_data(), plane_wave_data=plane_wave_data, lazy=st.booleans())
@pytest.mark.parametrize("builder", [Probe, PlaneWave])
def test_planewave_normalized(grid_data, plane_wave_data, lazy, builder):
    plane_wave = builder(**grid_data, **plane_wave_data)
    wave = plane_wave.build(lazy=lazy).compute()
    if plane_wave_data['normalize']:
        check_is_normalized(wave.array)


def check_is_normalized(array):
    xp = get_array_module(array)
    assert np.isclose((xp.abs(xp.fft.fft2(array)) ** 2).sum(), 1.)


@settings(deadline=None)
@given(atom_data=abst.empty_atoms_data(), plane_wave_data=plane_wave_data, gpts=abst.gpts(), lazy=st.booleans())
def test_planewave_empty_multislice(atom_data, plane_wave_data, gpts, lazy):
    atoms = Atoms(**atom_data)
    plane_wave = PlaneWave(gpts=gpts, **plane_wave_data)
    wave = plane_wave.multislice(atoms, lazy=lazy).compute()

    if plane_wave_data['normalize']:
        check_is_normalized(wave.array)


def antialias_cutoff_angle(sampling, energy):
    return energy2wavelength(energy) / sampling * 1e3 / 2 * 2 / 3


@settings(deadline=None)
@given(atom_data=abst.empty_atoms_data(),
       plane_wave_data=plane_wave_data,
       sampling=abst.sampling(),
       lazy=st.booleans())
def test_probe_empty_multislice(atom_data, plane_wave_data, sampling, lazy):
    atoms = Atoms(**atom_data)
    cutoff = 0.9 * antialias_cutoff_angle(max(ensure_is_tuple(sampling, 2)), plane_wave_data['energy'])
    plane_wave = Probe(semiangle_cutoff=cutoff, sampling=sampling, **plane_wave_data)
    wave = plane_wave.multislice(atoms, lazy=lazy).compute()
    if plane_wave_data['normalize']:
        check_is_normalized(wave.array)


@settings(deadline=None)
@given(atom_data=abst.random_atoms_data(),
       plane_wave_data=plane_wave_data,
       sampling=abst.sampling(min_value=.1),
       lazy=st.booleans())
def test_electrons_scatter(atom_data, plane_wave_data, sampling, lazy):
    atoms = Atoms(**atom_data)
    plane_wave = PlaneWave(sampling=sampling, **plane_wave_data)
    wave = plane_wave.multislice(atoms, lazy=lazy).compute()
    xp = get_array_module(wave.array)
    if plane_wave_data['normalize']:
        assert (xp.abs(xp.fft.fft2(wave.array)[..., 0, 0]) ** 2).sum() < 1.