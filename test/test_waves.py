import hypothesis.strategies as st
import numpy as np
import pytest
from ase import Atoms
from hypothesis import given, assume, settings, reproduce_failure
from matplotlib import pyplot as plt

from strategies import core as core_st
from strategies import atoms as atoms_st
from abtem.waves.waves import Probe, PlaneWave, Waves
from abtem.waves.prism import SMatrix
from abtem.core.backend import get_array_module
from abtem.core.energy import EnergyUndefinedError, energy2wavelength
from abtem.core.grid import GridUndefinedError
from test_grid import grid_data, check_grid_consistent
from utils import ensure_is_tuple, assert_array_matches_device, gpu, remove_dummy_dimensions, \
    assert_array_matches_laziness
import strategies.waves as waves_st
import strategies.prism as prism_st

import strategies.potentials as potentials_st
import strategies.scan as scan_st
import strategies.transfer as transfer_st


# @pytest.mark.parametrize("builder", [Probe, PlaneWave, SMatrix])
# @given(grid_data=grid_data())
# def test_grid_raises(grid_data, builder):
#     probe = builder(**grid_data)
#     try:
#         probe.grid.check_is_defined()
#     except GridUndefinedError:
#         with pytest.raises(GridUndefinedError):
#             probe.build()
#
#
# @given(grid_data=grid_data(), energy=core_st.energy(allow_none=True))
# @pytest.mark.parametrize("builder", [Probe, PlaneWave, SMatrix])
# def test_energy_raises(grid_data, energy, builder):
#     probe = builder(energy=energy, **grid_data)
#     assume(energy is None)
#     with pytest.raises(EnergyUndefinedError):
#         probe.build()


@pytest.mark.parametrize("waves_builder", [
    waves_st.random_probe,
    waves_st.random_planewave,
    prism_st.random_s_matrix
])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("lazy", [True, False])
@given(data=st.data())
def test_can_build(data, waves_builder, device, lazy):
    waves_builder = data.draw(waves_builder(device=device))

    waves = waves_builder.build(lazy=lazy)

    assert_array_matches_device(waves.array, device)

    assert waves.gpts == waves_builder.gpts
    assert remove_dummy_dimensions(waves_builder.ensemble_shape) == waves.ensemble_shape
    assert waves.array.shape[-2:] == waves.gpts
    assert waves.array.shape[:-len(waves.base_shape)] == waves.ensemble_shape

    assert np.all(np.isclose(waves_builder.extent, waves.extent))
    check_grid_consistent(waves.extent, waves.gpts, waves.sampling)

    assert np.isclose(waves.energy, waves_builder.energy)


@given(data=st.data())
@pytest.mark.parametrize("waves_builder", [
    waves_st.random_probe,
    waves_st.random_planewave,
    prism_st.random_s_matrix
])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("lazy", [True, False])
def test_can_compute(data, waves_builder, device, lazy):
    waves_builder = data.draw(waves_builder(device=device))
    waves = waves_builder.build(lazy=lazy)

    assert_array_matches_laziness(waves.array, lazy=lazy)
    waves.compute()

    assert_array_matches_laziness(waves.array, lazy=False)
    assert waves.array.shape[-2:] == waves_builder.gpts
    assert waves.array.shape[:-len(waves.base_shape)] == waves.ensemble_shape


@given(data=st.data(), potential=potentials_st.random_potential())
@pytest.mark.parametrize("waves_builder", [
    waves_st.random_probe,
    waves_st.random_planewave,
    prism_st.random_s_matrix,
])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
def test_can_multislice(data, potential, waves_builder, lazy, device):
    waves_builder = data.draw(waves_builder(device=device))
    waves_builder.grid.match(potential)

    waves = waves_builder.multislice(potential, lazy=lazy)
    assert remove_dummy_dimensions(waves_builder.ensemble_shape) + waves_builder.base_shape == waves.shape
    assert_array_matches_laziness(waves.array, lazy=lazy)
    waves.compute()
    assert remove_dummy_dimensions(waves_builder.ensemble_shape) + waves_builder.base_shape == waves.shape
    assert_array_matches_laziness(waves.array, lazy=False)


def assert_is_normalized(waves):
    assert np.allclose(waves.diffraction_patterns(max_angle=None).array.sum(axis=(-2, -1)), 1.)


@given(data=st.data())
@pytest.mark.parametrize("waves_builder", [
    waves_st.random_probe,
    waves_st.random_planewave,
    prism_st.random_s_matrix,
])
@pytest.mark.parametrize("lazy", [True, False])
def test_normalized(data, waves_builder, lazy):
    waves_builder = data.draw(waves_builder())
    waves = waves_builder.build(lazy=lazy)
    try:
        waves = waves.reduce()
    except AttributeError:
        pass
    waves.compute()
    assert_is_normalized(waves)


@given(data=st.data(), atoms=atoms_st.empty_atoms())
@pytest.mark.parametrize("waves_builder", [
    waves_st.random_probe,
    waves_st.random_planewave,
    prism_st.random_s_matrix,
])
@pytest.mark.parametrize("lazy", [True, False])
def test_empty_multislice_normalized(data, atoms, waves_builder, lazy):
    waves_builder = data.draw(waves_builder())
    waves = waves_builder.multislice(atoms, lazy=lazy)
    try:
        waves = waves.reduce()
    except AttributeError:
        pass
    waves.compute()
    assert_is_normalized(waves)


@given(data=st.data(), potential=potentials_st.gold_potential())
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("waves_builder", [
    waves_st.random_probe,
    waves_st.random_planewave,
    prism_st.random_s_matrix,
])
def test_multislice_scatter(data, potential, waves_builder, lazy):
    waves_builder = data.draw(waves_builder())
    waves_builder.grid.match(potential)

    waves = waves_builder.multislice(potential, lazy=lazy).compute()

    try:
        waves = waves.reduce()
    except AttributeError:
        pass

    assert np.all(waves.diffraction_patterns(max_angle=None).array.sum(axis=(-2, -1)) < 1.) or \
           np.allclose(waves.diffraction_patterns(max_angle=None).array.sum(axis=(-2, -1)), 1.)

    build_waves = waves_builder.build(lazy=lazy)
    build_waves = build_waves.multislice(potential).compute()

    try:
        build_waves = build_waves.reduce()
    except AttributeError:
        pass
    assert np.allclose(build_waves.array, waves.array)


@given(data=st.data())
@pytest.mark.parametrize("transform", [
    scan_st.grid_scan,
    scan_st.line_scan,
    scan_st.custom_scan,
    transfer_st.random_aberrations,
    transfer_st.random_aperture,
    transfer_st.random_temporal_envelope,
    transfer_st.random_spatial_envelope,
    transfer_st.random_composite_wave_transform,
    transfer_st.random_ctf,
])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
def test_apply_transform(data, transform, lazy, device):
    waves = data.draw(waves_st.random_waves(lazy=lazy, device=device))
    transform = data.draw(transform())
    assume(len(transform.ensemble_shape + waves.shape) < 5)
    transformed_waves = waves.apply_transform(transform)
    assert transformed_waves.shape == transform.ensemble_shape + waves.shape


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True])
@pytest.mark.parametrize("device", ['cpu'])
def test_expand_dims(data, lazy, device):
    waves = data.draw(waves_st.random_waves(lazy=lazy, device=device))
    expanded = waves.expand_dims((0,))
    assert expanded.shape[0] == 1
    expanded = expanded.expand_dims((1,))
    assert expanded.shape[1] == 1


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
def test_squeeze(data, lazy, device):
    waves = data.draw(waves_st.random_waves(lazy=lazy, device=device))
    squeezed = waves.squeeze()
    assert remove_dummy_dimensions(waves.ensemble_shape) + waves.base_shape == squeezed.shape

# @given(gpts=core_st.gpts(),
#        energy=core_st.energy(),
#        sampling=core_st.sampling(min_value=.1))
# def test_downsample(gpts, sampling, energy):
#     gpts = ensure_is_tuple(gpts, 2)
#     waves = Waves(np.ones(gpts, dtype=np.complex64), sampling=sampling, energy=energy)
#
#     amplitude = np.abs(waves.array).sum()
#
#     downsampled_waves = waves.downsample(waves.full_cutoff_angles[0] / 2, normalization='values')
#
#     assert downsampled_waves
#
#     assert np.allclose(downsampled_waves.array, 1)
#
#     downsampled_waves = waves.downsample(waves.full_cutoff_angles[0] / 2, normalization='amplitude')
#
#     assert np.allclose(np.abs(downsampled_waves.array).sum(), amplitude)
#
#
# @settings(deadline=None)
# @given(atoms_data=atoms_st.random_atoms_data(),
#        energy=core_st.energy(),
#        sampling=core_st.sampling(min_value=.1))
# @pytest.mark.parametrize("lazy", [True, False])
# def test_build_then_multislice(atoms_data, energy, sampling, lazy):
#     atoms = Atoms(**atoms_data)
#     plane_wave = PlaneWave(sampling=sampling, energy=energy)
#     wave = plane_wave.multislice(atoms, lazy=lazy).compute()
#     xp = get_array_module(wave.array)
#
#     assert (xp.abs(xp.fft.fft2(wave.array)[..., 0, 0]) ** 2).sum() < 1.
