import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings, reproduce_failure, assume

from strategies import core as core_st
from strategies import atoms as atoms_st
from abtem import Probe, SMatrix, GridScan, Potential
from abtem.core.backend import get_array_module
from strategies import detectors as detector_st
from utils import gpu, assume_valid_probe_and_detectors, assert_scanned_measurement_as_expected
from abtem.core.backend import cp


@given(gpts=core_st.gpts(),
       extent=core_st.extent(min_value=5, max_value=10),
       planewave_cutoff=st.floats(10, 15),
       energy=st.floats(100e3, 200e3),
       interpolation=st.integers(min_value=1, max_value=4))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_prism_normalized(gpts, extent, planewave_cutoff, energy, interpolation, lazy, device):
    s_matrix = SMatrix(potential=None,
                       planewave_cutoff=planewave_cutoff,
                       energy=energy,
                       extent=extent,
                       gpts=gpts,
                       interpolation=interpolation,
                       downsample='cutoff',
                       device=device)
    s_matrix = s_matrix.round_gpts_to_interpolation()
    assert np.isclose(s_matrix.reduce(lazy=lazy).diffraction_patterns(max_angle=None).array.sum(), 1.)


@given(extent=core_st.extent(min_value=5, max_value=10),
       gpts=core_st.gpts(min_value=32, max_value=64),
       planewave_cutoff=st.floats(10, 15),
       interpolation=st.integers(min_value=1, max_value=4),
       energy=st.floats(100e3, 200e3))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_prism_matches_probe(gpts, extent, planewave_cutoff, energy, lazy, interpolation, device):
    s_matrix = SMatrix(potential=None,
                       extent=extent,
                       gpts=gpts,
                       planewave_cutoff=planewave_cutoff,
                       energy=energy,
                       interpolation=interpolation,
                       device=device)

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    s_matrix_diffraction_patterns = s_matrix.reduce(lazy=lazy).diffraction_patterns(None)
    probe_diffraction_patterns = probe.build(lazy=lazy).diffraction_patterns(None)

    assert np.allclose(s_matrix_diffraction_patterns.array, probe_diffraction_patterns.array)


@given(atoms=atoms_st.random_atoms(min_side_length=5, max_side_length=10),
       gpts=core_st.gpts(min_value=32, max_value=64),
       planewave_cutoff=st.floats(10, 15),
       energy=st.floats(100e3, 200e3))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_prism_matches_probe_with_multislice(atoms, gpts, planewave_cutoff, energy, lazy, device):
    potential = Potential(atoms, gpts=gpts, device=device)
    s_matrix = SMatrix(potential=potential,
                       planewave_cutoff=planewave_cutoff,
                       energy=energy,
                       downsample=False,
                       device=device)

    s_matrix = s_matrix.round_gpts_to_interpolation()

    probe = Probe(energy=energy, semiangle_cutoff=planewave_cutoff)

    s_matrix_diffraction_patterns = s_matrix.reduce(lazy=lazy).diffraction_patterns(None)
    probe_diffraction_patterns = probe.multislice(potential=potential, lazy=lazy).diffraction_patterns(None)

    assert np.allclose(s_matrix_diffraction_patterns.array, probe_diffraction_patterns.array)


@given(data=st.data(),
       atoms=atoms_st.random_atoms(min_side_length=5, max_side_length=10),
       gpts=core_st.gpts(min_value=32, max_value=64),
       planewave_cutoff=st.floats(10, 15),
       energy=st.floats(100e3, 200e3))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_prism_matches_probe_with_scan(data, atoms, gpts, planewave_cutoff, energy, lazy, device):
    detectors = data.draw(detector_st.detectors(max_detectors=1))

    potential = Potential(atoms, device=device)
    s_matrix = SMatrix(potential=potential, gpts=gpts, planewave_cutoff=planewave_cutoff, energy=energy, device=device,
                       downsample=False)
    probe = Probe(gpts=gpts, extent=potential.extent, semiangle_cutoff=planewave_cutoff, energy=energy, device=device)

    assume_valid_probe_and_detectors(probe, detectors)

    measurement = probe.scan(potential, detectors=detectors, lazy=lazy)
    prism_measurement = s_matrix.scan(detectors=detectors, lazy=lazy)

    assert np.allclose(measurement.array, prism_measurement.array)


@given(atoms=atoms_st.random_atoms(min_side_length=5, max_side_length=10),
       gpts=core_st.gpts(min_value=32, max_value=64),
       planewave_cutoff=st.floats(10, 15),
       interpolation=st.integers(min_value=1, max_value=4),
       energy=st.floats(100e3, 200e3),
       data=st.data())
@pytest.mark.parametrize('lazy', [True])
@pytest.mark.parametrize('device', ['cpu'])
def test_prism_interpolation(data, atoms, gpts, planewave_cutoff, energy, lazy, device, interpolation):
    detectors = data.draw(detector_st.detectors(max_detectors=1))

    potential = Potential(atoms, gpts=gpts, device=device).build(lazy=lazy)
    # scan = GridScan(start=(0, 0), end=potential.extent)
    #
    # probe = Probe(semiangle_cutoff=planewave_cutoff, energy=energy, device=device)
    # probe.grid.match(potential)
    
    #tiled_potential = potential.tile((interpolation,) * 2)
    s_matrix = SMatrix(potential=potential, interpolation=interpolation, planewave_cutoff=planewave_cutoff,
                       energy=energy, device=device, downsample=False)

    #diffraction_pattern = probe.build(lazy=False).diffraction_patterns()
    prism_diffraction_pattern = s_matrix.build(stop=0, lazy=False).reduce().diffraction_patterns()



    # xp = get_array_module(device)
    # assume(xp.abs(diffraction_pattern.array - prism_diffraction_pattern.array).max() < 1e-6)
    # assume_valid_probe_and_detectors(probe, detectors)
    #
    # measurement = probe.scan(potential, scan=scan, detectors=detectors, lazy=lazy)
    # prism_measurement = s_matrix.scan(scan=scan, detectors=detectors, lazy=lazy)
    #
    # assert np.allclose(measurement.array, prism_measurement.array, atol=1e-6)

#
#
# @given(atoms=atoms_st.random_atoms(min_side_length=5, max_side_length=10),
#        gpts=core_st.gpts(min_value=32, max_value=64),
#        planewave_cutoff=st.floats(10, 15),
#        energy=st.floats(100e3, 200e3),
#        interpolation=st.integers(min_value=1, max_value=4),
#        data=st.data())
# @pytest.mark.parametrize('lazy', [False, True])
# @pytest.mark.skipif(cp is None, reason="no gpu")
# def test_store_on_host(data, atoms, gpts, planewave_cutoff, energy, lazy, interpolation):
#     detectors = data.draw(detector_st.detectors(allow_detect_every=lazy, max_detectors=1))
#
#     s_matrix = SMatrix(potential=atoms, gpts=gpts, interpolation=interpolation, planewave_cutoff=planewave_cutoff,
#                        energy=energy, device='gpu', store_on_host=True)
#
#     probe = s_matrix.build(stop=0, lazy=True).comparable_probe()
#
#     assert isinstance(s_matrix.build().compute().array, np.ndarray)
#
#     assume_valid_probe_and_detectors(probe, detectors)
#
#     scan = GridScan()
#     measurements = s_matrix.scan(scan=scan, detectors=detectors, lazy=lazy, downsample=False)
#     measurements.compute()
#
#     assert_scanned_measurement_as_expected(measurements, atoms, probe, detectors, scan)
#
#
#
# @given(atoms=atoms_st.random_atoms(min_side_length=5, max_side_length=10),
#        gpts=core_st.gpts(min_value=32, max_value=64),
#        planewave_cutoff=st.floats(5, 15),
#        energy=st.floats(100e3, 200e3),
#        interpolation=st.integers(min_value=2, max_value=4),
#        distribute_scan=st.tuples(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3)),
#        data=st.data())
# @pytest.mark.parametrize('lazy', [True])
# @pytest.mark.parametrize('device', ['cpu', gpu])
# @pytest.mark.parametrize('store_on_host', [False, True])
# def test_distribute_scan(data, atoms, gpts, planewave_cutoff, energy, lazy, distribute_scan, interpolation,
#                          device, store_on_host):
#     detectors = data.draw(detector_st.detectors(allow_detect_every=lazy, max_detectors=1))
#
#     s_matrix = SMatrix(potential=atoms, gpts=gpts, interpolation=interpolation, planewave_cutoff=planewave_cutoff,
#                        energy=energy, device=device, store_on_host=True)
#
#     probe = s_matrix.build(stop=0, lazy=True).comparable_probe()
#
#     assert isinstance(s_matrix.build().compute().array, np.ndarray)
#
#     assume_valid_probe_and_detectors(probe, detectors)
#
#     scan = GridScan()
#     measurements = s_matrix.scan(scan=scan, detectors=detectors, distribute_scan=distribute_scan, lazy=lazy,
#                                  downsample=False)
#     measurements.compute()
#
#     assert_scanned_measurement_as_expected(measurements, atoms, probe, detectors, scan)
