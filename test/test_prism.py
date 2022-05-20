# import hypothesis.strategies as st
# import numpy as np
# import pytest
# from hypothesis import given, settings, reproduce_failure, assume
#
# from strategies import core as core_st
# from strategies import atoms as atoms_st
# from abtem import Probe, SMatrix, GridScan, Potential
# from abtem.core.backend import get_array_module
# from strategies import detectors as detector_st
# from utils import gpu, assume_valid_probe_and_detectors, assert_scanned_measurement_as_expected
# from abtem.core.backend import cp
#
# #@reproduce_failure('6.29.3', b'AXicY2AgCTAyMAAAADUAAg==')
# #@settings(print_blob=True)
# @given(data=st.data(),
#        atoms=atoms_st.random_atoms(min_side_length=5, max_side_length=10),
#        gpts=core_st.gpts(min_value=32, max_value=64),
#        planewave_cutoff=st.floats(10, 15),
#        energy=st.floats(100e3, 200e3))
# @pytest.mark.parametrize('lazy', [True])
# @pytest.mark.parametrize('device', ['cpu'])
# def test_multislice_matches_prism(data, atoms, gpts, planewave_cutoff, energy, lazy, device):
#     detectors = data.draw(detector_st.detectors(allow_detect_every=lazy, max_detectors=1))
#
#     potential = Potential(atoms, device=device)
#     s_matrix = SMatrix(potential=potential, gpts=gpts, planewave_cutoff=planewave_cutoff, energy=energy, device=device)
#     probe = Probe(gpts=gpts, extent=potential.extent, semiangle_cutoff=planewave_cutoff, energy=energy, device=device)
#
#     assume_valid_probe_and_detectors(probe, detectors)
#
#     measurement = probe.scan(potential, detectors=detectors, lazy=lazy)
#     prism_measurement = s_matrix.scan(detectors=detectors, lazy=lazy, downsample=False)
#
#     assert np.allclose(measurement.array, prism_measurement.array)
#
#
# @given(atoms=atoms_st.random_atoms(min_side_length=5, max_side_length=10),
#        gpts=core_st.gpts(min_value=32, max_value=64),
#        planewave_cutoff=st.floats(10, 15),
#        interpolation=st.integers(min_value=1, max_value=4),
#        energy=st.floats(100e3, 200e3),
#        data=st.data())
# @pytest.mark.parametrize('lazy', [True, False])
# @pytest.mark.parametrize('device', ['cpu', gpu])
# def test_prism_interpolation(data, atoms, gpts, planewave_cutoff, energy, lazy, device, interpolation):
#     detectors = data.draw(detector_st.detectors(allow_detect_every=lazy, max_detectors=1))
#
#     potential = Potential(atoms, gpts=gpts, device=device).build().compute()
#     scan = GridScan(start=(0, 0), end=potential.extent)
#
#     probe = Probe(semiangle_cutoff=planewave_cutoff, energy=energy, device=device)
#     probe.grid.match(potential)
#
#     tiled_potential = potential.tile((interpolation,) * 2)
#     s_matrix = SMatrix(potential=tiled_potential, interpolation=interpolation, planewave_cutoff=planewave_cutoff,
#                        energy=energy, device=device)
#
#     diffraction_pattern = probe.build(lazy=False).diffraction_patterns()
#     prism_diffraction_pattern = s_matrix.build(stop=0, lazy=False).reduce().diffraction_patterns()
#
#     xp = get_array_module(device)
#     assume(xp.abs(diffraction_pattern.array - prism_diffraction_pattern.array).max() < 1e-6)
#     assume_valid_probe_and_detectors(probe, detectors)
#
#     measurement = probe.scan(potential, scan=scan, detectors=detectors, lazy=lazy)
#     prism_measurement = s_matrix.scan(scan=scan, detectors=detectors, lazy=lazy, downsample=False)
#
#     assert np.allclose(measurement.array, prism_measurement.array, atol=1e-6)
#
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
