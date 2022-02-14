import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings, reproduce_failure

import strats as abst
from abtem import Probe, SMatrix, GridScan, Potential
from abtem.core.backend import cp
from utils import gpu, assume_valid_probe_and_detectors, assert_scanned_measurement_as_expected


@settings(deadline=None, max_examples=20, print_blob=True)
@given(atoms=abst.random_atoms(min_side_length=5, max_side_length=10),
       gpts=abst.gpts(min_value=64, max_value=128),
       planewave_cutoff=st.floats(5, 15),
       energy=st.floats(100e3, 200e3),
       detectors=abst.detectors(max_detectors=1))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_multislice_match_prism(atoms, gpts, planewave_cutoff, energy, detectors, lazy, device):
    potential = Potential(atoms, device=device)
    s_matrix = SMatrix(potential=potential, gpts=gpts, planewave_cutoff=planewave_cutoff, energy=energy, device=device)
    probe = Probe(gpts=gpts, extent=potential.extent, semiangle_cutoff=planewave_cutoff, energy=energy, device=device)

    assume_valid_probe_and_detectors(probe, detectors)

    measurements = probe.scan(potential, detectors=detectors, lazy=lazy)
    prism_measurements = s_matrix.scan(detectors=detectors, lazy=lazy, downsample=False)

    assert np.allclose(measurements.array, prism_measurements.array)


@settings(deadline=None, max_examples=10, print_blob=True)
@given(atoms=abst.random_atoms(min_side_length=5, max_side_length=10),
       gpts=abst.gpts(min_value=64, max_value=128),
       planewave_cutoff=st.floats(15, 20),
       energy=st.floats(100e3, 200e3),
       interpolation=st.integers(min_value=2, max_value=3),
       detectors=abst.detectors(max_detectors=1))
@pytest.mark.parametrize('lazy', [False, True])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_prism_interpolation(atoms, gpts, planewave_cutoff, energy, detectors, lazy, device, interpolation):
    potential = Potential(atoms, gpts=gpts, device=device).build()
    scan = GridScan(start=(0, 0), end=potential.extent)

    probe = Probe(semiangle_cutoff=planewave_cutoff, energy=energy, device=device)
    probe.grid.match(potential)

    tiled_potential = potential.tile((interpolation,) * 2)
    s_matrix = SMatrix(potential=tiled_potential, interpolation=interpolation, planewave_cutoff=planewave_cutoff,
                       energy=energy, device=device)

    assume_valid_probe_and_detectors(probe, detectors)

    measurements = probe.scan(potential, scan=scan, detectors=detectors, lazy=lazy)
    prism_measurements = s_matrix.scan(scan=scan, detectors=detectors, lazy=lazy, downsample=False)

    assert np.allclose(measurements.array, prism_measurements.array, atol=1e-6)


@settings(deadline=None, max_examples=10, print_blob=True)
@given(atoms=abst.random_atoms(min_side_length=5, max_side_length=10),
       gpts=abst.gpts(min_value=64, max_value=128),
       planewave_cutoff=st.floats(5, 15),
       energy=st.floats(100e3, 200e3),
       interpolation=st.integers(min_value=1, max_value=4),
       detectors=abst.detectors(max_detectors=1))
@pytest.mark.parametrize('lazy', [False, True])
def test_store_on_host(atoms, gpts, planewave_cutoff, energy, detectors, lazy, interpolation):
    s_matrix = SMatrix(potential=atoms, gpts=gpts, interpolation=interpolation, planewave_cutoff=planewave_cutoff,
                       energy=energy, device='gpu', store_on_host=True)

    probe = s_matrix.meta_waves

    assert isinstance(s_matrix.build().compute().array, np.ndarray)

    assume_valid_probe_and_detectors(probe, detectors)

    scan = GridScan()
    measurements = s_matrix.scan(scan=scan, detectors=detectors, lazy=lazy, downsample=False)
    measurements.compute()

    assert_scanned_measurement_as_expected(measurements, atoms, probe, scan, detectors)


@settings(deadline=None, max_examples=10, print_blob=True)
@given(atoms=abst.random_atoms(min_side_length=5, max_side_length=10),
       gpts=abst.gpts(min_value=64, max_value=128),
       planewave_cutoff=st.floats(5, 15),
       energy=st.floats(100e3, 200e3),
       interpolation=st.integers(min_value=2, max_value=4),
       distribute_scan=st.tuples(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3)),
       detectors=abst.detectors(max_detectors=1))
@pytest.mark.parametrize('lazy', [True])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('store_on_host', [False, True])
def test_distribute_scan(atoms, gpts, planewave_cutoff, energy, detectors, lazy, distribute_scan, interpolation,
                         device, store_on_host):
    s_matrix = SMatrix(potential=atoms, gpts=gpts, interpolation=interpolation, planewave_cutoff=planewave_cutoff,
                       energy=energy, device=device, store_on_host=True)

    probe = s_matrix.meta_waves

    assert isinstance(s_matrix.build().compute().array, np.ndarray)

    assume_valid_probe_and_detectors(probe, detectors)

    scan = GridScan()
    measurements = s_matrix.scan(scan=scan, detectors=detectors, distribute_scan=distribute_scan, lazy=lazy)
    measurements.compute()

    assert_scanned_measurement_as_expected(measurements, atoms, probe, scan, detectors)
