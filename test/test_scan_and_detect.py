import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings, assume

from abtem import Probe, SMatrix, GridScan, PlaneWave, LineScan, CustomScan, AnnularDetector
from strategies import atoms as atoms_strats
from strategies import core as core_strats
from strategies import detectors as detector_strats
from utils import gpu, assume_valid_probe_and_detectors, assert_scanned_measurement_as_expected


@settings(deadline=None, max_examples=40, print_blob=True)
@given(atoms=atoms_strats.random_atoms(min_side_length=5, max_side_length=10) |
             atoms_strats.random_frozen_phonons(min_side_length=5, max_side_length=10),
       gpts=core_strats.gpts(min_value=64, max_value=128),
       semiangle_cutoff=st.floats(5, 10),
       energy=st.floats(100e3, 200e3),
       scan=st.one_of((st.just(CustomScan()), st.just(LineScan()), st.just(GridScan()))),
       detectors=detector_strats.detectors())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_probe_scan(atoms, gpts, semiangle_cutoff, energy, detectors, lazy, scan, device):
    probe = Probe(gpts=gpts,
                  semiangle_cutoff=semiangle_cutoff,
                  energy=energy,
                  extent=np.diag(atoms.cell)[:2],
                  device=device)

    assume_valid_probe_and_detectors(probe, detectors)

    assume(not (isinstance(scan, CustomScan) and
                any([isinstance(detector, AnnularDetector) for detector in detectors])))

    measurements = probe.scan(potential=atoms, scan=scan, detectors=detectors, lazy=lazy)
    measurements.compute()

    assert_scanned_measurement_as_expected(measurements, atoms, probe, detectors, scan)


@settings(deadline=None, max_examples=20, print_blob=True)
@given(atoms=atoms_strats.random_atoms(min_side_length=5, max_side_length=10) |
             atoms_strats.random_frozen_phonons(min_side_length=5, max_side_length=10),
       gpts=core_strats.gpts(min_value=64, max_value=128),
       energy=st.floats(100e3, 200e3),
       detectors=detector_strats.detectors())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_planewave_detect(atoms, gpts, energy, detectors, lazy, device):
    planewave = PlaneWave(gpts=gpts,
                          energy=energy,
                          extent=np.diag(atoms.cell)[:2],
                          device=device)

    assume_valid_probe_and_detectors(planewave, detectors)
    assume(not any([isinstance(detector, AnnularDetector) for detector in detectors]))

    measurements = planewave.multislice(potential=atoms, detectors=detectors, lazy=lazy)
    measurements.compute()

    assert_scanned_measurement_as_expected(measurements, atoms, planewave, detectors, scan=None)


@settings(deadline=None, max_examples=40, print_blob=True)
@given(atoms=atoms_strats.random_atoms(min_side_length=5, max_side_length=10) |
             atoms_strats.random_frozen_phonons(min_side_length=5, max_side_length=10),
       gpts=core_strats.gpts(min_value=64, max_value=128),
       planewave_cutoff=st.floats(5, 10),
       energy=st.floats(100e3, 200e3),
       downsample=st.just('valid') | st.just('cutoff') | st.just(False),
       interpolation=st.integers(min_value=1, max_value=3),
       detectors=detector_strats.detectors())
@pytest.mark.parametrize('lazy', [True])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_s_matrix_scan(atoms, gpts, planewave_cutoff, energy, detectors, lazy, device, downsample, interpolation):
    s_matrix = SMatrix(potential=atoms,
                       gpts=gpts,
                       planewave_cutoff=planewave_cutoff,
                       interpolation=interpolation,
                       energy=energy,
                       device=device)

    if downsample:
        probe = s_matrix.build(stop=0, lazy=True).downsample(max_angle=downsample).comparable_probe()
    else:
        probe = s_matrix.build(stop=0, lazy=True).comparable_probe()

    assume_valid_probe_and_detectors(probe, detectors)

    scan = GridScan()
    measurements = s_matrix.scan(scan=scan, detectors=detectors, lazy=lazy, downsample=downsample)
    measurements.compute()

    assert_scanned_measurement_as_expected(measurements, atoms, probe, detectors, scan=scan)
