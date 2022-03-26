import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings, assume, reproduce_failure

from abtem import Probe, SMatrix, GridScan, PlaneWave, LineScan, CustomScan, ParameterSeries
from strategies import atoms as atoms_st
from strategies import core as core_st
from strategies import detectors as detector_st
from utils import gpu, assert_scanned_measurement_as_expected

all_detectors = {'annular_detector': detector_st.annular_detector,
                 'segmented_detector': detector_st.segmented_detector,
                 'flexible_annular_detector': detector_st.flexible_annular_detector,
                 'pixelated_detector': detector_st.pixelated_detector,
                 'waves_detector': detector_st.waves_detector
                 }


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('detector', list(all_detectors.keys()))
@pytest.mark.parametrize('frozen_phonons', [True, False])
@pytest.mark.parametrize('scan', [CustomScan(), LineScan(), GridScan()])
def test_probe_scan(data, detector, lazy, scan, device, frozen_phonons):
    if frozen_phonons:
        atoms = data.draw(atoms_st.random_atoms(min_side_length=10, max_side_length=10))
    else:
        atoms = data.draw(atoms_st.random_frozen_phonons(min_side_length=10, max_side_length=10))

    probe = Probe(gpts=data.draw(core_st.gpts(min_value=32, max_value=64)),
                  semiangle_cutoff=data.draw(st.floats(5, 10)),
                  energy=data.draw(core_st.energy()),
                  extent=np.diag(atoms.cell)[:2],
                  device=device)

    detectors = [data.draw(all_detectors[detector](max_angle=min(probe.cutoff_angles), allow_detect_every=False))]

    if isinstance(scan, CustomScan) and detector == 'annular_detector':
        return

    measurements = probe.scan(potential=atoms, scan=scan, detectors=detectors, lazy=lazy)
    measurements.compute()

    assert_scanned_measurement_as_expected(measurements, atoms, probe, detectors, scan)


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('detector', ['pixelated_detector', 'segmented_detector', 'flexible_annular_detector',
                                      'waves_detector'])
@pytest.mark.parametrize('frozen_phonons', [True, False])
@pytest.mark.parametrize('defocus', [0., ParameterSeries(np.linspace(0, 100, 2))])
def test_cbed(data, detector, lazy, device, frozen_phonons, defocus):
    if frozen_phonons:
        atoms = data.draw(atoms_st.random_atoms(min_side_length=5, max_side_length=10))
    else:
        atoms = data.draw(atoms_st.random_frozen_phonons(min_side_length=5, max_side_length=10))

    wave = Probe(gpts=data.draw(core_st.gpts(min_value=32, max_value=64)),
                 energy=data.draw(core_st.energy()),
                 extent=np.diag(atoms.cell)[:2],
                 device=device,
                 semiangle_cutoff=10.,
                 defocus=defocus)

    detectors = [data.draw(all_detectors[detector](max_angle=min(wave.cutoff_angles), allow_detect_every=False))]

    measurements = wave.multislice(potential=atoms, detectors=detectors, lazy=lazy)
    measurements.compute()

    assert_scanned_measurement_as_expected(measurements, atoms, wave, detectors, scan=None, parameter_series=defocus)


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('detector', ['pixelated_detector', 'segmented_detector', 'flexible_annular_detector',
                                      'waves_detector'])
@pytest.mark.parametrize('frozen_phonons', [True, False])
def test_hrtem(data, detector, lazy, device, frozen_phonons):
    if frozen_phonons:
        atoms = data.draw(atoms_st.random_atoms(min_side_length=5, max_side_length=10))
    else:
        atoms = data.draw(atoms_st.random_frozen_phonons(min_side_length=5, max_side_length=10))

    wave = PlaneWave(gpts=data.draw(core_st.gpts(min_value=32, max_value=64)),
                     energy=data.draw(core_st.energy()),
                     extent=np.diag(atoms.cell)[:2],
                     device=device)

    detectors = [data.draw(all_detectors[detector](max_angle=min(wave.cutoff_angles), allow_detect_every=False))]

    measurements = wave.multislice(potential=atoms, detectors=detectors, lazy=lazy)
    measurements.compute()

    assert_scanned_measurement_as_expected(measurements, atoms, wave, detectors, scan=None)


@settings(print_blob=True)
@given(data=st.data(),
       gpts=core_st.gpts(min_value=32, max_value=64),
       planewave_cutoff=st.floats(5, 10),
       energy=st.floats(100e3, 200e3))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('detector', list(all_detectors.keys()))
@pytest.mark.parametrize('downsample', ['cutoff', False])
@pytest.mark.parametrize('interpolation', [1, 2, 3])
@pytest.mark.parametrize('frozen_phonons', [True, False])
def test_s_matrix_scan_and_detect(data,
                                  gpts,
                                  planewave_cutoff,
                                  energy,
                                  detector,
                                  lazy,
                                  device,
                                  downsample,
                                  interpolation,
                                  frozen_phonons):
    if frozen_phonons:
        atoms = data.draw(atoms_st.random_atoms(min_side_length=5, max_side_length=10))
    else:
        atoms = data.draw(atoms_st.random_frozen_phonons(min_side_length=5, max_side_length=10))

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

    detectors = [
        data.draw(all_detectors[detector](max_angle=np.floor(min(probe.cutoff_angles)), allow_detect_every=False))]

    scan = GridScan()
    measurements = s_matrix.scan(scan=scan, detectors=detectors, lazy=lazy, downsample=downsample)
    measurements.compute()

    assert_scanned_measurement_as_expected(measurements, atoms, probe, detectors, scan=scan)
