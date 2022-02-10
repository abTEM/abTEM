import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, assume, settings, reproduce_failure

import strats as abst
from abtem import Probe, SMatrix, GridScan, Potential
from abtem.potentials.temperature import AbstractFrozenPhonons
from utils import gpu, ensure_is_tuple


def assume_valid_probe_and_detectors(probe, detectors):
    integration_limits = [detector.angular_limits(probe) for detector in detectors]
    outer_limit = max([outer for inner, outer in integration_limits])
    min_range = min([outer - inner for inner, outer in integration_limits])
    assume(min(probe.angular_sampling) < min_range)
    assume(outer_limit < min(probe.cutoff_angles))


def assert_measurement_expected(measurements, atoms, waves, scan, detectors):
    if not isinstance(measurements, list):
        measurements = [measurements]

    assert len(measurements) == len(detectors)

    for detector, measurement in zip(detectors, measurements):

        if isinstance(atoms, AbstractFrozenPhonons) and len(atoms) > 1 and not detector.ensemble_mean:
            frozen_phonons_shape = (len(atoms),)
        else:
            frozen_phonons_shape = ()

        scan_shape = scan.gpts
        assert scan_shape == measurement.shape[len(frozen_phonons_shape): len(frozen_phonons_shape) + len(scan_shape)]

        if frozen_phonons_shape:
            assert frozen_phonons_shape[0] == measurement.shape[0]

        base_shape = detector.measurement_shape(waves)[len(frozen_phonons_shape + scan_shape):]
        assert base_shape == measurement.shape[len(measurement.shape) - len(base_shape):]
        assert not np.all(measurement.array == 0.)


@settings(deadline=None, max_examples=20, print_blob=True)
@given(atoms=abst.random_atoms(min_side_length=5, max_side_length=10) |
             abst.random_frozen_phonons(min_side_length=5, max_side_length=10),
       gpts=abst.gpts(min_value=64, max_value=128),
       semiangle_cutoff=st.floats(5, 10),
       energy=st.floats(100e3, 200e3),
       detectors=abst.detectors())
@pytest.mark.parametrize('lazy', [False, True])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_probe_scan(atoms, gpts, semiangle_cutoff, energy, detectors, lazy, device):
    probe = Probe(gpts=gpts,
                  semiangle_cutoff=semiangle_cutoff,
                  energy=energy,
                  extent=np.diag(atoms.cell)[:2],
                  device=device)

    assume_valid_probe_and_detectors(probe, detectors)

    scan = GridScan()
    measurements = probe.scan(potential=atoms, scan=scan, detectors=detectors, lazy=lazy)
    measurements.compute()

    assert_measurement_expected(measurements, atoms, probe, scan, detectors)


@settings(deadline=None, max_examples=20, print_blob=True)
@given(atoms=abst.random_atoms(min_side_length=5, max_side_length=10) |
             abst.random_frozen_phonons(min_side_length=5, max_side_length=10),
       gpts=abst.gpts(min_value=64, max_value=128),
       planewave_cutoff=st.floats(5, 10),
       energy=st.floats(100e3, 200e3),
       downsample=st.just('valid') | st.just('cutoff') | st.just(False),
       interpolation=st.integers(min_value=1, max_value=3),
       detectors=abst.detectors())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_s_matrix_scan(atoms, gpts, planewave_cutoff, energy, detectors, lazy, device, downsample, interpolation):
    s_matrix = SMatrix(potential=atoms,
                       gpts=gpts,
                       planewave_cutoff=planewave_cutoff,
                       interpolation=interpolation,
                       energy=energy,
                       device=device)

    probe = s_matrix.build(downsample=downsample).meta_waves
    assume_valid_probe_and_detectors(probe, detectors)

    scan = GridScan()
    measurements = s_matrix.scan(scan=scan, detectors=detectors, lazy=lazy, downsample=downsample)
    measurements.compute()

    assert_measurement_expected(measurements, atoms, probe, scan, detectors)


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


@settings(deadline=None, max_examples=100, print_blob=True)
@given(atoms=abst.random_atoms(min_side_length=5, max_side_length=10),
       gpts=abst.gpts(min_value=64, max_value=128),
       planewave_cutoff=st.floats(5, 15),
       energy=st.floats(100e3, 200e3),
       interpolation=st.integers(min_value=2, max_value=4),
       detectors=abst.detectors(max_detectors=1))
@pytest.mark.parametrize('lazy', [False, True])
@pytest.mark.parametrize('device', ['cpu'])
def test_prism_interpolation(atoms, gpts, planewave_cutoff, energy, detectors, lazy, device, interpolation):
    # gpts = ensure_is_tuple(gpts, 2)
    # gpts = (int(np.ceil(gpts[0] / 2) * 2), int(np.ceil(gpts[1] / 2) * 2))

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
