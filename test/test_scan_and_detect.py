import hypothesis.strategies as st
import numpy as np
import pytest
from ase import Atoms
from hypothesis import given, assume, settings, reproduce_failure

import strats as abst
from abtem import Probe, SMatrix, GridScan
from abtem.potentials.temperature import AbstractFrozenPhonons


def _test_scan(probe, atoms, detectors, lazy):
    integration_limits = [detector.angular_limits(probe) for detector in detectors]
    outer_limit = max([outer for inner, outer in integration_limits])
    min_range = min([outer - inner for inner, outer in integration_limits])

    assume(min(probe.angular_sampling) < min_range)
    assume(outer_limit < min(probe.cutoff_angles))

    scan = GridScan()
    measurements = probe.scan(potential=atoms, scan=scan, detectors=detectors, lazy=lazy)
    measurements.compute()

    if not isinstance(measurements, list):
        measurements = [measurements]

    assert len(measurements) == len(detectors)

    for detector, measurement in zip(detectors, measurements):

        if isinstance(atoms, AbstractFrozenPhonons) and len(atoms) > 1 and not detector.ensemble_mean:
            frozen_phonons_shape = (len(atoms),)
        else:
            frozen_phonons_shape = ()

        base_shape = detector.measurement_shape(probe)
        assert base_shape == measurement.shape[len(measurement.shape) - len(base_shape):]

        scan_shape = scan.gpts
        assert scan_shape == measurement.shape[len(frozen_phonons_shape): len(frozen_phonons_shape) + len(scan_shape)]

        if frozen_phonons_shape:
            assert frozen_phonons_shape[0] == measurement.shape[0]

        assert not np.all(measurement.array == 0.)


@settings(deadline=None, max_examples=20, print_blob=True)
@given(atoms=abst.random_atoms(min_side_length=5, max_side_length=10) |
             abst.random_frozen_phonons(min_side_length=5, max_side_length=10),
       gpts=abst.gpts(min_value=64, max_value=128),
       semiangle_cutoff=st.floats(5, 10),
       energy=st.floats(100e3, 200e3),
       detectors=abst.detectors())
@pytest.mark.parametrize('lazy', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_probe_scan(atoms, gpts, semiangle_cutoff, energy, detectors, lazy, device):
    probe = Probe(gpts=gpts,
                  semiangle_cutoff=semiangle_cutoff,
                  energy=energy,
                  extent=np.diag(atoms.cell)[:2],
                  device=device)
    _test_scan(probe, atoms, detectors, lazy)


@settings(deadline=None, max_examples=20, print_blob=True)
@given(atoms=abst.random_atoms(min_side_length=5, max_side_length=10) |
             abst.random_frozen_phonons(min_side_length=5, max_side_length=10),
       gpts=abst.gpts(min_value=64, max_value=128),
       planewave_cutoff=st.floats(5, 10),
       energy=st.floats(100e3, 200e3),
       detectors=abst.detectors())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['gpu', 'cpu'])
def test_s_matrix_scan(atoms, gpts, planewave_cutoff, energy, detectors, lazy, device):
    probe = SMatrix(potential=atoms,
                    gpts=gpts,
                    planewave_cutoff=planewave_cutoff,
                    energy=energy,
                    device=device)
    integration_limits = [detector.angular_limits(probe) for detector in detectors]
    outer_limit = max([outer for inner, outer in integration_limits])
    min_range = min([outer - inner for inner, outer in integration_limits])

    assume(min(probe.angular_sampling) < min_range)
    assume(outer_limit < min(probe.cutoff_angles))

    scan = GridScan()
    measurements = probe.scan(scan=scan, detectors=detectors, lazy=lazy)
    measurements.compute()
