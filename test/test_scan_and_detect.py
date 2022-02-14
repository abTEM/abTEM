import cupy as cp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

import strats as abst
from abtem import Probe, SMatrix, GridScan
from abtem.potentials.temperature import AbstractFrozenPhonons
from utils import gpu, assume_valid_probe_and_detectors, assert_scanned_measurement_as_expected




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

    assert_scanned_measurement_as_expected(measurements, atoms, probe, scan, detectors)


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

    assert_scanned_measurement_as_expected(measurements, atoms, probe, scan, detectors)
