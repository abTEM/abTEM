from typing import Iterable

import dask.array as da
import numpy as np
import pytest
from hypothesis import assume

from abtem.core.backend import get_array_module, cp
from abtem.potentials.potentials import Potential
from abtem.potentials.temperature import AbstractFrozenPhonons
from abtem.waves.core import Waves


def assert_array_matches_device(array, device):
    assert get_array_module(array) is get_array_module(device)


def assert_array_matches_laziness(array, lazy):
    if lazy:
        assert isinstance(array, da.core.Array)
    else:
        assert not isinstance(array, da.core.Array)


def remove_dummy_dimensions(shape):
    return tuple(s for s in shape if s > 1)


def ensure_is_tuple(x, length: int = 1):
    if not isinstance(x, tuple):
        x = (x,) * length
    elif isinstance(x, Iterable):
        x = tuple(x)
    assert len(x) == length
    return x


def array_is_close(a1, a2, rel_tol=np.inf, abs_tol=np.inf, check_above_abs=0., check_above_rel=0., mask=None):
    if mask is not None:
        a1 = a1[mask]
        a2 = a2[mask]

    if rel_tol < np.inf:
        element_is_checked = (a2 > check_above_abs) * (a2 > (a2.max() * check_above_rel))
        rel_error = (a1[element_is_checked] - a2[element_is_checked]) / a2[element_is_checked]
        if np.any(np.abs(rel_error) > rel_tol):
            return False

    if abs_tol < np.inf:
        if np.any(np.abs(a1 - a2) > abs_tol):
            return False

    return True


def assume_valid_probe_and_detectors(probe, detectors):
    integration_limits = [detector.angular_limits(probe) for detector in detectors]
    outer_limit = max([outer for inner, outer in integration_limits])
    min_range = min([outer - inner for inner, outer in integration_limits])
    assume(min(probe.angular_sampling) < min_range)
    assume(outer_limit <= min(probe.cutoff_angles))


def assert_scanned_measurement_as_expected(measurements, atoms, waves, detectors, scan=None, parameter_series=None):
    if not isinstance(measurements, list):
        measurements = [measurements]

    assert len(measurements) == len(detectors)

    for detector, measurement in zip(detectors, measurements):

        expected_shape = ()

        if isinstance(atoms, AbstractFrozenPhonons):
            if (not atoms.ensemble_mean) or isinstance(measurement, Waves):
                expected_shape = (len(atoms),)

        if parameter_series is not None:
            if hasattr(parameter_series, '__len__') and not parameter_series.ensemble_mean:
                expected_shape += (len(parameter_series),)

        if detector.detect_every:
            num_detect_thicknesses = len(Potential(atoms)) // detector.detect_every
            if len(Potential(atoms)) % detector.detect_every != 0:
                num_detect_thicknesses += 1

            if num_detect_thicknesses > 1:
                expected_shape += (num_detect_thicknesses,)

        if scan is not None:
            expected_shape += scan.shape

        expected_shape = tuple(s for s in expected_shape if s > 1)
        expected_shape += detector.measurement_shape(waves)

        assert expected_shape == measurement.shape
        # assert not np.all(measurement.array == 0.)

        if detector.to_cpu:
            assert isinstance(measurement.array, np.ndarray)
        elif waves.device == 'gpu':
            assert isinstance(measurement.array, cp.ndarray)


gpu = pytest.param('gpu', marks=pytest.mark.skipif(cp is None, reason='no gpu'))
