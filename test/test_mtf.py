"""Tests for abtem/mtf.py"""

import numpy as np
import pytest

from abtem.mtf import MTF, default_mtf_func


class TestDefaultMtfFunc:
    def test_at_zero_frequency(self):
        # k=0 → (c0-c1)/(1+0) + c1 = c0
        c0, c1, c2, c3 = 1.0, 0.2, 0.5, 2.0
        result = default_mtf_func(np.array([0.0]), c0, c1, c2, c3)
        assert np.isclose(result[0], c0)

    def test_approaches_c1_at_high_frequency(self):
        c0, c1, c2, c3 = 1.0, 0.1, 0.5, 2.0
        result = default_mtf_func(np.array([1e6]), c0, c1, c2, c3)
        assert np.isclose(result[0], c1, atol=1e-4)

    def test_monotonically_decreasing(self):
        k = np.linspace(0, 2, 50)
        c0, c1, c2, c3 = 1.0, 0.1, 0.5, 2.0
        values = default_mtf_func(k, c0, c1, c2, c3)
        assert np.all(np.diff(values) <= 0)

    def test_array_output_shape(self):
        k = np.ones((4, 4))
        result = default_mtf_func(k, 1.0, 0.1, 0.5, 2.0)
        assert result.shape == (4, 4)

    def test_values_between_c1_and_c0(self):
        c0, c1, c2, c3 = 0.9, 0.1, 0.5, 2.0
        k = np.linspace(0, 10, 100)
        result = default_mtf_func(k, c0, c1, c2, c3)
        assert np.all(result >= c1 - 1e-10)
        assert np.all(result <= c0 + 1e-10)


class TestMTF:
    """MTF.__call__ uses the legacy `calibrations` attribute (not `axes_metadata`).
    We test it via a minimal mock that matches the interface the code expects."""

    def _make_mock_measurement(self, shape=(32, 32), sampling=0.1):
        from unittest.mock import MagicMock

        arr = np.ones(shape)
        cal = MagicMock()
        cal.units = "Å"
        cal.sampling = sampling

        m = MagicMock()
        m.array = arr.copy()
        m.calibrations = [cal, cal]
        # copy() returns a new mock with the same array so MTF can write into it
        copy = MagicMock()
        copy.array = arr.copy()
        copy.calibrations = [cal, cal]
        m.copy.return_value = copy
        return m, copy

    def test_default_func_used_when_none(self):
        mtf = MTF(c0=1.0, c1=0.1, c2=0.5, c3=2.0)
        assert mtf.f is default_mtf_func

    def test_custom_func_stored(self):
        custom = lambda k, a: a * k
        mtf = MTF(func=custom, a=3.0)
        assert mtf.f is custom
        assert mtf.params == {"a": 3.0}

    def test_params_stored(self):
        mtf = MTF(c0=1.0, c1=0.2, c2=0.5, c3=2.0)
        assert mtf.params == {"c0": 1.0, "c1": 0.2, "c2": 0.5, "c3": 2.0}

    def test_call_returns_copy(self):
        m, copy = self._make_mock_measurement()
        mtf = MTF(c0=1.0, c1=0.0, c2=0.5, c3=2.0)
        result = mtf(m)
        assert result is copy

    def test_call_writes_finite_values(self):
        m, copy = self._make_mock_measurement()
        mtf = MTF(c0=1.0, c1=0.0, c2=0.5, c3=2.0)
        mtf(m)
        assert np.isfinite(copy.array).all()

    def test_identity_mtf_output_is_finite(self):
        """MTF=1 everywhere (c0=c1=1) should produce finite output."""
        arr = np.random.default_rng(0).random((32, 32))
        m, copy = self._make_mock_measurement()
        copy.array = arr.copy()
        mtf = MTF(c0=1.0, c1=1.0, c2=0.5, c3=2.0)
        mtf(m)
        assert np.isfinite(copy.array).all()
