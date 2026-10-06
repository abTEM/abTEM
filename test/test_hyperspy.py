import sys
import warnings

import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import given
from utils import devices, lazy_params

from abtem.core.backend import asnumpy

try:
    import hyperspy
except ImportError:
    hyperspy = None


@given(data=st.data())
@lazy_params
@devices
@pytest.mark.parametrize(
    "measurement",
    [
        abtem_st.images,
        abtem_st.line_profiles,
        abtem_st.diffraction_patterns,
        abtem_st.polar_measurements,
        abtem_st.potential_array,
        abtem_st.waves,
    ],
)
@pytest.mark.skipif("hyperspy" not in sys.modules, reason="requires hyperspy")
def test_hyperspy(data, measurement, lazy, device):
    measurement = data.draw(measurement(lazy=lazy, device=device))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        hyperspy_signal = measurement.to_hyperspy()
        expected = measurement.to_cpu().to_hyperspy().data

    signal_data = hyperspy_signal.data
    if lazy:
        signal_data, expected = signal_data.compute(), expected.compute()
    if device == "mps":
        # HyperSpy cannot hold torch backend arrays.
        assert isinstance(signal_data, np.ndarray)
    np.testing.assert_array_equal(asnumpy(signal_data), expected)
