import sys
import warnings

import hypothesis.strategies as st
import pytest
import strategies as abtem_st
from hypothesis import given
from utils import gpu

try:
    import hyperspy
except ImportError:
    hyperspy = None


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
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
