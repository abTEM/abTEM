import sys
import warnings

import hypothesis.strategies as st
import pytest
import strategies as abtem_st
from hypothesis import given
from utils import devices, lazy_params

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
def test_hyperspy(request, data, measurement, lazy, device):
    if device == "mps" and not lazy:
        # HyperSpy calls np.atleast_1d on the data, which the torch backend does not
        # implement for its arrays.
        request.applymarker(
            pytest.mark.xfail(
                raises=TypeError,
                strict=True,
                reason="np.atleast_1d is not implemented for torch backend arrays",
            )
        )
    measurement = data.draw(measurement(lazy=lazy, device=device))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        hyperspy_signal = measurement.to_hyperspy()
