import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from strategies import measurements as measurements_st

all_measurements = {'images': measurements_st.images,
                    'diffraction_patterns': measurements_st.diffraction_patterns,
                    'line_profiles': measurements_st.line_profiles,
                    'polar_measurements': measurements_st.polar_measurements
                    }


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('measurement', list(all_measurements.keys()))
def test_hyperspy(data, measurement, lazy):
    measurement = data.draw(all_measurements[measurement](lazy=lazy))
    hyperspy_signal = measurement.to_hyperspy()
