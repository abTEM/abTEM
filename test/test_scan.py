import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from abtem.waves.scan import LineScan
from strategies.core import sensible_floats


@given(position=st.tuples(sensible_floats(min_value=-100, max_value=100),
                          sensible_floats(min_value=-100, max_value=100)),
       extent=sensible_floats(min_value=.1, max_value=100),
       angle=sensible_floats(min_value=0, max_value=360))
def test_linescan_at_position(position, extent, angle):
    linescan = LineScan.at_position(position, extent=extent, angle=angle)
    vector = np.array(linescan.end) - np.array(linescan.start)

    assert np.allclose(extent, np.linalg.norm(vector))
    assert np.allclose(angle, np.rad2deg(np.arctan2(vector[1], vector[0])) % 360)
    assert np.allclose(position, (np.array(linescan.start) + np.array(linescan.end)) / 2)
