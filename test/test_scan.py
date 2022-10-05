import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import strategies as abtem_st
from abtem.scan import LineScan


@given(position=st.tuples(abtem_st.sensible_floats(min_value=-100, max_value=100),
                          abtem_st.sensible_floats(min_value=-100, max_value=100)),
       extent=abtem_st.sensible_floats(min_value=.1, max_value=100),
       angle=abtem_st.sensible_floats(min_value=0, max_value=360))
def test_linescan_at_position(position, extent, angle):
    linescan = LineScan.at_position(position, extent=extent, angle=angle)
    vector = np.array(linescan.end) - np.array(linescan.start)

    assert np.allclose(extent, np.linalg.norm(vector))
    assert np.allclose(angle, np.rad2deg(np.arctan2(vector[1], vector[0])) % 360)
    assert np.allclose(position, (np.array(linescan.start) + np.array(linescan.end)) / 2)

# def test_source_offset():
#     distribution = GaussianDistribution(4, num_samples=4, dimension=2)
#
#     s = SourceOffset(distribution)
#
#     blocks = s._ensemble_blockwise(1).compute()
#
#     for i in np.ndindex(blocks.shape):
#         blocks[i] = blocks[i].values
#
#     assert np.allclose(concatenate_blocks(blocks), s.get_positions())
