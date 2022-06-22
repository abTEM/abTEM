import numpy as np
import pytest
from hypothesis import given, settings, assume, reproduce_failure
from hypothesis import strategies as st

from abtem.core.ensemble import concatenate_array_blocks
from abtem.core.distributions import GaussianDistribution
from abtem.waves.scan import LineScan, SourceOffset
from strategies.core import sensible_floats
from strategies import scan as scan_st


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


@given(data=st.data(), chunks=st.integers(min_value=1, max_value=10))
@pytest.mark.parametrize('scan', [scan_st.grid_scan(), scan_st.line_scan(), scan_st.custom_scan()],
                         ids=['custom_scan', 'line_scan', 'grid_scan'])
def test_scan_ensemble(data, scan, chunks):
    scan = data.draw(scan)
    blocks = scan.ensemble_blocks(chunks).compute()

    for i in np.ndindex(blocks.shape):
        blocks[i] = blocks[i].get_positions()

    assert np.allclose(concatenate_array_blocks(blocks), scan.get_positions())


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