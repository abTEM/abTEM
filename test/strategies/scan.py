import hypothesis.strategies as st
from hypothesis.extra import numpy as numpy_st

from abtem.scan import CustomScan, LineScan, GridScan
from . import core as core_strats
from .core import sensible_floats


@st.composite
def custom_scan(draw):
    n = draw(st.integers(1, 10))
    positions = numpy_st.arrays(dtype=float,
                                shape=(n, 2),
                                elements=sensible_floats(min_value=0, max_value=10.))
    positions = draw(positions)
    return CustomScan(positions)


@st.composite
def line_scan(draw):
    endpoint = draw(st.booleans())
    min_gpts = 2 if endpoint else 1
    gpts = draw(st.integers(min_value=min_gpts, max_value=10))

    start = draw(st.tuples(st.floats(min_value=-5, max_value=5), st.floats(min_value=-5, max_value=5)))
    shift = draw(st.tuples(st.floats(min_value=.1, max_value=5), st.floats(min_value=.1, max_value=5)))
    end = (start[0] + shift[0], start[1] + shift[1])

    return LineScan(start=start, end=end, gpts=gpts, endpoint=endpoint)


@st.composite
def grid_scan(draw):
    endpoint = draw(st.booleans())
    min_gpts = 2 if endpoint else 1

    gpts = draw(core_strats.gpts(min_value=min_gpts, max_value=10, allow_none=False))
    start = draw(st.tuples(st.floats(min_value=-5, max_value=5), st.floats(min_value=-5, max_value=5)))
    shift = draw(st.tuples(st.floats(min_value=.1, max_value=5), st.floats(min_value=.1, max_value=5)))
    end = (start[0] + shift[0], start[1] + shift[1])

    return GridScan(start=start, end=end, gpts=gpts, endpoint=endpoint)


scan_strategies = [custom_scan, line_scan, grid_scan]
