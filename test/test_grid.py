import random

import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import assume, given
from utils import ensure_is_tuple

from abtem.core.grid import Grid, GridUndefinedError


def grid_data(allow_none=False, allow_overdefined=True):
    data = {
        "gpts": abtem_st.gpts(allow_none=allow_none),
        "sampling": abtem_st.sampling(allow_none=allow_none),
        "extent": abtem_st.extent(allow_none=allow_none),
    }

    if not allow_overdefined:
        keys = [key for key in random.sample(data.keys(), 2)]
        data = {key: data[key] for key in keys}

    return st.fixed_dictionaries(data)


def unpack_grid_data(grid_data):
    return grid_data["gpts"], grid_data["extent"], grid_data["sampling"]


def is_grid_data_defining(extent, gpts, sampling):
    return sum([1 if grid_prop else 0 for grid_prop in (extent, gpts, sampling)]) > 1


def check_grid_consistent(extent, gpts, sampling):
    if is_grid_data_defining(extent, gpts, sampling):
        assert np.allclose(sampling, np.array(extent) / np.array(gpts))


@given(grid_data=grid_data())
def test_create_grid(grid_data):
    grid = Grid(**grid_data)

    gpts, extent, sampling = unpack_grid_data(grid_data)

    if gpts is not None:
        assert grid.gpts == ensure_is_tuple(gpts, 2)

    if extent is not None:
        assert grid.extent == ensure_is_tuple(extent, 2)

    if sampling is not None:
        if extent is None:
            assert np.allclose(grid.sampling, ensure_is_tuple(sampling, 2))
        elif gpts is None:
            adjusted_sampling = extent / np.ceil(np.array(extent) / np.array(sampling))
            assert np.allclose(grid.sampling, adjusted_sampling)


@given(grid_data=grid_data())
def test_grid_raises(grid_data):
    grid = Grid(**grid_data)

    if is_grid_data_defining(*unpack_grid_data(grid_data)):
        grid.check_is_defined()
    else:
        with pytest.raises(GridUndefinedError):
            grid.check_is_defined()


@given(grid_data=grid_data())
def test_grid_consistent(grid_data):
    grid = Grid(**grid_data)
    assume(is_grid_data_defining(*unpack_grid_data(grid_data)))
    check_grid_consistent(grid.extent, grid.gpts, grid.sampling)


@given(grid_data=grid_data(), new_gpts=abtem_st.gpts())
def test_gpts_change(grid_data, new_gpts):
    grid = Grid(**grid_data)

    grid.gpts = new_gpts
    assert (
        grid.gpts == ensure_is_tuple(new_gpts, 2) if new_gpts is not None else new_gpts
    )
    check_grid_consistent(grid.extent, grid.gpts, grid.sampling)


@given(grid_data=grid_data(), new_extent=abtem_st.extent())
def test_gpts_change(grid_data, new_extent):
    grid = Grid(**grid_data)

    grid.extent = new_extent
    assert (
        grid.extent == ensure_is_tuple(new_extent, 2)
        if new_extent is not None
        else new_extent
    )
    check_grid_consistent(grid.extent, grid.gpts, grid.sampling)


@given(grid_data=grid_data(), new_sampling=abtem_st.sampling())
def test_sampling_change(grid_data, new_sampling):
    grid = Grid(**grid_data)

    grid.sampling = new_sampling
    if grid.extent is None:
        assert (
            grid.sampling == ensure_is_tuple(new_sampling, 2)
            if new_sampling is not None
            else new_sampling
        )
    else:
        adjusted_sampling = grid.extent / np.ceil(
            np.array(grid.extent) / np.array(new_sampling)
        )
        assert np.allclose(grid.sampling, adjusted_sampling)

    check_grid_consistent(grid.extent, grid.gpts, grid.sampling)
