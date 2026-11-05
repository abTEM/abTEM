import random

import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import assume, example, given
from utils import ensure_is_tuple, exactly_dividing_lengths

from abtem.core import config
from abtem.core.grid import Grid, GridUndefinedError, adjusted_gpts
from abtem.core.utils import safe_ceiling_int


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
def test_extent_change(grid_data, new_extent):
    grid = Grid(**grid_data)

    grid.extent = new_extent
    assert (
        grid.extent == ensure_is_tuple(new_extent, 2)
        if new_extent is not None
        else new_extent
    )
    check_grid_consistent(grid.extent, grid.gpts, grid.sampling)


@given(grid_data=grid_data(), new_sampling=abtem_st.sampling())
# 1.8 / 0.06 is 30.000000000000004 in floats; the grid has 30 points.
@example(grid_data={"gpts": 32, "extent": 1.8}, new_sampling=0.06)
def test_sampling_change(grid_data, new_sampling):
    # Pin the option: these assert the behaviour of the default mode,
    # which a user-level override of the config would otherwise change.
    with config.set({"grid.round-to-fast-fft": "auto"}):
        grid = Grid(**grid_data)

        grid.sampling = new_sampling
        if grid.extent is None:
            assert (
                grid.sampling == ensure_is_tuple(new_sampling, 2)
                if new_sampling is not None
                else new_sampling
            )
        else:
            adjusted_sampling = tuple(
                e / safe_ceiling_int(e / d)
                for e, d in zip(grid.extent, ensure_is_tuple(new_sampling, 2))
            )
            assert np.allclose(grid.sampling, adjusted_sampling)

        check_grid_consistent(grid.extent, grid.gpts, grid.sampling)


def test_fast_fft_rounding_off_by_default():
    # Pin the option: these assert the behaviour of the default mode,
    # which a user-level override of the config would otherwise change.
    with config.set({"grid.round-to-fast-fft": "auto"}):
        # 10 / 0.03 -> ceil = 334 = 2 * 167; 167 is prime, so 334 is not a fast
        # FFT length and must be kept exactly as derived when the option is off.
        grid = Grid(extent=10, sampling=0.03)
        assert grid.gpts == (334, 334)


def test_fast_fft_rounding_refines_sampling():
    with config.set({"grid.round-to-fast-fft": True}):
        grid = Grid(extent=10, sampling=0.03)

    assert grid.gpts == (336, 336)  # 336 = 2**4 * 3 * 7
    assert all(d <= 0.03 for d in grid.sampling)
    assert grid.extent == (10.0, 10.0)
    check_grid_consistent(grid.extent, grid.gpts, grid.sampling)


def test_fast_fft_rounding_keeps_fast_gpts():
    with config.set({"grid.round-to-fast-fft": True}):
        grid = Grid(extent=10, sampling=0.1)

    assert grid.gpts == (100, 100)  # 100 = 2**2 * 5**2 is already fast


def test_fast_fft_rounding_never_alters_explicit_gpts():
    with config.set({"grid.round-to-fast-fft": True}):
        grid = Grid(extent=10, gpts=334)

    assert grid.gpts == (334, 334)


def test_fast_fft_rounding_exempts_endpoint_grids():
    with config.set({"grid.round-to-fast-fft": True}):
        grid = Grid(extent=10, sampling=0.03, endpoint=True)

    assert grid.gpts == (335, 335)


def test_fast_fft_rounding_applies_on_sampling_change():
    grid = Grid(extent=10, sampling=0.1)
    with config.set({"grid.round-to-fast-fft": True}):
        grid.sampling = 0.03

    assert grid.gpts == (336, 336)
    check_grid_consistent(grid.extent, grid.gpts, grid.sampling)


def test_round_to_fast_fft_method():
    # Pin the option: these assert the behaviour of the default mode,
    # which a user-level override of the config would otherwise change.
    with config.set({"grid.round-to-fast-fft": "auto"}):
        grid = Grid(extent=10, sampling=0.03)
        assert grid.gpts == (334, 334)

        gpts = grid.round_to_fast_fft()

        assert gpts == (336, 336)
        assert grid.gpts == (336, 336)
        assert grid.extent == (10.0, 10.0)
        check_grid_consistent(grid.extent, grid.gpts, grid.sampling)


def test_round_to_fast_fft_leaves_non_fft_grids_alone():
    # A grid that is never Fourier transformed (a GridScan's probe positions)
    # gains nothing from a fast length, and rounding it would change what is
    # simulated rather than how fast it runs.
    assert Grid(extent=(131.15, 131.15), gpts=(2623, 2623)).round_to_fast_fft() == (
        2625,
        2625,
    )
    scan_grid = Grid(extent=(131.15, 131.15), gpts=(2623, 2623), fft_grid=False)
    assert scan_grid.round_to_fast_fft() == (2623, 2623)
    assert scan_grid.gpts == (2623, 2623)


def test_gpts_of_an_exactly_dividing_sampling():
    # 10.8 / 0.3 is 36.00000000000001 in floats; the grid still has 36 points.
    for extent, sampling, count in exactly_dividing_lengths():
        grid = Grid(extent=extent, sampling=sampling)
        assert grid.gpts == (count, count), (extent, sampling, grid.gpts, count)


@pytest.mark.parametrize("extent", [1.1, 3.7, 5.43, 7.3, 10.8, 13.37])
def test_gpts_round_trip_through_the_sampling(extent):
    for n in range(1, 200):
        grid = Grid(extent=extent, sampling=extent / n)
        assert grid.gpts == (n, n), (extent, n, grid.gpts)


@pytest.mark.parametrize("extent", [36.001, 36.0000002])
def test_gpts_still_round_up_a_genuine_remainder(extent):
    assert Grid(extent=extent, sampling=1.0).gpts == (37, 37)


def test_gpts_of_an_extent_far_below_the_sampling_is_one():
    assert Grid(extent=1e-9, sampling=1.0).gpts == (1, 1)
    assert Grid(extent=1e-9, sampling=1.0, endpoint=False).gpts == (1, 1)


def test_gpts_of_a_zero_extent_axis_with_an_endpoint_is_one():
    grid = Grid(extent=(2.0, 0.0), sampling=0.5, endpoint=True)
    assert grid.gpts == (5, 1)


def test_adjusted_gpts_of_a_target_far_above_the_sampling_is_one():
    sampling, gpts = adjusted_gpts((1.0, 1.0), (1e-9, 1e-9), (1, 1))
    assert gpts == (1, 1)
    assert sampling == pytest.approx((1e-9, 1e-9))


# ── default gpts yield to an explicit grid ───────────────────────────────────
def _default_grid():
    return Grid(
        extent=(8.0, 8.0), gpts=(160, 160), lock_extent=True, gpts_is_default=True
    )


def test_a_grid_with_no_gpts_adopts_default_gpts():
    waves = Grid()
    default = _default_grid()

    waves.match(default)

    assert waves.gpts == (160, 160)
    assert default.gpts == (160, 160)
    assert default.gpts_is_default


@pytest.mark.parametrize(
    "kwargs, expected_gpts",
    [({"gpts": (100, 100)}, (100, 100)), ({"sampling": 0.1}, (80, 80))],
)
def test_explicit_gpts_or_sampling_override_default_gpts(kwargs, expected_gpts):
    waves = Grid(**kwargs)
    default = _default_grid()

    waves.match(default)

    assert waves.gpts == expected_gpts
    assert default.gpts == expected_gpts
    assert not default.gpts_is_default
    assert default.extent == (8.0, 8.0)


def test_locked_gpts_override_default_gpts_instead_of_raising():
    built = Grid(extent=(8.0, 8.0), gpts=(100, 100), lock_extent=True, lock_gpts=True)
    default = _default_grid()

    built.match(default)

    assert default.gpts == (100, 100)


def test_gpts_that_are_not_a_default_still_win_over_explicit_gpts():
    waves = Grid(gpts=(100, 100))
    explicit = Grid(extent=(8.0, 8.0), gpts=(160, 160), lock_extent=True)

    waves.match(explicit)

    assert waves.gpts == (160, 160)
    assert explicit.gpts == (160, 160)


def test_setting_gpts_or_sampling_clears_the_default_flag():
    grid = _default_grid()
    grid.gpts = (100, 100)
    assert not grid.gpts_is_default

    grid = _default_grid()
    grid.sampling = 0.1
    assert not grid.gpts_is_default


def test_whether_gpts_are_a_default_does_not_affect_equality():
    default = _default_grid()
    explicit = Grid(extent=(8.0, 8.0), gpts=(160, 160), lock_extent=True)

    assert default == explicit


# ── auto gpts are kept, but an explicit grid meeting them is warned about ────
def _auto_grid():
    return Grid(
        extent=(8.0, 8.0), gpts=(160, 160), lock_extent=True, gpts_are_auto=True
    )


def test_explicit_gpts_do_not_override_auto_gpts_but_warn():
    waves = Grid(gpts=(100, 100))
    auto = _auto_grid()

    with pytest.warns(UserWarning, match="sampling='auto'"):
        waves.match(auto)

    assert waves.gpts == (160, 160)
    assert auto.gpts == (160, 160)


def test_a_grid_without_explicit_gpts_adopts_auto_gpts_quietly(recwarn):
    waves = Grid()

    waves.match(_auto_grid())

    assert waves.gpts == (160, 160)
    assert len(recwarn) == 0


def test_a_grid_that_adopted_its_gpts_is_not_explicit_when_matched_again(recwarn):
    waves = Grid()
    waves.match(Grid(extent=(8.0, 8.0), gpts=(100, 100), lock_extent=True))

    waves.match(_auto_grid())

    assert waves.gpts == (160, 160)
    assert len(recwarn) == 0


def test_setting_gpts_clears_the_auto_flag_and_the_adoption():
    grid = _auto_grid()
    grid.gpts = (100, 100)
    assert not grid.gpts_are_auto

    waves = Grid()
    waves.match(Grid(extent=(8.0, 8.0), gpts=(100, 100), lock_extent=True))
    waves.gpts = (50, 50)
    assert not waves._gpts_adopted
