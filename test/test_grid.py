import random

import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import assume, given
from utils import ensure_is_tuple

from abtem.core import config
from abtem.core.grid import Grid, GridUndefinedError, spatial_frequencies


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
            adjusted_sampling = grid.extent / np.ceil(
                np.array(grid.extent) / np.array(new_sampling)
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


def _hex_cell(a, angle_deg=60.0):
    th = np.deg2rad(angle_deg)
    return np.array([[a, 0.0], [a * np.cos(th), a * np.sin(th)]])


def test_grid_orthogonal_by_default():
    grid = Grid(extent=(20.0, 30.0), gpts=(64, 96))
    assert grid.cell is None
    assert grid.is_orthogonal


def test_k_squared_orthogonal_matches_spatial_frequencies():
    # the orthogonal path must be identical to the existing spatial_frequencies code
    grid = Grid(extent=(38.4, 51.2), gpts=(192, 256))
    kx, ky = spatial_frequencies(grid.gpts, grid.sampling)
    ref = kx[:, None] ** 2 + ky[None] ** 2
    assert np.array_equal(grid.k_squared(), ref)


def test_skew_grid_metric_and_components():
    a = 30.0
    cell = _hex_cell(a)
    grid = Grid(extent=tuple(np.linalg.norm(cell, axis=1)), gpts=(128, 128), cell=cell)

    assert not grid.is_orthogonal

    # reciprocal metric is symmetric with a non-zero cross term for a skew cell
    M = grid.reciprocal_metric
    assert np.allclose(M, M.T)
    assert abs(M[0, 1]) > 1e-9

    # |g|^2 from the metric equals gx^2 + gy^2 from the components
    gx, gy = grid.k_components()
    assert np.allclose(grid.k_squared(), gx**2 + gy**2, atol=1e-5)


def test_skew_grid_reduces_to_orthogonal_for_diagonal_cell():
    extent = (24.0, 32.0)
    gpts = (96, 128)
    ortho = Grid(extent=extent, gpts=gpts)
    diag = Grid(extent=extent, gpts=gpts, cell=np.diag(extent))
    assert np.allclose(diag.k_squared(), ortho.k_squared())


def test_cell_validation():
    # inconsistent row lengths vs extent are rejected
    with pytest.raises(ValueError):
        Grid(extent=(10.0, 10.0), gpts=(8, 8), cell=np.array([[20.0, 0.0], [0.0, 10.0]]))
    # wrong shape is rejected
    with pytest.raises(ValueError):
        Grid(extent=(10.0, 10.0), gpts=(8, 8), cell=np.eye(3))
    # a non-2D grid with a cell is rejected
    with pytest.raises(ValueError):
        Grid(extent=(10.0,) * 3, gpts=(8,) * 3, dimensions=3, cell=np.eye(2))


def test_polar_spatial_frequencies():
    from abtem.core.grid import polar_spatial_frequencies

    # orthogonal: bit-exact vs the module function
    ortho = Grid(extent=(20.0, 25.0), gpts=(64, 80))
    k, phi = ortho.polar_spatial_frequencies()
    k0, phi0 = polar_spatial_frequencies(ortho.gpts, ortho.sampling)
    assert np.array_equal(k, k0) and np.array_equal(phi, phi0)

    # skew: physical (k, phi) consistent with the metric components
    cell = _hex_cell(20.0, angle_deg=70.0)
    skew = Grid(extent=tuple(np.linalg.norm(cell, axis=1)), gpts=(80, 80), cell=cell)
    k, phi = skew.polar_spatial_frequencies()
    gx, gy = skew.k_components()
    assert np.allclose(k**2, skew.k_squared())
    assert np.allclose(phi, np.arctan2(gy, gx))


def test_skew_grid_copy_and_equality():
    cell = _hex_cell(25.0)
    grid = Grid(extent=tuple(np.linalg.norm(cell, axis=1)), gpts=(64, 64), cell=cell)
    assert grid.copy() == grid
    assert np.allclose(grid.copy().cell, grid.cell)
    ortho = Grid(extent=grid.extent, gpts=grid.gpts)
    assert not (grid == ortho)


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


def test_fast_fft_rounding_composes_with_a_skewed_cell():
    # fft_grid / round-to-fast-fft only ever touches gpts; a skewed cell's row
    # lengths track extent (via the extent setter), so rounding gpts on a
    # cell-bearing grid must leave the cell's row lengths matching extent.
    cell = _hex_cell(10.0)
    with config.set({"grid.round-to-fast-fft": True}):
        grid = Grid(extent=tuple(np.linalg.norm(cell, axis=1)), sampling=0.03, cell=cell)

    assert all(d <= 0.03 for d in grid.sampling)
    lengths = np.linalg.norm(np.asarray(grid.cell), axis=1)
    assert np.allclose(lengths, grid.extent, rtol=1e-6)


def test_round_to_fast_fft_method_preserves_skewed_cell():
    cell = _hex_cell(10.0)
    extent = tuple(np.linalg.norm(cell, axis=1))
    grid = Grid(extent=extent, sampling=0.03, cell=cell)
    assert grid.gpts == (334, 334)

    gpts = grid.round_to_fast_fft()

    assert gpts == (336, 336)
    lengths = np.linalg.norm(np.asarray(grid.cell), axis=1)
    assert np.allclose(lengths, grid.extent, rtol=1e-6)
