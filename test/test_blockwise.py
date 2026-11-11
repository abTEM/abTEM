"""Tests of abtem.core.blockwise: kernels mapped over lazy arrays must give the eager
result, whatever the chunking of the ensemble and base axes."""

import functools

import dask.array as da
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from abtem.core.blockwise import map_kernel, map_overlapping_kernel, merge_base_chunks

CHUNKINGS = [
    (3, 4, 16, 12),  # one chunk
    (1, 2, 16, 12),  # chunked ensemble axes
    (3, 4, 8, 6),  # chunked base axes
    (1, 2, 8, 6),  # both
]


@pytest.fixture
def array():
    return np.random.default_rng(0).random((3, 4, 16, 12)).astype(np.float32)


def _assert_lazy_matches_eager(eager, lazy):
    assert isinstance(lazy, da.Array)
    assert lazy.shape == eager.shape
    assert lazy.dtype == eager.dtype
    computed = lazy.compute()
    assert computed.dtype == eager.dtype
    np.testing.assert_allclose(computed, eager, rtol=1e-6)


def _scale(a, factor):
    return a * factor


def _total(a):
    return a.sum((-2, -1))


def _crop_and_repeat(a, n):
    return a[..., :n, :, None].repeat(3, -1)


def _to_complex(a):
    return (a + 1j).astype(np.complex64)


def _with_members(a, matrices, energies):
    # the member arguments keep their own trailing axes
    assert matrices.shape[-2:] == (3, 3)
    assert energies.shape[-2:] == (1, 1)
    return a.sum((-2, -1))[..., None] * matrices[..., 0, :] + energies[..., 0, :]


@pytest.mark.parametrize("chunks", CHUNKINGS)
@pytest.mark.parametrize(
    "kernel, kwargs, options",
    [
        (_scale, {"factor": 2.0}, {}),
        (_total, {}, {"base_shape_out": ()}),
        (_crop_and_repeat, {"n": 5}, {"base_shape_out": (5, 12, 3)}),
        (_to_complex, {}, {"dtype": np.complex64}),
    ],
    ids=["same shape", "drop base axes", "new base axes", "new dtype"],
)
def test_map_kernel_lazy_matches_eager(array, chunks, kernel, kwargs, options):
    eager = map_kernel(array, kernel, base_ndim=2, **kwargs, **options)
    np.testing.assert_array_equal(eager, kernel(array, **kwargs))

    lazy = map_kernel(
        da.from_array(array, chunks=chunks), kernel, base_ndim=2, **kwargs, **options
    )
    _assert_lazy_matches_eager(eager, lazy)


@pytest.mark.parametrize("chunks", CHUNKINGS)
def test_map_kernel_member_args_broadcast(array, chunks):
    rng = np.random.default_rng(1)
    # one matrix per member of the second ensemble axis, one energy per member of
    # the first: each broadcasts along the other axis
    matrices = rng.random((1, 4, 3, 3))
    energies = rng.random((3, 1, 1, 1))
    options = dict(base_ndim=2, base_shape_out=(3,), dtype=np.float64)

    eager = map_kernel(array, _with_members, matrices, energies, **options)
    lazy = map_kernel(
        da.from_array(array, chunks=chunks),
        _with_members,
        matrices,
        energies,
        **options,
    )
    _assert_lazy_matches_eager(eager, lazy)


def test_map_kernel_without_ensemble_axes(array):
    image = array[0, 0]
    lazy = map_kernel(
        da.from_array(image, chunks=(8, 6)), _total, base_ndim=2, base_shape_out=()
    )
    _assert_lazy_matches_eager(_total(image), lazy)


def test_map_kernel_bare_array_needs_base_ndim(array):
    with pytest.raises(ValueError, match="base_ndim"):
        map_kernel(array, _total)


def test_map_kernel_array_object():
    from abtem.core.axes import OrdinalAxis
    from abtem.measurements import Images

    images = Images(
        np.random.default_rng(2).random((5, 16, 12)).astype(np.float32),
        sampling=0.1,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(5)))],
    )
    lazy_images = images.ensure_lazy(chunks=(2, 8, 6))

    eager = map_kernel(images, _total, base_shape_out=())
    lazy = map_kernel(lazy_images, _total, base_shape_out=())
    _assert_lazy_matches_eager(eager, lazy)


def test_merge_base_chunks():
    array = da.zeros((6, 16, 12), chunks=(2, 8, 6))
    merged = merge_base_chunks(array, base_ndim=2)
    assert merged.chunks[1:] == ((16,), (12,))

    unchanged = da.zeros((6, 16, 12), chunks=(2, 16, 12))
    assert merge_base_chunks(unchanged, base_ndim=2) is unchanged


@pytest.mark.parametrize("chunks", CHUNKINGS)
@pytest.mark.parametrize(
    "boundary, mode",
    [("periodic", "wrap"), ("reflect", "reflect"), ("constant", "constant")],
)
def test_map_overlapping_kernel_lazy_matches_eager(array, chunks, boundary, mode):
    # the kernel fills its own edges with cval; the overlap needs it too
    kernel = functools.partial(
        gaussian_filter, sigma=(0, 0, 1.5, 1.5), mode=mode, cval=0.3
    )
    eager = kernel(array)

    lazy = map_overlapping_kernel(
        da.from_array(array, chunks=chunks),
        kernel,
        depth={2: 6, 3: 6},
        boundary=boundary,
        cval=0.3,
    )
    _assert_lazy_matches_eager(eager, lazy)


def test_map_overlapping_kernel_rejects_unknown_boundary(array):
    with pytest.raises(ValueError, match="boundary"):
        map_overlapping_kernel(
            da.from_array(array), gaussian_filter, {2: 1}, boundary="wrap", sigma=1
        )
