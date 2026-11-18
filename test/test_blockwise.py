"""Tests of abtem.core.blockwise: kernels mapped over lazy arrays must give the eager
result, whatever the chunking of the ensemble and base axes."""

import functools

import dask
import dask.array as da
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from utils import gpu

from abtem.core.backend import get_array_module
from abtem.core.blockwise import (
    array_namespace,
    map_kernel,
    map_overlapping_kernel,
    merge_base_chunks,
)

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


def _kernel_with_device_keyword(a, device="not passed"):
    # A kernel keyword named `device` (as calculate_structure_factors has) must
    # reach the kernel, lazy or eager: map_kernel's own device argument is `out_device`.
    assert device == "cpu"
    return a


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_map_kernel_forwards_a_kernel_keyword_named_device(array, lazy):
    source = da.from_array(array, chunks=(1, 2, 16, 12)) if lazy else array

    result = map_kernel(source, _kernel_with_device_keyword, base_ndim=2, device="cpu")

    np.testing.assert_array_equal(np.asarray(result), array)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_map_kernel_out_device_sets_the_meta_not_the_kernel_keywords(array, device):
    xp = get_array_module(device)
    result = map_kernel(
        da.from_array(array, chunks=(1, 2, 16, 12)),
        lambda a: xp.asarray(a),
        base_ndim=2,
        out_device=device,
    )

    assert get_array_module(result._meta) is xp
    assert get_array_module(result.compute()) is xp


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_array_namespace(device, lazy):
    xp = get_array_module(device)
    array = xp.zeros((2, 3))
    if lazy:
        array = da.from_array(array, chunks=(1, 3))

    namespace = array_namespace(array)

    assert namespace is (da if lazy else xp)
    # a function of the namespace must accept the array it was chosen for
    tiled = namespace.tile(array, (2, 1))
    assert tiled.shape == (4, 3)
    assert isinstance(tiled, da.Array) == lazy
    assert get_array_module(tiled.compute() if lazy else tiled) is xp


@pytest.mark.parametrize("base_ndim", [0, 1, 4])
def test_map_kernel_base_ndim_extremes(array, base_ndim):
    # no base axes: the kernel is applied element-wise; all base axes: no ensemble
    chunks = (1, 2, 8, 6)
    eager = map_kernel(array, _scale, base_ndim=base_ndim, factor=2.0)
    lazy = map_kernel(
        da.from_array(array, chunks=chunks), _scale, base_ndim=base_ndim, factor=2.0
    )
    _assert_lazy_matches_eager(eager, lazy)


def test_map_kernel_with_an_empty_ensemble(array):
    empty = array[:0]
    lazy = map_kernel(
        da.from_array(empty, chunks=(1, 2, 8, 6)),
        _total,
        base_ndim=2,
        base_shape_out=(),
    )
    assert lazy.compute().shape == (0, 4)


@pytest.mark.parametrize(
    "name", ["axis", "keepdims", "name", "token", "vectorize", "concatenate"]
)
def test_map_kernel_rejects_keywords_that_dask_takes(array, name):
    # Dask consumes these for a lazy array only (or silently drops them), so a kernel
    # keyword of that name would give different lazy and eager results.
    with pytest.raises(ValueError, match="reserved"):
        map_kernel(array, _total, base_ndim=2, base_shape_out=(), **{name: 1})
    with pytest.raises(ValueError, match="reserved"):
        map_kernel(
            da.from_array(array, chunks=(1, 2, 8, 6)),
            _total,
            base_ndim=2,
            base_shape_out=(),
            **{name: 1},
        )


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize(
    "member, message",
    [
        (np.ones(4), "leading axes"),  # fewer axes than the ensemble axes
        (np.ones((3, 5, 3, 3)), "leading axes"),  # wrong length on an ensemble axis
        ([1.0, 2.0, 3.0], "NumPy or CuPy array"),
        (da.ones((3, 4, 3, 3)), "NumPy or CuPy array"),
    ],
    ids=["too few axes", "wrong length", "list", "dask array"],
)
def test_map_kernel_validates_member_args(array, lazy, member, message):
    source = da.from_array(array, chunks=(1, 2, 8, 6)) if lazy else array
    with pytest.raises((ValueError, TypeError), match=message):
        map_kernel(
            source,
            lambda a, m: a,
            member,
            base_ndim=2,
        )


def test_merge_base_chunks_keeps_unknown_ensemble_chunks(array):
    lazy = da.from_array(array, chunks=(1, 2, 8, 6))
    lazy = lazy[lazy[:, 0, 0, 0] > -1]
    assert np.isnan(lazy.chunks[0]).all()

    merged = merge_base_chunks(lazy, base_ndim=2)

    assert merged.chunks[2:] == ((16,), (12,))
    assert np.isnan(merged.chunks[0]).all()  # kept, not re-chosen
    np.testing.assert_array_equal(merged.compute(), array)


def test_merge_base_chunks_respects_the_dask_chunk_size(array):
    # a block holds the whole base array of as many members as fit array.chunk-size,
    # so splitting the base axes can never produce an unbounded chunk
    big = da.zeros((64, 64, 128, 128), chunks=(1, 1, 64, 64), dtype=np.float32)

    with dask.config.set({"array.chunk-size": "16MiB"}):
        merged = merge_base_chunks(big, base_ndim=2)

    assert merged.chunks[2:] == ((128,), (128,))
    block_bytes = np.prod(merged.chunksize) * merged.dtype.itemsize
    assert block_bytes <= 16 * 2**20


@pytest.mark.parametrize("boundary", ["periodic", "reflect", "constant"])
def test_map_overlapping_kernel_complex(array, boundary):
    # the declared meta must keep the imaginary part
    values = (array + 1j * array[::-1]).astype(np.complex64)
    kernel = functools.partial(
        gaussian_filter,
        sigma=(0, 0, 1.5, 1.5),
        mode="wrap" if boundary == "periodic" else boundary,
    )
    eager = map_overlapping_kernel(values, kernel, {}, boundary=boundary)
    lazy = map_overlapping_kernel(
        da.from_array(values, chunks=(1, 2, 8, 6)),
        kernel,
        depth={2: 6, 3: 6},
        boundary=boundary,
        dtype=np.complex64,
    )

    assert np.iscomplexobj(lazy.compute())
    _assert_lazy_matches_eager(eager, lazy)


@pytest.mark.parametrize(
    "depth", [8, 12, 40], ids=["over a chunk", "over an axis", "far over"]
)
def test_map_overlapping_kernel_depth_larger_than_the_chunks(array, depth):
    # The filter radius is 8 pixels. Dask merges chunks that are smaller than the
    # overlap; an overlap larger than the axis is clipped to it. Either way the
    # kernel's own edge handling stays exact.
    kernel = functools.partial(gaussian_filter, sigma=(0, 0, 2.0, 2.0), mode="wrap")
    eager = kernel(array)
    lazy = map_overlapping_kernel(
        da.from_array(array, chunks=(1, 2, 4, 3)),
        kernel,
        depth={2: depth, 3: depth},
        boundary="periodic",
    )
    _assert_lazy_matches_eager(eager, lazy)
