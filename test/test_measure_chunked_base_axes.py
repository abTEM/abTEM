"""Lazy measurements whose base axes are split into several chunks.

PR 679 moved the measurement methods onto `abtem.core.blockwise.map_kernel`, which
merges split base axes before calling the kernel. The existing lazy tests chunk only
the ensemble axes, so these cover the merge itself: values, dask `meta` and the
unknown-chunk case.
"""

import dask.array as da
import numpy as np
import pytest
from utils import gpu

from abtem.core.axes import ScanAxis, UnknownAxis
from abtem.core.backend import asnumpy, get_array_module
from abtem.measurements import (
    DiffractionPatterns,
    Images,
    RealSpaceLineProfiles,
)

DP_KWARGS = dict(
    sampling=0.1,
    ensemble_axes_metadata=[ScanAxis(sampling=0.2, _main=True)] * 2,
    metadata={"energy": 100e3},
)


def _diffraction_patterns(array):
    return DiffractionPatterns(array, **DP_KWARGS)


def _images(array):
    return Images(
        array, sampling=0.1, ensemble_axes_metadata=[UnknownAxis()] * (array.ndim - 2)
    )


def _line_profiles(array):
    return RealSpaceLineProfiles(
        array, sampling=0.1, ensemble_axes_metadata=[UnknownAxis()] * (array.ndim - 1)
    )


CASES = {
    "center_of_mass": (
        _diffraction_patterns,
        (3, 4, 16, 12),
        lambda m: m.center_of_mass(),
    ),
    "integrate_radial": (
        _diffraction_patterns,
        (3, 4, 16, 12),
        lambda m: m.integrate_radial(0, 10),
    ),
    "azimuthal_average": (
        _diffraction_patterns,
        (3, 4, 16, 12),
        lambda m: m.azimuthal_average(),
    ),
    "interpolate_line": (
        _images,
        (3, 4, 16, 12),
        lambda m: m.interpolate_line((0, 0), (1.0, 0.8), gpts=13),
    ),
    "width": (
        _line_profiles,
        (3, 4, 32),
        lambda m: m.width(),
    ),
}


def _array(result):
    # `width` returns a bare array, the other methods a measurement
    return getattr(result, "array", result)


#  (ensemble chunks, base chunks) of the lazy input
CHUNKINGS = {
    "base-split": ((3, 4), (8, 6)),
    "both-split": ((1, 2), (8, 6)),
}


def _chunks(shape, chunking):
    ensemble, base = CHUNKINGS[chunking]
    base = base[-(len(shape) - 2) :]
    return ensemble + base


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("chunking", CHUNKINGS)
@pytest.mark.parametrize("case", CASES)
def test_lazy_with_split_base_axes_matches_eager(case, chunking, device):
    build, shape, operation = CASES[case]
    array = np.random.default_rng(1).random(shape).astype(np.float32)

    eager = operation(build(array).copy_to_device(device))
    lazy = operation(
        build(da.from_array(array, chunks=_chunks(shape, chunking))).copy_to_device(
            device
        )
    )

    lazy_array = _array(lazy)
    assert isinstance(lazy_array, da.Array)
    assert get_array_module(lazy_array._meta) is get_array_module(device)
    computed = _array(lazy.compute())
    assert get_array_module(computed) is get_array_module(device)
    assert computed.shape == _array(eager).shape
    np.testing.assert_allclose(
        asnumpy(computed), asnumpy(_array(eager)), rtol=1e-5, atol=1e-6
    )


def _with_unknown_ensemble_chunks(array, chunks):
    # boolean indexing leaves the length of the ensemble axis unknown (nan chunks)
    lazy = da.from_array(array, chunks=chunks)
    lazy = lazy[lazy[:, 0, 0] > -1]
    assert np.isnan(lazy.chunks[0]).all()
    return lazy


@pytest.mark.parametrize(
    "operation",
    [lambda m: m.integrate_radial(0, 10), lambda m: m.center_of_mass()],
    ids=["integrate_radial", "center_of_mass"],
)
def test_lazy_with_unknown_ensemble_chunks_and_split_base_axes(operation):
    # Regression: merging the base chunks re-chose the ensemble chunks with 'auto',
    # which dask refuses when their length is unknown. dev handled these methods.
    array = np.random.default_rng(2).random((4, 16, 16)).astype(np.float32)
    kwargs = dict(
        sampling=0.1,
        ensemble_axes_metadata=[UnknownAxis()],
        metadata={"energy": 100e3},
    )
    lazy = DiffractionPatterns(
        _with_unknown_ensemble_chunks(array, (1, 8, 8)), **kwargs
    )

    result = operation(lazy).compute().array
    expected = operation(DiffractionPatterns(array, **kwargs)).array

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
