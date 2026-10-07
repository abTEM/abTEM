"""A lazy result declares the dtype its blocks return, and the eager result has.

A dask reduction accumulates in the declared dtype, so a declaration below the
blocks loses precision, and one above them makes ``.dtype`` misreport the data.
Every case compares the declared dtype, the dtype of the computed blocks and the
dtype of the eager result of the same call.
"""

import dask.array as da
import numpy as np
import pytest

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.core.fft import fft2, fft2_convolve
from abtem.measurements import DiffractionPatterns, Images
from abtem.potentials.iam import PotentialArray

GRID = (16, 20)
MEMBERS = 3


def _complex_data(dtype):
    rng = np.random.default_rng(7)
    shape = (MEMBERS,) + GRID
    data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return data.astype(dtype)


def _real_data(dtype):
    rng = np.random.default_rng(11)
    return (10 * rng.standard_normal((MEMBERS,) + GRID)).astype(dtype)


def _members():
    return [OrdinalAxis(values=tuple(range(MEMBERS)))]


def _images(dtype, lazy):
    array = _real_data(dtype) if not np.iscomplexobj(dtype(0)) else _complex_data(dtype)
    if lazy:
        array = da.from_array(array, chunks=(1,) + GRID)
    return Images(array, sampling=0.1, ensemble_axes_metadata=_members())


def _patterns(dtype, lazy):
    array = np.abs(_real_data(dtype)) + 1
    if lazy:
        array = da.from_array(array, chunks=(1,) + GRID)
    return DiffractionPatterns(array, sampling=0.1, ensemble_axes_metadata=_members())


def _check(lazy, eager, member_offset_free=False):
    """The lazy result declares the dtype of its computed blocks and of the eager
    result. Values are compared up to each member's additive constant when
    `member_offset_free` is set."""
    declared = lazy.array.dtype
    computed = lazy.copy().compute().array
    assert declared == computed.dtype == eager.array.dtype
    expected = eager.array
    if member_offset_free:
        computed = computed - computed.min(axis=(-2, -1), keepdims=True)
        expected = expected - expected.min(axis=(-2, -1), keepdims=True)
    scale = np.abs(expected).max()
    np.testing.assert_allclose(
        computed,
        expected,
        rtol=0,
        atol=64 * np.finfo(expected.dtype).eps * scale,
    )


def _potential_array(dtype, lazy):
    array = np.abs(_real_data(dtype))[:, :, :]
    if lazy:
        array = da.from_array(array, chunks=(1,) + GRID)
    return PotentialArray(array, slice_thickness=1.0, sampling=0.1)


MEASUREMENTS = {
    "diffractograms_float64": (
        lambda: _images(np.float64, True),
        lambda: _images(np.float64, False),
        lambda m: m.diffractograms(),
    ),
    "diffractograms_int64": (
        lambda: _int_images(True),
        lambda: _int_images(False),
        lambda m: m.diffractograms(),
    ),
    "integrate_gradient_complex64": (
        lambda: _images(np.complex64, True),
        lambda: _images(np.complex64, False),
        lambda m: m.integrate_gradient(),
    ),
    "integrate_gradient_complex128": (
        lambda: _images(np.complex128, True),
        lambda: _images(np.complex128, False),
        lambda m: m.integrate_gradient(),
    ),
    "diffraction_patterns_interpolate": (
        lambda: _patterns(np.float64, True),
        lambda: _patterns(np.float64, False),
        lambda m: m.interpolate(sampling=0.25),
    ),
    "center_of_mass_float32": (
        lambda: _patterns(np.float32, True),
        lambda: _patterns(np.float32, False),
        lambda m: m.center_of_mass(),
    ),
    "center_of_mass_float64": (
        lambda: _patterns(np.float64, True),
        lambda: _patterns(np.float64, False),
        lambda m: m.center_of_mass(),
    ),
    "images_interpolate_fft_float64": (
        lambda: _images(np.float64, True),
        lambda: _images(np.float64, False),
        lambda m: m.interpolate(0.05),
    ),
    "images_interpolate_fft_float32": (
        lambda: _images(np.float32, True),
        lambda: _images(np.float32, False),
        lambda m: m.interpolate(0.05),
    ),
    # consistent on the base
    "images_interpolate_spline": (
        lambda: _images(np.float64, True),
        lambda: _images(np.float64, False),
        lambda m: m.interpolate(0.05, method="spline"),
    ),
}


def _int_images(lazy):
    array = (10 * _real_data(np.float64)).astype(np.int64)
    if lazy:
        array = da.from_array(array, chunks=(1,) + GRID)
    return Images(array, sampling=0.1, ensemble_axes_metadata=_members())


MEASUREMENT_PRECISION = {
    "images_interpolate_fft_float64": "float32",
    "images_interpolate_fft_float32": "float64",
}


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_lazy_measurements_declare_their_block_precision(name):
    make_lazy, make_eager, method = MEASUREMENTS[name]
    config = MEASUREMENT_PRECISION.get(name, "float32")
    with abtem.config.set({"precision": config, "fft": "numpy"}):
        lazy = method(make_lazy())
        eager = method(make_eager())
        assert lazy.is_lazy and not eager.is_lazy
        _check(lazy, eager, member_offset_free=name.startswith("integrate_gradient"))


def test_lazy_potential_array_transmission_function_declares_its_block_precision():
    with abtem.config.set({"precision": "float32"}):
        lazy = _potential_array(np.float64, True).transmission_function(100e3)
        eager = _potential_array(np.float64, False).transmission_function(100e3)
        assert lazy.is_lazy and not eager.is_lazy
        _check(lazy, eager)


def test_lazy_fft2_declares_the_input_precision():
    with abtem.config.set({"precision": "float32", "fft": "numpy"}):
        x = da.from_array(_complex_data(np.complex128), chunks=(1,) + GRID)
        result = fft2(x)
        assert result.dtype == result.compute().dtype == np.complex128


@pytest.mark.parametrize("kernel_dtype", [np.float32, np.float64, np.complex128])
@pytest.mark.parametrize("x_dtype", [np.complex64, np.complex128, np.float32])
def test_lazy_fft2_convolve_declares_the_dtype_of_its_blocks(x_dtype, kernel_dtype):
    # The product is taken in place, so the kernel's dtype does not matter.
    with abtem.config.set({"precision": "float32", "fft": "numpy"}):
        if np.issubdtype(x_dtype, np.complexfloating):
            data = _complex_data(x_dtype)
        else:
            data = _real_data(x_dtype)
        kernel = np.ones(GRID, dtype=kernel_dtype)
        lazy = fft2_convolve(da.from_array(data, chunks=(1,) + GRID), kernel)
        eager = fft2_convolve(data, kernel)
        assert lazy.dtype == lazy.compute().dtype == eager.dtype
