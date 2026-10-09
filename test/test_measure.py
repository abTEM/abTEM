import operator
import sys
import types
import warnings

import ase
import dask.array as da
import hypothesis.strategies as st
import numpy as np
import pytest
import scipy.ndimage
import scipy.signal
import strategies as abtem_st
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.strategies import composite
from utils import (
    array_is_close,
    assert_array_matches_device,
    devices,
    ensure_is_tuple,
    gpu,
    lazy_params,
    requires_gpu,
)

import abtem
from abtem.core.axes import OrdinalAxis, ScanAxis
from abtem.core.backend import asnumpy, copy_to_device, get_array_module
from abtem.core.energy import energy2wavelength
from abtem.core.utils import get_dtype
from abtem.measurements import (
    DiffractionPatterns,
    Images,
    PolarMeasurements,
    RealSpaceLineProfiles,
    ReciprocalSpaceLineProfiles,
    _apply_convolve_2d_on_axes,
    _gaussian_kernel_2d,
    _gaussian_kernels_1d,
    _interpolate_stack,
    _scan_sampling,
    _scan_shape,
)
from abtem.waves import Probe


def test_scanned_measurement_type():
    array = np.zeros((10, 10, 10, 10, 10))

    ensemble_axes_metadata = [
        ScanAxis(_main=False),
        OrdinalAxis(values=(1,) * 10),
        ScanAxis(),
    ]
    measurement = DiffractionPatterns(
        array,
        sampling=0.1,
        ensemble_axes_metadata=ensemble_axes_metadata,
        metadata={"energy": 100e3},
    )
    assert isinstance(
        measurement.integrate_radial(inner=0, outer=10), RealSpaceLineProfiles
    )

    ensemble_axes_metadata = [OrdinalAxis(values=(1,) * 10), ScanAxis(), ScanAxis()]
    measurement = DiffractionPatterns(
        array,
        sampling=0.1,
        ensemble_axes_metadata=ensemble_axes_metadata,
        metadata={"energy": 100e3},
    )
    assert isinstance(measurement.integrate_radial(inner=0, outer=10), Images)

    ensemble_axes_metadata = [ScanAxis(), ScanAxis(), OrdinalAxis(values=(1,) * 10)]
    measurement = DiffractionPatterns(
        array,
        sampling=0.1,
        ensemble_axes_metadata=ensemble_axes_metadata,
        metadata={"energy": 100e3},
    )

    # with pytest.raises(RuntimeError):
    #    measurement.integrate_radial(inner=0, outer=10)


@settings(max_examples=5)
@given(data=st.data())
@pytest.mark.parametrize("method", ["__add__", "__sub__", "__mul__", "__truediv__"])
@lazy_params
@devices
@pytest.mark.parametrize(
    "measurement",
    [
        abtem_st.images,
        abtem_st.line_profiles,
        abtem_st.diffraction_patterns,
        abtem_st.polar_measurements,
    ],
)
def test_add_subtract(data, measurement, method, lazy, device):
    measurement = data.draw(measurement(lazy=lazy, device=device))
    # A second operand distinct from the first, b = 2a + 1 (>= 1, so safe to
    # divide by), so that no two of +, -, *, / can give the same result.
    other = measurement.__class__(
        **{
            **measurement._copy_kwargs(exclude=("array",)),
            "array": measurement.array * 2 + 1,
        }
    )
    # compute() replaces a lazy object's array in place, so take the oracle
    # values from copies: computing the operands themselves would make the
    # operation below run eagerly even when lazy=True.
    a = asnumpy(measurement.copy().compute().array)
    b = asnumpy(other.copy().compute().array)

    new_measurement = getattr(measurement, method)(other)
    assert new_measurement.array is not measurement.array
    assert new_measurement.is_lazy == lazy

    # Oracle: the same elementwise operation on the plain numpy arrays.
    expected = getattr(np.asarray(a, dtype=np.float64), method)(b)
    np.testing.assert_allclose(
        asnumpy(new_measurement.compute().array), expected, rtol=1e-6
    )
    # Not in place: the left operand is left untouched.
    np.testing.assert_array_equal(asnumpy(measurement.compute().array), a)


@lazy_params
@devices
@pytest.mark.parametrize("scalar", [2.0, -0.5])
def test_reflected_arithmetic_with_a_scalar(scalar, lazy, device):
    # Oracle: numpy's own reflected operators on the plain array. The array is
    # not symmetric under any of the operations, so e.g. `scalar / m` computed
    # as `m / scalar` (as __rtruediv__ = __truediv__ used to do) fails.
    array = np.array([[1.0, 2.0, 4.0], [8.0, 0.5, 0.25]], dtype=get_dtype())
    measurement = Images(array, sampling=(0.1, 0.2))
    if lazy:
        measurement = Images(da.from_array(array, chunks=(1, 3)), sampling=(0.1, 0.2))
    measurement = measurement.copy_to_device(device)

    for result, expected in (
        (scalar / measurement, scalar / array),
        (scalar * measurement, scalar * array),
    ):
        assert isinstance(result, Images)
        np.testing.assert_allclose(
            asnumpy(result.compute().array), expected, rtol=1e-6
        )


@lazy_params
@devices
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
@pytest.mark.parametrize(
    "operand_type",
    [
        "numpy_float64",
        "numpy_int64",
        "0d_numpy_array",
        "0d_device_array",
        "numpy_float64_array",
    ],
)
def test_arithmetic_with_a_numpy_or_device_operand(
    operand_type, op, in_place, lazy, device
):
    # Oracle: the same operation on the plain arrays in double precision. NumPy
    # promotes a single-precision measurement to double with any of these
    # operands; the torch backend holds single precision only and must give the
    # single-precision result instead, eager and lazy alike. An in-place
    # operation keeps single precision on every backend.
    if in_place and lazy:
        pytest.skip("in-place arithmetic refuses lazy measurements")
    xp = get_array_module(device)
    host_operand = {
        "numpy_float64": np.float64(-0.5),
        "numpy_int64": np.int64(3),
        "0d_numpy_array": np.asarray(2.0),
        "0d_device_array": np.asarray(2.0, dtype=get_dtype()),
        # Broadcasts along the last axis, which is 3 long and the other 2.
        "numpy_float64_array": np.array([0.5, 2.0, 4.0]),
    }[operand_type]
    operand = (
        xp.asarray(host_operand) if operand_type == "0d_device_array" else host_operand
    )
    array = np.array([[1.0, 2.0, 4.0], [8.0, 0.5, 0.25]], dtype=get_dtype())
    measurement = Images(
        da.from_array(array, chunks=(1, 3)) if lazy else array.copy(),
        sampling=(0.1, 0.2),
    ).copy_to_device(device)

    result = getattr(operator, ("i" if in_place else "") + op)(measurement, operand)

    assert isinstance(result, Images)
    assert result.is_lazy == lazy
    computed = result.compute().array
    assert_array_matches_device(computed, device)
    if in_place or device == "mps":
        assert asnumpy(computed).dtype == get_dtype()
    expected = getattr(operator, op)(
        array.astype(np.float64), np.asarray(host_operand, dtype=np.float64)
    )
    np.testing.assert_allclose(
        asnumpy(computed), expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max()
    )


def test_in_place_true_division_refuses_lazy_measurements():
    # Like the other in-place operators, /= must refuse a lazy measurement
    # rather than silently returning a new (lazy) object.
    measurement = Images(da.ones((4, 4), chunks=2), sampling=0.1)
    with pytest.raises(RuntimeError, match="inplace"):
        measurement /= 2.0


@settings(max_examples=5)
@given(data=st.data())
@pytest.mark.parametrize("method", ["__iadd__", "__isub__", "__imul__", "__itruediv__"])
@devices
@pytest.mark.parametrize(
    "measurement",
    [
        abtem_st.images,
        abtem_st.line_profiles,
        abtem_st.diffraction_patterns,
        abtem_st.polar_measurements,
    ],
)
def test_inplace_add_subtract(data, measurement, method, device):
    measurement = data.draw(measurement(lazy=False, device=device))
    new_measurement = getattr(measurement, method)(measurement.copy())
    assert new_measurement.array is measurement.array


@given(data=st.data())
@pytest.mark.parametrize("method", ["sum", "mean", "std"])
@devices
@pytest.mark.parametrize(
    "measurement",
    [
        abtem_st.images,
        abtem_st.line_profiles,
        abtem_st.diffraction_patterns,
        abtem_st.polar_measurements,
    ],
)
def test_reduce(data, measurement, method, device):
    measurement = data.draw(measurement(lazy=True, device=device))

    axes_indices = st.integers(
        min_value=0, max_value=max(len(measurement.ensemble_shape) - 1, 0)
    )
    axes_indices = st.lists(
        elements=axes_indices,
        min_size=0,
        max_size=len(measurement.ensemble_shape),
        unique=True,
    )
    axes_indices = data.draw(axes_indices)

    axes = tuple(axes_indices)
    num_lost_dims = len(axes)

    new_measurement = getattr(measurement.compute(), method)(axes)

    assert len(new_measurement.shape) == len(measurement.shape) - num_lost_dims


@composite
def gpts_or_sampling(draw):
    return draw(
        st.one_of(
            st.fixed_dictionaries({"gpts": abtem_st.gpts(), "sampling": st.none()}),
            st.fixed_dictionaries({"gpts": st.none(), "sampling": abtem_st.sampling()}),
        )
    )


@given(data=st.data(), gpts_or_sampling=gpts_or_sampling())
@lazy_params
@devices
@pytest.mark.parametrize("method", ["spline", "fft"])
def test_interpolate_images(data, gpts_or_sampling, lazy, device, method):
    measurement = data.draw(abtem_st.images(lazy=lazy, device=device))
    interpolated = measurement.interpolate(**gpts_or_sampling, method=method)
    assert np.allclose(interpolated.extent, measurement.extent)
    if gpts_or_sampling["gpts"]:
        assert interpolated.base_shape == ensure_is_tuple(gpts_or_sampling["gpts"], 2)
    elif gpts_or_sampling["sampling"]:
        sampling = ensure_is_tuple(gpts_or_sampling["sampling"], 2)
        adjusted_sampling = tuple(
            l / np.ceil(l / d) for d, l in zip(sampling, measurement.extent)
        )
        assert np.allclose(interpolated.sampling, adjusted_sampling)


@given(
    data=st.data(),
    tile=st.tuples(
        st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3)
    ),
)
@lazy_params
@devices
def test_tile_images(data, tile, lazy, device):
    measurement = data.draw(abtem_st.images(lazy=lazy, device=device))
    tiled = measurement.tile(tile)
    assert np.allclose(np.array(measurement.extent) * tile, tiled.extent)
    assert (
        tuple(n * t for n, t in zip(measurement.base_shape, tile)) == tiled.base_shape
    )


def _sigma_strategy(max_value=5.0):
    """Physical-unit sigma/half-width, either exactly 0.0 or a "sensible"
    nonzero float.

    Excludes hypothesis' extreme near-zero (but nonzero) floats -- e.g.
    ~1e-300 -- which are physically indistinguishable from zero at any sane
    pixel sampling, provide no additional test coverage over the sigma=0.0
    case the filters already special-case, and can silently underflow to 0
    when squared (``sigma**2`` for such a value is smaller than the
    smallest representable float64), which previously raised a bare
    ZeroDivisionError deep inside the Gaussian kernel construction.
    """
    return st.one_of(st.just(0.0), st.floats(min_value=1e-6, max_value=max_value))


@composite
def sigma(draw, max_value=5.0):
    sigma = _sigma_strategy(max_value)
    return draw(st.one_of(st.tuples(sigma, sigma), sigma))


# Anisotropic pixel sampling (x != y), so that applying a sigma/sampling
# component to the wrong image axis changes the result.
_anisotropic_sampling = st.tuples(
    st.floats(min_value=0.02, max_value=0.05),
    st.floats(min_value=0.06, max_value=0.1),
)

# Binary-exact anisotropic sampling for the Lorentzian-family tests: with
# half-widths chosen as (multiple of 0.5 px) x sampling, the documented
# truncation window (``truncate`` half-widths along each axis) ends exactly on
# a pixel, so the reference kernel below is unambiguous.
_LORENTZIAN_SAMPLING = (0.125, 0.0625)
_lorentzian_hw_pixels = st.sampled_from([0.0, 0.5, 1.5, 2.5])

# How each filter's documented `boundary` maps onto scipy.ndimage's `mode`
# ('periodic' wraps around; the Gaussian's 'reflect' reflects about the edge
# of the last pixel, which is scipy's 'reflect').
_BOUNDARY_TO_SCIPY = {"periodic": "wrap", "reflect": "reflect", "constant": "constant"}


def _with_sampling(measurement, sampling):
    return Images(
        array=measurement.array,
        sampling=sampling,
        ensemble_axes_metadata=measurement.ensemble_axes_metadata,
        metadata=measurement.metadata,
    )


def _as_float64(measurement):
    return np.asarray(asnumpy(measurement.compute().array), dtype=np.float64)


def _assert_matches_reference(result, expected, rel=1e-5):
    """Compare with a tolerance relative to the reference signal's peak, so
    the tolerance can never exceed the signal itself (a zero image fails)."""
    result = np.asarray(asnumpy(result), dtype=np.float64)
    scale = np.abs(expected).max()
    assert scale > 0
    np.testing.assert_allclose(result, expected, rtol=0, atol=rel * scale)


def _analytic_lorentzian_kernel(hw_pixels, truncate=10.0):
    """The Lorentzian kernel as documented by ``Images.lorentzian_filter``:

        L(x, y) = 1 / (1 + (x/γ_x)² + (y/γ_y)²),   γ = HWHM in pixels,

    truncated at ``truncate`` half-widths along each axis and normalized to
    unit sum; the γ → 0 limit along an axis is a delta along that axis.
    """
    terms = []
    for axis, hw in enumerate(hw_pixels):
        radius = int(round(truncate * hw))
        assert radius == truncate * hw, "choose hw so the window ends on a pixel"
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        shape = (-1, 1) if axis == 0 else (1, -1)
        terms.append(((x / hw) ** 2 if hw > 0 else x * 0.0).reshape(shape))
    kernel = 1.0 / (1.0 + terms[0] + terms[1])
    return kernel / kernel.sum()


def _scipy_gaussian_kernel(sigma_pixels):
    """scipy.ndimage's own (truncate=4) 2-D Gaussian kernel, obtained as the
    impulse response of scipy.ndimage.gaussian_filter on a delta that fits
    the whole kernel."""
    radii = [int(4.0 * s + 0.5) for s in sigma_pixels]
    delta = np.zeros([2 * r + 1 for r in radii])
    delta[radii[0], radii[1]] = 1.0
    return scipy.ndimage.gaussian_filter(delta, sigma_pixels, mode="constant")


def _convolve_base_axes(array, kernel_2d, mode):
    """Reference convolution over the two trailing (image) axes."""
    kernel = kernel_2d.reshape((1,) * (array.ndim - 2) + kernel_2d.shape)
    return scipy.ndimage.convolve(array, kernel, mode=mode, cval=0.0)


@given(
    data=st.data(),
    sigma=sigma(),
    sampling=_anisotropic_sampling,
    boundary=st.sampled_from(["periodic", "reflect", "constant"]),
)
@lazy_params
@devices
def test_gaussian_filter_images(data, sigma, sampling, boundary, lazy, device):
    measurement = _with_sampling(
        data.draw(abtem_st.images(lazy=lazy, device=device)), sampling
    )
    try:
        filtered = measurement.gaussian_filter(sigma, boundary=boundary).compute()
    except OSError:
        pytest.skip(
            "Known CuPy error, but only reproducible in pytest https://github.com/cupy/cupy/issues/8218"
        )
    original = _as_float64(measurement)

    # Oracle: scipy.ndimage.gaussian_filter, with sigma converted from Å to
    # pixels separately along x (axis -2) and y (axis -1) and no smoothing
    # across the ensemble axes.
    sigma_x, sigma_y = ensure_is_tuple(sigma, 2)
    sigma_pixels = (0.0,) * (original.ndim - 2) + (
        sigma_x / sampling[0],
        sigma_y / sampling[1],
    )
    expected = scipy.ndimage.gaussian_filter(
        original, sigma_pixels, mode=_BOUNDARY_TO_SCIPY[boundary]
    )
    _assert_matches_reference(filtered.array, expected)


@given(
    data=st.data(),
    hw_pixels=st.tuples(_lorentzian_hw_pixels, _lorentzian_hw_pixels),
    boundary=st.sampled_from(["periodic", "constant"]),
)
@lazy_params
@devices
def test_lorentzian_filter_images(data, hw_pixels, boundary, lazy, device):
    measurement = _with_sampling(
        data.draw(abtem_st.images(lazy=lazy, device=device)), _LORENTZIAN_SAMPLING
    )
    half_width = tuple(h * d for h, d in zip(hw_pixels, _LORENTZIAN_SAMPLING))
    try:
        filtered = measurement.lorentzian_filter(
            half_width, boundary=boundary
        ).compute()
    except OSError:
        pytest.skip(
            "Known CuPy error, but only reproducible in pytest https://github.com/cupy/cupy/issues/8218"
        )
    original = _as_float64(measurement)

    # Oracle: direct (scipy.ndimage) convolution with the analytic kernel.
    expected = _convolve_base_axes(
        original,
        _analytic_lorentzian_kernel(hw_pixels),
        mode=_BOUNDARY_TO_SCIPY[boundary],
    )
    _assert_matches_reference(filtered.array, expected)


@given(
    data=st.data(),
    sigma_pixels=st.tuples(
        st.floats(min_value=0.0, max_value=3.0), st.floats(min_value=0.0, max_value=3.0)
    ),
    hw_pixels=st.tuples(_lorentzian_hw_pixels, _lorentzian_hw_pixels),
)
@lazy_params
@devices
def test_voigtian_filter_images(data, sigma_pixels, hw_pixels, lazy, device):
    measurement = _with_sampling(
        data.draw(abtem_st.images(lazy=lazy, device=device)), _LORENTZIAN_SAMPLING
    )
    sigma = tuple(s * d for s, d in zip(sigma_pixels, _LORENTZIAN_SAMPLING))
    half_width = tuple(h * d for h, d in zip(hw_pixels, _LORENTZIAN_SAMPLING))
    try:
        filtered = measurement.voigtian_filter(sigma, half_width).compute()
    except OSError:
        pytest.skip(
            "Known CuPy error, but only reproducible in pytest https://github.com/cupy/cupy/issues/8218"
        )
    original = _as_float64(measurement)

    # Oracle: a single periodic convolution with the Voigt kernel, built as
    # the full linear convolution of the Gaussian and Lorentzian kernels.
    voigt_kernel = scipy.signal.convolve2d(
        _scipy_gaussian_kernel(sigma_pixels), _analytic_lorentzian_kernel(hw_pixels)
    )
    expected = _convolve_base_axes(original, voigt_kernel, mode="wrap")
    _assert_matches_reference(filtered.array, expected)


def test_images_coordinates_spaced_by_sampling():
    # Pixel i of a periodic image with sampling s sits at i * s; the last
    # pixel is one sampling short of the extent, not at the extent.
    images = Images(np.zeros((4, 5)), sampling=(0.5, 0.2))
    x, y = images.coordinates
    assert np.allclose(x, [0.0, 0.5, 1.0, 1.5])
    assert np.allclose(y, [0.0, 0.2, 0.4, 0.6, 0.8])


def _delta_probe_image(gpts=64, lazy=False):
    """A throwaway probe-intensity image for the filter tests below -- only
    a non-trivial 2D image is needed, not any particular physics."""
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=gpts)
    return wave.build((0, 0), lazy=lazy).intensity()


def test_lorentzian_filter_changes_image():
    """Lorentzian filter with a non-trivial HWHM should change the image."""
    images = _delta_probe_image()
    filtered = images.lorentzian_filter(0.5)
    assert not np.allclose(filtered.array, images.array)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_lorentzian_filter_respects_precision_config(precision):
    """The Lorentzian family of filters must honour ``abtem.config['precision']``.

    The kernel is built via :func:`_lorentzian_kernel_2d`, which queries the
    config — and the output dtype must match the configured precision rather
    than being silently downcast to float32. Regression for a bug where the
    kernel was hardcoded to ``np.float32``.
    """
    expected_dtype = np.dtype(precision)
    n = 21
    with abtem.config.set({"precision": precision}):
        arr = np.zeros((n, n), dtype=expected_dtype)
        arr[n // 2, n // 2] = 1.0
        images = Images(arr, sampling=(0.1, 0.1))

        out_l = images.lorentzian_filter(0.3).array
        out_v = images.voigtian_filter(0.2, 0.3).array
        out_pv = images.pseudo_voigtian_filter(0.2, 0.3, eta=0.5).array

    assert out_l.dtype == expected_dtype, (
        f"lorentzian_filter: got {out_l.dtype}, expected {expected_dtype}"
    )
    assert out_v.dtype == expected_dtype, (
        f"voigtian_filter: got {out_v.dtype}, expected {expected_dtype}"
    )
    assert out_pv.dtype == expected_dtype, (
        f"pseudo_voigtian_filter: got {out_pv.dtype}, expected {expected_dtype}"
    )


@pytest.mark.parametrize(
    "filter_name,kwargs",
    [
        ("lorentzian_filter", dict(half_width=0.5, truncate=10.0)),
        ("voigtian_filter", dict(gaussian_sigma=0.1, lorentzian_gamma=0.5)),
        (
            "pseudo_voigtian_filter",
            dict(gaussian_sigma=0.1, lorentzian_gamma=0.5, eta=0.5),
        ),
    ],
)
def test_lorentzian_family_filters_are_rotationally_symmetric(filter_name, kwargs):
    """
    Filtering a delta image with the Lorentzian family of filters must produce
    a rotationally-symmetric impulse response. A naively separable 1-D × 1-D
    Lorentzian would be several times brighter along the x and y axes than
    along the diagonal at the same radius, producing visible cross-shaped halos
    around sharp features. Regression for that bug.
    """
    # Square delta image with isotropic sampling
    n = 81
    arr = np.zeros((n, n), dtype=np.float32)
    arr[n // 2, n // 2] = 1.0
    images = Images(arr, sampling=(0.1, 0.1))

    out = getattr(images, filter_name)(**kwargs).array
    c = n // 2

    # Sample at several radii; compare on-axis vs along-diagonal at the same r.
    for r in (4, 8, 12, 20):
        d = int(round(r / np.sqrt(2)))  # so √2·d ≈ r
        ax_val = float(out[c, c + r])
        diag_val = float(out[c + d, c + d])
        ratio = ax_val / diag_val
        # True 2-D Lorentzian / Voigt is rotationally symmetric → ratio ≈ 1.
        # The old separable implementation gave ratio ≳ 3 at large r.
        assert 0.85 < ratio < 1.15, (
            f"{filter_name}: axis/diag ratio = {ratio:.3g} at r={r} "
            f"(ax={ax_val:.4g}, diag={diag_val:.4g}); kernel is not "
            "rotationally symmetric"
        )


def test_voigtian_filter_changes_image():
    """Voigtian filter with non-trivial parameters should change the image."""
    images = _delta_probe_image()
    filtered = images.voigtian_filter(0.3, 0.3)
    assert not np.allclose(filtered.array, images.array)


def _unit_delta_image(shape=(64, 96), device="cpu"):
    """A unit delta on an anisotropically sampled grid, so each filter's
    output is its own (normalized) impulse response with peak << 1 but
    known exactly, and tolerances can be set relative to that peak."""
    array = np.zeros(shape, dtype=get_dtype(complex=False))
    array[shape[0] // 2, shape[1] // 2] = 1.0
    images = Images(array, sampling=_LORENTZIAN_SAMPLING)
    return images.to_gpu() if device == "gpu" else images


# Anisotropic widths in pixels; half-widths are multiples of 0.5 px so the
# Lorentzian truncation window ends on a pixel (see _analytic_lorentzian_kernel).
_LIMIT_SIGMA_PIXELS = (2.0, 1.25)
_LIMIT_HW_PIXELS = (1.5, 2.5)
_LIMIT_SIGMA = tuple(s * d for s, d in zip(_LIMIT_SIGMA_PIXELS, _LORENTZIAN_SAMPLING))
_LIMIT_HW = tuple(h * d for h, d in zip(_LIMIT_HW_PIXELS, _LORENTZIAN_SAMPLING))


def _reference_impulse_response(kernel, shape=(64, 96)):
    delta = np.zeros(shape)
    delta[shape[0] // 2, shape[1] // 2] = 1.0
    return scipy.ndimage.convolve(delta, kernel, mode="wrap")


@devices
def test_voigtian_filter_matches_convolved_kernels(device):
    """The Voigt impulse response must equal the numerical (linear)
    convolution of the Gaussian and the Lorentzian kernels."""
    out = _unit_delta_image(device=device).voigtian_filter(_LIMIT_SIGMA, _LIMIT_HW)
    voigt_kernel = scipy.signal.convolve2d(
        _scipy_gaussian_kernel(_LIMIT_SIGMA_PIXELS),
        _analytic_lorentzian_kernel(_LIMIT_HW_PIXELS),
    )
    _assert_matches_reference(out.array, _reference_impulse_response(voigt_kernel))


@devices
def test_voigtian_filter_pure_gaussian_limit(device):
    """voigtian_filter with lorentzian_gamma=0 must be a pure Gaussian."""
    out = _unit_delta_image(device=device).voigtian_filter(_LIMIT_SIGMA, 0.0)
    expected = _reference_impulse_response(_scipy_gaussian_kernel(_LIMIT_SIGMA_PIXELS))
    _assert_matches_reference(out.array, expected)


@devices
def test_voigtian_filter_pure_lorentzian_limit(device):
    """voigtian_filter with gaussian_sigma=0 must be a pure Lorentzian."""
    out = _unit_delta_image(device=device).voigtian_filter(0.0, _LIMIT_HW)
    expected = _reference_impulse_response(_analytic_lorentzian_kernel(_LIMIT_HW_PIXELS))
    _assert_matches_reference(out.array, expected)


def test_pseudo_voigtian_filter_changes_image():
    """Pseudo-Voigtian filter with non-trivial parameters should change the image."""
    images = _delta_probe_image()
    filtered = images.pseudo_voigtian_filter(0.3, 0.3, eta=0.5)
    assert not np.allclose(filtered.array, images.array)


@pytest.mark.parametrize("eta", [0.0, 0.3, 1.0])
@devices
def test_pseudo_voigtian_filter_mixes_components(eta, device):
    """pseudo_voigtian_filter = (1 - eta) * Gaussian + eta * Lorentzian
    (Nguyen et al. 2014), including the pure limits eta = 0 and eta = 1."""
    out = _unit_delta_image(device=device).pseudo_voigtian_filter(
        _LIMIT_SIGMA, _LIMIT_HW, eta=eta
    )
    gaussian = _reference_impulse_response(_scipy_gaussian_kernel(_LIMIT_SIGMA_PIXELS))
    lorentzian = _reference_impulse_response(
        _analytic_lorentzian_kernel(_LIMIT_HW_PIXELS)
    )
    _assert_matches_reference(out.array, (1 - eta) * gaussian + eta * lorentzian)


# Maps _apply_convolve_2d_on_axes' internal mode names onto the
# scipy.ndimage mode that defines the same boundary extension.
_CONVOLVE_MODE_TO_SCIPY = {
    "wrap": "wrap",
    "symmetric": "reflect",
    "constant": "constant",
    "reflect": "mirror",
}


@pytest.mark.parametrize("mode", list(_CONVOLVE_MODE_TO_SCIPY))
def test_convolve_handles_kernel_radius_larger_than_axis(mode):
    """_apply_convolve_2d_on_axes must stay exact -- and not blow up memory --
    when the kernel radius (sigma / sampling) vastly exceeds the array's own
    axis length along the filtered axes, for *every* boundary mode.

    This is exactly the shape gaussian_source_size hits for a small scan
    grid smoothed with a much finer sampling than the scan step (e.g. a
    handful of scan positions with sub-pixel-scale sampling and sigma of a
    few sampling units): the physical-to-pixel sigma conversion then yields
    a kernel radius far larger than the scan axis itself.

    Padding the array by that radius scales the padded axis -- and
    multiplicatively every other axis of the buffer -- by orders of
    magnitude; on GPU this produced a CuPy OutOfMemoryError trying to
    allocate tens of GB for an array whose raw data was a few KB. Each mode
    now has a route whose cost is bounded by the array instead: no padding
    at all for "wrap", one period of the mirrored signal for "reflect" and
    "symmetric", and dropping the unreachable taps for "constant".

    These helpers are backend-agnostic (they dispatch via
    get_array_module), so this runs on CPU without a GPU while exercising
    the same code GPU calls run.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    # Mimics a DiffractionPatterns array: 2 small scan axes + 2 base axes.
    array = rng.random((3, 4, 16, 16)).astype(np.float64)

    # sigma=1.7 physical units at sampling=0.01 -> ~170 px sigma -> radius
    # ~680, vastly larger than the scan axes' own length of 3 and 4.
    sigma_pixels = 1.7 / 0.01
    kernels_1d = _gaussian_kernels_1d((sigma_pixels, sigma_pixels))
    assert kernels_1d[0].shape[0] > 10 * array.shape[0]

    cval = 2.5 if mode == "constant" else 0.0
    got = _apply_convolve_2d_on_axes(
        array, None, axes=(0, 1), mode=mode, cval=cval, kernels_1d=kernels_1d
    )
    expected = gaussian_filter(
        array,
        sigma=(sigma_pixels, sigma_pixels, 0.0, 0.0),
        mode=_CONVOLVE_MODE_TO_SCIPY[mode],
        cval=cval,
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("mode", list(_CONVOLVE_MODE_TO_SCIPY))
def test_convolve_separable_kernels_match_dense_kernel(mode):
    """Passing the separable 1-D Gaussian kernels must be equivalent to
    passing their dense outer product.

    The 1-D form is what the filters actually use, so that kernel memory
    stays O(radius) rather than O(radius**2) -- the radius scales with
    sigma / sampling and can be far larger than the array itself.
    """
    rng = np.random.default_rng(0)
    array = rng.random((5, 7, 4)).astype(np.float64)

    kernels_1d = _gaussian_kernels_1d((1.7, 0.9))
    kernel_2d = _gaussian_kernel_2d((1.7, 0.9))

    separable = _apply_convolve_2d_on_axes(
        array, None, axes=(0, 1), mode=mode, cval=1.5, kernels_1d=kernels_1d
    )
    dense = _apply_convolve_2d_on_axes(
        array, kernel_2d, axes=(0, 1), mode=mode, cval=1.5
    )
    # Tolerance is set by the kernel dtype (abtem.config['precision'], float32
    # by default): the two representations sum the same weights in a different
    # order, so they agree only to kernel precision, not bitwise.
    np.testing.assert_allclose(separable, dense, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("boundary", ["periodic", "reflect", "constant"])
def test_filters_preserve_complex_images(boundary):
    """The filters must keep working on complex measurements.

    DiffractionPatterns.center_of_mass returns complex Images by default
    (real/imaginary hold the two CoM components), differential(
    return_complex=True) produces them, and integrate_gradient requires
    them -- so smoothing a complex image is an ordinary DPC/CoM step. The
    real-input FFTs (rfftn/irfftn) reject complex arrays, so the filters
    must dispatch to the full complex transforms instead.

    Each component must come back filtered exactly as if it had been
    filtered on its own.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    array = rng.random((16, 16)) + 1j * rng.random((16, 16))
    images = Images(array, sampling=0.1)

    sigma = 0.3
    filtered = images.gaussian_filter(sigma, boundary=boundary).array
    assert np.iscomplexobj(filtered)

    scipy_mode = {"periodic": "wrap", "reflect": "reflect", "constant": "constant"}[
        boundary
    ]
    sigma_pixels = sigma / images.sampling[0]
    expected = gaussian_filter(
        array.real, sigma=sigma_pixels, mode=scipy_mode
    ) + 1j * gaussian_filter(array.imag, sigma=sigma_pixels, mode=scipy_mode)
    np.testing.assert_allclose(filtered, expected, atol=1e-9)

    # The Lorentzian family shares the same convolution helper.
    for method, args, kwargs in [
        ("lorentzian_filter", (0.3,), {}),
        ("voigtian_filter", (0.3, 0.3), {}),
        ("pseudo_voigtian_filter", (0.3, 0.3), dict(eta=0.5)),
    ]:
        out = getattr(images, method)(*args, boundary=boundary, **kwargs).array
        assert np.iscomplexobj(out), method
        assert np.abs(out.imag).max() > 0, method


def test_dtype_preserving_operations_keep_complex():
    """Operations that pass their input values through must declare a dask
    dtype that follows the input, not the configured precision.

    The declared dtype is the invariant worth pinning: a lazy array declared
    real while its blocks are complex is already wrong, and whether it goes on
    to actually lose the imaginary part depends on the chunking and on which
    dask assembly path runs -- which is exactly what made this hide (it
    surfaced only for voigtian_filter with boundary="constant"). So assert on
    the graph's dtype rather than hoping a given shape happens to trigger the
    cast, and check the computed result against the eager one as well.

    Complex measurements are ordinary here: center_of_mass returns complex
    Images, differential(return_complex=True) produces them.
    """
    rng = np.random.default_rng(0)

    def pair(measurement_cls, array, chunks, **kwargs):
        return (
            measurement_cls(array, **kwargs),
            measurement_cls(da.from_array(array, chunks=chunks), **kwargs),
        )

    dp_kwargs = dict(
        sampling=0.1,
        ensemble_axes_metadata=[ScanAxis(sampling=0.2, _main=True)] * 2,
        metadata={"energy": 100e3},
    )

    im_e, im_l = pair(
        Images, rng.random((32, 32)) + 1j * rng.random((32, 32)), (16, 16), sampling=0.1
    )
    lp_e, lp_l = pair(
        RealSpaceLineProfiles, rng.random(64) + 1j * rng.random(64), 32, sampling=0.1
    )
    dp_e, dp_l = pair(
        DiffractionPatterns,
        rng.random((4, 4, 16, 16)) + 1j * rng.random((4, 4, 16, 16)),
        (2, 2, 16, 16),
        **dp_kwargs,
    )

    cases = {
        "interpolate_line": (lambda m: m.interpolate_line((0, 0), (2, 2)), im_e, im_l),
        "line_profile_interpolate": (lambda m: m.interpolate(gpts=128), lp_e, lp_l),
        "bandlimit": (lambda m: m.bandlimit(0, 10), dp_e, dp_l),
        "polar_binning": (lambda m: m.polar_binning(4, 4, 0, 10), dp_e, dp_l),
        "integrate_radial": (lambda m: m.integrate_radial(0, 10), dp_e, dp_l),
        "azimuthal_average": (lambda m: m.azimuthal_average(), dp_e, dp_l),
    }

    for name, (operation, eager_in, lazy_in) in cases.items():
        lazy_result = operation(lazy_in)
        assert np.iscomplexobj(
            np.empty(0, dtype=lazy_result.array.dtype)
        ), f"{name} declares a real dask dtype for complex input"

        # A silent downcast only warns, so make it fail loudly here.
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.exceptions.ComplexWarning)
            computed = np.asarray(lazy_result.compute().array)

        assert np.iscomplexobj(computed), f"{name} dropped the complex dtype"
        np.testing.assert_allclose(
            computed, np.asarray(operation(eager_in).compute().array), err_msg=name
        )


def test_filter_boundary_modes():
    """All four filter methods accept all three boundary modes without error."""
    images = _delta_probe_image(gpts=32)
    for boundary in ("periodic", "reflect", "constant"):
        images.gaussian_filter(0.3, boundary=boundary).array
        images.lorentzian_filter(0.3, boundary=boundary).array
        images.voigtian_filter(0.3, 0.3, boundary=boundary).array
        images.pseudo_voigtian_filter(0.3, 0.3, eta=0.5, boundary=boundary).array


@requires_gpu
@pytest.mark.parametrize("boundary", ["periodic", "reflect", "constant"])
@lazy_params
@pytest.mark.parametrize("complex_input", [False, True])
def test_gaussian_family_filters_match_cpu_and_gpu(boundary, lazy, complex_input):
    """gaussian_filter (and, through it, voigtian_filter/pseudo_voigtian_filter) uses
    a different implementation on GPU than on CPU -- FFT-based convolution instead of
    cupyx.scipy.ndimage.gaussian_filter -- to avoid per-(sigma, shape) CUDA kernel
    recompilation overhead. Nothing else in the suite checks the two backends agree
    numerically, so do that explicitly here for all three boundary modes.

    Complex measurements take a different branch again (the real-input FFTs reject
    them), and they are an ordinary case: DiffractionPatterns.center_of_mass returns
    complex Images, so smoothing one is a normal DPC/CoM step. A complex wave function
    stands in for that here -- it exercises the same dtype without needing a scan.
    """
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=48)
    built = wave.build((0, 0), lazy=lazy)

    if complex_input:
        images_cpu = Images(built.array, sampling=built.sampling)
    else:
        images_cpu = built.intensity()

    assert np.iscomplexobj(images_cpu.array) == complex_input
    images_gpu = images_cpu.to_gpu()

    sigma, gamma = 0.7, 0.4
    for method, kwargs in [
        ("gaussian_filter", dict(sigma=sigma)),
        ("voigtian_filter", dict(gaussian_sigma=sigma, lorentzian_gamma=gamma)),
        (
            "pseudo_voigtian_filter",
            dict(gaussian_sigma=sigma, lorentzian_gamma=gamma, eta=0.5),
        ),
    ]:
        cpu_array = getattr(images_cpu, method)(boundary=boundary, **kwargs)
        gpu_array = getattr(images_gpu, method)(boundary=boundary, **kwargs)
        cpu_array = cpu_array.compute().array
        gpu_array = gpu_array.to_cpu().compute().array

        assert np.iscomplexobj(gpu_array) == complex_input, method

        # Scale the absolute tolerance to the data: a probe's values are of
        # order 1e-5 here, so a fixed atol would pass no matter what the two
        # backends returned. The two paths (scipy.ndimage vs the FFT helper)
        # differ by ~2e-7 of the peak when measured on the same input, so
        # this leaves a comfortable margin for cuFFT rounding.
        np.testing.assert_allclose(
            cpu_array,
            gpu_array,
            atol=1e-5 * np.abs(cpu_array).max(),
            rtol=1e-5,
            err_msg=f"{method} disagrees between CPU and GPU",
        )


@requires_gpu
@lazy_params
def test_gaussian_source_size_matches_cpu_and_gpu(lazy):
    """gaussian_source_size hits the same GPU-only FFT code path as
    Images.gaussian_filter, but convolves along the (non-trailing) scan axes
    instead of the trailing two -- check CPU/GPU agreement there too.
    """
    rng = np.random.default_rng(0)
    array = rng.random((6, 5, 12, 12))
    if lazy:
        array = da.from_array(array, chunks=(2, 2, 12, 12))

    ensemble_axes_metadata = [
        ScanAxis(sampling=0.5, _main=True),
        ScanAxis(sampling=0.5, _main=True),
    ]
    measurement_cpu = DiffractionPatterns(
        array,
        sampling=0.1,
        ensemble_axes_metadata=ensemble_axes_metadata,
        metadata={"energy": 100e3},
    )
    measurement_gpu = measurement_cpu.to_gpu()

    cpu = measurement_cpu.gaussian_source_size(0.6)
    gpu_result = measurement_gpu.gaussian_source_size(0.6)
    np.testing.assert_allclose(
        cpu.compute().array,
        gpu_result.to_cpu().compute().array,
        atol=1e-5,
        rtol=1e-5,
    )


def test_gaussian_source_size_lazy_matches_eager_at_edges():
    """The lazy path must wrap periodically at the scan-grid edges, like the
    eager path (regression test for abTEM discussion 483)."""
    rng = np.random.default_rng(0)
    array = rng.random((12, 12, 4, 4))
    ensemble_axes_metadata = [
        ScanAxis(sampling=0.5, _main=True),
        ScanAxis(sampling=0.5, _main=True),
    ]

    def make(a):
        return DiffractionPatterns(
            a,
            sampling=0.1,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata={"energy": 100e3},
        )

    eager = make(array).gaussian_source_size(0.6).array
    lazy = (
        make(da.from_array(array, chunks=(6, 6, 4, 4)))
        .gaussian_source_size(0.6)
        .compute()
        .array
    )
    np.testing.assert_allclose(lazy, eager, atol=1e-10)


def test_lorentzian_filter_lazy():
    """Lorentzian filter works on a lazy (dask-backed) image."""
    images = _delta_probe_image(gpts=32, lazy=True)
    filtered = images.lorentzian_filter(0.5)
    assert np.isfinite(filtered.array.compute()).all()


def test_voigtian_filter_lazy():
    """Voigtian filter works on a lazy (dask-backed) image."""
    images = _delta_probe_image(gpts=32, lazy=True)
    filtered = images.voigtian_filter(0.3, 0.3)
    assert np.isfinite(filtered.array.compute()).all()


def test_pseudo_voigtian_filter_lazy():
    """Pseudo-Voigtian filter works on a lazy (dask-backed) image."""
    images = _delta_probe_image(gpts=32, lazy=True)
    filtered = images.pseudo_voigtian_filter(0.3, 0.3, eta=0.5)
    assert np.isfinite(filtered.array.compute()).all()


# @given(data=st.data())
# @pytest.mark.parametrize('lazy', [True, False])
# @pytest.mark.parametrize('device', ['cpu', gpu])
# def test_diffractograms(data, lazy, device):
#     measurement = data.draw(abtem_st.images(lazy=lazy, device=device))
#     measurement.diffractograms()


def _periodic_gaussian_blob(gpts, sampling, center, sigma):
    """exp(-|r - center|² / (2 sigma²)) on a periodic grid (minimum-image
    distance), peak 1."""
    extent = [n * d for n, d in zip(gpts, sampling)]
    coords = []
    for n, d, c, L in zip(gpts, sampling, center, extent):
        r = np.arange(n) * d - c
        coords.append(r - L * np.round(r / L))
    x, y = np.meshgrid(*coords, indexing="ij")
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))


def _make_images(array, sampling, lazy, device):
    array = array.astype(get_dtype(complex=False))
    if lazy:
        array = da.from_array(array, chunks=(array.shape[0] // 2, -1))
    images = Images(array, sampling=sampling)
    return images.to_gpu() if device == "gpu" else images


def _line_values(line):
    return np.asarray(asnumpy(line.compute().array), dtype=np.float64)


@lazy_params
@devices
def test_images_interpolate_line_through_grid_nodes(lazy, device):
    """A line running along a grid row/column, sampled at the grid spacing,
    samples only grid nodes, where spline interpolation is exact -- so the
    profile equals that row/column of the image. The image is anisotropic in
    shape and sampling so that an x/y mix-up changes the result."""
    rng = np.random.default_rng(7)
    gpts, sampling = (48, 64), (0.2, 0.15)
    array = rng.random(gpts)
    images = _make_images(array, sampling, lazy, device)
    extent = images.extent

    # Along y at x = 0 (the first row of the array).
    line = images.interpolate_line(start=(0, 0), end=(0, extent[1]), gpts=gpts[1])
    _assert_matches_reference(_line_values(line), array[0])

    # Along x at y = y_j (the j-th column).
    j = 17
    line = images.interpolate_line(
        start=(0, j * sampling[1]), end=(extent[0], j * sampling[1]), gpts=gpts[0]
    )
    _assert_matches_reference(_line_values(line), array[:, j])


@settings(max_examples=10, deadline=None)
@given(data=st.data())
@lazy_params
@devices
def test_images_interpolate_line_at_position(data, lazy, device):
    """A line of length L at any angle through the centre c of a rotationally
    symmetric Gaussian blob exp(-|r - c|²/2σ²) samples exp(-t²/2σ²),
    t = -L/2 ... L/2, independent of the angle (analytic profile)."""
    # Anisotropic sampling, so pixel/physical-unit mix-ups show; the line
    # may extend past the image edge, where the image is periodic.
    gpts, sampling, sigma, length, n = (128, 160), (0.1, 0.08), 0.8, 6.0, 121
    center = data.draw(
        st.tuples(
            *(
                st.floats(min_value=0, max_value=g * d, exclude_max=True)
                for g, d in zip(gpts, sampling)
            )
        ),
        label="center",
    )
    angle = data.draw(st.floats(min_value=0, max_value=360.0), label="angle")

    images = _make_images(
        _periodic_gaussian_blob(gpts, sampling, center, sigma), sampling, lazy, device
    )
    line = images.interpolate_line_at_position(
        center=center, angle=angle, extent=length, gpts=n, endpoint=True
    )

    t = np.linspace(-length / 2, length / 2, n)
    expected = np.exp(-(t**2) / (2 * sigma**2))
    # Cubic-spline interpolation of a Gaussian 8-10 pixels wide errs by
    # ~1e-4 of the peak; 1e-3 still rejects any off-centre or mis-rotated
    # line (a 0.1 Å offset changes the profile by ~1e-2 of the peak).
    _assert_matches_reference(_line_values(line), expected, rel=1e-3)

    # Averaging across a perpendicular width preserves the rotational
    # symmetry: the profile must not depend on the angle.
    width = data.draw(st.floats(min_value=0.1, max_value=2.0), label="width")
    other_angle = data.draw(st.floats(min_value=0, max_value=360.0), label="angle2")
    wide = [
        _line_values(
            images.interpolate_line_at_position(
                center=center, angle=a, extent=length, gpts=n, width=width
            )
        )
        for a in (angle, other_angle)
    ]
    _assert_matches_reference(wide[0], wide[1], rel=1e-3)


@devices
@pytest.mark.parametrize("order", [1, 3])
def test_interpolate_stack_just_below_grid_nodes(device, order):
    """Coordinates a hair below a grid node must give that node's value. With
    float64 coordinates on a float32 image, cupyx's spline kernel took the
    index from the float32-rounded coordinate and the weights from the
    float64 one, and returned the value at the next node instead."""
    rng = np.random.default_rng(11)
    array = rng.random((16, 24)).astype(np.float32)
    k = np.arange(1, 15)
    positions = np.stack([k - 1e-12, np.full(k.shape, 5.0)], axis=-1)
    if device == "gpu":
        cp = pytest.importorskip("cupy")
        values = _interpolate_stack(
            cp.asarray(array), cp.asarray(positions), mode="wrap", order=order
        )
    else:
        values = _interpolate_stack(array, positions, mode="wrap", order=order)
    np.testing.assert_allclose(asnumpy(values), array[k, 5], rtol=0, atol=1e-5)


def test_interpolate_line_lazy_matches_eager_with_ensemble_axis():
    """Regression: interpolate_line's lazy path computed drop_axis/new_axis
    as if the base (spatial) axes were the *first* axes of the array
    (range(len(base_shape))), rather than the actual trailing axes after any
    ensemble axis. This never raised -- it silently produced wrong output
    (extra, duplicated blocks) once an ensemble axis had more than one dask
    chunk, and silently wrong values once *both* base axes had more than one
    chunk each (reproduced via a DiffractionPatterns array whose spatial
    axes were split by a large zarr save/reload -- an all-zero
    momentum-resolved spectrum from the lazy load vs. a correct one from the
    eagerly-computed data)."""
    import dask.array as da

    from abtem.core.axes import EnergyLossAxis, ReciprocalSpaceAxis
    from abtem.measurements import DiffractionPatterns

    n_energy, gpts = 6, 64
    rng = np.random.default_rng(0)
    array = rng.random((n_energy, gpts, gpts))
    energies = tuple(float(e) for e in np.linspace(0.01, 0.1, n_energy))

    def make_dp(chunks):
        lazy = da.from_array(array, chunks=chunks)
        return DiffractionPatterns.from_array_and_metadata(
            lazy,
            axes_metadata=[
                EnergyLossAxis(values=energies),
                ReciprocalSpaceAxis(sampling=0.02, label="x", units="1/A"),
                ReciprocalSpaceAxis(sampling=0.02, label="y", units="1/A"),
            ],
            metadata={"energy": 100e3},
        )

    truth = make_dp((n_energy, gpts, gpts)).interpolate_line(
        start=(0.0, 0.0), end=(0.0, gpts * 0.02), gpts=20, width=0.3, order=1,
        endpoint=True,
    ).compute().array

    for name, chunks in [
        ("multi_chunk_ensemble_axis", (1, gpts, gpts)),
        ("both_base_axes_chunked", (n_energy, gpts // 2, gpts // 2)),
    ]:
        dp = make_dp(chunks)
        line = dp.interpolate_line(
            start=(0.0, 0.0), end=(0.0, gpts * 0.02), gpts=20, width=0.3, order=1,
            endpoint=True,
        ).compute()
        assert line.array.shape == truth.shape, name
        np.testing.assert_allclose(line.array, truth, err_msg=name)


@given(
    data=st.data(), dose_per_area=abtem_st.sensible_floats(min_value=1e8, max_value=1e9)
)
@lazy_params
@devices
@pytest.mark.parametrize("measurement", [abtem_st.images])
def test_poisson_noise(data, measurement, dose_per_area, lazy, device):
    measurement = data.draw(
        measurement(lazy=lazy, device=device, min_value=0.5, min_base_side=16)
    )

    assume(isinstance(measurement, Images) or len(_scan_shape(measurement)) == 2)
    measurement = measurement.no_base_chunks()
    noisy = measurement.poisson_noise(dose_per_area=dose_per_area, samples=16).compute()

    if isinstance(measurement, Images):
        area = np.prod(measurement.extent)
        expected_total_dose = area * dose_per_area * np.prod(measurement.ensemble_shape)
        actual_total_dose = (noisy.array.mean(axis=0)).sum() / measurement.array.mean()
    else:
        area = np.prod(measurement.scan_extent)
        expected_total_dose = (
            area * dose_per_area * np.prod(measurement.ensemble_shape[:-2])
        )
        actual_total_dose = (
            noisy.array.mean(axis=0) / measurement.array.sum((-2, -1), keepdims=True)
        ).sum()

    expected_total_dose = copy_to_device(expected_total_dose, "cpu")
    actual_total_dose = copy_to_device(actual_total_dose, "cpu")

    assert np.allclose(expected_total_dose, actual_total_dose, rtol=0.1)


@given(data=st.data())
@devices
def test_diffraction_patterns_polar_binning(data, device):
    """The lazy polar_binning must compute the same bins as the eager one."""
    measurement = data.draw(
        abtem_st.diffraction_patterns(lazy=True, device=device, min_base_side=16)
    )

    nbins_radial = data.draw(
        st.integers(min_value=1, max_value=min(measurement.base_shape))
    )

    nbins_azimuthal = data.draw(
        st.integers(min_value=1, max_value=min(measurement.base_shape))
    )

    outer = data.draw(
        abtem_st.sensible_floats(
            min_value=min(
                min(measurement.max_angles), max(measurement.angular_sampling)
            ),
            max_value=min(measurement.max_angles),
        )
    )

    inner = data.draw(
        abtem_st.sensible_floats(
            min_value=0.0, max_value=max(0.0, outer - max(measurement.angular_sampling))
        )
    )

    rotation = data.draw(abtem_st.sensible_floats(min_value=0.0, max_value=360.0))
    kwargs = dict(
        nbins_radial=nbins_radial,
        nbins_azimuthal=nbins_azimuthal,
        inner=inner,
        outer=outer,
        rotation=rotation,
    )

    lazy = measurement.polar_binning(**kwargs)
    assert lazy.is_lazy
    lazy = lazy.compute()
    eager = measurement.compute().polar_binning(**kwargs)

    assert isinstance(lazy, PolarMeasurements)
    assert lazy.shape == measurement.ensemble_shape + (nbins_radial, nbins_azimuthal)
    np.testing.assert_allclose(
        asnumpy(lazy.array), asnumpy(eager.array), rtol=1e-6, atol=0
    )


@given(data=st.data())
@lazy_params
@devices
def test_diffraction_patterns_center_of_mass(data, lazy, device):
    # Property: the center of mass is a physical quantity, so it cannot depend on
    # the storage order of the pattern. Re-storing the same pattern in the other
    # order (centred <-> unshifted; unshifted is by definition the ifftshift of
    # centred) must give the same center of mass, in both units.
    measurement = data.draw(
        abtem_st.diffraction_patterns(
            lazy=lazy, min_scan_dims=1, device=device, min_base_side=16
        )
    )
    assume(len(_scan_sampling(measurement)) > 0)

    array = measurement.compute().array
    xp = get_array_module(array)
    if measurement.fftshift:
        reordered = xp.fft.ifftshift(array, axes=(-2, -1))
    else:
        reordered = xp.fft.fftshift(array, axes=(-2, -1))

    other = DiffractionPatterns(
        reordered,
        sampling=measurement.sampling,
        ensemble_axes_metadata=measurement.ensemble_axes_metadata,
        metadata=measurement.metadata,
        fftshift=not measurement.fftshift,
    )

    angular_limits = measurement.angular_limits
    for units, scale in (
        ("1/Å", max(abs(k) for limit in measurement.limits for k in limit)),
        ("mrad", max(abs(a) for limit in angular_limits for a in limit)),
    ):
        com = asnumpy(measurement.center_of_mass(units=units).compute().array)
        other_com = asnumpy(other.center_of_mass(units=units).array)
        # Only the float32 summation order differs between the two (~1e-6 of
        # the scale); a storage-order bug displaces the COM by whole pixels,
        # i.e. by multiples of 1 / (n // 2) >= 1 / 16 of the scale (the
        # strategy's base side is <= 32).
        np.testing.assert_allclose(com, other_com, rtol=0, atol=1e-4 * scale)


def _delta_diffraction_patterns(gpts, sampling, fftshift, index, device):
    # A unit delta at integer frequency index `index` (fftfreq-style: negative
    # values count from the end), i.e. at k = index * sampling. In centred storage
    # the zero frequency sits at pixel n // 2 (np.fft.fftshift convention); in
    # unshifted storage it sits at pixel 0 (np.fft.fftfreq convention).
    array = np.zeros(gpts, dtype=np.float32)
    if fftshift:
        array[gpts[0] // 2 + index[0], gpts[1] // 2 + index[1]] = 1.0
    else:
        array[index[0] % gpts[0], index[1] % gpts[1]] = 1.0

    return DiffractionPatterns(
        copy_to_device(array, device),
        sampling=sampling,
        fftshift=fftshift,
        metadata={"energy": 100e3},
    )


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("units", ["1/Å", "mrad"])
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize(
    "gpts, sampling",
    [
        ((32, 32), (0.1, 0.1)),
        ((33, 33), (0.1, 0.1)),
        # Non-square grid and anisotropic sampling: an x/y swap of the shape or
        # of the sampling changes the answer.
        ((32, 33), (0.1, 0.12)),
    ],
)
def test_diffraction_patterns_center_of_mass_of_delta(
    device, units, fftshift, gpts, sampling
):
    # The COM of a single delta is its own position: k = (3 * dkx, -2 * dky). In
    # mrad, alpha = lambda * k * 1e3 (the convention of angular_sampling).
    index = (3, -2)
    measurement = _delta_diffraction_patterns(gpts, sampling, fftshift, index, device)

    expected = index[0] * sampling[0] + 1.0j * index[1] * sampling[1]
    if units == "mrad":
        expected = expected * energy2wavelength(100e3) * 1e3

    com = complex(asnumpy(measurement.center_of_mass(units=units).array))

    # One nonzero pixel: the only error is float32 rounding of the coordinate
    # (~6e-8 relative); 1e-6 is ~10 float32 ulps. A one-pixel error is >= 1/3.6
    # relative.
    assert com == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("units", ["1/Å", "mrad"])
@pytest.mark.parametrize("gpts", [(32, 33), (33, 32)])
def test_diffractogram_of_plane_wave_peaks_at_its_frequency(device, units, gpts):
    # |FFT|^2 of exp(2 pi i (kx x + ky y)), with k on the DFT grid, is a single
    # delta at +k (numpy's forward-FFT sign convention), so the diffractogram's
    # center of mass is k itself, and lambda * k * 1e3 in mrad.
    sampling = (0.2, 0.25)
    kx = 3 / (gpts[0] * sampling[0])
    ky = -2 / (gpts[1] * sampling[1])
    x = np.arange(gpts[0]) * sampling[0]
    y = np.arange(gpts[1]) * sampling[1]
    image = np.exp(2j * np.pi * (kx * x[:, None] + ky * y[None])).astype(np.complex64)

    images = Images(
        copy_to_device(image, device), sampling=sampling, metadata={"energy": 100e3}
    )
    com = complex(asnumpy(images.diffractograms().center_of_mass(units=units).array))

    expected = kx + 1.0j * ky
    if units == "mrad":
        expected = expected * energy2wavelength(100e3) * 1e3

    # float32 FFT leakage off the peak is ~1e-7 relative; a one-pixel error is
    # >= 1 / 3.6 relative.
    assert com == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("gpts", [(32, 32), (33, 33), (32, 33)])
def test_diffraction_patterns_coordinates_match_fftfreq(device, fftshift, gpts):
    # The frequencies of an n-point DFT with real-space spacing d are
    # np.fft.fftfreq(n, d) in unshifted order, and np.fft.fftshift of that in
    # centred order. A reciprocal sampling dk corresponds to d = 1 / (n * dk).
    sampling = (0.1, 0.12)
    measurement = DiffractionPatterns(
        copy_to_device(np.zeros(gpts, dtype=np.float32), device),
        sampling=sampling,
        fftshift=fftshift,
        metadata={"energy": 100e3},
    )
    wavelength_mrad = energy2wavelength(100e3) * 1e3

    for i in range(2):
        expected = np.fft.fftfreq(gpts[i], d=1 / (gpts[i] * sampling[i]))
        if fftshift:
            expected = np.fft.fftshift(expected)

        # float64 coordinates: 1e-12 is ~1e4 ulps of |k| <= 2 1/Å.
        np.testing.assert_allclose(
            measurement.coordinates[i], expected, rtol=0, atol=1e-12
        )
        # Angular coordinates follow the configured precision (float32 by
        # default, ~6e-8 relative); 1e-6 of a pixel is far below a 1-pixel error.
        np.testing.assert_allclose(
            asnumpy(measurement.angular_coordinates[i]),
            expected * wavelength_mrad,
            rtol=1e-6,
            atol=1e-6 * wavelength_mrad * sampling[i],
        )


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("units", ["1/Å", "mrad"])
@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("gpts", [64, 65])
@pytest.mark.parametrize("captured_fraction", [1.0, 0.5, 0.1])
def test_diffraction_patterns_center_of_mass_is_normalized(
    device, units, fftshift, gpts, captured_fraction
):
    # A center of mass is a normalized (intensity-weighted average) quantity, so
    # scaling the total intensity of a diffraction pattern must not change the
    # computed center of mass. This did not hold before the sum was normalized by
    # the total captured intensity: https://github.com/abTEM/abTEM/discussions/402
    #
    # A Gaussian sampled symmetrically about pixel (n // 2 + shift) has its COM
    # exactly at that pixel, i.e. at k = shift * sampling, provided it is neither
    # truncated nor wrapped: the nearest edge is >= 32 - 10 = 22 px = 7.3 sigma
    # away, where the tail is exp(-7.3**2 / 2) ~ 3e-12.
    sampling = 0.4436
    sigma = 3.0
    shift = (-10, 7)

    i, j = np.mgrid[0:gpts, 0:gpts]
    disk = np.exp(
        -((i - gpts // 2 - shift[0]) ** 2 + (j - gpts // 2 - shift[1]) ** 2)
        / (2 * sigma**2)
    )
    if not fftshift:
        # Unshifted storage is by definition the ifftshift of centred storage.
        disk = np.fft.ifftshift(disk)
    disk = copy_to_device(
        (disk / disk.sum() * captured_fraction).astype(np.float32), device
    )

    measurement = DiffractionPatterns(
        disk, sampling=sampling, fftshift=fftshift, metadata={"energy": 100e3}
    )

    com = complex(asnumpy(measurement.center_of_mass(units=units).array))

    expected = (shift[0] * sampling) + 1.0j * (shift[1] * sampling)
    if units == "mrad":
        expected = expected * energy2wavelength(100e3) * 1e3

    # float32 sums over ~4e3 pixels: ~1e-6 relative. A one-pixel error is
    # 1 / |shift| = 1 / 12.2 ~ 8% relative.
    assert abs(com - expected) < 1e-4 * abs(expected)


@given(data=st.data())
@lazy_params
@devices
def test_diffraction_patterns_integrated_center_of_mass(data, lazy, device):
    measurement = data.draw(
        abtem_st.diffraction_patterns(
            lazy=lazy, min_scan_dims=1, device=device, min_base_side=16
        )
    )
    assume(len(_scan_sampling(measurement)) > 1)
    measurement.integrated_center_of_mass().compute()


@given(data=st.data())
@lazy_params
@devices
def test_diffraction_patterns_bandlimit(data, lazy, device):
    measurement = data.draw(
        abtem_st.diffraction_patterns(lazy=lazy, device=device, min_base_side=16)
    )
    outer = data.draw(
        abtem_st.sensible_floats(min_value=0.0, max_value=min(measurement.max_angles))
    )
    inner = data.draw(abtem_st.sensible_floats(min_value=0.0, max_value=outer))
    measurement.bandlimit(inner, outer).compute()
    measurement.block_direct().compute()


@settings(deadline=None, max_examples=10)
@given(data=st.data(), sigma=_sigma_strategy(max_value=2.0))
@lazy_params
@devices
def test_diffraction_patterns_gaussian_source_size(data, sigma, lazy, device):
    measurement = data.draw(
        abtem_st.diffraction_patterns(
            lazy=lazy, min_scan_dims=2, device=device, min_base_side=16
        )
    )
    assume(len(_scan_sampling(measurement)) > 1)
    measurement.gaussian_source_size(sigma).compute()


@settings(suppress_health_check=(HealthCheck.data_too_large,))
@given(data=st.data())
@lazy_params
@devices
def test_polar_measurements_integrate(data, lazy, device):
    measurement = data.draw(abtem_st.polar_measurements(lazy=lazy, device=device))
    assume(len(_scan_shape(measurement)) > 0)

    radial_outer = data.draw(
        abtem_st.sensible_floats(min_value=0.0, max_value=measurement.outer_angle)
    )
    radial_inner = data.draw(
        abtem_st.sensible_floats(min_value=0.0, max_value=radial_outer)
    )
    radial_limits = data.draw(
        st.one_of(st.just((radial_inner, radial_outer)), st.none())
    )

    # Azimuthal limits are in radians.
    azimuthal_outer = data.draw(
        abtem_st.sensible_floats(min_value=0.0, max_value=2 * np.pi)
    )
    azimuthal_inner = data.draw(
        abtem_st.sensible_floats(min_value=0.0, max_value=azimuthal_outer)
    )
    azimuthal_limits = data.draw(
        st.one_of(st.just((azimuthal_inner, azimuthal_outer)), st.none())
    )

    measurement.integrate(
        radial_limits=radial_limits, azimuthal_limits=azimuthal_limits
    ).compute()
    measurement.integrate_radial(radial_inner, radial_outer).compute()

    max_region = int(np.prod(tuple(n - 1 for n in measurement.shape[-2:])))
    detector_regions = st.lists(
        min_size=0,
        max_size=max_region,
        elements=st.integers(min_value=0, max_value=max_region),
        unique=True,
    )

    measurement.integrate(detector_regions=data.draw(detector_regions)).compute()


@given(data=st.data())
@lazy_params
@devices
def test_line_profiles_interpolate(data, lazy, device):
    measurement = data.draw(abtem_st.line_profiles(lazy=lazy, device=device))
    measurement.interpolate().compute()


@given(data=st.data(), reps=st.integers(min_value=1, max_value=3))
@lazy_params
@devices
def test_line_profiles_tile(data, reps, lazy, device):
    measurement = data.draw(abtem_st.line_profiles(lazy=lazy, device=device))
    measurement.tile(reps).compute()


def test_images_interpolate_to_a_sampling_that_divides_the_extent():
    # 10.8 / 0.3 is 36.00000000000001 in floats; the images still have 36 pixels.
    images = Images(np.random.default_rng(0).random((10, 5)), sampling=1.08)
    assert images.interpolate(sampling=0.3).shape == (36, 18)


def test_line_profiles_interpolate_to_a_sampling_that_divides_the_extent():
    profiles = RealSpaceLineProfiles(
        np.random.default_rng(0).random(10), sampling=1.08
    )
    assert profiles.interpolate(sampling=0.3).shape == (36,)


@lazy_params
def test_line_profiles_interpolate_comparison(lazy):
    atoms = ase.build.bulk("Si", cubic=True)
    images = abtem.PlaneWave(energy=100e3, sampling=0.05).multislice(atoms).intensity()

    if not lazy:
        images.compute()

    assert np.allclose(
        images.interpolate_line().interpolate(0.01).array,
        images.interpolate(0.01).interpolate_line().array,
        rtol=0.01,
    )


@lazy_params
def test_interpolate_periodic_spline_and_fft(lazy):
    atoms = ase.build.bulk("Si", cubic=True)
    images = abtem.PlaneWave(energy=100e3, sampling=0.05).multislice(atoms).intensity()

    if not lazy:
        images.compute()

    spline_interpolated = images.interpolate(
        method="spline", sampling=0.05, boundary="periodic", order=5
    )
    fft_interpolated = images.interpolate(method="fft", sampling=0.05)
    assert array_is_close(
        spline_interpolated.array, fft_interpolated.array, rel_tol=0.01
    )


def _periodic_spline_interpolate(array, gpts, order, lazy, device):
    images = abtem.Images(copy_to_device(array, device), sampling=0.1)
    if lazy:
        images = images.lazy()
    interpolated = images.interpolate(
        gpts=gpts, method="spline", boundary="periodic", order=order
    )
    return asnumpy(interpolated.compute().array)


@devices
@lazy_params
@pytest.mark.parametrize("order", [2, 3])
def test_periodic_spline_interpolation_is_invariant_to_whole_pixel_rolls(
    lazy, order, device
):
    # A periodic interpolant commutes with rolling the image by whole pixels. The
    # image has 40 x 30 pixels and is interpolated to 80 x 90, so one old pixel is
    # 2 new pixels along x and 3 along y and the rolled output is a whole-pixel roll.
    array = np.random.default_rng(0).random((40, 30))

    interpolated_roll = _periodic_spline_interpolate(
        np.roll(array, (5, 7), axis=(0, 1)), (80, 90), order, lazy, device
    )
    rolled_interpolation = np.roll(
        _periodic_spline_interpolate(array, (80, 90), order, lazy, device),
        (10, 21),
        axis=(0, 1),
    )

    # The two sides agree to within 20 eps of the data scale in the dtype the device
    # stores (float64 on the CPU, float32 on Metal and torch); the tolerance of 200 eps
    # leaves a margin of 10 over that, and a roll that is off by one pixel differs by
    # about 0.5 of the data scale.
    eps = np.finfo(interpolated_roll.dtype).eps
    np.testing.assert_allclose(
        interpolated_roll, rolled_interpolation, rtol=0, atol=200 * eps * array.max()
    )


@devices
@lazy_params
@pytest.mark.parametrize("order", [2, 3])
def test_periodic_spline_interpolation_reproduces_a_band_limited_field(
    lazy, order, device
):
    # Interpolating a periodic field with a few Fourier components from 40 x 30 to
    # 53 x 41 points (a factor that is not an integer) must give the field itself at
    # the new positions j * extent / gpts. A constant shift of the coordinates, which
    # a roll test does not see, moves the result by up to 0.2 of the amplitude for a
    # shift of half a pixel. The spline's own error is 9e-4 at order 2 and 2e-4 at
    # order 3.
    extent = (4.0, 3.0)
    gpts, new_gpts = (40, 30), (53, 41)

    def field(x, y):
        return np.cos(2 * np.pi * (2 * x / extent[0] + y / extent[1] + 0.1)) + 0.5 * (
            np.sin(2 * np.pi * (x / extent[0] - 3 * y / extent[1]))
        )

    def sample(points):
        axes = [np.arange(n) * length / n for n, length in zip(points, extent)]
        return field(*np.meshgrid(*axes, indexing="ij"))

    sampling = tuple(e / n for e, n in zip(extent, gpts))
    images = abtem.Images(copy_to_device(sample(gpts), device), sampling=sampling)
    if lazy:
        images = images.lazy()
    interpolated = images.interpolate(
        gpts=new_gpts, method="spline", boundary="periodic", order=order
    ).compute()

    expected = sample(new_gpts)
    np.testing.assert_allclose(
        asnumpy(interpolated.array),
        expected,
        rtol=0,
        atol=2e-3 * np.abs(expected).max(),
    )


@given(
    gpts=st.integers(min_value=16, max_value=32),
    extent=st.floats(min_value=5, max_value=10),
)
def test_diffraction_patterns_interpolate_uniform(gpts, extent):
    probe = Probe(
        energy=100e3, semiangle_cutoff=20, extent=extent, gpts=gpts, soft=False
    )
    diffraction_patterns = probe.build().diffraction_patterns(max_angle=None)
    probe.gpts = (gpts * 2, gpts)
    probe.extent = (extent * 2, extent)
    interpolated_diffraction_patterns = (
        probe.build().diffraction_patterns(max_angle=None).interpolate("uniform")
    )
    assert np.allclose(
        interpolated_diffraction_patterns.array, diffraction_patterns.array
    )


_DISC_SIGMA = 1.0
_DISC_SAMPLING = (0.02, 0.03)
_DISC_GPTS = (500, 333)


@given(
    position=st.tuples(
        st.floats(min_value=0.0, max_value=_DISC_GPTS[0] * _DISC_SAMPLING[0]),
        st.floats(min_value=0.0, max_value=_DISC_GPTS[1] * _DISC_SAMPLING[1]),
    ),
    radius=st.floats(min_value=0.5 * _DISC_SIGMA, max_value=3.0 * _DISC_SIGMA),
)
def test_integrate_disc(position, radius):
    """A disc of radius R centred on a 2-D Gaussian blob of width σ (anywhere
    in the periodic image, including across its border) captures the
    fraction 1 - exp(-R²/2σ²) of the blob's total.

    Tolerance: integrate_disc anti-aliases the disc edge with a linear ramp
    one mean pixel (d = 0.025 Å) wide, which can grow the effective radius by
    at most d/2; the fraction then changes by at most
    (d/2) max_R dF/dR = (d/2σ) e^(-1/2) ≈ 0.30 d/σ = 0.0075 (σ = 1 Å). The
    sampling is anisotropic, so an x/y sampling mix-up misplaces the disc.
    """
    array = _periodic_gaussian_blob(_DISC_GPTS, _DISC_SAMPLING, position, _DISC_SIGMA)
    array /= array.sum()
    measurement = Images(array, sampling=_DISC_SAMPLING)

    captured = measurement.integrate_disc(position=position, radius=radius)

    expected = 1 - np.exp(-(radius**2) / (2 * _DISC_SIGMA**2))
    tolerance = 0.30 * np.mean(_DISC_SAMPLING) / _DISC_SIGMA
    assert abs(captured - expected) < tolerance


# @given(sigma=st.floats(min_value=.1, max_value=.5),
#        outer=st.floats(min_value=10., max_value=100))
# def test_gaussian_source_size_order(sigma, outer):
#     diffraction_patterns = from_zarr('data/silicon_diffraction_patterns.zarr').compute()
#     image1 = diffraction_patterns.gaussian_source_size(sigma).integrate_radial(0, outer)
#     image2 = diffraction_patterns.integrate_radial(0, outer).gaussian_filter(sigma)
#     assert np.allclose(image1.array, image2.array)


# ---------------------------------------------------------------------------
# Images — crop, complex accessors, abs, scan_noise, normalize_ensemble
# ---------------------------------------------------------------------------

def make_images(shape=(32, 32), sampling=(0.1, 0.1), value=None, complex_=False):
    """Build a bare ``Images`` object for tests that only need *some* image.

    ``value`` fills the array with a constant (as several noise/transform
    tests need); otherwise a reproducible random array is used, optionally
    complex-valued.
    """
    if value is not None:
        arr = np.full(shape, value, dtype=float)
    else:
        arr = np.random.default_rng(0).random(shape)
        if complex_:
            arr = arr + 1j * np.random.default_rng(1).random(shape)
    return Images(arr, sampling=sampling)


class TestImagesCrop:
    # Anisotropic shape and sampling, extent (3.2, 2.0) Å, so that x/y
    # mix-ups in the crop region change the result. Every crop below is an
    # exact whole number of pixels, so the expected region is unambiguous:
    # pixel i covers [i d, (i + 1) d).
    _shape, _sampling = (32, 40), (0.1, 0.05)

    def _images(self):
        return make_images(self._shape, self._sampling)

    def test_crop_reduces_extent(self):
        imgs = self._images()
        cropped = imgs.crop((1.5, 1.0))
        # 1.5 Å / 0.1 Å = 15 and 1.0 Å / 0.05 Å = 20 pixels from the origin.
        assert cropped.base_shape == (15, 20)
        assert np.allclose(cropped.extent, (1.5, 1.0))
        assert np.allclose(cropped.sampling, imgs.sampling)
        np.testing.assert_array_equal(cropped.array, imgs.array[:15, :20])

    def test_crop_centered(self):
        imgs = self._images()
        cropped = imgs.crop((1.2, 1.0), centered=True)
        # Centred: lower corner at extent/2 - crop/2 = (1.0, 0.5) Å, i.e.
        # pixel (10, 10); 12 x 20 pixels.
        assert cropped.base_shape == (12, 20)
        np.testing.assert_array_equal(cropped.array, imgs.array[10:22, 10:30])

    def test_crop_too_large_raises(self):
        imgs = make_images((32, 32), (0.1, 0.1))
        with pytest.raises(ValueError, match="smaller"):
            imgs.crop((999.0, 999.0))

    def test_crop_centered_with_offset_raises(self):
        imgs = make_images((32, 32), (0.1, 0.1))
        with pytest.raises(ValueError):
            imgs.crop((1.0, 1.0), offset=(0.1, 0.1), centered=True)

    def test_crop_with_offset(self):
        imgs = self._images()
        cropped = imgs.crop((1.5, 1.0), offset=(0.5, 0.3))
        # Lower corner (0.5, 0.3) Å = pixel (5, 6); 15 x 20 pixels.
        assert cropped.base_shape == (15, 20)
        np.testing.assert_array_equal(cropped.array, imgs.array[5:20, 6:26])


class TestImagesComplexAccessors:
    def test_real(self):
        imgs = make_images(complex_=True)
        real = imgs.real()
        assert not np.iscomplexobj(real.array)
        assert np.allclose(real.array, imgs.array.real)

    def test_imag(self):
        imgs = make_images(complex_=True)
        imag = imgs.imag()
        assert np.allclose(imag.array, imgs.array.imag)

    def test_phase(self):
        imgs = make_images(complex_=True)
        phase = imgs.phase()
        assert np.all(np.abs(phase.array) <= np.pi + 1e-10)

    def test_abs(self):
        imgs = make_images(complex_=True)
        ab = imgs.abs()
        assert np.all(ab.array >= 0)

    def test_real_on_real_raises(self):
        imgs = make_images(complex_=False)
        with pytest.raises(RuntimeError):
            imgs.real()


class TestImagesNormalizeEnsemble:
    # Two members, the second an affine transform (x -> 10 x + 5) of the first.
    _member = np.array([[1.0, 2.0], [3.0, 7.0]])
    _arr = np.stack([_member, 10 * _member + 5])

    def _images(self):
        return Images(
            self._arr,
            sampling=(0.1, 0.1),
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        )

    def test_normalize_reduces_spread(self):
        """Shifting by the min and scaling by the peak-to-peak range removes any
        per-member offset and scale: both members map onto the same image,
        spanning exactly [0, 1]."""
        normalized = self._images().normalize_ensemble(scale="ptp", shift="min")
        member = self._member
        # Per member (the whole 2-D image, not each row of it).
        expected = (member - member.min()) / (member.max() - member.min())
        np.testing.assert_allclose(normalized.array[0], expected)
        np.testing.assert_allclose(normalized.array[1], expected)

    def test_normalize_default_mean_max(self):
        """The defaults shift each member by its mean and divide by its max
        (evaluated before shifting): the members' means become zero."""
        normalized = self._images().normalize_ensemble()
        for original, result in zip(self._arr, normalized.array):
            np.testing.assert_allclose(
                result, (original - original.mean()) / original.max()
            )
            assert abs(result.mean()) < 1e-12

    @lazy_params
    @devices
    @pytest.mark.parametrize("scale, shift", [("max", "mean"), ("max", "min")])
    def test_matches_double_precision_oracle(self, scale, shift, lazy, device):
        """Each member of a (3, 5, 7) ensemble, chunked along both base axes when
        lazy, normalizes as its double-precision NumPy reduction says."""
        array = (np.random.default_rng(0).random((3, 5, 7)) + 0.5).astype(
            np.float32
        )
        images = Images(
            da.from_array(array, chunks=(1, 3, 4)) if lazy else array,
            sampling=0.1,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1, 2))],
        ).copy_to_device(device)

        normalized = images.normalize_ensemble(scale=scale, shift=shift)

        assert normalized.is_lazy == lazy
        computed = normalized.compute().array
        assert_array_matches_device(computed, device)
        reference = array.astype(np.float64)
        expected = (
            reference - getattr(np, shift)(reference, axis=(1, 2), keepdims=True)
        ) / getattr(np, scale)(reference, axis=(1, 2), keepdims=True)
        np.testing.assert_allclose(
            asnumpy(computed), expected, rtol=0, atol=1e-6 * np.abs(expected).max()
        )

    def test_normalize_line_profiles_per_profile(self):
        """For 1-D members the reduction runs along the single base axis."""
        arr = np.array([[1.0, 3.0, 5.0], [2.0, 2.0, 8.0]])
        profiles = RealSpaceLineProfiles(
            arr, sampling=0.1, ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))]
        )
        normalized = profiles.normalize_ensemble(scale="ptp", shift="min")
        np.testing.assert_allclose(normalized.array, [[0, 0.5, 1], [0, 0, 1]])


class TestImagesScanNoise:
    def test_scan_noise_returns_images(self):
        imgs = make_images((16, 16))
        result = imgs.scan_noise(
            rms_power=1.0, dwell_time=1e-6, flyback_time=1e-4,
            num_components=5
        ).compute()
        assert isinstance(result, Images)

    def test_scan_noise_shape_preserved(self):
        imgs = make_images((16, 16))
        result = imgs.scan_noise(1.0, 1e-6, 1e-4, num_components=5).compute()
        assert result.base_shape == imgs.base_shape


class TestImagesRelativeDifference:
    def test_zero_difference(self):
        imgs = make_images()
        diff = imgs.relative_difference(imgs.copy())
        assert np.allclose(diff.array[np.isfinite(diff.array)], 0.0, atol=1e-10)

    def test_wrong_type_raises(self):
        imgs = make_images()
        dp = DiffractionPatterns(
            np.ones((8, 8)), sampling=0.1, metadata={"energy": 100e3}
        )
        with pytest.raises(RuntimeError):
            imgs.relative_difference(dp)


# ---------------------------------------------------------------------------
# DiffractionPatterns — integrate_radial, crop, poisson_noise with samples
# ---------------------------------------------------------------------------

def _dp(shape=(32, 32), fill=1.0):
    return DiffractionPatterns(
        np.full(shape, fill), sampling=0.05, metadata={"energy": 100e3}
    )


class TestDiffractionPatternsIntegrateRadial:
    def test_returns_images_with_scan_axes(self):
        arr = np.ones((4, 4, 16, 16))
        dp = DiffractionPatterns(
            arr, sampling=0.05,
            ensemble_axes_metadata=[ScanAxis(), ScanAxis()],
            metadata={"energy": 100e3},
        )
        result = dp.integrate_radial(inner=0, outer=10)
        assert isinstance(result, Images)

    def test_inner_equals_outer_zero_result(self):
        dp = _dp()
        result = dp.integrate_radial(inner=5, outer=5)
        assert np.all(result.array == 0.0)

    def test_larger_outer_gives_larger_sum(self):
        dp = _dp()
        r1 = dp.integrate_radial(0, 5)
        r2 = dp.integrate_radial(0, 10)
        assert r2.array.sum() >= r1.array.sum()


def _frequency_index_positions(n, fftshift):
    """Map integer spatial-frequency index -> array position, in numpy's
    fftfreq convention (shifted: zero frequency at n // 2)."""
    freqs = np.fft.fftfreq(n, 1 / n)
    if fftshift:
        freqs = np.fft.fftshift(freqs)
    return {int(round(f)): i for i, f in enumerate(freqs)}


class TestDiffractionPatternsCrop:
    @pytest.mark.parametrize("fftshift", [True, False])
    def test_crop_reduces_max_angle(self, fftshift):
        """Cropping to max_angle keeps exactly the frequencies |m| dα <= max_angle
        along each axis (an odd grid centred on zero frequency), with their
        original values and unchanged sampling."""
        shape, sampling, energy = (64, 48), (0.05, 0.08), 100e3
        arr = np.random.default_rng(3).random(shape)
        dp = DiffractionPatterns(
            arr, sampling=sampling, fftshift=fftshift, metadata={"energy": energy}
        )
        # Anisotropic sampling: the same angle is a different number of
        # frequency steps along x and y.
        angular_sampling = [d * energy2wavelength(energy) * 1e3 for d in sampling]
        max_angle = 10.2 * angular_sampling[0]  # 10 steps along x, 6 along y
        n_max = [int(np.round(max_angle / a)) for a in angular_sampling]
        assert n_max == [10, 6]

        cropped = dp.crop(max_angle=max_angle)

        assert cropped.shape == (2 * n_max[0] + 1, 2 * n_max[1] + 1)
        assert np.allclose(cropped.sampling, dp.sampling)
        assert cropped.fftshift == fftshift

        # Each kept frequency (m_x, m_y) must hold the original value at that
        # frequency.
        old = [_frequency_index_positions(n, fftshift) for n in shape]
        new = [_frequency_index_positions(n, fftshift) for n in cropped.shape]
        assert sorted(new[0]) == list(range(-n_max[0], n_max[0] + 1))
        assert sorted(new[1]) == list(range(-n_max[1], n_max[1] + 1))
        expected = np.zeros(cropped.shape)
        for mx, i in new[0].items():
            for my, j in new[1].items():
                expected[i, j] = arr[old[0][mx], old[1][my]]
        np.testing.assert_array_equal(cropped.array, expected)


class TestDiffractionPatternsPoisson:
    def test_poisson_with_samples(self):
        dp = _dp((16, 16), fill=100.0)
        noisy = dp.poisson_noise(total_dose=1e6, samples=4).compute()
        assert noisy.shape[0] == 4

    def test_poisson_nonnegative(self):
        dp = _dp((16, 16), fill=50.0)
        noisy = dp.poisson_noise(total_dose=1e5).compute()
        assert np.all(noisy.array >= 0)


# ---------------------------------------------------------------------------
# RealSpaceLineProfiles
# ---------------------------------------------------------------------------

class TestRealSpaceLineProfiles:
    def _lp(self, n=64):
        from abtem.core.axes import RealSpaceAxis
        arr = np.ones(n)
        return RealSpaceLineProfiles(arr, sampling=0.1)

    def test_construction(self):
        lp = self._lp()
        assert lp.base_shape == (64,)

    def test_extent(self):
        lp = self._lp(32)
        # RealSpaceLineProfiles.extent is a scalar float, not a tuple
        assert np.isclose(lp.extent, 32 * 0.1)

    def test_interpolate(self):
        lp = self._lp()
        result = lp.interpolate(sampling=0.05)
        assert result.base_shape[0] > lp.base_shape[0]

    def test_tile(self):
        lp = self._lp(16)
        tiled = lp.tile(3)
        assert tiled.base_shape[0] == 48

    def test_sum_axis(self):
        arr = np.ones((4, 32))
        from abtem.core.axes import OrdinalAxis
        lp = RealSpaceLineProfiles(
            arr, sampling=0.1,
            ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(4)))]
        )
        result = lp.sum(axis=0)
        assert result.base_shape == (32,)


# ---------------------------------------------------------------------------
# PolarMeasurements — integrate, integrate_radial
# ---------------------------------------------------------------------------

class TestPolarMeasurements:
    # Base shape (3 radial, 4 azimuthal). With the default geometry below, radial
    # bin i spans [5 i, 5 i + 5) mrad and azimuthal bin j spans
    # [j pi / 2, (j + 1) pi / 2) rad, i.e. the bins of
    # DiffractionPatterns.polar_binning(nbins_radial=3, nbins_azimuthal=4,
    # inner=0, outer=15). Limits select the bins whose centers lie in
    # [lower, upper), with the azimuth periodic in 2 pi.
    #
    # The non-uniform array is A[i, j] = 10 i + j, so
    #   row sums    R_i = sum_j A[i, j] = 40 i + 6        -> R = (6, 46, 86)
    #   column sums C_j = sum_i A[i, j] = 30 + 3 j        -> C = (30, 33, 36, 39)
    #   total       = 6 + 46 + 86 = 138
    # The second scan position holds 2 A, so every expected value doubles there.

    def _polar(
        self,
        values="indexed",
        radial_sampling=5.0,
        radial_offset=0.0,
        azimuthal_offset=0.0,
        lazy=False,
        device="cpu",
    ):
        i, j = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
        base = np.ones((3, 4)) if values == "ones" else 10.0 * i + j
        arr = np.stack([base, 2 * base])
        arr = copy_to_device(arr, device)
        if lazy:
            arr = da.from_array(arr, chunks=(1, -1, -1))
        return PolarMeasurements(
            arr,
            radial_sampling=radial_sampling,
            azimuthal_sampling=2 * np.pi / 4,
            radial_offset=radial_offset,
            azimuthal_offset=azimuthal_offset,
            ensemble_axes_metadata=[ScanAxis()],
            metadata={"energy": 100e3},
        )

    @staticmethod
    def _values(result):
        return np.asarray(result.compute().to_cpu().array)

    def test_construction(self):
        pm = self._polar()
        assert pm.shape[-2] == 3
        assert pm.shape[-1] == 4
        assert pm.outer_angle == 15.0

    @pytest.mark.parametrize(
        "values, radial_limits, azimuthal_limits, expected",
        [
            # ones: every selected bin contributes 1.
            ("ones", None, None, 12.0),  # all 3 x 4 bins
            ("ones", (0, 10), None, 8.0),  # radial bins 0, 1 -> 2 x 4
            ("ones", None, (0, np.pi), 6.0),  # azimuthal bins 0, 1 -> 3 x 2
            ("ones", (0, 10), (0, np.pi), 4.0),  # 2 x 2
            # indexed: A[i, j] = 10 i + j.
            ("indexed", None, None, 138.0),  # total
            ("indexed", (0, 10), None, 52.0),  # R_0 + R_1 = 6 + 46
            ("indexed", (5, 15), None, 132.0),  # R_1 + R_2 = 46 + 86
            ("indexed", None, (0, np.pi), 63.0),  # C_0 + C_1 = 30 + 33
            ("indexed", None, (np.pi / 2, 2 * np.pi), 108.0),  # C_1 + C_2 + C_3
            # Limits inside a bin: centers are 2.5, 7.5, 12.5 mrad.
            ("indexed", (2, 12), None, 52.0),  # 2 <= 2.5, 7.5 < 12 -> R_0 + R_1
            ("indexed", (3, 12), None, 46.0),  # 2.5 < 3 -> R_1 only
            # Periodic azimuth: (-pi/2, pi/2) and (3pi/2, 5pi/2) are bins 3, 0.
            ("indexed", None, (-np.pi / 2, np.pi / 2), 69.0),  # C_3 + C_0
            ("indexed", None, (3 * np.pi / 2, 5 * np.pi / 2), 69.0),  # C_3 + C_0
            # Combined: i in {1, 2}, j = 1 -> A[1, 1] + A[2, 1] = 11 + 21.
            ("indexed", (5, 15), (np.pi / 2, np.pi), 32.0),
        ],
    )
    @lazy_params
    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_integrate_values(
        self, values, radial_limits, azimuthal_limits, expected, lazy, device
    ):
        pm = self._polar(values=values, lazy=lazy, device=device)
        result = pm.integrate(
            radial_limits=radial_limits, azimuthal_limits=azimuthal_limits
        )
        np.testing.assert_allclose(self._values(result), [expected, 2 * expected])

    @pytest.mark.parametrize(
        "radial_limits, azimuthal_limits, expected",
        [
            # radial_offset = 20: radial bin i spans [20 + 5 i, 25 + 5 i).
            ((25, 35), None, 132.0),  # R_1 + R_2
            ((0, 25), None, 6.0),  # nothing below the offset -> R_0 only
            # azimuthal_offset = pi/4: bin j spans [pi/4 + j pi/2, 3pi/4 + j pi/2).
            (None, (np.pi / 4, 5 * np.pi / 4), 63.0),  # C_0 + C_1
            (None, (3 * np.pi / 4, 7 * np.pi / 4), 69.0),  # C_1 + C_2
            # Bin 3 spans [7pi/4, 9pi/4), i.e. it wraps through 0.
            (None, (-np.pi / 4, np.pi / 4), 39.0),  # C_3
            # Combined: i = 0, j in {1, 2} -> A[0, 1] + A[0, 2] = 1 + 2.
            ((20, 25), (3 * np.pi / 4, 7 * np.pi / 4), 3.0),
        ],
    )
    @lazy_params
    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_integrate_values_with_offsets(
        self, radial_limits, azimuthal_limits, expected, lazy, device
    ):
        pm = self._polar(
            radial_offset=20.0, azimuthal_offset=np.pi / 4, lazy=lazy, device=device
        )
        result = pm.integrate(
            radial_limits=radial_limits, azimuthal_limits=azimuthal_limits
        )
        np.testing.assert_allclose(self._values(result), [expected, 2 * expected])

    def test_integrate_limits_on_inexact_bin_edges(self):
        # With radial_sampling = 0.1 the edge 0.3 is bin index 3, although
        # 0.3 / 0.1 == 2.9999999999999996 in floating point. (0, 0.3) must select
        # all three radial bins: total = 138.
        pm = self._polar(radial_sampling=0.1)
        result = pm.integrate(radial_limits=(0, 0.3))
        np.testing.assert_allclose(self._values(result), [138.0, 276.0])

    def test_integrate_radial_limit_exceeded(self):
        # outer_angle is 15 mrad; a limit extending over a whole further bin
        # (center 17.5 mrad) asks for data that does not exist.
        with pytest.raises(RuntimeError):
            self._polar().integrate(radial_limits=(0, 20))

    def test_integrate_radial(self):
        # integrate_radial(5, 15) == integrate(radial_limits=(5, 15)) = R_1 + R_2.
        result = self._polar().integrate_radial(5, 15)
        np.testing.assert_allclose(self._values(result), [132.0, 264.0])

    def test_integrate_with_detector_regions(self):
        # Region k is flat index k of the (3, 4) base: region 1 = A[0, 1] = 1,
        # region 4 = A[1, 0] = 10.
        pm = self._polar()
        result = pm.integrate(detector_regions=[1, 4])
        np.testing.assert_allclose(self._values(result), [11.0, 22.0])


# ---------------------------------------------------------------------------
# ReciprocalSpaceLineProfiles
# ---------------------------------------------------------------------------

class TestReciprocalSpaceLineProfiles:
    def test_from_ctf(self):
        from abtem.transfer import CTF
        ctf = CTF(energy=100e3, gpts=(64, 64), sampling=(0.1, 0.1), defocus=200.0)
        profiles = ctf.profiles()
        assert isinstance(profiles, ReciprocalSpaceLineProfiles)

    def test_shape(self):
        from abtem.transfer import CTF
        ctf = CTF(energy=100e3, gpts=(64, 64), sampling=(0.1, 0.1))
        profiles = ctf.profiles()
        assert len(profiles.base_shape) == 1
        assert profiles.base_shape[0] > 0


def test_lazy_filters_set_the_warning_filters_once(monkeypatch):
    # The CuPy branch of the FFT convolution imports cupyx.scipy.signal under
    # catch_warnings, which swaps the process-wide filter list; entered from
    # dask's threads at once, it can leave the import's filter installed or drop
    # the user's. Run that branch on NumPy blocks, with cupyx.scipy.signal
    # stood in by an empty module, and count the entries.
    import abtem.measurements as measurements

    names = ("cupyx", "cupyx.scipy", "cupyx.scipy.signal")
    modules = {name: types.ModuleType(name) for name in names}
    modules["cupyx"].scipy = modules["cupyx.scipy"]
    modules["cupyx.scipy"].signal = modules["cupyx.scipy.signal"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, modules[name])
    monkeypatch.setattr(measurements, "cp", np)

    entries = []

    class CountingWarnings:
        def __getattr__(self, name):
            return getattr(warnings, name)

        def catch_warnings(self, *args, **kwargs):
            entries.append(None)
            return warnings.catch_warnings(*args, **kwargs)

    monkeypatch.setattr(measurements, "warnings", CountingWarnings())
    # monkeypatch restores the flag, so the stand-in is not recorded as the real
    # import
    monkeypatch.setattr(measurements, "_cupyx_signal_imported", False)

    images = Images(
        da.ones((16, 32, 32), chunks=(1, 32, 32)),
        sampling=0.1,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(16)))],
    )
    images.lorentzian_filter(0.3).compute(scheduler="threads", num_workers=4)

    assert len(entries) == 1
