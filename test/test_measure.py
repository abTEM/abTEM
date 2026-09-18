import warnings

import ase
import dask.array as da
import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.strategies import composite
from utils import array_is_close, ensure_is_tuple, gpu, requires_gpu

import abtem
from abtem.core.axes import OrdinalAxis, ScanAxis
from abtem.core.backend import copy_to_device
from abtem.measurements import (
    DiffractionPatterns,
    Images,
    PolarMeasurements,
    RealSpaceLineProfiles,
    ReciprocalSpaceLineProfiles,
    _apply_convolve_2d_on_axes,
    _gaussian_kernel_2d,
    _gaussian_kernels_1d,
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
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
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
    new_measurement = getattr(measurement, method)(measurement.copy())
    assert new_measurement.array is not measurement.array


@settings(max_examples=5)
@given(data=st.data())
@pytest.mark.parametrize("method", ["__iadd__", "__isub__", "__imul__", "__itruediv__"])
@pytest.mark.parametrize("device", ["cpu", gpu])
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
@pytest.mark.parametrize("device", [gpu, "cpu"])
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
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
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
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
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


@given(data=st.data(), sigma=sigma())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_gaussian_filter_images(data, sigma, lazy, device):
    if lazy is True and device == gpu.values[0]:
        return

    measurement = data.draw(abtem_st.images(lazy=lazy, device=device))
    assume(all(n > 1 for n in measurement.base_shape))
    try:
        filtered = measurement.gaussian_filter(sigma)
        filtered.compute()
        measurement.compute()
    except OSError:
        pytest.skip(
            "Known CuPy error, but only reproducible in pytest https://github.com/cupy/cupy/issues/8218"
        )

    if np.any(np.array(sigma)) > 1:
        assert not np.allclose(filtered.array, measurement.array)


@given(data=st.data(), sigma=sigma())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_lorentzian_filter_images(data, sigma, lazy, device):
    if lazy is True and device == gpu.values[0]:
        return

    measurement = data.draw(abtem_st.images(lazy=lazy, device=device))
    assume(all(n > 1 for n in measurement.base_shape))
    try:
        filtered = measurement.lorentzian_filter(sigma)
        filtered.compute()
        measurement.compute()
    except OSError:
        pytest.skip(
            "Known CuPy error, but only reproducible in pytest https://github.com/cupy/cupy/issues/8218"
        )

    if np.any(np.array(sigma)) > 1:
        assert not np.allclose(filtered.array, measurement.array)


@given(data=st.data(), sigma=sigma())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_voigtian_filter_images(data, sigma, lazy, device):
    if lazy is True and device == gpu.values[0]:
        return

    measurement = data.draw(abtem_st.images(lazy=lazy, device=device))
    assume(all(n > 1 for n in measurement.base_shape))
    try:
        # Use sigma as both gaussian_sigma and lorentzian_gamma
        filtered = measurement.voigtian_filter(sigma, sigma)
        filtered.compute()
        measurement.compute()
    except OSError:
        pytest.skip(
            "Known CuPy error, but only reproducible in pytest https://github.com/cupy/cupy/issues/8218"
        )

    if np.any(np.array(sigma)) > 1:
        assert not np.allclose(filtered.array, measurement.array)


def test_lorentzian_filter_changes_image():
    """Lorentzian filter with a non-trivial HWHM should change the image."""
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=64)
    images = wave.build((0, 0), lazy=False).intensity()
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
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=64)
    images = wave.build((0, 0), lazy=False).intensity()
    filtered = images.voigtian_filter(0.3, 0.3)
    assert not np.allclose(filtered.array, images.array)


def test_voigtian_filter_pure_gaussian_limit():
    """voigtian_filter with lorentzian_gamma=0 must equal gaussian_filter."""
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=64)
    images = wave.build((0, 0), lazy=False).intensity()
    sigma = 0.4
    gauss = images.gaussian_filter(sigma)
    voigt = images.voigtian_filter(sigma, 0.0)
    assert np.allclose(gauss.array, voigt.array, atol=1e-5)


def test_voigtian_filter_pure_lorentzian_limit():
    """voigtian_filter with gaussian_sigma=0 must equal lorentzian_filter."""
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=64)
    images = wave.build((0, 0), lazy=False).intensity()
    hw = 0.4
    lor = images.lorentzian_filter(hw)
    voigt = images.voigtian_filter(0.0, hw)
    assert np.allclose(lor.array, voigt.array, atol=1e-5)


def test_pseudo_voigtian_filter_changes_image():
    """Pseudo-Voigtian filter with non-trivial parameters should change the image."""
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=64)
    images = wave.build((0, 0), lazy=False).intensity()
    filtered = images.pseudo_voigtian_filter(0.3, 0.3, eta=0.5)
    assert not np.allclose(filtered.array, images.array)


def test_pseudo_voigtian_filter_pure_gaussian_limit():
    """pseudo_voigtian_filter with eta=0 must equal gaussian_filter."""
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=64)
    images = wave.build((0, 0), lazy=False).intensity()
    sigma = 0.4
    gauss = images.gaussian_filter(sigma)
    pv = images.pseudo_voigtian_filter(sigma, 1.0, eta=0.0)
    assert np.allclose(gauss.array, pv.array, atol=1e-5)


def test_pseudo_voigtian_filter_pure_lorentzian_limit():
    """pseudo_voigtian_filter with eta=1 must equal lorentzian_filter."""
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=64)
    images = wave.build((0, 0), lazy=False).intensity()
    hw = 0.4
    lor = images.lorentzian_filter(hw)
    pv = images.pseudo_voigtian_filter(1.0, hw, eta=1.0)
    assert np.allclose(lor.array, pv.array, atol=1e-5)


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
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=32)
    images = wave.build((0, 0), lazy=False).intensity()
    for boundary in ("periodic", "reflect", "constant"):
        images.gaussian_filter(0.3, boundary=boundary).array
        images.lorentzian_filter(0.3, boundary=boundary).array
        images.voigtian_filter(0.3, 0.3, boundary=boundary).array
        images.pseudo_voigtian_filter(0.3, 0.3, eta=0.5, boundary=boundary).array


@requires_gpu
@pytest.mark.parametrize("boundary", ["periodic", "reflect", "constant"])
@pytest.mark.parametrize("lazy", [False, True])
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
@pytest.mark.parametrize("lazy", [False, True])
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


def test_lorentzian_filter_lazy():
    """Lorentzian filter works on a lazy (dask-backed) image."""
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=32)
    images = wave.build((0, 0), lazy=True).intensity()
    filtered = images.lorentzian_filter(0.5)
    assert np.isfinite(filtered.array.compute()).all()


def test_voigtian_filter_lazy():
    """Voigtian filter works on a lazy (dask-backed) image."""
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=32)
    images = wave.build((0, 0), lazy=True).intensity()
    filtered = images.voigtian_filter(0.3, 0.3)
    assert np.isfinite(filtered.array.compute()).all()


def test_pseudo_voigtian_filter_lazy():
    """Pseudo-Voigtian filter works on a lazy (dask-backed) image."""
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=10, gpts=32)
    images = wave.build((0, 0), lazy=True).intensity()
    filtered = images.pseudo_voigtian_filter(0.3, 0.3, eta=0.5)
    assert np.isfinite(filtered.array.compute()).all()


# @given(data=st.data())
# @pytest.mark.parametrize('lazy', [True, False])
# @pytest.mark.parametrize('device', ['cpu', gpu])
# def test_diffractograms(data, lazy, device):
#     measurement = data.draw(abtem_st.images(lazy=lazy, device=device))
#     measurement.diffractograms()


@given(data=st.data())
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_images_interpolate_line(data, lazy, device):
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=20, gpts=256, device=device)
    image = wave.build((0, 0), lazy=lazy).intensity()

    line = image.interpolate_line(start=(0, 0), end=(0, wave.extent[1]), width=0.0)
    assert np.allclose(
        image.to_cpu().compute().array[0], line.to_cpu().compute().array,
        rtol=1e-6, atol=1e-6,
    )

    coordinate = st.floats(min_value=0, max_value=wave.extent[0])
    center = data.draw(st.tuples(coordinate, coordinate))
    angle1 = data.draw(st.floats(min_value=0, max_value=360.0))
    angle2 = data.draw(st.floats(min_value=0, max_value=360.0))
    width = data.draw(st.floats(min_value=0, max_value=2.0))

    image = wave.build(center, lazy=lazy).intensity()
    line1 = image.interpolate_line_at_position(
        center=center, angle=angle1, extent=wave.extent[0] / 2, width=width, gpts=128
    ).to_cpu().compute()
    line2 = image.interpolate_line_at_position(
        center=center, angle=angle2, extent=wave.extent[0] / 2, width=width, gpts=128
    ).to_cpu().compute()

    assert np.allclose(line1.array, line2.array, rtol=1e-6, atol=10)


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
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
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
@pytest.mark.parametrize("lazy", [True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_diffraction_patterns_polar_binning(data, lazy, device):
    measurement = data.draw(
        abtem_st.diffraction_patterns(lazy=lazy, device=device, min_base_side=16)
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
    print(nbins_radial)
    measurement.polar_binning(
        nbins_radial=nbins_radial,
        nbins_azimuthal=nbins_azimuthal,
        inner=inner,
        outer=outer,
        rotation=rotation,
    )

    step_size = data.draw(
        abtem_st.sensible_floats(
            min_value=min(measurement.angular_sampling),
            max_value=max(min(measurement.angular_sampling), outer - inner),
        )
    )

    # measurement.radial_binning(step_size=step_size,
    #                           inner=inner,
    #                           outer=outer)


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_diffraction_patterns_center_of_mass(data, lazy, device):
    measurement = data.draw(
        abtem_st.diffraction_patterns(
            lazy=lazy, min_scan_dims=1, device=device, min_base_side=16
        )
    )
    assume(len(_scan_sampling(measurement)) > 0)

    print(measurement.shape, measurement.axes_metadata)

    measurement.center_of_mass().compute()


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_diffraction_patterns_integrated_center_of_mass(data, lazy, device):
    measurement = data.draw(
        abtem_st.diffraction_patterns(
            lazy=lazy, min_scan_dims=1, device=device, min_base_side=16
        )
    )
    assume(len(_scan_sampling(measurement)) > 1)
    measurement.integrated_center_of_mass().compute()


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_diffraction_patterns_integrated_center_of_mass(data, lazy, device):
    measurement = data.draw(
        abtem_st.diffraction_patterns(
            lazy=lazy, min_scan_dims=1, device=device, min_base_side=16
        )
    )
    assume(len(_scan_sampling(measurement)) > 1)
    measurement.integrated_center_of_mass().compute()


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
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
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
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
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
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

    azimuthal_outer = data.draw(
        abtem_st.sensible_floats(min_value=0.0, max_value=360.0)
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
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_line_profiles_interpolate(data, lazy, device):
    measurement = data.draw(abtem_st.line_profiles(lazy=lazy, device=device))
    measurement.interpolate().compute()


@given(data=st.data(), reps=st.integers(min_value=1, max_value=3))
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_line_profiles_tile(data, reps, lazy, device):
    measurement = data.draw(abtem_st.line_profiles(lazy=lazy, device=device))
    measurement.tile(reps).compute()


@pytest.mark.parametrize("lazy", [True, False])
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


@pytest.mark.parametrize("lazy", [True, False])
def test_interpolate_periodic_spline_and_fft(lazy):
    atoms = ase.build.bulk("Si", cubic=True)
    images = abtem.PlaneWave(energy=100e3, sampling=0.05).multislice(atoms).intensity()

    if not lazy:
        images.compute()

    spline_interpolated = images.interpolate(
        method="spline", sampling=0.05, boundary="periodic", order=5
    )
    fft_interpolated = images.interpolate(method="fft", sampling=0.05)
    array_is_close(spline_interpolated.array, fft_interpolated.array, rel_tol=0.01)


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


@given(
    gpts=st.tuples(
        st.integers(min_value=50, max_value=100),
        st.integers(min_value=50, max_value=100),
    ),
    radius=st.floats(min_value=5, max_value=20),
    sampling=st.tuples(
        st.floats(min_value=0.05, max_value=1), st.floats(min_value=0.05, max_value=1)
    ),
    position=st.tuples(
        st.floats(min_value=0.0, max_value=0), st.floats(min_value=0.0, max_value=0.0)
    ),
)
def test_integrate_disc(gpts, radius, sampling, position):
    array = np.ones(gpts)
    measurement = Images(array, sampling=sampling)
    output = measurement.integrate_disc(position=position, radius=radius)
    expected = (radius / sampling[0]) * (radius / sampling[1]) * np.pi
    assert np.abs(output - expected) < 4 * np.pi * radius


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

def _make_images(shape=(32, 32), sampling=(0.1, 0.1), complex_=False):
    arr = np.random.default_rng(0).random(shape)
    if complex_:
        arr = arr + 1j * np.random.default_rng(1).random(shape)
    return Images(arr, sampling=sampling)


class TestImagesCrop:
    def test_crop_reduces_extent(self):
        imgs = _make_images((32, 32), (0.1, 0.1))
        cropped = imgs.crop((1.5, 1.5))
        assert cropped.extent[0] <= imgs.extent[0]
        assert cropped.extent[1] <= imgs.extent[1]

    def test_crop_centered(self):
        imgs = _make_images((32, 32), (0.1, 0.1))
        cropped = imgs.crop((1.0, 1.0), centered=True)
        assert cropped.base_shape[0] <= imgs.base_shape[0]

    def test_crop_too_large_raises(self):
        imgs = _make_images((32, 32), (0.1, 0.1))
        with pytest.raises(ValueError, match="smaller"):
            imgs.crop((999.0, 999.0))

    def test_crop_centered_with_offset_raises(self):
        imgs = _make_images((32, 32), (0.1, 0.1))
        with pytest.raises(ValueError):
            imgs.crop((1.0, 1.0), offset=(0.1, 0.1), centered=True)

    def test_crop_with_offset(self):
        imgs = _make_images((32, 32), (0.1, 0.1))
        cropped = imgs.crop((1.0, 1.0), offset=(0.5, 0.5))
        assert cropped.base_shape[0] <= imgs.base_shape[0]


class TestImagesComplexAccessors:
    def test_real(self):
        imgs = _make_images(complex_=True)
        real = imgs.real()
        assert not np.iscomplexobj(real.array)
        assert np.allclose(real.array, imgs.array.real)

    def test_imag(self):
        imgs = _make_images(complex_=True)
        imag = imgs.imag()
        assert np.allclose(imag.array, imgs.array.imag)

    def test_phase(self):
        imgs = _make_images(complex_=True)
        phase = imgs.phase()
        assert np.all(np.abs(phase.array) <= np.pi + 1e-10)

    def test_abs(self):
        imgs = _make_images(complex_=True)
        ab = imgs.abs()
        assert np.all(ab.array >= 0)

    def test_real_on_real_raises(self):
        imgs = _make_images(complex_=False)
        with pytest.raises(RuntimeError):
            imgs.real()


class TestImagesNormalizeEnsemble:
    def test_normalize_reduces_spread(self):
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]],
                        [[10.0, 20.0], [30.0, 40.0]]])
        from abtem.core.axes import OrdinalAxis
        imgs = Images(arr, sampling=(0.1, 0.1),
                      ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))])
        normalized = imgs.normalize_ensemble()
        assert normalized.array.shape == arr.shape


class TestImagesScanNoise:
    def test_scan_noise_returns_images(self):
        imgs = _make_images((16, 16))
        result = imgs.scan_noise(
            rms_power=1.0, dwell_time=1e-6, flyback_time=1e-4,
            num_components=5
        ).compute()
        assert isinstance(result, Images)

    def test_scan_noise_shape_preserved(self):
        imgs = _make_images((16, 16))
        result = imgs.scan_noise(1.0, 1e-6, 1e-4, num_components=5).compute()
        assert result.base_shape == imgs.base_shape


class TestImagesRelativeDifference:
    def test_zero_difference(self):
        imgs = _make_images()
        diff = imgs.relative_difference(imgs.copy())
        assert np.allclose(diff.array[np.isfinite(diff.array)], 0.0, atol=1e-10)

    def test_wrong_type_raises(self):
        imgs = _make_images()
        dp = DiffractionPatterns(
            np.ones((8, 8)), sampling=0.1, metadata={"energy": 100e3}
        )
        with pytest.raises(RuntimeError):
            imgs.relative_difference(dp)


# ---------------------------------------------------------------------------
# DiffractionPatterns — integrate_radial, crop, poisson_noise with samples
# ---------------------------------------------------------------------------

class TestDiffractionPatternsIntegrateRadial:
    def _dp(self, shape=(32, 32)):
        return DiffractionPatterns(
            np.ones(shape), sampling=0.05, metadata={"energy": 100e3}
        )

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
        dp = self._dp()
        result = dp.integrate_radial(inner=5, outer=5)
        assert np.all(result.array == 0.0)

    def test_larger_outer_gives_larger_sum(self):
        dp = self._dp()
        r1 = dp.integrate_radial(0, 5)
        r2 = dp.integrate_radial(0, 10)
        assert r2.array.sum() >= r1.array.sum()


class TestDiffractionPatternsCrop:
    def test_crop_reduces_max_angle(self):
        dp = DiffractionPatterns(
            np.ones((64, 64)), sampling=0.05, metadata={"energy": 100e3}
        )
        max_before = min(dp.max_angles)
        cropped = dp.crop(max_angle=max_before / 2)
        assert min(cropped.max_angles) <= min(dp.max_angles)


class TestDiffractionPatternsPoisson:
    def test_poisson_with_samples(self):
        dp = DiffractionPatterns(
            np.ones((16, 16)) * 100, sampling=0.05, metadata={"energy": 100e3}
        )
        noisy = dp.poisson_noise(total_dose=1e6, samples=4).compute()
        assert noisy.shape[0] == 4

    def test_poisson_nonnegative(self):
        dp = DiffractionPatterns(
            np.ones((16, 16)) * 50, sampling=0.05, metadata={"energy": 100e3}
        )
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
    def _polar(self, nbins_radial=8, nbins_azimuthal=6):
        arr = np.ones((4, 4, nbins_radial, nbins_azimuthal))
        return PolarMeasurements(
            arr,
            radial_sampling=5.0,
            azimuthal_sampling=360.0 / nbins_azimuthal,
            radial_offset=0.0,
            azimuthal_offset=0.0,
            ensemble_axes_metadata=[ScanAxis(), ScanAxis()],
            metadata={"energy": 100e3},
        )

    def test_construction(self):
        pm = self._polar()
        assert pm.shape[-2] == 8
        assert pm.shape[-1] == 6

    def test_integrate_radial(self):
        pm = self._polar()
        result = pm.integrate_radial(0, pm.outer_angle)
        assert isinstance(result, Images)

    def test_integrate_all(self):
        pm = self._polar()
        result = pm.integrate(
            radial_limits=(0, pm.outer_angle),
            azimuthal_limits=None,
        )
        assert result.shape == pm.ensemble_shape

    def test_integrate_with_detector_regions(self):
        pm = self._polar()
        n_regions = pm.shape[-2] * pm.shape[-1]
        result = pm.integrate(detector_regions=list(range(n_regions))).compute()
        assert result is not None


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
