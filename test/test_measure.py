import operator

import ase
import dask.array as da
import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.strategies import composite
from utils import array_is_close, assert_array_matches_device, ensure_is_tuple, gpu

import abtem
from abtem.core.axes import OrdinalAxis, ScanAxis
from abtem.core.backend import asnumpy, copy_to_device, get_array_module
from abtem.core.energy import energy2wavelength
from abtem.core.utils import get_dtype
from abtem.measurements import (
    DiffractionPatterns,
    Images,
    MeasurementsEnsemble,
    PolarMeasurements,
    RealSpaceLineProfiles,
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


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv", "pow"])
@pytest.mark.parametrize(
    "scalar_type", ["python", "numpy_float64", "numpy_float32", "0d_array"]
)
def test_reflected_arithmetic_with_a_scalar(scalar_type, op, lazy, device):
    # Oracle: the same operation with the scalar on the left of the plain array.
    # The array is not symmetric under any of the operations, so a reflected
    # operation computed in the forward order (`2 / m` as `m / 2`) fails. NumPy
    # scalars and arrays on the left must defer to the measurement instead of
    # coercing it to an array.
    xp = get_array_module(device)
    host_scalar = {
        "python": 2.0,
        "numpy_float64": np.float64(1.5),
        "numpy_float32": np.float32(3.0),
        "0d_array": np.asarray(2.0),
    }[scalar_type]
    scalar = xp.asarray(host_scalar) if scalar_type == "0d_array" else host_scalar
    array = np.array([[1.0, 2.0, 4.0], [8.0, 0.5, 0.25]], dtype=get_dtype())
    measurement = Images(
        da.from_array(array, chunks=(1, 3)) if lazy else array, sampling=(0.1, 0.2)
    ).copy_to_device(device)

    result = getattr(operator, op)(scalar, measurement)

    assert isinstance(result, Images)
    assert result.is_lazy == lazy
    expected = getattr(operator, op)(host_scalar, array)
    np.testing.assert_allclose(
        asnumpy(result.compute().array),
        expected,
        rtol=1e-6,
        atol=1e-6 * np.abs(expected).max(),
    )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_numpy_scalar_on_the_left_of_a_measurement_without_base_axes(device):
    # MeasurementsEnsemble has no base axes, so NumPy's coercion through
    # __len__/__getitem__ would succeed and build an object array; a NumPy scalar
    # on the left must defer to the reflected operators instead.
    array = np.array([[1.0, 2.0, 4.0], [8.0, 0.5, 0.25]], dtype=get_dtype())
    ensemble = MeasurementsEnsemble(
        array,
        ensemble_axes_metadata=[
            OrdinalAxis(values=(0, 1)),
            OrdinalAxis(values=(0, 1, 2)),
        ],
    ).copy_to_device(device)
    for op in (operator.mul, operator.truediv, operator.add, operator.sub):
        result = op(np.float64(2.0), ensemble)
        assert isinstance(result, MeasurementsEnsemble)
        assert_array_matches_device(result.array, device)
        np.testing.assert_allclose(asnumpy(result.array), op(2.0, array), rtol=1e-6)


@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv", "pow"])
def test_unsupported_left_operand_raises_type_error(op):
    # A reflected operator declines an operand it does not support, so Python
    # raises its own TypeError rather than the measurement raising another error.
    measurement = Images(np.ones((2, 3)), sampling=0.1)
    with pytest.raises(TypeError, match="unsupported operand"):
        getattr(operator, op)(object(), measurement)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv", "pow"])
def test_dask_array_on_the_left_of_a_measurement(op, lazy, device):
    # Oracle: the same operation on the plain arrays. The two operands differ in
    # every element, so an operation computed in the forward order fails for
    # sub, truediv and pow. The result is lazy whenever either operand is.
    left = np.array([[3.0, 1.0, 2.0], [5.0, 7.0, 11.0]], dtype=get_dtype())
    array = np.array([[1.0, 2.0, 4.0], [8.0, 0.5, 0.25]], dtype=get_dtype())
    measurement = Images(
        da.from_array(array, chunks=(1, 3)) if lazy else array, sampling=(0.1, 0.2)
    ).copy_to_device(device)
    dask_left = copy_to_device(da.from_array(left, chunks=(1, 3)), device)

    result = getattr(operator, op)(dask_left, measurement)

    assert isinstance(result, Images)
    assert result.is_lazy
    computed = result.compute().array
    assert_array_matches_device(computed, device)
    expected = getattr(operator, op)(left.astype(np.float64), array.astype(np.float64))
    np.testing.assert_allclose(
        asnumpy(computed), expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max()
    )


@pytest.mark.parametrize("reflected", [False, True], ids=["forward", "reflected"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
@pytest.mark.parametrize("operand_shape", [(), (2, 3)], ids=["0d_array", "2d_array"])
def test_numpy_array_operand_of_a_measurement_on_the_device(
    operand_shape, op, device, lazy, reflected
):
    # CuPy refuses a host array as an operand, a 0-d one included, so a NumPy
    # array has to be moved to the device of the measurement. The array operand
    # differs from the measurement in every element, so a reflected operation
    # computed in the forward order fails.
    rng = np.random.default_rng(0)
    operand = rng.uniform(1.0, 2.0, size=operand_shape).astype(get_dtype())
    array = np.array([[1.0, 2.0, 4.0], [8.0, 0.5, 0.25]], dtype=get_dtype())
    measurement = Images(
        da.from_array(array, chunks=(1, 3)) if lazy else array, sampling=(0.1, 0.2)
    ).copy_to_device(device)

    if reflected:
        result = getattr(measurement, f"__r{op}__")(operand)
        expected = getattr(operator, op)(operand, array)
    else:
        result = getattr(operator, op)(measurement, operand)
        expected = getattr(operator, op)(array, operand)

    assert isinstance(result, Images)
    assert result.is_lazy == lazy
    computed = result.compute().array
    assert_array_matches_device(computed, device)
    np.testing.assert_allclose(
        asnumpy(computed), expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_builtin_sum_of_measurements(lazy, device):
    # sum() starts from the integer 0, so it needs 0 + m.
    arrays = [np.full((2, 3), value, dtype=get_dtype()) for value in (1.0, 2.0, 4.0)]
    measurements = [
        Images(
            da.from_array(array, chunks=(1, 3)) if lazy else array, sampling=0.1
        ).copy_to_device(device)
        for array in arrays
    ]

    result = sum(measurements)

    assert isinstance(result, Images)
    assert result.is_lazy == lazy
    np.testing.assert_array_equal(asnumpy(result.compute().array), sum(arrays))


def test_in_place_true_division_refuses_lazy_measurements():
    # Like the other in-place operators, /= must refuse a lazy measurement
    # (it used to fail with an AttributeError on the dask array).
    measurement = Images(da.ones((4, 4), chunks=2), sampling=0.1)
    with pytest.raises(RuntimeError, match="inplace"):
        measurement /= 2.0


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


def test_images_coordinates_spaced_by_sampling():
    # Pixel i of a periodic image with sampling s sits at i * s; the last
    # pixel is one sampling short of the extent, not at the extent.
    images = Images(np.zeros((4, 5)), sampling=(0.5, 0.2))
    x, y = images.coordinates
    assert np.allclose(x, [0.0, 0.5, 1.0, 1.5])
    assert np.allclose(y, [0.0, 0.2, 0.4, 0.6, 0.8])


@composite
def sigma(draw):
    sigma = st.floats(min_value=0.0, max_value=5.0)
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


@pytest.mark.parametrize("cval", [0.0, 0.5])
def test_gaussian_filter_lazy_constant_boundary_matches_eager(cval):
    """dask's map_overlap only understands reflect/periodic/nearest/none or a
    numeric fill value for `boundary`, so the string 'constant' has to be
    translated to `cval`; passing it through raised ValueError."""
    rng = np.random.default_rng(0)
    array = rng.random((32, 32)).astype(np.float32)
    images = Images(array, sampling=0.1)

    eager = images.gaussian_filter(0.3, boundary="constant", cval=cval)
    lazy = (
        images.lazy()
        .rechunk((16, 16))
        .gaussian_filter(0.3, boundary="constant", cval=cval)
    )
    assert np.allclose(lazy.compute().array, eager.array, atol=1e-5)


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
        image.to_cpu().compute().array[0],
        line.to_cpu().compute().array,
        rtol=1e-6,
        atol=1e-6,
    )

    coordinate = st.floats(min_value=0, max_value=wave.extent[0])
    center = data.draw(st.tuples(coordinate, coordinate))
    angle1 = data.draw(st.floats(min_value=0, max_value=360.0))
    angle2 = data.draw(st.floats(min_value=0, max_value=360.0))
    width = data.draw(st.floats(min_value=0, max_value=2.0))

    image = wave.build(center, lazy=lazy).intensity()
    line1 = (
        image.interpolate_line_at_position(
            center=center,
            angle=angle1,
            extent=wave.extent[0] / 2,
            width=width,
            gpts=128,
        )
        .to_cpu()
        .compute()
    )
    line2 = (
        image.interpolate_line_at_position(
            center=center,
            angle=angle2,
            extent=wave.extent[0] / 2,
            width=width,
            gpts=128,
        )
        .to_cpu()
        .compute()
    )

    assert np.allclose(line1.array, line2.array, rtol=1e-6, atol=10)


def _periodic_gaussian_blob(gpts, sampling, center, sigma):
    """exp(-|r - center|^2 / (2 sigma^2)) on a periodic grid (minimum-image
    distance), peak 1."""
    extent = [n * d for n, d in zip(gpts, sampling)]
    coords = []
    for n, d, c, length in zip(gpts, sampling, center, extent):
        r = np.arange(n) * d - c
        coords.append(r - length * np.round(r / length))
    x, y = np.meshgrid(*coords, indexing="ij")
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))


def _line_values(line):
    return np.asarray(asnumpy(line.compute().array), dtype=np.float64)


@settings(max_examples=10, deadline=None)
@given(data=st.data())
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_images_interpolate_line_at_position(data, lazy, device):
    """A line of length L at any angle through the centre c of a rotationally
    symmetric Gaussian blob exp(-|r - c|^2/2 sigma^2) samples
    exp(-t^2/2 sigma^2), t = -L/2 ... L/2, independent of the angle (analytic
    profile)."""
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

    array = _periodic_gaussian_blob(gpts, sampling, center, sigma)
    if lazy:
        array = da.from_array(array, chunks=(gpts[0] // 2, -1))
    images = Images(array, sampling=sampling).copy_to_device(device)
    line = images.interpolate_line_at_position(
        center=center, angle=angle, extent=length, gpts=n, endpoint=True
    )

    t = np.linspace(-length / 2, length / 2, n)
    expected = np.exp(-(t**2) / (2 * sigma**2))
    # Cubic-spline interpolation of a Gaussian 8-10 pixels wide errs by
    # ~1e-4 of the peak; 1e-3 still rejects any off-centre or mis-rotated
    # line (a 0.1 Å offset changes the profile by ~1e-2 of the peak).
    np.testing.assert_allclose(_line_values(line), expected, rtol=0, atol=1e-3)

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
    np.testing.assert_allclose(wide[0], wide[1], rtol=0, atol=1e-3)


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

    from abtem.core.axes import ReciprocalSpaceAxis
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
                OrdinalAxis(label="energy", values=energies),
                ReciprocalSpaceAxis(sampling=0.02, label="x", units="1/A"),
                ReciprocalSpaceAxis(sampling=0.02, label="y", units="1/A"),
            ],
            metadata={"energy": 100e3},
        )

    truth = (
        make_dp((n_energy, gpts, gpts))
        .interpolate_line(
            start=(0.0, 0.0),
            end=(0.0, gpts * 0.02),
            gpts=20,
            width=0.3,
            order=1,
            endpoint=True,
        )
        .compute()
        .array
    )

    for name, chunks in [
        ("multi_chunk_ensemble_axis", (1, gpts, gpts)),
        ("both_base_axes_chunked", (n_energy, gpts // 2, gpts // 2)),
    ]:
        dp = make_dp(chunks)
        line = dp.interpolate_line(
            start=(0.0, 0.0),
            end=(0.0, gpts * 0.02),
            gpts=20,
            width=0.3,
            order=1,
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


def test_scan_noise():
    array = np.random.default_rng(0).random((16, 16)) + 1.0
    image = Images(array, sampling=0.1, metadata={"units": "electrons"})

    noisy = image.scan_noise(
        rms_power=0.1, dwell_time=1e-6, flyback_time=1e-5, seed=1
    ).compute()

    assert isinstance(noisy, Images)
    assert noisy.base_shape == image.base_shape
    assert not np.allclose(noisy.array, array)


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


@pytest.mark.parametrize("lazy", [True, False])
def test_diffraction_patterns_float64_precision(lazy):
    """Polar binning, center of mass and differential images must follow the
    configured precision instead of falling back to float32 / complex64."""
    array = np.random.default_rng(0).random((3, 3, 32, 32))
    ensemble_axes_metadata = [ScanAxis(sampling=1.0), ScanAxis(sampling=1.0)]

    with abtem.config.set({"precision": "float64"}):
        measurement = DiffractionPatterns(
            array,
            sampling=0.5,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata={"energy": 100e3},
        )
        if lazy:
            measurement = measurement.lazy()

        polar = measurement.polar_binning(
            nbins_radial=2, nbins_azimuthal=4, inner=0.0, outer=8.0
        )
        assert polar.array.dtype == np.float64
        assert polar.compute().array.dtype == np.float64

        differentials = polar.compute().differentials((0, 2), (1, 3))
        assert differentials.array.dtype == np.complex128

        com = measurement.center_of_mass()
        assert com.array.dtype == np.complex128
        assert com.compute().array.dtype == np.complex128


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


@pytest.mark.parametrize("chunks", [((2, 1), 16, 20), ((1, 1, 1), 16, 20)])
def test_integrate_gradient_offset_is_per_member(chunks):
    rng = np.random.default_rng(0)
    gradient = rng.normal(size=(3, 16, 20)) + 1j * rng.normal(size=(3, 16, 20))
    gradient *= np.array([1.0, 5.0, 0.3])[:, None, None]
    kwargs = dict(sampling=0.1, ensemble_axes_metadata=[OrdinalAxis(values=(0, 1, 2))])

    eager = Images(gradient, **kwargs).integrate_gradient()
    lazy = Images(da.from_array(gradient, chunks=chunks), **kwargs).integrate_gradient()

    np.testing.assert_allclose(eager.array.min(axis=(-2, -1)), 0.0, atol=1e-12)
    np.testing.assert_allclose(lazy.compute().array, eager.array, atol=1e-12)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "method", ["integrate_gradient", "center_of_mass", "center_of_mass_mrad"]
)
def test_single_precision_measurements_stay_single_precision(method, lazy):
    # A float64 factor, here the frequency grid or the coordinates, must not widen
    # single-precision data; the declared precision of a lazy result must be the
    # computed one.
    rng = np.random.default_rng(0)
    ensemble_axes_metadata = [OrdinalAxis(values=(0, 1))]
    with abtem.config.set({"precision": "float32"}):
        if method == "integrate_gradient":
            data = rng.normal(size=(2, 16, 20)) + 1j * rng.normal(size=(2, 16, 20))
            data = data.astype(np.complex64)
            kwargs, expected = {}, np.dtype(np.float32)
        else:
            data = (np.abs(rng.normal(size=(2, 16, 20))) + 1).astype(np.float32)
            kwargs, expected = {"metadata": {"energy": 100e3}}, np.dtype(np.complex64)
        array = da.from_array(data, chunks=(1, 16, 20)) if lazy else data
        cls = Images if method == "integrate_gradient" else DiffractionPatterns
        measurement = cls(
            array, sampling=0.1, ensemble_axes_metadata=ensemble_axes_metadata, **kwargs
        )

        if method == "integrate_gradient":
            result = measurement.integrate_gradient()
        else:
            result = measurement.center_of_mass(
                units="mrad" if method.endswith("mrad") else "1/Å"
            )

        assert result.array.dtype == expected
        assert result.compute().array.dtype == expected


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_center_of_mass_of_integer_counts_follows_the_configured_precision(
    precision, lazy
):
    counts = np.random.default_rng(3).integers(0, 50, size=(2, 16, 20))
    kwargs = dict(
        sampling=0.1,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        metadata={"energy": 100e3},
    )
    with abtem.config.set({"precision": precision}):
        array = da.from_array(counts, chunks=(1, 16, 20)) if lazy else counts
        com = DiffractionPatterns(array, **kwargs).center_of_mass()
        expected = np.result_type(np.dtype(precision), np.complex64)
        assert com.array.dtype == expected
        computed = com.compute().array
        assert computed.dtype == expected

    with abtem.config.set({"precision": "float64"}):
        reference = DiffractionPatterns(counts.astype(np.float64), **kwargs)
        reference = reference.center_of_mass().array
    rtol = 1e-5 if precision == "float32" else 1e-12
    np.testing.assert_allclose(computed, reference, rtol=rtol, atol=1e-6)


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
@given(data=st.data(), sigma=st.floats(min_value=0.0, max_value=2.0))
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


# Base shape (3 radial, 4 azimuthal). With the default geometry below, radial bin
# i spans [5 i, 5 i + 5) mrad and azimuthal bin j spans [j pi / 2, (j + 1) pi / 2)
# rad. Limits select the bins whose centers lie in [lower, upper), with the
# azimuth periodic in 2 pi.
#
# The non-uniform array is A[i, j] = 10 i + j, so
#   row sums    R_i = sum_j A[i, j] = 40 i + 6        -> R = (6, 46, 86)
#   column sums C_j = sum_i A[i, j] = 30 + 3 j        -> C = (30, 33, 36, 39)
#   total       = 6 + 46 + 86 = 138
# The second scan position holds 2 A, so every expected value doubles there.
def _polar(
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


def _polar_values(result):
    return np.asarray(result.compute().to_cpu().array)


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
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_polar_measurements_integrate_values(
    values, radial_limits, azimuthal_limits, expected, lazy, device
):
    pm = _polar(values=values, lazy=lazy, device=device)
    result = pm.integrate(
        radial_limits=radial_limits, azimuthal_limits=azimuthal_limits
    )
    np.testing.assert_allclose(_polar_values(result), [expected, 2 * expected])


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
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_polar_measurements_integrate_values_with_offsets(
    radial_limits, azimuthal_limits, expected, lazy, device
):
    pm = _polar(
        radial_offset=20.0, azimuthal_offset=np.pi / 4, lazy=lazy, device=device
    )
    result = pm.integrate(
        radial_limits=radial_limits, azimuthal_limits=azimuthal_limits
    )
    np.testing.assert_allclose(_polar_values(result), [expected, 2 * expected])


def test_polar_measurements_integrate_limits_on_inexact_bin_edges():
    # With radial_sampling = 0.1 the edge 0.3 is bin index 3, although
    # 0.3 / 0.1 == 2.9999999999999996 in floating point. (0, 0.3) must select
    # all three radial bins: total = 138.
    pm = _polar(radial_sampling=0.1)
    result = pm.integrate(radial_limits=(0, 0.3))
    np.testing.assert_allclose(_polar_values(result), [138.0, 276.0])


def test_polar_measurements_integrate_radial_limit_exceeded():
    # outer_angle is 15 mrad; a limit extending over a whole further bin
    # (center 17.5 mrad) asks for data that does not exist.
    with pytest.raises(RuntimeError):
        _polar().integrate(radial_limits=(0, 20))


def test_polar_measurements_integrate_radial():
    # integrate_radial(5, 15) == integrate(radial_limits=(5, 15)) = R_1 + R_2.
    result = _polar().integrate_radial(5, 15)
    np.testing.assert_allclose(_polar_values(result), [132.0, 264.0])


def test_polar_measurements_integrate_detector_regions():
    # Region k is flat index k of the (3, 4) base: region 1 = A[0, 1] = 1,
    # region 4 = A[1, 0] = 10.
    result = _polar().integrate(detector_regions=[1, 4])
    np.testing.assert_allclose(_polar_values(result), [11.0, 22.0])


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
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64])
def test_line_profiles_interpolate_preserves_dtype(lazy, dtype):
    """The 1D spline interpolation allocated its output as float32: a complex
    profile raised 'output must have complex dtype' and a float64 profile was
    silently downgraded."""
    rng = np.random.default_rng(0)
    array = rng.random(40).astype(dtype)
    if np.iscomplexobj(array):
        array = array + 1j * rng.random(40).astype(array.real.dtype)
    profiles = RealSpaceLineProfiles(array, sampling=0.1)
    if lazy:
        profiles = profiles.lazy()

    interpolated = profiles.interpolate(gpts=80)
    if lazy:
        assert interpolated.array.dtype == array.dtype
    result = interpolated.compute().array

    assert result.dtype == array.dtype
    assert result.shape == (80,)
    # Every second output point is an input point (periodic grid, no endpoint).
    assert np.allclose(result[::2], array, atol=1e-5)


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


def _periodic_spline_interpolate(array, gpts, order, lazy):
    images = abtem.Images(array, sampling=0.1)
    if lazy:
        images = images.lazy()
    interpolated = images.interpolate(
        gpts=gpts, method="spline", boundary="periodic", order=order
    )
    return interpolated.compute().array


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("order", [2, 3])
def test_periodic_spline_interpolation_is_invariant_to_whole_pixel_rolls(lazy, order):
    # A periodic interpolant commutes with rolling the image by whole pixels. The
    # image has 40 x 30 pixels and is interpolated to 80 x 90, so one old pixel is
    # 2 new pixels along x and 3 along y and the rolled output is a whole-pixel roll.
    array = np.random.default_rng(0).random((40, 30))

    interpolated_roll = _periodic_spline_interpolate(
        np.roll(array, (5, 7), axis=(0, 1)), (80, 90), order, lazy
    )
    rolled_interpolation = np.roll(
        _periodic_spline_interpolate(array, (80, 90), order, lazy),
        (10, 21),
        axis=(0, 1),
    )

    np.testing.assert_allclose(
        interpolated_roll, rolled_interpolation, rtol=0, atol=1e-12 * array.max()
    )


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("order", [2, 3])
def test_periodic_spline_interpolation_reproduces_a_band_limited_field(lazy, order):
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
    images = abtem.Images(sample(gpts), sampling=sampling)
    if lazy:
        images = images.lazy()
    interpolated = images.interpolate(
        gpts=new_gpts, method="spline", boundary="periodic", order=order
    ).compute()

    expected = sample(new_gpts)
    np.testing.assert_allclose(
        interpolated.array, expected, rtol=0, atol=2e-3 * np.abs(expected).max()
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

    def test_normalize_removes_offset_and_scale_per_member(self):
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

    def test_normalize_line_profiles_per_profile(self):
        """For 1-D members the reduction runs along the single base axis."""
        arr = np.array([[1.0, 3.0, 5.0], [2.0, 2.0, 8.0]])
        profiles = RealSpaceLineProfiles(
            arr, sampling=0.1, ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))]
        )
        normalized = profiles.normalize_ensemble(scale="ptp", shift="min")
        np.testing.assert_allclose(normalized.array, [[0, 0.5, 1], [0, 0, 1]])

    @pytest.mark.parametrize("device", ["cpu", gpu])
    @pytest.mark.parametrize("scale, shift", [("ptp", "min"), ("max", "ptp")])
    def test_lazy_matches_eager(self, scale, shift, device):
        """A lazy measurement, chunked along both base axes, normalizes to the
        eager result for 'ptp' as scale and as shift."""
        array = (np.random.default_rng(0).random((3, 5, 7)) + 0.5).astype(get_dtype())
        axes = [OrdinalAxis(values=(0, 1, 2))]
        eager = Images(array, sampling=0.1, ensemble_axes_metadata=axes)
        eager = eager.copy_to_device(device)
        lazy = Images(
            da.from_array(array, chunks=(1, 3, 4)),
            sampling=0.1,
            ensemble_axes_metadata=axes,
        ).copy_to_device(device)
        expected = asnumpy(eager.normalize_ensemble(scale=scale, shift=shift).array)
        normalized = lazy.normalize_ensemble(scale=scale, shift=shift)
        assert normalized.is_lazy
        computed = normalized.array.compute()
        assert_array_matches_device(computed, device)
        np.testing.assert_allclose(
            asnumpy(computed),
            expected,
            rtol=0,
            atol=1e-6 * np.abs(expected).max(),
        )


def _frequency_index_positions(n, fftshift):
    """Map integer spatial-frequency index -> array position, in numpy's
    fftfreq convention (shifted: zero frequency at n // 2)."""
    freqs = np.fft.fftfreq(n, 1 / n)
    if fftshift:
        freqs = np.fft.fftshift(freqs)
    return {int(round(f)): i for i, f in enumerate(freqs)}


@pytest.mark.parametrize("fftshift", [True, False])
def test_diffraction_patterns_crop_keeps_the_frequencies_within_max_angle(fftshift):
    """Cropping to max_angle keeps exactly the frequencies |m| d_alpha <= max_angle
    along each axis (an odd grid centred on zero frequency), with their original
    values and unchanged sampling, for either fftshift layout."""
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


@pytest.mark.parametrize("cutoff", [np.array(10.0), np.array([10.0])])
def test_block_direct_leaves_an_array_cutoff_in_the_metadata_unchanged(cutoff):
    from abtem.waves import Waves

    rng = np.random.default_rng(0)
    array = (
        rng.standard_normal((100, 100)) + 1j * rng.standard_normal((100, 100))
    ).astype(np.complex64)

    def patterns(semiangle_cutoff):
        waves = Waves(
            array,
            energy=200e3,
            sampling=0.5,
            metadata={"semiangle_cutoff": semiangle_cutoff},
        )
        return waves, waves.diffraction_patterns()

    expected = patterns(10.0)[1].block_direct().array
    waves, diffraction_patterns = patterns(cutoff)

    for _ in range(3):
        np.testing.assert_array_equal(
            diffraction_patterns.block_direct().array, expected
        )

    assert np.asarray(waves.metadata["semiangle_cutoff"]) == 10.0
    assert np.asarray(diffraction_patterns.metadata["semiangle_cutoff"]) == 10.0


@pytest.mark.parametrize("extent", [(19.525, 19.525), (10.0, 14.0)])
def test_block_direct_of_radius_zero_blocks_the_zero_angle_pixel_on_every_grid(extent):
    # The zero-angle coordinate of a float32 grid is on some grids a tiny number of
    # either sign instead of 0, which a radius of exactly 0 does not reach.
    missed = []
    for gpts in range(32, 301):
        shape = (gpts, gpts + 4)
        patterns = DiffractionPatterns(
            np.ones(shape, dtype=np.float32),
            sampling=(1 / extent[0], 1 / extent[1]),
            fftshift=True,
            metadata={"energy": 200e3},
        )
        blocked = patterns.block_direct(radius=0, margin=False).array
        zeroed = [tuple(int(i) for i in z) for z in np.argwhere(blocked == 0)]
        if zeroed != [(shape[0] // 2, shape[1] // 2)]:
            missed.append(gpts)

    assert missed == []


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("fftshift", [False, True])
def test_block_direct_without_a_cutoff_blocks_only_the_zero_angle_pixel(fftshift, lazy):
    array = np.ones((15, 20), dtype=np.float32)
    if lazy:
        array = da.from_array(array)
    patterns = DiffractionPatterns(
        array, sampling=(0.05, 0.04), fftshift=fftshift, metadata={"energy": 100e3}
    )
    assert "semiangle_cutoff" not in patterns.metadata

    blocked = patterns.block_direct().compute()

    zeroed = np.argwhere(blocked.array == 0)
    zero_angle = (7, 10) if fftshift else (0, 0)
    assert [tuple(index) for index in zeroed] == [zero_angle]


def test_block_direct_without_a_cutoff_keeps_the_first_order_reflections():
    from ase.spacegroup import crystal

    atoms = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[3.905] * 3 + [90] * 3,
    )
    potential = abtem.Potential(atoms, sampling=0.1)
    waves = abtem.PlaneWave(energy=100e3).multislice(potential)
    patterns = waves.diffraction_patterns(max_angle=None, fftshift=True).compute()

    blocked = patterns.block_direct()

    center = tuple(n // 2 for n in patterns.shape[-2:])
    first_order = (center[0] + 1, center[1])
    assert patterns.array[first_order] > 0
    assert blocked.array[first_order] == patterns.array[first_order]
    assert blocked.array[center] == 0


@pytest.mark.parametrize("fftshift", [True, False])
def test_reciprocal_coordinates_are_whole_multiples_of_the_sampling(fftshift):
    # Zero frequency is then exactly 0 and opposite frequencies are exact negatives,
    # so bandlimit(inner=0) removes the zero-angle pixel on every grid.
    sampling = (1 / 19.525, 1 / 10.0)
    for gpts in range(32, 301):
        shape = (gpts, gpts + 3)
        patterns = DiffractionPatterns(
            np.ones(shape, dtype=np.float32),
            sampling=sampling,
            fftshift=fftshift,
            metadata={"energy": 200e3},
        )
        for coordinates, n, d in zip(patterns.coordinates, shape, sampling):
            expected = (np.arange(n) - n // 2) * d
            if not fftshift:
                expected = np.fft.ifftshift(expected)
            np.testing.assert_array_equal(np.asarray(coordinates), expected)

        center = (shape[0] // 2, shape[1] // 2) if fftshift else (0, 0)
        assert patterns.bandlimit().array[center] == 0, gpts


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("weighting_function", ["step", "gaussian"])
def test_azimuthal_average_on_a_device_equals_the_reference(
    weighting_function, lazy, device
):
    # The averages were collected in a host array and the radii made with NumPy, so
    # on CuPy every eager or lazy average failed.
    data = np.random.default_rng(1).random((3, 33, 40)).astype(np.float32)
    patterns = DiffractionPatterns(
        data,
        sampling=1 / 20.0,
        fftshift=True,
        metadata={"energy": 200e3},
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1, 2))],
    )

    kx, ky = patterns.angular_coordinates
    radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    width = min(patterns.angular_sampling)
    centers = np.arange(0, -min(min(patterns.angular_limits)), width)
    expected = np.zeros((3, len(centers)))
    for i, center in enumerate(centers):
        if weighting_function == "step":
            mask = np.abs(radius - center) < width
        else:
            mask = np.exp(-((radius - center) ** 2) / (width**2 / 2))
        expected[:, i] = (data * mask).sum(axis=(-2, -1)) / mask.sum()

    on_device = patterns.copy_to_device(device)
    if lazy:
        on_device = on_device.ensure_lazy(chunks=(1, -1, -1))
    result = on_device.azimuthal_average(weighting_function=weighting_function)
    result = result.compute()

    assert_array_matches_device(result.array, device)
    np.testing.assert_allclose(asnumpy(result.array), expected, rtol=1e-5, atol=1e-7)
