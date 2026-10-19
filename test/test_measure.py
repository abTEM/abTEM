import ase
import dask.array as da
import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.strategies import composite
from utils import array_is_close, ensure_is_tuple, gpu

import abtem
from abtem.core.axes import OrdinalAxis, ScanAxis
from abtem.core.backend import asnumpy, copy_to_device
from abtem.core.energy import energy2wavelength
from abtem.measurements import (
    DiffractionPatterns,
    Images,
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
@pytest.mark.parametrize("scalar", [2.0, -0.5])
def test_reflected_arithmetic_with_a_scalar(scalar, lazy):
    # Oracle: numpy's own reflected operators on the plain array. The array is
    # not symmetric under any of the operations, so e.g. `scalar / m` computed
    # as `m / scalar` (as __rtruediv__ = __truediv__ used to do) fails.
    array = np.array([[1.0, 2.0, 4.0], [8.0, 0.5, 0.25]])
    if lazy:
        measurement = Images(da.from_array(array, chunks=(1, 3)), sampling=(0.1, 0.2))
    else:
        measurement = Images(array, sampling=(0.1, 0.2))

    for result, expected in (
        (scalar / measurement, scalar / array),
        (scalar * measurement, scalar * array),
    ):
        assert isinstance(result, Images)
        np.testing.assert_allclose(result.compute().array, expected, rtol=1e-6)


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
    measurement = data.draw(
        abtem_st.diffraction_patterns(
            lazy=lazy, min_scan_dims=1, device=device, min_base_side=16
        )
    )
    assume(len(_scan_sampling(measurement)) > 0)

    print(measurement.shape, measurement.axes_metadata)

    measurement.center_of_mass().compute()


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("captured_fraction", [1.0, 0.5, 0.1])
def test_diffraction_patterns_center_of_mass_is_normalized(device, captured_fraction):
    # A center of mass is a normalized (intensity-weighted average) quantity, so
    # scaling the total intensity of a diffraction pattern must not change the
    # computed center of mass. This did not hold before the sum was normalized by
    # the total captured intensity: https://github.com/abTEM/abTEM/discussions/402
    gpts = 33
    sampling = 0.4436
    shift = (-10, 10)

    y, x = np.mgrid[0:gpts, 0:gpts]
    disk = np.exp(-((y - gpts // 2) ** 2 + (x - gpts // 2) ** 2) / (2 * 3.0**2))
    disk = np.roll(disk, shift, axis=(0, 1))
    disk = disk / disk.sum() * captured_fraction
    disk = copy_to_device(disk.astype(np.float32), device)

    measurement = DiffractionPatterns(disk, sampling=sampling, fftshift=True)

    com = measurement.center_of_mass(units="1/Å").array

    expected = (shift[0] * sampling) + 1.0j * (shift[1] * sampling)
    assert abs(com - expected) < 0.1 * abs(expected)


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
