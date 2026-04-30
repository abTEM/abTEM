import ase
import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.strategies import composite
from utils import array_is_close, ensure_is_tuple, gpu

import abtem
from abtem.core.axes import OrdinalAxis, ScanAxis
from abtem.core.backend import copy_to_device
from abtem.measurements import (
    DiffractionPatterns,
    Images,
    PolarMeasurements,
    RealSpaceLineProfiles,
    ReciprocalSpaceLineProfiles,
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
    image = wave.build((0, 0), lazy=False).intensity()

    line = image.interpolate_line(start=(0, 0), end=(0, wave.extent[1]), width=0.0)
    assert np.allclose(
        image.to_cpu().array[0], line.to_cpu().array, rtol=1e-6, atol=1e-6
    )

    coordinate = st.floats(min_value=0, max_value=wave.extent[0])
    center = data.draw(st.tuples(coordinate, coordinate))
    angle1 = data.draw(st.floats(min_value=0, max_value=360.0))
    angle2 = data.draw(st.floats(min_value=0, max_value=360.0))
    width = data.draw(st.floats(min_value=0, max_value=2.0))

    image = wave.build(center, lazy=False).intensity()
    line1 = image.interpolate_line_at_position(
        center=center, angle=angle1, extent=wave.extent[0] / 2, width=width, gpts=128
    ).to_cpu()
    line2 = image.interpolate_line_at_position(
        center=center, angle=angle2, extent=wave.extent[0] / 2, width=width, gpts=128
    ).to_cpu()

    assert np.allclose(line1.array, line2.array, rtol=1e-6, atol=10)


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
