from numbers import Number

import hypothesis.extra.numpy as numpy_st
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck, reproduce_failure
from hypothesis.strategies import composite

from abtem import Probe
from abtem.core.axes import OrdinalAxis
from abtem.core.backend import cp
from abtem.measure.measure import stack_measurements, Images, from_zarr
from strategies import core as core_st
from strategies import detectors as detectors_st
from strategies import measurements as measurements_st
from utils import ensure_is_tuple, gpu, array_is_close

all_measurements = {'images': measurements_st.images,
                    'diffraction_patterns': measurements_st.diffraction_patterns,
                    'line_profiles': measurements_st.line_profiles,
                    'polar_measurements': measurements_st.polar_measurements
                    }


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('measurement', list(all_measurements.keys()))
def test_indexing(data, measurement, lazy, device):
    measurement = data.draw(all_measurements[measurement](lazy=lazy, device=device))
    indices = data.draw(numpy_st.basic_indices(measurement.extra_axes_shape, allow_newaxis=False, allow_ellipsis=False))

    if isinstance(indices, Number):
        num_lost_axes = 1
    elif isinstance(indices, slice):
        num_lost_axes = 0
    else:
        num_lost_axes = sum(1 for i in indices if isinstance(i, Number))

    assert len(measurement[indices].shape) == len(measurement.shape) - num_lost_axes


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('measurement', list(all_measurements.keys()))
def test_copy_equals_measurement(data, measurement, lazy, device):
    measurement = data.draw(all_measurements[measurement](lazy=lazy, device=device))

    measurement_copy = measurement.copy()
    assert measurement_copy == measurement
    assert measurement_copy.array is not measurement.array
    assert measurement_copy._extra_axes_metadata is not measurement._extra_axes_metadata
    assert measurement_copy.metadata is not measurement.metadata


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('measurement', list(all_measurements.keys()))
def test_squeeze(data, measurement, lazy, device):
    measurement = data.draw(all_measurements[measurement](lazy=lazy, device=device))
    squeezed_measurement = measurement.squeeze()
    assert all(tuple(n > 1 for n in squeezed_measurement.extra_axes_shape))


@given(data=st.data())
@pytest.mark.parametrize('method', ['__add__', '__sub__', '__mul__', '__truediv__'])
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('measurement', list(all_measurements.keys()))
def test_add_subtract(data, measurement, method, lazy, device):
    measurement = data.draw(all_measurements[measurement](lazy=lazy, device=device))
    new_measurement = getattr(measurement, method)(measurement.copy())
    assert new_measurement.array is not measurement.array


@given(data=st.data())
@pytest.mark.parametrize('method', ['__iadd__', '__isub__', '__imul__', '__itruediv__'])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('measurement', list(all_measurements.keys()))
def test_inplace_add_subtract(data, measurement, method, device):
    measurement = data.draw(all_measurements[measurement](lazy=False, device=device))
    new_measurement = getattr(measurement, method)(measurement.copy())
    assert new_measurement.array is measurement.array


@given(data=st.data())
@pytest.mark.parametrize('method', ['mean', 'sum', 'std'])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('measurement', list(all_measurements.keys()))
def test_reduce(data, measurement, method, device):
    measurement = data.draw(all_measurements[measurement](lazy=False, device=device))

    axes_indices = st.integers(min_value=0, max_value=max(len(measurement.extra_axes) - 1, 0))
    axes_indices = st.lists(elements=axes_indices, min_size=0, max_size=len(measurement.extra_axes), unique=True)
    axes_indices = data.draw(axes_indices)

    axes = tuple(measurement.extra_axes[i] for i in axes_indices)
    num_lost_dims = len(axes)
    new_measurement = getattr(measurement, method)(axes)
    assert len(new_measurement.shape) == len(measurement.shape) - num_lost_dims


@settings(deadline=None, max_examples=40, print_blob=True)
@given(data=st.data(), url=detectors_st.temporary_path(allow_none=False))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('measurement', list(all_measurements.keys()))
@pytest.mark.skipif(cp is None, reason="no gpu")
def test_to_cpu(data, measurement, url, lazy):
    measurement = data.draw(all_measurements[measurement](lazy=lazy, device='gpu'))
    measurement = measurement.to_cpu()
    if measurement.is_lazy:
        assert isinstance(measurement.array._meta, np.ndarray)
    measurement.compute()
    assert isinstance(measurement.array, np.ndarray)


@given(data=st.data(), url=detectors_st.temporary_path(allow_none=False))
@pytest.mark.parametrize('lazy', [False, True])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('measurement', list(all_measurements.keys()))
def test_to_zarr_from_zarr(data, measurement, url, lazy, device):
    measurement = data.draw(all_measurements[measurement](lazy=lazy, device=device))
    measurement.to_zarr(url)

    imported_measurement = measurement.__class__.from_zarr(url)
    imported_measurement.compute()
    assert imported_measurement == measurement.to_cpu().compute()


@composite
def gpts_or_sampling(draw):
    return draw(st.one_of(st.fixed_dictionaries({'gpts': core_st.gpts(), 'sampling': st.none()}),
                          st.fixed_dictionaries({'gpts': st.none(), 'sampling': core_st.sampling()})))


@given(data=st.data(), gpts_or_sampling=gpts_or_sampling())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('method', ['spline', 'fft'])
def test_interpolate_images(data, gpts_or_sampling, lazy, device, method):
    measurement = data.draw(measurements_st.images(lazy=lazy, device=device))
    interpolated = measurement.interpolate(**gpts_or_sampling, method=method)
    assert np.allclose(interpolated.extent, measurement.extent)
    if gpts_or_sampling['gpts']:
        assert interpolated.base_shape == ensure_is_tuple(gpts_or_sampling['gpts'], 2)
    elif gpts_or_sampling['sampling']:
        sampling = ensure_is_tuple(gpts_or_sampling['sampling'], 2)
        adjusted_sampling = tuple(l / np.ceil(l / d) for d, l in zip(sampling, measurement.extent))
        assert np.allclose(interpolated.sampling, adjusted_sampling)


@given(data=st.data(),
       tile=st.tuples(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3)))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_tile_images(data, tile, lazy, device):
    measurement = data.draw(measurements_st.images(lazy=lazy, device=device))
    tiled = measurement.tile(tile)
    assert np.allclose(np.array(measurement.extent) * tile, tiled.extent)
    assert tuple(n * t for n, t in zip(measurement.base_shape, tile)) == tiled.base_shape


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('measurement', list(all_measurements.keys()))
def test_stacks_with_self(data, measurement, lazy, device):
    measurement = data.draw(all_measurements[measurement](lazy=lazy, device=device))
    stacked_measurement = stack_measurements((measurement, measurement), axes_metadata=OrdinalAxis())
    stacked_measurement.compute()


@composite
def sigma(draw):
    sigma = st.floats(min_value=0., max_value=5.)
    return draw(st.one_of(st.tuples(sigma, sigma), sigma))


@given(data=st.data(), sigma=sigma())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_gaussian_filter_images(data, sigma, lazy, device):
    if lazy is True and device == gpu.values[0]:
        return

    measurement = data.draw(measurements_st.images(lazy=lazy, device=device))
    assume(all(n > 1 for n in measurement.base_shape))
    filtered = measurement.gaussian_filter(sigma)
    filtered.compute()
    measurement.compute()

    if np.any(np.array(sigma)) > 1e-6:
        assert not np.allclose(filtered.array, measurement.array)


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_diffractograms(data, lazy, device):
    measurement = data.draw(measurements_st.images(lazy=lazy, device=device))
    measurement.diffractograms()


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_images_interpolate_line(data, lazy, device):
    wave = Probe(energy=100e3, semiangle_cutoff=30, extent=20, gpts=256, device=device)
    image = wave.build((0, 0), lazy=False).intensity()

    line = image.interpolate_line(start=(0, 0), end=(0, wave.extent[1]), width=0.)
    assert np.allclose(image.array[0], line.array)

    coordinate = st.floats(min_value=0, max_value=wave.extent[0])
    center = data.draw(st.tuples(coordinate, coordinate))
    angle1 = data.draw(st.floats(min_value=0, max_value=360.))
    angle2 = data.draw(st.floats(min_value=0, max_value=360.))
    width = data.draw(st.floats(min_value=0, max_value=2.))

    image = wave.build(center, lazy=False).intensity()
    line1 = image.interpolate_line_at_position(center=center, angle=angle1, extent=wave.extent[0] / 2, width=width,
                                               gpts=32)
    line2 = image.interpolate_line_at_position(center=center, angle=angle2, extent=wave.extent[0] / 2, width=width,
                                               gpts=32)

    assert np.allclose(line1.array, line2.array, rtol=1e-6, atol=1e-6)


@given(data=st.data(), dose=core_st.sensible_floats(min_value=1e8, max_value=1e9))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('measurement', ['images', 'diffraction_patterns'])
def test_poisson_noise(data, measurement, dose, lazy, device):
    measurement = data.draw(all_measurements[measurement](lazy=lazy, device=device, min_value=.5, min_side=16))

    assume(isinstance(measurement, Images) or len(measurement.scan_sampling) == 2)
    noisy = measurement.poisson_noise(dose=dose, samples=16).compute()

    if isinstance(measurement, Images):
        area = np.prod(measurement.extent)
        expected_total_dose = area * dose * np.prod(measurement.extra_axes_shape)
        actual_total_dose = (noisy.array.mean(axis=0)).sum() / measurement.array.mean()
    else:
        area = np.prod(measurement.scan_extent)
        expected_total_dose = area * dose * np.prod(measurement.extra_axes_shape[:-2])
        actual_total_dose = (noisy.array.mean(axis=0) / measurement.array.sum((-2, -1), keepdims=True)).sum()

    assert np.allclose(expected_total_dose, actual_total_dose, rtol=0.1)


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_diffraction_patterns_polar_binning(data, lazy, device):
    measurement = data.draw(measurements_st.diffraction_patterns(lazy=lazy, device=device, min_side=16))

    nbins_radial = data.draw(st.integers(min_value=1, max_value=min(measurement.base_shape)))

    nbins_azimuthal = data.draw(st.integers(min_value=1, max_value=min(measurement.base_shape)))

    outer = data.draw(core_st.sensible_floats(min_value=min(min(measurement.max_angles),
                                                            max(measurement.angular_sampling)),
                                              max_value=min(measurement.max_angles)))

    inner = data.draw(core_st.sensible_floats(min_value=0.,
                                              max_value=max(0., outer - max(measurement.angular_sampling))))

    rotation = data.draw(core_st.sensible_floats(min_value=0., max_value=360.))

    measurement.polar_binning(nbins_radial=nbins_radial,
                              nbins_azimuthal=nbins_azimuthal,
                              inner=inner,
                              outer=outer,
                              rotation=rotation)

    step_size = data.draw(core_st.sensible_floats(min_value=min(measurement.angular_sampling),
                                                  max_value=max(min(measurement.angular_sampling),
                                                                outer - inner)))

    measurement.radial_binning(step_size=step_size,
                               inner=inner,
                               outer=outer)


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_diffraction_patterns_polar_binning(data, lazy, device):
    measurement = data.draw(measurements_st.diffraction_patterns(lazy=lazy, device=device, min_side=16))

    nbins_radial = data.draw(st.integers(min_value=1, max_value=min(measurement.base_shape)))

    nbins_azimuthal = data.draw(st.integers(min_value=1, max_value=min(measurement.base_shape)))

    outer = data.draw(core_st.sensible_floats(min_value=min(min(measurement.max_angles),
                                                            max(measurement.angular_sampling)),
                                              max_value=min(measurement.max_angles)))

    inner = data.draw(core_st.sensible_floats(min_value=0.,
                                              max_value=max(0., outer - max(measurement.angular_sampling))))

    rotation = data.draw(core_st.sensible_floats(min_value=0., max_value=360.))

    measurement.polar_binning(nbins_radial=nbins_radial,
                              nbins_azimuthal=nbins_azimuthal,
                              inner=inner,
                              outer=outer,
                              rotation=rotation)

    step_size = data.draw(core_st.sensible_floats(min_value=min(measurement.angular_sampling),
                                                  max_value=max(min(measurement.angular_sampling),
                                                                outer - inner)))

    measurement.radial_binning(step_size=step_size,
                               inner=inner,
                               outer=outer)


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_diffraction_patterns_center_of_mass(data, lazy, device):
    measurement = data.draw(measurements_st.diffraction_patterns(lazy=lazy, device=device, min_side=16))
    assume(len(measurement.scan_sampling) > 0)
    measurement.center_of_mass().compute()


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_diffraction_patterns_integrated_center_of_mass(data, lazy, device):
    measurement = data.draw(measurements_st.diffraction_patterns(lazy=lazy, device=device, min_side=16))
    assume(len(measurement.scan_sampling) > 1)
    measurement.integrated_center_of_mass().compute()


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_diffraction_patterns_integrated_center_of_mass(data, lazy, device):
    measurement = data.draw(measurements_st.diffraction_patterns(lazy=lazy, device=device, min_side=16))
    assume(len(measurement.scan_sampling) > 1)
    measurement.integrated_center_of_mass().compute()


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_diffraction_patterns_bandlimit(data, lazy, device):
    measurement = data.draw(measurements_st.diffraction_patterns(lazy=lazy, device=device, min_side=16))
    outer = data.draw(core_st.sensible_floats(min_value=0., max_value=min(measurement.max_angles)))
    inner = data.draw(core_st.sensible_floats(min_value=0., max_value=outer))
    measurement.bandlimit(inner, outer).compute()
    measurement.block_direct().compute()


@settings(deadline=None, max_examples=10)
@given(data=st.data(), sigma=st.floats(min_value=0., max_value=2.))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_diffraction_patterns_gaussian_source_size(data, sigma, lazy, device):
    measurement = data.draw(measurements_st.diffraction_patterns(lazy=lazy, device=device, min_side=16))
    assume(measurement.num_scan_axes > 1)
    measurement.gaussian_source_size(sigma).compute()


@settings(deadline=None, max_examples=10)
@given(data=st.data(), sigma=st.floats(min_value=0., max_value=2.))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_diffraction_patterns_gaussian_source_size(data, sigma, lazy, device):
    measurement = data.draw(measurements_st.diffraction_patterns(lazy=lazy, device=device, min_side=16))
    assume(measurement.num_scan_axes > 1)
    measurement.gaussian_source_size(sigma).compute()


@settings(suppress_health_check=(HealthCheck.data_too_large,))
@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_polar_measurements_integrate(data, lazy, device):
    measurement = data.draw(measurements_st.polar_measurements(lazy=lazy, device=device))
    assume(measurement.num_scan_axes > 0)

    radial_outer = data.draw(core_st.sensible_floats(min_value=0., max_value=measurement.outer_angle))
    radial_inner = data.draw(core_st.sensible_floats(min_value=0., max_value=radial_outer))
    radial_limits = data.draw(st.one_of(st.just((radial_inner, radial_outer)), st.none()))

    azimuthal_outer = data.draw(core_st.sensible_floats(min_value=0., max_value=360.))
    azimuthal_inner = data.draw(core_st.sensible_floats(min_value=0., max_value=azimuthal_outer))
    azimuthal_limits = data.draw(st.one_of(st.just((azimuthal_inner, azimuthal_outer)), st.none()))

    measurement.integrate(radial_limits=radial_limits, azimuthal_limits=azimuthal_limits).compute()
    measurement.integrate_radial(radial_inner, radial_outer).compute()

    max_region = int(np.prod(tuple(n - 1 for n in measurement.shape[-2:])))
    detector_regions = st.lists(min_size=0, max_size=max_region,
                                elements=st.integers(min_value=0, max_value=max_region),
                                unique=True)

    measurement.integrate(detector_regions=data.draw(detector_regions)).compute()


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_line_profiles_interpolate(data, lazy, device):
    measurement = data.draw(measurements_st.line_profiles(lazy=lazy, device=device))
    measurement.interpolate().compute()


@given(data=st.data(), reps=st.integers(min_value=1, max_value=3))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_line_profiles_tile(data, reps, lazy, device):
    measurement = data.draw(measurements_st.line_profiles(lazy=lazy, device=device))
    measurement.tile(reps).compute()


def test_line_profiles_interpolate_comparison():
    images = Images.from_zarr('data/silicon_image.zarr').compute().interpolate(.2)
    assert np.allclose(images.interpolate_line().interpolate(.01).array,
                       images.interpolate(.01).interpolate_line().array, rtol=.01)


def test_interpolate_periodic_spline_and_fft():
    images = Images.from_zarr('data/silicon_image.zarr')
    spline_interpolated = images.interpolate(method='spline', sampling=.05, boundary='periodic', order=5)
    fft_interpolated = images.interpolate(method='fft', sampling=.05)
    array_is_close(spline_interpolated.array, fft_interpolated.array, rel_tol=0.01)


@given(gpts=st.integers(min_value=16, max_value=32), extent=st.floats(min_value=5, max_value=10))
def test_diffraction_patterns_interpolate_uniform(gpts, extent):
    probe = Probe(energy=100e3, semiangle_cutoff=20, extent=extent, gpts=gpts)
    diffraction_patterns = probe.build().diffraction_patterns(max_angle=None)
    probe.gpts = (gpts * 2, gpts)
    probe.extent = (extent * 2, extent)
    interpolated_diffraction_patterns = probe.build().diffraction_patterns(max_angle=None).interpolate('uniform')
    assert np.allclose(interpolated_diffraction_patterns.array, diffraction_patterns.array)


@given(sigma=st.floats(min_value=.1, max_value=.5),
       outer=st.floats(min_value=10., max_value=100))
def test_gaussian_source_size_order(sigma, outer):
    diffraction_patterns = from_zarr('data/silicon_diffraction_patterns.zarr').compute()
    image1 = diffraction_patterns.gaussian_source_size(sigma).integrate_radial(0, outer)
    image2 = diffraction_patterns.integrate_radial(0, outer).gaussian_filter(sigma)
    assert np.allclose(image1.array, image2.array)
