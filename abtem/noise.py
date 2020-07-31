import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _pixel_times(dwell_time, flyback_time, shape):
    line_time = (dwell_time * shape[1]) + flyback_time
    slow_time = np.tile(np.linspace(line_time, shape[0] * line_time, shape[0])[:, None], (1, shape[1]))
    fast_time = np.tile(np.linspace((line_time - flyback_time) / shape[1],
                                    line_time - flyback_time, shape[1]), (shape[0], 1))
    return slow_time + fast_time


def _single_axis_distortion(time, max_frequency, num_components):
    frequencies = np.random.rand(num_components, 1, 1) * max_frequency
    amplitudes = np.random.rand(num_components, 1, 1) / np.sqrt(frequencies)
    displacements = np.random.rand(num_components, 1, 1) / frequencies
    return (amplitudes * np.sin(2 * np.pi * (time + displacements) * frequencies)).sum(axis=0)


def _make_displacement_field(time, max_frequency, num_components, rms_power):
    profile_x = _single_axis_distortion(time, max_frequency, num_components)
    profile_y = _single_axis_distortion(time, max_frequency, num_components)

    x_mag_deviation = np.gradient(profile_x, axis=1)
    y_mag_deviation = np.gradient(profile_y, axis=0)

    frame_mag_deviation = (1 + x_mag_deviation) * (1 + y_mag_deviation) - 1
    frame_mag_deviation = np.sqrt(np.mean(frame_mag_deviation ** 2))

    profile_x *= rms_power / (235.5 * frame_mag_deviation)
    profile_y *= rms_power / (235.5 * frame_mag_deviation)

    return profile_x, profile_y


def _apply_displacement_field(image, distortion_x, distortion_y):
    x = np.arange(0, image.shape[0])
    y = np.arange(0, image.shape[1])

    interpolating_function = RegularGridInterpolator([x, y], image, fill_value=None)

    y, x = np.meshgrid(y, x)
    p = np.array([(x + distortion_x).ravel(), (y + distortion_y).ravel()]).T

    p[:, 0] = np.clip(p[:, 0], 0, x.max())
    p[:, 1] = np.clip(p[:, 1], 0, y.max())

    warped = interpolating_function(p)
    return warped.reshape(image.shape)


def add_scan_noise(image, dwell_time, flyback_time, max_frequency, rms_power, num_components=200):
    image = image.copy()
    time = _pixel_times(dwell_time, flyback_time, image.array.T.shape)
    displacement_x, displacement_y = _make_displacement_field(time, max_frequency, num_components, rms_power)
    array = _apply_displacement_field(image.array[:].T, displacement_x, displacement_y)
    image.array[:] = array.T
    return image


def poisson_noise(self, dose):
    pixel_area = np.product([calibration.sampling for calibration in self.calibrations])
    new_copy = copy(self)
    array = new_copy.array
    array[:] = array / np.sum(array) * dose * pixel_area * np.prod(array.shape)
    array[:] = np.random.poisson(array).astype(np.float)
    return new_copy