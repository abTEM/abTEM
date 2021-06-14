"""Module to describe electron waves and their propagation."""
from typing import Union, Sequence, Tuple

import dask.array as da
import numpy as np

from abtem.base_classes import HasGridMixin, HasAcceleratorMixin, HasBeamTiltMixin, HasAntialiasAperture
from abtem.detect import AbstractDetector
from abtem.device import HasDeviceMixin
from abtem.device import get_array_module_from_device
from abtem.measure.old_measure import Measurement
from abtem.utils.antialias import AntialiasFilter
from abtem.utils.coordinates import spatial_frequencies


class _WavesLike(HasGridMixin, HasAcceleratorMixin, HasDeviceMixin, HasBeamTiltMixin, HasAntialiasAperture):

    # def __init__(self, tilt: Tuple[float, float] = None, antialiasing_aperture: Tuple[float, float] = None):
    #     self.tilt = tilt
    #
    #     if antialiasing_aperture is None:
    #         antialiasing_aperture = (2 / 3.,) * 2
    #
    #     self.antialiasing_aperture = antialiasing_aperture

    # @property
    # @abstractmethod
    # def tilt(self):
    #     pass

    # @property
    # @abstractmethod
    # def antialias_aperture(self):
    #     pass

    @property
    def cutoff_scattering_angles(self) -> Tuple[float, float]:
        interpolated_grid = self._interpolated_grid
        kcut = [1 / d / 2 * a for d, a in zip(interpolated_grid.sampling, self.antialias_aperture)]
        kcut = min(kcut)
        kcut = (
            np.ceil(2 * interpolated_grid.extent[0] * kcut) / (
                    2 * interpolated_grid.extent[0]) * self.wavelength * 1e3,
            np.ceil(2 * interpolated_grid.extent[1] * kcut) / (
                    2 * interpolated_grid.extent[1]) * self.wavelength * 1e3)
        return kcut

    @property
    def rectangle_cutoff_scattering_angles(self) -> Tuple[float, float]:
        rolloff = AntialiasFilter.rolloff
        interpolated_grid = self._interpolated_grid
        kcut = [(a / (d * 2) - rolloff) / np.sqrt(2) for d, a in
                zip(interpolated_grid.sampling, self.antialias_aperture)]

        kcut = min(kcut)
        kcut = (
            np.floor(2 * interpolated_grid.extent[0] * kcut) / (
                    2 * interpolated_grid.extent[0]) * self.wavelength * 1e3,
            np.floor(2 * interpolated_grid.extent[1] * kcut) / (
                    2 * interpolated_grid.extent[1]) * self.wavelength * 1e3)
        return kcut

    @property
    def angular_sampling(self):
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        return tuple([1 / l * self.wavelength * 1e3 for l in self._interpolated_grid.extent])

    def get_spatial_frequencies(self):
        xp = get_array_module_from_device(self.device)
        kx, ky = spatial_frequencies(self.grid.gpts, self.grid.sampling)
        # TODO : should beam tilt be added here?
        kx = xp.asarray(kx)
        ky = xp.asarray(ky)
        return kx, ky

    def get_scattering_angles(self):
        kx, ky = self.get_spatial_frequencies()
        alpha, phi = polar_coordinates(kx * self.wavelength, ky * self.wavelength)
        return alpha, phi

    @property
    def _interpolated_grid(self):
        return self.grid

    def downsampled_gpts(self, max_angle: Union[float, str]):
        interpolated_gpts = self._interpolated_grid.gpts

        if max_angle is None:
            gpts = interpolated_gpts

        elif isinstance(max_angle, str):
            if max_angle == 'limit':
                cutoff_scattering_angle = self.cutoff_scattering_angles
            elif max_angle == 'valid':
                cutoff_scattering_angle = self.rectangle_cutoff_scattering_angles
            else:
                raise RuntimeError()

            angular_sampling = self.angular_sampling
            gpts = (int(np.ceil(cutoff_scattering_angle[0] / angular_sampling[0] * 2 - 1e-12)),
                    int(np.ceil(cutoff_scattering_angle[1] / angular_sampling[1] * 2 - 1e-12)))
        else:
            try:
                gpts = [int(2 * np.ceil(max_angle / d)) + 1 for n, d in zip(interpolated_gpts, self.angular_sampling)]
            except:
                raise RuntimeError()

        return (min(gpts[0], interpolated_gpts[0]), min(gpts[1], interpolated_gpts[1]))


class _Scanable(_WavesLike):

    def _validate_detectors(self, detectors):
        if isinstance(detectors, AbstractDetector):
            detectors = [detectors]
        return detectors

    def _validate_scan_measurements(self, detectors, scan, measurements=None):

        if isinstance(measurements, Measurement):
            if len(detectors) > 1:
                raise ValueError('more than one detector, measurements must be mapping or None')

            return {detectors[0]: measurements}

        if measurements is None:
            measurements = {}

        for detector in detectors:
            if detector not in measurements.keys():
                measurements[detector] = detector.allocate_measurement(self, scan)
            # if not set(measurements.keys()) == set(detectors):
            #    raise ValueError('measurements dict keys does not match detectors')
        # else:
        #    raise ValueError('measurements must be Measurement or dict of AbtractDetector: Measurement')
        return measurements

    def _validate_positions(self, positions: Sequence = None) -> np.ndarray:
        if positions is None:
            positions = da.array((self.extent[0] / 2, self.extent[1] / 2), dtype=np.float32)
        else:
            positions = da.array(positions, dtype=np.float32)

        if len(positions.shape) == 1:
            positions = np.expand_dims(positions, axis=0)

        if positions.shape[1] != 2:
            raise ValueError('positions must be of shape Nx2')

        return positions
