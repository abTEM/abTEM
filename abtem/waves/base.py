"""Module to describe electron waves and their propagation."""
from typing import Union, Sequence, Tuple

import dask
import dask.array as da
import numpy as np
from abc import ABCMeta, abstractmethod
from abtem.device import HasDeviceMixin
from abtem.measure.measure import Measurement
from abtem.utils.antialias import HasAntialiasAperture
from abtem.utils.energy import HasAcceleratorMixin
from abtem.utils.event import HasEventMixin, Event, watched_method
from abtem.utils.grid import HasGridMixin
from abtem.measure.detect import AbstractDetector
from abtem.waves.scan import AbstractScan
from copy import copy


class BeamTilt(HasEventMixin):

    def __init__(self, tilt: Tuple[float, float] = (0., 0.)):
        self._tilt = tilt
        self._event = Event()

    @property
    def tilt(self) -> Tuple[float, float]:
        """Beam tilt [mrad]."""
        return self._tilt

    @tilt.setter
    @watched_method('_event')
    def tilt(self, value: Tuple[float, float]):
        self._tilt = value


class HasBeamTiltMixin:
    _beam_tilt: BeamTilt

    @property
    def tilt(self) -> Tuple[float, float]:
        return self._beam_tilt.tilt

    @tilt.setter
    def tilt(self, value: Tuple[float, float]):
        self.tilt = value


class AbstractWaves(HasGridMixin, HasAcceleratorMixin, HasDeviceMixin, HasBeamTiltMixin, HasAntialiasAperture,
                    metaclass=ABCMeta):

    @abstractmethod
    def multislice(self, *args, **kwargs):
        pass

    @abstractmethod
    def __copy__(self):
        pass

    def copy(self):
        """Make a copy."""
        return copy(self)

    @property
    def antialias_valid_gpts(self):
        return self._valid_rectangle(self.gpts, self.sampling)

    @property
    def antialias_cutoff_gpts(self):
        return self._cutoff_rectangle(self.gpts, self.sampling)

    def _gpts_in_angle(self, angle):
        return tuple(int(2 * np.ceil(angle / d)) + 1 for n, d in zip(self.gpts, self.angular_sampling))

    @property
    def cutoff_angles(self) -> Tuple[float, float]:
        return (self.antialias_cutoff_gpts[0] // 2 * self.angular_sampling[0],
                self.antialias_cutoff_gpts[1] // 2 * self.angular_sampling[1])

    @property
    def rectangle_cutoff_angles(self) -> Tuple[float, float]:
        return (self.antialias_valid_gpts[0] // 2 * self.angular_sampling[0],
                self.antialias_valid_gpts[1] // 2 * self.angular_sampling[1])

    @property
    def angular_sampling(self):
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        return tuple(1 / l * self.wavelength * 1e3 for l in self.extent)


class _Scanable(AbstractWaves):

    def compute_chunks(self):
        chunk_size = dask.utils.parse_bytes(dask.config.get('array.chunk-size'))
        return int(np.floor(chunk_size / (2 * 4 * np.prod(self.gpts))))

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

    def _validate_positions(self, positions: Union[Sequence, AbstractScan] = None):
        if isinstance(positions, AbstractScan):
            positions = positions.get_positions()

        if positions is None:
            positions = (self.extent[0] / 2, self.extent[1] / 2)

        if not isinstance(positions, da.core.Array):
            positions = np.array(positions, dtype=np.float32)

        if len(positions.shape) == 1:
            positions = positions[None]

        if positions.shape[-1] != 2:
            raise ValueError()

        if not isinstance(positions, da.core.Array):
            dims = len(positions.shape) - 1
            chunks = int(np.floor(self.compute_chunks() ** (1 / dims)))
            positions = da.from_array(positions, chunks=(chunks,) * dims + (self.compute_chunks(),))

        return positions
