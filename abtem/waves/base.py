"""Module to describe electron waves and their propagation."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Union, Sequence, Tuple

import dask
import dask.array as da
import numpy as np
from ase import Atoms

from abtem.basic.antialias import HasAntialiasAperture
from abtem.basic.energy import HasAcceleratorMixin
from abtem.basic.event import HasEventMixin, Event, watched_method
from abtem.basic.grid import HasGridMixin
from abtem.device import HasDeviceMixin
from abtem.measure.detect import AbstractDetector
from abtem.measure.old_measure import Measurement
from abtem.potentials import Potential
from abtem.waves.scan import AbstractScan


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

    @property
    def _base_axes_metadata(self):
        return [{'label': 'x', 'type': 'real_space', 'sampling': self.sampling[0]},
                {'label': 'y', 'type': 'real_space', 'sampling': self.sampling[1]}]

    # def _gpts_in_angle(self, angle):
    #    return tuple(int(2 * np.ceil(angle / d)) + 1 for n, d in zip(self.gpts, self.angular_sampling))

    def _gpts_within_angle(self, angle):

        if angle is None:
            return self.gpts

        if not isinstance(angle, str):
            return tuple(int(2 * np.ceil(angle / d)) + 1 for n, d in zip(self.gpts, self.angular_sampling))

        if angle == 'cutoff':
            return self.antialias_cutoff_gpts

        if angle == 'valid':
            return self.antialias_valid_gpts

        raise ValueError()

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

    def _validate_potential(self, potential):
        if isinstance(potential, Atoms):
            potential = Potential(potential)

        self.grid.match(potential)
        return potential


class AbstractScannedWaves(AbstractWaves):

    def _compute_chunks(self, dims):
        chunk_size = dask.utils.parse_bytes(dask.config.get('array.chunk-size'))
        chunks = int(np.floor(chunk_size / (2 * 4 * np.prod(self.gpts))))
        chunks = int(np.floor(chunks ** (1 / dims)))
        return (chunks,) * dims + (2,)

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

        return positions
