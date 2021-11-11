"""Module to describe electron waves and their propagation."""
from copy import copy
from typing import Union, Sequence, Tuple

import dask
import dask.array as da
import numpy as np
from ase import Atoms

from abtem.basic.antialias import HasAntialiasApertureMixin
from abtem.basic.energy import HasAcceleratorMixin
from abtem.basic.grid import HasGridMixin
from abtem.potentials import Potential
from abtem.waves.scan import AbstractScan


class BeamTilt:

    def __init__(self, tilt: Tuple[float, float] = (0., 0.)):
        self._tilt = tilt

    @property
    def tilt(self) -> Tuple[float, float]:
        """Beam tilt [mrad]."""
        return self._tilt

    @tilt.setter
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


class WavesLikeMixin(HasGridMixin, HasAcceleratorMixin, HasBeamTiltMixin, HasAntialiasApertureMixin):

    # @abstractmethod
    # def multislice(self, *args, **kwargs):
    #    pass

    # @abstractmethod
    # def __copy__(self):
    #    pass

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
        self.grid.check_is_defined()
        return [{'label': 'x', 'type': 'real_space', 'sampling': self.sampling[0]},
                {'label': 'y', 'type': 'real_space', 'sampling': self.sampling[1]}]

    @property
    def _fourier_space_axes_metadata(self):
        return [{'label': 'alpha_x', 'type': 'fourier_space', 'sampling': self.angular_sampling[0]},
                {'label': 'alpha_y', 'type': 'fourier_space', 'sampling': self.angular_sampling[1]}]

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


class AbstractScannedWaves(WavesLikeMixin):

    def _compute_chunks(self, dims):
        chunk_size = dask.utils.parse_bytes(dask.config.get('array.chunk-size'))
        chunks = int(np.floor(chunk_size / (2 * 4 * np.prod(self.gpts))))
        chunks = int(np.floor(chunks ** (1 / dims)))
        return (chunks,) * dims + (2,)

    def _validate_detectors(self, detectors):
        if hasattr(detectors, 'detect'):
            detectors = [detectors]
        return detectors

    def _bytes_per_wave(self):
        return 2 * 4 * np.prod(self.gpts)

    def _validate_chunks(self, chunks):
        if chunks == 'auto':
            chunk_size = dask.utils.parse_bytes(dask.config.get('array.chunk-size'))
            return int(chunk_size / self._bytes_per_wave())

        return chunks

    def _validate_positions(self,
                            positions: Union[Sequence, AbstractScan] = None,
                            lazy: bool = True,
                            chunks: Union[int, str] = 'auto'):

        chunks = self._validate_chunks(chunks)

        if hasattr(positions, 'get_positions'):
            return positions.get_positions(lazy=lazy, chunks=chunks), positions.axes_metadata

        if positions is None:
            positions = (self.extent[0] / 2, self.extent[1] / 2)

        if not isinstance(positions, da.core.Array):
            positions = np.array(positions, dtype=np.float32)

        if isinstance(positions, np.ndarray) and lazy:
            positions = da.from_array(positions)

        if len(positions.shape) == 1:
            positions = positions[None]

        if positions.shape[-1] != 2:
            raise ValueError()

        return positions, [{'type': 'positions'}] * (len(positions.shape) - 1)
