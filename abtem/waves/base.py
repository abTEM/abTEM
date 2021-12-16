"""Module to describe electron waves and their propagation."""
from copy import copy
from typing import Union, Sequence, Tuple, List, Dict

import dask
import dask.array as da
import numpy as np
from ase import Atoms

from abtem.core.antialias import HasAntialiasApertureMixin
from abtem.core.dask import _validate_lazy
from abtem.core.energy import HasAcceleratorMixin
from abtem.core.grid import HasGridMixin
from abtem.potentials.potentials import Potential, AbstractPotential
from abtem.waves.scan import AbstractScan
from abtem.waves.multislice import FresnelPropagator
from abtem.waves.tilt import HasBeamTiltMixin
from abtem.measure.detect import AbstractDetector, WavesDetector


class WavesLikeMixin(HasGridMixin, HasAcceleratorMixin, HasBeamTiltMixin, HasAntialiasApertureMixin):

    @property
    def num_base_axes(self) -> int:
        return 2

    @property
    def _base_axes_metadata(self) -> List[Dict]:
        self.grid.check_is_defined()
        return [{'label': 'x', 'type': 'real_space', 'sampling': self.sampling[0]},
                {'label': 'y', 'type': 'real_space', 'sampling': self.sampling[1]}]



    @property
    def antialias_valid_gpts(self) -> Tuple[int, int]:
        return self._valid_rectangle(self.gpts, self.sampling)

    @property
    def antialias_cutoff_gpts(self) -> Tuple[int, int]:
        return self._cutoff_rectangle(self.gpts, self.sampling)

    @property
    def _fourier_space_axes_metadata(self) -> List[Dict]:
        return [{'label': 'alpha_x', 'type': 'fourier_space', 'sampling': self.angular_sampling[0]},
                {'label': 'alpha_y', 'type': 'fourier_space', 'sampling': self.angular_sampling[1]}]

    def _gpts_within_angle(self, angle: Union[None, float, str]) -> Tuple[int, int]:

        if angle is None:
            return self.gpts

        elif not isinstance(angle, str):
            return (int(2 * np.ceil(angle / self.angular_sampling[0])) + 1,
                    int(2 * np.ceil(angle / self.angular_sampling[1])) + 1)

        elif angle == 'cutoff':
            return self.antialias_cutoff_gpts

        elif angle == 'valid':
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
    def angular_sampling(self) -> Tuple[float, float]:
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        return 1 / self.extent[0] * self.wavelength * 1e3, 1 / self.extent[1] * self.wavelength * 1e3

    def _validate_potential(self, potential: Union[Atoms, AbstractPotential]) -> AbstractPotential:
        if isinstance(potential, Atoms):
            potential = Potential(potential)

        self.grid.match(potential)
        return potential


class AbstractScannedWaves(WavesLikeMixin):

    def _compute_chunks(self, dims: int) -> Tuple[int, ...]:
        chunk_size = dask.utils.parse_bytes(dask.config.get('array.chunk-size'))
        chunks = int(np.floor(chunk_size / (2 * 4 * np.prod(self.gpts))))
        chunks = int(np.floor(chunks ** (1 / dims)))
        return (chunks,) * dims + (2,)

    def _validate_detectors(self, detectors) -> List[AbstractDetector]:
        if hasattr(detectors, 'detect'):
            detectors = [detectors]

        if detectors is None:
            detectors = [WavesDetector()]

        return detectors

    def _bytes_per_wave(self) -> int:
        return 2 * 4 * np.prod(self.gpts)

    def _validate_chunks(self, chunks: int) -> int:
        if chunks == 'auto':
            chunk_size = dask.utils.parse_bytes(dask.config.get('array.chunk-size'))
            return int(chunk_size / self._bytes_per_wave())

        return chunks

