import hashlib
import itertools
import operator
from abc import abstractmethod
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from functools import partial, reduce
from typing import Union, Tuple, Dict, List

import dask
import dask.array as da
import numpy as np
from ase import Atoms

from abtem.core.axes import OrdinalAxis, RealSpaceAxis, AxisMetadata
from abtem.core.backend import get_array_module, cp, validate_device
from abtem.core.blockwise import ensemble_blockwise
from abtem.core.dask import validate_lazy, ComputableList, validate_chunks, chunk_range, equal_sized_chunks
from abtem.core.energy import Accelerator, HasAcceleratorMixin
from abtem.core.grid import Grid, HasGridMixin
from abtem.core.intialize import initialize
from abtem.core.utils import subdivide_into_chunks
from abtem.measure.detect import AbstractDetector, validate_detectors
from abtem.measure.measure import AbstractMeasurement
from abtem.potentials.potentials import AbstractPotential, validate_potential
from abtem.waves.base import WavesLikeMixin
from abtem.waves.multislice import multislice, allocate_multislice_measurements
from abtem.waves.prism_utils import prism_wave_vectors, plane_waves, wrapped_crop_2d, prism_coefficients, minimum_crop, \
    batch_crop_2d
from abtem.waves.scan import AbstractScan, GridScan, LineScan, CustomScan, validate_scan
from abtem.waves.tilt import BeamTilt
from abtem.waves.transfer import CTF
from abtem.waves.waves import Waves, Probe, finalize_lazy_measurements


class AbstractSMatrix(WavesLikeMixin):
    planewave_cutoff: float
    interpolation: Tuple[int, int]

    @property
    @abstractmethod
    def interpolated_gpts(self):
        pass

    @property
    def interpolated_extent(self):
        return self.interpolated_gpts[0] * self.sampling[0], self.interpolated_gpts[1] * self.sampling[1]

    @property
    def interpolated_antialias_cutoff_gpts(self):
        if self.antialias_cutoff_gpts is None:
            return None

        return (self.antialias_cutoff_gpts[0] // self.interpolation[0],
                self.antialias_cutoff_gpts[1] // self.interpolation[1])

    def comparable_probe(self, ctf: CTF = None, interpolated: bool = True):
        gpts = self.interpolated_gpts if interpolated else self.gpts
        extent = self.interpolated_extent if interpolated else self.extent
        antialias_cutoff_gpts = self.interpolated_antialias_cutoff_gpts if interpolated else self.antialias_cutoff_gpts

        probe = Probe(gpts=gpts, extent=extent, energy=self.energy, device=self.device,
                      semiangle_cutoff=self.planewave_cutoff, ctf=ctf)

        probe._antialias_cutoff_gpts = antialias_cutoff_gpts
        return probe

    def _validate_ctf(self, ctf: CTF) -> CTF:
        if ctf is None:
            ctf = CTF(semiangle_cutoff=self.planewave_cutoff, energy=self.energy)

        if isinstance(ctf, dict):
            ctf = CTF(energy=self.energy, **ctf)

        return ctf

    def _validate_positions(self, positions, ctf):
        if positions is None:
            positions = (0., 0.)

        if isinstance(positions, GridScan):
            if positions.start is None:
                positions.start = (0., 0.)
            if positions.end is None:
                positions.end = self.extent
            if positions.sampling is None and ctf is not None:
                positions.sampling = 0.9 * ctf.nyquist_sampling
            return positions

        elif isinstance(positions, LineScan):
            raise NotImplementedError()

        elif isinstance(positions, CustomScan):
            return positions

        elif isinstance(positions, (list, tuple, np.ndarray)):
            return CustomScan(positions)

        elif not isinstance(positions, CustomScan):
            raise NotImplementedError


def _validate_interpolation(interpolation: Union[int, Tuple[int, int]]):
    if isinstance(interpolation, int):
        interpolation = (interpolation,) * 2
    elif not len(interpolation) == 2:
        raise ValueError('interpolation factor must be int')
    return interpolation


# class SMatrixArray(HasDaskArray, AbstractSMatrix):
#     """
#     Scattering matrix array object.
#
#     The scattering matrix array object represents a plane wave expansion of a probe, it is used for STEM simulations
#     with the PRISM algorithm.
#
#     Parameters
#     ----------
#     array : 3d array or 4d array
#         The array representation of the scattering matrix.
#     energy : float
#         Electron energy [eV].
#     wave_vectors : 2d array
#         The spatial frequencies of each plane in the plane wave expansion.
#     planewave_cutoff : float
#         The angular cutoff of the plane wave expansion [mrad].
#     interpolation : int or two int
#         Interpolation factor. Default is 1 (no interpolation).
#     sampling : one or two float, optional
#         Lateral sampling of wave functions [1 / Å]. Default is None (inherits sampling from the potential).
#     tilt : two float, optional
#         Small angle beam tilt [mrad].
#     antialias_aperture : two float, optional
#         Assumed antialiasing aperture as a fraction of the real space Nyquist frequency. Default is 2/3.
#     device : str, optional
#         The calculations will be carried out on this device. Default is 'cpu'.
#     extra_axes_metadata : list of dicts
#     metadata : dict
#     """
#
#     def __init__(self,
#                  array: Union[np.ndarray, da.core.Array],
#                  energy: float,
#                  wave_vectors: np.ndarray,
#                  planewave_cutoff: float,
#                  interpolation: Union[int, Tuple[int, int]] = 1,
#                  partitions: int = None,
#                  sampling: Union[float, Tuple[float, float]] = None,
#                  tilt: Tuple[float, float] = None,
#                  accumulated_defocus: float = 0.,
#                  crop_offset: Tuple[int, int] = (0, 0),
#                  uncropped_gpts: Tuple[int, int] = None,
#                  antialias_cutoff_gpts: Tuple[int, int] = None,
#                  normalization: str = 'probe',
#                  device: str = None,
#                  extra_axes_metadata: List[Dict] = None,
#                  metadata: Dict = None):
#
#         self._interpolation = self._validate_interpolation(interpolation)
#         self._grid = Grid(gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
#
#         self._beam_tilt = BeamTilt(tilt=tilt)
#         self._antialias_cutoff_gpts = antialias_cutoff_gpts
#         self._accelerator = Accelerator(energy=energy)
#         self._device = validate_device(device)
#
#         self._array = array
#         self._wave_vectors = wave_vectors
#         self._planewave_cutoff = planewave_cutoff
#
#         super().__init__(array)
#
#         if extra_axes_metadata is None:
#             extra_axes_metadata = []
#
#         if metadata is None:
#             metadata = {}
#
#         self._extra_axes_metadata = extra_axes_metadata
#         self._metadata = metadata
#
#         self._accumulated_defocus = accumulated_defocus
#         self._partitions = partitions
#
#         self._normalization = normalization
#
#         self._crop_offset = crop_offset
#         self._uncropped_gpts = uncropped_gpts
#
#         self._check_axes_metadata()
#

#
#
#
#     def _validate_probes_per_reduction(self, probes_per_reduction: int) -> int:
#
#         if probes_per_reduction == 'auto' or probes_per_reduction is None:
#             probes_per_reduction = 300
#         return probes_per_reduction
#
#     def rechunk(self, chunks: int = None, **kwargs):
#         if not isinstance(self.array, da.core.Array):
#             raise RuntimeError()
#
#         if chunks is None:
#             chunks = self.array.chunks[:-3] + ((sum(self.array.chunks[-3]),),) + self.array.chunks[-2:]
#
#         self._array = self._array.rechunk(chunks=chunks, **kwargs)
#         return self
#
#     def crop_to_positions(self, positions: Union[np.ndarray, AbstractScan]):
#         xp = get_array_module(self.array)
#         if self.interpolation == (1, 1):
#             corner = (0, 0)
#             cropped_array = self.array
#         else:
#             corner, size, _ = _minimum_crop(positions, self.sampling, self.interpolated_gpts)
#             corner = (corner[0] if self.interpolation[0] > 1 else 0, corner[1] if self.interpolation[1] > 1 else 0)
#
#             size = (size[0] if self.interpolation[0] > 1 else self.gpts[0],
#                     size[1] if self.interpolation[1] > 1 else self.gpts[1])
#
#             if self.is_lazy:
#                 cropped_array = self.array.map_blocks(wrapped_crop_2d,
#                                                       corner=corner,
#                                                       size=size,
#                                                       chunks=self.array.chunks[:-2] + ((size[0],), (size[1],)),
#                                                       meta=xp.array((), dtype=xp.complex64))
#             else:
#                 cropped_array = wrapped_crop_2d(self.array, corner=corner, size=size)
#
#         d = self._copy_as_dict(copy_array=False)
#         d['array'] = cropped_array
#         d['crop_offset'] = corner
#         d['uncropped_gpts'] = self.uncropped_gpts
#         return self.__class__(**d)
#
#     def downsample(self, max_angle: Union[str, float] = 'cutoff') -> 'SMatrixArray':
#         waves = Waves(self.array, sampling=self.sampling, energy=self.energy,
#                       extra_axes_metadata=self.axes_metadata[:-2])
#
#         if self.normalization == 'probe':
#             waves = waves.downsample(max_angle=max_angle, normalization='amplitude')
#         elif self.normalization == 'planewaves':
#             waves = waves.downsample(max_angle=max_angle, normalization='values')
#         else:
#             raise RuntimeError()
#
#         d = self._copy_as_dict(copy_array=False)
#         d['array'] = waves.array
#         d['sampling'] = waves.sampling
#         return self.__class__(**d)
#
#     def streaming_multislice(self, potential, chunks=None, **kwargs):
#
#         for chunk_start, chunk_stop in generate_chunks(len(self), chunks=chunks):
#             extra_axes_metadata = self.extra_axes_metadata + [PrismPlaneWavesAxis()]
#             waves = Waves(self.array[chunk_start:chunk_stop], energy=self.energy, sampling=self.sampling,
#                           extra_axes_metadata=extra_axes_metadata)
#             waves = waves.copy('gpu')
#             self._array[chunk_start:chunk_stop] = waves.multislice(potential, **kwargs).copy('cpu').array
#
#         return self
#
#     def multislice(self,
#                    potential: Union[Atoms, AbstractPotential],
#                    start: int = 0,
#                    stop: int = None,
#                    chunks: int = None,
#                    conjugate: bool = False) -> 'SMatrixArray':
#         """
#         Propagate the scattering matrix through the provided potential.
#
#         Parameters
#         ----------
#         potential : AbstractPotential object
#             Scattering potential.
#
#         Returns
#         -------
#         Waves object.
#             Probe exit wave functions for the provided positions.
#         """
#
#         if chunks is None:
#             chunks = len(self)
#
#         potential = validate_potential(potential)
#
#         waves = self.to_waves()
#
#         if self._is_streaming:
#             waves._array = waves._array.map_blocks(cp.asarray)
#             waves = waves.multislice(potential, start=start, stop=stop, conjugate=conjugate)
#             waves._array = waves._array.map_blocks(cp.asnumpy)
#
#         else:
#             waves = waves.multislice(potential, start=start, stop=stop)
#
#         return self.from_waves(waves)
#

#     def _distributed_reduce(self,
#                             detectors: List[AbstractDetector],
#                             scan: AbstractScan,
#                             scan_divisions: Tuple[int, int],
#                             ctf: CTF,
#                             probes_per_reduction: int):
#
#         scans = scan.divide(scan_divisions)
#
#         scans = [item for sublist in scans for item in sublist]
#
#         measurements = []
#         for scan in scans:
#             cropped_s_matrix_array = self.crop_to_positions(scan)
#
#             if self._is_streaming:
#                 cropped_s_matrix_array._array = cropped_s_matrix_array._array.map_blocks(cp.asarray)
#
#             measurement = cropped_s_matrix_array._apply_reduction_func(_reduce_to_measurements,
#                                                                        detectors=detectors,
#                                                                        scan=scan,
#                                                                        ctf=ctf,
#                                                                        probes_per_reduction=probes_per_reduction)
#             measurements.append(measurement)
#
#         measurements = list(map(list, zip(*measurements)))
#
#         for i, measurement in enumerate(measurements):
#             cls = measurement[0].__class__
#             d = measurement[0]._copy_as_dict(copy_array=False)
#
#             measurement = [measurement[i:i + scan_divisions[1]] for i in range(0, len(measurement), scan_divisions[1])]
#
#             array = da.concatenate([da.concatenate([item.array for item in block], axis=1)
#                                     for block in measurement], axis=0)
#
#             d['array'] = array
#             measurements[i] = cls(**d)
#
#         return measurements


@dataclass
class DummyPotential:
    ensemble_axes_metadata: list
    ensemble_shape: tuple


class SMatrixWaves(HasGridMixin, HasAcceleratorMixin):

    def __init__(self,
                 waves,
                 wave_vectors: np.ndarray,
                 planewave_cutoff: float,
                 interpolation,
                 cropping_window=(0, 0),
                 window_offset=(0, 0),
                 device=None):

        self._waves = waves
        self._wave_vectors = wave_vectors
        self._planewave_cutoff = planewave_cutoff
        self._cropping_window = cropping_window
        self._window_offset = window_offset
        self._interpolation = interpolation
        self._device = device

        self._grid = waves.grid
        self._accelerator = waves.accelerator

    @property
    def interpolation(self):
        return self._interpolation

    def rechunk_planewaves(self, chunks=None):

        if chunks is None:
            chunks = self.waves.chunks[-2:]
        else:
            chunks = validate_chunks(self.waves.gpts, chunks)

        chunks = self.waves.chunks[:-3] + ((self.waves.shape[-3],),) + chunks

        self.waves.rechunk(chunks)

        return self

    @property
    def planewave_cutoff(self):
        return self._planewave_cutoff

    @property
    def waves(self) -> Waves:
        return self._waves

    @property
    def wave_vectors(self):
        return self._wave_vectors

    @property
    def cropping_window(self):
        return self._cropping_window

    @property
    def window_extent(self):
        return self.cropping_window[0] * self.sampling[0], self.cropping_window[1] * self.sampling[1]

    @property
    def antialias_cutoff_gpts(self):
        return self.waves.antialias_cutoff_gpts

    @property
    def interpolated_antialias_cutoff_gpts(self):
        if self.antialias_cutoff_gpts is None:
            return None

        return (self.antialias_cutoff_gpts[0] // self.interpolation[0],
                self.antialias_cutoff_gpts[1] // self.interpolation[1])

    def dummy_probes(self, scan):

        probe = Probe(gpts=self.cropping_window,
                      extent=self.window_extent,
                      energy=self.energy,
                      device=self._device,
                      semiangle_cutoff=self.planewave_cutoff)

        probe.add_transform(scan)

        return probe

    def reduce_to_waves(self, scan: AbstractScan, ctf: CTF) -> 'Waves':
        array = self.waves.array

        if self._device == 'gpu' and isinstance(array, np.ndarray):
            array = cp.asarray(array)

        xp = get_array_module(array)

        positions = scan.get_positions()
        positions = xp.asarray(positions)
        coefficients = prism_coefficients(positions, ctf=ctf, wave_vectors=self.wave_vectors, xp=xp)

        if self.cropping_window != self.gpts:
            pixel_positions = positions / xp.array(self.waves.sampling) - xp.asarray(self._window_offset)


            crop_corner, size, corners = minimum_crop(pixel_positions, self.cropping_window)
            array = wrapped_crop_2d(array, crop_corner, size)

            array = xp.tensordot(coefficients, array, axes=[-1, -3])
            array = batch_crop_2d(array, corners, self.cropping_window)

            # import matplotlib.pyplot as plt
            # print(array.shape)
            # plt.imshow(array[0,0,0].real)
            # plt.show()


        else:
            array = xp.tensordot(coefficients, self.waves.array, axes=[-1, -3])

        if len(self.waves.shape) > 3:
            num_extra_axes = len(self.waves.shape) - 3
            source = range(len(scan.shape))
            dest = range(num_extra_axes, num_extra_axes + len(scan.shape))
            array = xp.moveaxis(array, source, dest)

        ensemble_axes_metadata = self.waves.ensemble_axes_metadata[:-1] + scan.ensemble_axes_metadata

        waves = Waves(array, sampling=self.sampling, energy=self.energy, ensemble_axes_metadata=ensemble_axes_metadata)

        return waves

    def batch_reduce_to_measurements(self, scan, ctf, detectors, reduction_max_batch):
        probe = self.dummy_probes(scan)

        measurements = allocate_multislice_measurements(probe,
                                                        detectors,
                                                        extra_ensemble_axes_shape=self.waves.ensemble_shape[:-1],
                                                        extra_ensemble_axes_metadata=
                                                        self.waves.ensemble_axes_metadata[:-1])

        for indices, sub_scan in scan.generate_scans(reduction_max_batch):
            waves = self.reduce_to_waves(sub_scan, ctf)

            indices = (slice(None),) * (len(self.waves.shape) - 3) + indices

            for detector, measurement in measurements.items():
                measurement.array[indices] = detector.detect(waves).array

        return tuple(measurements.values())

    def partial(self):
        self.waves.ensemble_partial()

        def prism_partial(array,
                          waves_partial,
                          window_overlap,
                          kwargs,
                          block_info=None):

            waves = waves_partial(array)

            if block_info is not None:
                window_offset = (block_info[0]['array-location'][-2][0] -
                                 (block_info[0]['chunk-location'][-2] * 2 + 1) * window_overlap[0],
                                 block_info[0]['array-location'][-1][0] -
                                 (block_info[0]['chunk-location'][-1] * 2 + 1) * window_overlap[1])

            else:
                window_offset = (0, 0)

            return SMatrixWaves(waves, window_offset=window_offset, **kwargs)

        kwargs = {'wave_vectors': self.wave_vectors,
                  'planewave_cutoff': self.planewave_cutoff,
                  'device': self._device,
                  'cropping_window': self.cropping_window,
                  'interpolation': self.interpolation, }

        return partial(prism_partial,
                       waves_partial=self.waves.ensemble_partial(),
                       kwargs=kwargs)

    def _scan_chunks(self, scan):
        chunks = self.waves.chunks[-2:]
        extents = tuple(tuple((cc[0] * d, cc[1] * d) for cc in c) for c, d in zip(chunk_range(chunks), self.sampling))
        return scan.chunks_from_extents(extents)

    def _window_overlap(self):
        return self.cropping_window[0] // 2, self.cropping_window[1] // 2

    def _overlap_depth(self):
        window_overlap = self._window_overlap()
        return [{**{i: 0 for i in range(0, len(self.waves.shape) - 2)},
                 **{j: window_overlap[i] for i, j in
                    enumerate(range(len(self.waves.shape) - 2, len(self.waves.shape)))}}]

    def _validate_rechunk_scheme(self, rechunk_scheme='auto'):

        if rechunk_scheme == 'auto':
            rechunk_scheme = 'interpolation'

        if rechunk_scheme == 'interpolation':
            num_chunks = self.interpolation

        elif isinstance(rechunk_scheme, tuple):
            num_chunks = rechunk_scheme

            assert len(rechunk_scheme) == 2
        else:
            raise RuntimeError

        return tuple(equal_sized_chunks(n, num_chunks=nsc) for n, nsc in zip(self.gpts, num_chunks))

    def _validate_reduction_max_batch(self, scan, reduction_max_batch='auto'):

        shape = (len(scan),) + self.cropping_window
        chunks = (reduction_max_batch, -1, -1)

        return validate_chunks(shape, chunks, dtype=np.dtype('complex64'))[0][0]

    @staticmethod
    def lazy_reduce(array,
                    scan,
                    scan_chunks,
                    s_matrix_partial,
                    window_overlap,
                    ctf,
                    detectors,
                    reduction_max_batch,
                    block_info=None):

        scan = scan.select_block(block_info[0]['chunk-location'][-2:], scan_chunks)

        s_matrix = s_matrix_partial(array, window_overlap=window_overlap, block_info=block_info)

        measurements = s_matrix.batch_reduce_to_measurements(scan, ctf, detectors, reduction_max_batch)

        arr = np.zeros((1,) * (len(array.shape) - 1), dtype=object)
        arr.itemset(measurements)
        return arr

    def reduce(self,
               scan: AbstractScan,
               ctf: CTF = None,
               detectors: Union[AbstractDetector, List[AbstractDetector]] = None,
               reduction_max_batch: Union[int, str] = 'auto',
               rechunk_scheme: Union[Tuple[int, int], str] = 'auto'):

        """
        Scan the probe across the potential and record a measurement for each detector.

        Parameters
        ----------
        detectors : List of Detector objects
            The detectors recording the measurements.
        positions : Scan object
            Scan defining the positions of the probe wave functions.
        ctf: CTF object, optional
            The probe contrast transfer function. Default is None (aperture is set by the planewave cutoff).
        rechunk_scheme : two int or str, optional
            Partitioning of the scan. The scattering matrix will be reduced in similarly partitioned chunks.
            Should be equal to or greater than the interpolation.
        reduction_batch : int or str, optional
            Number of positions per reduction operation. A large number of positions better utilize thread
            parallelization, but requires more memory and floating point operations. The optimal value is highly
            dependent on
        """

        if ctf is None:
            ctf = CTF(energy=self.waves.energy)

        if self.waves.is_lazy:
            probe = self.dummy_probes(scan)

            chunks = self._validate_rechunk_scheme(rechunk_scheme=rechunk_scheme)

            self.rechunk_planewaves(chunks=chunks)

            reduction_max_batch = self._validate_reduction_max_batch(scan, reduction_max_batch)

            chunks = self.waves.chunks[:-3] + self._scan_chunks(scan)

            array = da.map_overlap(self.lazy_reduce,
                                   self.waves.array,
                                   scan=scan,
                                   scan_chunks=chunks[-2:],
                                   drop_axis=len(self.waves.shape) - 3,
                                   align_arrays=False,
                                   chunks=chunks,
                                   depth=self._overlap_depth(),
                                   s_matrix_partial=self.partial(),
                                   window_overlap=self._window_overlap(),
                                   ctf=ctf,
                                   detectors=detectors,
                                   reduction_max_batch=reduction_max_batch,
                                   trim=False,
                                   boundary='periodic',
                                   meta=np.array((), dtype=np.complex64))

            return finalize_lazy_measurements(array,
                                              waves=probe,
                                              detectors=detectors,
                                              extra_ensemble_axes_metadata=self.waves.ensemble_axes_metadata[:-1])

        else:
            return self.batch_reduce_to_measurements(scan, ctf, detectors, reduction_max_batch)


class SMatrix(WavesLikeMixin):
    """
    Scattering matrix builder class

    The scattering matrix builder object is used for creating scattering matrices and simulating STEM experiments using
    the PRISM algorithm.

    Parameters
    ----------
    potential : Atoms or AbstractPotential
        Potential or atoms
    energy : float
        Electron energy [eV].
    planewave_cutoff : float
        The radial cutoff of the plane wave expansion [mrad]. Default is 30 mrad.
    interpolation : one or two int, optional
        Interpolation factor. Default is 1 (no interpolation).
    extent : one or two float, optional
        Lateral extent of wave functions [Å]. Default is None (inherits the extent from the potential).
    gpts : one or two int, optional
        Number of grid points describing the wave functions. Default is None (inherits the gpts from the potential).
    sampling : one or two float, optional
        Lateral sampling of wave functions [1 / Å]. Default is None (inherits the sampling from the potential).
    chunks : int, optional
        Number of PRISM plane waves in each chunk.
    tilt : two float
        Small angle beam tilt [mrad].
    downsample : float or str or bool
        If not False, the scattering matrix is downsampled to a maximum given scattering angle after running the
        multislice algorithm. If downsample is given as a float angle may be given as a float

        is given the scattering matrix is downsampled to a maximum scattering angle
    device : str, optional
        The calculations will be carried out on this device. Default is 'cpu'.
    store_on_host : bool
        If true, store the scattering matrix in host (cpu) memory. The necessary memory is transferred as chunks to
        the device to run calculations.
    """

    def __init__(self,
                 potential: Union[Atoms, AbstractPotential] = None,
                 energy: float = None,
                 planewave_cutoff: float = 30.,
                 interpolation: Union[int, Tuple[int, int]] = 1,
                 normalize: bool = True,
                 downsample: bool = 'cutoff',
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 tilt: Tuple[float, float] = None,
                 device: str = None,
                 store_on_host: bool = False):

        self._device = validate_device(device)
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self._potential = validate_potential(potential, self)

        if potential is not None:
            self._grid = self._potential.grid

        self._interpolation = _validate_interpolation(interpolation)
        self._planewave_cutoff = planewave_cutoff
        self._downsample = downsample

        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)

        self._normalize = normalize
        self._store_on_host = store_on_host

        self._extra_axes_metadata = []

    @property
    def downsample(self):
        return self._downsample

    @property
    def store_on_host(self):
        return self._store_on_host

    @property
    def metadata(self):
        return {'energy': self.energy}

    @property
    def shape(self):
        return (len(self),) + self.gpts

    def __len__(self) -> int:
        return len(self.wave_vectors)

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        self.grid.check_is_defined()
        return [OrdinalAxis(),
                RealSpaceAxis(label='x', sampling=self.sampling[0], units='Å', endpoint=False),
                RealSpaceAxis(label='y', sampling=self.sampling[0], units='Å', endpoint=False)]

    @property
    def wave_vectors(self) -> np.ndarray:
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        xp = np if self.store_on_host else get_array_module(self.device)
        wave_vectors = prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy, self.interpolation, xp=xp)
        return wave_vectors

    @property
    def potential(self) -> AbstractPotential:
        return self._potential

    @potential.setter
    def potential(self, potential):
        self._potential = potential
        self._grid = potential.grid

    @property
    def normalize(self):
        return self._normalize

    @property
    def planewave_cutoff(self) -> float:
        """Plane wave expansion cutoff."""
        return self._planewave_cutoff

    @planewave_cutoff.setter
    def planewave_cutoff(self, value: float):
        self._planewave_cutoff = value

    @property
    def interpolation(self) -> Tuple[int, int]:
        """Interpolation factor."""
        return self._interpolation

    @property
    def interpolated_gpts(self) -> Tuple[int, int]:
        return self.gpts[0] // self.interpolation[0], self.gpts[1] // self.interpolation[0]

    def _wave_vector_chunks(self, max_batch):
        if isinstance(max_batch, int):
            max_batch = max_batch * reduce(operator.mul, self.gpts)

        chunks = validate_chunks(shape=(len(self),) + self.gpts,
                                 chunks=('auto', -1, -1),
                                 limit=max_batch,
                                 dtype=np.dtype('complex64'),
                                 device=self.device)
        return chunks

    @staticmethod
    def _build_plane_waves_multislice_downsample(*args,
                                                 downsample,
                                                 planewave_cutoff,
                                                 extent,
                                                 gpts,
                                                 energy,
                                                 interpolation,
                                                 store_on_host,
                                                 start,
                                                 stop,
                                                 device='cpu',
                                                 potential_partial=None):
        xp = get_array_module(device)

        wave_vectors = prism_wave_vectors(planewave_cutoff, extent, energy, interpolation, xp=xp)
        array = plane_waves(wave_vectors[slice(*args[-1][0])], extent, gpts)

        normalization_constant = np.prod(gpts) * xp.sqrt(len(wave_vectors)) / np.prod(interpolation)
        array = array / normalization_constant.astype(xp.float32)
        # else:
        #    array = array / xp.sqrt(np.prod(gpts).astype(xp.float32))

        waves = Waves(array, energy=energy, extent=extent, ensemble_axes_metadata=[OrdinalAxis()])

        if potential_partial is not None:
            potential = potential_partial(args[0]).item()
            waves = multislice(waves, potential, start, stop=stop)

        if downsample:
            waves = waves.downsample(max_angle=downsample)

        array = waves.array[None]

        if store_on_host and device == 'gpu':
            with cp.cuda.Stream():
                array = cp.asnumpy(array)

        return array

    def _lazy_build_plane_waves_multislice_downsample(self, start: int = 0, stop: int = None):

        if self.potential is not None:
            potential_partial = self.potential.ensemble_partial()
        else:
            potential_partial = None

        return partial(self._build_plane_waves_multislice_downsample,
                       downsample=self.downsample,
                       planewave_cutoff=self.planewave_cutoff,
                       extent=self.extent,
                       gpts=self.gpts,
                       energy=self.energy,
                       interpolation=self.interpolation,
                       store_on_host=self.store_on_host,
                       start=start,
                       stop=stop,
                       potential_partial=potential_partial,
                       device=self.device)

    def build(self,
              start: int = 0,
              stop: int = None,
              lazy: bool = None,
              max_batch: Union[int, str] = 'auto'):

        """
        Build the plane waves of the scattering matrix and propagate the waves through the potential using the
        multislice algorithm.

        Parameters
        ----------
        start : int
            First slice index for running the multislice algorithm. Default is first slice of the potential.
        stop : int
            Last slice for running the multislice algorithm. If smaller than start the multislice algorithm will run
            in the reverse direction. Default is last slice of the potential.
        lazy : bool
            If True, build the scattering matrix lazily with dask array.


        Returns
        -------
        SMatrixArray
        """

        initialize()

        lazy = validate_lazy(lazy)

        wave_vector_chunks = self._wave_vector_chunks(max_batch)

        potential_blocks = self.potential.ensemble_blocks()[0]

        gpts = self._gpts_within_angle(self.downsample)

        s_matrix_arrays = []
        for i in range(len(potential_blocks)):
            wave_vector_blocks = da.from_array(chunk_range(wave_vector_chunks)[0], chunks=(1, 2), name=False)

            arr = da.blockwise(self._lazy_build_plane_waves_multislice_downsample(start=start, stop=stop),
                               (0, 1, 2, 3),
                               potential_blocks[[i]], (0,),
                               wave_vector_blocks, (1, -1),
                               new_axes={2: (gpts[0],), 3: (gpts[1],)},
                               adjust_chunks={1: wave_vector_chunks[0]},
                               concatenate=True,
                               meta=np.array((), dtype=np.complex64))

            s_matrix_arrays.append(arr)

        concat_s_matrix_arrays = da.concatenate(s_matrix_arrays)

        waves = Waves(concat_s_matrix_arrays,
                      energy=self.energy,
                      extent=self.extent,

                      ensemble_axes_metadata=self.potential.ensemble_axes_metadata + self.base_axes_metadata[:1])

        waves = waves.squeeze()

        cropping_window = self.gpts[0] // self.interpolation[0], self.gpts[1] // self.interpolation[1]

        return SMatrixWaves(waves,
                            wave_vectors=self.wave_vectors,
                            interpolation=self.interpolation,
                            planewave_cutoff=self.planewave_cutoff,
                            cropping_window=cropping_window,
                            device=self._device)

    def scan(self,
             scan: Union[np.ndarray, AbstractScan] = None,
             detectors: Union[AbstractDetector, List[AbstractDetector]] = None,
             ctf: Union[CTF, Dict] = None,
             distribute_scan: Union[bool, Tuple[int, int]] = False,
             probes_per_reduction: int = None,
             downsample: Union[float, str] = 'cutoff',
             lazy: bool = None) -> Union[Waves, AbstractMeasurement, List[AbstractMeasurement]]:

        """
        Scan the probe across the potential and record a measurement for each detector.

        Parameters
        ----------
        detectors : detector and list of Detector
            The detectors recording the measurements.
        scan : Scan object
            Scan defining the positions of the probe wave functions.
        ctf: CTF object, optional
            The probe contrast transfer function. Default is None (aperture is set by the planewave cutoff).
        distribute_scan : two int, optional
            Partitioning of the scan. The scattering matrix will be reduced in similarly partitioned chunks.
            Should be equal to or greater than the interpolation.
        probes_per_reduction : int, optional
            Number of probe positions per reduction operation.
        lazy : bool

        """

        ctf = self._validate_ctf(ctf)

        if ctf.semiangle_cutoff is None:
            ctf.semiangle_cutoff = self.planewave_cutoff

        scan = validate_scan(scan, self.comparable_probe(ctf, interpolated=False))

        detectors = validate_detectors(detectors)

        multislice_start_stop, detect_every, axes_metadata = thickness_series_precursor(detectors, self.potential)

        if len(multislice_start_stop) == 1:
            return self.build(lazy=lazy, downsample=downsample).reduce(positions=scan,
                                                                       detectors=detectors,
                                                                       distribute_scan=distribute_scan,
                                                                       probes_per_reduction=probes_per_reduction)
        else:
            if not lazy:
                raise RuntimeError()

        measurements = []
        for i, (s_matrix, potential) in enumerate(self.generate_distribution(stop=0, yield_potential=True)):

            thickness_series = defaultdict(list)
            for i, (start, stop) in enumerate(multislice_start_stop):

                s_matrix = s_matrix.multislice(potential, start, stop)

                if downsample:
                    downsampled_s_matrix = s_matrix.downsample(max_angle=downsample)
                else:
                    downsampled_s_matrix = s_matrix

                detectors_at = detectors_at_stop_slice(detect_every, stop)

                new_measurements = downsampled_s_matrix.reduce(detectors_at,
                                                               scan,
                                                               distribute_scan=distribute_scan,
                                                               probes_per_reduction=probes_per_reduction)

                new_measurements = [new_measurements] if not isinstance(new_measurements, list) else new_measurements

                for detector, new_measurement in zip(detectors_at, new_measurements):
                    thickness_series[detector].append(new_measurement)

            measurements.append(stack_thickness_series(thickness_series, detectors, axes_metadata))

        measurements = stack_frozen_phonons(measurements, detectors)

        measurements = [measurement.squeeze() for measurement in measurements]

        if len(measurements) == 1:
            return measurements[0]
        else:
            return ComputableList(measurements)

    def _copy_as_dict(self, copy_potential: bool = True):
        potential = self.potential

        if copy_potential and self.potential is not None:
            potential = potential.copy()

        d = {'potential': potential,
             'energy': self.energy,
             'planewave_cutoff': self.planewave_cutoff,
             'interpolation': self.interpolation,
             'partitions': self.partitions,
             'normalize': self.normalize,
             'extent': self.extent,
             'gpts': self.gpts,
             'sampling': self.sampling,
             'chunks': self.chunks,
             'store_on_host': self._store_on_host,
             'tilt': self.tilt,
             'device': self._device}
        return d

    def __copy__(self) -> 'SMatrix':
        return self.__class__(**self._copy_as_dict())

    def copy(self) -> 'SMatrix':
        """Make a copy."""
        return copy(self)
