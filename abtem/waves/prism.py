import operator
import warnings
from copy import copy
from functools import partial, reduce
from typing import Union, Tuple, Dict, List

import dask.array as da
import numpy as np
from ase import Atoms

from abtem.core.axes import OrdinalAxis, RealSpaceAxis, AxisMetadata, FrozenPhononsAxis
from abtem.core.backend import get_array_module, cp, validate_device
from abtem.core.complex import abs2
from abtem.core.dask import validate_lazy, validate_chunks, chunk_range, equal_sized_chunks
from abtem.core.energy import Accelerator, HasAcceleratorMixin
from abtem.core.fft import fft2
from abtem.core.grid import Grid, HasGridMixin, GridUndefinedError
from abtem.core.intialize import initialize
from abtem.measure.detect import AbstractDetector, validate_detectors
from abtem.measure.measure import AbstractMeasurement
from abtem.potentials.potentials import AbstractPotential, validate_potential
from abtem.waves.base import WavesLikeMixin
from abtem.waves.multislice import multislice, allocate_multislice_measurements
from abtem.waves.prism_utils import prism_wave_vectors, plane_waves, wrapped_crop_2d, prism_coefficients, minimum_crop, \
    batch_crop_2d
from abtem.waves.scan import AbstractScan, validate_scan, GridScan
from abtem.waves.tilt import BeamTilt
from abtem.waves.transfer import CTF
from abtem.waves.waves import Waves, Probe, finalize_lazy_measurements


class AbstractSMatrix:
    pass


def _validate_interpolation(interpolation: Union[int, Tuple[int, int]]):
    if isinstance(interpolation, int):
        interpolation = (interpolation,) * 2
    elif not len(interpolation) == 2:
        raise ValueError('interpolation factor must be int')
    return interpolation


class SMatrixWaves(HasGridMixin, HasAcceleratorMixin, AbstractSMatrix):

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
        else:
            array = xp.tensordot(coefficients, array, axes=[-1, -3])

        if len(self.waves.shape) > 3:
            num_extra_axes = len(self.waves.shape) - 3
            source = range(len(scan.shape))
            dest = range(num_extra_axes, num_extra_axes + len(scan.shape))
            array = xp.moveaxis(array, source, dest)

        ensemble_axes_metadata = self.waves.ensemble_axes_metadata[:-1] + scan.ensemble_axes_metadata

        waves = Waves(array, sampling=self.sampling, energy=self.energy, ensemble_axes_metadata=ensemble_axes_metadata)

        return waves

    def dummy_probes(self, scan=None, ctf=None):

        if ctf is None:
            ctf = CTF(energy=self.waves.energy, semiangle_cutoff=self.planewave_cutoff)

        probes = Probe.from_ctf(extent=self.window_extent,
                                gpts=self.cropping_window,
                                ctf=ctf,
                                energy=self.energy,
                                device=self._device)

        if scan is not None:
            probes = probes.insert_transform(scan)

        return probes

    def batch_reduce_to_measurements(self, scan, ctf, detectors, reduction_max_batch):
        dummy_probes = self.dummy_probes(scan=scan, ctf=ctf)

        measurements = allocate_multislice_measurements(dummy_probes,
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

    def _chunk_extents(self):
        chunks = self.waves.chunks[-2:]
        return tuple(tuple((cc[0] * d, cc[1] * d) for cc in c) for c, d in zip(chunk_range(chunks), self.sampling))

    def _window_overlap(self):
        return self.cropping_window[0] // 2, self.cropping_window[1] // 2

    def _overlap_depth(self):
        if self.cropping_window == self.gpts:
            return 0

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

        if len(scan.ensemble_shape) == 1:
            block_index = np.ravel_multi_index(block_info[0]['chunk-location'][-2:], block_info[0]['num-chunks'][-2:]),
        else:
            block_index = block_info[0]['chunk-location'][-2:]

        scan = scan.select_block(block_index, scan_chunks)

        s_matrix = s_matrix_partial(array, window_overlap=window_overlap, block_info=block_info)

        measurements = s_matrix.batch_reduce_to_measurements(scan, ctf, detectors, reduction_max_batch)

        arr = np.zeros((1,) * (len(array.shape) - 1), dtype=object)
        arr.itemset(measurements)
        return arr

    # def multislice(self,
    #                potential: Union[Atoms, AbstractPotential],
    #                start: int = 0,
    #                stop: int = None,
    #                chunks: int = None,
    #                conjugate: bool = False) -> 'SMatrixArray':
    #     """
    #     Propagate the scattering matrix through the provided potential.
    #
    #     Parameters
    #     ----------
    #     potential : AbstractPotential object
    #         Scattering potential.
    #
    #     Returns
    #     -------
    #     Waves object.
    #         Probe exit wave functions for the provided positions.
    #     """
    #
    #     if chunks is None:
    #         chunks = len(self)
    #
    #     potential = validate_potential(potential)
    #
    #     waves = self.to_waves()
    #
    #     if self._is_streaming:
    #         waves._array = waves._array.map_blocks(cp.asarray)
    #         waves = waves.multislice(potential, start=start, stop=stop, conjugate=conjugate)
    #         waves._array = waves._array.map_blocks(cp.asnumpy)
    #
    #     else:
    #         waves = waves.multislice(potential, start=start, stop=stop)
    #
    #     return self.from_waves(waves)
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

    def reduce(self,
               scan: AbstractScan = None,
               ctf: CTF = None,
               detectors: Union[AbstractDetector, List[AbstractDetector]] = None,
               reduction_max_batch: Union[int, str] = 'auto',
               rechunk_scheme: Union[Tuple[int, int], str] = 'auto',
               lazy: bool = None):

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

        self.accelerator.check_is_defined()

        if ctf is None:
            ctf = CTF(energy=self.waves.energy, semiangle_cutoff=self.planewave_cutoff)

        if scan is None:
            scan = self.waves.extent[0] / 2, self.waves.extent[1] / 2

        scan = validate_scan(scan, Probe.from_ctf(extent=self.extent, ctf=ctf, energy=self.energy))

        detectors = validate_detectors(detectors)

        reduction_max_batch = self._validate_reduction_max_batch(scan, reduction_max_batch)

        if self.waves.is_lazy:
            chunks = self._validate_rechunk_scheme(rechunk_scheme=rechunk_scheme)

            self.rechunk_planewaves(chunks=chunks)

            scan, scan_chunks = scan.sort_into_extents(self._chunk_extents())

            chunks = self.waves.chunks[:-3] + (1, 1)

            array = da.map_overlap(self.lazy_reduce,
                                   self.waves.array,
                                   scan=scan,
                                   scan_chunks=scan_chunks,
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

            dummy_probes = self.dummy_probes(scan=scan, ctf=ctf)

            if len(scan.ensemble_shape) != 2:
                array = array.reshape(array.shape[:-2] + (array.shape[-2] * array.shape[-1],))

            chunks = self.waves.chunks[:-3] + scan_chunks

            return finalize_lazy_measurements(array,
                                              waves=dummy_probes,
                                              detectors=detectors,
                                              extra_ensemble_axes_metadata=self.waves.ensemble_axes_metadata[:-1],
                                              chunks=chunks)

        else:

            measurements = self.batch_reduce_to_measurements(scan, ctf, detectors, reduction_max_batch)

            measurements = tuple(measurement.squeeze() for measurement in measurements)

            return measurements if len(measurements) > 1 else measurements[0]

    def compute(self, **kwargs):
        self.waves.compute(**kwargs)
        return self


def round_gpts_to_multiple_of_interpolation(gpts: Tuple[int, int], interpolation: Tuple[int, int]) -> Tuple[int, int]:
    return tuple(n + (-n) % f for f, n in zip(interpolation, gpts))  # noqa


class SMatrix(WavesLikeMixin, AbstractSMatrix):
    """
    The SMatrix may be used for creating scattering matrices and simulating STEM experiments using the PRISM algorithm.

    Parameters
    ----------
    planewave_cutoff : float
        The radial cutoff of the plane wave expansion [mrad].
    potential : Atoms or AbstractPotential, optional
        Potential or atoms represented by the scattering matrix. If given as atoms, a default Potential will be created.
        If not provided the scattering matrix will represent a vacuum potential, in this case the sampling and extent
        should be provided.
    energy : float, optional
        Electron energy [eV].
    interpolation : one or two int, optional
        Interpolation factor. Default is 1 (no interpolation).
    extent : one or two float, optional
        Lateral extent of wave functions [Å]. Provide only if potential is not given.
    gpts : one or two int, optional
        Number of grid points describing the wave functions. Provide only if potential is not given.
    sampling : one or two float, optional
        Lateral sampling of wave functions [1 / Å]. Provide only if potential is not given.
    tilt : two float
        Small angle beam tilt [mrad].
    downsample : {'cutoff', 'valid'} or float or bool
        If not False, the scattering matrix is downsampled to a maximum given scattering angle after running the
        multislice algorithm.
            ``cutoff`` or True :
                Downsample to the antialias cutoff scattering angle.

            ``valid`` :
                Downsample to the largest rectangle inside the circle with a the radius defined by the antialias cutoff
                scattering angle.

            float :
                Downsample to a maximum scattering angle specified by a float.
    normalize : {'probe', 'planewaves'}
    device : str, optional
        The calculations will be carried out on this device. Default is 'cpu'.
    store_on_host : bool
        If true, store the scattering matrix in host (cpu) memory. The necessary memory is transferred as chunks to
        the device to run calculations.
    """

    def __init__(self,
                 planewave_cutoff: float,
                 potential: Union[Atoms, AbstractPotential] = None,
                 energy: float = None,
                 interpolation: Union[int, Tuple[int, int]] = 1,
                 normalize: str = 'probe',
                 downsample: Union[bool, str] = 'cutoff',
                 tilt: Tuple[float, float] = None,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 device: str = None,
                 store_on_host: bool = False):

        self._device = validate_device(device)
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self._potential = validate_potential(potential, self)

        if potential is not None:
            self._grid = self._potential.grid
        else:
            try:
                self.grid.check_is_defined()
            except GridUndefinedError:
                raise ValueError('provide a potential or provide extent and gpts')

        self._interpolation = _validate_interpolation(interpolation)
        self._planewave_cutoff = planewave_cutoff
        self._downsample = downsample

        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)

        self._normalize = normalize
        self._store_on_host = store_on_host

        if not all(n % f == 0 for f, n in zip(self.interpolation, self.gpts)):
            warnings.warn('the interpolation factor does not exactly divide gpts, normalization may not be exactly '
                          'preserved')

    def round_gpts_to_interpolation(self):
        rounded = round_gpts_to_multiple_of_interpolation(self.gpts, self.interpolation)
        if rounded == self.gpts:
            return self

        self.gpts = rounded
        return self

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

    def downsampled_gpts(self) -> Tuple[int, int]:
        if self.downsample:
            downsampled_gpts = self._gpts_within_angle(self.downsample)
            return round_gpts_to_multiple_of_interpolation(downsampled_gpts, self.interpolation)
        else:
            return self.gpts

    def _build_s_matrix(self, wave_vector_range=None, start=0, stop=None):

        xp = get_array_module(self.device)

        wave_vectors = xp.asarray(self.wave_vectors)

        if wave_vector_range is not None:
            wave_vectors = wave_vectors[wave_vector_range]

        array = plane_waves(wave_vectors, self.extent, self.gpts)

        if all(n % f == 0 for f, n in zip(self.interpolation, self.gpts)):
            normalization_constant = np.prod(self.gpts) * xp.sqrt(len(wave_vectors)) / np.prod(self.interpolation)

        else:
            cropping_window = self.gpts[0] // self.interpolation[0], self.gpts[1] // self.interpolation[1]
            corner = cropping_window[0] // 2, cropping_window[1] // 2
            cropped_array = wrapped_crop_2d(array, corner, cropping_window)
            normalization_constant = xp.sqrt(abs2(fft2(cropped_array.sum(0))).sum())

        array = array / normalization_constant.astype(xp.float32)
        #    array = array / xp.sqrt(np.prod(gpts).astype(xp.float32))

        waves = Waves(array, energy=self.energy, extent=self.extent, ensemble_axes_metadata=[OrdinalAxis()])

        if self.potential is not None:
            waves = multislice(waves, self.potential, start, stop=stop)

        if self.downsampled_gpts() != self.gpts:
            waves = waves.downsample(gpts=self.downsampled_gpts(), normalization='intensity')

        if self.store_on_host and self.device == 'gpu':
            with cp.cuda.Stream():
                waves._array = cp.asnumpy(waves.array)

        return waves

    @staticmethod
    def _wrapped_build_s_matrix(*args, s_matrix_partial):
        s_matrix = s_matrix_partial(*args[:-1])
        wave_vector_range = slice(*args[-1][0, 0])
        return s_matrix._build_s_matrix(wave_vector_range).array[None, None]

    def _s_matrix_partial(self):
        def s_matrix(*args, potential_partial, **kwargs):
            if potential_partial is not None:
                potential = potential_partial(*args + (np.array([None], dtype=object),)).item()
            else:
                potential = None
            return SMatrix(potential=potential, **kwargs)

        potential_partial = self.potential.ensemble_partial() if self.potential is not None else None
        return partial(s_matrix, potential_partial=potential_partial, **self._copy_as_dict(copy_potential=False))

    def dummy_probes(self, scan=None, ctf=None):
        return self.build(lazy=True).dummy_probes(scan=scan, ctf=ctf)

    def build(self,
              start: int = 0,
              stop: int = None,
              lazy: bool = None,
              max_batch: Union[int, str] = 'auto') -> SMatrixWaves:

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
        max_batch : 'auto' or str
            The maximum number of plane waves


        Returns
        -------
        SMatrixWaves
        """

        initialize()

        lazy = validate_lazy(lazy)

        wave_vector_chunks = self._wave_vector_chunks(max_batch)

        downsampled_gpts = self.downsampled_gpts()

        if self.potential is not None:
            potential_blocks = self.potential.ensemble_blocks()[0], (0,), self.potential.ensemble_blocks()[1], (1,)
            ensemble_axes_metadata = self.potential.ensemble_axes_metadata
        else:
            potential_blocks = (da.from_array(np.array([None], dtype=object), chunks=1), (0,),
                                da.from_array(np.array([None], dtype=object), chunks=1), (1,))
            ensemble_axes_metadata = [AxisMetadata(), AxisMetadata()]

        wave_vector_blocks = np.array(chunk_range(wave_vector_chunks)[0], dtype=int)
        wave_vector_blocks = np.tile(wave_vector_blocks[None], (len(potential_blocks[0]), 1, 1))

        if lazy:
            wave_vector_blocks = da.from_array(wave_vector_blocks, chunks=(1, 1, 2), name=False)
            blocks = potential_blocks + (wave_vector_blocks, (0, 2, -1))

            arr = da.blockwise(self._wrapped_build_s_matrix,
                               tuple(range(5)),
                               *blocks,
                               new_axes={3: (downsampled_gpts[0],), 4: (downsampled_gpts[1],)},
                               adjust_chunks={2: wave_vector_chunks[0]},
                               concatenate=True,
                               meta=np.array((), dtype=np.complex64),
                               **{'s_matrix_partial': self._s_matrix_partial()})

            waves = Waves(arr,
                          energy=self.energy,
                          extent=self.extent,
                          ensemble_axes_metadata=ensemble_axes_metadata + self.base_axes_metadata[:1])

        else:
            waves = self._build_s_matrix()

        waves = waves.squeeze(axis=tuple(range(len(waves.shape) - 3)))

        cropping_window = downsampled_gpts[0] // self.interpolation[0], downsampled_gpts[1] // self.interpolation[1]

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
             multislice_max_batch: Union[str, int] = 'auto',
             reduction_max_batch: Union[str, int] = 'auto',
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
        if scan is None:
            scan = GridScan()

        return self.build(max_batch=multislice_max_batch, lazy=lazy).reduce(scan=scan,
                                                                            detectors=detectors,
                                                                            reduction_max_batch=reduction_max_batch,
                                                                            ctf=ctf)

    def reduce(self,
               scan: Union[np.ndarray, AbstractScan] = None,
               detectors: Union[AbstractDetector, List[AbstractDetector]] = None,
               ctf: Union[CTF, Dict] = None,
               multislice_max_batch: Union[str, int] = 'auto',
               reduction_max_batch: Union[str, int] = 'auto',
               lazy: bool = None):

        return self.build(max_batch=multislice_max_batch, lazy=lazy).reduce(scan=scan,
                                                                            detectors=detectors,
                                                                            reduction_max_batch=reduction_max_batch,
                                                                            ctf=ctf)

    def _copy_as_dict(self, copy_potential: bool = True):

        d = {'energy': self.energy,
             'planewave_cutoff': self.planewave_cutoff,
             'interpolation': self.interpolation,
             'downsample': self.downsample,
             'normalize': self.normalize,
             'extent': self.extent,
             'gpts': self.gpts,
             'sampling': self.sampling,
             'store_on_host': self._store_on_host,
             'tilt': self.tilt,
             'device': self._device}

        if copy_potential:
            potential = self.potential.copy() if self.potential is not None else None
            d['potential'] = potential

        return d

    def __copy__(self) -> 'SMatrix':
        return self.__class__(**self._copy_as_dict())

    def copy(self) -> 'SMatrix':
        """Make a copy."""
        return copy(self)
