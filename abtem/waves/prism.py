import inspect
import operator
import warnings
from abc import abstractmethod
from functools import partial, reduce
from typing import Union, Tuple, Dict, List

import dask.array as da
import numpy as np
from ase import Atoms

from abtem.core.array import validate_lazy, HasArray, ComputableList
from abtem.core.axes import OrdinalAxis, AxisMetadata
from abtem.core.backend import get_array_module, cp, validate_device
from abtem.core.chunks import chunk_ranges, validate_chunks, equal_sized_chunks
from abtem.core.complex import abs2
from abtem.core.energy import Accelerator
from abtem.core.fft import fft2
from abtem.core.grid import Grid, GridUndefinedError
from abtem.core.intialize import initialize
from abtem.measure.detect import AbstractDetector, validate_detectors, WavesDetector
from abtem.measure.measure import AbstractMeasurement
from abtem.potentials.potentials import AbstractPotential, validate_potential
from abtem.waves.base import WavesLikeMixin
from abtem.waves.multislice import allocate_multislice_measurements, multislice_and_detect
from abtem.waves.prism_utils import prism_wave_vectors, plane_waves, wrapped_crop_2d, prism_coefficients, minimum_crop, \
    batch_crop_2d
from abtem.waves.scan import AbstractScan, validate_scan, GridScan
from abtem.waves.tilt import BeamTilt
from abtem.waves.transfer import CTF
from abtem.waves.waves import Waves, Probe, finalize_lazy_measurements


class AbstractSMatrix(WavesLikeMixin):

    @property
    @abstractmethod
    def wave_vectors(self):
        pass

    def __len__(self) -> int:
        return len(self.wave_vectors)

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        return [OrdinalAxis(values=self.wave_vectors)] + super().base_axes_metadata  # noqa

    @property
    def base_shape(self):
        return (len(self),) + super().base_shape


def validate_interpolation(interpolation: Union[int, Tuple[int, int]]):
    if isinstance(interpolation, int):
        interpolation = (interpolation,) * 2
    elif not len(interpolation) == 2:
        raise ValueError('interpolation factor must be int')
    return tuple(interpolation)


def common_kwargs(a, b):
    a_kwargs = inspect.signature(a).parameters.keys()
    b_kwargs = inspect.signature(b).parameters.keys()
    return set(a_kwargs).intersection(b_kwargs)


def validate_wave_vectors(wave_vectors):
    return [(float(wave_vector[0]), float(wave_vector[1])) for wave_vector in wave_vectors]


class SMatrixArray(HasArray, AbstractSMatrix):
    _base_dims = 3

    def __init__(self,
                 array: np.ndarray,
                 wave_vectors: np.ndarray,
                 planewave_cutoff: float,
                 interpolation: Union[int, Tuple[int, int]],
                 energy: float = None,
                 extent: Union[float, Tuple[float, float]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 tilt: Tuple[float, float] = (0., 0.),
                 cropping_window: Tuple[int, int] = (0, 0),
                 window_offset: Tuple[int, int] = (0, 0),
                 device=None,
                 ensemble_axes_metadata: List[AxisMetadata] = None,
                 metadata: Dict = None):

        if len(array.shape) < 2:
            raise RuntimeError('Wave function array should have 2 dimensions or more')

        self._array = array
        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)

        self._ensemble_axes_metadata = [] if ensemble_axes_metadata is None else ensemble_axes_metadata

        self._metadata = {} if metadata is None else metadata

        self._wave_vectors = validate_wave_vectors(wave_vectors)
        self._planewave_cutoff = planewave_cutoff
        self._cropping_window = tuple(cropping_window)
        self._window_offset = tuple(window_offset)
        self._interpolation = validate_interpolation(interpolation)
        self._device = device

        self.check_axes_metadata()

    @property
    def tilt(self):
        return self._beam_tilt.tilt

    @property
    def device(self):
        return self._device

    @classmethod
    def from_waves(cls, waves, **kwargs):
        kwargs.update({key: getattr(waves, key) for key in common_kwargs(cls, Waves)})
        kwargs['ensemble_axes_metadata'] = kwargs['ensemble_axes_metadata'][:-1]
        return cls(**kwargs)

    @property
    def waves(self):
        kwargs = {key: getattr(self, key) for key in common_kwargs(self.__class__, Waves)}
        kwargs['ensemble_axes_metadata'] = kwargs['ensemble_axes_metadata'] + self.base_axes_metadata[-1:]
        return Waves(**kwargs)

    def _copy_with_new_waves(self, waves):
        keys = set(inspect.signature(self.__class__).parameters.keys()) - common_kwargs(self.__class__, Waves)
        kwargs = {key: getattr(self, key) for key in keys}
        return self.from_waves(waves, **kwargs)

    @property
    def metadata(self) -> Dict:
        self._metadata['energy'] = self.energy
        return self._metadata

    @property
    def ensemble_axes_metadata(self):
        return self._ensemble_axes_metadata

    @property
    def window_offset(self):
        return self._window_offset

    @property
    def ensemble_shape(self):
        return self.array.shape[:-3]

    @property
    def interpolation(self):
        return self._interpolation

    def rechunk_planewaves(self, chunks=None):

        if chunks is None:
            chunks = self.chunks[-2:]
        else:
            chunks = validate_chunks(self.gpts, chunks)

        chunks = self.chunks[:-3] + ((self.shape[-3],),) + chunks

        self._array = self.array.rechunk(chunks)

        return self

    @property
    def planewave_cutoff(self):
        return self._planewave_cutoff

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

    def multislice(self,
                   potential: AbstractPotential = None,
                   start: int = 0,
                   stop: int = None,
                   lazy: bool = None,
                   max_batch: Union[int, str] = 'auto'):
        waves = self.waves.multislice(potential, start=start, stop=stop, lazy=lazy, max_batch=max_batch)
        return self._copy_with_new_waves(waves)

    def reduce_to_waves(self, scan: AbstractScan, ctf: CTF) -> 'Waves':
        array = self.waves.array

        if self._device == 'gpu' and isinstance(array, np.ndarray):
            array = cp.asarray(array)

        xp = get_array_module(array)

        positions = scan.get_positions()
        positions = xp.asarray(positions)
        coefficients = prism_coefficients(positions, ctf=ctf, wave_vectors=self.wave_vectors, xp=xp)

        if self.cropping_window != self.gpts:
            pixel_positions = positions / xp.array(self.waves.sampling) - xp.asarray(self.window_offset)

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

    def batch_reduce_to_measurements(self,
                                     scan: AbstractScan,
                                     ctf: CTF,
                                     detectors: List[AbstractDetector],
                                     reduction_max_batch: int) \
            -> Tuple[Union[AbstractMeasurement, Waves], ...]:

        dummy_probes = self.dummy_probes(scan=scan, ctf=ctf)

        measurements = allocate_multislice_measurements(dummy_probes,
                                                        detectors,
                                                        extra_ensemble_axes_shape=self.waves.ensemble_shape[:-1],
                                                        extra_ensemble_axes_metadata=
                                                        self.waves.ensemble_axes_metadata[:-1])

        for _, slics, sub_scan in scan.generate_blocks(reduction_max_batch):
            waves = self.reduce_to_waves(sub_scan, ctf)

            indices = (slice(None),) * (len(self.waves.shape) - 3) + slics

            for detector, measurement in measurements.items():
                measurement.array[indices] = detector.detect(waves).array

        return tuple(measurements.values())

    def partial(self):
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

            return SMatrixArray.from_waves(waves, window_offset=window_offset, **kwargs)

        kwargs = {'wave_vectors': self.wave_vectors,
                  'planewave_cutoff': self.planewave_cutoff,
                  'device': self.device,
                  'cropping_window': self.cropping_window,
                  'interpolation': self.interpolation, }

        return partial(prism_partial,
                       waves_partial=self.waves.from_partitioned_args(),
                       kwargs=kwargs)

    def _chunk_extents(self):
        chunks = self.waves.chunks[-2:]
        return tuple(tuple((cc[0] * d, cc[1] * d) for cc in c) for c, d in zip(chunk_ranges(chunks), self.sampling))

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

        num_positions = reduce(operator.mul, tuple(scan_chunk[index]
                                                   for scan_chunk, index in zip(scan_chunks, block_index)))

        if num_positions == 0:
            raise RuntimeError()

        scan = scan.select_block(block_index, scan_chunks)

        s_matrix = s_matrix_partial(array, window_overlap=window_overlap, block_info=block_info)

        measurements = s_matrix.batch_reduce_to_measurements(scan, ctf, detectors, reduction_max_batch)

        arr = np.zeros((1,) * (len(array.shape) - 1), dtype=object)
        arr.itemset(measurements)
        return arr

    def reduce(self,
               scan: AbstractScan = None,
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
        reduction_max_batch : int or str, optional
            Number of positions per reduction operation. A large number of positions better utilize thread
            parallelization, but requires more memory and floating point operations. The optimal value is highly
            dependent on
        rechunk_scheme : two int or str, optional
            Partitioning of the scan. The scattering matrix will be reduced in similarly partitioned chunks.
            Should be equal to or greater than the interpolation.
        """

        self.accelerator.check_is_defined()

        if ctf is None:
            ctf = CTF(energy=self.waves.energy, semiangle_cutoff=self.planewave_cutoff)

        squeeze_scan = False
        if scan is None:
            squeeze_scan = True
            scan = self.extent[0] / 2, self.extent[1] / 2

        scan = validate_scan(scan, Probe.from_ctf(extent=self.extent, ctf=ctf, energy=self.energy))

        detectors = validate_detectors(detectors)

        reduction_max_batch = self._validate_reduction_max_batch(scan, reduction_max_batch)

        if self.is_lazy:
            chunks = self._validate_rechunk_scheme(rechunk_scheme=rechunk_scheme)

            s_matrix = self.rechunk_planewaves(chunks=chunks)

            scan, scan_chunks = scan.sort_into_extents(self._chunk_extents())

            chunks = s_matrix.chunks[:-3] + (1, 1)

            array = da.map_overlap(self.lazy_reduce,
                                   self.array,
                                   scan=scan,
                                   scan_chunks=scan_chunks,
                                   drop_axis=len(self.shape) - 3,
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

            chunks = self.chunks[:-3] + scan_chunks

            measurements = finalize_lazy_measurements(array,
                                                      waves=dummy_probes,
                                                      detectors=detectors,
                                                      extra_ensemble_axes_metadata=self.ensemble_axes_metadata,
                                                      chunks=chunks)

            measurements = (measurements,) if not isinstance(measurements, tuple) else measurements

        else:
            measurements = self.batch_reduce_to_measurements(scan, ctf, detectors, reduction_max_batch)

        if squeeze_scan:
            if isinstance(measurements, tuple):
                measurements = tuple(measurement.squeeze((-3,)) for measurement in measurements)
            else:
                measurements = measurements.squeeze((-3,))

        return measurements[0] if len(measurements) == 1 else ComputableList(measurements)


def round_gpts_to_multiple_of_interpolation(gpts: Tuple[int, int], interpolation: Tuple[int, int]) -> Tuple[int, int]:
    return tuple(n + (-n) % f for f, n in zip(interpolation, gpts))  # noqa


class SMatrix(AbstractSMatrix):
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
                 tilt: Tuple[float, float] = (0., 0.),
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 device: str = None,
                 store_on_host: bool = False):

        self._device = validate_device(device)
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self.set_potential(potential)

        self._interpolation = validate_interpolation(interpolation)
        self._planewave_cutoff = planewave_cutoff
        self._downsample = downsample

        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)

        self._normalize = normalize
        self._store_on_host = store_on_host

        assert planewave_cutoff > 0.

        if not all(n % f == 0 for f, n in zip(self.interpolation, self.gpts)):
            warnings.warn('the interpolation factor does not exactly divide gpts, normalization may not be exactly '
                          'preserved')

    def set_potential(self, potential):
        self._potential = validate_potential(potential)

        if self._potential is not None:
            self.grid.match(self._potential)
            self._grid = self._potential.grid
        else:
            try:
                self.grid.check_is_defined()
            except GridUndefinedError:
                raise ValueError('provide a potential or provide extent and gpts')

    @property
    def tilt(self):
        return self._beam_tilt.tilt

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

    @property
    def ensemble_shape(self):
        if self.potential is not None:
            return self.potential.ensemble_shape
        else:
            return ()

    @property
    def wave_vectors(self) -> List[Tuple[float, float]]:
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        xp = np if self.store_on_host else get_array_module(self.device)
        wave_vectors = prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy, self.interpolation, xp=xp)
        return validate_wave_vectors(wave_vectors)

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

    def _build_s_matrix(self, wave_vector_range=slice(None), start=0, stop=None):

        xp = get_array_module(self.device)

        wave_vectors = xp.asarray(self.wave_vectors, dtype=xp.float32)[wave_vector_range]

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

        waves = Waves(array, energy=self.energy, extent=self.extent,
                      ensemble_axes_metadata=[OrdinalAxis(values=wave_vectors)])

        if self.potential is not None:
            waves = multislice_and_detect(waves, self.potential, [WavesDetector()], start=start, stop=stop)[0]

        if self.downsampled_gpts() != self.gpts:
            waves = waves.downsample(gpts=self.downsampled_gpts(), normalization='intensity')

        if self.store_on_host and self.device == 'gpu':
            with cp.cuda.Stream():
                waves._array = cp.asnumpy(waves.array)

        return waves

    @staticmethod
    def _wrapped_build_s_matrix(*args, s_matrix_partial):
        s_matrix = s_matrix_partial(*tuple(arg.item() for arg in args[:-1]))

        wave_vector_range = slice(*np.squeeze(args[-1]))

        return s_matrix._build_s_matrix(wave_vector_range).array

    def _s_matrix_partial(self):
        def s_matrix(*args, potential_partial, **kwargs):
            if potential_partial is not None:
                potential = potential_partial(*args + (np.array([None], dtype=object),))
            else:
                potential = None
            return SMatrix(potential=potential, **kwargs)

        potential_partial = self.potential.from_partitioned_args() if self.potential is not None else None
        return partial(s_matrix, potential_partial=potential_partial, **self.copy_kwargs(exclude=('potential',)))

    def dummy_probes(self, scan=None, ctf=None):
        return self.build(lazy=True).dummy_probes(scan=scan, ctf=ctf)

    def multislice(self, potential=None, start=0, stop=None, lazy: bool = None, max_batch: Union[int, str] = 'auto'):
        s_matrix = self.__class__(potential=potential, **self.copy_kwargs(exclude=('potential',)))
        return s_matrix.build(start=start, stop=stop, lazy=lazy, max_batch=max_batch)

    def build(self,
              lazy: bool = None,
              max_batch: Union[int, str] = 'auto') -> SMatrixArray:

        """
        Build the plane waves of the scattering matrix and propagate the waves through the potential using the
        multislice algorithm.

        Parameters
        ----------
        lazy : bool
            If True, build the scattering matrix lazily.
        max_batch : 'auto' or str
            The maximum number of plane waves

        Returns
        -------
        s_matrix_array  : SMatrixArray
        """

        initialize()

        lazy = validate_lazy(lazy)

        wave_vector_chunks = self._wave_vector_chunks(max_batch)

        downsampled_gpts = self.downsampled_gpts()

        wave_vector_blocks = np.array(chunk_ranges(wave_vector_chunks)[0], dtype=int)

        if self.potential is not None:
            if len(self.potential.exit_planes) > 1:
                raise NotImplementedError('thickness series not yet implemented for PRISM')

            potential_blocks = self.potential.partition_args()[0], (0,)
            ensemble_axes_metadata = self.potential.ensemble_axes_metadata

            if self.potential.ensemble_shape:
                symbols = tuple(range(4))
            else:
                symbols = tuple(range(1, 4))

            wave_vector_blocks = np.tile(wave_vector_blocks[None], (len(potential_blocks[0]), 1, 1))
            wave_vector_blocks = da.from_array(wave_vector_blocks, chunks=(1, 1, 2), name=False)
            blocks = potential_blocks + (wave_vector_blocks, (0, 1, -1))

            adjust_chunks = {1: wave_vector_chunks[0]}
            new_axes = {2: (downsampled_gpts[0],), 3: (downsampled_gpts[1],)}
        else:
            potential_blocks = ()
            ensemble_axes_metadata = []
            symbols = tuple(range(0, 3))
            adjust_chunks = {0: wave_vector_chunks[0]}
            new_axes = {1: (downsampled_gpts[0],), 2: (downsampled_gpts[1],)}
            wave_vector_blocks = da.from_array(wave_vector_blocks, chunks=(1, 2), name=False)
            blocks = potential_blocks + (wave_vector_blocks, (0, -1))

        if lazy:
            xp = get_array_module(self.device)

            arr = da.blockwise(self._wrapped_build_s_matrix,
                               symbols,
                               *blocks,
                               new_axes=new_axes,
                               adjust_chunks=adjust_chunks,
                               concatenate=True,
                               meta=xp.array((), dtype=np.complex64),
                               **{'s_matrix_partial': self._s_matrix_partial()})

            waves = Waves(arr,
                          energy=self.energy,
                          extent=self.extent,
                          ensemble_axes_metadata=ensemble_axes_metadata + self.base_axes_metadata[:1])

        else:
            waves = self._build_s_matrix()

        cropping_window = downsampled_gpts[0] // self.interpolation[0], downsampled_gpts[1] // self.interpolation[1]

        return SMatrixArray.from_waves(waves,
                                       wave_vectors=self.wave_vectors,
                                       interpolation=self.interpolation,
                                       planewave_cutoff=self.planewave_cutoff,
                                       cropping_window=cropping_window,
                                       device=self.device)

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
