from abc import abstractmethod
from collections import defaultdict
from copy import copy, deepcopy
from typing import Union, Sequence, Tuple, Dict, List

import dask
import dask.array as da
import numpy as np
from ase import Atoms

from abtem.core.axes import FrozenPhononsAxis, OrdinalAxis, RealSpaceAxis, AxisMetadata, PrismPlaneWavesAxis
from abtem.core.backend import get_array_module, cp, copy_to_device, _validate_device
from abtem.core.complex import complex_exponential
from abtem.core.dask import HasDaskArray, validate_lazy, ComputableList
from abtem.core.energy import Accelerator
from abtem.core.grid import Grid
from abtem.core.utils import generate_chunks
from abtem.measure.detect import AbstractDetector, validate_detectors, stack_measurements
from abtem.measure.measure import AbstractMeasurement
from abtem.measure.thickness import thickness_series_precursor, detectors_at_stop_slice, stack_thickness_series
from abtem.potentials.potentials import AbstractPotential, validate_potential
from abtem.potentials.temperature import stack_frozen_phonons
from abtem.waves.base import WavesLikeMixin
from abtem.waves.multislice import multislice
from abtem.waves.prism_utils import prism_wave_vectors, partitioned_prism_wave_vectors, plane_waves, \
    wrapped_crop_2d
from abtem.waves.scan import AbstractScan, GridScan, LineScan, CustomScan, validate_scan
from abtem.waves.tilt import BeamTilt
from abtem.waves.transfer import CTF
from abtem.waves.waves import Waves, Probe


def batch_crop_2d(array: np.ndarray, corners: Tuple[int, int], new_shape: Tuple[int, int]):
    xp = get_array_module(array)
    if xp is cp:
        i = xp.arange(array.shape[0])[:, None, None]
        ix = cp.arange(new_shape[0]) + cp.asarray(corners[:, 0, None])
        iy = cp.arange(new_shape[1]) + cp.asarray(corners[:, 1, None])
        ix = ix[:, :, None]
        iy = iy[:, None]
        return array[i, ix, iy]
    else:
        array = np.lib.stride_tricks.sliding_window_view(array, (1,) + new_shape)
        return array[xp.arange(array.shape[0]), corners[:, 0], corners[:, 1], 0]


def prism_coefficients(positions: np.ndarray, wave_vectors: np.ndarray, wavelength: float, ctf: CTF):
    xp = get_array_module(wave_vectors)
    positions = copy_to_device(positions, xp)

    def calculate_ctf_coefficient(wave_vectors, wavelength, ctf):
        alpha = xp.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2) * wavelength
        phi = xp.arctan2(wave_vectors[:, 0], wave_vectors[:, 1])
        coefficients = ctf.evaluate(alpha, phi)
        return coefficients

    def calculate_translation_coefficients(wave_vectors, positions):
        coefficients = complex_exponential(-2. * xp.pi * positions[..., 0, None] * wave_vectors[:, 0][None])
        coefficients *= complex_exponential(-2. * xp.pi * positions[..., 1, None] * wave_vectors[:, 1][None])
        return coefficients

    return calculate_ctf_coefficient(wave_vectors, wavelength=wavelength, ctf=ctf) * \
           calculate_translation_coefficients(wave_vectors, positions)


def _minimum_crop(positions: Union[Sequence[float], AbstractScan], sampling, shape):
    if isinstance(positions, AbstractScan):
        positions = np.array(positions.limits)

    xp = get_array_module(positions)

    offset = (shape[0] // 2, shape[1] // 2)
    corners = xp.rint(xp.array(positions) / xp.asarray(sampling) - xp.asarray(offset)).astype(int)
    upper_corners = corners + xp.asarray(shape)

    crop_corner = (xp.min(corners[..., 0]).item(), xp.min(corners[..., 1]).item())

    size = (xp.max(upper_corners[..., 0]).item() - crop_corner[0],
            xp.max(upper_corners[..., 1]).item() - crop_corner[1])

    corners -= xp.asarray(crop_corner)
    return crop_corner, size, corners


def _reduce_to_waves(s_matrix, positions, ctf, axes_metadata):
    xp = get_array_module(s_matrix._device)
    positions = xp.asarray(positions)

    if len(axes_metadata) != (len(positions.shape) - 1):
        raise RuntimeError()

    wave_vectors = xp.asarray(s_matrix.wave_vectors)
    alpha = xp.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2) * s_matrix.wavelength
    phi = xp.arctan2(wave_vectors[:, 0], wave_vectors[:, 1])
    basis = ctf.evaluate(alpha, phi)

    out_shape = positions.shape[:-1]
    positions = positions.reshape((-1, 2))
    offset_positions = positions - xp.array(s_matrix.crop_offset) * xp.array(s_matrix.sampling)

    wave_vectors = xp.asarray(s_matrix.wave_vectors)

    coefficients = complex_exponential(-2. * xp.pi * positions[..., 0, None] * wave_vectors[:, 0][None])
    coefficients *= complex_exponential(-2. * xp.pi * positions[..., 1, None] * wave_vectors[:, 1][None])
    coefficients *= basis

    if not s_matrix.array.shape[-2:] == s_matrix.interpolated_gpts:
        crop_corner, size, corners = _minimum_crop(offset_positions, s_matrix.sampling, s_matrix.interpolated_gpts)

        array = wrapped_crop_2d(s_matrix.array, crop_corner, size)

        if s_matrix._device == 'gpu':
            array = xp.asarray(array)

        array = xp.tensordot(coefficients, array, axes=[-1, -3])

        array = batch_crop_2d(array, corners.reshape((-1, 2)), s_matrix.interpolated_gpts)
    else:
        array = s_matrix.array
        if s_matrix._device == 'gpu':
            array = xp.asarray(array)
        array = xp.tensordot(coefficients, array, axes=[-1, -3])

    array = array.reshape(out_shape + array.shape[-2:])

    antialias_cutoff_gpts = (s_matrix._antialias_cutoff_gpts[0] // s_matrix.interpolation[0],
                             s_matrix._antialias_cutoff_gpts[1] // s_matrix.interpolation[1])

    waves = Waves(array,
                  sampling=s_matrix.sampling,
                  energy=s_matrix.energy,
                  extra_axes_metadata=axes_metadata,
                  antialias_cutoff_gpts=antialias_cutoff_gpts,
                  metadata=s_matrix.metadata)

    return waves


def _reduce_to_measurements(s_matrix, detectors, scan, ctf, probes_per_reduction):
    waves = s_matrix.comparable_probe().build(positions=scan, lazy=True)

    measurements = [detector.allocate_measurement(waves) for detector in detectors]

    for i, (indices, waves) in enumerate(s_matrix._generate_waves(scan, ctf, probes_per_reduction)):

        for j, detector in enumerate(detectors):
            measurements[j].array[indices] = detector.detect(waves).array

    return measurements


def stack_s_matrices(s_matrices, axes_metadata):
    arrays = [s_matrix.array for s_matrix in s_matrices]
    d = s_matrices[0]._copy_as_dict(copy_array=False)

    if s_matrices[0].is_lazy:
        d['array'] = da.stack(arrays)
    else:
        xp = get_array_module(arrays[0])
        d['array'] = xp.stack(arrays)
    d['extra_axes_metadata'] = [axes_metadata]

    return SMatrixArray(**d)


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

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        self.grid.check_is_defined()
        return [OrdinalAxis(),
                RealSpaceAxis(label='x', sampling=self.sampling[0], units='Å', endpoint=False),
                RealSpaceAxis(label='y', sampling=self.sampling[0], units='Å', endpoint=False)]

    def _validate_ctf(self, ctf: CTF) -> CTF:
        if ctf is None:
            ctf = CTF(semiangle_cutoff=self.planewave_cutoff, energy=self.energy)

        if isinstance(ctf, dict):
            ctf = CTF(energy=self.energy, **ctf)

        return ctf

    def _validate_interpolation(self, interpolation: Union[int, Tuple[int, int]]):
        if isinstance(interpolation, int):
            interpolation = (interpolation,) * 2
        elif not len(interpolation) == 2:
            raise ValueError('interpolation factor must be int')
        return interpolation

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


class SMatrixArray(HasDaskArray, AbstractSMatrix):
    """
    Scattering matrix array object.

    The scattering matrix array object represents a plane wave expansion of a probe, it is used for STEM simulations
    with the PRISM algorithm.

    Parameters
    ----------
    array : 3d array or 4d array
        The array representation of the scattering matrix.
    energy : float
        Electron energy [eV].
    wave_vectors : 2d array
        The spatial frequencies of each plane in the plane wave expansion.
    planewave_cutoff : float
        The angular cutoff of the plane wave expansion [mrad].
    interpolation : int or two int
        Interpolation factor. Default is 1 (no interpolation).
    sampling : one or two float, optional
        Lateral sampling of wave functions [1 / Å]. Default is None (inherits sampling from the potential).
    tilt : two float, optional
        Small angle beam tilt [mrad].
    antialias_aperture : two float, optional
        Assumed antialiasing aperture as a fraction of the real space Nyquist frequency. Default is 2/3.
    device : str, optional
        The calculations will be carried out on this device. Default is 'cpu'.
    extra_axes_metadata : list of dicts
    metadata : dict
    """

    def __init__(self,
                 array: Union[np.ndarray, da.core.Array],
                 energy: float,
                 wave_vectors: np.ndarray,
                 planewave_cutoff: float,
                 interpolation: Union[int, Tuple[int, int]] = 1,
                 partitions: int = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 tilt: Tuple[float, float] = None,
                 accumulated_defocus: float = 0.,
                 crop_offset: Tuple[int, int] = (0, 0),
                 uncropped_gpts: Tuple[int, int] = None,
                 antialias_cutoff_gpts: Tuple[int, int] = None,
                 normalization: str = 'probe',
                 device: str = None,
                 extra_axes_metadata: List[Dict] = None,
                 metadata: Dict = None):

        self._interpolation = self._validate_interpolation(interpolation)
        self._grid = Grid(gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)

        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_cutoff_gpts = antialias_cutoff_gpts
        self._accelerator = Accelerator(energy=energy)
        self._device = _validate_device(device)

        self._array = array
        self._wave_vectors = wave_vectors
        self._planewave_cutoff = planewave_cutoff

        super().__init__(array)

        if extra_axes_metadata is None:
            extra_axes_metadata = []

        if metadata is None:
            metadata = {}

        self._extra_axes_metadata = extra_axes_metadata
        self._metadata = metadata

        self._accumulated_defocus = accumulated_defocus
        self._partitions = partitions

        self._normalization = normalization

        self._crop_offset = crop_offset
        self._uncropped_gpts = uncropped_gpts

        self._check_axes_metadata()

    def __len__(self) -> int:
        return len(self.wave_vectors)

    @property
    def normalization(self) -> str:
        return self._normalization

    @property
    def full_wave_vectors(self):
        return prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy, self.interpolation)

    @property
    def crop_offset(self) -> Tuple[int, int]:
        return self._crop_offset

    @property
    def uncropped_gpts(self) -> Tuple[int, int]:
        if self._uncropped_gpts is None:
            return self.gpts
        return self._uncropped_gpts

    @property
    def is_cropped(self) -> bool:
        return self.uncropped_gpts != self.gpts

    @property
    def planewave_cutoff(self) -> float:
        return self._planewave_cutoff

    @property
    def num_axes(self) -> int:
        return len(self.array.shape)

    @property
    def num_base_axes(self) -> int:
        return 3

    @property
    def chunks(self) -> int:
        return self.array.chunks[:-2]

    @property
    def metadata(self) -> Dict:
        return self._metadata

    @property
    def accumulated_defocus(self) -> float:
        return self._accumulated_defocus

    @accumulated_defocus.setter
    def accumulated_defocus(self, value):
        self._accumulated_defocus = value

    @property
    def wave_vectors(self) -> np.ndarray:
        """The spatial frequencies of each wave in the plane wave expansion."""
        return self._wave_vectors

    @property
    def interpolation(self) -> Tuple[int, int]:
        """Interpolation factor."""
        return self._interpolation

    @property
    def partitions(self) -> int:
        return self._partitions

    @property
    def interpolated_gpts(self) -> Tuple[int, int]:
        return self.uncropped_gpts[0] // self.interpolation[0], self.uncropped_gpts[1] // self.interpolation[1]

    def to_waves(self):
        extra_axes_metadata = self.extra_axes_metadata + [PrismPlaneWavesAxis()]

        waves = Waves(self.array,
                      energy=self.energy,
                      sampling=self.sampling,
                      extra_axes_metadata=extra_axes_metadata)

        return waves

    def from_waves(self, waves):
        d = self._copy_as_dict(copy_array=False)
        d['array'] = waves.array
        return self.__class__(**d)

    def to_list(self):
        if len(self.array.shape) == 3:
            return [self]

        s_matrices = []
        for i in range(len(self.array)):
            d = self._copy_as_dict(copy_array=False)
            d['array'] = self.array[i]
            d['extra_axes_metadata'] = []
            s_matrices.append(self.__class__(**d))
        return s_matrices

    @property
    def _is_stored_on_host(self):
        if hasattr(self.array, '_meta'):
            return isinstance(self.array._meta, np.ndarray)

        return isinstance(self.array, np.ndarray)

    @property
    def _is_streaming(self):
        return self._device == 'gpu' and self._is_stored_on_host

    def _validate_probes_per_reduction(self, probes_per_reduction: int) -> int:

        if probes_per_reduction == 'auto' or probes_per_reduction is None:
            probes_per_reduction = 300
        return probes_per_reduction

    def rechunk(self, chunks: int = None, **kwargs):
        if not isinstance(self.array, da.core.Array):
            raise RuntimeError()

        if chunks is None:
            chunks = self.array.chunks[:-3] + ((sum(self.array.chunks[-3]),),) + self.array.chunks[-2:]

        self._array = self._array.rechunk(chunks=chunks, **kwargs)
        return self

    def crop_to_positions(self, positions: Union[np.ndarray, AbstractScan]):
        xp = get_array_module(self.array)
        if self.interpolation == (1, 1):
            corner = (0, 0)
            cropped_array = self.array
        else:
            corner, size, _ = _minimum_crop(positions, self.sampling, self.interpolated_gpts)
            corner = (corner[0] if self.interpolation[0] > 1 else 0, corner[1] if self.interpolation[1] > 1 else 0)

            size = (size[0] if self.interpolation[0] > 1 else self.gpts[0],
                    size[1] if self.interpolation[1] > 1 else self.gpts[1])

            if self.is_lazy:
                cropped_array = self.array.map_blocks(wrapped_crop_2d,
                                                      corner=corner,
                                                      size=size,
                                                      chunks=self.array.chunks[:-2] + ((size[0],), (size[1],)),
                                                      meta=xp.array((), dtype=xp.complex64))
            else:
                cropped_array = wrapped_crop_2d(self.array, corner=corner, size=size)

        d = self._copy_as_dict(copy_array=False)
        d['array'] = cropped_array
        d['crop_offset'] = corner
        d['uncropped_gpts'] = self.uncropped_gpts
        return self.__class__(**d)

    def downsample(self, max_angle: Union[str, float] = 'cutoff') -> 'SMatrixArray':
        waves = Waves(self.array, sampling=self.sampling, energy=self.energy,
                      extra_axes_metadata=self.axes_metadata[:-2])

        if self.normalization == 'probe':
            waves = waves.downsample(max_angle=max_angle, normalization='amplitude')
        elif self.normalization == 'planewaves':
            waves = waves.downsample(max_angle=max_angle, normalization='values')
        else:
            raise RuntimeError()

        d = self._copy_as_dict(copy_array=False)
        d['array'] = waves.array
        d['sampling'] = waves.sampling
        return self.__class__(**d)

    def streaming_multislice(self, potential, chunks=None, **kwargs):

        for chunk_start, chunk_stop in generate_chunks(len(self), chunks=chunks):
            extra_axes_metadata = self.extra_axes_metadata + [PrismPlaneWavesAxis()]
            waves = Waves(self.array[chunk_start:chunk_stop], energy=self.energy, sampling=self.sampling,
                          extra_axes_metadata=extra_axes_metadata)
            waves = waves.copy('gpu')
            self._array[chunk_start:chunk_stop] = waves.multislice(potential, **kwargs).copy('cpu').array

        return self

    def multislice(self,
                   potential: Union[Atoms, AbstractPotential],
                   start: int = 0,
                   stop: int = None,
                   chunks: int = None,
                   conjugate: bool = False) -> 'SMatrixArray':
        """
        Propagate the scattering matrix through the provided potential.

        Parameters
        ----------
        potential : AbstractPotential object
            Scattering potential.

        Returns
        -------
        Waves object.
            Probe exit wave functions for the provided positions.
        """

        if chunks is None:
            chunks = len(self)

        potential = validate_potential(potential)

        waves = self.to_waves()

        if self._is_streaming:
            waves._array = waves._array.map_blocks(cp.asarray)
            waves = waves.multislice(potential, start=start, stop=stop, conjugate=conjugate)
            waves._array = waves._array.map_blocks(cp.asnumpy)

        else:
            waves = waves.multislice(potential, start=start, stop=stop)

        return self.from_waves(waves)

    def _apply_reduction_func(self, func, detectors, scan, **kwargs):
        detectors = validate_detectors(detectors)

        waves = self.comparable_probe().build(positions=scan)

        new_cls = [detector.measurement_type(waves) for detector in detectors]
        new_cls_kwargs = [detector.measurement_kwargs(waves) for detector in detectors]

        signatures = []
        output_sizes = {}
        meta = []
        i = 3
        for detector in detectors:
            shape = detector.measurement_shape(waves)[self.num_extra_axes:]
            signatures.append(f'({",".join([str(i) for i in range(i, i + len(shape))])})')
            output_sizes.update({str(index): n for index, n in zip(range(i, i + len(shape)), shape)})
            meta.append(np.array((), dtype=detector.measurement_dtype))
            i += len(shape)

        signature = '(0,1,2)->' + ','.join(signatures)

        measurements = self.apply_gufunc(func,
                                         detectors=detectors,
                                         scan=scan,
                                         new_cls=new_cls,
                                         new_cls_kwargs=new_cls_kwargs,
                                         signature=signature,
                                         output_sizes=output_sizes,
                                         allow_rechunk=True,
                                         meta=meta,
                                         **kwargs)

        return measurements

    def _generate_waves(self, scan: AbstractScan, ctf: CTF, probes_per_reduction: int) -> Tuple[np.ndarray, Waves]:

        ctf = self._validate_ctf(ctf)

        scan = self._validate_positions(scan, ctf)

        probes_per_reduction = self._validate_probes_per_reduction(probes_per_reduction)

        for indices, positions in scan.generate_positions(chunks=probes_per_reduction):
            yield indices, _reduce_to_waves(self, positions, ctf, scan.axes_metadata)

    def _distributed_reduce(self,
                            detectors: List[AbstractDetector],
                            scan: AbstractScan,
                            scan_divisions: Tuple[int, int],
                            ctf: CTF,
                            probes_per_reduction: int):

        scans = scan.divide(scan_divisions)

        scans = [item for sublist in scans for item in sublist]

        measurements = []
        for scan in scans:
            cropped_s_matrix_array = self.crop_to_positions(scan)

            if self._is_streaming:
                cropped_s_matrix_array._array = cropped_s_matrix_array._array.map_blocks(cp.asarray)

            measurement = cropped_s_matrix_array._apply_reduction_func(_reduce_to_measurements,
                                                                       detectors=detectors,
                                                                       scan=scan,
                                                                       ctf=ctf,
                                                                       probes_per_reduction=probes_per_reduction)
            measurements.append(measurement)

        measurements = list(map(list, zip(*measurements)))

        for i, measurement in enumerate(measurements):
            cls = measurement[0].__class__
            d = measurement[0]._copy_as_dict(copy_array=False)

            measurement = [measurement[i:i + scan_divisions[1]] for i in range(0, len(measurement), scan_divisions[1])]

            array = da.concatenate([da.concatenate([item.array for item in block], axis=1)
                                    for block in measurement], axis=0)

            d['array'] = array
            measurements[i] = cls(**d)

        return measurements

    def reduce(self,
               detectors: Union[AbstractDetector, List[AbstractDetector]] = None,
               positions: Union[np.ndarray, AbstractScan] = None,
               ctf: Union[CTF, Dict] = None,
               distribute_scan: Union[int, Tuple[int, int]] = False,
               probes_per_reduction: int = None) -> Union[Waves, AbstractMeasurement, List[AbstractMeasurement]]:

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
        distribute_scan : two int, optional
            Partitioning of the scan. The scattering matrix will be reduced in similarly partitioned chunks.
            Should be equal to or greater than the interpolation.
        probes_per_reduction : int, optional
            Number of positions per reduction operation. To utilize thread
        """

        probes_per_reduction = self._validate_probes_per_reduction(probes_per_reduction)
        detectors = validate_detectors(detectors)
        ctf = self._validate_ctf(ctf)
        scan = self._validate_positions(positions, ctf)

        measurements = []
        for i, s_matrix in enumerate(self.to_list()):
            if distribute_scan:
                measurement = s_matrix._distributed_reduce(detectors,
                                                           scan,
                                                           distribute_scan,
                                                           ctf,
                                                           probes_per_reduction)

            else:
                measurement = s_matrix._apply_reduction_func(_reduce_to_measurements,
                                                             detectors=detectors,
                                                             scan=scan,
                                                             ctf=ctf,
                                                             probes_per_reduction=probes_per_reduction)



            measurements.append(measurement)

        measurements = list(map(list, zip(*measurements)))

        if len(self.array.shape) > 3:
            axes_metadata = self.axes_metadata[-4]
        else:
            axes_metadata = FrozenPhononsAxis()

        for i in range(len(measurements)):
            measurements[i] = stack_measurements(measurements[i], axes_metadata=axes_metadata)
            if hasattr(measurements[i], '_reduce_ensemble'):
                measurements[i] = measurements[i]._reduce_ensemble()

        measurements = [measurement.squeeze() for measurement in measurements]

        if len(measurements) == 1:
            return measurements[0]
        else:
            return ComputableList(measurements)

    def _copy_as_dict(self, copy_array: bool = True):
        d = {'energy': self.energy,
             'wave_vectors': self.wave_vectors.copy(),
             'interpolation': self.interpolation,
             'planewave_cutoff': self.planewave_cutoff,
             'sampling': self.sampling,
             'accumulated_defocus': self.accumulated_defocus,
             'crop_offset': self.crop_offset,
             'uncropped_gpts': self._uncropped_gpts,
             'tilt': self.tilt,
             'partitions': self.partitions,
             'antialias_cutoff_gpts': self.antialias_cutoff_gpts,
             'device': self._device,
             'extra_axes_metadata': deepcopy(self._extra_axes_metadata),
             'metadata': copy(self.metadata)}

        if copy_array:
            d['array'] = self.array.copy()
        return d

    def copy(self, device: str = None):
        d = self._copy_as_dict(copy_array=False)

        if device is not None:
            array = copy_to_device(self.array, device)
        else:
            array = self.array.copy()

        d['array'] = array
        return self.__class__(**d)


class SMatrix(AbstractSMatrix):
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
                 partitions: int = None,
                 normalize: bool = True,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 chunks: int = None,
                 tilt: Tuple[float, float] = None,
                 device: str = None,
                 store_on_host: bool = False):

        self._device = _validate_device(device)
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self._potential = validate_potential(potential, self)

        if potential is not None:
            self._grid = self._potential.grid

        self._interpolation = self._validate_interpolation(interpolation)
        self._planewave_cutoff = planewave_cutoff

        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._partitions = partitions

        self._normalize = normalize
        self._chunks = chunks
        self._store_on_host = store_on_host

        self._extra_axes_metadata = []
        self._antialias_cutoff_gpts = None

    @property
    def metadata(self):
        return {'energy': self.energy}

    @property
    def shape(self):
        return (len(self),) + self.gpts

    def __len__(self) -> int:
        return len(self.wave_vectors)

    @property
    def wave_vectors(self) -> np.ndarray:
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        if self._store_on_host:
            xp = np
        else:
            xp = get_array_module(self._device)

        if self.partitions is None:
            wave_vectors = prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy, self.interpolation,
                                              xp=xp)

        else:
            wave_vectors = partitioned_prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy,
                                                          num_rings=self.partitions, xp=xp)

        return wave_vectors

    @property
    def potential(self) -> AbstractPotential:
        return self._potential

    @potential.setter
    def potential(self, potential):
        self._potential = potential
        self._grid = potential.grid

    @property
    def chunks(self) -> int:
        if self._chunks is None:
            chunk_size = dask.utils.parse_bytes(dask.config.get('array.chunk-size'))
            chunks = int(chunk_size / self._bytes_per_wave())
            return chunks

        return self._chunks

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

    @property
    def partitions(self):
        return self._partitions

    def _build_chunk(self,
                     chunk_start: int = 0,
                     chunk_stop: int = None,
                     start: int = 0,
                     stop: int = None,
                     potential=None,
                     downsample: Union[bool, str] = False) -> np.ndarray:

        xp = get_array_module(self._device)

        wave_vectors = prism_wave_vectors(self.planewave_cutoff, self.extent, self.energy, self.interpolation, xp)

        if chunk_stop is None:
            chunk_stop = len(wave_vectors)

        array = plane_waves(wave_vectors[chunk_start:chunk_stop], self.extent, self.gpts)

        if self.normalize:
            normalization_constant = np.prod(self.gpts) * xp.sqrt(len(wave_vectors)) / np.prod(self.interpolation)
            array = array / normalization_constant.astype(xp.float32)
        else:
            array = array / xp.sqrt(np.prod(self.gpts).astype(xp.float32))

        if potential is not None and stop != 0:
            waves = Waves(array, extent=self.extent, energy=self.energy, extra_axes_metadata=[OrdinalAxis()])

            waves = multislice(waves, potential, start=start, stop=stop)

            if downsample:
                waves = waves.downsample(max_angle=downsample)

            array = waves.array

        if self._store_on_host and self._device == 'gpu':
            with cp.cuda.Stream():
                array = cp.asnumpy(array)

        return array

    def generate_distribution(self,
                              lazy: bool = None,
                              downsample: Union[float, str] = False,
                              start: int = 0,
                              stop: int = None,
                              yield_potential: bool = False) -> List[SMatrixArray]:

        lazy = validate_lazy(lazy)

        if self.potential:
            self.grid.match(self.potential)

        self.grid.check_is_defined()

        xp = get_array_module(self._device)

        if self._store_on_host:
            storage_xp = np
        else:
            storage_xp = xp

        gpts = self._gpts_within_angle(downsample) if downsample else self.gpts

        for potential in [None] if self.potential is None else self.potential.get_configurations(lazy=lazy):

            arrays = []
            for chunk_start, chunk_stop in generate_chunks(len(self), chunks=self.chunks):

                if lazy:
                    delayed_potential = potential.to_delayed() if potential is not None else potential
                    array = dask.delayed(self._build_chunk)(chunk_start, chunk_stop, start, stop, delayed_potential,
                                                            downsample)
                    array = da.from_delayed(array,
                                            shape=(chunk_stop - chunk_start,) + gpts,
                                            meta=storage_xp.array((), dtype=storage_xp.complex64))

                else:

                    array = self._build_chunk(chunk_start, chunk_stop, start, stop, potential, downsample)

                arrays.append(array)

            if lazy:
                array = da.concatenate(arrays)
            else:
                array = storage_xp.concatenate(arrays)

            s_matrix = SMatrixArray(array,
                                    interpolation=self.interpolation,
                                    planewave_cutoff=self.planewave_cutoff,
                                    sampling=(self.extent[0] / gpts[0], self.extent[1] / gpts[1]),
                                    energy=self.energy,
                                    tilt=self.tilt,
                                    partitions=self.partitions,
                                    wave_vectors=self.wave_vectors,
                                    antialias_cutoff_gpts=self.antialias_cutoff_gpts,
                                    device=self._device,
                                    metadata=self.metadata)

            if self.potential is not None:
                s_matrix.accumulated_defocus = self.potential.thickness

            if yield_potential:
                yield s_matrix, potential
            else:
                yield s_matrix

    def build(self,
              start: int = 0,
              stop: int = None,
              lazy: bool = None,
              downsample: Union[float, str] = False) -> SMatrixArray:
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
        downsample : float or str or False
            If not False, the scattering matrix is downsampled to a maximum given scattering angle after running the
            multislice algorithm. If downsample is given as a float angle may be given as a float

            is given the scattering matrix is downsampled to a maximum scattering angle

        Returns
        -------
        SMatrixArray
        """


        generator = self.generate_distribution(lazy=lazy, start=start, stop=stop, downsample=downsample)
        s_matrices = [s_matrix for s_matrix in generator]

        if len(s_matrices) > 1:
            return stack_s_matrices(s_matrices, FrozenPhononsAxis(ensemble_mean=self.potential.ensemble_mean))
        else:
            return s_matrices[0]

    def _validate_ctf(self, ctf):
        if ctf is None:
            ctf = CTF(energy=self.energy, semiangle_cutoff=self.planewave_cutoff)

        if ctf.semiangle_cutoff is None:
            ctf.semiangle_cutoff = self.planewave_cutoff

        ctf.accelerator.check_is_defined()
        return ctf

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

        lazy = validate_lazy(lazy)

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
