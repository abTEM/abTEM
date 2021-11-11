import itertools
from copy import copy
from typing import Union, Sequence, Tuple, Dict, List

import dask
import dask.array as da
import numpy as np
from ase import Atoms

from abtem.basic.antialias import AntialiasAperture
from abtem.basic.axes import frozen_phonons_axes_metadata
from abtem.basic.backend import get_array_module, cp, copy_to_device
from abtem.basic.complex import complex_exponential
from abtem.basic.dask import HasDaskArray, ComputableList
from abtem.basic.energy import energy2wavelength
from abtem.basic.grid import Grid
from abtem.basic.utils import generate_chunks
from abtem.measure.detect import AbstractDetector
from abtem.potentials.potentials import AbstractPotential
from abtem.waves.base import BeamTilt, AbstractScannedWaves
from abtem.waves.multislice import multislice
from abtem.waves.scan import AbstractScan, GridScan
from abtem.waves.transfer import CTF

from dask import graph_manipulation

from abtem.waves.waves import Waves, Probe

if cp is not None:
    from abtem.basic.cuda import batch_crop_2d as batch_crop_2d_cuda
else:
    batch_crop_2d_cuda = None


def wrapped_slices(start, stop, n):
    # if stop - start > n:
    # raise RuntimeError(f'{start} {stop} {n} {stop - start}')

    if start < 0:
        if stop > n:
            raise RuntimeError()

        a = slice(start % n, None)
        b = slice(0, stop)

    elif stop > n:
        if start < 0:
            raise RuntimeError()

        a = slice(start, None)
        b = slice(0, stop - n)

    else:
        a = slice(start, stop)
        b = slice(0, 0)
    return a, b


def wrapped_crop_2d(array, corner, size):
    upper_corner = (corner[0] + size[0], corner[1] + size[1])

    xp = get_array_module(array)

    a, c = wrapped_slices(corner[0], upper_corner[0], array.shape[-2])
    b, d = wrapped_slices(corner[1], upper_corner[1], array.shape[-1])

    A = array[..., a, b]
    B = array[..., c, b]
    D = array[..., c, d]
    C = array[..., a, d]

    if A.size == 0:
        AB = B
    elif B.size == 0:
        AB = A
    else:
        AB = xp.concatenate([A, B], axis=-2)

    if C.size == 0:
        CD = D
    elif D.size == 0:
        CD = C
    else:
        CD = xp.concatenate([C, D], axis=-2)

    if CD.size == 0:
        return AB

    if AB.size == 0:
        return CD

    return xp.concatenate([AB, CD], axis=-1)


def batch_crop_2d(array, corners, new_shape):
    xp = get_array_module(array)

    if xp is cp:
        return batch_crop_2d_cuda(array, corners, new_shape)
    else:
        array = np.lib.stride_tricks.sliding_window_view(array, (1,) + new_shape)
        return array[xp.arange(array.shape[0]), corners[:, 0], corners[:, 1], 0]


_plane_waves_axes_metadata = {'label': 'plane_waves', 'type': 'ensemble'}


def prism_coefficients(positions, wave_vectors, wavelength, ctf):
    xp = get_array_module(wave_vectors)
    positions = copy_to_device(positions, xp)

    def _calculate_ctf_coefficient(wave_vectors, wavelength, ctf):
        alpha = xp.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2) * wavelength
        phi = xp.arctan2(wave_vectors[:, 0], wave_vectors[:, 1])
        coefficients = ctf.evaluate(alpha, phi)
        return coefficients

    def _calculate_coefficents(wave_vectors, positions):
        coefficients = complex_exponential(-2. * xp.pi * positions[..., 0, None] * wave_vectors[:, 0][None])
        coefficients *= complex_exponential(-2. * xp.pi * positions[..., 1, None] * wave_vectors[:, 1][None])
        return coefficients

    coefficients = _calculate_coefficents(wave_vectors, positions)
    ctf_coefficients = _calculate_ctf_coefficient(wave_vectors, wavelength=wavelength, ctf=ctf)
    return coefficients * ctf_coefficients


def prism_wave_vectors(cutoff, extent, energy, interpolation, xp):
    xp = get_array_module(xp)
    wavelength = energy2wavelength(energy)

    n_max = int(xp.ceil(cutoff / 1.e3 / (wavelength / extent[0] * interpolation)))
    m_max = int(xp.ceil(cutoff / 1.e3 / (wavelength / extent[1] * interpolation)))

    n = xp.arange(-n_max, n_max + 1, dtype=np.float32)
    w = xp.asarray(extent[0], dtype=np.float32)
    m = xp.arange(-m_max, m_max + 1, dtype=np.float32)
    h = xp.asarray(extent[1], dtype=np.float32)

    kx = n / w * np.float32(interpolation)
    ky = m / h * np.float32(interpolation)

    mask = kx[:, None] ** 2 + ky[None, :] ** 2 < (cutoff / 1.e3 / wavelength) ** 2
    kx, ky = xp.meshgrid(kx, ky, indexing='ij')
    kx = kx[mask]
    ky = ky[mask]
    return xp.asarray((kx, ky)).T


def _minimum_crop(positions: Union[Sequence[float], GridScan], sampling, shape):
    if all(hasattr(positions, attr) for attr in ('start', 'end')):
        positions = [positions.start, positions.end]

    offset = (shape[0] // 2, shape[1] // 2)
    corners = np.rint(np.array(positions) / sampling - offset).astype(np.int)
    upper_corners = corners + np.asarray(shape)
    crop_corner = (np.min(corners[..., 0]).item(), np.min(corners[..., 1]).item())

    size = (np.max(upper_corners[..., 0]).item() - crop_corner[0],
            np.max(upper_corners[..., 1]).item() - crop_corner[1])

    corners -= crop_corner
    return crop_corner, size, corners


class SMatrixArray(HasDaskArray, AbstractScannedWaves):
    """
    Scattering matrix array object.

    The scattering matrix array object represents a plane wave expansion of a probe, it is used for STEM simulations
    with the PRISM algorithm.

    Parameters
    ----------
    array : 3d array
        The array representation of the scattering matrix.
    expansion_cutoff : float
        The angular cutoff of the plane wave expansion [mrad].
    energy : float
        Electron energy [eV].
    k : 2d array
        The spatial frequencies of each plane in the plane wave expansion.
    ctf : CTF object, optional
        The probe contrast transfer function. Default is None.
    extent : one or two float, optional
        Lateral extent of wave functions [Å]. Default is None (inherits extent from the potential).
    sampling : one or two float, optional
        Lateral sampling of wave functions [1 / Å]. Default is None (inherits sampling from the potential).
    tilt : two float, optional
        Small angle beam tilt [mrad].
    periodic : bool, optional
        Should the scattering matrix array be considered periodic. This may be false if the scattering matrix is a
        cropping of a larger scattering matrix.
    interpolated_gpts : two int, optional
        The gpts of the probe window after Fourier interpolation. This may differ from the shape determined by dividing
        each side by the interpolation is the scattering matrix array is cropped from a larger scattering matrix.
    antialiasing_aperture : two float, optional
        Assumed antialiasing aperture as a fraction of the real space Nyquist frequency. Default is 2/3.
    device : str, optional
        The calculations will be carried out on this device. Default is 'cpu'.
    """

    def __init__(self,
                 array: Union[np.ndarray, da.core.Array],
                 energy: float,
                 wave_vectors: np.ndarray,
                 interpolation: int = 1,
                 extent: Union[float, Tuple[float, float]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 tilt: Tuple[float, float] = None,
                 ctf: CTF = None,
                 antialias_aperture: float = None,
                 device: str = 'cpu',
                 max_concurrent: int = None,
                 axes_metadata: List[Dict] = None,
                 metadata: Dict = None):

        if ctf is None:
            ctf = CTF()

        if ctf.energy is None:
            ctf.energy = energy

        if ctf.energy != energy:
            raise RuntimeError

        self._interpolation = interpolation

        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_aperture = AntialiasAperture(cutoff=antialias_aperture)

        self._ctf = ctf
        self._accelerator = self._ctf._accelerator
        self._max_concurrent = max_concurrent

        self._device = device
        self._array = array
        self._wave_vectors = wave_vectors

        if axes_metadata is None:
            axes_metadata = []

        self._axes_metadata = axes_metadata
        self._metadata = metadata
        super().__init__(array)

    @property
    def chunks(self):
        return self.array.chunks[:-2]

    @property
    def axes_metadata(self):
        return self._axes_metadata + [_plane_waves_axes_metadata] + self._base_axes_metadata

    @property
    def metadata(self):
        return self._metadata

    def rechunk(self, **kwargs):
        self._array = self._array.rechunk(**kwargs)
        self._wave_vectors = self._wave_vectors.rechunk((self._array.chunks[0], (2,)))

    @property
    def ctf(self) -> CTF:
        """Probe contrast transfer function."""
        return self._ctf

    @property
    def wave_vectors(self) -> np.ndarray:
        """The spatial frequencies of each wave in the plane wave expansion."""
        return self._wave_vectors

    @property
    def interpolation(self) -> int:
        """Interpolation factor."""
        return self._interpolation

    @property
    def interpolated_gpts(self) -> Tuple[int, int]:
        return (self.gpts[0] // self.interpolation, self.gpts[1] // self.interpolation)

    def downsample(self, max_angle='cutoff') -> 'SMatrixArray':

        waves = Waves(self.array, extent=self.extent, energy=self.energy, antialias_aperture=self.antialias_aperture)

        downsampled_waves = waves.downsample(max_angle=max_angle)

        return self.__class__(array=downsampled_waves.array,
                              wave_vectors=self.wave_vectors.copy(),
                              interpolation=self.interpolation,
                              ctf=self.ctf.copy(),
                              extent=self.extent,
                              energy=self.energy,
                              axes_metadata=self._axes_metadata,
                              antialias_aperture=downsampled_waves.antialias_aperture)

    def multislice(self, potential: AbstractPotential, chunks: int = 1):
        """
        Propagate the scattering matrix through the provided potential.

        Parameters
        ----------
        potential : AbstractPotential object
            Scattering potential.
        max_batch : int, optional
            The probe batch size. Larger batches are faster, but require more memory. Default is None.

        Returns
        -------
        Waves object.
            Probe exit wave functions for the provided positions.
        """
        raise NotImplementedError

    def _get_coefficient(self, positions):
        return prism_coefficients(positions, self.wave_vectors, self.wavelength, self.ctf)

    def _get_minimum_crop(self, positions):
        return _minimum_crop(positions, self.sampling, self.interpolated_gpts)

    def reduce(self, positions, detectors, scan_partitions, positions_per_reduction, lazy: bool = False):
        detectors = self._validate_detectors(detectors)
        scans = positions.divide(scan_partitions)
        xp = get_array_module(self.array)

        def reduce_subchunk(array,
                            positions,
                            coefficients,
                            sampling,
                            interpolated_gpts):

            xp = get_array_module(array)

            if not array.shape[-2:] == interpolated_gpts:
                crop_corner, size, corners = _minimum_crop(positions, sampling, interpolated_gpts)
                array = wrapped_crop_2d(array, crop_corner, size)
                window = xp.tensordot(coefficients, array, axes=[-1, -3])
                window = batch_crop_2d(window, corners.reshape((-1, 2)), self.interpolated_gpts)
            else:
                window = xp.tensordot(coefficients, array, axes=[-1, -3])

            window = window.reshape(positions.shape[:-1] + window.shape[-2:])
            axes_metadata = [{'type': 'positions'}, {'type': 'positions'}]
            waves = Waves(window, sampling=self.sampling, energy=self.energy, axes_metadata=axes_metadata)
            return waves

        def reshape_list(l, n):
            return [l[i:i + n] for i in range(0, len(l), n)]

        def reduce(array,
                   scan,
                   detectors,
                   sub_chunks,
                   corner,
                   wave_vectors,
                   wavelength,
                   ctf,
                   sampling,
                   interpolated_gpts):

            positions = scan.get_positions(chunks=sub_chunks, lazy=False)
            xp = get_array_module(array)

            measurements = {detector: [] for detector in detectors}
            for positions_chunk in itertools.chain(*positions):

                coefficients = prism_coefficients(positions_chunk.reshape((-1, 2)), wave_vectors, wavelength, ctf)
                shifted_positions = positions_chunk - np.array(corner) * np.array(sampling)

                waves = reduce_subchunk(array,
                                        shifted_positions,
                                        coefficients,
                                        sampling,
                                        interpolated_gpts
                                        )

                for detector in detectors:
                    measurements[detector].append(detector.detect(waves).array)

            result = {}
            for detector, measurement in measurements.items():
                result[detector] = xp.block(reshape_list(measurement, len(positions[0])))

            return list(result.values())

        def scan_reduce(array, scans, detectors):

            measurements = [[] for _ in range(len(detectors))]
            for scan in itertools.chain(*scans):

                if self.interpolation == 1:
                    corner = [0., 0.]
                    cropped_array = array
                else:
                    corner, size, _ = self._get_minimum_crop(scan)
                    cropped_array = array.map_blocks(wrapped_crop_2d,
                                                     corner=corner,
                                                     size=size,
                                                     chunks=(self.array.chunks[1], (size[0],), (size[1],)),
                                                     meta=xp.array((), dtype=xp.complex64))

                new_measurements = dask.delayed(reduce, nout=len(detectors))(cropped_array,
                                                                             scan,
                                                                             detectors,
                                                                             positions_per_reduction,
                                                                             corner,
                                                                             wave_vectors=self.wave_vectors,
                                                                             wavelength=self.wavelength,
                                                                             ctf=self.ctf,
                                                                             sampling=self.sampling,
                                                                             interpolated_gpts=self.interpolated_gpts,
                                                                             )

                for i, measurement in enumerate(new_measurements):
                    measurement = da.from_delayed(measurement,
                                                  shape=scan.gpts,
                                                  meta=xp.array((), dtype=xp.float32))

                    measurements[i].append(measurement)

            output = []
            for measurement in measurements:
                output.append(da.block(reshape_list(measurement, len(scans))))

            return output

        measurements = []
        for i in range(len(self)):
            measurement = scan_reduce(self.array[i], scans, detectors)

            if self._max_concurrent is not None:
                if i >= self._max_concurrent:
                    measurement = graph_manipulation.bind(measurement, measurements[i - self._max_concurrent])
            else:
                measurement = graph_manipulation.wait_on(measurement)

            measurements.append(measurement)

        measurements = list(map(da.stack, map(list, zip(*measurements))))

        for i, (detector, measurement) in enumerate(zip(detectors, measurements)):
            from dask.graph_manipulation import wait_on
            if detector.ensemble_mean:
                # measurement = wait_on(measurement)

                measurement = measurement.mean(0)
                axes_metadata = []
            else:
                axes_metadata = [{'label': 'frozen_phonons', 'type': 'ensemble'}]

            measurements[i] = detector.measurement_from_array(measurement, scan=positions, waves=self,
                                                              axes_metadata=axes_metadata)

        if len(measurements) == 1:
            output = measurements[0]
        else:
            output = ComputableList(measurements)

        if not lazy:
            output.compute()

        return output

    def scan(self,
             scan: AbstractScan,
             detectors: Sequence[AbstractDetector]):

        """
        Raster scan the probe across the potential and record a measurement for each detector.

        Parameters
        ----------
        scan : Scan object
            Scan defining the positions of the probe wave functions.
        detectors : List of Detector objects
            The detectors recording the measurements.
        max_batch_probes : int, optional
            The probe batch size. Larger batches are faster, but require more memory. Default is None.
        max_batch_expansion : int, optional
            The expansion plane wave batch size. Default is None.
        pbar : bool, optional
            Display progress bars. Default is True.

        Returns
        -------
        dict
            Dictionary of measurements with keys given by the detector.
        """
        return self.reduce(scan).detect(detectors)

    def __copy__(self):
        return self.__class__(array=self.array.copy(),
                              wave_vectors=self.wave_vectors.copy(),
                              ctf=self.ctf.copy(),
                              extent=self.extent,
                              energy=self.energy,
                              antialias_aperture=self.antialias_aperture)

    def copy(self):
        """Make a copy."""
        return copy(self)


def plane_waves(wave_vectors, extent, gpts):
    xp = get_array_module(wave_vectors)
    x = xp.linspace(0, extent[0], gpts[0], endpoint=False, dtype=np.float32)
    y = xp.linspace(0, extent[1], gpts[1], endpoint=False, dtype=np.float32)
    array = (complex_exponential(2 * np.pi * wave_vectors[:, 0, None, None] * x[:, None]) *
             complex_exponential(2 * np.pi * wave_vectors[:, 1, None, None] * y[None, :]))
    return array


class SMatrix(AbstractScannedWaves):
    """
    Scattering matrix builder class

    The scattering matrix builder object is used for creating scattering matrices and simulating STEM experiments using
    the PRISM algorithm.

    Parameters
    ----------
    expansion_cutoff : float
        The angular cutoff of the plane wave expansion [mrad].
    energy : float
        Electron energy [eV].
    interpolation : one or two int, optional
        Interpolation factor. Default is 1 (no interpolation).
    ctf: CTF object, optional
        The probe contrast transfer function. Default is None (aperture is set by the cutoff of the expansion).
    num_partitions : int
        The number of partitions used for in the parent scattering matrix.
    extent : one or two float, optional
        Lateral extent of wave functions [Å]. Default is None (inherits the extent from the potential).
    gpts : one or two int, optional
        Number of grid points describing the wave functions. Default is None (inherits the gpts from the potential).
    sampling : one or two float, None
        Lateral sampling of wave functions [1 / Å]. Default is None (inherits the sampling from the potential.
    tilt : two float
        Small angle beam tilt [mrad].
    device : str, optional
        The calculations will be carried out on this device. Default is 'cpu'.
    storage : str, optional
        The scattering matrix will be stored on this device. Default is None (uses the option chosen for device).
    kwargs :
        The parameters of a new CTF object as keyword arguments.
    """

    def __init__(self,
                 energy: float,
                 expansion_cutoff: float = None,
                 interpolation: int = 1,
                 ctf: CTF = None,
                 num_partitions: int = None,
                 chunks: int = None,
                 max_concurrent: int = None,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 tilt: Tuple[float, float] = None,
                 device: str = 'cpu',
                 storage: str = None,
                 **kwargs):

        if not isinstance(interpolation, int):
            raise ValueError('Interpolation factor must be int')

        self._interpolation = interpolation

        if ctf is None:
            ctf = CTF(**kwargs)

        if ctf.energy is None:
            ctf.energy = energy

        if (ctf.energy != energy):
            raise RuntimeError

        if (expansion_cutoff is None) and ('semiangle_cutoff' in kwargs):
            expansion_cutoff = kwargs['semiangle_cutoff']

        if expansion_cutoff is None:
            raise ValueError('')

        self._expansion_cutoff = expansion_cutoff

        self._ctf = ctf
        self._accelerator = self._ctf._accelerator
        self._antialias_aperture = AntialiasAperture()
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._max_concurrent = max_concurrent

        self._device = device

        if storage is None:
            storage = device

        self._storage = storage
        self._chunks = chunks
        self._num_partitions = num_partitions

    @property
    def axes_metadata(self):
        return [_plane_waves_axes_metadata] + self._base_axes_metadata

    @property
    def chunks(self):
        return self._chunks

    @property
    def ctf(self):
        """The contrast transfer function of the probes."""
        return self._ctf

    @ctf.setter
    def ctf(self, value):
        self._ctf = value

    @property
    def expansion_cutoff(self) -> float:
        """Plane wave expansion cutoff."""
        return self._expansion_cutoff

    @expansion_cutoff.setter
    def expansion_cutoff(self, value: float):
        self._expansion_cutoff = value

    @property
    def interpolation(self) -> int:
        """Interpolation factor."""
        return self._interpolation

    @property
    def interpolated_gpts(self) -> Tuple[int, int]:
        return (self.gpts[0] // self.interpolation, self.gpts[1] // self.interpolation)

    def as_probe(self):
        extent = (self.extent[0] / self.interpolation, self.extent[1] / self.interpolation)

        return Probe(extent=extent,
                     gpts=self.interpolated_gpts,
                     energy=self.energy,
                     ctf=self.ctf,
                     device=self._device)

    def multislice(self, potential: AbstractPotential):
        """
        Build scattering matrix and propagate the scattering matrix through the provided potential.

        Parameters
        ----------
        potential : AbstractPotential
            Scattering potential.
        max_batch : int, optional
            The probe batch size. Larger batches are faster, but require more memory. Default is None.
        pbar : bool, optional
            Display progress bars. Default is True.

        Returns
        -------
        Waves object
            Probe exit wave functions as a Waves object.
        """

        potential = self._validate_potential(potential)
        self.grid.check_is_defined()

        # exit_waves.append(multislice(self.build(), p))

        def multislice_reduce(wave_vectors, potential, extent, gpts, energy):
            array = plane_waves(wave_vectors, extent, gpts)
            waves = Waves(array, extent=extent, energy=energy)

            return array

        wave_vectors = self.wave_vectors

        xp = get_array_module(self._device)

        for p in potential.get_slice_iterators():
            s_matrix_chunks = []
            for start, end in generate_chunks(len(wave_vectors), chunks=self.chunks):
                chunk = dask.delayed(multislice_reduce)(wave_vectors[start:end],
                                                        potential=p,
                                                        extent=self.extent,
                                                        gpts=self.gpts,
                                                        energy=self.energy
                                                        )

                chunk = da.from_delayed(chunk, shape=(end - start,) + self.gpts, meta=xp.array((), dtype=xp.complex64))

                s_matrix_chunks.append(chunk)

            stack = da.concatenate(s_matrix_chunks)

        return stack

        # exit_waves = []
        # exit_waves.append(multislice(self.build(), potential))
        # # for p in potential.frozen_phonon_potentials():
        # #    exit_waves.append(multislice(self.build(), p))
        #
        # if len(exit_waves) > 1:
        #     array = da.stack([exit_wave.array for exit_wave in exit_waves], axis=0)
        #     extra_axes_metadata = [{'label': 'frozen_phonons', 'type': 'ensemble'}]
        # else:
        #     array = exit_waves[0].array
        #     extra_axes_metadata = []
        #
        # return SMatrixArray(array,
        #                     interpolation=self.interpolation,
        #                     extent=self.extent,
        #                     energy=self.energy,
        #                     tilt=self.tilt,
        #                     wave_vectors=self.wave_vectors,
        #                     ctf=self.ctf.copy(),
        #                     antialias_aperture=self.antialias_aperture,
        #                     device=self._device,
        #                     extra_axes_metadata=extra_axes_metadata,
        #                     metadata={'energy': self.energy})

        # return self.build().multislice(potential)

    def scan(self,
             scan: AbstractScan,
             detectors: Sequence[AbstractDetector],
             potential: Union[Atoms, AbstractPotential],
             scan_partitions,
             positions_per_reduction,
             lazy: bool = False):
        """
        Build the scattering matrix. Raster scan the probe across the potential, record a measurement for each detector.

        Parameters
        ----------
        scan : Scan object
            Scan defining the positions of the probe wave functions.
        detectors : List of Detector objects
            The detectors recording the measurements.
        potential : Potential object
            The potential to scan the probe over.
        max_batch_probes : int, optional
            The probe batch size. Larger batches are faster, but require more memory. Default is None.
        max_batch_expansion : int, optional
            The expansion plane wave batch size. Default is None.
        pbar : bool, optional
            Display progress bars. Default is True.

        Returns
        -------
        dict
            Dictionary of measurements with keys given by the detector.
        """

        self.grid.match(potential.grid)
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        if hasattr(scan, 'match'):
            scan.match(self)

        S = self._build_multislice_downsample(potential)

        measurements = S.reduce(scan,
                                detectors,
                                scan_partitions=scan_partitions,
                                positions_per_reduction=positions_per_reduction, lazy=lazy)

        return measurements

    def __len__(self):
        return len(self.wave_vectors)

    @property
    def wave_vectors(self):
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        wave_vectors = prism_wave_vectors(self.expansion_cutoff,
                                          self.extent,
                                          self.energy,
                                          self.interpolation,
                                          self._device)

        return wave_vectors

    def _build_partial(self, start, end):
        xp = get_array_module(self._device)
        wave_vectors = prism_wave_vectors(self.expansion_cutoff, self.extent, self.energy, self.interpolation, xp)
        array = plane_waves(wave_vectors[start:end], self.extent, self.gpts)
        # xp = get_array_module(array)
        # interpolated_gpts = (gpts[0] // interpolation, self.gpts[1] // interpolation)
        # probe = (xp.abs(array.sum(0)) ** 2)[:interpolated_gpts[0], :interpolated_gpts[1]]
        # array /= xp.sqrt(probe.sum()) * xp.sqrt(interpolated_gpts[0] * interpolated_gpts[1])
        return array

    def _build_multislice_downsample(self, potential):

        def build_multislice_downsample(start, end, potential):
            array = self._build_partial(start, end)
            waves = Waves(array, extent=self.extent, energy=self.energy)
            waves = waves.multislice(potential, lazy=False)
            waves = waves.downsample(max_angle='cutoff')
            return waves.array

        xp = get_array_module(self._device)

        frozen_phonon_arrays = []
        for i, projected_potential in enumerate(potential.get_projected_potentials(lazy=True)):

            arrays = []
            for start, end in generate_chunks(len(self), chunks=self.chunks):
                array = dask.delayed(build_multislice_downsample)(start, end, projected_potential)
                array = da.from_delayed(array,
                                        shape=(end - start,) + self.antialias_cutoff_gpts,
                                        meta=xp.array((), dtype=xp.complex64))

                arrays.append(array)

            frozen_phonon_arrays.append(da.concatenate(arrays, axis=0))

        array = da.stack(frozen_phonon_arrays)
        axes_metadata = [frozen_phonons_axes_metadata]

        return SMatrixArray(array,
                            interpolation=self.interpolation,
                            extent=self.extent,
                            energy=self.energy,
                            tilt=self.tilt,
                            wave_vectors=self.wave_vectors,
                            ctf=self.ctf.copy(),
                            antialias_aperture=self.antialias_aperture,
                            device=self._device,
                            max_concurrent=self._max_concurrent,
                            axes_metadata=axes_metadata,
                            metadata={'energy': self.energy})

    def build(self, lazy=True) -> Union[SMatrixArray]:
        """Build the scattering matrix."""

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        arrays = []
        for start, end in generate_chunks(len(self), chunks=self.chunks):
            array = dask.delayed(self._build_partial)(start, end)
            array = da.from_delayed(array, shape=(end - start,) + self.gpts, dtype=np.complex64)
            arrays.append(array)

        array = da.concatenate(arrays)
        return SMatrixArray(array,
                            interpolation=self.interpolation,
                            extent=self.extent,
                            energy=self.energy,
                            tilt=self.tilt,
                            wave_vectors=self.wave_vectors,
                            ctf=self.ctf.copy(),
                            antialias_aperture=self.antialias_aperture,
                            device=self._device,
                            axes_metadata=[],
                            metadata={'energy': self.energy})

    def profile(self, angle=0.):
        return self.as_probe().profile(angle=angle)

    def show(self, **kwargs):
        """
        Show the probe wave function.

        Parameters
        ----------
        angle : float, optional
            Angle along which the profile is shown [deg]. Default is 0 degrees.
        kwargs : Additional keyword arguments for the abtem.plot.show_image function.
        """
        return self.build().reduce().intensity().show(**kwargs)

    def __copy__(self) -> 'SMatrix':
        return self.__class__(expansion_cutoff=self.expansion_cutoff,
                              interpolation=self.interpolation,
                              ctf=self.ctf.copy(),
                              extent=self.extent,
                              gpts=self.gpts,
                              energy=self.energy,
                              device=self._device,
                              storage=self._storage)

    def copy(self) -> 'SMatrix':
        """Make a copy."""
        return copy(self)
