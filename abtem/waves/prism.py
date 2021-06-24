from copy import copy
from typing import Union, Sequence, Tuple, Dict

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

from abtem.base_classes import Cache, cached_method, HasEventMixin, Event
from abtem.basic.antialias import AntialiasAperture
from abtem.basic.complex import complex_exponential
from abtem.basic.fft import fft2_shift_kernel
from abtem.basic.grid import Grid
from abtem.basic.utils import subdivide_into_batches, periodic_crop, array_row_intersection
from abtem.device import get_array_module, get_device_function, get_array_module_from_device, \
    copy_to_device, get_available_memory, get_device_from_array
from abtem.measure.detect import AbstractDetector
from abtem.measure.measure import Measurement, probe_profile
from abtem.potentials import Potential, AbstractPotential
from abtem.waves.base import BeamTilt, AbstractScannedWaves
from abtem.waves.scan import AbstractScan, GridScan
from abtem.waves.transfer import CTF
from abtem.waves.waves import Waves
import dask


class SMatrixArray(AbstractScannedWaves):
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
                 array: np.ndarray,
                 energy: float,
                 k: np.ndarray,
                 ctf: CTF = None,
                 extent: Union[float, Tuple[float, float]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 tilt: Tuple[float, float] = None,
                 periodic: bool = True,
                 offset: Tuple[int, int] = (0, 0),
                 interpolated_gpts: Tuple[int, int] = None,
                 antialias_aperture: float = None,
                 device: str = 'cpu'):

        if ctf is None:
            ctf = CTF()

        if ctf.energy is None:
            ctf.energy = energy

        if (ctf.energy != energy):
            raise RuntimeError

        self._ctf = ctf
        self._accelerator = self._ctf._accelerator
        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_aperture = AntialiasAperture(cutoff=antialias_aperture)

        self._device = device
        self._array = array
        self._k = k
        self._interpolated_gpts = interpolated_gpts

        self._periodic = periodic
        self._offset = offset

    @property
    def ctf(self) -> CTF:
        """Probe contrast transfer function."""
        return self._ctf

    @property
    def array(self) -> np.ndarray:
        """Array representing the scattering matrix."""
        return self._array

    @property
    def k(self) -> np.ndarray:
        """The spatial frequencies of each wave in the plane wave expansion."""
        return self._k

    @property
    def periodic(self) -> bool:
        return self._periodic

    @property
    def interpolated_gpts(self) -> Tuple[int, int]:
        """The grid of the interpolated scattering matrix."""
        return self._interpolated_gpts

    @property
    def _interpolated_grid(self):
        return Grid(gpts=self._interpolated_gpts, sampling=self.sampling)

    @property
    def offset(self) -> Tuple[int, int]:
        return self._offset

    def __len__(self) -> int:
        """Number of plane waves in expansion."""
        return len(self._array)

    def _raise_not_periodic(self):
        raise RuntimeError('not implemented for non-periodic/cropped scattering matrices')

    def downsample(self, max_angle='limit') -> 'SMatrixArray':
        if not self.periodic:
            self._raise_not_periodic()

        xp = get_array_module(self.array)
        gpts = next(self._generate_partial(1, pbar=False))[2].downsample(max_angle).gpts

        new_array = xp.zeros((len(self.array),) + gpts, dtype=self.array.dtype)
        max_batch = self._max_batch_expansion()

        for start, end, partial_s_matrix in self._generate_partial(max_batch, pbar=False):
            downsampled = partial_s_matrix.downsample(max_angle)
            new_array[start:end] = copy_to_device(downsampled.array, xp)

        if self.interpolated_gpts == self.gpts:
            interpolated_gpts = gpts
        else:
            interpolated_gpts = tuple(n // (self.gpts[i] // self.interpolated_gpts[i]) for i, n in enumerate(gpts))

        antialias_aperture = downsampled.antialias_aperture

        return self.__class__(array=new_array,
                              k=self.k.copy(),
                              ctf=self.ctf,
                              extent=self.extent,
                              energy=self.energy,
                              periodic=self.periodic,
                              offset=self._offset,
                              interpolated_gpts=interpolated_gpts,
                              antialias_aperture=antialias_aperture,
                              device=self.device)

    def crop_to_scan(self, scan) -> 'SMatrixArray':

        if not isinstance(scan, GridScan):
            raise NotImplementedError()

        crop_corner, size = self._get_requisite_crop(np.array([scan.start, scan.end]))
        new_array = periodic_crop(self.array, crop_corner, size)

        return self.__class__(array=new_array,
                              k=self.k.copy(),
                              ctf=self.ctf,
                              sampling=self.sampling,
                              energy=self.energy,
                              periodic=False,
                              offset=crop_corner,
                              interpolated_gpts=self.interpolated_gpts,
                              device=self.device)

    def _max_batch_expansion(self) -> int:
        memory_per_wave = 2 * 4 * self.gpts[0] * self.gpts[1]
        available_memory = .2 * get_available_memory(self._device)
        return min(int(available_memory / memory_per_wave), len(self))

    def _max_batch_probes(self) -> int:
        max_batch_plane_waves = self._max_batch_expansion()
        memory_per_wave = 2 * 4 * self.interpolated_gpts[0] * self.interpolated_gpts[1]
        memory_per_plane_wave_batch = 2 * 4 * self.gpts[0] * self.gpts[1] * max_batch_plane_waves
        available_memory = .2 * get_available_memory(self._device) - memory_per_plane_wave_batch
        return max(min(int(available_memory / memory_per_wave), 1024), 1)

    def _generate_partial(self, max_batch: int = None, pbar: Union[bool] = True) -> Waves:
        if max_batch is None:
            n_batches = 1
        else:
            n_batches = (len(self) + (-len(self) % max_batch)) // max_batch

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(self), desc='Plane waves', disable=(not pbar) or (n_batches == 1))
            close_pbar = True
        else:
            pbar.reset()
            close_pbar = False

        xp = get_array_module_from_device(self._device)

        n = 0
        for batch_size in subdivide_into_batches(len(self), n_batches):
            start = n
            end = n + batch_size

            if xp != get_array_module(self.array):
                yield start, end, Waves(copy_to_device(self._array[start:end], xp),
                                        extent=self.extent, energy=self.energy,
                                        antialias_aperture=self.antialias_aperture)
            else:
                yield start, end, Waves(self._array[start:end], extent=self.extent, energy=self.energy,
                                        antialias_aperture=self.antialias_aperture)

            n += batch_size
            pbar.update(batch_size)

        pbar.refresh()
        if close_pbar:
            pbar.close()

    def multislice(self,
                   potential: AbstractPotential,
                   max_batch: int = None,
                   multislice_pbar: Union[bool] = True,
                   plane_waves_pbar: Union[bool] = True):
        """
        Propagate the scattering matrix through the provided potential.

        Parameters
        ----------
        potential : AbstractPotential object
            Scattering potential.
        max_batch : int, optional
            The probe batch size. Larger batches are faster, but require more memory. Default is None.
        multislice_pbar : bool, optional
            Display multislice progress bar. Default is True.
        plane_waves_pbar : bool, optional
            Display plane waves progress bar. Default is True.

        Returns
        -------
        Waves object.
            Probe exit wave functions for the provided positions.
        """

        if not self.periodic:
            self._raise_not_periodic()

        if isinstance(potential, Atoms):
            potential = Potential(potential)

        if not isinstance(max_batch, int):
            max_batch = self._max_batch_expansion()

        if isinstance(multislice_pbar, bool):
            multislice_pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not multislice_pbar)
            close_pbar = True
        else:
            close_pbar = False

        propagator = FresnelPropagator()

        for start, end, partial_s_matrix in self._generate_partial(max_batch, pbar=plane_waves_pbar):
            multislice(partial_s_matrix, potential, propagator=propagator, pbar=multislice_pbar)
            self._array[start: end] = copy_to_device(partial_s_matrix.array, get_array_module(self._array))

        self._antialiasing_aperture = (2 / 3.,) * 2

        if close_pbar:
            multislice_pbar.close()
        return self

    def _get_ctf_coefficients(self):
        xp = get_array_module(self._array)
        alpha = xp.sqrt(self.k[:, 0] ** 2 + self.k[:, 1] ** 2) * self.wavelength
        phi = xp.arctan2(self.k[:, 0], self.k[:, 1])
        return self._ctf.evaluate(alpha, phi)

    def _get_translation_coefficients(self, positions: Sequence[float]):
        xp = get_array_module_from_device(self.device)
        complex_exponential = get_device_function(xp, 'complex_exponential')
        positions = xp.asarray(positions)
        k = xp.asarray(self.k)
        return (complex_exponential(-2. * np.pi * k[:, 0][None] * positions[:, 0, None]) *
                complex_exponential(-2. * np.pi * k[:, 1][None] * positions[:, 1, None]))

    def _get_coefficients(self, positions: Sequence[float]):
        return self._get_translation_coefficients(positions) * self._get_ctf_coefficients()

    def _get_requisite_crop(self, positions: Sequence[float], return_per_position: bool = False):
        offset = (self.interpolated_gpts[0] // 2, self.interpolated_gpts[1] // 2)
        corners = np.rint(np.array(positions) / self.sampling - offset).astype(np.int)
        upper_corners = corners + np.asarray(self.interpolated_gpts)
        crop_corner = (np.min(corners[:, 0]).item(), np.min(corners[:, 1]).item())
        size = (np.max(upper_corners[:, 0]).item() - crop_corner[0],
                np.max(upper_corners[:, 1]).item() - crop_corner[1])

        if return_per_position:
            return crop_corner, size, corners
        else:
            return crop_corner, size

    def reduce(self, positions: Sequence[Sequence[float]] = None, max_batch_expansion: int = None) -> Waves:
        """
        Collapse the scattering matrix to probe wave functions centered on the provided positions.

        Parameters
        ----------
        positions : array of xy-positions
            The positions of the probe wave functions.
        max_batch_expansion : int, optional
            The maximum number of plane waves the reduction is applied to simultanously. If set to None, the number is
            chosen automatically based on available memory. Default is None.

        Returns
        -------
        Waves object
            Probe wave functions for the provided positions.
        """
        xp = get_array_module_from_device(self.device)
        batch_crop = get_device_function(xp, 'batch_crop')

        if max_batch_expansion is None:
            max_batch_expansion = self._max_batch_expansion()

        positions = self._validate_positions(positions)

        coefficients = self._get_coefficients(positions)

        if self.interpolated_gpts != self.gpts:
            crop_corner, size, corners = self._get_requisite_crop(positions, return_per_position=True)

            if self.offset is not None:
                corners -= self.offset
                crop_corner = (crop_corner[0] - self.offset[0], crop_corner[1] - self.offset[1])

            array = copy_to_device(periodic_crop(self.array, crop_corner, size), device=self._device)
            window = xp.tensordot(coefficients, array, axes=[(1,), (0,)])
            corners -= crop_corner
            window = batch_crop(window, corners, self.interpolated_gpts)

        elif max_batch_expansion <= len(self):
            window = xp.zeros((len(positions),) + self.gpts, dtype=xp.complex64)
            for start, end, partial_s_matrix in self._generate_partial(max_batch_expansion, pbar=False):
                partial_coefficients = coefficients[:, start:end]
                window += xp.tensordot(partial_coefficients, partial_s_matrix.array, axes=[(1,), (0,)])

        else:
            window = xp.tensordot(coefficients, copy_to_device(self.array, device=self._device), axes=[(1,), (0,)])

        return Waves(window, sampling=self.sampling, energy=self.energy, tilt=self.tilt,
                     antialias_aperture=self.antialias_aperture)

    def _generate_probes(self, scan: AbstractScan, max_batch_probes, max_batch_expansion):

        if not isinstance(max_batch_expansion, int):
            max_batch_expansion = self._max_batch_expansion()

        if not isinstance(max_batch_probes, int):
            max_batch_probes = self._max_batch_probes()

        for indices, positions in scan.generate_positions(max_batch=max_batch_probes):
            yield indices, self.collapse(positions, max_batch_expansion=max_batch_expansion)

    def scan(self,
             scan: AbstractScan,
             detectors: Sequence[AbstractDetector],
             measurements: Union[Measurement, Dict[AbstractDetector, Measurement]] = None,
             max_batch_probes: int = None,
             max_batch_expansion: int = None,
             pbar: Union[bool] = True):

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

        self.grid.check_is_defined()

        detectors = self._validate_detectors(detectors)
        measurements = self._validate_scan_measurements(detectors, scan, measurements)

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(scan), desc='Scan', disable=not pbar)

        for indices, exit_probes in self._generate_probes(scan, max_batch_probes, max_batch_expansion):
            for detector in detectors:
                new_measurement = detector.detect(exit_probes)
                scan.insert_new_measurement(measurements[detector], indices, new_measurement)

            pbar.update(len(indices))

        pbar.refresh()
        pbar.close()

        measurements = list(measurements.values())
        if len(measurements) == 1:
            return measurements[0]
        else:
            return measurements

    def transfer(self, device):
        return self.__class__(array=copy_to_device(self.array, device),
                              k=self.k.copy(),
                              ctf=self.ctf.copy(),
                              extent=self.extent,
                              offset=self.offset,
                              interpolated_gpts=self.interpolated_gpts,
                              energy=self.energy,
                              antialias_aperture=self.antialias_aperture,
                              device=self.device)

    def __copy__(self, device=None):
        return self.__class__(array=self.array.copy(),
                              k=self.k.copy(),
                              ctf=self.ctf.copy(),
                              extent=self.extent,
                              offset=self.offset,
                              interpolated_gpts=self.interpolated_gpts,
                              energy=self.energy,
                              antialias_aperture=self.antialias_aperture,
                              device=self.device)

    def copy(self):
        """Make a copy."""
        return copy(self)


class PartitionedSMatrix(AbstractScannedWaves):

    def __init__(self, parent_s_matrix, wave_vectors):
        self._parent_s_matrix = parent_s_matrix
        self._wave_vectors = wave_vectors

        self._grid = self._parent_s_matrix.grid
        self._ctf = self._parent_s_matrix.ctf
        self._accelerator = self._parent_s_matrix.accelerator
        self._antialias_aperture = self._parent_s_matrix.antialias_aperture
        self._device = self._parent_s_matrix._device

        self._event = Event()
        self._beamlet_weights_cache = Cache(1)
        self._beamlet_basis_cache = Cache(1)

        self._ctf_has_changed = Event()

        self._has_tilt = True
        self._accumulated_defocus = 0.

    @property
    def parent_wave_vectors(self):
        return self._parent_s_matrix.k

    @property
    def ctf(self):
        return self._ctf

    @property
    def wave_vectors(self):
        return self._wave_vectors

    @cached_method('_beamlet_weights_cache')
    def get_beamlet_weights(self):
        from scipy.spatial import Delaunay
        from abtem.waves.natural_neighbors import find_natural_neighbors, natural_neighbor_weights

        parent_wavevectors = self.parent_wave_vectors
        wave_vectors = self.wave_vectors

        n = len(parent_wavevectors)
        tri = Delaunay(parent_wavevectors)

        kx, ky = self.get_spatial_frequencies()
        kx, ky = np.meshgrid(kx, ky, indexing='ij')

        k = np.asarray((kx.ravel(), ky.ravel())).T

        weights = np.zeros((n,) + kx.shape)

        intersection = np.where(array_row_intersection(k, wave_vectors))[0]

        members, circumcenters = find_natural_neighbors(tri, k)

        for i in intersection:
            j, l = np.unravel_index(i, kx.shape)
            weights[:, j, l] = natural_neighbor_weights(parent_wavevectors, k[i], tri, members[i],
                                                        circumcenters)

        return weights

    def get_weights(self):
        from scipy.spatial import Delaunay
        from abtem.waves.natural_neighbors import find_natural_neighbors, natural_neighbor_weights

        parent_wavevectors = self.parent_wave_vectors
        n = len(parent_wavevectors)
        tri = Delaunay(parent_wavevectors)

        k = self.wave_vectors
        weights = np.zeros((n, len(k)))

        members, circumcenters = find_natural_neighbors(tri, k)
        for i, p in enumerate(k):
            weights[:, i] = natural_neighbor_weights(parent_wavevectors, p, tri, members[i], circumcenters)

        return weights

    def downsample(self, **kwargs):
        new_s_matrix_array = self._parent_s_matrix.downsample(**kwargs)
        return self.__class__(new_s_matrix_array, wave_vectors=self.wave_vectors)

    def _add_plane_wave_tilt(self):
        if self._has_tilt:
            return

    def _remove_plane_wave_tilt(self):
        if not self._has_tilt:
            return

        xp = get_array_module(self._parent_s_matrix.array)
        storage = get_device_from_array(self._parent_s_matrix.array)
        complex_exponential = get_device_function(xp, 'complex_exponential')

        x = xp.linspace(0, self.extent[0], self.gpts[0], endpoint=self.grid.endpoint[0], dtype=np.float32)
        y = xp.linspace(0, self.extent[1], self.gpts[1], endpoint=self.grid.endpoint[1], dtype=np.float32)

        array = self._parent_s_matrix.array
        k = self.parent_wave_vectors

        alpha = xp.sqrt(k[:, 0] ** 2 + k[:, 1] ** 2) * self.wavelength
        phi = xp.arctan2(k[:, 0], k[:, 1])

        ctf = CTF(defocus=self._accumulated_defocus, energy=self.energy)
        coeff = ctf.evaluate(alpha, phi)

        array *= coeff[:, None, None]

        for i in range(len(array)):
            array[i] *= copy_to_device(complex_exponential(-2 * np.pi * k[i, 0, None, None] * x[:, None]) *
                                       complex_exponential(-2 * np.pi * k[i, 1, None, None] * y[None, :]),
                                       storage)

        self._has_tilt = False

    def multislice(self, potential: AbstractPotential,
                   max_batch: int = None,
                   multislice_pbar: Union[bool] = True,
                   plane_waves_pbar: Union[bool] = True):

        if isinstance(potential, Atoms):
            potential = Potential(potential)

        self._add_plane_wave_tilt()

        self._accumulated_defocus += potential.thickness

        self._parent_s_matrix.multislice(potential, max_batch, multislice_pbar, plane_waves_pbar)
        return self

    def _fourier_translation_operator(self, positions):
        xp = get_array_module(positions)
        # positions /= xp.array(self.sampling)
        return fft2_shift_kernel(positions, self.gpts)

    @cached_method('_beamlet_basis_cache')
    def get_beamlet_basis(self):
        alpha, phi = self.get_scattering_angles()

        ctf = self._ctf.copy()
        ctf.defocus = ctf.defocus - self._accumulated_defocus
        # ctf = CTF(defocus=-self._accumulated_defocus, energy=self.energy, semiangle_cutoff=10)
        coeff = ctf.evaluate(alpha, phi)
        weights = self.get_beamlet_weights() * coeff
        return np.fft.ifft2(weights, axes=(1, 2))

    def _build_planewaves(self, k):
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        # xp = get_array_module_from_device(self._device)
        xp = np
        # storage_xp = get_array_module_from_device(self._storage)
        complex_exponential = get_device_function(xp, 'complex_exponential')

        x = xp.linspace(0, self.extent[0], self.gpts[0], endpoint=self.grid.endpoint[0], dtype=np.float32)
        y = xp.linspace(0, self.extent[1], self.gpts[1], endpoint=self.grid.endpoint[1], dtype=np.float32)

        shape = (len(k),) + self.gpts

        array = xp.zeros(shape, dtype=np.complex64)
        for i in range(len(k)):
            array[i] = (complex_exponential(2 * np.pi * k[i, 0, None, None] * x[:, None]) *
                        complex_exponential(2 * np.pi * k[i, 1, None, None] * y[None, :]))

        return array

    def interpolate_full(self):
        self._remove_plane_wave_tilt()

        plane_waves = self._build_planewaves(self.wave_vectors)
        weights = self.get_weights()

        for i, plane_wave in enumerate(plane_waves):
            plane_wave *= (self._parent_s_matrix.array * weights[:, i, None, None]).sum(0)

        alpha = np.sqrt(self.wave_vectors[:, 0] ** 2 + self.wave_vectors[:, 1] ** 2) * self.wavelength
        phi = np.arctan2(self.wave_vectors[:, 0], self.wave_vectors[:, 1])

        plane_waves *= CTF(defocus=-self._accumulated_defocus, energy=self.energy).evaluate(alpha, phi)[:, None, None]

        return SMatrixArray(plane_waves, energy=self.energy, k=self.wave_vectors, extent=self.extent,
                            interpolated_gpts=self.gpts, antialias_aperture=self._parent_s_matrix.antialias_aperture)

    def get_beamlets(self, positions, subpixel_shift=False):
        xp = get_array_module(positions)
        positions = positions / xp.array(self.sampling)

        if subpixel_shift:
            weights = self.get_beamlet_weights() * self._fourier_translation_operator(positions)
            return np.fft.ifft2(weights, axes=(1, 2))
        else:
            positions = np.round(positions).astype(np.int)
            weights = np.roll(self.get_beamlet_basis(), positions, axis=(1, 2))
            return weights

    def reduce(self, positions, subpixel_shift=False):
        self._remove_plane_wave_tilt()
        beamlets = self.get_beamlets(positions, subpixel_shift=subpixel_shift)
        array = np.einsum('ijk,ijk->jk', self._parent_s_matrix.array, beamlets)
        return Waves(array=array, extent=self.extent, energy=self.energy,
                     antialias_aperture=self._parent_s_matrix._antialias_aperture.antialias_aperture)

    def show_interpolation_weights(self, ax=None):
        from matplotlib.colors import to_rgb
        from abtem.measure.measure import fourier_space_offset
        weights = np.fft.fftshift(self.get_beamlet_weights(), axes=(1, 2))

        color_cycle = [['c', 'r'], ['m', 'g'], ['b', 'y']]
        colors = ['w']
        i = 1
        while True:
            colors += color_cycle[(i - 1) % 3] * (3 + (i - 1) * 3)
            i += 1
            if len(colors) >= len(weights):
                break

        colors = np.array([to_rgb(color) for color in colors])
        color_map = np.zeros(weights.shape[1:] + (3,))

        for i, color in enumerate(colors):
            color_map += weights[i, ..., None] * color[None, None]

        alpha, phi = self.get_scattering_angles()
        intensity = np.abs(self._ctf.evaluate(alpha, phi)) ** 2
        color_map *= np.fft.fftshift(intensity[..., None])

        if ax is None:
            fig, ax = plt.subplots()

        offsets = [fourier_space_offset(n, d) for n, d in zip(self.gpts, self.sampling)]

        extent = [offsets[0], offsets[0] + 1 / self.extent[0] * self.gpts[0] - 1 / self.extent[0],
                  offsets[1], offsets[1] + 1 / self.extent[1] * self.gpts[1] - 1 / self.extent[1]]
        extent = [l * 1000 * self.wavelength for l in extent]

        ax.imshow(color_map, extent=extent, origin='lower')
        ax.set_xlim([min(self.wave_vectors[:, 0]) * 1.1 * 1000 * self.wavelength,
                     max(self.wave_vectors[:, 0]) * 1.1 * 1000 * self.wavelength])
        ax.set_ylim([min(self.wave_vectors[:, 1]) * 1.1 * 1000 * self.wavelength,
                     max(self.wave_vectors[:, 1]) * 1.1 * 1000 * self.wavelength])

        ax.set_xlabel('alpha_x [mrad]')
        ax.set_ylabel('alpha_y [mrad]')
        return ax


def plane_waves(wave_vectors, extent, gpts):
    xp = get_array_module(wave_vectors)
    x = xp.linspace(0, extent[0], gpts[0], endpoint=False, dtype=np.float32)
    y = xp.linspace(0, extent[1], gpts[1], endpoint=False, dtype=np.float32)
    array = (complex_exponential(2 * np.pi * wave_vectors[:, 0, None, None] * x[:, None]) *
             complex_exponential(2 * np.pi * wave_vectors[:, 1, None, None] * y[None, :]))
    return array


class SMatrix(AbstractScannedWaves, HasEventMixin):
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
                 chunks=None,
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
        self._event = Event()

        self._device = device

        if storage is None:
            storage = device

        self._storage = storage
        self._chunks = chunks
        self._num_partitions = num_partitions

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

    @interpolation.setter
    def interpolation(self, value: int):
        self._interpolation = value

    @property
    def interpolated_gpts(self) -> Tuple[int, int]:
        return (self.gpts[0] // self.interpolation, self.gpts[1] // self.interpolation)

    @property
    def _interpolated_grid(self) -> Grid:
        """The grid of the interpolated probe wave functions."""
        interpolated_gpts = tuple(n // self.interpolation for n in self.gpts)
        return Grid(gpts=interpolated_gpts, sampling=self.sampling, lock_gpts=True)

    def equivalent_probe(self):
        return Probe(extent=self.extent, gpts=self.gpts, sampling=self.sampling, energy=self.energy, ctf=self.ctf,
                     device=self.device)

    def _generate_probes(self,
                         scan: AbstractScan,
                         potential: AbstractPotential,
                         max_batch_probes: int,
                         max_batch_expansion: int,
                         pbar: bool = True):

        potential_pbar = ProgressBar(total=len(potential), desc='Potential',
                                     disable=(not pbar) or (not potential._precalculate))

        multislice_pbar = ProgressBar(total=len(self), desc='Multislice', disable=not pbar)

        scan_bar = ProgressBar(total=len(scan), desc='Scan', disable=not pbar)

        tds_bar = ProgressBar(total=potential.num_frozen_phonon_configs, desc='TDS',
                              disable=(not pbar) or (potential.num_frozen_phonon_configs == 1))

        for potential_config in potential.generate_frozen_phonon_potentials(pbar=potential_pbar):
            scan_bar.reset()
            S = self.build()

            S = S.multislice(potential_config,
                             max_batch=max_batch_expansion,
                             multislice_pbar=False,
                             plane_waves_pbar=multislice_pbar)

            S = S.downsample('limit')

            for indices, exit_probes in S._generate_probes(scan, max_batch_probes, max_batch_expansion):
                yield indices, exit_probes
                scan_bar.update(len(indices))

            tds_bar.update(1)
            scan_bar.refresh()

        multislice_pbar.refresh()
        multislice_pbar.close()
        potential_pbar.refresh()
        potential_pbar.close()
        scan_bar.close()
        tds_bar.refresh()
        tds_bar.close()

    def multislice(self,
                   potential: AbstractPotential,
                   max_batch: int = None,
                   pbar: Union[bool] = True):
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

        if isinstance(potential, Atoms):
            potential = Potential(potential)

        self.grid.match(potential)

        return self.build().multislice(potential,
                                       max_batch=max_batch,
                                       multislice_pbar=pbar,
                                       plane_waves_pbar=pbar)

    def scan(self,
             scan: AbstractScan,
             detectors: Sequence[AbstractDetector],
             potential: Union[Atoms, AbstractPotential],
             measurements: Union[Measurement, Dict[AbstractDetector, Measurement]] = None,
             max_batch_probes: int = None,
             max_batch_expansion: int = None,
             pbar: bool = True) -> Union[Measurement, Sequence[Measurement]]:
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

        if isinstance(potential, Atoms):
            potential = Potential(potential)

        self.grid.match(potential.grid)
        self.grid.check_is_defined()

        detectors = self._validate_detectors(detectors)
        measurements = self._validate_scan_measurements(detectors, scan, measurements)

        probe_generator = self._generate_probes(scan,
                                                potential,
                                                max_batch_probes=max_batch_probes,
                                                max_batch_expansion=max_batch_expansion,
                                                pbar=pbar)

        for indices, exit_probes in probe_generator:
            for detector in detectors:
                new_measurement = detector.detect(exit_probes) / potential.num_frozen_phonon_configs
                scan.insert_new_measurement(measurements[detector], indices, new_measurement)

        measurements = list(measurements.values())
        if len(measurements) == 1:
            return measurements[0]
        else:
            return measurements

    @property
    def is_partial(self):
        return self._num_partitions is not None

    def __len__(self):
        if self.is_partial:
            return len(self.get_parent_wavevectors())
        else:
            return len(self.get_wavevectors())

    @property
    def k(self):
        return self.get_wavevectors()

    @property
    def wave_vectors(self):
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        xp = get_array_module_from_device(self._device)
        n_max = int(xp.ceil(self.expansion_cutoff / 1.e3 / (self.wavelength / self.extent[0] * self.interpolation)))
        m_max = int(xp.ceil(self.expansion_cutoff / 1.e3 / (self.wavelength / self.extent[1] * self.interpolation)))

        n = xp.arange(-n_max, n_max + 1, dtype=np.float32)
        w = xp.asarray(self.extent[0], dtype=np.float32)
        m = xp.arange(-m_max, m_max + 1, dtype=np.float32)
        h = xp.asarray(self.extent[1], dtype=np.float32)

        kx = n / w * np.float32(self.interpolation)
        ky = m / h * np.float32(self.interpolation)

        mask = kx[:, None] ** 2 + ky[None, :] ** 2 < (self.expansion_cutoff / 1.e3 / self.wavelength) ** 2
        kx, ky = xp.meshgrid(kx, ky, indexing='ij')
        kx = kx[mask]
        ky = ky[mask]
        return xp.asarray((kx, ky)).T

    def get_parent_wavevectors(self):
        rings = [np.array((0., 0.))]
        n = 6
        if self._num_partitions == 1:
            raise NotImplementedError()

        for r in np.linspace(self.expansion_cutoff / (self._num_partitions - 1), self.expansion_cutoff,
                             self._num_partitions - 1):
            angles = np.arange(n, dtype=np.int32) * 2 * np.pi / n + np.pi / 2
            kx = np.round(r * np.sin(angles) / 1000. / self.wavelength * self.extent[0]) / self.extent[0]
            ky = np.round(r * np.cos(-angles) / 1000. / self.wavelength * self.extent[1]) / self.extent[1]
            n += 6
            rings.append(np.array([kx, ky]).T)

        return np.vstack(rings)

    def _build_partial(self):
        k_parent = self.get_parent_wavevectors()
        array = self._build_planewaves(k_parent)

        parent_s_matrix = SMatrixArray(array,
                                       interpolated_gpts=self.interpolated_gpts,
                                       extent=self.extent,
                                       energy=self.energy,
                                       tilt=self.tilt,
                                       k=k_parent,
                                       ctf=self.ctf.copy(),
                                       antialias_aperture=self.antialias_aperture,
                                       device=self._device)
        return PartitionedSMatrix(parent_s_matrix, wave_vectors=self.get_wavevectors())

    def _build_convential(self):
        k = da.from_array(self.wave_vectors, chunks=(self.chunks, -1))

        def _build_s_matrix(k, extent, gpts, interpolation):
            array = plane_waves(k, extent, gpts)
            xp = get_array_module(array)

            interpolated_gpts = (gpts[0] // interpolation, self.gpts[1] // interpolation)

            #probe = (xp.abs(array.sum(0)) ** 2)[:interpolated_gpts[0], :interpolated_gpts[1]]
            #array /= xp.sqrt(probe.sum()) * xp.sqrt(interpolated_gpts[0] * interpolated_gpts[1])

            return array

        array = k.map_blocks(_build_s_matrix, extent=self.extent, gpts=self.gpts, interpolation=self.interpolation,
                             drop_axis=1, new_axis=(1, 2),
                             chunks=k.chunks[:-1] + ((self.gpts[0],), (self.gpts[1],)), dtype=np.complex64)

        return SMatrixArray(array,
                            interpolated_gpts=self.interpolated_gpts,
                            extent=self.extent,
                            energy=self.energy,
                            tilt=self.tilt,
                            k=k,
                            ctf=self.ctf.copy(),
                            antialias_aperture=self.antialias_aperture,
                            device=self._device)

    def build(self) -> Union[SMatrixArray, PartitionedSMatrix]:
        """Build the scattering matrix."""

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        if self._num_partitions is None:
            return self._build_convential()
        else:
            return self._build_partial()

    def profile(self, angle=0.) -> Measurement:
        measurement = self.build().collapse((self.extent[0] / 2, self.extent[1] / 2)).intensity()
        return probe_profile(measurement, angle=angle)

    def interact(self, sliders=None, profile: bool = False, throttling: float = 0.01):
        from abtem.visualize.widgets import quick_sliders, throttle
        from abtem.visualize.interactive.apps import MeasurementView1d, MeasurementView2d
        import ipywidgets as widgets

        if profile:
            view = MeasurementView1d()

            def callback(*args):
                view.measurement = self.profile()
        else:
            view = MeasurementView2d()

            def callback(*args):
                view.measurement = self.build().collapse().intensity()[0]

        if throttling:
            callback = throttle(throttling)(callback)

        self.observe(callback)
        callback()

        if sliders:
            sliders = quick_sliders(self.ctf, **sliders)
            return widgets.HBox([view.figure, widgets.VBox(sliders)])
        else:
            return view.figure

    def show(self, **kwargs):
        """
        Show the probe wave function.

        Parameters
        ----------
        angle : float, optional
            Angle along which the profile is shown [deg]. Default is 0 degrees.
        kwargs : Additional keyword arguments for the abtem.plot.show_image function.
        """
        return self.build().collapse((self.extent[0] / 2, self.extent[1] / 2)).intensity().show(**kwargs)

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
