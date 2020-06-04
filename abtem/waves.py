from typing import Union, Sequence, Tuple, Type, Generic
from abc import ABC, abstractmethod
import h5py
import numpy as np
import pyfftw as fftw
from ase import Atoms
import cupy as cp

from functools import lru_cache
from copy import copy
from abtem.analyse import fwhm
from abtem.bases import Grid, Accelerator, HasGridMixin, HasAcceleratorMixin, cache_clear_callback, Event, \
    watched_property, DeviceManager
from abtem.config import DTYPE, COMPLEX_DTYPE, FFTW_THREADS
from abtem.detect import DetectorBase
from abtem.potentials import Potential, PotentialBase
from abtem.prism import window_and_collapse
from abtem.scan import GridScan, LineScan, CustomScan, ScanBase
from abtem.transfer import CTF
from abtem.utils import complex_exponential, abs2, BatchGenerator
from tqdm.auto import tqdm
from contextlib import ExitStack


class ArrayWaves(HasGridMixin, HasAcceleratorMixin):
    """
    Waves object.

    The waves object can define a stack of n arbitrary 2d wavefunctions of shape (w, h) defined by a (n, w, h) complex
    numpy array.

    Parameters
    ----------
    array : complex ndarray of shape (n, gpts_x, gpts_y)
        Stack of n complex wave functions
    extent : sequence of float, float, optional
        Lateral extent of wave functions [Å]
    sampling : sequence of float, float, optional
        Lateral sampling of wave functions [1 / Å]
    energy : float, optional
        Wave function energy [eV]
    """

    def __init__(self, array: np.ndarray, extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None, energy: float = None, device=None):

        self._array = array
        self.grid = Grid(extent=extent, sampling=sampling, lock_gpts=True)
        self.accelerator = Accelerator(energy=energy)
        self.device_manager = DeviceManager(device)
        self.grid.changed.register(cache_clear_callback(self.fresnel))
        self.accelerator.changed.register(cache_clear_callback(self.fresnel))

    @property
    def array(self):
        return self._array

    @lru_cache(1)
    def fresnel(self, dz):
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        xp = self.device_manager.get_array_library()
        kx = xp.fft.fftfreq(self.gpts[0], self.sampling[0]).astype(DTYPE)
        ky = xp.fft.fftfreq(self.gpts[1], self.sampling[1]).astype(DTYPE)
        k = (kx ** 2)[:, None] + (ky ** 2)[None]
        return complex_exponential(-k * np.pi * self.wavelength * dz)[None]

    def propagate(self, dz):
        return self.fft_convolve(self.fresnel(dz))

    def fft_convolve(self, kernel):
        if self.device_manager.device == 'cpu':
            self._fft_convolve_fftw(self._array, kernel)
        else:
            self._fft_convolve_gpu(self._array, kernel)

    def _fft_convolve_gpu(self, array, kernel):
        array[:] = cp.fft.ifft2(cp.fft.fft2(array) * kernel)

    @lru_cache(1)
    def _create_fftw_objects(self, array):
        target_array = fftw.empty_aligned(array.shape, dtype=array.dtype)
        fftw_forward = fftw.FFTW(target_array, target_array, axes=(-1, -2), threads=FFTW_THREADS,
                                 flags=('FFTW_ESTIMATE',))
        fftw_backward = fftw.FFTW(target_array, target_array, axes=(-1, -2), direction='FFTW_BACKWARD',
                                  threads=FFTW_THREADS, flags=('FFTW_ESTIMATE',))
        return fftw_forward, fftw_backward, target_array

    def _fft_convolve_fftw(self, array, kernel):
        fftw_forward, fftw_backward, target_array = self._create_fftw_objects(array)
        target_array[:] = array
        fftw_forward()
        array *= kernel
        fftw_backward()

    def transmit(self, potential_slice):
        self._array *= complex_exponential(self.accelerator.sigma * potential_slice.array)

    def apply_ctf(self, ctf=None, **kwargs):
        """
        Apply the aberrations defined by a CTF object to wave function.

        Parameters
        ----------
        ctf : CTF object
            Contrast Transfer Function object to be applied
        in_place : bool
            If true modify the array representing the wave in place, otherwise create a copy.
        kwargs :

        Returns
        -------
        Waves
            The wave functions with aberrations applied.
        """

        if ctf is None:
            ctf = CTF(**kwargs)

        ctf.accelerator.match(self.accelerator)

        return self.fft_convolve(ctf.evaluate_on_grid(self.grid))

    def multislice(self, potential, show_progress: bool = True):
        """
        Propagate the wave function through a potential using the multislice

        Parameters
        ----------
        potential : Potential object or Atoms object
            The potential to propaget the waves through.
        in_place : bool
            Modify the wavefunction arrays in place.
        show_progress : bool
            If true create a progress bar.

        Returns
        -------
        Waves object
            Wave functions after multislice propagation through the potential.

        """
        for potential_slice in potential:
            self.transmit(potential_slice)
            self.propagate(potential_slice.thickness)

    def write(self, path, overwrite=True) -> None:
        """
        Write Waves object to file.

        Parameters
        ----------
        path : str
            Path of the file to write to.

        Returns
        -------
        None
        """

        with h5py.File(path, 'w') as f:
            f.create_dataset('array', data=self.array)
            f.create_dataset('energy', data=self.energy)
            f.create_dataset('extent', data=self.extent)

    def __getitem__(self, item):
        if len(self.array.shape) <= self.grid.dimensions:
            raise RuntimeError()
        return self.__class__(array=self._array[item], extent=self.extent.copy(), energy=self.energy)

    def __copy__(self):
        """
        Return a copy.

        Parameters
        ----------
        copy_array : bool
            If true copy the underlying numpy array.

        Returns
        -------
        Waves object
            A copy of itself.
        """
        new_copy = self.__class__(array=self._array.copy())
        new_copy.grid = copy(self.grid)
        new_copy.accelerator = copy(self.accelerator)
        new_copy.device_manager = copy(self.accelerator)
        return new_copy


class PlaneWaves(HasGridMixin, HasAcceleratorMixin):
    """
    Plane waves object

    The plane waves object can represent a stack of plane waves.

    Parameters
    ----------
    num_waves : int
        Number of plane waves in stack
    extent : sequence of float, float, optional
        Lateral extent of wave functions [Å]
    gpts : sequence of int, int, optional
        Number of grid points describing the wave functions
    sampling : sequence of float, float, optional
        Lateral sampling of wave functions [1 / Å]
    energy : float, optional
        Energy of electrons represented by wave functions [eV]
    """

    def __init__(self, num_waves=1, extent=None, gpts=None, sampling=None, energy=None):
        self._num_waves = num_waves

        self.grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self.accelerator = Accelerator(energy=energy)

        self.changed = Event()
        self.changed.register(cache_clear_callback(self.as_array_waves))

    @property
    def num_waves(self) -> int:
        return self._num_waves

    @num_waves.setter
    @watched_property('changed')
    def num_waves(self, value: int):
        self._num_waves = value

    def multislice(self, potential, show_progress=True):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)
        potential.match_grid(self)
        return self.as_array_waves().multislice(potential, show_progress=show_progress)

    def transmit(self, potential_slice):
        self.grid.match(potential_slice)
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        return self.as_array_waves().transmit(potential_slice)

    def propagate(self, dz):
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        return self.as_array_waves().propagate(dz)

    @lru_cache(1)
    def as_array_waves(self):
        self.grid.check_is_defined()
        array = np.ones((self.num_waves, self.gpts[0], self.gpts[1]), dtype=COMPLEX_DTYPE)
        return ArrayWaves(array, extent=self.extent, energy=self.energy)

    def copy(self):
        return self.__class__(num_waves=self.num_waves, extent=self.extent, gpts=self.gpts, sampling=self.sampling,
                              energy=self.energy)


class ProbeWaves(HasGridMixin, HasAcceleratorMixin):
    """
    Probe waves object

    The probe waves object can represent a stack of electron probe wave function for simulating scanning transmission
    electron microscopy.

    Parameters
    ----------
    semiangle_cutoff : float
        Convergence semi-angle [rad.].
    rolloff : float
        Softens the cutoff. A value of 0 gives a hard cutoff, while 1 gives the softest possible cutoff.
    focal_spread : float
        The focal spread due to, among other factors, chromatic aberrations and lens current instabilities.
    parameters : dict
        The parameters describing the phase aberrations using polar notation or the alias. See the documentation for the
        CTF object for a more detailed description. Convert from cartesian to polar parameters using
        ´utils.cartesian2polar´.
    normalize : bool
        If true normalize the absolute square of probe array.
    extent : sequence of float, float, optional
        Lateral extent of wavefunctions [Å].
    gpts : sequence of int, int, optional
        Number of grid points describing the wavefunctions
    sampling : sequence of float, float, optional
        Lateral sampling of wavefunctions [1 / Å].
    energy : float, optional
        Waves energy [eV].
    **kwargs
        Provide the aberration coefficients as keyword arguments.

    """

    def __init__(self, semiangle_cutoff: float = np.inf, rolloff: float = 0., focal_spread: float = 0.,
                 angular_spread: float = 0., ctf_parameters: dict = None,
                 normalize: bool = False,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 device=None,
                 **kwargs):

        self._normalize = normalize
        self.changed = Event()
        self.grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self.accelerator = Accelerator(energy=energy)
        self.device_manager = DeviceManager(device)
        self.ctf = CTF(semiangle_cutoff=semiangle_cutoff, rolloff=rolloff, focal_spread=focal_spread,
                       angular_spread=angular_spread, parameters=ctf_parameters, extent=extent, sampling=sampling,
                       energy=energy, **kwargs)

        self.changed = Event()
        self.changed.register(cache_clear_callback(self.as_array_waves))

    @property
    def normalize(self) -> bool:
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool):
        self._normalize = value

    def get_fwhm(self):
        return fwhm(self)

    def _fourier_translation_operator(self, positions):
        xp = self.device_manager.get_array_library()

        kx, ky = self.grid.spatial_frequencies()
        kx = kx.reshape((1, -1, 1))
        ky = ky.reshape((1, 1, -1))

        kx = xp.asarray(kx)
        ky = xp.asarray(ky)
        positions = xp.asarray(positions)

        x = positions[:, 0].reshape((-1,) + (1, 1))
        y = positions[:, 1].reshape((-1,) + (1, 1))
        return complex_exponential(2 * np.pi * kx * x) * complex_exponential(2 * np.pi * ky * y)

    def as_array_waves(self, positions: Sequence[Sequence[float]] = None) -> ArrayWaves:
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        if positions is None:
            positions = np.zeros((1, 2), dtype=DTYPE)
        else:
            positions = np.array(positions, dtype=DTYPE)

        if len(positions.shape) == 1:
            positions = np.expand_dims(positions, axis=0)

        xp = self.device_manager.get_array_library()
        array = self.ctf.evaluate_on_grid(self.grid) * self._fourier_translation_operator(positions)

        if self.device_manager.is_cuda:
            array[:] = xp.fft.fft2(array)
        else:
            fftw.FFTW(array, array, axes=(1, 2), threads=FFTW_THREADS, flags=('FFTW_ESTIMATE',))()

        if self.normalize:
            array[:] = array / np.sum(xp.abs(array) ** 2, axis=(1, 2), keepdims=True) * xp.prod(array.shape[1:])

        return ArrayWaves(array, extent=self.extent, energy=self.energy)

    def multislice(self, positions, potential, show_progress=False):
        self.grid.match(potential)
        return self.as_array_waves(positions).multislice(potential, show_progress=show_progress)

    def generate_probes(self, scan: Union[LineScan, GridScan], potential: Union[Type[PotentialBase], Atoms],
                        max_batch='50%'):

        for start, end, positions in scan.generate_positions(max_batch=max_batch):
            yield start, end, self.multislice(positions, potential)

    def scan(self, scan, potential, detectors, max_batch, show_progress=False):
        measurements = scan.allocate_measurements(detectors)

        with tqdm(total=len(scan)) if show_progress else ExitStack() as pbar:
            for start, end, exit_probes in self.generate_probes(scan, potential, max_batch):

                for detector, measurement in measurements.items():
                    scan.insert_new_measurement(measurement, start, end, detector.detect(exit_probes))

                if show_progress:
                    pbar.update(end - start)

        return measurements
#
#
# def _prism_translate(positions, kx, ky):
#     """
#     Create Fourier space translation multiplier.
#
#     Parameters
#     ----------
#     positions : Nx2 numpy array
#     kx :
#     ky :
#
#     Returns
#     -------
#
#     """
#     array = (complex_exponential(2 * np.pi * kx[None] * positions[:, 0, None]) *
#              complex_exponential(2 * np.pi * ky[None] * positions[:, 1, None]))
#     return array


# class ScatteringMatrix(Waves, CTFBase):
#     """
#     Scattering matrix object
#
#     The scattering matrix object represents a plane wave expansion of a scanning transmission electron microscopy probe.
#
#     Parameters
#     ----------
#     array : 3d numpy array
#         The array representation of the scattering matrix.
#     interpolation : int
#
#     expansion_cutoff : float
#         The angular cutoff of the plane wave expansion.
#     kx : sequence of floats
#         The
#     ky : sequence of floats
#     extent : two floats, float, optional
#         Lateral extent of the scattering matrix, if the unit cell of the atoms is too small it will be repeated. Units of Angstrom.
#     sampling : two floats, float, optional
#         Lateral sampling of the scattering matrix. Units of 1 / Angstrom.
#     energy :
#     always_recenter :
#     """
#
#     def __init__(self, array: np.ndarray, interpolation: int, expansion_cutoff: float, kx: np.ndarray, ky: np.ndarray,
#                  extent: Union[float, Sequence[float]] = None,
#                  sampling: Union[float, Sequence[float]] = None,
#                  energy: float = None, always_recenter: bool = False):
#
#         self._interpolation = interpolation
#         self._expansion_cutoff = expansion_cutoff
#         self._kx = kx
#         self._ky = ky
#         self.always_recenter = always_recenter
#
#         super().__init__(array=array, spatial_dimensions=2, extent=extent, sampling=sampling, energy=energy)
#
#     @property
#     def kx(self) -> np.ndarray:
#         return self._kx
#
#     @property
#     def ky(self) -> np.ndarray:
#         return self._ky
#
#     @property
#     def interpolation(self) -> int:
#         return self._interpolation
#
#     @property
#     def probe_extent(self):
#         return self.probe_shape * self.sampling
#
#     @property
#     def probe_shape(self):
#         return np.array((self.gpts[0] // self.interpolation, self.gpts[1] // self.interpolation))
#
#     def build_at(self, positions: Sequence[Sequence[float]]) -> Waves:
#         xp = cp.get_array_module(self.array)
#
#         coefficients = super().get_array()[0] * _prism_translate(positions, self.kx, self.ky)
#
#         if (self.interpolation > 1) | self.always_recenter:
#             window_shape = (len(positions), self.gpts[0] // self.interpolation, self.gpts[1] // self.interpolation)
#             window = xp.zeros(window_shape, dtype=np.complex)
#
#             corners = xp.round(positions / self.sampling - np.floor_divide(window.shape[1:], 2)).astype(np.int)
#             corners = xp.remainder(corners, self.gpts)
#
#             window_and_collapse(self.array, window, corners, coefficients)
#         else:
#             window = (self.array[None] * coefficients[:, :, None, None]).sum(1)
#
#         return Waves(window, extent=self.extent, energy=self.energy)
#
#     def _get_scan_waves_maker(self):
#         def scan_waves_func(waves, positions):
#             waves = waves.build_at(positions)
#             return waves
#
#         return scan_waves_func
#
#     def build(self):
#         return Waves(np.fft.fftshift(self.array.sum(0)), extent=self.extent, energy=self.energy)
#
#     @cached_method(('extent', 'gpts', 'sampling', 'energy'))
#     def get_alpha(self):
#         return np.sqrt(self._kx ** 2 + self._ky ** 2) * self.wavelength
#
#     @cached_method(('extent', 'gpts', 'sampling', 'energy'))
#     def get_phi(self):
#         return np.arctan2(self._kx, self._ky)
#
#
# class PrismWaves(Grid, Energy, Buildable, Cache):
#
#     def __init__(self, expansion_cutoff: float, interpolation: int = 1,
#                  extent: Union[float, Sequence[float]] = None,
#                  gpts: Union[int, Sequence[int]] = None,
#                  sampling: Union[float, Sequence[float]] = None,
#                  energy: float = None, always_recenter: bool = False,
#                  build_on_gpu=True):
#
#         """
#
#         Parameters
#         ----------
#         expansion_cutoff :
#         interpolation :
#         extent :
#         gpts :
#         sampling :
#         energy :
#         always_recenter :
#         """
#
#         if not isinstance(interpolation, int):
#             raise ValueError('interpolation factor must be int')
#
#         self._interpolation = interpolation
#         self._expansion_cutoff = expansion_cutoff
#         self.always_recenter = always_recenter
#
#         super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy, build_on_gpu=build_on_gpu)
#
#     @property
#     def expansion_cutoff(self) -> float:
#         return self._expansion_cutoff
#
#     @expansion_cutoff.setter
#     @notify
#     def expansion_cutoff(self, value: float):
#         self._expansion_cutoff = value
#
#     @property
#     def interpolation(self) -> int:
#         return self._interpolation
#
#     @interpolation.setter
#     @notify
#     def interpolation(self, value: int):
#         self._interpolation = value
#
#     @cached_method()
#     def get_spatial_frequencies(self) -> Tuple[np.ndarray, np.ndarray]:
#         self.check_is_grid_defined()
#         self.check_is_energy_defined()
#
#         xp = self._array_module()
#
#         n_max = int(np.ceil(self.expansion_cutoff / (self.wavelength / self.extent[0] * self.interpolation)))
#         m_max = int(np.ceil(self.expansion_cutoff / (self.wavelength / self.extent[1] * self.interpolation)))
#
#         kx = xp.arange(-n_max, n_max + 1) / xp.asarray(self.extent[0]) * self.interpolation
#         ky = xp.arange(-m_max, m_max + 1) / xp.asarray(self.extent[1]) * self.interpolation
#
#         mask = kx[:, None] ** 2 + ky[None, :] ** 2 < (self.expansion_cutoff / self.wavelength) ** 2
#         kx, ky = xp.meshgrid(kx, ky, indexing='ij')
#
#         return kx[mask], ky[mask]
#
#     def multislice(self, potential, show_progress=True) -> ScatteringMatrix:
#
#         if isinstance(potential, Atoms):
#             potential = Potential(atoms=potential)
#
#         if self.extent is None:
#             self.extent = potential.extent
#
#         if self.gpts is None:
#             self.gpts = potential.gpts
#
#         return self.build().multislice(potential, in_place=True, show_progress=show_progress)
#
#     def build(self) -> ScatteringMatrix:
#         self.check_is_grid_defined()
#         self.check_is_energy_defined()
#
#         xp = self._array_module()
#         kx, ky = self.get_spatial_frequencies()
#
#         x = xp.linspace(0, self.extent[0], self.gpts[0], endpoint=self.endpoint)
#         y = xp.linspace(0, self.extent[1], self.gpts[1], endpoint=self.endpoint)
#
#         array = (complex_exponential(-2 * np.pi * kx[:, None, None] * x[None, :, None]) *
#                  complex_exponential(-2 * np.pi * ky[:, None, None] * y[None, None, :]))
#
#         return ScatteringMatrix(array,
#                                 interpolation=self.interpolation, expansion_cutoff=self.expansion_cutoff,
#                                 extent=self.extent, energy=self.energy, kx=kx, ky=ky,
#                                 always_recenter=self.always_recenter)
