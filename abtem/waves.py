from __future__ import annotations

from contextlib import ExitStack
from copy import copy
from typing import Union, Sequence

import h5py
import numpy as np
from ase import Atoms
from tqdm.auto import tqdm

from abtem.bases import Grid, Accelerator, HasGridMixin, HasAcceleratorMixin, cache_clear_callback, Event, \
    watched_method, Cache, cached_method
from abtem.config import DTYPE
from abtem.detect import AbstractDetector
from abtem.device import get_array_module, get_device_function, asnumpy
from abtem.measure import calibrations_from_grid, Measurement
from abtem.plot import show_image
from abtem.potentials import Potential, AbstractPotential
from abtem.scan import AbstractScan
from abtem.transfer import CTF
from abtem.utils import polargrid


class FresnelPropagator:

    def __init__(self):
        self.cache = Cache(1)  # TODO : lru

    @cached_method('cache')
    def get_array(self, waves, dz):
        xp = get_array_module(waves.array)
        complex_exponential = get_device_function(xp, 'complex_exponential')

        kx = xp.fft.fftfreq(waves.grid.gpts[0], waves.grid.sampling[0]).astype(DTYPE)
        ky = xp.fft.fftfreq(waves.grid.gpts[1], waves.grid.sampling[1]).astype(DTYPE)
        f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * waves.wavelength * dz) *
             complex_exponential(-(ky ** 2)[None] * np.pi * waves.wavelength * dz))
        return f

    def propagate(self, waves, dz):
        fft2_convolve = get_device_function(get_array_module(waves.array), 'fft2_convolve')
        return fft2_convolve(waves._array, self.get_array(waves, dz))


def multislice(waves: Union[Waves, SMatrix], potential: AbstractPotential, show_progress: bool = True):
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
    waves.grid.match(potential)
    waves.accelerator.check_is_defined()
    waves.grid.check_is_defined()
    xp = get_array_module(waves.array)
    complex_exponential = get_device_function(xp, 'complex_exponential')

    propagator = FresnelPropagator()
    with tqdm(total=len(potential)) if show_progress else ExitStack() as pbar:
        for potential_slice, dz in potential:
            if np.iscomplexobj(potential_slice):
                waves._array *= potential_slice[None]
            else:
                waves._array *= complex_exponential(waves.accelerator.sigma * potential_slice[None])

            propagator.propagate(waves, dz)

            if show_progress:
                pbar.update(1)
    return waves


def _scan(scanned, scan, detectors, potential=None, max_batch=1, show_progress=True):
    try:
        probe_generator = scanned.generate_probes(scan, potential=potential, max_batch=max_batch)
    except TypeError:
        probe_generator = scanned.generate_probes(scan, max_batch=max_batch)

    for detector in detectors:
        if hasattr(detector, 'adapt_to_waves'):
            detector.adapt_to_waves(scanned)

    measurements = scan.allocate_measurements(detectors)
    with tqdm(total=len(scan)) if show_progress else ExitStack() as pbar:
        for start, end, exit_probes in probe_generator:
            for detector, measurement in measurements.items():
                scan.insert_new_measurement(measurement, start, end, detector.detect(exit_probes))

            if show_progress:
                pbar.update(end - start)

    return measurements


class Waves(HasGridMixin, HasAcceleratorMixin):
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
                 sampling: Union[float, Sequence[float]] = None, energy: float = None):

        if len(array.shape) == 2:
            array = array[None]

        self._array = array
        self._grid = Grid(extent=extent, gpts=array.shape[1:], sampling=sampling, lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)

    @property
    def array(self):
        return self._array

    @property
    def intensity(self):
        calibrations = [None] + calibrations_from_grid(self.grid, ['x', 'y'])
        abs2 = get_device_function(get_array_module(self.array), 'abs2')
        return Measurement(abs2(self.array), calibrations)

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
        xp = get_array_module(self.array)
        fft2_convolve = get_device_function(get_array_module(self.array), 'fft2_convolve')

        if ctf is None:
            ctf = CTF(**kwargs)

        ctf.accelerator.match(self.accelerator)
        kx, ky = self.grid.spatial_frequencies()
        alpha, phi = polargrid(xp.asarray(kx * self.wavelength), xp.asarray(ky * self.wavelength))
        kernel = ctf.evaluate(alpha, phi)
        return self.__class__(fft2_convolve(self.array, kernel, overwrite_x=False),
                              extent=self.extent.copy(),
                              energy=self.energy)

    def multislice(self, potential: AbstractPotential, show_progress: bool = True):
        return multislice(self, potential, show_progress)

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

    @classmethod
    def read(cls, path):
        with h5py.File(path, 'r') as f:
            datasets = {}
            for key in f.keys():
                datasets[key] = f.get(key)[()]

        return cls(array=datasets['array'], extent=datasets['extent'], energy=datasets['energy'])

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
        new_copy._grid = copy(self.grid)
        new_copy._accelerator = copy(self.accelerator)
        return new_copy

    def show(self, **kwargs):
        xp = get_array_module(self.array)
        array = xp.squeeze(self.array)
        array = asnumpy(array)
        return show_image(array, calibrations_from_grid(self.grid, names=['x', 'y']), **kwargs)


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

    def __init__(self, num_waves: int = 1, extent=None, gpts=None, sampling=None, energy=None):
        self._num_waves = num_waves
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)

    @property
    def num_waves(self) -> int:
        return self._num_waves

    @num_waves.setter
    def num_waves(self, value: int):
        self._num_waves = value

    def multislice(self, potential, show_progress=True):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)
        potential.grid.match(self)
        return self.build().multislice(potential, show_progress=show_progress)

    def build(self, xp=np):
        self.grid.check_is_defined()
        array = xp.ones((self.num_waves, self.gpts[0], self.gpts[1]), dtype=xp.float32)
        return Waves(array, extent=self.extent, energy=self.energy)

    def copy(self):
        return self.__class__(num_waves=self.num_waves, extent=self.extent, gpts=self.gpts, sampling=self.sampling,
                              energy=self.energy)


class Probe(HasGridMixin, HasAcceleratorMixin):
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
                 **kwargs):

        self._normalize = normalize

        self._ctf = CTF(semiangle_cutoff=semiangle_cutoff, rolloff=rolloff, focal_spread=focal_spread,
                        angular_spread=angular_spread, parameters=ctf_parameters, energy=energy, **kwargs)

        self._accelerator = self._ctf._accelerator
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self.cache = Cache(1)

    @property
    def normalize(self) -> bool:
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool):
        self._normalize = value

    def get_fwhm(self):
        array = self.build(self.extent / 2).array
        y = array[0, array.shape[1] // 2]
        peak_idx = np.argmax(y)
        peak_value = y[peak_idx]
        left = np.argmin(np.abs(y[:peak_idx] - peak_value / 2))
        right = peak_idx + np.argmin(np.abs(y[peak_idx:] - peak_value / 2))
        return (right - left) * self.sampling[0]

    def _fourier_translation_operator(self, positions):
        xp = get_array_module(positions)
        complex_exponential = get_device_function(xp, 'complex_exponential')

        kx, ky = self.grid.spatial_frequencies()
        kx = kx.reshape((1, -1, 1))
        ky = ky.reshape((1, 1, -1))
        kx = xp.asarray(kx)
        ky = xp.asarray(ky)
        positions = xp.asarray(positions)
        x = positions[:, 0].reshape((-1,) + (1, 1))
        y = positions[:, 1].reshape((-1,) + (1, 1))

        return complex_exponential(2 * np.pi * kx * x) * complex_exponential(2 * np.pi * ky * y)

    @cached_method('cache')
    def _evaluate_ctf(self, xp):
        kx, ky = self.grid.spatial_frequencies()
        alpha, phi = polargrid(xp.asarray(kx * self.wavelength), xp.asarray(ky * self.wavelength))
        return self._ctf.evaluate(alpha, phi)

    def build(self, positions: Sequence[Sequence[float]] = None) -> Waves:
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        xp = get_array_module(positions)
        fft2 = get_device_function(xp, 'fft2')

        if positions is None:
            positions = xp.zeros((1, 2), dtype=xp.float32)
        else:
            positions = xp.array(positions, dtype=xp.float32)

        if len(positions.shape) == 1:
            positions = xp.expand_dims(positions, axis=0)

        array = self._evaluate_ctf(xp) * self._fourier_translation_operator(positions)

        fft2(array, overwrite_x=True)

        if self.normalize:
            array[:] = array / np.sum(xp.abs(array) ** 2, axis=(1, 2), keepdims=True) * xp.prod(array.shape[1:])

        return Waves(array, extent=self.extent, energy=self.energy)

    def multislice(self, positions: Sequence[Sequence[float]], potential: AbstractPotential,
                   show_progress=False) -> Waves:
        self.grid.match(potential)
        return self.build(positions).multislice(potential, show_progress=show_progress)

    def generate_probes(self, scan: AbstractScan, potential: Union[AbstractPotential, Atoms], max_batch: int,
                        show_progress=False):
        with tqdm(total=len(scan)) if show_progress else ExitStack() as pbar:
            for start, end, positions in scan.generate_positions(max_batch=max_batch):
                if show_progress:
                    pbar.update(end - start)

                yield start, end, self.multislice(positions, potential)

    def scan(self, scan: AbstractScan, detectors: Sequence[AbstractDetector],
             potential: Union[Atoms, AbstractPotential], max_batch=1, show_progress=True):
        self.grid.match(potential)
        return _scan(self, scan, detectors, potential, max_batch, show_progress)

    def show(self, **kwargs):
        array = self.build(self.extent / 2).array[0]
        abs2 = get_device_function(get_array_module(array), 'abs2')
        array = asnumpy(abs2(array))
        return show_image(array, calibrations_from_grid(self.grid, names=['x', 'y']), **kwargs)


class SMatrix(HasGridMixin, HasAcceleratorMixin):
    """
    Scattering matrix object

    The scattering matrix object represents a plane wave expansion of a scanning transmission electron microscopy probe.

    Parameters
    ----------
    array : 3d numpy array
        The array representation of the scattering matrix.
    interpolation : int

    expansion_cutoff : float
        The angular cutoff of the plane wave expansion.
    kx : sequence of floats
        The
    ky : sequence of floats
    extent : two floats, float, optional
        Lateral extent of the scattering matrix, if the unit cell of the atoms is too small it will be repeated. Units of Angstrom.
    sampling : two floats, float, optional
        Lateral sampling of the scattering matrix. Units of 1 / Angstrom.
    energy :
    always_recenter :
    """

    def __init__(self,
                 array: np.ndarray,
                 interpolation: int,
                 expansion_cutoff: float,
                 kx: np.ndarray,
                 ky: np.ndarray,
                 position: Sequence[float] = None,
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 always_recenter: bool = False,
                 ctf: CTF = None):

        # TODO : Should multiple positions be supported?

        self._array = array
        self._position = np.array(position)
        self._interpolation = interpolation
        self._expansion_cutoff = expansion_cutoff
        self._kx = kx
        self._ky = ky
        self.always_recenter = always_recenter
        self._grid = Grid(extent=extent, gpts=array.shape[1:], sampling=sampling, lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)
        self.set_ctf(ctf)
        self.cache = Cache(1)
        self.changed = Event()
        self.changed.register(cache_clear_callback(self.alpha))
        self.changed.register(cache_clear_callback(self.phi))

    def set_ctf(self, ctf: CTF = None, **kwargs):
        if ctf is None:
            self._ctf = CTF(**kwargs)
        else:
            self._ctf = copy(ctf)
        self._ctf._accelerator = self._accelerator

    @property
    def ctf(self):
        return self._ctf

    @property
    def position(self):
        return self._position

    @property
    def array(self):
        return self._array

    @property
    def kx(self) -> np.ndarray:
        return self._kx

    @property
    def ky(self) -> np.ndarray:
        return self._ky

    @property
    def interpolation(self) -> int:
        return self._interpolation

    @property
    def interpolated_gpts(self):
        return self.gpts // self.interpolation

    @property
    def interpolated_extent(self):
        return self.interpolated_gpts * self.sampling

    @property
    @cached_method('cache')
    def alpha(self):
        return np.sqrt(self._kx ** 2 + self._ky ** 2) * self.wavelength

    @property
    @cached_method('cache')
    def phi(self):
        return np.arctan2(self._kx, self._ky)

    def multislice(self, potential: AbstractPotential, show_progress: bool = True):
        return self.propagator.multislice(self, potential, show_progress)

    def _allocate(self, n):
        xp = self.device_manager.get_array_library()
        return xp.zeros((n,) + tuple(self.interpolated_gpts), dtype=np.complex64)

    def collapse(self, positions):
        xp = self.device_manager.get_array_library()
        positions = xp.array(positions, dtype=np.float32)

        if positions.shape == (2,):
            positions = positions[None]
        elif (len(positions.shape) != 2) or (positions.shape[-1] != 2):
            raise RuntimeError()

        translation = (complex_exponential(2. * np.pi * self.kx[None] * positions[:, 0, None]) *
                       complex_exponential(2. * np.pi * self.ky[None] * positions[:, 1, None]))

        coefficients = self.ctf.evaluate(self.alpha, self.phi)[None] * translation
        window = self._allocate(len(positions))
        # window[:] = 0.

        crop_corners = xp.round(positions / self.sampling - xp.floor_divide(window.shape[1:], 2)).astype(xp.int)
        crop_corners = xp.remainder(crop_corners, self.gpts)

        # kx = kx[:len(kx) // 2 + 1]
        # ky = ky[:len(ky) // 2 + 1]
        # (self.array[None] * coefficients[:, :, None, None])
        # print((self.array[None] * coefficients[:, :, None, None]).shape)
        window[:] = (self.array[None] * coefficients[:, :, None, None]).sum(1)
        # window_and_collapse(window, self.array, crop_corners, coefficients)

        return Waves(window, extent=self.extent, energy=self.energy)

    def generate_probes(self, scan: AbstractScan, max_batch):
        for start, end, positions in scan.generate_positions(max_batch=max_batch):
            yield start, end, self.collapse(positions)

    def scan(self, scan, detectors, max_batch=1, show_progress=True):
        return _scan(self, scan, detectors, max_batch=max_batch, show_progress=show_progress)


class SMatrixBuilder(HasGridMixin, HasAcceleratorMixin):

    def __init__(self,
                 expansion_cutoff: float,
                 interpolation: int = 1,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None, always_recenter: bool = False,
                 ctf=None,
                 device=None):

        """

        Parameters
        ----------
        expansion_cutoff :
        interpolation :
        extent :
        gpts :
        sampling :
        energy :
        always_recenter :
        """

        if not isinstance(interpolation, int):
            raise ValueError('interpolation factor must be int')

        self._interpolation = interpolation
        self._expansion_cutoff = expansion_cutoff
        self.always_recenter = always_recenter

        self.device_manager = DeviceManager(device)
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)

    @property
    def expansion_cutoff(self) -> float:
        return self._expansion_cutoff

    @expansion_cutoff.setter
    @watched_method('changed')
    def expansion_cutoff(self, value: float):
        self._expansion_cutoff = value

    @property
    def interpolation(self) -> int:
        return self._interpolation

    @interpolation.setter
    @watched_method('changed')
    def interpolation(self, value: int):
        self._interpolation = value

    def multislice(self, potential, show_progress=True) -> SMatrix:
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)
        self.grid.match(potential)
        return self.build().multislice(potential, show_progress=show_progress)

    def build(self) -> SMatrix:
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        xp = self.device_manager.get_array_library()

        n_max = int(xp.ceil(self.expansion_cutoff / (self.wavelength / self.extent[0] * self.interpolation)))
        m_max = int(xp.ceil(self.expansion_cutoff / (self.wavelength / self.extent[1] * self.interpolation)))

        n = xp.arange(-n_max, n_max + 1, dtype=xp.float32)
        w = xp.asarray(self.extent[0], dtype=xp.float32)
        m = xp.arange(-m_max, m_max + 1, dtype=xp.float32)
        h = xp.asarray(self.extent[1], dtype=xp.float32)

        kx = n / w * xp.float32(self.interpolation)
        ky = m / h * xp.float32(self.interpolation)

        mask = kx[:, None] ** 2 + ky[None, :] ** 2 < (self.expansion_cutoff / self.wavelength) ** 2
        kx, ky = xp.meshgrid(kx, ky, indexing='ij')
        kx = kx[mask]
        ky = ky[mask]

        x, y = self.grid.coordinates()
        array = (complex_exponential(-2 * np.pi * kx[:, None, None] * x[None, :, None]) *
                 complex_exponential(-2 * np.pi * ky[:, None, None] * y[None, None, :]))

        return SMatrix(array,
                       interpolation=self.interpolation,
                       expansion_cutoff=self.expansion_cutoff,
                       extent=self.extent,
                       energy=self.energy,
                       kx=kx,
                       ky=ky,
                       always_recenter=self.always_recenter)
