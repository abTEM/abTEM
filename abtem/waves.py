from typing import Union, Sequence, Tuple, Type

import h5py
import numpy as np
import pyfftw as fftw
from ase import Atoms
import cupy as cp

from abtem.analyse import fwhm
from abtem.bases import cached_method, Grid, Energy, cached_method_with_args, ArrayWithGridAndEnergy2D, Cache, \
    notify, ArrayWithGrid, Buildable
from abtem.config import DTYPE, COMPLEX_DTYPE, FFTW_THREADS
from abtem.detect import DetectorBase
from abtem.potentials import Potential, PotentialBase
from abtem.prism import window_and_collapse
from abtem.scan import GridScan, LineScan, CustomScan, ScanBase
from abtem.transfer import CTF, CTFBase
from abtem.utils import complex_exponential, abs2, BatchGenerator


class Waves(ArrayWithGridAndEnergy2D, Cache):
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

        super().__init__(array=array, extent=extent, sampling=sampling, energy=energy)

    def apply_ctf(self, ctf=None, in_place=False, **kwargs):
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

        if ctf is not None:
            self.match_grid(ctf)

        else:
            ctf = CTF(**kwargs, extent=self.extent, gpts=self.gpts, energy=self.energy)

        return self.fft_convolve(ctf.get_array(), in_place)

    @cached_method()
    def get_fftw_plans(self):
        fftw_forward = fftw.FFTW(self._array, self._array, axes=(1, 2), threads=FFTW_THREADS,
                                 flags=('FFTW_ESTIMATE',))
        fftw_backward = fftw.FFTW(self._array, self._array, axes=(1, 2), direction='FFTW_BACKWARD',
                                  threads=FFTW_THREADS, flags=('FFTW_ESTIMATE',))

        return fftw_forward, fftw_backward

    def _on_gpu(self):
        return cp.get_array_module(self.array) is cp

    def fft_convolve(self, array, in_place=True):
        if not in_place:
            waves = self.copy()
        else:
            waves = self

        if self._on_gpu():
            waves._array = cp.fft.ifft2(cp.fft.fft2(waves._array) * array)
        else:
            fftw_forward, fftw_backward = waves.get_fftw_plans()
            fftw_forward()
            waves._array *= array
            fftw_backward()
        return waves

    def transmit(self, potential_slice, in_place=True):
        if not in_place:
            waves = self.copy()
        else:
            waves = self
        waves._array *= complex_exponential(waves.sigma * potential_slice.array)
        return waves

    @cached_method_with_args(('gpts', 'extent', 'sampling', 'energy'))
    def get_propagator(self, dz):
        self.check_is_grid_defined()
        self.check_is_energy_defined()
        xp = cp.get_array_module(self.array)

        kx = xp.fft.fftfreq(self.gpts[0], self.sampling[0]).astype(DTYPE)
        ky = xp.fft.fftfreq(self.gpts[1], self.sampling[1]).astype(DTYPE)
        k = (kx ** 2)[:, None] + (ky ** 2)[None]
        return complex_exponential(-k * np.pi * self.wavelength * dz)[None]

    def intensities(self):
        return abs2(self.array)

    def diffraction_patterns(self):
        return abs2(np.fft.fft2(np.fft.fftshift(self.array)))

    def propagate(self, dz, in_place=True):
        return self.fft_convolve(self.get_propagator(dz), in_place=in_place)

    def multislice(self, potential, in_place: bool = False, show_progress: bool = True):
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
        if not in_place:
            waves = self.copy()
        else:
            waves = self

        for potential_slice in potential:
            waves.transmit(potential_slice)
            waves.propagate(potential_slice.thickness)
        return waves

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
        if len(self.array.shape) <= self.spatial_dimensions:
            raise RuntimeError()
        return self.__class__(array=self._array[item], extent=self.extent.copy(), energy=self.energy)

    def copy(self, copy_array=True) -> 'Waves':
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
        try:
            extent = self.extent.copy()
        except AttributeError:
            extent = self.extent

        if copy_array:
            array = self.array.copy()
        else:
            array = self.array

        new = self.__class__(array=array, extent=extent, energy=self.energy)
        return new


class PlaneWaves(Grid, Energy, Cache):
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
        super().__init__(extent=extent, gpts=gpts, sampling=sampling, dimensions=2, energy=energy)

    @property
    def num_waves(self) -> int:
        return self._num_waves

    @num_waves.setter
    @notify
    def num_waves(self, value: int):
        self._num_waves = value

    def multislice(self, potential, show_progress=True):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        potential.match_grid(self)

        return self.build().multislice(potential, in_place=True, show_progress=show_progress)

    def transmit(self, potential_slice):
        self.match_grid(potential_slice)
        self.check_is_energy_defined()
        self.check_is_grid_defined()
        return self.build().transmit(potential_slice, in_place=False)

    def propagate(self):
        pass

    @cached_method()
    def build(self):
        if self.gpts is None:
            raise RuntimeError('gpts not defined')

        array = np.ones((self.num_waves, self.gpts[0], self.gpts[1]), dtype=COMPLEX_DTYPE)
        return Waves(array, extent=self.extent, energy=self.energy)

    def copy(self):
        return self.__class__(num_waves=self.num_waves, extent=self.extent, gpts=self.gpts, sampling=self.sampling,
                              energy=self.energy)


class Scanable:

    def generate_probes(self):
        raise NotImplementedError()

    def scan(self, scan: Type[ScanBase], potential: Union[Type[PotentialBase], Atoms], detectors, max_batch):
        measurements = scan



class ProbeWaves(CTF, Scanable):
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
                 angular_spread: float = 0., parameters: dict = None,
                 normalize: bool = False,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 build_on_gpu=False,
                 **kwargs):

        self._normalize = normalize

        super().__init__(semiangle_cutoff=semiangle_cutoff, rolloff=rolloff, focal_spread=focal_spread,
                         angular_spread=angular_spread,
                         parameters=parameters, extent=extent, gpts=gpts, sampling=sampling, energy=energy,
                         build_on_gpu=build_on_gpu, **kwargs)

    @property
    def normalize(self) -> bool:
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool):
        self._normalize = value

    def get_image(self):
        return self.build().get_image()

    def fwhm(self):
        return fwhm(self)

    def _translation_multiplier(self, positions):
        xp = self._array_module()

        kx, ky = self.fftfreq()
        kx = kx.reshape((1, -1, 1))
        ky = ky.reshape((1, 1, -1))

        kx = xp.asarray(kx)
        ky = xp.asarray(ky)
        positions = xp.asarray(positions)

        x = positions[:, 0].reshape((-1,) + (1, 1))
        y = positions[:, 1].reshape((-1,) + (1, 1))
        return complex_exponential(2 * np.pi * kx * x) * complex_exponential(2 * np.pi * ky * y)

    def build_at(self, positions: Sequence[Sequence[float]] = None) -> Waves:
        self.check_is_grid_defined()
        self.check_is_energy_defined()

        if positions is None:
            positions = np.zeros((1, 2), dtype=DTYPE)
        else:
            positions = np.array(positions, dtype=DTYPE)

        if len(positions.shape) == 1:
            positions = np.expand_dims(positions, axis=0)

        xp = self._array_module()

        array = super().get_array() * self._translation_multiplier(positions)

        if self._build_on_gpu:
            array[:] = xp.fft.fft2(array)
        else:
            fftw.FFTW(array, array, axes=(1, 2), threads=FFTW_THREADS, flags=('FFTW_ESTIMATE',))()

        if self.normalize:
            array[:] = array / np.sum(xp.abs(array) ** 2, axis=(1, 2), keepdims=True) * xp.prod(array.shape[1:])

        return Waves(array, extent=self.extent, energy=self.energy)

    def multislice_at(self, positions, potential, show_progress=False):
        self.match_grid(potential)
        return self.build_at(positions).multislice(potential, in_place=True, show_progress=show_progress)

    def build(self):
        waves = super().build()
        waves._array = np.fft.fftshift(np.fft.fft2(waves._array))
        return waves

    def get_ctf(self) -> CTF:
        return super().build()

    def generate_probes(self, scan: Type[ScanBase], potential: Union[Type[PotentialBase], Atoms], max_batch='50%'):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        for start, end, positions in scan.generate_positions(max_batch=max_batch):
            yield start, end, self.multislice_at(positions, potential)

    def scan(self, scan):
        pass

    def custom_scan(self, potential: Union[Atoms, PotentialBase],
                    detectors: Union[Sequence[DetectorBase], DetectorBase],
                    positions: Sequence[Sequence[float]]):

        return CustomScan(probe=self, potential=potential, detectors=detectors, positions=positions)

    def line_scan(self, potential: Union[Atoms, PotentialBase],
                  detectors: Union[Sequence[DetectorBase], DetectorBase],
                  start: Sequence[float], end: Sequence[float], gpts: int = None, sampling: float = None,
                  endpoint: bool = True):
        scan = LineScan(probe=self, start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)
        return self.scan(self.exit_probe_factory(potential), scan)

    def grid_scan(self, potential: Union[Atoms, PotentialBase],
                  detectors: Union[Sequence[DetectorBase], DetectorBase],
                  start: Sequence[float], end: Sequence[float], gpts: Union[int, Sequence[int]] = None,
                  sampling: Union[float, Sequence[float]] = None, endpoint: bool = False):

        return GridScan(probe=self, potential=potential, detectors=detectors, start=start, end=end, gpts=gpts,
                        sampling=sampling, endpoint=endpoint)


def _prism_translate(positions, kx, ky):
    """
    Create Fourier space translation multiplier.

    Parameters
    ----------
    positions : Nx2 numpy array
    kx :
    ky :

    Returns
    -------

    """
    array = (complex_exponential(2 * np.pi * kx[None] * positions[:, 0, None]) *
             complex_exponential(2 * np.pi * ky[None] * positions[:, 1, None]))
    return array


class ScatteringMatrix(ArrayWithGrid, CTFBase, Cache):
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

    def __init__(self, array: np.ndarray, interpolation: int, expansion_cutoff: float, kx: np.ndarray, ky: np.ndarray,
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None, always_recenter: bool = False):

        self._interpolation = interpolation
        self._expansion_cutoff = expansion_cutoff
        self._kx = kx
        self._ky = ky
        self.always_recenter = always_recenter

        super().__init__(array=array, spatial_dimensions=2, extent=extent, sampling=sampling, energy=energy)

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
    def probe_extent(self):
        return self.probe_shape * self.sampling

    @property
    def probe_shape(self):
        return np.array((self.gpts[0] // self.interpolation, self.gpts[1] // self.interpolation))

    def build_at(self, positions: Sequence[Sequence[float]]) -> Waves:
        xp = cp.get_array_module(self.array)

        coefficients = super().get_array()[0] * _prism_translate(positions, self.kx, self.ky)

        if (self.interpolation > 1) | self.always_recenter:
            window_shape = (len(positions), self.gpts[0] // self.interpolation, self.gpts[1] // self.interpolation)
            window = xp.zeros(window_shape, dtype=np.complex)

            corners = xp.round(positions / self.sampling - np.floor_divide(window.shape[1:], 2)).astype(np.int)
            corners = xp.remainder(corners, self.gpts)

            window_and_collapse(self.array, window, corners, coefficients)
        else:
            window = (self.array[None] * coefficients[:, :, None, None]).sum(1)

        return Waves(window, extent=self.extent, energy=self.energy)

    def _get_scan_waves_maker(self):
        def scan_waves_func(waves, positions):
            waves = waves.build_at(positions)
            return waves

        return scan_waves_func

    def build(self):
        return Waves(np.fft.fftshift(self.array.sum(0)), extent=self.extent, energy=self.energy)

    @cached_method(('extent', 'gpts', 'sampling', 'energy'))
    def get_alpha(self):
        return np.sqrt(self._kx ** 2 + self._ky ** 2) * self.wavelength

    @cached_method(('extent', 'gpts', 'sampling', 'energy'))
    def get_phi(self):
        return np.arctan2(self._kx, self._ky)

    # def multislice_at(self, positions, in_place=True, show_progress=False):
    #    return multislice(self, potential, in_place=in_place, show_progress=show_progress)

    def custom_scan(self, detectors: Union[Sequence[DetectorBase], DetectorBase],
                    positions: Sequence[Sequence[float]],
                    show_progress: bool = True):

        scan = CustomScan(positions=positions)

        return _do_scan(self, self._get_scan_waves_maker(), scan=scan, detectors=detectors, max_batch=1,
                        show_progress=show_progress)

    def line_scan(self, detectors: Union[Sequence[DetectorBase], DetectorBase],
                  start: Sequence[float], end: Sequence[float], gpts: int = None, sampling: float = None,
                  endpoint: bool = True, max_batch: int = 1, show_progress: bool = True):

        scan = LineScan(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)
        return _do_scan(self, self._get_scan_waves_maker(), scan=scan, detectors=detectors, max_batch=max_batch,
                        show_progress=show_progress)

    def grid_scan(self, detectors: Union[Sequence[DetectorBase], DetectorBase],
                  start: Sequence[float], end: Sequence[float], gpts: Union[int, Sequence[int]] = None,
                  sampling: Union[float, Sequence[float]] = None, endpoint: bool = True, max_batch: int = 1,
                  show_progress: bool = True):

        if start is None:
            start = np.zeros(2, dtype=DTYPE)

        if end is None:
            end = self.extent

        scan = GridScan(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)

        return _do_scan(self, self._get_scan_waves_maker(), scan=scan, detectors=detectors, max_batch=max_batch,
                        show_progress=show_progress)


class PrismWaves(Grid, Energy, Buildable, Cache):

    def __init__(self, expansion_cutoff: float, interpolation: int = 1,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None, always_recenter: bool = False,
                 build_on_gpu=True):

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

        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy, build_on_gpu=build_on_gpu)

    @property
    def expansion_cutoff(self) -> float:
        return self._expansion_cutoff

    @expansion_cutoff.setter
    @notify
    def expansion_cutoff(self, value: float):
        self._expansion_cutoff = value

    @property
    def interpolation(self) -> int:
        return self._interpolation

    @interpolation.setter
    @notify
    def interpolation(self, value: int):
        self._interpolation = value

    @cached_method()
    def get_spatial_frequencies(self) -> Tuple[np.ndarray, np.ndarray]:
        self.check_is_grid_defined()
        self.check_is_energy_defined()

        xp = self._array_module()

        n_max = int(np.ceil(self.expansion_cutoff / (self.wavelength / self.extent[0] * self.interpolation)))
        m_max = int(np.ceil(self.expansion_cutoff / (self.wavelength / self.extent[1] * self.interpolation)))

        kx = xp.arange(-n_max, n_max + 1) / xp.asarray(self.extent[0]) * self.interpolation
        ky = xp.arange(-m_max, m_max + 1) / xp.asarray(self.extent[1]) * self.interpolation

        mask = kx[:, None] ** 2 + ky[None, :] ** 2 < (self.expansion_cutoff / self.wavelength) ** 2
        kx, ky = xp.meshgrid(kx, ky, indexing='ij')

        return kx[mask], ky[mask]

    def multislice(self, potential, show_progress=True) -> ScatteringMatrix:

        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        if self.extent is None:
            self.extent = potential.extent

        if self.gpts is None:
            self.gpts = potential.gpts

        return self.build().multislice(potential, in_place=True, show_progress=show_progress)

    def build(self) -> ScatteringMatrix:
        self.check_is_grid_defined()
        self.check_is_energy_defined()

        xp = self._array_module()
        kx, ky = self.get_spatial_frequencies()

        x = xp.linspace(0, self.extent[0], self.gpts[0], endpoint=self.endpoint)
        y = xp.linspace(0, self.extent[1], self.gpts[1], endpoint=self.endpoint)

        array = (complex_exponential(-2 * np.pi * kx[:, None, None] * x[None, :, None]) *
                 complex_exponential(-2 * np.pi * ky[:, None, None] * y[None, None, :]))

        return ScatteringMatrix(array,
                                interpolation=self.interpolation, expansion_cutoff=self.expansion_cutoff,
                                extent=self.extent, energy=self.energy, kx=kx, ky=ky,
                                always_recenter=self.always_recenter)
