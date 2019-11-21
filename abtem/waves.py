from typing import Union, Sequence, Tuple

import h5py
import numpy as np
import pyfftw as fftw
from ase import Atoms
from tqdm.auto import tqdm

from abtem.bases import cached_method, Grid, Energy, cached_method_with_args, ArrayWithGridAndEnergy, Cache, \
    notify, ArrayWithGrid
from abtem.detect import DetectorBase
from abtem.potentials import Potential, PotentialBase
from abtem.prism import window_and_collapse
from abtem.scan import GridScan, LineScan, CustomScan, ScanBase
from abtem.transfer import CTF, CTFBase
from abtem.utils import complex_exponential, fftfreq, BatchGenerator


def get_fft_plans(shape, threads=16):
    temp_1 = fftw.empty_aligned(shape, dtype='complex128')
    temp_2 = fftw.empty_aligned(shape, dtype='complex128')
    fft_object_forward = fftw.FFTW(temp_1, temp_2, axes=(1, 2), threads=threads)
    fft_object_backward = fftw.FFTW(temp_2, temp_1, axes=(1, 2), direction='FFTW_BACKWARD', threads=threads)
    return temp_1, temp_2, fft_object_forward, fft_object_backward


class Propagator(Grid, Energy, Cache):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None):
        super().__init__(gpts=gpts, extent=extent, sampling=sampling, energy=energy)

    @cached_method_with_args(('gpts', 'extent', 'sampling', 'energy'))
    def build(self, dz):
        self.check_is_grid_defined()
        self.check_is_energy_defined()
        kx = np.fft.fftfreq(self.gpts[0], self.sampling[0])
        ky = np.fft.fftfreq(self.gpts[1], self.sampling[1])
        k = (kx ** 2)[:, None] + (ky ** 2)[None]
        return complex_exponential(-k * np.pi * self.wavelength * dz)[None]


def multislice(waves, potential, in_place=False, show_progress=False):
    if isinstance(potential, Atoms):
        potential = Potential(atoms=potential)

    if not in_place:
        waves = waves.copy()

    if (waves.extent is not None) & np.all(potential.extent != waves.extent):
        raise RuntimeError('inconsistent extent')

    if potential.gpts is None:
        potential = potential.copy(copy_atoms=False)
        potential.gpts = waves.gpts

    elif np.all(potential.gpts != waves.gpts):
        raise RuntimeError('inconsistent grid points')

    temp_1, temp_2, fft_object_forward, fft_object_backward = get_fft_plans(waves.array.shape)

    propagator = Propagator(extent=potential.extent, gpts=potential.gpts, energy=waves.energy)

    for i in tqdm(range(potential.num_slices), disable=not show_progress):
        temp_1[:] = waves.array * complex_exponential(waves.sigma * potential.get_slice(i))
        temp_2[:] = fft_object_forward() * propagator.build(potential.slice_thickness(i))
        waves.array[:] = fft_object_backward()

    return waves


class Waves(ArrayWithGridAndEnergy):
    """
    Waves object.

    The waves object can define a stack of arbitrary 2d wavefunctions defined by a complex array.

    Parameters
    ----------
    array : complex ndarray of shape (n, gpts_x, gpts_y)
        the complex wavefunctions
    extent : sequence of float, float, optional
        lateral extent of wavefunctions [Å]
    sampling : sequence of float, float, optional
        lateral sampling of wavefunctions [1 / Å]
    energy : float, optional
        waves energy [eV]
    """

    def __init__(self, array, extent=None, sampling=None, energy=None):
        array = np.array(array, dtype=np.complex)
        if len(array.shape) == 2:
            array = array[None]
        super().__init__(array=array, spatial_dimensions=2, extent=extent, sampling=sampling, energy=energy)

    def apply_ctf(self, ctf=None, in_place=False, **kwargs):
        if not in_place:
            waves = self.copy()
        else:
            waves = self

        if ctf is not None:
            if ctf.extent is None:
                ctf.extent = self.extent

            if ctf.gpts is None:
                ctf.gpts = self.gpts

            if self.extent is None:
                self.extent = ctf.extent

            self.check_same_grid(ctf)
            self.check_same_energy(ctf)

        else:
            ctf = CTF(**kwargs, extent=self.extent, gpts=self.gpts, energy=self.energy)

        waves.check_is_grid_defined()
        waves.check_is_energy_defined()

        temp_1, temp_2, fft_object_forward, fft_object_backward = get_fft_plans(self.array.shape)
        temp_1[:] = waves.array
        temp_2[:] = fft_object_forward() * ctf.get_array()
        array = fft_object_backward()

        return self.__class__(array, extent=self.extent, energy=self.energy)

    def propagate(self, dz):
        new = self.copy()
        new.array[:] = np.fft.ifft2(
            Propagator(extent=self.extent, gpts=self.gpts, energy=self.energy).build(dz) * np.fft.fft2(new.array))
        return new

    def multislice(self, potential, in_place=False, show_progress=True):
        return multislice(self, potential, in_place=in_place, show_progress=show_progress)

    def copy(self, copy_array=True):
        try:
            extent = self.extent.copy()
        except AttributeError:
            extent = self.extent

        new = self.__class__(array=self.array.copy(), extent=extent, energy=self.energy)
        return new


class PlaneWaves(Grid, Energy, Cache):
    """
    Plane waves object

    The plane waves object can represent a stack of plane waves.

    Parameters
    ----------
    num_waves : int
        number of plane waves in stack
    extent : sequence of float, float, optional
        lateral extent of wavefunctions [Å]
    gpts : sequence of int, int, optional
        number of grid points describing the wavefunctions
    sampling : sequence of float, float, optional
        lateral sampling of wavefunctions [1 / Å]
    energy : float, optional
        waves energy [eV]
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

        if self.extent is None:
            self.extent = potential.extent

        if self.gpts is None:
            self.gpts = potential.gpts

        return self.build().multislice(potential, in_place=True, show_progress=show_progress)

    @cached_method()
    def build(self):
        if self.gpts is None:
            raise RuntimeError('gpts not defined')

        array = np.ones((self.num_waves, self.gpts[0], self.gpts[1]), dtype=np.complex64)
        return Waves(array, extent=self.extent, energy=self.energy)

    def copy(self):
        return self.__class__(num_waves=self.num_waves, extent=self.extent.copy(), energy=self.energy)


def do_scan(probe, scan_waves_maker: callable, scan: ScanBase, detectors: Union[Sequence[DetectorBase], DetectorBase],
            max_batch: int, show_progress: bool = True):
    if not isinstance(detectors, Sequence):
        detectors = [detectors]

    for detector in detectors:
        detector.extent = probe.probe_extent
        detector.gpts = probe.probe_shape
        detector.energy = probe.energy
        scan.open_measurements(detector)

    for start, stop, positions in tqdm(scan.generate_positions(max_batch),
                                       total=int(np.ceil(np.prod(scan.gpts) / max_batch)),
                                       disable=not show_progress):

        waves = scan_waves_maker(probe, positions)

        for detector in detectors:
            if detector.export is not None:
                with h5py.File(detector.export, 'a') as f:
                    f['data'][start:start + stop] = detector.detect(waves)

            else:
                scan.measurements[detector][start:start + stop] = detector.detect(waves)

    for detector in detectors:
        scan.finalize_measurements(detector)

    return scan


def translate(positions, kx, ky):
    kx = kx.reshape((1, -1, 1))
    ky = ky.reshape((1, 1, -1))
    x = positions[:, 0].reshape((-1,) + (1, 1))
    y = positions[:, 1].reshape((-1,) + (1, 1))
    return complex_exponential(2 * np.pi * (kx * x + ky * y))


def normalize_in_place(array):
    array[:] = array / np.sum(np.abs(array) ** 2, axis=(1, 2)) * np.prod(array.shape[1:])[None]
    return array


class ProbeWaves(CTF):
    """
    Probe waves object

    The probe waves object can represent a stack of electron probe wave function for simulating scanning transmission
    electron microscopy.

    Parameters
    ----------
    cutoff : float
        Convergence semi-angle [rad.].
    rolloff : float
        Softens the cutoff. a value of 0 gives a hard cutoff, while 1 gives the softest possible cutoff.
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
        Provide the parameters as keyword arguments.
    """

    def __init__(self, cutoff: float = np.inf, rolloff: float = 0., focal_spread: float = 0., parameters: dict = None,
                 normalize: bool = False,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 **kwargs):

        self._normalize = normalize

        super().__init__(cutoff=cutoff, rolloff=rolloff, focal_spread=focal_spread, parameters=parameters,
                         extent=extent, gpts=gpts, sampling=sampling, energy=energy, **kwargs)

    @property
    def normalize(self) -> bool:
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool):
        self._normalize = value

    def get_image(self):
        return self.build().get_image()

    @property
    def probe_extent(self):
        return self.extent

    @property
    def probe_shape(self):
        return self.gpts

    def build_at(self, positions: Sequence[Sequence[float]]) -> Waves:
        positions = np.array(positions)

        if len(positions.shape) == 1:
            positions = np.expand_dims(positions, axis=0)

        kx, ky = fftfreq(self)

        temp_1 = fftw.empty_aligned((len(positions), self.gpts[0], self.gpts[1]), dtype='complex128')
        temp_2 = fftw.empty_aligned((len(positions), self.gpts[0], self.gpts[1]), dtype='complex128')

        fft_object = fftw.FFTW(temp_1, temp_2, axes=(1, 2))

        temp_1[:] = super().get_array() * translate(positions, kx, ky)
        fft_object()

        return Waves(temp_2, extent=self.extent, energy=self.energy)

    def get_ctf(self) -> CTF:
        return super().build()

    def build(self) -> Waves:
        self.check_is_grid_defined()
        self.check_is_energy_defined()

        array = np.fft.fftshift(np.fft.fft2(super().get_array()))

        if self.normalize:
            normalize_in_place(array)

        return Waves(array, extent=self.extent, energy=self.energy)

    def _get_scan_waves_maker(self, potential):

        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        if self.extent is None:
            self.extent = potential.extent

        if self.gpts is None:
            self.gpts = potential.gpts

        def scan_waves_func(waves, positions):
            waves = waves.build_at(positions)
            waves.multislice(potential=potential, in_place=True, show_progress=False)
            return waves

        return scan_waves_func

    def custom_scan(self, potential: Union[Atoms, PotentialBase],
                    detectors: Union[Sequence[DetectorBase], DetectorBase],
                    positions: Sequence[Sequence[float]],
                    show_progress: bool = True):

        scan = CustomScan(positions=positions)
        return do_scan(self, self._get_scan_waves_maker(potential), scan=scan, detectors=detectors, max_batch=1,
                       show_progress=show_progress)

    def line_scan(self, potential: Union[Atoms, PotentialBase],
                  detectors: Union[Sequence[DetectorBase], DetectorBase],
                  start: Sequence[float], end: Sequence[float], gpts: int = None, sampling: float = None,
                  endpoint: bool = True, max_batch: int = 1, show_progress: bool = True):

        scan = LineScan(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)
        return do_scan(self, self._get_scan_waves_maker(potential), scan=scan, detectors=detectors, max_batch=max_batch,
                       show_progress=show_progress)

    def grid_scan(self, potential: Union[Atoms, PotentialBase],
                  detectors: Union[Sequence[DetectorBase], DetectorBase],
                  start: Sequence[float], end: Sequence[float], gpts: Union[int, Sequence[int]] = None,
                  sampling: Union[float, Sequence[float]] = None, endpoint: bool = False, max_batch: int = 1,
                  show_progress: bool = True):

        if isinstance(potential, Atoms):
            potential = Potential(potential)

        if start is None:
            start = potential.origin

        if end is None:
            end = potential.extent

        scan = GridScan(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)

        return do_scan(self, self._get_scan_waves_maker(potential), scan=scan, detectors=detectors, max_batch=max_batch,
                       show_progress=show_progress)


def prism_translate(positions, kx, ky):
    return complex_exponential(2 * np.pi * (kx[None] * positions[:, 0, None] + ky[None] * positions[:, 1, None]))


class ScatteringMatrix(ArrayWithGrid, CTFBase, Cache):

    def __init__(self, array, interpolation, cutoff, kx, ky, extent=None, sampling=None, energy=None,
                 always_recenter: bool = False):
        self._interpolation = interpolation
        self._cutoff = cutoff
        self._kx = kx
        self._ky = ky
        self._positions = np.array([[0., 0.]])
        self.always_recenter = always_recenter
        super().__init__(array=array, spatial_dimensions=2, extent=extent, sampling=sampling, energy=energy)
        self.cutoff = cutoff

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
        coefficients = super().get_array()[0] * prism_translate(positions, self.kx, self.ky)

        if (self.interpolation > 1) | self.always_recenter:
            window_shape = (len(positions), self.gpts[0] // self.interpolation, self.gpts[1] // self.interpolation)
            window = np.zeros(window_shape, dtype=np.complex)

            corners = np.round(positions / self.sampling - np.floor_divide(window.shape[1:], 2)).astype(np.int)
            corners = np.remainder(corners, self.gpts)

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

    def multislice(self, potential, in_place=True, show_progress=False):
        return multislice(self, potential, in_place=in_place, show_progress=show_progress)

    def custom_scan(self, detectors: Union[Sequence[DetectorBase], DetectorBase],
                    positions: Sequence[Sequence[float]],
                    show_progress: bool = True):

        scan = CustomScan(positions=positions)

        return do_scan(self, self._get_scan_waves_maker(), scan=scan, detectors=detectors, max_batch=1,
                       show_progress=show_progress)

    def line_scan(self, detectors: Union[Sequence[DetectorBase], DetectorBase],
                  start: Sequence[float], end: Sequence[float], gpts: int = None, sampling: float = None,
                  endpoint: bool = True, max_batch: int = 1, show_progress: bool = True):

        scan = LineScan(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)
        return do_scan(self, self._get_scan_waves_maker(), scan=scan, detectors=detectors, max_batch=max_batch,
                       show_progress=show_progress)

    def grid_scan(self, detectors: Union[Sequence[DetectorBase], DetectorBase],
                  start: Sequence[float], end: Sequence[float], gpts: Union[int, Sequence[int]] = None,
                  sampling: Union[float, Sequence[float]] = None, endpoint: bool = True, max_batch: int = 1,
                  show_progress: bool = True):

        if start is None:
            start = (0., 0.)

        if end is None:
            end = self.extent

        scan = GridScan(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)

        return do_scan(self, self._get_scan_waves_maker(), scan=scan, detectors=detectors, max_batch=max_batch,
                       show_progress=show_progress)


class PrismWaves(Grid, Energy, Cache):

    def __init__(self, cutoff: float, interpolation: int = 1,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None, always_recenter: bool = False):

        if not isinstance(interpolation, int):
            raise ValueError('interpolation factor must be int')

        self._interpolation = interpolation
        self._cutoff = cutoff
        self.always_recenter = always_recenter

        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy)

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @cutoff.setter
    @notify
    def cutoff(self, value: float):
        self._cutoff = value

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

        n_max = np.ceil(self.cutoff / (self.wavelength / self.extent[0] * self.interpolation))
        m_max = np.ceil(self.cutoff / (self.wavelength / self.extent[1] * self.interpolation))

        kx = np.arange(-n_max, n_max + 1) / self.extent[0] * self.interpolation
        ky = np.arange(-m_max, m_max + 1) / self.extent[1] * self.interpolation

        mask = kx[:, None] ** 2 + ky[None, :] ** 2 < (self.cutoff / self.wavelength) ** 2
        kx, ky = np.meshgrid(kx, ky, indexing='ij')

        return kx[mask], ky[mask]

    def multislice(self, potential, show_progress=True) -> ScatteringMatrix:

        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        if self.extent is None:
            self.extent = potential.extent

        if self.gpts is None:
            self.gpts = potential.gpts

        return self.build().multislice(potential, in_place=True, show_progress=show_progress)

    def generate_expansion(self, max_batch: int = None):
        kx, ky = self.get_spatial_frequencies()
        x = np.linspace(0, self.extent[0], self.gpts[0], endpoint=self.endpoint)
        y = np.linspace(0, self.extent[1], self.gpts[1], endpoint=self.endpoint)

        if max_batch is None:
            max_batch = len(kx)

        batch_generator = BatchGenerator(len(kx), max_batch)

        for start, length in batch_generator.generate():
            kx_batch = kx[start:start + length]
            ky_batch = ky[start:start + length]
            yield ScatteringMatrix(complex_exponential(-2 * np.pi *
                                                       (kx_batch[:, None, None] * x[None, :, None] +
                                                        ky_batch[:, None, None] * y[None, None, :])),
                                   interpolation=self.interpolation, cutoff=self.cutoff, extent=self.extent,
                                   energy=self.energy, kx=kx_batch, ky=ky_batch, always_recenter=self.always_recenter)

    def build(self) -> ScatteringMatrix:
        return next(self.generate_expansion())
