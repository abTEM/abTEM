from copy import copy
from typing import Union, Sequence

import h5py
import numpy as np
from ase import Atoms

from abtem.bases import Grid, Accelerator, HasGridMixin, HasAcceleratorMixin, cache_clear_callback, watched_property, \
    Cache, cached_method
from abtem.detect import AbstractDetector, crop_to_center
from abtem.device import get_array_module, get_device_function, asnumpy, HasDeviceMixin, get_array_module_from_device, \
    copy_to_device
from abtem.measure import calibrations_from_grid, Measurement
from abtem.plot import show_line
from abtem.potentials import Potential, AbstractPotential, AbstractTDSPotentialBuilder, AbstractPotentialBuilder
from abtem.scan import AbstractScan
from abtem.transfer import CTF
from abtem.utils import polargrid, ProgressBar, cosine_window, spatial_frequencies, coordinates, split_integer


class FresnelPropagator:

    def __init__(self):
        self.cache = Cache(1)

    def antialiasing_aperture(self, gpts, xp):
        x = 1 - cosine_window(np.abs(xp.fft.fftfreq(gpts[0])), .25, .1, 'high')
        y = 1 - cosine_window(np.abs(xp.fft.fftfreq(gpts[1])), .25, .1, 'high')
        return x[:, None] * y[None]

    @cached_method('cache')
    def get_array(self, gpts, sampling, wavelength, dz, xp):
        complex_exponential = get_device_function(xp, 'complex_exponential')
        kx = xp.fft.fftfreq(gpts[0], sampling[0]).astype(xp.float32)
        ky = xp.fft.fftfreq(gpts[1], sampling[1]).astype(xp.float32)
        f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * wavelength * dz) *
             complex_exponential(-(ky ** 2)[None] * np.pi * wavelength * dz))
        f *= self.antialiasing_aperture(gpts, xp)
        return f

    def propagate(self, waves, dz):
        p = self.get_array(waves.grid.gpts, waves.grid.sampling, dz, waves.wavelength, get_array_module(waves.array))

        fft2_convolve = get_device_function(get_array_module(waves.array), 'fft2_convolve')
        fft2_convolve(waves._array, p)
        return waves


def transmit(waves, potential_slice):
    xp = get_array_module(waves.array)
    complex_exponential = get_device_function(xp, 'complex_exponential')
    dim_padding = len(waves._array.shape) - len(potential_slice.array.shape)
    slice_array = potential_slice.array.reshape((1,) * dim_padding + potential_slice.array.shape)

    if np.iscomplexobj(slice_array):
        waves._array *= slice_array
    else:
        waves._array *= complex_exponential(waves.accelerator.sigma * slice_array)

    return waves


def multislice(waves: Union['Waves', 'SMatrix', 'PartialSMatrix'], potential: AbstractPotential, propagator=None,
               pbar: Union[ProgressBar, bool] = True):
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

    if propagator is None:
        propagator = FresnelPropagator()

    if isinstance(pbar, bool):
        pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not pbar)

    pbar.reset()
    for potential_slice in potential:
        transmit(waves, potential_slice)
        propagator.propagate(waves, potential_slice.thickness)
        pbar.update(1)

    pbar.refresh()
    return waves


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

        # if len(array.shape) == 2:
        #    array = array[None]

        self._array = array
        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)

    @property
    def array(self):
        return self._array

    def intensity(self):
        calibrations = calibrations_from_grid(self.grid.gpts, self.grid.sampling, ['x', 'y'])
        calibrations = [None] * (len(self.array.shape) - 2) + calibrations

        abs2 = get_device_function(get_array_module(self.array), 'abs2')
        return Measurement(abs2(self.array), calibrations)

    def diffraction_pattern(self):
        calibrations = calibrations_from_grid(self.grid.antialiased_gpts,
                                              self.grid.antialiased_sampling,
                                              names=['alpha_x', 'alpha_y'],
                                              units='mrad.',
                                              scale_factor=self.wavelength,
                                              fourier_space=True)

        calibrations = [None] * (len(self.array.shape) - 2) + calibrations

        xp = get_array_module(self.array)
        abs2 = get_device_function(xp, 'abs2')
        fft2 = get_device_function(xp, 'fft2')
        pattern = asnumpy(abs2(crop_to_center(xp.fft.fftshift(fft2(self.array, overwrite_x=False)))))
        return Measurement(pattern, calibrations)

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
        kx, ky = spatial_frequencies(self.grid.gpts, self.grid.sampling)
        alpha, phi = polargrid(xp.asarray(kx * self.wavelength), xp.asarray(ky * self.wavelength))
        kernel = ctf.evaluate(alpha, phi)

        return self.__class__(fft2_convolve(self.array, kernel, overwrite_x=False),
                              extent=self.extent,
                              energy=self.energy)

    def multislice(self, potential: AbstractPotential, pbar: Union[ProgressBar, bool] = True):
        propagator = FresnelPropagator()

        if isinstance(potential, AbstractTDSPotentialBuilder):
            xp = get_array_module(self.array)
            N = len(potential.frozen_phonons)
            out_array = xp.zeros((N,) + self.array.shape, dtype=xp.complex64)
            tds_waves = self.__class__(out_array, extent=self.extent, energy=self.energy)

            tds_pbar = ProgressBar(total=N, desc='TDS', disable=(not pbar) or (N == 1))
            multislice_pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not pbar)

            for i, potential in enumerate(potential.generate_frozen_phonon_potentials()):
                multislice_pbar.reset()

                exit_waves = multislice(self.copy(), potential, propagator=propagator, pbar=multislice_pbar)
                tds_waves.array[i] = exit_waves.array
                tds_pbar.update(1)

            multislice_pbar.close()
            tds_pbar.close()

            return tds_waves
        else:
            return multislice(self, potential, propagator, pbar)

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
        return self.__class__(array=self._array[item], extent=self.extent, energy=self.energy)

    def copy(self):
        """
        Return a copy.
        """
        new_copy = self.__class__(array=self._array.copy())
        new_copy._grid = self.grid.copy()
        new_copy._accelerator = self.accelerator.copy()
        return new_copy

    def show(self, **kwargs):
        self.intensity().show(**kwargs)


class PlaneWave(HasGridMixin, HasAcceleratorMixin):
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

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, device='cpu'):
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)
        self._device = device

    def multislice(self, potential, pbar=True):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)
        potential.grid.match(self)
        return self.build().multislice(potential, pbar=pbar)

    def build(self):
        xp = get_array_module_from_device(self._device)
        self.grid.check_is_defined()
        array = xp.ones((self.gpts[0], self.gpts[1]), dtype=xp.complex64)
        return Waves(array, extent=self.extent, energy=self.energy)

    def copy(self):
        return self.__class__(extent=self.extent, gpts=self.gpts, sampling=self.sampling, energy=self.energy)


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

    def __init__(self,
                 semiangle_cutoff: float = np.inf,
                 rolloff: float = 0.05,
                 focal_spread: float = 0.,
                 angular_spread: float = 0.,
                 ctf_parameters: dict = None,
                 normalize: bool = False,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 device='cpu',
                 **kwargs):

        self._normalize = normalize
        self._ctf = CTF(semiangle_cutoff=semiangle_cutoff, rolloff=rolloff, focal_spread=focal_spread,
                        angular_spread=angular_spread, parameters=ctf_parameters, energy=energy, **kwargs)
        self._accelerator = self._ctf._accelerator
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self.cache = Cache(1)

        self._ctf.changed.register(cache_clear_callback(self.cache))
        self._grid.changed.register(cache_clear_callback(self.cache))
        self._accelerator.changed.register(cache_clear_callback(self.cache))

        self._device = device

    @property
    def ctf(self):
        return self._ctf

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

        kx, ky = spatial_frequencies(self.grid.gpts, self.grid.sampling)
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
        kx, ky = spatial_frequencies(self.grid.gpts, self.grid.sampling)
        alpha, phi = polargrid(xp.asarray(kx * self.wavelength), xp.asarray(ky * self.wavelength))
        return self._ctf.evaluate(alpha, phi)

    def build(self, positions: Sequence[Sequence[float]] = None) -> Waves:
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        xp = get_array_module_from_device(self._device)
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

    def multislice(self, positions: Sequence[Sequence[float]], potential: AbstractPotential, pbar=True) -> Waves:
        self.grid.match(potential)
        return multislice(self.build(positions), potential, None, pbar)

    def generate_probes(self, scan: AbstractScan, potential: Union[AbstractPotential, Atoms], max_batch: int):
        for start, end, positions in scan.generate_positions(max_batch=max_batch):
            yield start, end, self.multislice(positions, potential, pbar=False)

    def generate_tds_probes(self, scan, potential, max_batch, pbar):
        tds_bar = ProgressBar(total=len(potential.frozen_phonons), desc='TDS',
                              disable=(not pbar) or (len(potential.frozen_phonons) == 1))
        potential_pbar = ProgressBar(total=len(potential), desc='Potential', disable=not pbar)

        for potential_config in potential.generate_frozen_phonon_potentials(pbar=potential_pbar):
            yield self.generate_probes(scan, potential_config, max_batch)
            tds_bar.update(1)

        potential_pbar.close()
        tds_bar.refresh()
        tds_bar.close()

    def scan(self, scan: AbstractScan, detectors: Sequence[AbstractDetector],
             potential: Union[Atoms, AbstractPotential], max_batch=1, pbar: Union[ProgressBar, bool] = True):

        self.grid.match(potential.grid)
        self.grid.check_is_defined()

        measurements = {}
        for detector in detectors:
            measurements[detector] = detector.allocate_measurement(self.grid, self.wavelength, scan)

        scan_bar = ProgressBar(total=len(scan), desc='Scan', disable=not pbar)

        if isinstance(potential, AbstractTDSPotentialBuilder):
            probe_generators = self.generate_tds_probes(scan, potential, max_batch, pbar)
        else:
            if isinstance(potential, AbstractPotentialBuilder):
                potential = potential.build(pbar=True)

            probe_generators = [self.generate_probes(scan, potential, max_batch)]

        for probe_generator in probe_generators:
            scan_bar.reset()
            for start, end, exit_probes in probe_generator:
                for detector, measurement in measurements.items():
                    scan.insert_new_measurement(measurement, start, end, detector.detect(exit_probes))

                scan_bar.update(end - start)

            scan_bar.refresh()

        scan_bar.close()

        return measurements

    def show(self, profile=False, **kwargs):
        measurement = self.build((self.extent[0] / 2, self.extent[1] / 2)).intensity()

        if profile:
            array = measurement.array[0]
            array = array[array.shape[0] // 2, :]
            calibration = calibrations_from_grid(gpts=(self.grid.gpts[1],),
                                                 sampling=(self.grid.sampling[1],),
                                                 names=['x'])[0]
            show_line(array, calibration, **kwargs)
        else:
            return measurement.show(**kwargs)

    def show_interactive(self):
        from abtem.interactive import BokehImage

        def new_measurement_callback(*args, **kwargs):
            return self.build((self.extent[0] / 2, self.extent[1] / 2)).intensity()[0]

        image = BokehImage(new_measurement_callback, push_notebook=True)
        image.update()

        self.ctf.changed.register(image.update)
        self.grid.changed.register(image.update)
        self.accelerator.changed.register(image.update)

    def copy(self):
        new_copy = self.__class__(normalize=self._normalize)
        new_copy._grid = self.grid.copy()
        new_copy._ctf = self.ctf.copy()
        new_copy._accelerator = self._ctf._accelerator
        return new_copy


class PartialSMatrix(HasGridMixin, HasAcceleratorMixin):

    def __init__(self, start, stop, parent):
        self._start = start
        self._stop = stop
        self._parent = parent

    @property
    def _array(self):
        return self._parent._array[self._start:self._stop]

    @_array.setter
    def _array(self, value):
        self._parent._array[self._start:self._stop] = value

    @property
    def array(self):
        return self._array

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def kx(self):
        return self._parent._kx[self._start:self._stop]

    @property
    def ky(self):
        return self._parent._ky[self._start:self._stop]

    @property
    def _grid(self):
        return self._parent._grid

    @property
    def _accelerator(self):
        return self._parent._accelerator


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
    def interpolated_grid(self):
        interpolated_gpts = tuple(n // self.interpolation for n in self.gpts)
        return Grid(gpts=interpolated_gpts, sampling=self.sampling, lock_gpts=True)

    def _evaluate_ctf(self, xp):
        alpha = xp.sqrt(self._kx ** 2 + self._ky ** 2) * self.wavelength
        phi = xp.arctan2(self._kx, self._ky)
        return self._ctf.evaluate(alpha, phi)

    def __len__(self):
        return len(self._array)

    def generate_partial(self, max_batch=None, pbar=True):
        if max_batch is None:
            n_batches = 1
        else:
            n_batches = (len(self) + (-len(self) % max_batch)) // max_batch

        batch_pbar = ProgressBar(total=len(self), desc='Batches', disable=(not pbar) or (n_batches == 1))
        batch_sizes = split_integer(len(self), n_batches)
        N = 0
        for batch_size in batch_sizes:
            yield PartialSMatrix(N, N + batch_size, self)
            N += batch_size
            batch_pbar.update(batch_size)

        batch_pbar.refresh()
        batch_pbar.close()

    def multislice(self, potential: AbstractPotential, max_batch=None, pbar: bool = True):
        propagator = FresnelPropagator()

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not pbar)

        for partial_s_matrix in self.generate_partial(max_batch):
            multislice(partial_s_matrix, potential, propagator=propagator, pbar=pbar)

        pbar.refresh()

        return self

    def collapse(self, positions, max_batch=None):
        xp = get_array_module(self.array)
        complex_exponential = get_device_function(xp, 'complex_exponential')
        scale_reduce = get_device_function(xp, 'scale_reduce')
        windowed_scale_reduce = get_device_function(xp, 'windowed_scale_reduce')

        positions = np.array(positions, dtype=xp.float32)

        if positions.shape == (2,):
            positions = positions[None]
        elif (len(positions.shape) != 2) or (positions.shape[-1] != 2):
            raise RuntimeError()

        interpolated_grid = self.interpolated_grid  # np.array(self.gpts) // self.interpolation
        W = np.floor_divide(interpolated_grid.gpts, 2)
        corners = np.rint(positions / self.sampling - W).astype(np.int)
        corners = np.asarray(corners, dtype=xp.int)
        corners = np.remainder(corners, np.asarray(self.gpts))
        corners = xp.asarray(corners)

        window = xp.zeros((len(positions), interpolated_grid.gpts[0], interpolated_grid.gpts[1],), dtype=xp.complex64)

        positions = xp.asarray(positions)

        translation = (complex_exponential(2. * np.pi * self.kx[None] * positions[:, 0, None]) *
                       complex_exponential(2. * np.pi * self.ky[None] * positions[:, 1, None]))

        coefficients = translation * self._evaluate_ctf(xp)

        for partial_s_matrix in self.generate_partial(max_batch, pbar=False):
            partial_coefficients = coefficients[:, partial_s_matrix.start:partial_s_matrix.stop]

            if self.interpolation > 1:
                windowed_scale_reduce(window, partial_s_matrix.array, corners, partial_coefficients)
            else:
                scale_reduce(window, partial_s_matrix.array, partial_coefficients)

        return Waves(window, extent=interpolated_grid.extent, energy=self.energy)

    def generate_probes(self, scan: AbstractScan, max_batch_probes, max_batch_expansion):
        for start, end, positions in scan.generate_positions(max_batch=max_batch_probes):
            yield start, end, self.collapse(positions, max_batch=max_batch_expansion)

    def scan(self,
             scan: AbstractScan,
             detectors: Sequence[AbstractDetector],
             max_batch_positions=1,
             max_batch_expansion=None,
             pbar: Union[ProgressBar, bool] = True):

        measurements = {}
        for detector in detectors:
            measurements[detector] = detector.allocate_measurement(self.interpolated_grid, self.wavelength, scan)

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(scan), desc='Scan', disable=not pbar)

        for start, end, exit_probes in self.generate_probes(scan, max_batch_positions, max_batch_expansion):
            for detector, measurement in measurements.items():
                scan.insert_new_measurement(measurement, start, end, detector.detect(exit_probes))
            pbar.update(end - start)

        pbar.refresh()
        pbar.close()
        return measurements


class SMatrixBuilder(HasGridMixin, HasAcceleratorMixin, HasDeviceMixin):

    def __init__(self,
                 expansion_cutoff: float,
                 interpolation: int = 1,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None, always_recenter: bool = False,
                 ctf=None,
                 device='cpu',
                 storage=None):
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
        self._ctf = ctf

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)

        self._device = device
        if storage is None:
            storage = device

        self._storage = storage

    @property
    def expansion_cutoff(self) -> float:
        return self._expansion_cutoff

    @expansion_cutoff.setter
    @watched_property('changed')
    def expansion_cutoff(self, value: float):
        self._expansion_cutoff = value

    @property
    def interpolation(self) -> int:
        return self._interpolation

    @interpolation.setter
    @watched_property('changed')
    def interpolation(self, value: int):
        self._interpolation = value

    @property
    def interpolated_grid(self):
        interpolated_gpts = tuple(n // self.interpolation for n in self.gpts)
        return Grid(gpts=interpolated_gpts, sampling=self.sampling, lock_gpts=True)

    def generate_tds_probes(self, scan,
                            potential,
                            max_batch_probes,
                            max_batch_expansion,
                            potential_pbar: Union[ProgressBar, bool] = True,
                            multislice_pbar: Union[ProgressBar, bool] = True):

        if isinstance(potential_pbar, bool):
            potential_pbar = ProgressBar(total=len(potential), desc='Potential', disable=not potential_pbar)

        if isinstance(multislice_pbar, bool):
            multislice_pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not multislice_pbar)

        for potential_config in potential.generate_frozen_phonon_potentials(pbar=potential_pbar):
            S = self.multislice(potential_config, max_batch=max_batch_expansion, pbar=multislice_pbar)
            yield S.generate_probes(scan, max_batch_probes, max_batch_expansion)

        multislice_pbar.refresh()
        multislice_pbar.close()

        potential_pbar.refresh()
        potential_pbar.close()

    def multislice(self, potential, max_batch=None, pbar: Union[ProgressBar, bool] = True):
        return self.build().multislice(potential, max_batch=max_batch, pbar=pbar)

    def scan(self,
             potential: Union[Atoms, AbstractPotential],
             scan: AbstractScan,
             detectors: Sequence[AbstractDetector],
             max_batch_probes=1,
             max_batch_expansion=None,
             pbar=True):

        self.grid.match(potential.grid)
        self.grid.check_is_defined()

        measurements = {}
        for detector in detectors:
            measurements[detector] = detector.allocate_measurement(self.interpolated_grid, self.wavelength, scan)

        if isinstance(potential, AbstractTDSPotentialBuilder):
            probe_generators = self.generate_tds_probes(scan,
                                                        potential,
                                                        max_batch_probes=max_batch_probes,
                                                        max_batch_expansion=max_batch_expansion,
                                                        potential_pbar=pbar,
                                                        multislice_pbar=pbar)
        else:
            if isinstance(potential, AbstractPotentialBuilder):
                potential = potential.build(pbar=True)

            S = self.multislice(potential, max_batch=max_batch_probes, pbar=pbar)
            probe_generators = [S.generate_probes(scan, max_batch_probes, max_batch_expansion)]

        tds_bar = ProgressBar(total=len(potential.frozen_phonons), desc='TDS',
                              disable=(not pbar) or (len(potential.frozen_phonons) == 1))

        scan_bar = ProgressBar(total=len(scan), desc='Scan', disable=not pbar)

        for probe_generator in probe_generators:
            scan_bar.reset()

            for start, end, exit_probes in probe_generator:
                for detector, measurement in measurements.items():
                    scan.insert_new_measurement(measurement, start, end, detector.detect(exit_probes))

                scan_bar.update(end - start)

            scan_bar.refresh()
            tds_bar.update(1)

        scan_bar.close()

        tds_bar.refresh()
        tds_bar.close()

        return measurements

    def build(self):
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        xp = self.get_array_module()
        storage_xp = get_array_module_from_device(self._storage)
        complex_exponential = get_device_function(xp, 'complex_exponential')

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

        x, y = coordinates(extent=self.extent, gpts=self.gpts, endpoint=self.grid.endpoint)
        x = xp.asarray(x)
        y = xp.asarray(y)

        array = storage_xp.zeros((len(kx),) + (self.gpts[0], self.gpts[1]), dtype=np.complex64)

        for i in range(len(kx)):
            array[i] = copy_to_device(complex_exponential(-2 * np.pi * kx[i, None, None] * x[:, None]) *
                                      complex_exponential(-2 * np.pi * ky[i, None, None] * y[None, :]), self._storage)

        return SMatrix(array,
                       interpolation=self.interpolation,
                       expansion_cutoff=self.expansion_cutoff,
                       extent=self.extent,
                       energy=self.energy,
                       kx=kx,
                       ky=ky,
                       always_recenter=self.always_recenter)
