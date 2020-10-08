"""Module to describe electron waves and their propagation."""
from copy import copy
from numbers import Number
from typing import Union, Sequence, Tuple

import h5py
import numpy as np
from ase import Atoms

from abtem.base_classes import Grid, Accelerator, cache_clear_callback, Cache, cached_method, \
    HasGridAndAcceleratorMixin, AntialiasFilter, Event
from abtem.detect import AbstractDetector
from abtem.device import get_array_module, get_device_function, asnumpy, get_array_module_from_device, \
    copy_to_device, get_available_memory, HasDeviceMixin
from abtem.measure import calibrations_from_grid, Measurement, block_zeroth_order_spot, probe_profile
from abtem.potentials import Potential, AbstractPotential, AbstractTDSPotentialBuilder, AbstractPotentialBuilder
from abtem.scan import AbstractScan, GridScan
from abtem.transfer import CTF
from abtem.utils import polar_coordinates, ProgressBar, spatial_frequencies, subdivide_into_batches, periodic_crop, \
    fft_crop, fft_interpolate_2d


class FresnelPropagator:
    """
    Fresnel propagator object.

    This class is used for propagating a wave function object using the near-field approximation (Fresnel diffraction).
    The array representing the Fresnel propagator function is cached.
    """

    def __init__(self):
        self._cache = Cache(1)

    @cached_method('_cache')
    def _evaluate_propagator_array(self,
                                   gpts: Tuple[int, int],
                                   sampling: Tuple[float, float],
                                   wavelength: float,
                                   dz: float,
                                   xp) -> np.ndarray:
        complex_exponential = get_device_function(xp, 'complex_exponential')
        kx = xp.fft.fftfreq(gpts[0], sampling[0]).astype(xp.float32)
        ky = xp.fft.fftfreq(gpts[1], sampling[1]).astype(xp.float32)
        f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * wavelength * dz) *
             complex_exponential(-(ky ** 2)[None] * np.pi * wavelength * dz))

        return f * AntialiasFilter().get_mask(gpts, sampling, xp)

    def propagate(self, waves: Union['Waves', 'SMatrixArray'], dz: float) -> Union['Waves', 'SMatrixArray']:
        """
        Propagate wave functions or scattering matrix.

        Parameters
        ----------
        waves : Waves or SMatrixArray object
            Wave function or scattering matrix to propagate.
        dz : float
            Propagation distance [Å].

        Returns
        -------
        Waves or SMatrixArray object
            The propagated wave functions.
        """
        fft2_convolve = get_device_function(get_array_module(waves.array), 'fft2_convolve')

        propagator_array = self._evaluate_propagator_array(waves.grid.gpts,
                                                           waves.grid.sampling,
                                                           dz,
                                                           waves.wavelength,
                                                           get_array_module(waves.array))

        fft2_convolve(waves._array, propagator_array, overwrite_x=True)
        waves._antialiasing = 2 / 3.

        return waves


def _multislice(waves: Union['Waves', 'SMatrixArray'],
                potential: AbstractPotential,
                propagator: FresnelPropagator = None,
                pbar: Union[ProgressBar, bool] = True) -> Union['Waves', 'SMatrixArray']:
    waves.grid.match(potential)
    waves.accelerator.check_is_defined()
    waves.grid.check_is_defined()

    if propagator is None:
        propagator = FresnelPropagator()

    if isinstance(pbar, bool):
        pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not pbar)

    pbar.reset()
    for start, end, t in potential.generate_transmission_functions(energy=waves.energy):
        waves = t.transmit(waves)
        waves = propagator.propagate(waves, t.thickness)
        pbar.update(1)

    pbar.refresh()
    return waves


class Waves(HasGridAndAcceleratorMixin):
    """
    Waves object

    The waves object can define a batch of arbitrary 2D wave functions defined by a complex numpy array.

    Parameters
    ----------
    extent : one or two float
        Lateral extent of wave function [Å].
    sampling : one or two float
        Lateral sampling of wave functions [1 / Å].
    energy : float
        Electron energy [eV].
    """

    def __init__(self,
                 array: np.ndarray,
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 antialiasing: float = None):

        if len(array.shape) < 2:
            raise RuntimeError('Wave function array should be have 2 dimensions or more')

        self._array = array
        self._antialiasing = antialiasing

        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)

    def __len__(self):
        return len(self.array)

    @property
    def array(self) -> np.ndarray:
        """Array representing the wave functions."""
        return self._array

    def intensity(self) -> Measurement:
        """
        Calculate the intensity of the wave functions at the image plane.

        Returns
        -------
        Measurement
            The wave function intensity.
        """

        calibrations = calibrations_from_grid(self.grid.gpts, self.grid.sampling, ['x', 'y'])
        calibrations = (None,) * (len(self.array.shape) - 2) + calibrations

        abs2 = get_device_function(get_array_module(self.array), 'abs2')
        return Measurement(abs2(self.array), calibrations)

    def diffraction_pattern(self, max_angle=None, block_zeroth_order=False) -> Measurement:
        """
        Calculate the intensity of the wave functions at the diffraction plane.

        Returns
        -------
        Measurement object
            The intensity of the diffraction pattern(s).
        """
        xp = get_array_module(self.array)
        abs2 = get_device_function(xp, 'abs2')
        fft2 = get_device_function(xp, 'fft2')

        array = fft2(self.array, overwrite_x=False)

        if max_angle in ('valid', 'limit'):
            antialias_filter = AntialiasFilter()
            array = antialias_filter.crop(array, self.sampling, max_angle)

        elif isinstance(max_angle, Number):
            new_gpts = self._resampled_gpts(max_angle)
            array = fft_crop(array, self.array.shape[:-2] + new_gpts)

        elif max_angle is not None:
            raise ValueError('max_angle must be "valid", "limit", float or None')

        sampling = (self.extent[0] / array.shape[-2], self.extent[1] / array.shape[-1])

        calibrations = calibrations_from_grid(array.shape[-2:],
                                              sampling,
                                              names=['alpha_x', 'alpha_y'],
                                              units='mrad',
                                              scale_factor=self.wavelength * 1000,
                                              fourier_space=True)

        calibrations = (None,) * (len(self.array.shape) - 2) + calibrations

        pattern = np.fft.fftshift(asnumpy(abs2(array)))

        measurement = Measurement(pattern, calibrations)

        if block_zeroth_order:
            block_zeroth_order_spot(measurement)

        return measurement

    def apply_ctf(self, ctf: CTF = None, in_place=False, **kwargs):
        """
        Apply the aberrations defined by a CTF object to wave function.

        Parameters
        ----------
        ctf : CTF
            Contrast Transfer Function object to be applied.
        kwargs :
            Provide the aberration coefficients as keyword arguments.

        Returns
        -------
        Waves object
            The wave functions with aberrations applied.
        """

        xp = get_array_module(self.array)
        fft2_convolve = get_device_function(get_array_module(self.array), 'fft2_convolve')

        if ctf is None:
            ctf = CTF(**kwargs)

        if not ctf.accelerator.energy:
            ctf.accelerator.match(self.accelerator)

        self.accelerator.check_is_defined()
        self.grid.check_is_defined()

        kx, ky = spatial_frequencies(self.grid.gpts, self.grid.sampling)
        alpha, phi = polar_coordinates(xp.asarray(kx * self.wavelength), xp.asarray(ky * self.wavelength))
        kernel = ctf.evaluate(alpha, phi)

        return self.__class__(fft2_convolve(self.array, kernel, overwrite_x=in_place),
                              extent=self.extent,
                              energy=self.energy)

    def multislice(self,
                   potential: AbstractPotential,
                   # reduce_tds: str = None,
                   pbar: Union[ProgressBar, bool] = True) -> 'Waves':
        """
        Propagate and transmit wave function through the provided potential.

        Parameters
        ----------
        potential : Potential
            The potential through which to propagate the wave function.
        pbar : bool
            If true, display a progress bar.

        Returns
        -------
        Waves object
            Wave function at the exit plane of the potential.
        """

        self.grid.match(potential)

        propagator = FresnelPropagator()

        if isinstance(potential, AbstractTDSPotentialBuilder):
            if len(potential.frozen_phonons) > 1:
                xp = get_array_module(self.array)
                n = len(potential.frozen_phonons)

                if n > 1:
                    out_array = xp.zeros((n,) + self.array.shape, dtype=xp.complex64)
                else:
                    out_array = xp.zeros(self.array.shape, dtype=xp.complex64)

                # TODO : implement reduction operation
                tds_pbar = ProgressBar(total=n, desc='TDS', disable=(not pbar) or (n == 1))
                multislice_pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not pbar)

                for i, potential_config in enumerate(potential.generate_frozen_phonon_potentials(pbar=pbar)):
                    multislice_pbar.reset()

                    exit_waves = _multislice(copy(self), potential_config, propagator=propagator, pbar=multislice_pbar)
                    out_array[i] = exit_waves.array

                    tds_pbar.update(1)

                tds_waves = self.__class__(out_array, extent=self.extent, energy=self.energy)

                multislice_pbar.close()
                tds_pbar.close()

                return tds_waves

        return _multislice(self, potential, propagator, pbar)

    def write(self, path: str):
        """
        Write wave functions to a hdf5 file.

        path : str
            The path to write the file.
        """

        with h5py.File(path, 'w') as f:
            f.create_dataset('array', data=self.array)
            f.create_dataset('energy', data=self.energy)
            f.create_dataset('extent', data=self.extent)

    @classmethod
    def read(cls, path: str) -> 'Waves':
        """
        Read wave functions from a hdf5 file.

        path : str
            The path to read the file.
        """

        with h5py.File(path, 'r') as f:
            datasets = {}
            for key in f.keys():
                datasets[key] = f.get(key)[()]

        return cls(array=datasets['array'], extent=datasets['extent'], energy=datasets['energy'])

    def __getitem__(self, item):
        if len(self.array.shape) <= self.grid.dimensions:
            raise RuntimeError()
        return self.__class__(array=self._array[item], extent=self.extent, energy=self.energy)

    def show(self, ax=None, **kwargs):
        """
        Show the wave function.

        kwargs :
            Additional keyword arguments for the abtem.plot.show_image function.
        """
        return self.intensity().show(**kwargs)

    def __copy__(self):
        new_copy = self.__class__(array=self._array.copy())
        new_copy._grid = copy(self.grid)
        new_copy._accelerator = copy(self.accelerator)
        return new_copy

    def copy(self):
        """Make a copy."""
        return copy(self)


class PlaneWave(HasGridAndAcceleratorMixin, HasDeviceMixin):
    """
    Plane wave object

    The plane wave object is used for building plane waves.

    Parameters
    ----------
    extent : two float
        Lateral extent of wave function [Å].
    gpts : two int
        Number of grid points describing the wave function.
    sampling : two float
        Lateral sampling of wave functions [1 / Å].
    energy : float
        Electron energy [eV].
    device : str
        The plane waves will be build on this device.
    """

    def __init__(self,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 device: str = 'cpu'):
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)
        self._device = device

    def multislice(self, potential: Union[AbstractPotential, Atoms], pbar: bool = True) -> Waves:
        """
        Build plane wave function and propagate it through the potential. The grid of the two will be matched.

        Parameters
        ----------
        potential : Potential or Atoms object
            The potential through which to propagate the wave function.
        pbar : bool, optional
            Display a progress bar. Default is True.

        Returns
        -------
        Waves object
            Wave function at the exit plane of the potential.
        """

        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)
        potential.grid.match(self)

        return self.build().multislice(potential, pbar=pbar)

    def build(self) -> Waves:
        """Build the plane wave function as a Waves object."""
        xp = get_array_module_from_device(self._device)
        self.grid.check_is_defined()
        array = xp.ones((1, self.gpts[0], self.gpts[1]), dtype=xp.complex64)
        return Waves(array, extent=self.extent, energy=self.energy)

    def __copy__(self, a) -> 'PlaneWave':
        return self.__class__(extent=self.extent, gpts=self.gpts, sampling=self.sampling, energy=self.energy)

    def copy(self):
        """Make a copy."""
        return copy(self)


class Probe(HasGridAndAcceleratorMixin, HasDeviceMixin):
    """
    Probe wave function object

    The probe object can represent a stack of electron probe wave function for simulating scanning transmission
    electron microscopy.

    See the docs of abtem.transfer.CTF for a description of the parameters related to the contrast transfer function.

    Parameters
    ----------
    semiangle_cutoff : float
        Convergence semi-angle [mrad].
    rolloff : float
        Softens the cutoff. A value of 0 gives a hard cutoff, while 1 gives the softest possible cutoff.
    focal_spread : float
        The focal spread of the probe.
    angular_spread : float
        The angular spread of the probe
    ctf_parameters : dict
        The parameters describing the phase aberrations using polar notation or an alias.
    extent : two float, optional
        Lateral extent of wave functions [Å].
    gpts : two int, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å].
    energy : float, optional
        Electron energy [eV].
    device : str
        The probe wave functions will be build on this device.
    kwargs :
        Provide the aberration coefficients as keyword arguments.
    """

    def __init__(self,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 device='cpu',
                 **kwargs):

        ctf = CTF(energy=energy, **kwargs)

        self._ctf = ctf
        self._accelerator = self._ctf._accelerator

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self.changed = Event()

        self._ctf.changed.register(self.changed.notify)
        self._grid.changed.register(self.changed.notify)
        self._accelerator.changed.register(self.changed.notify)

        self._device = device

        self._ctf_cache = Cache(1)
        self.changed.register(cache_clear_callback(self._ctf_cache))

    @property
    def ctf(self) -> CTF:
        """Probe contrast transfer function."""
        return self._ctf

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

    @cached_method('_ctf_cache')
    def _evaluate_ctf(self, xp):
        kx, ky = spatial_frequencies(self.grid.gpts, self.grid.sampling)
        alpha, phi = polar_coordinates(xp.asarray(kx * self.wavelength), xp.asarray(ky * self.wavelength))
        return self._ctf.evaluate(alpha, phi)

    def build(self, positions: Sequence[Sequence[float]] = None) -> Waves:
        """
        Build probe wave functions at the provided positions.

        Parameters
        ----------
        positions : array of xy-positions
            Positions of the probe wave functions

        Returns
        -------
        Waves object
            Probe wave functions as a Waves object.
        """

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        xp = get_array_module_from_device(self._device)
        fft2 = get_device_function(xp, 'fft2')

        if positions is None:
            positions = xp.array((self.extent[0] / 2, self.extent[1] / 2), dtype=xp.float32)
        else:
            positions = xp.array(positions, dtype=xp.float32)

        if len(positions.shape) == 1:
            positions = xp.expand_dims(positions, axis=0)

        array = fft2(self._evaluate_ctf(xp) * self._fourier_translation_operator(positions), overwrite_x=True)

        return Waves(array, extent=self.extent, energy=self.energy)

    def multislice(self, positions: Sequence[Sequence[float]], potential: AbstractPotential, pbar=True) -> Waves:
        """
        Build probe wave functions at the provided positions and propagate them through the potential.

        Parameters
        ----------
        positions : array of xy-positions
            Positions of the probe wave functions.
        potential : Potential or Atoms object
            The scattering potential.
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
        return _multislice(self.build(positions), potential, None, pbar)

    def _generate_probes(self, scan: AbstractScan, potential: Union[AbstractPotential, Atoms], max_batch: int):
        if not isinstance(max_batch, int):
            memory_per_wave = 2 * 4 * np.prod(self.gpts)
            available_memory = get_available_memory(self._device)
            max_batch = min(int(available_memory * .4 / memory_per_wave), 32)

        for indices, positions in scan.generate_positions(max_batch=max_batch):
            yield indices, self.multislice(positions, potential, pbar=False)

    def _generate_tds_probes(self, scan, potential, max_batch, pbar):
        tds_bar = ProgressBar(total=len(potential.frozen_phonons), desc='TDS',
                              disable=(not pbar) or (len(potential.frozen_phonons) == 1))
        potential_pbar = ProgressBar(total=len(potential), desc='Potential', disable=not pbar)

        for potential_config in potential.generate_frozen_phonon_potentials(pbar=potential_pbar):
            yield self._generate_probes(scan, potential_config, max_batch)
            tds_bar.update(1)

        potential_pbar.close()
        tds_bar.refresh()
        tds_bar.close()

    def scan(self,
             scan: AbstractScan,
             detectors: Union[AbstractDetector, Sequence[AbstractDetector]],
             potential: Union[Atoms, AbstractPotential],
             max_batch: int = None,
             pbar: bool = True) -> dict:

        """
        Raster scan the probe across the potential and record a measurement for each detector.

        Parameters
        ----------
        scan : Scan object
            Scan object defining the positions of the probe wave functions.
        detectors : List of detector objects
            The detectors recording the measurements.
        potential : Potential
            The potential to scan the probe over.
        max_batch : int, optional
            The probe batch size. Larger batches are faster, but require more memory. Default is None.
        pbar : bool, optional
            Display progress bars. Default is True.

        Returns
        -------
        dict
            Dictionary of measurements with keys given by the detector.
        """

        self.grid.match(potential.grid)
        self.grid.check_is_defined()

        if isinstance(detectors, AbstractDetector):
            detectors = [detectors]

        scan_bar = ProgressBar(total=len(scan), desc='Scan', disable=not pbar)

        if isinstance(potential, AbstractTDSPotentialBuilder):
            probe_generators = self._generate_tds_probes(scan, potential, max_batch, pbar)
        else:
            if isinstance(potential, AbstractPotentialBuilder):
                potential = potential.build(pbar=True)

            probe_generators = [self._generate_probes(scan, potential, max_batch)]

        measurements = {}
        for probe_generator in probe_generators:
            scan_bar.reset()
            for indices, exit_probes in probe_generator:

                for detector in detectors:
                    new_measurement = detector.detect(exit_probes)

                    if isinstance(potential, AbstractTDSPotentialBuilder):
                        new_measurement /= len(potential.frozen_phonons)

                    try:
                        scan.insert_new_measurement(measurements[detector], indices, new_measurement)
                    except KeyError:
                        measurements[detector] = detector.allocate_measurement(exit_probes, scan)
                        scan.insert_new_measurement(measurements[detector], indices, new_measurement)

                scan_bar.update(len(indices))

            scan_bar.refresh()

        scan_bar.close()
        return measurements

    def profile(self, angle=0.):
        measurement = self.build((self.extent[0] / 2, self.extent[1] / 2)).intensity()
        return probe_profile(measurement, angle=angle)

    def interact(self, sliders=None, profile=False):
        from abtem.visualize.bqplot import show_measurement_1d, show_measurement_2d
        from abtem.visualize.widgets import quick_sliders
        import ipywidgets as widgets

        if profile:
            figure, callback = show_measurement_1d(lambda: [self.profile()])
        else:
            figure, callback = show_measurement_2d(lambda: self.build().intensity())

        self.changed.register(callback)

        if sliders:
            sliders = quick_sliders(self.ctf, **sliders)
            return widgets.HBox([figure, widgets.VBox(sliders)])
        else:
            return figure

    def show(self, **kwargs):
        """
        Show the probe wave function.

        Parameters
        ----------
        angle : float, optional
            Angle along which the profile is shown [deg]. Default is 0 degrees.
        kwargs : Additional keyword arguments for the abtem.plot.show_image function.
        """
        return self.build((self.extent[0] / 2, self.extent[1] / 2)).intensity().show(**kwargs)


class SMatrixArray(HasGridAndAcceleratorMixin, HasDeviceMixin):
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
    interpolation : one or two int
        Interpolation factor.
    k : 2d array
        The spatial frequencies of each plane in the plane wave expansion.
    ctf : CTF object, optional
        The probe contrast transfer function. Default is None.
    extent : one or two float, optional
        Lateral extent of wave functions [Å]. Default is None (inherits extent from the potential).
    sampling : one or two float, optional
        Lateral sampling of wave functions [1 / Å]. Default is None (inherits sampling from the potential).
    device : str, optional
        The calculations will be carried out on this device. Default is 'cpu'.
    """

    def __init__(self,
                 array: np.ndarray,
                 expansion_cutoff: float,
                 energy: float,
                 k: np.ndarray,
                 interpolation: int = None,
                 ctf: CTF = None,
                 sampling: Union[float, Sequence[float]] = None,
                 extent: Union[float, Sequence[float]] = None,
                 periodic: bool = True,
                 offset: Sequence[float] = None,
                 cropped_shape: Tuple[int, int] = None,
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
        self.changed = Event()

        self._ctf.changed.register(self.changed.notify)
        self._grid.changed.register(self.changed.notify)
        self._accelerator.changed.register(self.changed.notify)

        self._device = device

        self._array = array
        self._expansion_cutoff = expansion_cutoff
        self._k = k

        self._periodic = periodic

        if offset is None:
            offset = (0, 0)

        self._offset = np.array(offset, dtype=np.int)

        if (cropped_shape is None) & (interpolation is not None):
            cropped_shape = (self.gpts[0] // interpolation, self.gpts[1] // interpolation)

        self._cropped_shape = cropped_shape

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
    def expansion_cutoff(self) -> float:
        """Expansion cutoff."""
        return self._expansion_cutoff

    @property
    def periodic(self):
        return self._periodic

    @property
    def cropped_grid(self):
        return Grid(gpts=self.cropped_shape, sampling=self.sampling, lock_gpts=True)

    @property
    def cropped_shape(self) -> Tuple[int, int]:
        """The grid of the interpolated scattering matrix."""
        return self._cropped_shape

    @property
    def offset(self):
        return self._offset

    def __len__(self) -> int:
        """Number of plane waves in expansion."""
        return len(self._array)

    def _raise_not_periodic(self):
        raise RuntimeError('not implemented for non-periodic/cropped scattering matrices')

    def downsample(self, max_angle='limit', normalization=None):
        if not self.periodic:
            self._raise_not_periodic()

        xp = get_array_module(self.array)

        if max_angle in ('valid', 'limit'):
            antialias_filter = AntialiasFilter()
            new_gpts = antialias_filter._cropped_gpts(self.gpts, self.sampling, max_angle)

        elif isinstance(max_angle, Number):
            new_gpts = self._resampled_gpts(max_angle)

        else:
            raise ValueError('max_angle must be "valid", "limit", float or None')

        new_array = xp.zeros((len(self.array),) + new_gpts, dtype=self.array.dtype)
        max_batch = self._max_batch_expansion()

        for start, end, partial_s_matrix in self._generate_partial(max_batch, pbar=False):
            new_array[start:end] = fft_interpolate_2d(self.array[start:end], new_gpts, normalization=normalization)

        if self.cropped_shape == self.gpts:
            cropped_shape = new_gpts
        else:
            cropped_shape = tuple(n // (self.gpts[i] // self.cropped_shape[i]) for i, n in enumerate(new_gpts))

        return self.__class__(array=new_array,
                              expansion_cutoff=self._expansion_cutoff,
                              k=self.k.copy(),
                              ctf=self.ctf,
                              extent=self.extent,
                              energy=self.energy,
                              periodic=self.periodic,
                              offset=self._offset,
                              cropped_shape=cropped_shape,
                              device=self.device)

    def crop_to_scan(self, scan):

        if not isinstance(scan, GridScan):
            raise NotImplementedError()

        crop_corner, size = self._get_requisite_crop(np.array([scan.start, scan.end]))
        new_array = periodic_crop(self.array, crop_corner, size)

        return self.__class__(array=new_array,
                              expansion_cutoff=self._expansion_cutoff,
                              interpolation=1,
                              k=self.k.copy(),
                              ctf=self.ctf,
                              sampling=self.sampling,
                              energy=self.energy,
                              periodic=False,
                              offset=crop_corner,
                              cropped_shape=self.cropped_shape,
                              device=self.device)

    def _max_batch_expansion(self):
        memory_per_wave = 2 * 4 * self.gpts[0] * self.gpts[1]
        available_memory = .25 * get_available_memory(self._device)
        return min(int(available_memory / memory_per_wave), len(self))

    def _max_batch_probes(self):
        max_batch_plane_waves = self._max_batch_expansion()
        memory_per_wave = 2 * 4 * self.cropped_shape[0] * self.cropped_shape[1]
        memory_per_plane_wave_batch = 2 * 4 * self.gpts[0] * self.gpts[1] * max_batch_plane_waves
        available_memory = .25 * get_available_memory(self._device) - memory_per_plane_wave_batch
        return max(min(int(available_memory / memory_per_wave), 1024), 1)

    def _generate_partial(self, max_batch: int = None, pbar: Union[ProgressBar, bool] = True):
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

        batch_sizes = subdivide_into_batches(len(self), n_batches)

        xp = get_array_module_from_device(self._device)

        if xp != get_array_module(self.array):
            stream = xp.cuda.Stream(non_blocking=False)
            partial_array = xp.empty((batch_sizes[0],) + self.gpts, dtype=xp.complex64)

        else:
            stream = None
            partial_array = None

        n = 0
        for batch_size in batch_sizes:
            start = n
            end = n + batch_size

            if stream is not None:
                partial_array[:batch_size].set(self._array[start:end], stream=stream)
                yield start, end, Waves(partial_array[:batch_size], extent=self.extent, energy=self.energy)

            else:
                yield start, end, Waves(self._array[start:end], extent=self.extent, energy=self.energy)

            n += batch_size
            pbar.update(batch_size)

        pbar.refresh()
        if close_pbar:
            pbar.close()

    def multislice(self,
                   potential: AbstractPotential,
                   max_batch=None,
                   multislice_pbar: Union[ProgressBar, bool] = True,
                   plane_waves_pbar: Union[ProgressBar, bool] = True):
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

        if not isinstance(max_batch, int):
            max_batch = self._max_batch_expansion()

        propagator = FresnelPropagator()

        storage = get_array_module(self._array)

        if isinstance(multislice_pbar, bool):
            multislice_pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not multislice_pbar)

        for start, end, partial_s_matrix in self._generate_partial(max_batch, pbar=plane_waves_pbar):
            _multislice(partial_s_matrix, potential, propagator=propagator, pbar=multislice_pbar)
            self._array[start: end] = copy_to_device(partial_s_matrix.array, storage)

        # multislice_pbar.close()
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
        return (complex_exponential(2. * np.pi * k[:, 0][None] * positions[:, 0, None]) *
                complex_exponential(2. * np.pi * k[:, 1][None] * positions[:, 1, None]))

    def _get_coefficients(self, positions: Sequence[float]):
        return self._get_translation_coefficients(positions) * self._get_ctf_coefficients()

    def _get_requisite_crop(self, positions: Sequence[float], return_per_position: bool = False):
        offset = (self.cropped_shape[0] // 2, self.cropped_shape[1] // 2)
        corners = np.rint(np.array(positions) / self.sampling - offset).astype(np.int)
        upper_corners = corners + np.asarray(self.cropped_shape)

        crop_corner = (np.min(corners[:, 0]).item(), np.min(corners[:, 1]).item())

        size = (np.max(upper_corners[:, 0]).item() - crop_corner[0],
                np.max(upper_corners[:, 1]).item() - crop_corner[1])

        if return_per_position:
            return crop_corner, size, corners
        else:
            return crop_corner, size

    def collapse(self, positions: Sequence[float], max_batch_expansion: int = None) -> Waves:
        """
        Collapse the scattering matrix to probe wave functions centered on the provided positions.

        Parameters
        ----------
        positions : array of xy-positions
            The positions of the probe wave functions.
        max_batch_expansion : int, optional
            The maximum number of plane waves the reduction is applied to simultanously. Default is None.

        Returns
        -------
        Waves object
            Probe wave functions for the provided positions.
        """

        if max_batch_expansion is None:
            max_batch_expansion = self._max_batch_expansion()

        xp = get_array_module_from_device(self.device)
        positions = np.array(positions, dtype=xp.float32)

        if positions.shape == (2,):
            positions = positions[None]
        elif (len(positions.shape) != 2) or (positions.shape[-1] != 2):
            raise RuntimeError()

        coefficients = self._get_coefficients(positions)

        if self.cropped_shape != self.gpts:
            crop_corner, size, corners = self._get_requisite_crop(positions, return_per_position=True)

            if self._offset is not None:
                corners -= self._offset
                crop_corner = (crop_corner[0] - self._offset[0], crop_corner[1] - self._offset[1])

            array = copy_to_device(periodic_crop(self.array, crop_corner, size), device=self._device)
            window = xp.tensordot(coefficients, array, axes=[(1,), (0,)])

            corners -= crop_corner
            for i in range(len(corners)):
                window[i, :self.cropped_shape[0], :self.cropped_shape[1]] = periodic_crop(window[i], corners[i],
                                                                                          self.cropped_shape)

            window = window[:, :self.cropped_shape[0], :self.cropped_shape[1]].copy()

        elif max_batch_expansion >= len(self):
            for start, end, partial_s_matrix in self._generate_partial(max_batch_expansion, pbar=False):
                partial_coefficients = coefficients[:, start:end]
                window = xp.tensordot(partial_coefficients, partial_s_matrix.array, axes=[(1,), (0,)])

        else:
            window = xp.tensordot(coefficients, self.array, axes=[(1,), (0,)])
            # window = xp.tensordot(coefficients, copy_to_device(self.array, device=self._device), axes=[(1,), (0,)])

        return Waves(window, sampling=self.sampling, energy=self.energy)

    def intensity(self):
        return self.collapse((self.extent[0] / 2, self.extent[1] / 2)).intensity()[0]

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
             max_batch_probes=None,
             max_batch_expansion=None,
             pbar: Union[ProgressBar, bool] = True):

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

        if isinstance(detectors, dict):
            measurements = detectors
        else:
            measurements = {}
            #for detector in detectors:
            #    measurements[detector] = detector.allocate_measurement(self.cropped_grid, self.wavelength, scan)

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(scan), desc='Scan', disable=not pbar)

        for indices, exit_probes in self._generate_probes(scan, max_batch_probes, max_batch_expansion):
            for detector in detectors:
            #for detector, measurement in measurements.items():
                new_measurement = detector.detect(exit_probes)
                try:
                    scan.insert_new_measurement(measurements[detector], indices, new_measurement)
                except KeyError:
                    measurements[detector] = detector.allocate_measurement(exit_probes, scan)
                    scan.insert_new_measurement(measurements[detector], indices, new_measurement)

                #scan.insert_new_measurement(measurement, indices, detector.detect(exit_probes))
            pbar.update(len(indices))

        pbar.refresh()
        pbar.close()
        return measurements

    def transfer(self, device):
        return self.__class__(array=copy_to_device(self.array, device),
                              expansion_cutoff=self._expansion_cutoff,
                              k=self.k.copy(),
                              ctf=self.ctf,
                              extent=self.extent,
                              offset=self.offset,
                              cropped_shape=self.cropped_shape,
                              energy=self.energy,
                              device=self.device)

    def __copy__(self, device=None):
        return self.__class__(array=self.array.copy(),
                              expansion_cutoff=self._expansion_cutoff,
                              k=self.k.copy(),
                              ctf=self.ctf,
                              extent=self.extent,
                              offset=self.offset,
                              cropped_shape=self.cropped_shape,
                              energy=self.energy,
                              device=self.device)

    def copy(self):
        """Make a copy."""
        return copy(self)


class SMatrix(HasGridAndAcceleratorMixin, HasDeviceMixin):
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
    extent : one or two float, optional
        Lateral extent of wave functions [Å]. Default is None (inherits the extent from the potential).
    gpts : one or two int, optional
        Number of grid points describing the wave functions. Default is None (inherits the gpts from the potential).
    sampling : one or two float, None
        Lateral sampling of wave functions [1 / Å]. Default is None (inherits the sampling from the potential.
    device : str, optional
        The calculations will be carried out on this device. Default is 'cpu'.
    storage : str, optional
        The scattering matrix will be stored on this device. Default is None (uses the option chosen for device).
    kwargs :
        The parameters of a new CTF object as keyword arguments.
    """

    def __init__(self,
                 expansion_cutoff: float,
                 energy: float,
                 interpolation: int = 1,
                 ctf: CTF = None,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 device: str = 'cpu',
                 storage: str = None,
                 **kwargs):

        if not isinstance(interpolation, int):
            raise ValueError('Interpolation factor must be int')

        self._interpolation = interpolation
        self._expansion_cutoff = expansion_cutoff

        if ctf is None:
            ctf = CTF(**kwargs)

        if ctf.energy is None:
            ctf.energy = energy

        if (ctf.energy != energy):
            raise RuntimeError

        self._ctf = ctf
        self._accelerator = self._ctf._accelerator

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self.changed = Event()

        self._ctf.changed.register(self.changed.notify)
        self._grid.changed.register(self.changed.notify)
        self._accelerator.changed.register(self.changed.notify)

        self._device = device
        if storage is None:
            storage = device

        self._storage = storage

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
    def interpolated_grid(self) -> Grid:
        """The grid of the interpolated probe wave functions."""
        interpolated_gpts = tuple(n // self.interpolation for n in self.gpts)
        return Grid(gpts=interpolated_gpts, sampling=self.sampling, lock_gpts=True)

    def _generate_tds_probes(self,
                             scan: AbstractScan,
                             potential: AbstractTDSPotentialBuilder,
                             max_batch_probes: int,
                             max_batch_expansion: int,
                             potential_pbar: Union[ProgressBar, bool] = True,
                             multislice_pbar: Union[ProgressBar, bool] = True,
                             plane_waves_pbar: Union[ProgressBar, bool] = True):

        if isinstance(potential_pbar, bool):
            potential_pbar = ProgressBar(total=len(potential), desc='Potential', disable=not potential_pbar)

        if isinstance(multislice_pbar, bool):
            multislice_pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not multislice_pbar)

        if isinstance(plane_waves_pbar, bool):
            plane_waves_pbar = ProgressBar(total=len(self), desc='Plane waves', disable=not plane_waves_pbar)

        for potential_config in potential.generate_frozen_phonon_potentials(pbar=potential_pbar):
            S = self.build().multislice(potential_config,
                                        max_batch=max_batch_expansion,
                                        multislice_pbar=multislice_pbar,
                                        plane_waves_pbar=plane_waves_pbar)

            yield S._generate_probes(scan, max_batch_probes, max_batch_expansion)

        multislice_pbar.refresh()
        multislice_pbar.close()

        plane_waves_pbar.refresh()
        plane_waves_pbar.close()

        potential_pbar.refresh()
        potential_pbar.close()

    def multislice(self,
                   potential: AbstractPotential,
                   max_batch: int = None,
                   pbar: Union[ProgressBar, bool] = True):
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
             max_batch_probes: int = None,
             max_batch_expansion: int = None,
             pbar: bool = True):
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


        #for detector in detectors:
        #    measurements[detector] = detector.allocate_measurement(self.interpolated_grid, self.wavelength, scan)

        if isinstance(potential, AbstractTDSPotentialBuilder):
            probe_generators = self._generate_tds_probes(scan,
                                                         potential,
                                                         max_batch_probes=max_batch_probes,
                                                         max_batch_expansion=max_batch_expansion,
                                                         potential_pbar=True,
                                                         multislice_pbar=True)

        else:
            if isinstance(potential, AbstractPotentialBuilder):
                potential = potential.build(pbar=True)

            S = self.multislice(potential, max_batch=max_batch_expansion, pbar=pbar)
            probe_generators = [S._generate_probes(scan,
                                                   max_batch_probes=max_batch_probes,
                                                   max_batch_expansion=max_batch_expansion)]

        if isinstance(potential, AbstractTDSPotentialBuilder):
            tds_bar = ProgressBar(total=len(potential.frozen_phonons), desc='TDS',
                                  disable=(not pbar) or (len(potential.frozen_phonons) == 1))
        else:
            tds_bar = None

        scan_bar = ProgressBar(total=len(scan), desc='Scan', disable=not pbar)

        measurements = {}
        for probe_generator in probe_generators:
            scan_bar.reset()

            for indices, exit_probes in probe_generator:

                for detector in detectors:
                #for detector, measurement in measurements.items():
                    new_measurement = detector.detect(exit_probes)

                    if isinstance(potential, AbstractTDSPotentialBuilder):
                        new_measurement /= len(potential.frozen_phonons)

                    try:
                        scan.insert_new_measurement(measurements[detector], indices, new_measurement)
                    except KeyError:
                        measurements[detector] = detector.allocate_measurement(exit_probes, scan)
                        scan.insert_new_measurement(measurements[detector], indices, new_measurement)

                    #scan.insert_new_measurement(measurement, indices, new_measurement)

                scan_bar.update(len(indices))

            scan_bar.refresh()
            if tds_bar:
                tds_bar.update(1)

        scan_bar.close()

        if tds_bar:
            tds_bar.refresh()
            tds_bar.close()

        return measurements

    def __len__(self):
        return len(self.k)

    @property
    def k(self):
        xp = get_array_module_from_device(self._device)
        n_max = int(
            xp.ceil(self.expansion_cutoff / 1000. / (self.wavelength / self.extent[0] * self.interpolation)))
        m_max = int(
            xp.ceil(self.expansion_cutoff / 1000. / (self.wavelength / self.extent[1] * self.interpolation)))

        n = xp.arange(-n_max, n_max + 1, dtype=xp.float32)
        w = xp.asarray(self.extent[0], dtype=xp.float32)
        m = xp.arange(-m_max, m_max + 1, dtype=xp.float32)
        h = xp.asarray(self.extent[1], dtype=xp.float32)

        kx = n / w * xp.float32(self.interpolation)
        ky = m / h * xp.float32(self.interpolation)

        mask = kx[:, None] ** 2 + ky[None, :] ** 2 < (self.expansion_cutoff / 1000. / self.wavelength) ** 2
        kx, ky = xp.meshgrid(kx, ky, indexing='ij')
        kx = kx[mask]
        ky = ky[mask]
        return xp.asarray((kx, ky)).T

    def build(self) -> SMatrixArray:
        """Build the scattering matrix."""
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        xp = get_array_module_from_device(self._device)
        storage_xp = get_array_module_from_device(self._storage)
        complex_exponential = get_device_function(xp, 'complex_exponential')

        x = xp.linspace(0, self.extent[0], self.gpts[0], endpoint=self.grid.endpoint[0], dtype=xp.float32)
        y = xp.linspace(0, self.extent[1], self.gpts[1], endpoint=self.grid.endpoint[1], dtype=xp.float32)

        k = self.k

        shape = (len(k),) + self.gpts
        array = storage_xp.zeros(shape, dtype=np.complex64)

        for i in range(len(k)):
            array[i] = copy_to_device(complex_exponential(-2 * np.pi * k[i, 0, None, None] * x[:, None]) *
                                      complex_exponential(-2 * np.pi * k[i, 1, None, None] * y[None, :]),
                                      self._storage)

        return SMatrixArray(array,
                            expansion_cutoff=self.expansion_cutoff,
                            interpolation=self.interpolation,
                            extent=self.extent,
                            energy=self.energy,
                            k=k,
                            ctf=self.ctf,
                            device=self._device)

    def profile(self, angle=0.):
        measurement = self.build().collapse((self.extent[0] / 2, self.extent[1] / 2)).intensity()
        return probe_profile(measurement, angle=angle)

    def interact(self, sliders=None, profile=False):
        from abtem.visualize.bqplot import show_measurement_1d, show_measurement_2d
        from abtem.visualize.widgets import quick_sliders
        import ipywidgets as widgets

        if profile:
            figure, callback = show_measurement_1d(lambda: [self.profile()])
        else:
            figure, callback = show_measurement_2d(lambda: self.build().collapse((self.extent[0] / 2,
                                                                                  self.extent[1] / 2)).intensity())

        self.changed.register(callback)

        if sliders:
            sliders = quick_sliders(self.ctf, **sliders)
            return widgets.HBox([figure, widgets.VBox(sliders)])
        else:
            return figure

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

    def __copy__(self):
        return self.__class__(expansion_cutoff=self.expansion_cutoff,
                              interpolation=self.interpolation,
                              ctf=copy(self.ctf),
                              extent=self.extent,
                              gpts=self.gpts,
                              energy=self.energy,
                              device=self._device,
                              storage=self._storage)

    def copy(self):
        """Make a copy."""
        return copy(self)
