"""Module to describe electron waves and their propagation."""
from copy import copy
from numbers import Number
from typing import Union, Sequence, Tuple, List, Dict

import h5py
import numpy as np
from ase import Atoms

from abtem.base_classes import Grid, Accelerator, cache_clear_callback, Cache, cached_method, \
    HasGridMixin, HasAcceleratorMixin, HasEventMixin, AntialiasFilter, Event
from abtem.detect import AbstractDetector
from abtem.device import get_array_module, get_device_function, asnumpy, get_array_module_from_device, \
    copy_to_device, get_available_memory, HasDeviceMixin, get_device_from_array
from abtem.measure import calibrations_from_grid, Measurement, block_zeroth_order_spot, probe_profile
from abtem.potentials import Potential, AbstractPotential, AbstractPotentialBuilder
from abtem.scan import AbstractScan, GridScan
from abtem.transfer import CTF
from abtem.utils import polar_coordinates, ProgressBar, spatial_frequencies, subdivide_into_batches, periodic_crop, \
    fft_crop, fourier_translation_operator


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
                                   tilt: Tuple[float, float],
                                   xp) -> np.ndarray:
        complex_exponential = get_device_function(xp, 'complex_exponential')
        kx = xp.fft.fftfreq(gpts[0], sampling[0]).astype(xp.float32)
        ky = xp.fft.fftfreq(gpts[1], sampling[1]).astype(xp.float32)
        f = (complex_exponential(-(kx ** 2)[:, None] * np.pi * wavelength * dz) *
             complex_exponential(-(ky ** 2)[None] * np.pi * wavelength * dz))

        if tilt is not None:
            # TODO : this is specimen tilt, beam tilt really be independent
            f *= (complex_exponential(-kx[:, None] * xp.tan(tilt[0] / 1e3) * dz * 2 * np.pi) *
                  complex_exponential(-ky[None] * xp.tan(tilt[1] / 1e3) * dz * 2 * np.pi))

        return f * AntialiasFilter().get_mask(gpts, sampling, xp)

    def propagate(self,
                  waves: Union['Waves', 'SMatrixArray'],
                  dz: float,
                  in_place: bool = True) -> Union['Waves', 'SMatrixArray']:
        """
        Propagate wave functions or scattering matrix.

        Parameters
        ----------
        waves : Waves or SMatrixArray object
            Wave function or scattering matrix to propagate.
        dz : float
            Propagation distance [Å].
        in_place : bool, optional
            If True the wavefunction array will be modified in place. Default is True.

        Returns
        -------
        Waves or SMatrixArray object
            The propagated wave functions.
        """
        if not in_place:
            waves = waves.copy()

        fft2_convolve = get_device_function(get_array_module(waves.array), 'fft2_convolve')

        propagator_array = self._evaluate_propagator_array(waves.grid.gpts,
                                                           waves.grid.sampling,
                                                           waves.wavelength,
                                                           dz,
                                                           waves.tilt,
                                                           get_array_module(waves.array))

        fft2_convolve(waves._array, propagator_array, overwrite_x=True)
        waves._antialiasing_aperture = (2 / 3.,) * 2
        return waves


def _multislice(waves: Union['Waves', 'SMatrixArray'],
                potential: AbstractPotential,
                propagator: FresnelPropagator = None,
                pbar: Union[ProgressBar, bool] = True,
                max_batch: int = 1,
                ) -> Union['Waves', 'SMatrixArray']:
    waves.grid.match(potential)
    waves.accelerator.check_is_defined()
    waves.grid.check_is_defined()

    if propagator is None:
        propagator = FresnelPropagator()

    if isinstance(pbar, bool):
        pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not pbar)
        close_pbar = True
    else:
        close_pbar = False

    pbar.reset()
    if max_batch == 1:
        for start, end, t in potential.generate_transmission_functions(energy=waves.energy, max_batch=1):
            waves = t.transmit(waves)
            waves = propagator.propagate(waves, t.thickness)
            pbar.update(1)
    else:
        for start, end, t_chunk in potential.generate_transmission_functions(energy=waves.energy, max_batch=max_batch):
            for _, __, t_slice in t_chunk.generate_slices(max_batch=1):
                waves = t_slice.transmit(waves)
                waves = propagator.propagate(waves, t_slice.thickness)

            pbar.update(end - start)

    pbar.refresh()
    if close_pbar:
        pbar.close()

    return waves


class _WavesLike(HasGridMixin, HasAcceleratorMixin, HasDeviceMixin):

    def __init__(self, tilt: Tuple[float, float] = None, antialiasing_aperture: float = None):
        self._tilt = tilt
        if antialiasing_aperture is None:
            antialiasing_aperture = (AntialiasFilter.cutoff,) * 2

        self._antialiasing_aperture = antialiasing_aperture

    @property
    def tilt(self):
        return self._tilt

    @tilt.setter
    def tilt(self, value):
        self._tilt = value

    @property
    def antialiasing_aperture(self):
        return self._antialiasing_aperture

    @property
    def cutoff_scattering_angles(self) -> Tuple[float, float]:
        interpolated_grid = self._interpolated_grid
        # print(self.antialiasing_aperture)
        kcut = [1 / d / 2 * a for d, a in zip(interpolated_grid.sampling, self.antialiasing_aperture)]
        kcut = min(kcut)
        kcut = (
            np.ceil(2 * interpolated_grid.extent[0] * kcut) / (
                    2 * interpolated_grid.extent[0]) * self.wavelength * 1e3,
            np.ceil(2 * interpolated_grid.extent[1] * kcut) / (
                    2 * interpolated_grid.extent[1]) * self.wavelength * 1e3)
        return kcut

    @property
    def rectangle_cutoff_scattering_angles(self) -> Tuple[float, float]:
        rolloff = AntialiasFilter.rolloff
        interpolated_grid = self._interpolated_grid
        # print(self.antialiasing_aperture / max(interpolated_grid.sampling))
        # print(interpolated_grid.sampling)
        kcut = [(a / (d * 2) - rolloff) / np.sqrt(2) for d, a in
                zip(interpolated_grid.sampling, self.antialiasing_aperture)]

        # print(self.antialiasing_aperture / max(interpolated_grid.sampling))
        # print(self.antialiasing_aperture / max(interpolated_grid.sampling))
        kcut = min(kcut)
        kcut = (
            np.floor(2 * interpolated_grid.extent[0] * kcut) / (
                    2 * interpolated_grid.extent[0]) * self.wavelength * 1e3,
            np.floor(2 * interpolated_grid.extent[1] * kcut) / (
                    2 * interpolated_grid.extent[1]) * self.wavelength * 1e3)
        return kcut

    @property
    def angular_sampling(self):
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        return tuple([1 / l * self.wavelength * 1e3 for l in self._interpolated_grid.extent])

    def get_spatial_frequencies(self):
        xp = get_array_module_from_device(self.device)
        kx, ky = spatial_frequencies(self.grid.gpts, self.grid.sampling)
        # TODO : should beam tilt be added here?
        kx = xp.asarray(kx)
        ky = xp.asarray(ky)
        return kx, ky

    def get_scattering_angles(self):
        kx, ky = self.get_spatial_frequencies()
        alpha, phi = polar_coordinates(kx * self.wavelength, ky * self.wavelength)
        return alpha, phi

    @property
    def _interpolated_grid(self):
        return self.grid

    def downsampled_gpts(self, max_angle: Union[float, str]):
        interpolated_grid = self._interpolated_grid

        if max_angle is None:
            gpts = interpolated_grid.gpts

        elif isinstance(max_angle, str):
            if max_angle == 'limit':
                cutoff_scattering_angle = self.cutoff_scattering_angles
            elif max_angle == 'valid':
                cutoff_scattering_angle = self.rectangle_cutoff_scattering_angles
            else:
                raise RuntimeError()

            angular_sampling = self.angular_sampling
            gpts = (int(round(cutoff_scattering_angle[0] / angular_sampling[0] * 2)),
                    int(round(cutoff_scattering_angle[1] / angular_sampling[1] * 2)))

        elif isinstance(max_angle, Number):
            gpts = [int(2 * np.floor(max_angle / d)) + 1 for n, d in zip(interpolated_grid.gpts, self.angular_sampling)]
        else:
            raise RuntimeError()

        return (min(gpts[0], interpolated_grid.gpts[0]), min(gpts[1], interpolated_grid.gpts[1]))


class _Scanable(_WavesLike):

    def _validate_detectors(self, detectors):
        if isinstance(detectors, AbstractDetector):
            detectors = [detectors]
        return detectors

    def _validate_scan_measurements(self, detectors, scan, measurements=None):

        if isinstance(measurements, Measurement):
            if len(detectors) > 1:
                raise ValueError('more than one detector, measurements must be mapping or None')

            return {detectors[0]: measurements}

        if measurements is None:
            measurements = {}

        for detector in detectors:
            if detector not in measurements.keys():
                measurements[detector] = detector.allocate_measurement(self, scan)
            # if not set(measurements.keys()) == set(detectors):
            #    raise ValueError('measurements dict keys does not match detectors')
        # else:
        #    raise ValueError('measurements must be Measurement or dict of AbtractDetector: Measurement')
        return measurements


class Waves(_WavesLike):
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
    tilt : two float
        Small angle beam tilt [mrad].
    antialiasing_aperture : float
        Assumed antialiasing aperture as a fraction of the real space Nyquist frequency. Default is 2/3.
    """

    def __init__(self,
                 array: np.ndarray,
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 tilt: Sequence[float] = None,
                 antialiasing_aperture: Tuple[float, float] = None):

        if len(array.shape) < 2:
            raise RuntimeError('Wave function array should be have 2 dimensions or more')

        self._array = array

        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)
        self._device = get_device_from_array(self._array)

        super().__init__(tilt=tilt, antialiasing_aperture=antialiasing_aperture)

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
        array = self.array

        calibrations = (None,) * (len(array.shape) - 2) + calibrations
        abs2 = get_device_function(get_array_module(self.array), 'abs2')
        return Measurement(abs2(array), calibrations)

    def downsample(self, max_angle='valid', return_fourier_space=False):
        xp = get_array_module(self.array)
        fft2 = get_device_function(xp, 'fft2')
        ifft2 = get_device_function(xp, 'ifft2')
        array = fft2(self.array, overwrite_x=False)

        gpts = self.downsampled_gpts(max_angle)

        if gpts != self.gpts:
            array = fft_crop(array, self.array.shape[:-2] + gpts)

        # if max_angle in ('valid', 'limit'):
        #    antialiasing_aperture = 1.
        # else:
        antialiasing_aperture = (self.antialiasing_aperture[0] * self.gpts[0] / gpts[0],
                                 self.antialiasing_aperture[1] * self.gpts[1] / gpts[1])

        # print(antialiasing_aperture)

        if return_fourier_space:
            return Waves(array, extent=self.extent, energy=self.energy, antialiasing_aperture=antialiasing_aperture)
        else:
            return Waves(ifft2(array), extent=self.extent, energy=self.energy,
                         antialiasing_aperture=antialiasing_aperture)

    def far_field(self, max_angle='valid'):
        return self.downsample(max_angle=max_angle, return_fourier_space=True)

    def diffraction_pattern(self, max_angle='valid', block_zeroth_order=False) -> Measurement:
        """
        Calculate the intensity of the wave functions at the diffraction plane.

        Returns
        -------
        Measurement object
            The intensity of the diffraction pattern(s).
        """
        xp = get_array_module(self.array)
        abs2 = get_device_function(xp, 'abs2')
        waves = self.far_field(max_angle)

        pattern = np.fft.fftshift(asnumpy(abs2(waves.array)), axes=(-1, -2))

        calibrations = calibrations_from_grid(waves.gpts,
                                              waves.sampling,
                                              names=['alpha_x', 'alpha_y'],
                                              units='mrad',
                                              scale_factor=self.wavelength * 1000,
                                              fourier_space=True)

        calibrations = (None,) * (len(pattern.shape) - 2) + calibrations

        measurement = Measurement(pattern, calibrations)

        if block_zeroth_order:
            block_zeroth_order_spot(measurement, block_zeroth_order)

        return measurement

    def allocate_measurement(self, fourier_space=False):
        """
        Allocate a measurement object

        Parameters
        ----------
        fourier_space

        Returns
        -------

        """
        calibrations = calibrations_from_grid(self.grid.gpts, self.grid.sampling, ['x', 'y'])
        calibrations = (None,) * (len(self.array.shape) - 2) + calibrations
        array = np.zeros_like(self.array, dtype=np.float32)
        return Measurement(array, calibrations)

    def apply_ctf(self, ctf: CTF = None, in_place=False, **kwargs):
        """
        Apply the aberrations defined by a CTF object to wave function.

        Parameters
        ----------
        ctf : CTF
            Contrast Transfer Function object to be applied.
        kwargs :
            Provide the parameters of the contrast transfer function as keyword arguments. See the documentation for the
            CTF object.

        Returns
        -------
        Waves object
            The wave functions with aberrations applied.
        """

        fft2_convolve = get_device_function(get_array_module(self.array), 'fft2_convolve')

        if ctf is None:
            ctf = CTF(**kwargs)

        if not ctf.accelerator.energy:
            ctf.accelerator.match(self.accelerator)

        self.accelerator.match(ctf.accelerator, check_match=True)

        self.accelerator.check_is_defined()
        self.grid.check_is_defined()

        alpha, phi = self.get_scattering_angles()
        kernel = ctf.evaluate(alpha, phi)

        return self.__class__(fft2_convolve(self.array, kernel, overwrite_x=in_place),
                              extent=self.extent,
                              energy=self.energy,
                              tilt=self.tilt)

    def multislice(self,
                   potential: AbstractPotential,
                   pbar: Union[ProgressBar, bool] = True,
                   detector=None,
                   max_batch_potential=1) -> 'Waves':
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

        result = None

        if potential.num_frozen_phonon_configs > 1:
            xp = get_array_module(self.array)
            n = potential.num_frozen_phonon_configs

            if detector:
                result = detector.allocate_measurement(self, self.array.shape[:-2])
            else:
                if n > 1:
                    if self.array.shape[0] == 1:
                        out_array = xp.zeros((n,) + self.array.shape[1:], dtype=xp.complex64)
                    else:
                        out_array = xp.zeros((n,) + self.array.shape, dtype=xp.complex64)
                else:
                    out_array = xp.zeros(self.array.shape, dtype=xp.complex64)

                result = self.__class__(out_array,
                                        extent=self.extent,
                                        energy=self.energy,
                                        antialiasing_aperture=(2 / 3.,) * 2)

            tds_pbar = ProgressBar(total=n, desc='TDS', disable=(not pbar) or (n == 1))
            multislice_pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not pbar)

            for i, potential_config in enumerate(potential.generate_frozen_phonon_potentials(pbar=pbar)):
                multislice_pbar.reset()

                exit_waves = _multislice(copy(self),
                                         potential_config,
                                         propagator=propagator,
                                         pbar=multislice_pbar,
                                         max_batch=max_batch_potential)

                if detector:
                    result._array += asnumpy(detector.detect(exit_waves)) / n
                else:
                    result._array[i] = xp.squeeze(exit_waves.array)

                tds_pbar.update(1)

            multislice_pbar.close()
            tds_pbar.close()

        if result is None:
            if isinstance(potential, AbstractPotentialBuilder):
                if potential.precalculate:
                    potential = potential.build(pbar=pbar)

            exit_wave = _multislice(self, potential, propagator, pbar, max_batch=max_batch_potential)

            if detector:
                result = detector.allocate_measurement(self, self.array.shape[:-2])
                result._array = asnumpy(detector.detect(exit_wave))
            else:
                result = exit_wave

        return result

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
        return self.intensity().show(ax=ax, **kwargs)

    def __copy__(self) -> 'Waves':
        new_copy = self.__class__(array=self._array.copy(), tilt=self.tilt,
                                  antialiasing_aperture=self.antialiasing_aperture)
        new_copy._grid = copy(self.grid)
        new_copy._accelerator = copy(self.accelerator)
        return new_copy

    def copy(self) -> 'Waves':
        """Make a copy."""
        return copy(self)


class PlaneWave(_WavesLike, HasDeviceMixin):
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
    tilt : two floats
        Small angle beam tilt [mrad].
    device : str
        The plane waves will be build on this device.
    """

    def __init__(self,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 energy: float = None,
                 tilt: Tuple[float, float] = None,
                 device: str = 'cpu'):
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)
        self._device = device

        super().__init__(tilt=tilt)

    def multislice(self,
                   potential: Union[AbstractPotential, Atoms],
                   pbar: bool = True,
                   max_batch_potential: int = 1) -> Waves:
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

        return self.build().multislice(potential, pbar=pbar, max_batch_potential=max_batch_potential)

    def build(self) -> Waves:
        """Build the plane wave function as a Waves object."""
        xp = get_array_module_from_device(self._device)
        self.grid.check_is_defined()
        array = xp.ones((1, self.gpts[0], self.gpts[1]), dtype=xp.complex64)
        # array = array / np.sqrt(np.prod(array.shape))
        return Waves(array, extent=self.extent, energy=self.energy)

    def __copy__(self, a) -> 'PlaneWave':
        return self.__class__(extent=self.extent, gpts=self.gpts, sampling=self.sampling, energy=self.energy)

    def copy(self):
        """Make a copy."""
        return copy(self)


class Probe(_Scanable, HasDeviceMixin, HasEventMixin):
    """
    Probe wavefunction object

    The probe object can represent a stack of electron probe wavefunctions for simulating scanning transmission
    electron microscopy.

    See the docs of abtem.transfer.CTF for a description of the parameters related to the contrast transfer function.

    Parameters
    ----------
    extent : two float, optional
        Lateral extent of wave functions [Å].
    gpts : two int, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å].
    energy : float, optional
        Electron energy [eV].
    ctf : CTF
        Contrast transfer function object. Note that this can be specified
    device : str
        The probe wave functions will be build on this device.
    kwargs :
        Provide the parameters of the contrast transfer function as keyword arguments. See the documentation for the
        CTF object.
    """

    def __init__(self,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 energy: float = None,
                 ctf: CTF = None,
                 tilt: Tuple[float, float] = None,
                 device: str = 'cpu',
                 **kwargs):

        if ctf is None:
            ctf = CTF(energy=energy, **kwargs)

        if ctf.energy is None:
            ctf.energy = energy

        if ctf.energy != energy:
            raise RuntimeError('CTF energy does match probe energy')

        self._ctf = ctf
        self._accelerator = self._ctf._accelerator

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._event = Event()

        self._ctf.observe(self.event.notify)
        self._grid.observe(self.event.notify)
        self._accelerator.observe(self.event.notify)

        self._device = device
        self._ctf_cache = Cache(1)
        self.observe(cache_clear_callback(self._ctf_cache))
        super().__init__(tilt=tilt)

    @property
    def ctf(self) -> CTF:
        """Probe contrast transfer function."""
        return self._ctf

    def _fourier_translation_operator(self, positions):
        xp = get_array_module(positions)
        positions /= xp.array(self.sampling)
        return fourier_translation_operator(positions, self.gpts)

    @cached_method('_ctf_cache')
    def _evaluate_ctf(self):
        alpha, phi = self.get_scattering_angles()

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
        ifft2 = get_device_function(xp, 'ifft2')

        if positions is None:
            positions = xp.array((self.extent[0] / 2, self.extent[1] / 2), dtype=xp.float32)
        else:
            positions = xp.array(positions, dtype=xp.float32)

        if len(positions.shape) == 1:
            positions = xp.expand_dims(positions, axis=0)

        array = ifft2(self._evaluate_ctf() * self._fourier_translation_operator(positions), overwrite_x=True)

        array = array / xp.sqrt((xp.abs(array[0]) ** 2).sum()) / xp.sqrt(np.prod(array.shape[1:]))

        return Waves(array, extent=self.extent, energy=self.energy, tilt=self.tilt)

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
        exit_probes = _multislice(self.build(positions), potential, None, pbar)
        exit_probes._antialiasing_aperture = (2 / 3.,) * 2
        return exit_probes

    def _estimate_max_batch(self):
        memory_per_wave = 2 * 4 * np.prod(self.gpts)
        available_memory = get_available_memory(self._device)
        return min(int(available_memory * .4 / memory_per_wave), 32)

    def _generate_probes(self, scan, potential, max_batch, pbar):

        potential_pbar = ProgressBar(total=len(potential), desc='Potential',
                                     disable=(not pbar) or (not potential._precalculate))

        tds_bar = ProgressBar(total=potential.num_frozen_phonon_configs, desc='TDS',
                              disable=(not pbar) or (potential.num_frozen_phonon_configs == 1))

        scan_bar = ProgressBar(total=len(scan), desc='Scan', disable=not pbar)

        for potential_config in potential.generate_frozen_phonon_potentials(pbar=potential_pbar):
            scan_bar.reset()

            if max_batch is None:
                max_batch = self._estimate_max_batch()

            for indices, positions in scan.generate_positions(max_batch=max_batch):
                yield indices, self.multislice(positions, potential_config, pbar=False)
                scan_bar.update(len(indices))

            scan_bar.refresh()
            tds_bar.update(1)

        potential_pbar.close()
        potential_pbar.refresh()
        tds_bar.refresh()
        tds_bar.close()
        scan_bar.close()

    def scan(self,
             scan: AbstractScan,
             detectors: Union[AbstractDetector, Sequence[AbstractDetector]],
             potential: Union[Atoms, AbstractPotential],
             measurements: Union[Measurement, Dict[AbstractDetector, Measurement]] = None,
             max_batch: int = None,
             pbar: bool = True) -> Union[Measurement, List[Measurement]]:

        """
        Raster scan the probe across the potential and record a measurement for each detector.

        Parameters
        ----------
        scan : Scan object
            Scan object defining the positions of the probe wave functions.
        detectors : Detector or list of detectors
            The detectors recording the measurements.
        potential : Potential
            The potential to scan the probe over.
        measurements : Measurement or list of measurements
            Diction
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

        detectors = self._validate_detectors(detectors)
        measurements = self._validate_scan_measurements(detectors, scan, measurements)

        for indices, exit_probes in self._generate_probes(scan, potential, max_batch, pbar):
            for detector, measurement in measurements.items():
                new_entries = detector.detect(exit_probes) / potential.num_frozen_phonon_configs
                scan.insert_new_measurement(measurement, indices, new_entries)

        measurements = list(measurements.values())
        if len(measurements) == 1:
            return measurements[0]
        else:
            return measurements

    def profile(self, angle=0.):
        self.grid.check_is_defined()
        measurement = self.build((self.extent[0] / 2, self.extent[1] / 2)).intensity()
        return probe_profile(measurement, angle=angle)

    def interact(self, sliders=None, profile=False, throttling: float = 0.01):
        from abtem.visualize.interactive.utils import quick_sliders, throttle
        from abtem.visualize.interactive import Canvas, MeasurementArtist2d
        from abtem.visualize.interactive.apps import MeasurementView1d
        import ipywidgets as widgets

        if profile:
            view = MeasurementView1d()

            def callback(*args):
                view.measurement = self.profile()
        else:
            canvas = Canvas(lock_scale=False)
            artist = MeasurementArtist2d()
            canvas.artists = {'image': artist}

            def callback(*args):
                artist.measurement = self.build().intensity()[0]
                canvas.adjust_limits_to_artists(adjust_y=False)
                canvas.adjust_labels_to_artists()

        if throttling:
            callback = throttle(throttling)(callback)

        self.observe(callback)
        callback()

        if sliders:
            sliders = quick_sliders(self.ctf, **sliders)
            return widgets.HBox([canvas.figure, widgets.VBox(sliders)])
        else:
            return canvas.figure

    def __copy__(self):
        return self.__class__(gpts=self.gpts,
                              extent=self.extent,
                              sampling=self.sampling,
                              energy=self.energy,
                              ctf=self.ctf.copy(),
                              device=self.device)

    def copy(self):
        return copy(self)

    def show(self, **kwargs):
        """
        Show the probe wave function.

        Parameters
        ----------
        angle : float, optional
            Angle along which the profile is shown [deg]. Default is 0 degrees.
        kwargs : Additional keyword arguments for the abtem.plot.show_image function.
        """
        self.grid.check_is_defined()
        return self.build((self.extent[0] / 2, self.extent[1] / 2)).intensity().show(**kwargs)


class SMatrixArray(_Scanable, HasDeviceMixin, HasEventMixin):
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
    antialiasing_aperture : float, optional
        Assumed antialiasing aperture as a fraction of the real space Nyquist frequency. Default is 2/3.
    device : str, optional
        The calculations will be carried out on this device. Default is 'cpu'.
    """

    def __init__(self,
                 array: np.ndarray,
                 expansion_cutoff: float,
                 energy: float,
                 k: np.ndarray,
                 ctf: CTF = None,
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 tilt: Tuple[float, float] = None,
                 periodic: bool = True,
                 offset: Sequence[float] = None,
                 interpolated_gpts: Tuple[int, int] = None,
                 antialiasing_aperture: Tuple[float, float] = None,
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
        self._event = Event()

        self._ctf.observe(self.event.notify)
        self._grid.event.observe(self.event.notify)
        self._accelerator.event.observe(self.event.notify)

        self._device = device

        self._array = array
        self._antialiasing_aperture = antialiasing_aperture

        self._expansion_cutoff = expansion_cutoff
        self._k = k
        self._periodic = periodic

        if offset is None:
            offset = (0, 0)

        self._offset = np.array(offset, dtype=np.int)
        self._interpolated_gpts = interpolated_gpts

        super().__init__(tilt=tilt, antialiasing_aperture=antialiasing_aperture)

    @property
    def ctf(self) -> CTF:
        """Probe contrast transfer function."""
        return self._ctf

    @property
    def array(self) -> np.ndarray:
        """Array representing the scattering matrix."""
        return self._array

    @property
    def antialiasing_aperture(self):
        return self._antialiasing_aperture

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
    def interpolated_gpts(self) -> Tuple[int, int]:
        """The grid of the interpolated scattering matrix."""
        return self._interpolated_gpts

    @property
    def _interpolated_grid(self):
        return Grid(gpts=self._interpolated_gpts, sampling=self.sampling)

    @property
    def offset(self):
        return self._offset

    def __len__(self) -> int:
        """Number of plane waves in expansion."""
        return len(self._array)

    def _raise_not_periodic(self):
        raise RuntimeError('not implemented for non-periodic/cropped scattering matrices')

    def downsample(self, max_angle='limit'):
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

        antialiasing_aperture = downsampled.antialiasing_aperture

        return self.__class__(array=new_array,
                              expansion_cutoff=self._expansion_cutoff,
                              k=self.k.copy(),
                              ctf=self.ctf,
                              extent=self.extent,
                              energy=self.energy,
                              periodic=self.periodic,
                              offset=self._offset,
                              interpolated_gpts=interpolated_gpts,
                              antialiasing_aperture=antialiasing_aperture,
                              device=self.device)

    def crop_to_scan(self, scan):

        if not isinstance(scan, GridScan):
            raise NotImplementedError()

        crop_corner, size = self._get_requisite_crop(np.array([scan.start, scan.end]))
        new_array = periodic_crop(self.array, crop_corner, size)

        return self.__class__(array=new_array,
                              expansion_cutoff=self._expansion_cutoff,
                              k=self.k.copy(),
                              ctf=self.ctf,
                              sampling=self.sampling,
                              energy=self.energy,
                              periodic=False,
                              offset=crop_corner,
                              interpolated_gpts=self.interpolated_gpts,
                              device=self.device)

    def _max_batch_expansion(self):
        memory_per_wave = 2 * 4 * self.gpts[0] * self.gpts[1]
        available_memory = .2 * get_available_memory(self._device)
        return min(int(available_memory / memory_per_wave), len(self))

    def _max_batch_probes(self):
        max_batch_plane_waves = self._max_batch_expansion()
        memory_per_wave = 2 * 4 * self.interpolated_gpts[0] * self.interpolated_gpts[1]
        memory_per_plane_wave_batch = 2 * 4 * self.gpts[0] * self.gpts[1] * max_batch_plane_waves
        available_memory = .2 * get_available_memory(self._device) - memory_per_plane_wave_batch
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

        xp = get_array_module_from_device(self._device)

        n = 0
        for batch_size in subdivide_into_batches(len(self), n_batches):
            start = n
            end = n + batch_size

            if xp != get_array_module(self.array):
                yield start, end, Waves(copy_to_device(self._array[start:end], xp),
                                        extent=self.extent, energy=self.energy,
                                        antialiasing_aperture=self.antialiasing_aperture)
            else:
                yield start, end, Waves(self._array[start:end], extent=self.extent, energy=self.energy,
                                        antialiasing_aperture=self.antialiasing_aperture)

            n += batch_size
            pbar.update(batch_size)

        pbar.refresh()
        if close_pbar:
            pbar.close()

    def multislice(self,
                   potential: AbstractPotential,
                   max_batch: int = None,
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

        if isinstance(multislice_pbar, bool):
            multislice_pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not multislice_pbar)
            close_pbar = True
        else:
            close_pbar = False

        propagator = FresnelPropagator()

        for start, end, partial_s_matrix in self._generate_partial(max_batch, pbar=plane_waves_pbar):
            _multislice(partial_s_matrix, potential, propagator=propagator, pbar=multislice_pbar)
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
        return (complex_exponential(2. * np.pi * k[:, 0][None] * positions[:, 0, None]) *
                complex_exponential(2. * np.pi * k[:, 1][None] * positions[:, 1, None]))

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

    def collapse(self, positions: Sequence[Sequence[float]] = None, max_batch_expansion: int = None) -> Waves:
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

        if positions is None:
            positions = np.array((self.extent[0] / 2, self.extent[1] / 2), dtype=xp.float32)
        else:
            positions = np.array(positions, dtype=xp.float32)

        if len(positions.shape) == 1:
            positions = np.expand_dims(positions, axis=0)

        if positions.shape[1] != 2:
            raise ValueError()

        coefficients = self._get_coefficients(positions)

        if self.interpolated_gpts != self.gpts:
            crop_corner, size, corners = self._get_requisite_crop(positions, return_per_position=True)

            if self._offset is not None:
                corners -= self._offset
                crop_corner = (crop_corner[0] - self._offset[0], crop_corner[1] - self._offset[1])

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
                     antialiasing_aperture=self.antialiasing_aperture)

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
                              expansion_cutoff=self._expansion_cutoff,
                              k=self.k.copy(),
                              ctf=self.ctf.copy(),
                              extent=self.extent,
                              offset=self.offset,
                              interpolated_gpts=self.interpolated_gpts,
                              energy=self.energy,
                              antialiasing_aperture=self.antialiasing_aperture,
                              device=self.device)

    def __copy__(self, device=None):
        return self.__class__(array=self.array.copy(),
                              expansion_cutoff=self._expansion_cutoff,
                              k=self.k.copy(),
                              ctf=self.ctf.copy(),
                              extent=self.extent,
                              offset=self.offset,
                              interpolated_gpts=self.interpolated_gpts,
                              energy=self.energy,
                              antialiasing_aperture=self.antialiasing_aperture,
                              device=self.device)

    def copy(self):
        """Make a copy."""
        return copy(self)


class SMatrix(_Scanable, HasDeviceMixin, HasEventMixin):
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

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self._event = Event()

        self._ctf.observe(self.event.notify)
        self._grid.observe(self.event.notify)
        self._accelerator.observe(self.event.notify)

        self._device = device

        if storage is None:
            storage = device

        self._storage = storage

        super().__init__(tilt=tilt)

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

        interpolated_gpts = (self.gpts[0] // self.interpolation, self.gpts[1] // self.interpolation)

        probe = (storage_xp.abs(array.sum(0)) ** 2)[:interpolated_gpts[0], :interpolated_gpts[1]]
        array /= storage_xp.sqrt(probe.sum()) * storage_xp.sqrt(interpolated_gpts[0] * interpolated_gpts[1])

        return SMatrixArray(array,
                            expansion_cutoff=self.expansion_cutoff,
                            interpolated_gpts=self.interpolated_gpts,
                            extent=self.extent,
                            energy=self.energy,
                            tilt=self.tilt,
                            k=k,
                            ctf=self.ctf.copy(),
                            device=self._device)

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
