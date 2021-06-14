"""Module to describe electron waves and their propagation."""
from copy import copy
from typing import Union, Sequence, Tuple, List

import dask.array as da
import h5py
import numpy as np
import xarray as xr
from ase import Atoms

from abtem.base_classes import Grid, Accelerator, cache_clear_callback, Cache, cached_method, HasEventMixin, Event, \
    BeamTilt, AntialiasAperture
from abtem.detect import AbstractDetector
from abtem.device import get_array_module, get_device_function, asnumpy, get_array_module_from_device, \
    get_available_memory, get_device_from_array
from abtem.measure.old_measure import calibrations_from_grid, Measurement, block_zeroth_order_spot, probe_profile
from abtem.potentials import Potential, AbstractPotential
from abtem.scan import AbstractScan
from abtem.transfer import CTF
from abtem.utils import ProgressBar, fft_crop, fourier_translation_operator, fft
from abtem.utils.complex import abs2
from abtem.waves.base import _WavesLike, _Scanable
from abtem.waves.multislice import FresnelPropagator, _multislice


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
                 tilt: Tuple[float, float] = None,
                 antialias_aperture: Tuple[float, float] = (2 / 3., 2 / 3.)):

        if len(array.shape) < 2:
            raise RuntimeError('Wave function array should be have 2 dimensions or more')

        self._array = array
        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_aperture = AntialiasAperture(antialias_aperture=antialias_aperture)
        self._device = get_device_from_array(self._array)

    def compute(self):
        return self.__class__(self._array.compute(), extent=self.extent, energy=self.energy, tilt=self.tilt,
                              antialias_aperture=self.antialias_aperture)

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
        coords = [np.arange(len(self))] + self.grid.coords()
        measurement = xr.DataArray(abs2(self.array), coords=coords, dims=['batch', 'x', 'y'], name='Intensity',
                                   attrs={'units': 'arb. units'})
        return measurement

        # calibrations = calibrations_from_grid(self.grid.gpts, self.grid.sampling, ['x', 'y'])
        # array = self.array
        # calibrations = (None,) * (len(array.shape) - 2) + calibrations
        # return Measurement(abs2(array), calibrations)

    def downsample(self, max_angle='valid', return_fourier_space: bool = False) -> 'Waves':
        xp = get_array_module(self.array)
        fft2 = get_device_function(xp, 'fft2')
        ifft2 = get_device_function(xp, 'ifft2')
        array = fft2(self.array, overwrite_x=False)

        gpts = self.downsampled_gpts(max_angle)

        if gpts != self.gpts:
            array = fft_crop(array, self.array.shape[:-2] + gpts)

        antialias_aperture = (self.antialias_aperture[0] * self.gpts[0] / gpts[0],
                              self.antialias_aperture[1] * self.gpts[1] / gpts[1])

        if return_fourier_space:
            return Waves(array, extent=self.extent, energy=self.energy, antialias_aperture=antialias_aperture)
        else:
            return Waves(ifft2(array), extent=self.extent, energy=self.energy, antialias_aperture=antialias_aperture)

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

    def allocate_measurement(self, fourier_space=False) -> Measurement:
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

    def apply_ctf(self, ctf: CTF = None, in_place=False, **kwargs) -> 'Waves':
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
                   max_batch_potential: int = 1) -> 'Waves':
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

        if isinstance(potential, Atoms):
            potential = Potential(potential)

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
                                        antialias_aperture=(2 / 3.,) * 2)

            tds_pbar = ProgressBar(total=n, desc='TDS', disable=(not pbar) or (n == 1))
            multislice_pbar = ProgressBar(total=len(potential), desc='Multislice', disable=not pbar)

            for i, potential_config in enumerate(potential.generate_frozen_phonon_potentials(pbar=pbar)):
                multislice_pbar.reset()

                exit_waves = _multislice(copy(self),
                                         potential_config,
                                         propagator=propagator)

                if detector:
                    result._array += asnumpy(detector.detect(exit_waves)) / n
                else:
                    result._array[i] = xp.squeeze(exit_waves.array)

                tds_pbar.update(1)

            multislice_pbar.close()
            tds_pbar.close()

        if result is None:
            exit_wave = _multislice(self, potential, propagator)

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
                                  antialias_aperture=self.antialias_aperture)
        new_copy._grid = copy(self.grid)
        new_copy._accelerator = copy(self.accelerator)
        return new_copy

    def copy(self) -> 'Waves':
        """Make a copy."""
        return copy(self)


class PlaneWave(_WavesLike):
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
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_aperture = AntialiasAperture()
        self._device = device

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
        array = da.ones((self.gpts[0], self.gpts[1]), dtype=xp.complex64)

        # array = array / np.sqrt(np.prod(array.shape))
        return Waves(array, extent=self.extent, energy=self.energy)

    def __copy__(self) -> 'PlaneWave':
        return self.__class__(extent=self.extent, gpts=self.gpts, sampling=self.sampling, energy=self.energy)

    def copy(self):
        """Make a copy."""
        return copy(self)


class Probe(_Scanable, HasEventMixin):
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
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._antialias_aperture = AntialiasAperture()
        self._event = Event()

        self._ctf.observe(self.event.notify)
        self._grid.observe(self.event.notify)
        self._accelerator.observe(self.event.notify)

        self._device = device
        self._ctf_cache = Cache(1)
        self.observe(cache_clear_callback(self._ctf_cache))

    @property
    def ctf(self) -> CTF:
        """Probe contrast transfer function."""
        return self._ctf

    def _fourier_translation_operator(self, positions):
        xp = get_array_module(positions)
        positions /= xp.array(self.sampling).astype(np.float32)
        return fourier_translation_operator(positions, self.gpts)

    @cached_method('_ctf_cache')
    def _evaluate_ctf(self):
        alpha, phi = self.get_scattering_angles()
        return da.array(self._ctf.evaluate(alpha, phi))

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

        positions = self._validate_positions(positions)

        array = fft.ifft2(self._evaluate_ctf() * self._fourier_translation_operator(positions))
        array = array / xp.sqrt((xp.abs(array[0]) ** 2).sum()) / xp.sqrt(np.prod(array.shape[1:]))

        return Waves(array, extent=self.extent, energy=self.energy, tilt=self.tilt,
                     antialias_aperture=self.antialias_aperture)

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
             chunk_size: int = None,
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

        positions = da.from_array(scan.get_positions(), chunks=(chunk_size, 2))
        exit_probes = self.multislice(positions, potential)

        measurements = []

        for i, potential_config in enumerate(potential.generate_frozen_phonon_potentials()):
            for detector in detectors:
                if i == 0:
                    measurement = detector.detect(exit_probes, scan) / potential.num_frozen_phonon_configs
                    measurements.append(measurement)
                else:
                    measurements[i] += detector.detect(exit_probes, scan) / potential.num_frozen_phonon_configs

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
