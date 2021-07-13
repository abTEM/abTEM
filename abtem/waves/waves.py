"""Module to describe electron waves and their propagation."""
from copy import copy
from typing import Union, Sequence, Tuple, List

import dask
import dask.array as da
import h5py
import numpy as np
from ase import Atoms

from abtem.basic.antialias import AntialiasAperture
from abtem.basic.axes import HasAxesMetadata
from abtem.basic.backend import get_array_module
from abtem.basic.complex import abs2
from abtem.basic.energy import Accelerator
from abtem.basic.event import HasEventMixin
from abtem.basic.fft import fft2, ifft2, fft2_convolve, fft2_shift_kernel, fft2_crop, fft2_interpolate
from abtem.basic.grid import Grid
from abtem.device import get_array_module_from_device, get_device_from_array
from abtem.measure.detect import AbstractDetector
from abtem.measure.measure import DiffractionPatterns, Images
from abtem.measure.old_measure import Measurement, probe_profile
from abtem.potentials import Potential, AbstractPotential
from abtem.waves.base import AbstractWaves, AbstractScannedWaves, BeamTilt
from abtem.waves.multislice import multislice
from abtem.waves.scan import AbstractScan
from abtem.waves.transfer import CTF


class Waves(AbstractWaves, HasAxesMetadata):
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
                 energy: float,
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 tilt: Tuple[float, float] = (0., 0.),
                 antialias_aperture: float = None,
                 axes_metadata=None,
                 metadata=None):

        if len(array.shape) < 2:
            raise RuntimeError('Wave function array should be have 2 dimensions or more')

        self._array = array
        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_aperture = AntialiasAperture(cutoff=antialias_aperture)
        self._device = get_device_from_array(self._array)

        self._axes_metadata = axes_metadata
        self._metadata = metadata

    @property
    def axes_metadata(self):
        return self._axes_metadata

    @property
    def metadata(self):
        return self._metadata

    def compute(self):
        return self.__class__(self._array.compute(), extent=self.extent, energy=self.energy, tilt=self.tilt,
                              antialias_aperture=self.antialias_aperture)

    def __len__(self):
        return len(self.array)

    @property
    def array(self) -> np.ndarray:
        """Array representing the wave functions."""
        return self._array

    @property
    def shape(self):
        return self.array.shape

    def intensity(self) -> Images:
        """
        Calculate the intensity of the wave functions at the image plane.

        Returns
        -------
        Measurement
            The wave function intensity.
        """
        return Images(abs2(self.array), axes_metadata=self.axes_metadata)

    def downsample(self, max_angle='valid') -> 'Waves':
        gpts = self._gpts_within_angle(max_angle)

        array = self.array

        if gpts != self.gpts:
            array = fft2_interpolate(array, new_shape=self.array.shape[:-2] + gpts)

        antialias_aperture = self.antialias_aperture * min(self.gpts[0] / gpts[0], self.gpts[1] / gpts[1])

        return Waves(array, extent=self.extent, energy=self.energy, antialias_aperture=antialias_aperture)

    def diffraction_patterns(self, max_angle='valid', fftshift=True) -> DiffractionPatterns:
        """
        Calculate the intensity of the wave functions at the diffraction plane.

        Returns
        -------
        Measurement object
            The intensity of the diffraction pattern(s).
        """

        def _diffraction_pattern(array, new_gpts, fftshift):
            xp = get_array_module(array)

            array = fft2(array, overwrite_x=False)

            if array.shape[-2:] != new_gpts:
                array = fft2_crop(array, new_shape=array.shape[:-2] + new_gpts)

            array = abs2(array)

            if fftshift:
                return xp.fft.fftshift(array, axes=(-1, -2))

            return array

        xp = get_array_module(self.array)

        new_gpts = self._gpts_within_angle(max_angle)

        pattern = self.array.map_blocks(_diffraction_pattern, new_gpts=new_gpts, fftshift=fftshift,
                                        chunks=self.array.chunks[:-2] + ((new_gpts[0],), (new_gpts[1],)),
                                        meta=xp.array((), dtype=xp.float32))

        axes_metadata = [{'type': 'fourier_space', 'sampling': self.angular_sampling[0]},
                         {'type': 'fourier_space', 'sampling': self.angular_sampling[1]}]
        axes_metadata = self.axes_metadata[:-2] + axes_metadata

        return DiffractionPatterns(pattern, fftshift=fftshift, axes_metadata=axes_metadata, metadata=self._metadata)

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

        if ctf is None:
            ctf = CTF(**kwargs)

        if not ctf.accelerator.energy:
            ctf.accelerator.match(self.accelerator)

        self.accelerator.match(ctf.accelerator, check_match=True)

        self.accelerator.check_is_defined()
        self.grid.check_is_defined()

        kernel = ctf.evaluate_on_grid(extent=self.extent, gpts=self.gpts, sampling=self.sampling)

        xp = get_array_module(self.array)

        kernel = xp.asarray(kernel)

        #print(type(kernel), type(self.array), fft2_convolve(self.array, kernel, overwrite_x=in_place))
        #sss

        return self.__class__(fft2_convolve(self.array, kernel, overwrite_x=in_place),
                              extent=self.extent,
                              energy=self.energy,
                              axes_metadata=self._axes_metadata,
                              tilt=self.tilt)

    def multislice(self, potential: AbstractPotential, splits=1) -> 'Waves':
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

        if potential.num_frozen_phonons == 1:
            return multislice(self, potential, splits=splits)

        exit_waves = []
        for p in potential.frozen_phonon_potentials():
            exit_waves.append(multislice(self.copy(), p, splits=splits))

        array = da.stack([exit_wave.array for exit_wave in exit_waves], axis=0)

        axes_metadata = [{'type': 'ensemble', 'name': 'frozen_phonons'}] + self.axes_metadata

        return self.__class__(array=array, extent=self.extent, energy=self.energy, tilt=self.tilt,
                              antialias_aperture=2 / 3., axes_metadata=axes_metadata)

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
        new_copy = self.__class__(array=self._array.copy(), tilt=self.tilt, energy=self.energy,
                                  antialias_aperture=self.antialias_aperture)
        new_copy._grid = copy(self.grid)
        new_copy._accelerator = copy(self.accelerator)
        return new_copy

    def copy(self) -> 'Waves':
        """Make a copy."""
        return copy(self)


class PlaneWave(AbstractWaves):
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

    def multislice(self, potential: Union[AbstractPotential, Atoms], splits=1) -> Waves:
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
        return self.build().multislice(potential, splits=splits)

    def build(self) -> Waves:
        """Build the plane wave function as a Waves object."""
        xp = get_array_module_from_device(self._device)
        self.grid.check_is_defined()
        array = da.from_array(xp.ones((self.gpts[0], self.gpts[1]), dtype=xp.complex64), chunks=(-1, -1))

        metadata = {'energy': self.energy}
        return Waves(array, extent=self.extent, energy=self.energy, axes_metadata=self._base_axes_metadata,
                     metadata=metadata)

    def __copy__(self) -> 'PlaneWave':
        return self.__class__(extent=self.extent, gpts=self.gpts, sampling=self.sampling, energy=self.energy)


class Probe(AbstractScannedWaves, HasEventMixin):
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
        self._antialias_aperture = AntialiasAperture()
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._device = device

    @property
    def ctf(self) -> CTF:
        """Probe contrast transfer function."""
        return self._ctf

    def _fourier_translation_operator(self, positions):
        xp = get_array_module(positions)
        positions /= xp.array(self.sampling).astype(np.float32)
        drop_axis = len(positions.shape) - 1
        new_axis = (len(positions.shape) - 1, len(positions.shape))
        return positions.map_blocks(fft2_shift_kernel, shape=self.gpts, meta=xp.array((), dtype=np.complex64),
                                    drop_axis=drop_axis, new_axis=new_axis,
                                    chunks=positions.chunks[:-1] + ((self.gpts[0],), (self.gpts[1],)))

    def _evaluate_ctf(self):
        xp = get_array_module(self._device)
        array = self._ctf.evaluate_on_grid(gpts=self.gpts, sampling=self.sampling, xp=xp)
        array = array / xp.sqrt(abs2(array).sum())  # / np.sqrt(np.prod(array.shape))
        return array

    def build(self, positions: Union[Sequence[Sequence[float]], AbstractScan] = None) -> Waves:
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

        if isinstance(positions, AbstractScan):
            axes_metadata = positions.axes_metadata + self._base_axes_metadata
            positions = positions.get_positions()
        else:
            axes_metadata = [{'type': 'positions'}] + self._base_axes_metadata

        positions = self._validate_positions(positions)

        positions = da.from_array(positions, chunks=self._compute_chunks(len(positions.array) - 1))

        xp = get_array_module(self._device)

        positions = positions.map_blocks(xp.asarray)

        ctf = da.from_delayed(dask.delayed(self._evaluate_ctf)(), shape=self.gpts,
                              meta=xp.array((), dtype=np.complex64))

        array = ifft2(ctf * self._fourier_translation_operator(positions))

        return Waves(array, extent=self.extent, energy=self.energy, tilt=self.tilt, axes_metadata=axes_metadata)

    def multislice(self,
                   potential: AbstractPotential,
                   positions: Union[Sequence[Sequence[float]], AbstractScan] = None) -> Waves:
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

        potential = self._validate_potential(potential)

        return self.build(positions).multislice(potential)

    def scan(self,
             scan: AbstractScan,
             detectors: Union[AbstractDetector, Sequence[AbstractDetector]],
             potential: Union[Atoms, AbstractPotential],
             chunk_size: int = None,
             ) -> Union[Measurement, List[Measurement]]:

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

        for i, potential_config in enumerate(potential.frozen_phonon_potentials()):
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
