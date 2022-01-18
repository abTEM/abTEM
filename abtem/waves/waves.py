"""Module to describe electron waves and their propagation."""
import dataclasses
from collections import Iterable
from copy import copy, deepcopy
from typing import Union, Sequence, Tuple, List, Dict

import dask
import dask.array as da
import numpy as np
import zarr
from ase import Atoms

from abtem.core.antialias import AntialiasAperture
from abtem.core.axes import HasAxes, FrozenPhononsAxis, AxisMetadata, axis_from_dict, axis_to_dict, PositionsAxis
from abtem.core.backend import get_array_module, _validate_device, copy_to_device
from abtem.core.complex import abs2
from abtem.core.dask import HasDaskArray, ComputableList, validate_lazy
from abtem.core.energy import Accelerator
from abtem.core.fft import fft2, ifft2, fft2_convolve, fft_crop, fft2_interpolate, fft_shift_kernel
from abtem.core.grid import Grid
from abtem.ionization.multislice import transition_potential_multislice
from abtem.ionization.transitions import AbstractTransitionCollection, AbstractTransitionPotential
from abtem.measure.detect import AbstractDetector, validate_detectors, PixelatedDetector
from abtem.measure.measure import DiffractionPatterns, Images, AbstractMeasurement, stack_measurements
from abtem.potentials.potentials import Potential, AbstractPotential, validate_potential
from abtem.waves.base import WavesLikeMixin
from abtem.waves.multislice import multislice
from abtem.waves.scan import AbstractScan
from abtem.waves.tilt import BeamTilt
from abtem.waves.transfer import CTF


def stack_waves(waves, axes_metadata):
    if len(waves) == 0:
        return waves[0]
    array = np.stack([waves.array for waves in waves], axis=0)
    d = waves[0]._copy_as_dict(copy_array=False)
    d['array'] = array
    d['extra_axes_metadata'] = [axes_metadata] + waves[0].extra_axes_metadata
    return Waves(**d)


class Waves(HasDaskArray, WavesLikeMixin, HasAxes):
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
    antialias_aperture : float
        Assumed antialiasing aperture as a fraction of the real space Nyquist frequency. Default is 2/3.
    """

    def __init__(self,
                 array: np.ndarray,
                 energy: float = None,
                 extent: Union[float, Tuple[float, float]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 tilt: Tuple[float, float] = (0., 0.),
                 antialias_aperture: float = 2 / 3.,
                 extra_axes_metadata: List[AxisMetadata] = None,
                 metadata: Dict = None):

        if len(array.shape) < 2:
            raise RuntimeError('Wave function array should have 2 dimensions or more')

        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_aperture = AntialiasAperture(cutoff=antialias_aperture)

        super().__init__(array=array)

        if extra_axes_metadata is None:
            extra_axes_metadata = []

        self._extra_axes_metadata = extra_axes_metadata

        if metadata is None:
            metadata = {}

        self._metadata = metadata

        self._check_axes_metadata()

    def lazy(self):
        self._array = da.from_array(self.array)

    def squeeze(self) -> 'Waves':
        shape = self.shape[:-2]
        squeezed = tuple(np.where([n == 1 for n in shape])[0])
        xp = get_array_module(self.array)
        d = self._copy_as_dict(copy_array=False)
        d['array'] = xp.squeeze(self.array, axis=squeezed)
        d['extra_axes_metadata'] = [element for i, element in enumerate(self.extra_axes_metadata) if not i in squeezed]
        return self.__class__(**d)

    @property
    def metadata(self) -> Dict:
        return self._metadata

    def intensity(self) -> Images:
        """
        Calculate the intensity of the wave functions at the image plane.

        Returns
        -------
        Images
            The wave function intensity.
        """
        return Images(abs2(self.array), sampling=self.sampling, extra_axes_metadata=self.extra_axes_metadata,
                      metadata=self.metadata)

    def downsample(self, max_angle: Union[str, float] = 'cutoff', normalization: str = 'values') -> 'Waves':
        """
        Downsample the wave function to a lower maximum scattering angle.

        Parameters
        ----------
        max_angle : str or float
            Maximum scattering angle after downsampling.
        normalization :


        Returns
        -------
        Waves
            The downsampled wave functions.
        """

        xp = get_array_module(self.array)
        gpts = self._gpts_within_angle(max_angle)
        if self.is_lazy:

            array = self.array.map_blocks(fft2_interpolate, new_shape=gpts,
                                          normalization=normalization,
                                          chunks=self.array.chunks[:-2] + gpts,
                                          meta=xp.array((), dtype=xp.complex64))
        else:
            array = fft2_interpolate(self.array, new_shape=gpts, normalization=normalization)

        antialias_aperture = self.antialias_aperture * min(self.gpts[0] / gpts[0], self.gpts[1] / gpts[1])

        d = self._copy_as_dict(copy_array=False)
        d['array'] = array
        d['antialias_aperture'] = antialias_aperture
        return self.__class__(**d)

    def detect(self, detectors: Union[AbstractDetector, List[AbstractDetector]]) \
            -> Union[AbstractMeasurement, List[AbstractMeasurement]]:

        if not isinstance(detectors, Iterable):
            detectors = (detectors,)

        measurements = []
        for detector in detectors:
            measurements += [detector.detect(self)]

        if len(measurements) == 1:
            return measurements[0]

        return measurements

    def apply_detector_func(self, func, detectors, scan=None, **kwargs):
        detectors = validate_detectors(detectors)

        new_cls = [detector.measurement_type(self, scan) for detector in detectors]
        new_cls_kwargs = [detector.measurement_kwargs(self, scan) for detector in detectors]

        signatures = []
        output_sizes = {}
        meta = []
        i = 2
        for detector in detectors:
            shape = detector.measurement_shape(self, scan=scan)[self.num_extra_axes:]
            signatures.append(f'({",".join([str(i) for i in range(i, i + len(shape))])})')
            output_sizes.update({str(index): n for index, n in zip(range(i, i + len(shape)), shape)})
            meta.append(np.array((), dtype=detector.measurement_dtype))
            i += len(shape)

        signature = '(0,1)->' + ','.join(signatures)

        measurements = self.apply_gufunc(func,
                                         detectors=detectors,
                                         new_cls=new_cls,
                                         new_cls_kwargs=new_cls_kwargs,
                                         signature=signature,
                                         output_sizes=output_sizes,
                                         meta=meta,
                                         **kwargs)

        return measurements

    def diffraction_patterns(self, max_angle: Union[str, float] = 'valid', block_direct: bool = False,
                             fftshift: bool = True) -> DiffractionPatterns:
        """
        Calculate the intensity of the wave functions at the diffraction plane.

        Returns
        -------
        DiffractionPatterns
            The intensity of the diffraction pattern(s).
        """

        def _diffraction_pattern(array, new_gpts, fftshift):
            xp = get_array_module(array)

            array = fft2(array, overwrite_x=False)

            if array.shape[-2:] != new_gpts:
                array = fft_crop(array, new_shape=array.shape[:-2] + new_gpts)

            array = abs2(array)

            if fftshift:
                return xp.fft.fftshift(array, axes=(-1, -2))

            return array

        xp = get_array_module(self.array)
        new_gpts = self._gpts_within_angle(max_angle)

        if self.is_lazy:
            pattern = self.array.map_blocks(_diffraction_pattern, new_gpts=new_gpts, fftshift=fftshift,
                                            chunks=self.array.chunks[:-2] + ((new_gpts[0],), (new_gpts[1],)),
                                            meta=xp.array((), dtype=xp.float32))
        else:
            pattern = _diffraction_pattern(self.array, new_gpts=new_gpts, fftshift=fftshift)

        diffraction_patterns = DiffractionPatterns(pattern,
                                                   angular_sampling=self.angular_sampling,
                                                   fftshift=fftshift,
                                                   extra_axes_metadata=self.extra_axes_metadata,
                                                   metadata=self.metadata)

        if block_direct:
            diffraction_patterns = diffraction_patterns.block_direct(radius=block_direct)

        return diffraction_patterns

    def apply_ctf(self, ctf: CTF = None, in_place: bool = False, **kwargs) -> 'Waves':
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
        Waves
            The wave functions with the contrast transfer function applied.
        """

        if ctf is None:
            ctf = CTF(**kwargs)

        if not ctf.accelerator.energy:
            ctf.accelerator.match(self.accelerator)

        self.accelerator.match(ctf.accelerator, check_match=True)
        self.accelerator.check_is_defined()
        self.grid.check_is_defined()

        xp = get_array_module(self.array)
        kernel = xp.asarray(ctf.evaluate_on_grid(extent=self.extent, gpts=self.gpts, sampling=self.sampling))

        d = self._copy_as_dict(copy_array=False)
        d['array'] = fft2_convolve(self.array, kernel, overwrite_x=in_place)
        return self.__class__(**d)

    def transition_multislice(self, potential: AbstractPotential, transitions, detectors=None, ctf=None):
        potential = validate_potential(potential, self)
        detectors = validate_detectors(detectors)

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        if hasattr(transitions, 'get_transition_potentials'):
            if self.is_lazy:
                transitions = dask.delayed(transitions.get_transition_potentials)()
            else:
                transitions = transitions.get_transition_potentials()

        measurements = []
        for potential_config in potential.get_potential_configurations(lazy=self.is_lazy):
            config_measurements = self.copy().apply_detector_func(transition_potential_multislice,
                                                                  potential=potential_config,
                                                                  detectors=detectors,
                                                                  transition_potentials=transitions,
                                                                  ctf=ctf)

            measurements.append(config_measurements)

        measurements = list(map(list, zip(*measurements)))
        measurements = [stack_measurements(measurements, FrozenPhononsAxis()) for measurements in measurements]

        for i, (detector, measurement) in enumerate(zip(detectors, measurements)):
            if detector.ensemble_mean:
                measurements[i] = measurement.mean(0)

        return measurements[0] if len(measurements) == 1 else measurements

    def multislice(self,
                   potential: AbstractPotential,
                   start: int = 0,
                   stop: int = None,
                   transpose: bool = False,
                   in_place: bool = False) -> 'Waves':
        """
        Propagate and transmit wave function through the provided potential.

        Parameters
        ----------
        potential : Potential
            The potential through which to propagate the wave function.

        Returns
        -------
        Waves object
            Wave function at the exit plane of the potential.
        """

        self.grid.match(potential)
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        xp = get_array_module(self.array)

        potential = self._validate_potential(potential)
        potential_configurations = potential.get_potential_configurations(lazy=self.is_lazy)

        exit_waves = []
        for potential_configuration in potential_configurations:
            waves = self
            if not in_place and (len(potential_configurations) == 1):
                waves = waves.copy()

            waves = waves.map_blocks(multislice,
                                     potential=potential_configuration,
                                     start=start,
                                     stop=stop,
                                     transpose=transpose,
                                     meta=xp.array((), dtype=xp.complex64))

            exit_waves.append(waves)

        exit_waves = stack_waves(exit_waves, FrozenPhononsAxis()) if len(exit_waves) > 1 else exit_waves[0]

        exit_waves.antialias_aperture = 2 / 3.

        return exit_waves

    def to_zarr(self, url: str, overwrite: bool = False):
        """
        Write potential to a zarr file.

        Parameters
        ----------
        url: str
            url to which the data is saved.
        """

        with zarr.open(url, mode='w') as root:
            if not self.is_lazy:
                self.lazy()
            self.array.to_zarr(url, component='array', overwrite=overwrite)
            for key, value in self._copy_as_dict(copy_array=False).items():
                if key == 'extra_axes_metadata':
                    root.attrs[key] = [axis_to_dict(axis) for axis in value]
                else:
                    root.attrs[key] = value

    @classmethod
    def from_zarr(cls, url: str, chunks: int = None) -> 'Waves':
        """
        Read wave functions from a hdf5 file.

        path : str
            The path to read the file.
        """

        with zarr.open(url, mode='r') as f:
            energy = f.attrs['energy']
            extent = f.attrs['extent']
            tilt = f.attrs['tilt']
            antialias_aperture = f.attrs['antialias_aperture']
            extra_axes_metadata = [axis_from_dict(d) for d in f.attrs['extra_axes_metadata']]
            metadata = f.attrs['metadata']
            shape = f['array'].shape

        if chunks is None:
            chunks = (-1,) * (len(shape) - 2)

        array = da.from_zarr(url, component='array', chunks=chunks + (-1, -1))
        return cls(array=array, energy=energy, extent=extent, tilt=tilt, antialias_aperture=antialias_aperture,
                   extra_axes_metadata=extra_axes_metadata, metadata=metadata)

    def __getitem__(self, item) -> 'Waves':
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

    def _copy_as_dict(self, copy_array: bool = True) -> dict:
        d = {'tilt': self.tilt,
             'energy': self.energy,
             'extent': self.extent,
             'antialias_aperture': self.antialias_aperture,
             'extra_axes_metadata': deepcopy(self._extra_axes_metadata),
             'metadata': copy(self.metadata)}

        if copy_array:
            d['array'] = self.array
        return d

    def copy(self, device: str = None):
        """Make a copy."""
        d = self._copy_as_dict(copy_array=False)

        if device is not None:
            array = copy_to_device(self.array, device)
        else:
            array = self.array.copy()

        d['array'] = array
        return self.__class__(**d)


class PlaneWave(WavesLikeMixin):
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
                 normalize: bool = False,
                 device: str = None):
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_aperture = AntialiasAperture()
        self._device = _validate_device(device)
        self._normalize = normalize

    def transition_multislice(self,
                              potential: Union[Atoms, AbstractPotential],
                              transitions: Union[AbstractTransitionCollection, AbstractTransitionPotential],
                              detectors: Union[AbstractDetector, List[AbstractDetector]] = None,
                              ctf: CTF = None,
                              lazy: bool = None):

        lazy = validate_lazy(lazy)
        potential = validate_potential(potential, self)

        if detectors is None:
            detectors = PixelatedDetector(ensemble_mean=True, fourier_space=False)

        measurements = self.build(lazy=lazy).transition_multislice(potential, transitions, detectors, ctf=ctf)
        return measurements

    def multislice(self, potential: Union[AbstractPotential, Atoms], lazy: bool = None) -> Waves:
        """
        Build plane wave function and propagate it through the potential. The grid of the two will be matched.

        Parameters
        ----------
        potential : Potential or Atoms object
            The potential through which to propagate the wave function.
        lazy : bool, optional
            Return lazy computation. Default is True.

        Returns
        -------
        Waves object
            Wave function at the exit plane of the potential.
        """

        lazy = validate_lazy(lazy)
        potential = validate_potential(potential, self)

        exit_waves = []
        for potential_config in potential.get_potential_configurations(lazy=lazy):
            waves = self.build(lazy=lazy)
            waves = waves.map_blocks(multislice, potential=potential_config, dtype=np.complex64)
            exit_waves.append(waves)

        if len(exit_waves) > 1:
            return stack_waves(exit_waves, FrozenPhononsAxis())
        else:
            return exit_waves[0]

    def build(self, lazy: bool = None) -> Waves:
        """Build the plane wave function as a Waves object."""
        xp = get_array_module(self._device)
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        def plane_wave(gpts, normalize, xp):
            if normalize:
                wave = xp.full(gpts, (1 / np.prod(gpts)).astype(xp.complex64), dtype=xp.complex64)
            else:
                wave = xp.ones(gpts, dtype=xp.complex64)
            return wave

        if validate_lazy(lazy):
            array = dask.delayed(plane_wave)(self.gpts, self._normalize, xp)
            array = da.from_delayed(array, shape=self.gpts, meta=xp.array((), dtype=xp.complex64))
        else:
            array = plane_wave(self.gpts, self._normalize, xp)

        return Waves(array, extent=self.extent, energy=self.energy)


class Probe(WavesLikeMixin):
    """
    Probe wave function object

    The probe object can represent a stack of electron probe wave functions for simulating scanning transmission
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
                 normalize: bool = False,
                 device: str = None,
                 **kwargs):

        if ctf is None:
            ctf = CTF(energy=energy, **kwargs)
        else:
            ctf = ctf.copy()

        if ctf.energy is None:
            ctf.energy = energy

        if ctf.energy != energy:
            raise RuntimeError('CTF energy does match probe energy')

        self._ctf = ctf
        self._accelerator = self._ctf._accelerator
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._antialias_aperture = AntialiasAperture()
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._device = _validate_device(device)
        self._normalize = normalize

    @property
    def ctf(self) -> CTF:
        """Probe contrast transfer function."""
        return self._ctf

    def _validate_positions(self,
                            positions: Union[Sequence, AbstractScan] = None,
                            lazy: bool = None,
                            chunks: int = None):

        lazy = validate_lazy(lazy)

        if chunks is None:
            chunk_size = dask.utils.parse_bytes(dask.config.get('array.chunk-size'))
            chunks = int(chunk_size / self._bytes_per_wave())

        if hasattr(positions, 'match_probe'):
            positions.match_probe(self)

        if hasattr(positions, 'get_positions'):
            if lazy:
                return positions.get_positions(lazy=lazy, chunks=chunks)
            else:
                return positions.get_positions(lazy=lazy)

        if positions is None:
            positions = (self.extent[0] / 2, self.extent[1] / 2)

        if not isinstance(positions, da.core.Array):
            positions = np.array(positions, dtype=np.float32)

        if len(positions.shape) == 1:
            positions = positions

        if isinstance(positions, np.ndarray) and lazy:
            positions = da.from_array(positions)

        if positions.shape[-1] != 2:
            raise ValueError()

        return positions

    def build(self,
              positions: Union[AbstractScan, Sequence] = None,
              chunks: int = None,
              lazy: bool = None) -> Waves:

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

        validated_positions = self._validate_positions(positions, lazy=lazy, chunks=chunks)
        xp = get_array_module(self._device)

        def calculate_probes(positions, xp):
            positions = xp.asarray(positions)
            xp = get_array_module(positions)
            positions = positions / xp.array(self.sampling).astype(np.float32)
            array = fft_shift_kernel(positions, shape=self.gpts)
            array *= self.ctf.evaluate_on_grid(gpts=self.gpts, sampling=self.sampling, xp=xp)
            if self._normalize:
                array /= xp.sqrt(abs2(array).sum((-2, -1), keepdims=True))
            return ifft2(array)

        if isinstance(validated_positions, da.core.Array):
            drop_axis = len(validated_positions.shape) - 1
            new_axis = (len(validated_positions.shape) - 1, len(validated_positions.shape))
            array = validated_positions.map_blocks(calculate_probes,
                                                   xp=xp,
                                                   meta=xp.array((), dtype=np.complex64),
                                                   drop_axis=drop_axis, new_axis=new_axis,
                                                   chunks=validated_positions.chunks[:-1] + (
                                                       (self.gpts[0],), (self.gpts[1],)))

        else:
            array = calculate_probes(validated_positions, xp)

        if hasattr(positions, 'axes_metadata'):
            extra_axes_metadata = positions.axes_metadata
        else:
            extra_axes_metadata = [PositionsAxis()] * (len(array.shape) - 2)

        return Waves(array, extent=self.extent, energy=self.energy, tilt=self.tilt,
                     extra_axes_metadata=extra_axes_metadata)

    def multislice(self,
                   potential: Union[AbstractPotential, Atoms],
                   positions: Union[AbstractScan, Sequence] = None,
                   chunks: int = None,
                   lazy: bool = None):
        """
        Build probe wave functions at the provided positions and propagate them through the potential.

        Parameters
        ----------
        positions : array of xy-positions
            Positions of the probe wave functions.
        potential : Potential or Atoms object
            The scattering potential.

        Returns
        -------
        Waves object
            Probe exit wave functions as a Waves object.
        """

        potential = validate_potential(potential, self)
        lazy = validate_lazy(lazy)

        exit_waves = []
        for potential_configuration in potential.get_potential_configurations(lazy=lazy):
            waves = self.build(positions, chunks=chunks, lazy=lazy)
            waves = waves.map_blocks(multislice, potential=potential_configuration, dtype=np.complex64)
            exit_waves.append(waves)

        if len(exit_waves) > 1:
            return stack_waves(exit_waves, FrozenPhononsAxis())
        else:
            return exit_waves[0]

    def scan(self,
             scan: Union[AbstractScan, np.ndarray, Sequence],
             detectors: Union[AbstractDetector, Sequence[AbstractDetector]],
             potential: Union[AbstractPotential],
             chunks: int = None,
             lazy: bool = None) -> Union[List, AbstractMeasurement]:

        waves = self.multislice(potential, scan, chunks=chunks, lazy=lazy)
        detectors = validate_detectors(detectors)

        def detect(waves, detectors):
            return [detector.detect(waves) for detector in detectors]

        measurements = list(waves.apply_detector_func(detect, detectors))

        for i, (detector, measurement) in enumerate(zip(detectors, measurements)):
            if detector.ensemble_mean:
                measurements[i] = measurement.mean(0)

        return measurements[0] if len(measurements) == 1 else ComputableList(measurements)

    def profile(self, angle: float = 0.):
        self.grid.check_is_defined()

        def _line_intersect_rectangle(point0, point1, lower_corner, upper_corner):
            if point0[0] == point1[0]:
                return (point0[0], lower_corner[1]), (point0[0], upper_corner[1])

            m = (point1[1] - point0[1]) / (point1[0] - point0[0])

            def y(x):
                return m * (x - point0[0]) + point0[1]

            def x(y):
                return (y - point0[1]) / m + point0[0]

            if y(0) < lower_corner[1]:
                intersect0 = (x(lower_corner[1]), y(x(lower_corner[1])))
            else:
                intersect0 = (0, y(lower_corner[0]))

            if y(upper_corner[0]) > upper_corner[1]:
                intersect1 = (x(upper_corner[1]), y(x(upper_corner[1])))
            else:
                intersect1 = (upper_corner[0], y(upper_corner[0]))

            return intersect0, intersect1

        point1 = np.array((self.extent[0] / 2, self.extent[1] / 2))

        measurement = self.build(point1).intensity()

        point2 = point1 + np.array([np.cos(np.pi * angle / 180), np.sin(np.pi * angle / 180)])
        point1, point2 = _line_intersect_rectangle(point1, point2, (0., 0.), self.extent)
        return measurement.interpolate_line(point1, point2)

    def __copy__(self) -> 'Probe':
        return self.__class__(gpts=self.gpts,
                              extent=self.extent,
                              sampling=self.sampling,
                              energy=self.energy,
                              tilt=self.tilt,
                              ctf=self.ctf.copy())

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
