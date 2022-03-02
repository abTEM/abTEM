"""Module to describe electron waves and their propagation."""
import itertools
from copy import copy, deepcopy
from numbers import Number
from typing import Union, Sequence, Tuple, List, Dict, Iterable

import dask
import dask.array as da
import numpy as np
import zarr
from ase import Atoms

from abtem.core.antialias import AntialiasAperture
from abtem.core.axes import FrozenPhononsAxis, AxisMetadata, axis_from_dict, axis_to_dict, PositionsAxis, OrdinalAxis
from abtem.core.backend import get_array_module, _validate_device, copy_to_device, device_name_from_array_module
from abtem.core.complex import abs2
from abtem.core.dask import HasDaskArray, validate_lazy, ComputableList
from abtem.core.energy import Accelerator
from abtem.core.fft import fft2, ifft2, fft2_convolve, fft_crop, fft2_interpolate, fft_shift_kernel
from abtem.core.grid import Grid
from abtem.ionization.multislice import transition_potential_multislice
from abtem.ionization.transitions import AbstractTransitionCollection, AbstractTransitionPotential
from abtem.measure.detect import AbstractDetector, validate_detectors, PixelatedDetector, WavesDetector, \
    FlexibleAnnularDetector
from abtem.measure.measure import DiffractionPatterns, Images, AbstractMeasurement, stack_measurements
from abtem.measure.thickness import thickness_series_axes_metadata, thickness_series_precursor
from abtem.potentials.potentials import Potential, AbstractPotential, validate_potential
from abtem.waves.base import WavesLikeMixin
from abtem.waves.multislice import multislice, multislice_and_detect_with_frozen_phonons, multislice_and_detect
from abtem.waves.scan import AbstractScan, GridScan, validate_scan
from abtem.waves.tilt import BeamTilt
from abtem.waves.transfer import CTF


def lazy_multislice_and_detect(waves, potential, detectors, func, func_kwargs=None):
    if func_kwargs is None:
        func_kwargs = {}

    if not waves.is_lazy:
        return multislice_and_detect(waves, potential, detectors, func, func_kwargs)

    chunks = waves.array.chunks[:-2]
    delayed_waves = waves.to_delayed()

    def wrapped_multislice_detect(waves, potential, detectors, func, func_kwargs):
        output = [measurement.array for measurement in multislice_and_detect(waves,
                                                                             potential,
                                                                             detectors,
                                                                             func,
                                                                             func_kwargs)]
        return output

    dwrapped = dask.delayed(wrapped_multislice_detect, nout=len(detectors))
    delayed_potential = potential.to_delayed()

    collections = np.empty_like(delayed_waves, dtype=object)
    for index, waves_block in np.ndenumerate(delayed_waves):
        collections[index] = dwrapped(waves_block, delayed_potential, detectors, func, func_kwargs)

    _, _, thickness_axes_metadata = thickness_series_precursor(detectors, potential)

    measurements = []
    for i, (detector, axes_metadata) in enumerate(zip(detectors, thickness_axes_metadata)):
        arrays = np.empty_like(collections, dtype=object)

        thicknesses = detector.num_detections(potential)
        measurement_shape = detector.measurement_shape(waves)[waves.num_extra_axes:]

        for (index, collection), chunk in zip(np.ndenumerate(collections), itertools.product(*chunks)):
            shape = (thicknesses,) + chunk + measurement_shape
            arrays[index] = da.from_delayed(collection[i], shape=shape, meta=np.array((), dtype=np.float32))

        if len(arrays.shape) == 0:
            arrays = arrays.item()
        elif len(arrays.shape) == 1:
            arrays = da.concatenate(arrays, axis=1)
        elif len(arrays.shape) == 2:
            arrays = da.concatenate([da.concatenate(arrays[:, i], axis=1) for i in range(arrays.shape[1])], axis=2)
        else:
            raise RuntimeError()

        d = detector.measurement_kwargs(waves)
        d['extra_axes_metadata'] = [axes_metadata] + d['extra_axes_metadata']

        measurement = detector.measurement_type(waves)(arrays, **d)
        measurement = measurement.squeeze()
        measurements.append(measurement)

    return ComputableList(measurements)


class Waves(HasDaskArray, WavesLikeMixin):
    """
    Waves object

    The waves object can define a batch of arbitrary 2D wave functions defined by a complex numpy or cupy array.

    Parameters
    ----------
    array : array
        Complex array defining the one or more 2d wave functions.
    extent : one or two float
        Lateral extent of wave function [Å].
    sampling : one or two float
        Lateral sampling of wave functions [1 / Å].
    energy : float
        Electron energy [eV].
    tilt : two float
        Small angle beam tilt [mrad].
    antialias_cutoff_gpts : two int, optional
        Shape of rectangle defining the maximum
        Assumed antialiasing aperture as a fraction of the real space Nyquist frequency. Default is 2/3.
    metadata : dict
        A dictionary defining simulation metadata. All items will be added to the metadata of measurements derived from
        the waves.
    extra_axes_metadata : list of AxesMetadata
    """

    def __init__(self,
                 array: np.ndarray,
                 energy: float = None,
                 extent: Union[float, Tuple[float, float]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 tilt: Tuple[float, float] = (0., 0.),
                 extra_axes_metadata: List[AxisMetadata] = None,
                 metadata: Dict = None,
                 antialias_cutoff_gpts: Tuple[int, int] = None):

        if len(array.shape) < 2:
            raise RuntimeError('Wave function array should have 2 dimensions or more')

        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_cutoff_gpts = antialias_cutoff_gpts

        super().__init__(array=array)

        if extra_axes_metadata is None:
            extra_axes_metadata = []

        if metadata is None:
            metadata = {}

        self._extra_axes_metadata = extra_axes_metadata

        self._metadata = metadata

        self._check_axes_metadata()

    @property
    def device(self):
        return device_name_from_array_module(get_array_module(self.array))

    def lazy(self):
        self._array = da.from_array(self.array)

    def squeeze(self) -> 'Waves':
        shape = self.shape[:-2]
        squeezed = tuple(np.where([n == 1 for n in shape])[0])
        xp = get_array_module(self.array)
        d = self._copy_as_dict(copy_array=False)
        d['array'] = xp.squeeze(self.array, axis=squeezed)
        d['extra_axes_metadata'] = [element for i, element in enumerate(self.extra_axes_metadata) if i not in squeezed]
        return self.__class__(**d)

    @property
    def metadata(self) -> Dict:
        metadata = self._metadata
        metadata['energy'] = self.energy
        return metadata

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
            array = self.array.map_blocks(fft2_interpolate,
                                          new_shape=gpts,
                                          normalization=normalization,
                                          chunks=self.array.chunks[:-2] + gpts,
                                          meta=xp.array((), dtype=xp.complex64))
        else:
            array = fft2_interpolate(self.array, new_shape=gpts, normalization=normalization)

        # antialias_aperture = max(new_sampling) / max(self.sampling) * self.antialias_aperture

        d = self._copy_as_dict(copy_array=False)
        d['array'] = array
        d['sampling'] = (self.extent[0] / gpts[0], self.extent[1] / gpts[1])
        # d['antialias_aperture'] = antialias_aperture
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

    def to_delayed(self):
        d = self._copy_as_dict(copy_array=False)

        def wrap_array(array):
            return self.__class__(array, **d)

        dwrap_array = dask.delayed(wrap_array)

        arrays = self.array.to_delayed()
        arrays = arrays.reshape(arrays.shape[:-2])

        waves = np.empty_like(arrays, dtype=object)
        for index, array in np.ndenumerate(arrays):
            waves[index] = dwrap_array(array)

        return waves

    def diffraction_patterns(self,
                             max_angle: Union[str, float, None] = 'valid',
                             block_direct: bool = False,
                             fftshift: bool = True) -> DiffractionPatterns:
        """
        Calculate the intensity of the wave functions at the diffraction plane.

        Parameters
        ----------
        max_angle : float, str, optional
            Maximum scattering angle of diffraction patterns in mrad. The
        block_direct : bool
            If true the direct beam is will be masked.
        fftshift : bool
            If true

        Returns
        -------
        DiffractionPatterns
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
                                                   sampling=self.fourier_space_sampling,
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
        for potential_config in potential.get_distribution(lazy=self.is_lazy):
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
                   detectors: AbstractDetector = None,
                   start: int = 0,
                   stop: int = None,
                   conjugate: bool = False,
                   **kwargs):
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

        if detectors is None:
            xp = get_array_module(self.array)

            if self.is_lazy:
                potential = potential.to_delayed()

            return self.map_blocks(multislice,
                                   potential=potential,
                                   start=start,
                                   stop=stop,
                                   conjugate=conjugate,
                                   **kwargs,
                                   meta=xp.array((), dtype=xp.complex64))

        def func(waves, potential, detectors):
            waves = waves.copy()

            def func(waves, detectors):
                return [detector.detect(waves) for detector in detectors]

            return lazy_multislice_and_detect(waves, potential, detectors, func)

        if detectors is None:
            detectors = [WavesDetector()]

        return multislice_with_frozen_phonons(self, potential, detectors, lazy=self.is_lazy, func=func)

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
            antialias_cutoff_gpts = f.attrs['antialias_cutoff_gpts']
            extra_axes_metadata = [axis_from_dict(d) for d in f.attrs['extra_axes_metadata']]
            metadata = f.attrs['metadata']
            shape = f['array'].shape

        if chunks is None:
            chunks = (-1,) * (len(shape) - 2)

        array = da.from_zarr(url, component='array', chunks=chunks + (-1, -1))
        return cls(array=array, energy=energy, extent=extent, tilt=tilt, antialias_cutoff_gpts=antialias_cutoff_gpts,
                   extra_axes_metadata=extra_axes_metadata, metadata=metadata)

    def __getitem__(self, items) -> 'Waves':
        if isinstance(items, (Number, slice)):
            items = (items,)

        if len(self.array.shape) <= self.grid.dimensions:
            raise RuntimeError()

        removed_axes = []
        for i, item in enumerate(items):
            if isinstance(item, Number):
                removed_axes.append(i)

        # if self._check_is_base_axes(removed_axes):
        #    raise RuntimeError('base axes cannot be indexed')

        axes = [element for i, element in enumerate(self.extra_axes_metadata) if not i in removed_axes]

        d = self._copy_as_dict(copy_array=False)
        d['array'] = self._array[items]
        d['extra_axes_metadata'] = axes

        return self.__class__(**d)

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
             'sampling': self.sampling,
             'antialias_cutoff_gpts': self.antialias_cutoff_gpts,
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

    @property
    def base_axes(self):
        return 0, 1

    @property
    def shape(self):
        return self.gpts

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

    def multislice(self,
                   potential: Union[AbstractPotential, Atoms],
                   detectors: AbstractDetector = None,
                   lazy: bool = None) -> Waves:
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

        def func(waves, potential, detectors):
            waves = waves.build(lazy=lazy)

            def func(waves, detectors):
                return [detector.detect(waves) for detector in detectors]

            return lazy_multislice_and_detect(waves, potential, detectors, func)

        if detectors is None:
            detectors = [WavesDetector()]

        return multislice_and_detect_with_frozen_phonons(self, potential, detectors, lazy=lazy, func=func)

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
                 normalize: bool = True,
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
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._device = _validate_device(device)
        self._normalize = normalize

        self._extra_axes_metadata = []
        self._antialias_cutoff_gpts = None

    @property
    def metadata(self):
        return {'energy': self.energy}

    @property
    def shape(self):
        return self.gpts

    @property
    def ctf(self) -> CTF:
        """Probe contrast transfer function."""
        return self._ctf

    def _validate_chunks(self, chunks):
        if chunks is None:
            chunk_size = dask.utils.parse_bytes(dask.config.get('array.chunk-size'))
            chunks = int(chunk_size / self._bytes_per_wave())

        return chunks

    def _validate_positions(self,
                            positions: Union[Sequence, AbstractScan] = None,
                            lazy: bool = None,
                            chunks: int = None):

        lazy = validate_lazy(lazy)

        chunks = self._validate_chunks(chunks)

        if hasattr(positions, 'match_probe'):
            positions.match_probe(self)

        if hasattr(positions, 'get_positions'):
            if lazy:
                return positions.get_positions(lazy=lazy, chunks=chunks)
            else:
                return positions.get_positions(lazy=lazy)

        if positions is None:
            positions = [self.extent[0] / 2, self.extent[1] / 2]

        if not isinstance(positions, da.core.Array):
            positions = np.array(positions, dtype=np.float32)

        if len(positions.shape) == 1:
            positions = positions

        if isinstance(positions, np.ndarray) and lazy:
            positions = da.from_array(positions)

        if positions.shape[-1] != 2:
            raise ValueError()

        return positions

    def _calculate_array(self, positions):
        xp = get_array_module(self._device)
        positions = xp.asarray(positions)
        xp = get_array_module(positions)
        positions = positions / xp.array(self.sampling).astype(np.float32)
        array = fft_shift_kernel(positions, shape=self.gpts)
        array *= self.ctf.evaluate_on_grid(gpts=self.gpts, sampling=self.sampling, xp=xp)
        if self._normalize:
            array /= xp.sqrt(abs2(array).sum((-2, -1), keepdims=True))
        return ifft2(array)

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

        if isinstance(validated_positions, da.core.Array):
            drop_axis = len(validated_positions.shape) - 1
            new_axis = (len(validated_positions.shape) - 1, len(validated_positions.shape))
            array = validated_positions.map_blocks(self._calculate_array,
                                                   meta=xp.array((), dtype=np.complex64),
                                                   drop_axis=drop_axis, new_axis=new_axis,
                                                   chunks=validated_positions.chunks[:-1] + (
                                                       (self.gpts[0],), (self.gpts[1],)))

        else:
            array = self._calculate_array(validated_positions)

        if hasattr(positions, 'axes_metadata'):
            extra_axes_metadata = positions.axes_metadata
        else:
            extra_axes_metadata = [PositionsAxis()] * (len(array.shape) - 2)

        metadata = {'semiangle_cutoff': self.ctf.semiangle_cutoff}

        return Waves(array,
                     extent=self.extent,
                     energy=self.energy,
                     tilt=self.tilt,
                     extra_axes_metadata=extra_axes_metadata,
                     metadata=metadata,
                     antialias_cutoff_gpts=self.antialias_cutoff_gpts)

    def _generate_waves(self, scan, potential, chunks):
        extra_axes_metadata = scan.axes_metadata

        chunks = self._validate_chunks(chunks)

        for indices, positions in scan.generate_positions(chunks=chunks):
            array = self._calculate_array(positions)
            waves = Waves(array, extent=self.extent, energy=self.energy, tilt=self.tilt,
                          extra_axes_metadata=extra_axes_metadata, metadata=self.metadata)
            yield indices, multislice(waves, potential=potential)

    def multislice(self,
                   potential: Union[AbstractPotential, Atoms],
                   positions: AbstractScan = None,
                   detectors: AbstractDetector = None,
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
        lazy = validate_lazy(lazy)
        potential = validate_potential(potential, self)

        if positions is None:
            positions = (self.extent[0] / 2, self.extent[1] / 2)

        positions = validate_scan(positions, self)

        if detectors is None:
            detectors = [WavesDetector()]

        def func(waves, potential, detectors, positions, lazy, chunks):

            waves = waves.build(positions=positions, lazy=lazy, chunks=chunks)

            def func(waves, detectors):
                return [detector.detect(waves) for detector in detectors]

            return lazy_multislice_and_detect(waves, potential, detectors, func)

        return multislice_and_detect_with_frozen_phonons(self,
                                                         potential,
                                                         detectors,
                                                         lazy=lazy,
                                                         func=func,
                                                         func_kwargs={'positions': positions,
                                                                      'lazy': lazy,
                                                                      'chunks': chunks})

    def scan(self,
             potential: Union[Atoms, AbstractPotential],
             scan: Union[AbstractScan, np.ndarray, Sequence] = None,
             detectors: Union[AbstractDetector, Sequence[AbstractDetector]] = None,
             chunks: int = None,
             lazy: bool = None) -> Union[List, AbstractMeasurement]:

        if scan is None:
            scan = GridScan()

        if detectors is None:
            detectors = FlexibleAnnularDetector()

        return self.multislice(potential, scan, detectors, chunks=chunks, lazy=lazy)

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

    def copy(self):
        return deepcopy(self)

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
