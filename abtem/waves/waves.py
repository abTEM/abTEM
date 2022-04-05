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
from matplotlib.axes import Axes

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
from abtem.potentials.potentials import Potential, AbstractPotential, validate_potential
from abtem.waves.base import WavesLikeMixin
from abtem.waves.multislice import multislice, multislice_and_detect_with_frozen_phonons, multislice_and_detect
from abtem.waves.scan import AbstractScan, GridScan, validate_scan
from abtem.waves.tilt import BeamTilt
from abtem.waves.transfer import CTF
from dask.graph_manipulation import clone


def stack_waves(waves, axes_metadata):
    if len(waves) == 0:
        return waves[0]
    array = np.stack([waves.array for waves in waves], axis=0)
    d = waves[0]._copy_as_dict(copy_array=False)
    d['array'] = array
    d['extra_axes_metadata'] = [axes_metadata] + waves[0].extra_axes_metadata
    return waves[0].__class__(**d)


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
        Metadata associated with an axis of
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
        """
        Remove axes of length one from the waves.

        Returns
        -------
        squeezed : Waves
        """

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
        intensity_images : Images
            The wave function intensity.
        """

        return Images(abs2(self.array), sampling=self.sampling, extra_axes_metadata=self.extra_axes_metadata,
                      metadata=self.metadata)

    def downsample(self, max_angle: Union[str, float] = 'cutoff', normalization: str = 'values') -> 'Waves':
        """
        Downsample the wave function to a lower maximum scattering angle.

        Parameters
        ----------
        max_angle : {'cutoff', 'valid'} or float
            Maximum scattering angle after downsampling.
            ``cutoff`` :
            The maximum scattering angle after downsampling will be the cutoff of the antialias aperture.
            ``valid`` :
            The maximum scattering angle after downsampling will be the largest rectangle that fits inside the
            circular antialias aperture.

        normalization : {'values', 'amplitude'}
            The normalization parameter determines the preserved quantity after normalization.
                ``values`` :
                The pixelwise values of the wave function are preserved.
                ``amplitude`` :
                The total amplitude of the wave function is preserved.

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

        d = self._copy_as_dict(copy_array=False)
        d['array'] = array
        d['sampling'] = (self.extent[0] / gpts[0], self.extent[1] / gpts[1])
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
        """
        Convert into an array of :class:`dask.delayed.Delayed` objects, one per chunk of the ``array.
        The delayed objects can be evaluated to :class:`abtem.waves.Waves` objects.

        Returns
        -------

        """

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
                             block_direct: Union[bool, float] = False,
                             fftshift: bool = True) -> DiffractionPatterns:
        """
        Calculate the intensity of the wave functions at the diffraction plane.

        Parameters
        ----------
        max_angle : {'cutoff', 'valid'} or float
            Maximum scattering angle of the diffraction patterns.
            ``cutoff`` :
            The maximum scattering angle will be the cutoff of the antialias aperture.
            ``valid`` :
            The maximum scattering angle will be the largest rectangle that fits inside the circular antialias aperture.
        block_direct : bool or float
            If true the direct beam is masked.
        fftshift : bool
            If true, shift the zero-angle component to the center of the diffraction patterns.

        Returns
        -------
        diffraction_patterns : DiffractionPatterns
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

    def apply_ctf(self,
                  ctf: CTF = None,
                  in_place: bool = False, **kwargs) -> 'Waves':
        """
        Apply the aberrations defined by a CTF to this collection of wave functions.

        Parameters
        ----------
        ctf : CTF
            Contrast Transfer Function object to be applied.
        kwargs :
            Provide the parameters of the contrast transfer function as keyword arguments. See the documentation for the
            CTF object.

        Returns
        -------
        aberrated_waves : Waves
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

        def apply_ctf(array, ctf_parameters=None, weights=None, ctf=None):
            xp = get_array_module(array)

            if ctf_parameters is not None:
                ctf_parameters = ctf_parameters.item()
                ctf = ctf.copy()
                ctf.set_parameters(ctf_parameters)

            kernel = xp.asarray(ctf.evaluate_on_grid(extent=self.extent, gpts=self.gpts, sampling=self.sampling))

            array = fft2_convolve(array, kernel, overwrite_x=False)

            if weights is not None:
                array *= weights.item()
                array = xp.expand_dims(array, axis=list(range(len(weights.shape))))

            return array

        if not self.is_lazy:
            array = apply_ctf(self.array, ctf=ctf)
            axes_metadata = []

            if ctf.is_distribution:
                raise NotImplementedError

        elif ctf.is_distribution:
            values, weights, axes_metadata = ctf.parameter_series()

            values = da.from_array(values, chunks=(1,) * len(values.shape))
            weights = da.from_array(weights, chunks=(1,) * len(weights.shape))

            n = tuple(range(len(values.shape)))
            m = tuple(range(len(n), len(n) + len(self.shape)))

            array = da.blockwise(apply_ctf,
                                 n + m,
                                 self.array, m,
                                 values, n,
                                 weights, n,
                                 ctf=ctf,
                                 concatenate=True,
                                 meta=xp.array((), dtype=np.complex64), )
        else:
            array = self.array.map_blocks(apply_ctf,
                                          ctf=ctf,
                                          meta=xp.array((), dtype=np.complex64))

            axes_metadata = []

        # kernel = xp.asarray(ctf.evaluate_on_grid(extent=self.extent, gpts=self.gpts, sampling=self.sampling))

        d = self._copy_as_dict(copy_array=False)
        d['array'] = array  # fft2_convolve(self.array, kernel, overwrite_x=in_place)
        d['extra_axes_metadata'] = axes_metadata + d['extra_axes_metadata']
        return self.__class__(**d)

    def apply_detector_func(self, func, detectors, scan=None, **kwargs):
        detectors = validate_detectors(detectors)

        new_cls = [detector.measurement_type(self) for detector in detectors]
        new_cls_kwargs = [detector.measurement_kwargs(self) for detector in detectors]

        signatures = []
        output_sizes = {}
        meta = []
        i = 2
        for detector in detectors:
            shape = detector.measurement_shape(self)[self.num_extra_axes:]
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
        for potential_config in potential.get_configurations(lazy=self.is_lazy):
            #potential_config = potential_config.to_delayed()

            config_measurements = self.copy().apply_detector_func(transition_potential_multislice,
                                                                  potential=potential_config,
                                                                  detectors=detectors,
                                                                  transition_potentials=transitions,
                                                                  ctf=ctf)

            measurements.append(config_measurements)

        measurements = list(map(list, zip(*measurements)))
        measurements = [stack_measurements(measurements, FrozenPhononsAxis()) for measurements in measurements]

        #for i, (detector, measurement) in enumerate(zip(detectors, measurements)):
        #    if detector.ensemble_mean:
        #        measurements[i] = measurement.mean(0)

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
        detectors : detector or list of detectors
        start : int
        stop : int
        conjugate : bool

        Returns
        -------
        Waves object
            Wave function at the exit plane of the potential.
        """

        if detectors is None:
            if self.is_lazy:
                potential = potential.to_delayed()

            xp = get_array_module(self.array)

            return self.map_blocks(multislice,
                                   potential=potential,
                                   start=start, stop=stop,
                                   conjugate=conjugate,
                                   meta=xp.array((), dtype=xp.complex64))

        if detectors is None:
            detectors = [WavesDetector()]

        return multislice_and_detect_with_frozen_phonons(self, potential, detectors,
                                                         start=start, stop=stop, conjugate=conjugate)

    def to_zarr(self, url: str, overwrite: bool = False):
        """
        Write wave functions to a zarr file.

        Parameters
        ----------
        url : str
            Location of the data, typically a path to a local file. A URL can also include a protocol specifier like
            s3:// for remote data.
        overwrite : bool
            If given array already exists, overwrite=False will cause an error, where overwrite=True will replace the
            existing data.
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

        url : str
            Location of the data, typically a path to a local file. A URL can also include a protocol specifier like
            s3:// for remote data.
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

        axes = [element for i, element in enumerate(self.extra_axes_metadata) if not i in removed_axes]

        d = self._copy_as_dict(copy_array=False)
        d['array'] = self._array[items]
        d['extra_axes_metadata'] = axes
        return self.__class__(**d)

    def show(self, ax: Axes = None, **kwargs):
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

    def clone(self):
        if not self.is_lazy:
            raise RuntimeError()

        d = self._copy_as_dict(copy_array=False)
        d['array'] = clone(self.array, assume_layers=False)
        return self.__class__(**d)

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
    tilt : two float
        Small angle beam tilt [mrad].
    normalize : bool
        If True,
    device : str
        The plane waves will be build on this device.
    """

    def __init__(self,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 energy: float = None,
                 tilt: Tuple[float, float] = None,
                 normalize: bool = True,
                 device: str = None):

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_aperture = AntialiasAperture()
        self._device = _validate_device(device)
        self._normalize = normalize

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
            detectors = PixelatedDetector(fourier_space=False)

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

        if detectors is None:
            detectors = [WavesDetector()]

        waves = self.build(True)

        measurements = multislice_and_detect_with_frozen_phonons(waves, potential, detectors)

        if not lazy:
            measurements.compute()

        return measurements

    def build(self, lazy: bool = None) -> Waves:
        """
        Build the plane wave function as a Waves object.

        Parameters
        ----------
        lazy :

        Returns
        -------

        """
        xp = get_array_module(self._device)
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        def plane_wave(gpts, xp):
            if self._normalize:
                wave = xp.full(gpts, (1 / np.prod(gpts)).astype(xp.complex64), dtype=xp.complex64)
            else:
                wave = xp.ones(gpts, dtype=xp.complex64)

            return wave

        if validate_lazy(lazy):
            array = dask.delayed(plane_wave)(self.gpts, xp)
            array = da.from_delayed(array, shape=self.gpts, meta=xp.array((), dtype=xp.complex64))
        else:
            array = plane_wave(self.gpts, xp)

        return Waves(array, extent=self.extent, energy=self.energy)


class Probe(WavesLikeMixin):
    """
    Probe wave function

    The probe can represent a stack of electron probe wave functions for simulating scanning transmission
    electron microscopy.

    See the documentation for abtem.transfer.CTF for a description of the parameters related to the contrast transfer
    function.

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
    tilt : two float, optional
        Small angle beam tilt [mrad].
    device : str
        The probe wave functions will be build on this device.
    ctf : CTF
        Contrast transfer function object. Note that this can be specified
    kwargs :
        Provide the parameters of the contrast transfer function as keyword arguments. See the documentation for the
        CTF object.
    """

    def __init__(self,
                 extent: Union[float, Tuple[float, float]] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 energy: float = None,
                 tilt: Tuple[float, float] = None,
                 device: str = None,
                 ctf: CTF = None,
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
        self._extra_axes_metadata = []
        self._antialias_cutoff_gpts = None

    @property
    def metadata(self):
        return {'energy': self.energy, 'semiangle_cutoff': self.ctf.semiangle_cutoff}

    @property
    def shape(self):
        """ Shape of Waves. """
        return self.gpts

    @property
    def ctf(self) -> CTF:
        """ Probe contrast transfer function. """
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

    def _calculate_probe(self, positions, ctf_parameters=None, weights=None, sampling=None, gpts=None):
        xp = get_array_module(self._device)

        if ctf_parameters is not None:
            ctf = self.ctf.copy()
            ctf.set_parameters(ctf_parameters.item())
        else:
            ctf = self.ctf

        positions = xp.asarray(positions)
        xp = get_array_module(positions)
        positions = positions / xp.array(sampling).astype(np.float32)
        array = fft_shift_kernel(positions, shape=gpts)
        array *= ctf.evaluate_on_grid(gpts=gpts, sampling=sampling, xp=xp)

        array /= xp.sqrt(abs2(array).sum((-2, -1), keepdims=True))

        if ctf_parameters is not None:
            array *= weights.item()
            array = np.expand_dims(array, axis=tuple(range(len(weights.shape))))

        return ifft2(array)

    def build(self,
              positions: Union[AbstractScan, Sequence] = None,
              chunks: int = None,
              lazy: bool = None) -> Waves:

        """
        Build probe wave functions at the provided positions.

        Parameters
        ----------
        positions : scan object or array of xy-positions
            Positions of the probe wave functions
        chunks : int
            Specifies the number of wave functions in each chunk of a the created dask array. If None, the number
            of chunks are automatically estimated based on the "dask.chunk-size" parameter in the configuration.
        lazy : boolean, optional
            If True, create the wave functions lazily, otherwise, calculate instantly. If None, this defaults to the
            value set in the configuration file.

        Returns
        -------
        probe_wave_functions : Waves
        """

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        lazy = validate_lazy(lazy)

        validated_positions = self._validate_positions(positions, lazy=lazy, chunks=chunks)
        xp = get_array_module(self._device)

        if lazy and self.ctf.is_distribution:
            values, weights, axes_metadata = self.ctf.parameter_series()

            values = da.from_array(values, chunks=(1,) * len(values.shape))
            weights = da.from_array(weights, chunks=(1,) * len(weights.shape))

            n = tuple(range(len(values.shape)))
            m = tuple(range(len(n), len(n) + len(validated_positions.shape)))
            k = tuple(range(len(n) + len(m), len(n) + len(m) + 2))
            new_axes = dict(zip(k, self.gpts))

            array = da.blockwise(self._calculate_probe,
                                 n + m[:-1] + k,
                                 validated_positions, m,
                                 values, n,
                                 weights, n,
                                 sampling=self.sampling,
                                 gpts=self.gpts,
                                 concatenate=True,
                                 meta=xp.array((), dtype=np.complex64),
                                 new_axes=new_axes)

        elif lazy:
            drop_axis = len(validated_positions.shape) - 1
            new_axis = len(validated_positions.shape) - 1, len(validated_positions.shape)

            array = validated_positions.map_blocks(self._calculate_probe,
                                                   meta=xp.array((), dtype=np.complex64),
                                                   sampling=self.sampling,
                                                   gpts=self.gpts,
                                                   drop_axis=drop_axis,
                                                   new_axis=new_axis,
                                                   chunks=validated_positions.chunks[:-1] + (
                                                       (self.gpts[0],), (self.gpts[1],)))

            axes_metadata = []

        else:
            if self.ctf.is_distribution:
                raise NotImplementedError

            array = self._calculate_probe(validated_positions, sampling=self.sampling, gpts=self.gpts)
            axes_metadata = []

        if hasattr(positions, 'axes_metadata'):
            axes_metadata = axes_metadata + positions.axes_metadata
        else:
            axes_metadata = axes_metadata + [PositionsAxis()] * (len(validated_positions.shape) - 1)

        metadata = {'semiangle_cutoff': self.ctf.semiangle_cutoff}

        return Waves(array,
                     extent=self.extent,
                     energy=self.energy,
                     tilt=self.tilt,
                     extra_axes_metadata=axes_metadata,
                     metadata=metadata,
                     antialias_cutoff_gpts=self.antialias_cutoff_gpts)

    def multislice(self,
                   potential: Union[AbstractPotential, Atoms],
                   positions: AbstractScan = None,
                   detectors: AbstractDetector = None,
                   chunks: int = None,
                   lazy: bool = None) -> Union[AbstractMeasurement, Waves, List[Union[AbstractMeasurement, Waves]]]:
        """
        Build probe wave functions at the provided positions and run the multislice algorithm using the wave functions
        as initial.

        Parameters
        ----------
        potential : Potential or Atoms object
            The scattering potential.
        positions : scan object, array of xy-positions, optional
            Positions of the probe wave functions. If None, the positions are a single position at the center of the
            potential.
        detectors : detector, list of detectors, optional
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm. See abtem.measure.detect for a list of implemented detectors.
        chunks : int, optional
            Specifices the number of wave functions in each chunk of a the created dask array. If None, the number
            of chunks are automatically estimated based on the "dask.chunk-size" parameter in the configuration.
        lazy : boolean, optional
            If True, create the measurements lazily, otherwise, calculate instantly. If None, this defaults to the value
            set in the configuration file.

        Returns
        -------
        measurements : AbstractMeasurement or Waves or list of AbstractMeasurement
        """
        lazy = validate_lazy(lazy)

        potential = validate_potential(potential, self)

        if positions is None:
            positions = (self.extent[0] / 2, self.extent[1] / 2)

        positions = validate_scan(positions, self)

        if detectors is None:
            detectors = [WavesDetector()]

        waves = self.build(positions=positions, lazy=True, chunks=chunks)

        measurements = multislice_and_detect_with_frozen_phonons(waves, potential, detectors)

        if not lazy:
            measurements.compute()

        return measurements

    def scan(self,
             potential: Union[Atoms, AbstractPotential],
             scan: Union[AbstractScan, np.ndarray, Sequence] = None,
             detectors: Union[AbstractDetector, Sequence[AbstractDetector]] = None,
             chunks: int = None,
             lazy: bool = None) -> Union[AbstractMeasurement, Waves, List[Union[AbstractMeasurement, Waves]]]:
        """
        Build probe wave functions at the provided positions and propagate them through the potential.

        Parameters
        ----------
        potential : Potential or Atoms object
            The scattering potential.
        scan : scan object
            Positions of the probe wave functions. If None, the positions are a single position at the center of the
            potential.
        detectors : detector, list of detectors, optional
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm. See abtem.measure.detect for a list of implemented detectors.
        chunks : int, optional
            Specifices the number of wave functions in each chunk of a the created dask array. If None, the number
            of chunks are automatically estimated based on the "dask.chunk-size" parameter in the configuration.
        lazy : boolean, optional
            If True, create the measurements lazily, otherwise, calculate instantly. If None, this defaults to the value
            set in the configuration file.

        Returns
        -------
        #list_of_measurements : measurement, wave functions, list of measurements
        """

        if scan is None:
            scan = GridScan()

        if detectors is None:
            detectors = FlexibleAnnularDetector()

        return self.multislice(potential, scan, detectors, chunks=chunks, lazy=lazy)

    def profile(self, angle: float = 0.):
        """
        Creates a line profile through the center of the probe.

        Parameters
        ----------
        angle : float
            Angle to the x-axis of the line profile in radians.
        """

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
        """ Make a copy. """
        return deepcopy(self)

    def show(self, **kwargs):
        """
        Show the probe wave function.

        Parameters
        ----------
        kwargs : Additional keyword arguments for the abtem.plot.show_image function.
        """
        return self.build((self.extent[0] / 2, self.extent[1] / 2)).intensity().show(**kwargs)
