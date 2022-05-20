"""Module to describe electron waves and their propagation."""
import itertools
from copy import copy, deepcopy
from functools import partial
from numbers import Number
from typing import Union, Sequence, Tuple, List, Dict

import dask.array as da
import numpy as np
import zarr
from ase import Atoms
from matplotlib.axes import Axes

from abtem.core.antialias import AntialiasAperture
from abtem.core.axes import AxisMetadata, axis_from_dict, axis_to_dict
from abtem.core.backend import get_array_module, validate_device, copy_to_device, device_name_from_array_module
from abtem.core.blockwise import Ensemble, ensemble_blockwise
from abtem.core.complex import abs2
from abtem.core.dask import HasDaskArray, validate_lazy, ComputableList, validate_chunks
from abtem.core.energy import Accelerator
from abtem.core.fft import fft2, ifft2, fft_crop, fft2_interpolate
from abtem.core.grid import Grid
from abtem.measure.detect import AbstractDetector, validate_detectors, WavesDetector, \
    FlexibleAnnularDetector
from abtem.measure.measure import DiffractionPatterns, Images, AbstractMeasurement
from abtem.potentials.potentials import Potential, AbstractPotential, validate_potential
from abtem.waves.base import WavesLikeMixin
from abtem.waves.multislice import multislice_and_detect
from abtem.waves.scan import AbstractScan, GridScan, validate_scan
from abtem.waves.tilt import BeamTilt
from abtem.waves.transfer import CTF, Aperture, Aberrations, WaveTransferFunction


def stack_waves(waves, axes_metadata):
    if len(waves) == 0:
        return waves[0]
    array = np.stack([waves.array for waves in waves], axis=0)
    d = waves[0]._copy_as_dict(copy_array=False)
    d['array'] = array
    d['ensemble_axes_metadata'] = [axes_metadata] + waves[0].ensemble_axes_metadata
    return waves[0].__class__(**d)


def _get_lazy_measurements_from_arrays(arrays,
                                       waves,
                                       detectors,
                                       potential: AbstractPotential = None):
    def extract_measurement(array, index):
        array = array.item()[index].array
        return array

    measurements = []
    for i, detector in enumerate(detectors):

        meta = detector.measurement_meta(waves)
        shape = detector.measurement_shape(waves)

        new_axis = tuple(range(len(arrays.shape), len(arrays.shape) + len(shape)))
        drop_axis = tuple(range(len(arrays.shape), len(arrays.shape)))

        chunks = arrays.chunks + tuple((n,) for n in shape)

        array = arrays.map_blocks(extract_measurement,
                                  i,
                                  chunks=chunks,
                                  drop_axis=drop_axis,
                                  new_axis=new_axis,
                                  meta=meta)

        axes_metadata = []

        if potential is not None:
            axes_metadata += potential.ensemble_axes_metadata

        axes_metadata += waves.ensemble_axes_metadata

        axes_metadata += detector.measurement_axes_metadata(waves)

        cls = detector.measurement_type(waves)

        measurement = cls.from_array_and_metadata(array, axes_metadata=axes_metadata, metadata=waves.metadata)

        if hasattr(measurement, 'reduce_ensemble'):
            measurement = measurement.reduce_ensemble()

        measurement = measurement.squeeze()
        measurements.append(measurement)

    return measurements[0] if len(measurements) == 1 else ComputableList(measurements)


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
    ensemble_axes_metadata : list of AxesMetadata
        Metadata associated with an axis of
    """

    def __init__(self,
                 array: np.ndarray,
                 energy: float = None,
                 extent: Union[float, Tuple[float, float]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 tilt: Tuple[float, float] = (0., 0.),
                 fourier_space: bool = False,
                 ensemble_axes_metadata: List[AxisMetadata] = None,
                 metadata: Dict = None,
                 antialias_cutoff_gpts: Tuple[int, int] = None):

        if len(array.shape) < 2:
            raise RuntimeError('Wave function array should have 2 dimensions or more')

        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._antialias_cutoff_gpts = antialias_cutoff_gpts
        self._fourier_space = fourier_space

        super().__init__(array=array)

        if ensemble_axes_metadata is None:
            ensemble_axes_metadata = []

        if metadata is None:
            metadata = {}

        self._ensemble_axes_metadata = ensemble_axes_metadata
        self._metadata = metadata
        self._check_axes_metadata()

    @property
    def shape(self):
        return self._array.shape

    def default_ensemble_chunks(self):
        return self.array.chunks[:-2]

    def ensemble_blocks(self):
        return self.array,

    def ensemble_partial(self):
        d = self._copy_as_dict(copy_array=False)

        def build_waves(*args, **kwargs):
            array = args[0]
            return Waves(array, **kwargs)

        return partial(build_waves, **d)

    @property
    def fourier_space(self):
        return self._fourier_space

    @property
    def metadata(self) -> Dict:
        self._metadata['energy'] = self.energy
        return self._metadata

    @classmethod
    def from_array_and_metadata(cls, array, axes_metadata, metadata=None):
        energy = metadata['energy']
        sampling = axes_metadata[-2].sampling, axes_metadata[-1].sampling
        return cls(array, sampling=sampling, energy=energy, ensemble_axes_metadata=axes_metadata[:-2],
                   metadata=metadata)

    @property
    def device(self):
        return device_name_from_array_module(get_array_module(self.array))

    def renormalize(self, mode: str, in_place: bool = False):
        xp = get_array_module(self.device)

        if mode == 'intensity':
            f = xp.sqrt(abs2(self.array).sum((-2, -1), keepdims=True))

        elif mode == 'amplitude':
            f = xp.abs(self.array).sum((-2, -1), keepdims=True)

        else:
            raise RuntimeError()

        if in_place:
            waves = self
        else:
            waves = self.copy()

        waves._array /= f

        return waves

    def ensure_lazy(self):
        if self.is_lazy:
            return self

        d = self._copy_as_dict(copy_array=False)
        d['array'] = da.from_array(self.array)
        return self.__class__(**d)

    def ensure_fourier_space(self):
        if self.fourier_space:
            return self

        d = self._copy_as_dict(copy_array=False)
        d['array'] = fft2(self.array, overwrite_x=True)
        d['fourier_space'] = True
        return self.__class__(**d)

    def ensure_real_space(self):
        if not self.fourier_space:
            return self

        d = self._copy_as_dict(copy_array=False)
        d['array'] = ifft2(self.array, overwrite_x=True)
        d['fourier_space'] = False
        return self.__class__(**d)

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
        d['ensemble_axes_metadata'] = [element for i, element in enumerate(self.ensemble_axes_metadata) if
                                       i not in squeezed]
        return self.__class__(**d)

    def intensity(self) -> Images:
        """
        Calculate the intensity of the wave functions at the image plane.

        Returns
        -------
        intensity_images : Images
            The wave function intensity.
        """

        return Images(abs2(self.array), sampling=self.sampling, ensemble_axes_metadata=self.ensemble_axes_metadata,
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
                                                   ensemble_axes_metadata=self.ensemble_axes_metadata,
                                                   metadata=self.metadata)

        if block_direct:
            diffraction_patterns = diffraction_patterns.block_direct(radius=block_direct)

        return diffraction_patterns

    def apply_ctf(self,
                  ctf: WaveTransferFunction = None,
                  max_batch: str = 'auto',
                  **kwargs) -> 'Waves':
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

        if self.is_lazy:
            def apply_ctf(*args, ctf, waves_kwargs, ctf_ensemble_dims):
                ctf = ctf(*args[:ctf_ensemble_dims]).item()
                waves = Waves(args[ctf_ensemble_dims], **waves_kwargs)
                waves = ctf.apply(waves)
                return waves.array

            if isinstance(max_batch, int):
                max_batch = max_batch * self.gpts[0] * self.gpts[1]

            chunks = validate_chunks(ctf.ensemble_shape + self.shape,
                                     ctf.default_ensemble_chunks + tuple(max(chunk) for chunk in self.array.chunks),
                                     limit=max_batch,
                                     dtype=np.dtype('complex64'))

            blocks = tuple((block, (i,)) for i, block in enumerate(ctf.ensemble_blocks(chunks[:-len(self.shape)])))

            array = da.blockwise(apply_ctf,
                                 tuple(range(len(blocks) + len(self.shape))),
                                 *tuple(itertools.chain(*blocks)),
                                 self.array,
                                 tuple(range(len(blocks), len(blocks) + len(self.shape))),
                                 adjust_chunks={i: chunk for i, chunk in enumerate(chunks)},
                                 ctf=ctf.ensemble_partial(),
                                 ctf_ensemble_dims=len(ctf.ensemble_shape),
                                 waves_kwargs=self._copy_as_dict(copy_array=False),
                                 meta=np.array((), dtype=np.complex64))

            d = self._copy_as_dict(copy_array=False)
            d['array'] = array
            d['ensemble_axes_metadata'] = ctf.ensemble_axes_metadata + d['ensemble_axes_metadata']
            return self.__class__(**d)

        else:
            return ctf.apply(self)

    def multislice(self,
                   potential: AbstractPotential,
                   detectors: AbstractDetector = None,
                   start: int = 0,
                   stop: int = None,
                   conjugate: bool = False,
                   keep_ensemble_dims: bool = False,
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

        detectors = validate_detectors(detectors)

        if self.is_lazy:
            def multislice(*args, potential, waves_kwargs, detectors):
                potential = potential(*args[:2]).item()
                waves = Waves(args[2], **waves_kwargs)
                measurements = waves.multislice(potential, detectors=detectors, keep_ensemble_dims=True)
                measurements = (measurements,) if hasattr(measurements, 'array') else measurements
                arr = np.zeros((1,) * (len(args) + 1), dtype=object)
                arr.itemset(measurements)
                return arr

            chunks = potential.default_ensemble_chunks + self.array.chunks
            blocks = potential.ensemble_blocks(chunks[:-len(self.shape)])
            blocks = tuple((block, (i,)) for i, block in enumerate(blocks))
            arrays = da.blockwise(multislice,
                                  tuple(range(len(blocks) + len(self.shape) - 2)),
                                  *tuple(itertools.chain(*blocks)),
                                  self.array,
                                  tuple(range(len(blocks), len(blocks) + len(self.shape))),
                                  adjust_chunks={i: chunk for i, chunk in enumerate(chunks)},
                                  potential=potential.ensemble_partial(),
                                  detectors=detectors,
                                  waves_kwargs=self._copy_as_dict(copy_array=False),
                                  concatenate=True,
                                  meta=np.array((), dtype=np.complex64))

            print(arrays)

            return _get_lazy_measurements_from_arrays(arrays, self, detectors, potential=potential)
        else:

            # if potential.num_frozen_phonons > 1:
            #    waves.multislice(potential)
            # array = xp.zeros((len(potential),) + self.shape, dtype=np.complex64)

            measurements = multislice_and_detect(self,
                                                 potential=potential,
                                                 detectors=detectors,
                                                 keep_ensemble_dims=keep_ensemble_dims)

        return measurements[0] if len(measurements) == 1 else ComputableList(measurements)

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
                if key == 'ensemble_axes_metadata':
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
            ensemble_axes_metadata = [axis_from_dict(d) for d in f.attrs['ensemble_axes_metadata']]
            metadata = f.attrs['metadata']
            shape = f['array'].shape

        if chunks is None:
            chunks = (-1,) * (len(shape) - 2)

        array = da.from_zarr(url, component='array', chunks=chunks + (-1, -1))
        return cls(array=array, energy=energy, extent=extent, tilt=tilt, antialias_cutoff_gpts=antialias_cutoff_gpts,
                   ensemble_axes_metadata=ensemble_axes_metadata, metadata=metadata)

    def __getitem__(self, items) -> 'Waves':
        if isinstance(items, (Number, slice)):
            items = (items,)

        if len(self.array.shape) <= self.grid.dimensions:
            raise RuntimeError()

        removed_axes = []
        for i, item in enumerate(items):
            if isinstance(item, Number):
                removed_axes.append(i)

        axes = [element for i, element in enumerate(self.ensemble_axes_metadata) if not i in removed_axes]

        d = self._copy_as_dict(copy_array=False)
        d['array'] = self._array[items]
        d['ensemble_axes_metadata'] = axes
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
             'fourier_space': self.fourier_space,
             'antialias_cutoff_gpts': self.antialias_cutoff_gpts,
             'ensemble_axes_metadata': deepcopy(self._ensemble_axes_metadata),
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


class WavesBuilder(WavesLikeMixin, Ensemble):

    @property
    def metadata(self):
        raise NotImplementedError

    @property
    def ensembles(self):
        raise NotImplementedError

    @property
    def default_ensemble_chunks(self):
        chunks = ()
        for ensemble in self.ensembles:
            chunks += ensemble.default_ensemble_chunks
        return chunks

    @property
    def ensemble_axes_metadata(self):
        axes_metadata = []
        for ensemble in self.ensembles:
            axes_metadata += ensemble.ensemble_axes_metadata
        return axes_metadata

    @property
    def ensemble_shape(self):
        shape = ()
        for ensemble in self.ensembles:
            shape += ensemble.ensemble_shape

        return shape

    def ensemble_chunks(self, max_batch=None):
        chunks = self.default_ensemble_chunks

        shape = self.ensemble_shape

        chunks = chunks + (-1, -1)
        shape = shape + self.gpts

        if isinstance(max_batch, int):
            max_batch = max_batch * self.gpts[0] * self.gpts[1]

        return validate_chunks(shape, chunks, limit=max_batch, dtype=np.dtype('complex64'))[:-2]

    def ensemble_blocks(self, chunks=None):
        if chunks is None:
            chunks = self.ensemble_chunks()

        blocks = ()
        start = 0
        for ensemble in self.ensembles:
            stop = start + ensemble.ensemble_dims
            blocks = blocks + ensemble.ensemble_blocks(chunks=chunks[start:stop])
            start = stop

        return blocks

    @staticmethod
    def build_multislice_detect(*args, detectors, waves_builder, potential=None):

        waves = waves_builder[0](*[args[i] for i in waves_builder[1]]).item()
        waves = waves.build(lazy=False, keep_ensemble_dims=True)

        if potential is not None:
            potential = potential[0](*[args[i] for i in potential[1]]).item()
            measurements = multislice_and_detect(waves,
                                                 potential=potential,
                                                 detectors=detectors,
                                                 keep_ensemble_dims=True)
        else:
            measurements = tuple(detector.detect(waves) for detector in detectors)

        arr = np.zeros((1,) * len(args), dtype=object)
        arr.itemset(measurements)
        return arr

    def _lazy_build_multislice_detect(self, detectors, max_batch=None, potential=None):
        chunks = self.ensemble_chunks(max_batch)
        blocks = self.ensemble_blocks(chunks)

        if potential is not None:
            potential_partial = potential.ensemble_partial()
            blocks = potential.ensemble_blocks() + blocks
            chunks = potential.default_ensemble_chunks + chunks

            potential_ensemble_dims = potential.ensemble_dims
            partial_potential = potential_partial, range(potential_ensemble_dims)
        else:
            partial_potential = None
            potential_ensemble_dims = 0

        waves_builder = self.ensemble_partial(), range(potential_ensemble_dims,
                                                       potential_ensemble_dims + self.ensemble_dims)

        build_multislice_detect_partial = partial(self.build_multislice_detect,
                                                  potential=partial_potential,
                                                  waves_builder=waves_builder,
                                                  detectors=detectors)

        array = ensemble_blockwise(build_multislice_detect_partial,
                                   blocks,
                                   chunks)

        measurements = _get_lazy_measurements_from_arrays(array,
                                                          waves=self,
                                                          detectors=detectors,
                                                          potential=potential)

        return measurements


class PlaneWave(WavesBuilder):
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
        self._device = validate_device(device)
        self._normalize = normalize

    @property
    def ensembles(self):
        return ()

    def ensemble_partial(self):

        def plane_wave(*args, **kwargs):
            pw = PlaneWave(**kwargs)
            arr = np.zeros((1,) * len(args), dtype=object)
            arr.itemset(pw)
            return arr

        return partial(plane_wave, **self._copy_as_dict())

    @property
    def metadata(self):
        return {'energy': self.energy}

    @property
    def shape(self):
        return self.gpts

    def build(self, lazy: bool = None, max_batch='auto', keep_ensemble_dims: bool = True) -> Waves:
        """
        Build the plane wave function as a Waves object.

        Parameters
        ----------
        lazy :

        Returns
        -------

        """

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        lazy = validate_lazy(lazy)

        detectors = [WavesDetector()]

        if lazy:
            return self._lazy_build_multislice_detect(detectors=detectors, max_batch=max_batch)

        xp = get_array_module(self._device)

        if self._normalize:
            array = xp.full(self.gpts, (1 / np.prod(self.gpts)).astype(xp.complex64), dtype=xp.complex64)
        else:
            array = xp.ones(self.gpts, dtype=xp.complex64)

        return Waves(array, extent=self.extent, energy=self.energy)

    def multislice(self,
                   potential: Union[AbstractPotential, Atoms],
                   detectors: AbstractDetector = None,
                   lazy: bool = None,
                   max_batch='auto') -> Waves:
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

        if detectors is None:
            detectors = [WavesDetector()]

        potential = validate_potential(potential)
        lazy = validate_lazy(lazy)
        detectors = validate_detectors(detectors)

        self.grid.match(potential)
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        measurements = self._lazy_build_multislice_detect(detectors=detectors, potential=potential, max_batch=max_batch)

        if not lazy:
            measurements = measurements.compute()

        return measurements

    def _copy_as_dict(self):
        return {'extent': self.extent,
                'gpts': self.gpts,
                'energy': self.energy,
                'tilt': self.tilt,
                'normalize': self._normalize,
                'device': self._device}


class Probe(WavesBuilder):
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
                 normalize: str = 'intensity',
                 source_offset=None,
                 tilt: Tuple[float, float] = (0., 0.),
                 device: str = None,
                 aperture: float = 30.,
                 aberrations: Union[Aberrations, dict] = None,
                 **kwargs):

        self._accelerator = Accelerator(energy=energy)

        if np.isscalar(aperture):
            aperture = Aperture(semiangle_cutoff=aperture)
            aperture._accelerator = self._accelerator

        if aberrations is None:
            aberrations = {}

        if isinstance(aberrations, dict):
            aberrations = Aberrations(**aberrations, **kwargs)
            aberrations._accelerator = self._accelerator

        self._aperture = aperture
        self._aberrations = aberrations
        self._source_offset = source_offset
        self._beam_tilt = BeamTilt(tilt=tilt)
        self._normalize = normalize
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self._device = validate_device(device)
        self._antialias_cutoff_gpts = None

        self._scan = None

    @property
    def normalize(self):
        return self._normalize

    @property
    def ensembles(self):
        ensembles = ()

        if self.aberrations.ensemble_dims:
            ensembles += (self._aberrations,)

        if self.source_offset is not None:
            ensembles += (self.source_offset,)

        return ensembles

    def ensemble_partial(self):

        def probe(*args, source_offset=None, aberrations=None, kwargs):
            if aberrations is not None:
                kwargs['aberrations'] = aberrations[0](*[args[i] for i in aberrations[1]]).item()

            if source_offset is not None:
                kwargs['source_offset'] = source_offset[0](*[args[i] for i in source_offset[1]]).item()

            probe = Probe(**kwargs)
            arr = np.zeros((1,) * max(len(args), 1), dtype=object)
            arr.itemset(probe)
            return arr

        kwargs = self._copy_as_dict()
        start = 0
        if self.aberrations.ensemble_dims:
            del kwargs['aberrations']
            stop = start + self.aberrations.ensemble_dims
            aberrations = (self.aberrations.ensemble_partial(), tuple(range(start, stop)))
            start = stop
        else:
            aberrations = None

        if self.source_offset is not None:
            del kwargs['source_offset']
            stop = start + self.source_offset.ensemble_dims
            source_offset = (self.source_offset.ensemble_partial(), tuple(range(start, start + stop)))
        else:
            source_offset = None

        partial_probe = partial(probe,
                                aberrations=aberrations,
                                source_offset=source_offset,
                                kwargs=kwargs)

        return partial_probe

    @property
    def source_offset(self):
        return self._source_offset

    @source_offset.setter
    def source_offset(self, source_offset: AbstractScan):
        self._source_offset = source_offset

    @property
    def aperture(self) -> Aperture:
        return self._aperture

    @aperture.setter
    def aperture(self, aperture: Aperture):
        self._aperture = aperture

    @property
    def aberrations(self) -> Aberrations:
        """ Probe contrast transfer function. """
        return self._aberrations

    @aberrations.setter
    def aberrations(self, aberrations: Aberrations):
        """ Probe contrast transfer function. """
        self._aberrations = aberrations

    @property
    def metadata(self):
        return {'energy': self.energy}

    @property
    def shape(self):
        """ Shape of Waves. """
        return self.ensemble_shape + self.gpts

    def build(self,
              scan: Union[AbstractScan, Sequence] = None,
              max_batch: int = None,
              lazy: bool = None,
              keep_ensemble_dims: bool = False) -> Waves:

        """
        Build probe wave functions at the provided positions.

        Parameters
        ----------
        scan : scan object or array of xy-positions
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

        if scan is None:
            scan = (self.extent[0] / 2, self.extent[1] / 2)

        scan = validate_scan(scan, self)

        probe = self.copy()

        if probe.source_offset is None:
            probe.source_offset = scan

        detectors = [WavesDetector()]

        if lazy:
            return probe._lazy_build_multislice_detect(detectors=detectors, max_batch=max_batch)

        xp = get_array_module(probe.device)

        array = xp.ones(probe.gpts, dtype=xp.complex64)

        waves = Waves(array=array, energy=probe.energy, extent=probe.extent, fourier_space=True)

        waves = probe.source_offset.apply_fft_shift(waves)

        ctf = probe.aberrations * probe.aperture

        waves = ctf.apply(waves)

        waves = waves.renormalize(mode=self.normalize, in_place=True)

        waves = waves.ensure_real_space()

        if not keep_ensemble_dims:
            waves = waves.squeeze()

        return waves

    def multislice(self,
                   potential: Union[AbstractPotential, Atoms],
                   scan: AbstractScan = None,
                   detectors: AbstractDetector = None,
                   max_batch: Union[int, str] = 'auto',
                   lazy: bool = None) -> Union[AbstractMeasurement, Waves, List[Union[AbstractMeasurement, Waves]]]:
        """
        Build probe wave functions at the provided positions and run the multislice algorithm using the wave functions
        as initial.

        Parameters
        ----------
        potential : Potential or Atoms object
            The scattering potential.
        scan : scan object, array of xy-positions, optional
            Positions of the probe wave functions. If None, the positions are a single position at the center of the
            potential.
        detectors : detector, list of detectors, optional
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm. See abtem.measure.detect for a list of implemented detectors.
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

        potential = validate_potential(potential)
        self.grid.match(potential)

        if scan is None:
            scan = self.extent[0] / 2, self.extent[1] / 2

        scan = validate_scan(scan, self)

        if detectors is None:
            detectors = [WavesDetector()]

        lazy = validate_lazy(lazy)

        detectors = validate_detectors(detectors)

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        probe = self.copy()
        probe.source_offset = scan

        measurements = probe._lazy_build_multislice_detect(potential=potential, detectors=detectors,
                                                           max_batch=max_batch)

        if not lazy:
            measurements.compute()

        return measurements

    def scan(self,
             potential: Union[Atoms, AbstractPotential],
             scan: Union[AbstractScan, np.ndarray, Sequence] = None,
             detectors: Union[AbstractDetector, Sequence[AbstractDetector]] = None,
             max_batch: Union[int, str] = 'auto',
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
        max_batch : int, optional
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

        return self.multislice(potential, scan, detectors, max_batch=max_batch, lazy=lazy)

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

    def _copy_as_dict(self):
        new = {'extent': self.extent,
               'gpts': self.gpts,
               'sampling': self.sampling,
               'energy': self.energy,
               'tilt': self.tilt,
               'device': self.device,
               'aberrations': self.aberrations.copy(),
               'aperture': self.aperture.copy(),
               'source_offset': copy(self.source_offset)}

        return new

    def copy(self):
        """ Make a copy. """
        return self.__class__(**self._copy_as_dict())

    def show(self, **kwargs):
        """
        Show the probe wave function.

        Parameters
        ----------
        kwargs : Additional keyword arguments for the abtem.plot.show_image function.
        """
        return self.build((self.extent[0] / 2, self.extent[1] / 2)).intensity().show(**kwargs)
