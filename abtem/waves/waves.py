"""Module to describe electron waves and their propagation."""
import itertools
import warnings
from abc import abstractmethod
from copy import copy
from functools import partial
from typing import Union, Sequence, Tuple, List, Dict

import dask.array as da
import numpy as np
from ase import Atoms

from abtem.core.array import HasArray, validate_lazy, ComputableList
from abtem.core.axes import AxisMetadata, TiltAxis, AxisAlignedTiltAxis
from abtem.core.backend import get_array_module, validate_device
from abtem.core.chunks import validate_chunks
from abtem.core.complex import abs2
from abtem.core.distributions import Distribution
from abtem.core.energy import Accelerator
from abtem.core.fft import fft2, ifft2, fft_crop, fft_interpolate
from abtem.core.grid import Grid, validate_gpts, polar_spatial_frequencies
from abtem.core.intialize import initialize
from abtem.ionization.multislice import transition_potential_multislice_and_detect
from abtem.measure.detect import (
    AbstractDetector,
    validate_detectors,
    WavesDetector,
    FlexibleAnnularDetector,
)
from abtem.measure.measure import DiffractionPatterns, Images, AbstractMeasurement
from abtem.potentials.potentials import Potential, AbstractPotential, validate_potential
from abtem.waves.base import WavesLikeMixin
from abtem.waves.multislice import multislice_and_detect, FresnelPropagator
from abtem.waves.scan import AbstractScan, GridScan, validate_scan
from abtem.waves.tilt import validate_tilt
from abtem.waves.transfer import (
    Aperture,
    Aberrations,
    WaveTransform,
    CompositeWaveTransform,
    CTF,
    WaveRenormalization,
)
from abtem.core.complex import complex_exponential


def _extract_measurement(array, index):
    if array.size == 0:
        return array

    array = array.item()[index].array
    return array


def finalize_lazy_measurements(
    arrays, waves, detectors, extra_ensemble_axes_metadata=None, chunks=None
):
    measurements = []
    for i, detector in enumerate(detectors):

        meta = detector.measurement_meta(waves)
        shape = detector.measurement_shape(waves)

        new_axis = tuple(range(len(arrays.shape), len(arrays.shape) + len(shape)))
        drop_axis = tuple(range(len(arrays.shape), len(arrays.shape)))

        if chunks is None:
            chunks = arrays.chunks

        array = arrays.map_blocks(
            _extract_measurement,
            i,
            chunks=chunks + tuple((n,) for n in shape),
            drop_axis=drop_axis,
            new_axis=new_axis,
            meta=meta,
        )

        axes_metadata = []

        if extra_ensemble_axes_metadata is not None:
            axes_metadata += extra_ensemble_axes_metadata

        axes_metadata += waves.ensemble_axes_metadata

        axes_metadata += detector.measurement_axes_metadata(waves)

        cls = detector.measurement_type(waves)

        metadata = detector.measurement_metadata(waves)

        measurement = cls.from_array_and_metadata(
            array, axes_metadata=axes_metadata, metadata=metadata
        )

        if hasattr(measurement, "reduce_ensemble"):
            measurement = measurement.reduce_ensemble()

        measurements.append(measurement)

    return measurements[0] if len(measurements) == 1 else ComputableList(measurements)


class Waves(HasArray, WavesLikeMixin):
    """
    Waves define a batch of arbitrary 2D wave functions defined by a complex numpy or cupy array.

    Parameters
    ----------
    array : array
        Complex array defining one or more 2d wave functions. The second-to-last and last dimensions are the wave
        function y- and x-axis, respectively.
    energy : float
        Electron energy [eV].
    extent : one or two float
        Extent of wave functions in x and y [Å].
    sampling : one or two float
        Sampling of wave functions in x and y [1 / Å].
    tilt : two float, optional
        Small angle beam tilt [mrad]. Implemented by shifting the wave function at every slice. Default is (0., 0.).
    fourier_space : bool, optional
        If True, the wave functions are assumed to be represented in Fourier space instead of real space.
    ensemble_axes_metadata : list of AxesMetadata
        Axis metadata for each ensemble axis. The axis metadata must be compatible with the shape of the array.
    metadata : dict
        A dictionary defining simulation metadata. All items will be added to the metadata of measurements derived from
        the waves.
    """

    _base_dims = 2

    def __init__(
        self,
        array: np.ndarray,
        energy: float = None,
        extent: Union[float, Tuple[float, float]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        fourier_space: bool = False,
        ensemble_axes_metadata: List[AxisMetadata] = None,
        metadata: Dict = None,
    ):

        if len(array.shape) < 2:
            raise RuntimeError("Wave function array should have 2 dimensions or more")

        self._array = array
        self._grid = Grid(
            extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True
        )
        self._accelerator = Accelerator(energy=energy)
        self._ensemble_axes_metadata = (
            [] if ensemble_axes_metadata is None else ensemble_axes_metadata
        )
        self._metadata = {} if metadata is None else metadata

        self._fourier_space = fourier_space
        self.check_axes_metadata()

    # def fourier_space_measurements(self):
    #     waves = self.ensure_fourier_space(in_place=False)
    #     return DiffractionPatterns(waves.array, sampling=self.sampling,
    #                                ensemble_axes_metadata=self.ensemble_axes_metadata,
    #                                metadata=self.metadata)

    @property
    def base_tilt(self):
        return (
            self.metadata.get("base_tilt_x", 0.0),
            self.metadata.get("base_tilt_y", 0.0),
        )

    @property
    def tilt_axes(self):
        return tuple(
            i
            for i, axis in enumerate(self.ensemble_axes_metadata)
            if isinstance(axis, (TiltAxis, AxisAlignedTiltAxis))
        )

    @property
    def tilt_axes_metadata(self):
        return [self.ensemble_axes_metadata[i] for i in self.tilt_axes]

    def fresnel_propagator(self, thickness: float) -> FresnelPropagator:
        return FresnelPropagator(self, thickness)

    def complex_images(self):
        waves = self.ensure_real_space(in_place=False)
        return Images(
            waves.array,
            sampling=self.sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self.metadata,
        )

    @property
    def ensemble_axes_metadata(self):
        return self._ensemble_axes_metadata

    @property
    def fourier_space(self):
        return self._fourier_space

    @property
    def metadata(self) -> Dict:
        self._metadata["energy"] = self.energy
        return self._metadata

    def from_partitioned_args(self):
        d = self.copy_kwargs(exclude=("array", "extent"))
        return partial(lambda *args, **kwargs: self.__class__(args[0], **kwargs), **d)

    @classmethod
    def from_array_and_metadata(cls, array, axes_metadata, metadata=None):
        energy = metadata["energy"]
        sampling = axes_metadata[-2].sampling, axes_metadata[-1].sampling
        return cls(
            array,
            sampling=sampling,
            energy=energy,
            ensemble_axes_metadata=axes_metadata[:-2],
            metadata=metadata,
        )

    def _angular_grid(self):
        xp = get_array_module(self.array)
        alpha, phi = polar_spatial_frequencies(self.gpts, self.sampling, xp=xp)
        alpha *= self.wavelength
        return alpha, phi

    def convolve(
        self,
        kernel: np.ndarray,
        axes_metadata: List[AxisMetadata],
        out_space: str = "in_space",
    ):

        if out_space == "in_space":
            fourier_space_out = self.fourier_space
        elif out_space in ("fourier_space", "real_space"):
            fourier_space_out = out_space == "fourier_space"
        else:
            raise ValueError

        xp = get_array_module(self.device)

        waves = self.ensure_fourier_space(in_place=False)

        waves_dims = tuple(range(len(kernel.shape) - 2))
        kernel_dims = tuple(
            range(
                len(kernel.shape) - 2,
                len(waves.array.shape) - 2 + len(kernel.shape) - 2,
            )
        )

        array = xp.expand_dims(waves.array, axis=waves_dims) * xp.expand_dims(
            kernel, axis=kernel_dims
        )

        if not fourier_space_out:
            array = ifft2(array, overwrite_x=False)

        d = waves.copy_kwargs(exclude=("array",))
        d["fourier_space"] = fourier_space_out
        d["array"] = array
        d["ensemble_axes_metadata"] = axes_metadata + d["ensemble_axes_metadata"]

        return waves.__class__(**d)

    def renormalize(self, space: str = "fourier"):
        xp = get_array_module(self.device)

        fourier_space = self.fourier_space

        if space == "fourier":
            waves = self.ensure_fourier_space()
            f = xp.sqrt(abs2(waves.array).sum((-2, -1), keepdims=True))
            waves = waves / f
            if not fourier_space:
                waves = waves.ensure_real_space()
        elif space == "real":
            raise NotImplementedError
        else:
            raise ValueError()

        return waves

    def tile(
        self, repetitions: Tuple[int, int], normalization: str = "values"
    ) -> "Waves":
        """
        Tile wave functions.

        Parameters
        ----------
        repetitions : two int
            The number of repetitions of the wave functions along the x- and y-axis.
        normalization : {'intensity', 'values'}


        Returns
        -------
        tiled_wave_functions : Waves
        """

        d = self.copy_kwargs(exclude=("array", "extent"))
        xp = get_array_module(self.device)

        if self.is_lazy:
            tile_func = da.tile
        else:
            tile_func = xp.tile

        array = tile_func(self.array, (1,) * len(self.ensemble_shape) + repetitions)

        if hasattr(array, "rechunk"):
            array = array.rechunk(array.chunks[:-2] + (-1, -1))

        d["array"] = array

        if normalization == "intensity":
            d["array"] /= xp.asarray(np.prod(repetitions))
        elif normalization != "values":
            raise ValueError()

        return self.__class__(**d)

    def ensure_fourier_space(self, in_place: bool = False):
        if self.fourier_space:
            return self

        d = self.copy_kwargs(exclude=("array",))
        d["array"] = fft2(self.array, overwrite_x=in_place)
        d["fourier_space"] = True
        return self.__class__(**d)

    def ensure_real_space(self, in_place: bool = False):
        if not self.fourier_space:
            return self

        d = self.copy_kwargs(exclude=("array",))
        d["array"] = ifft2(self.array, overwrite_x=in_place)
        d["fourier_space"] = False
        return self.__class__(**d)

    def phase_shift(self, shift):
        """
        Shift the phase of the wave functions by

        Parameters
        ----------
        shift :

        Returns
        -------

        """
        xp = get_array_module(self.array)
        arr = xp.exp(1.0j * shift) * self.array

        kwargs = self.copy_kwargs(exclude=("array",))

        return self.__class__(array=arr, **kwargs)

    def intensity(self) -> Images:
        """
        Calculate the intensity of the wave functions at the image plane.

        Returns
        -------
        intensity_images : Images
            The wave function intensity.
        """

        array = abs2(self.array)

        metadata = copy(self.metadata)

        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"

        return Images(
            array,
            sampling=self.sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=metadata,
        )

    def as_complex_image(self):
        """
        Calculate the complex array of the wave functions at the image plane.

        Rerturns
        -------
        intensity_images : Images
            The wave function intensity.
        """

        array = self.array

        metadata = copy(self.metadata)

        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"

        return Images(
            array,
            sampling=self.sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=metadata,
        )

    def downsample(
        self,
        max_angle: Union[str, float] = "cutoff",
        gpts: Tuple[int, int] = None,
        normalization: str = "values",
    ) -> "Waves":
        """
        Downsample the wave function to a lower maximum scattering angle.

        Parameters
        ----------
        max_angle : {'cutoff', 'valid'} or float, optional
            If not False, the scattering matrix is downsampled to a maximum given scattering angle after running the
            multislice algorithm.

                ``cutoff`` :
                    Downsample to the antialias cutoff scattering angle.

                ``valid`` :
                    Downsample to the largest rectangle inside the circle with a radius defined by the antialias
                    cutoff scattering angle.

                float :
                    Downsample to a maximum scattering angle specified by a float.

        gpts : two int, optional
            Number of grid points of the waves after downsampling. If given, `max_angle` is not used.

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

        if gpts is None:
            gpts = self._gpts_within_angle(max_angle)

        if self.is_lazy:
            array = self.array.map_blocks(
                fft_interpolate,
                new_shape=gpts,
                normalization=normalization,
                chunks=self.array.chunks[:-2] + gpts,
                meta=xp.array((), dtype=xp.complex64),
            )
        else:
            array = fft_interpolate(
                self.array, new_shape=gpts, normalization=normalization
            )

        kwargs = self.copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        kwargs["sampling"] = (self.extent[0] / gpts[0], self.extent[1] / gpts[1])
        return self.__class__(**kwargs)

    def diffraction_patterns(
        self,
        max_angle: Union[str, float, None] = "cutoff",
        block_direct: Union[bool, float] = False,
        fftshift: bool = True,
        parity: str = "same",
    ) -> DiffractionPatterns:
        """
        Calculate the intensity of the wave functions at the diffraction plane.

        Parameters
        ----------
        max_angle : {'cutoff', 'valid'} or float
            Maximum scattering angle of the diffraction patterns.

                ``cutoff`` :
                    The maximum scattering angle will be the cutoff of the antialias aperture.

                ``valid`` :
                    The maximum scattering angle will be the largest rectangle that fits inside the circular antialias
                    aperture.

        block_direct : bool or float
            If true the direct beam is masked.
        fftshift : bool
            If true, shift the zero-angle component to the center of the diffraction patterns.
        parity : {'same', 'even', 'odd', 'none'}
            The parity of the shape of the diffraction patterns.

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
        new_gpts = self._gpts_within_angle(max_angle, parity=parity)
        validate_gpts(new_gpts)

        if self.is_lazy:
            pattern = self.array.map_blocks(
                _diffraction_pattern,
                new_gpts=new_gpts,
                fftshift=fftshift,
                chunks=self.array.chunks[:-2] + ((new_gpts[0],), (new_gpts[1],)),
                meta=xp.array((), dtype=xp.float32),
            )
        else:
            pattern = _diffraction_pattern(
                self.array, new_gpts=new_gpts, fftshift=fftshift
            )

        metadata = copy(self.metadata)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"

        diffraction_patterns = DiffractionPatterns(
            pattern,
            sampling=self.fourier_space_sampling,
            fftshift=fftshift,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=metadata,
        )

        if block_direct:
            diffraction_patterns = diffraction_patterns.block_direct(
                radius=block_direct
            )

        return diffraction_patterns

    def as_complex_diffraction(
        self,
        max_angle: Union[str, float, None] = "cutoff",
        block_direct: Union[bool, float] = False,
        fftshift: bool = True,
        parity: str = "same",
    ) -> DiffractionPatterns:
        """
        Calculate the complex array of the wave functions at the diffraction plane.

        Parameters
        ----------
        max_angle : {'cutoff', 'valid'} or float
            Maximum scattering angle of the diffraction patterns.

                ``cutoff`` :
                    The maximum scattering angle will be the cutoff of the antialias aperture.

                ``valid`` :
                    The maximum scattering angle will be the largest rectangle that fits inside the circular antialias
                    aperture.

        block_direct : bool or float
            If true the direct beam is masked.
        fftshift : bool
            If true, shift the zero-angle component to the center of the diffraction patterns.
        parity : {'same', 'even', 'odd', 'none'}
            The parity of the shape of the diffraction patterns.

        Returns
        -------
        diffraction_patterns : DiffractionPatterns
        """

        def _as_complex_diffraction(array, new_gpts, fftshift):
            array = self.array
            xp = get_array_module(array)

            array = fft2(xp.fft.fftshift(array))

            if array.shape[-2:] != new_gpts:
                array = fft_crop(array, new_shape=array.shape[:-2] + new_gpts)

            if fftshift:
                array = xp.fft.ifftshift(array, axes=(-1, -2))

            return array

        xp = get_array_module(self.array)
        new_gpts = self._gpts_within_angle(max_angle, parity=parity)
        validate_gpts(new_gpts)

        if self.is_lazy:
            pattern = self.array.map_blocks(
                _as_complex_diffraction,
                new_gpts=new_gpts,
                fftshift=fftshift,
                chunks=self.array.chunks[:-2] + ((new_gpts[0],), (new_gpts[1],)),
                meta=xp.array((), dtype=xp.float32),
            )
        else:
            pattern = _as_complex_diffraction(
                self.array, new_gpts=new_gpts, fftshift=fftshift
            )

        metadata = copy(self.metadata)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"

        diffraction_patterns = DiffractionPatterns(
            pattern,
            sampling=self.fourier_space_sampling,
            fftshift=fftshift,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=metadata,
        )

        if block_direct:
            diffraction_patterns = diffraction_patterns.block_direct(
                radius=block_direct
            )

        return diffraction_patterns

    @staticmethod
    def _apply_wave_transform(*args, waves_partial, transform_partial):
        waves = waves_partial(*args[-1:])
        transform = transform_partial(*(arg.item() for arg in args[:-1]))
        waves = transform.apply(waves)
        return waves.array

    def apply_transform(
        self, transform: WaveTransform, max_batch: int = None
    ) -> "Waves":
        """
        Apply wave function transformation to the wave functions.

        Parameters
        ----------
        transform : WaveTransform
            Wave function transformation to apply.
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array. If None, the number of chunks are
            automatically estimated based on "dask.chunk-size" in the user configuration.

        Returns
        -------
        transformed_waves : Waves
        """

        if self.is_lazy:
            if isinstance(max_batch, int):
                max_batch = max_batch * self.gpts[0] * self.gpts[1]

            chunks = validate_chunks(
                transform.ensemble_shape + self.shape,
                transform.default_ensemble_chunks
                + tuple(max(chunk) for chunk in self.array.chunks),
                limit=max_batch,
                dtype=np.dtype("complex64"),
            )

            args = tuple(
                (arg, (i,))
                for i, arg in enumerate(
                    transform.partition_args(chunks[: -len(self.shape)])
                )
            )

            xp = get_array_module(self.device)

            kwargs = self.copy_kwargs(exclude=("array",))

            array = da.blockwise(
                self._apply_wave_transform,
                tuple(range(len(args) + len(self.shape))),
                *tuple(itertools.chain(*args)),
                self.array,
                tuple(range(len(args), len(args) + len(self.shape))),
                adjust_chunks={i: chunk for i, chunk in enumerate(chunks)},
                transform_partial=transform.from_partitioned_args(),
                waves_partial=self.from_partitioned_args(),  # noqa
                meta=xp.array((), dtype=np.complex64)
            )

            kwargs["array"] = array
            kwargs["ensemble_axes_metadata"] = (
                transform.ensemble_axes_metadata + kwargs["ensemble_axes_metadata"]
            )
            return self.__class__(**kwargs)
        else:
            return transform.apply(self)

    def apply_ctf(self, ctf: CTF = None, max_batch: str = None, **kwargs) -> "Waves":
        """
        Apply the aberrations and apertures of a Contrast Transfer Function to the wave functions.

        Parameters
        ----------
        ctf : CTF, optional
            Contrast Transfer Function to be applied.
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array. If None, the number of chunks are
            automatically estimated based on "dask.chunk-size" in the user configuration.
        kwargs :
            Provide the parameters of the contrast transfer function as keyword arguments. See `abtem.transfer.CTF`.

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

        return self.apply_transform(ctf, max_batch=max_batch)

    @staticmethod
    def _lazy_multislice(*args, potential_partial, waves_partial, detectors):
        potential = potential_partial(*(arg.item() for arg in args[:1]))
        waves = waves_partial(*args[-1:])

        measurements = waves.multislice(
            potential, detectors=detectors, keep_ensemble_dims=True
        )
        measurements = (
            (measurements,) if hasattr(measurements, "array") else measurements
        )
        arr = np.zeros((1,) * (len(args) - 1), dtype=object)
        arr.itemset(measurements)
        return arr

    def multislice(
        self,
        potential: AbstractPotential,
        detectors: Union[AbstractDetector, List[AbstractDetector]] = None,
        conjugate: bool = False,
        transpose: bool = False,
    ) -> "Waves":
        """
        Propagate and transmit wave function through the provided potential using the multislice algorithm.

        Parameters
        ----------
        potential : Potential
            The potential through which to propagate the wave function.
        detectors : detector or list of detectors
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm. See abtem.measure.detect for a list of implemented detectors.
        conjugate : bool, optional
            If True, use the conjugate of the transmission function. Default is False.
        transpose : bool, optional
            If True, reverse the order of propagation and transmission. Default is False.

        Returns
        -------
        exit_waves : Waves
            Wave function(s) at the exit plane(s) of the potential.
        """

        potential = validate_potential(potential, self)

        detectors = validate_detectors(detectors)

        if self.is_lazy:
            blocks = potential.partition_args()
            blocks = tuple((block, (i,)) for i, block in enumerate(blocks))

            num_new_symbols = len(potential.ensemble_shape)
            extra_ensemble_axes_metadata = potential.ensemble_axes_metadata

            if len(potential.exit_planes) > 1:
                new_axes = {num_new_symbols: (potential.num_exit_planes,)}
                num_new_symbols += 1
                extra_ensemble_axes_metadata = extra_ensemble_axes_metadata + [
                    potential.exit_planes_axes_metadata
                ]
            else:
                new_axes = None

            arrays = da.blockwise(
                self._lazy_multislice,
                tuple(range(num_new_symbols + len(self.shape) - 2)),
                *tuple(itertools.chain(*blocks)),
                self.array,
                tuple(range(num_new_symbols, num_new_symbols + len(self.shape))),
                potential_partial=potential.from_partitioned_args(),
                new_axes=new_axes,
                detectors=detectors,  # noqa
                conjugate=conjugate,  # noqa
                transpose=transpose,  # noqa
                waves_partial=self.from_partitioned_args(),  # noqa
                concatenate=True,
                meta=np.array((), dtype=np.complex64)
            )

            return finalize_lazy_measurements(
                arrays,
                self,
                detectors,
                extra_ensemble_axes_metadata=extra_ensemble_axes_metadata,
            )
        else:
            measurements = multislice_and_detect(
                self,
                potential=potential,
                detectors=detectors,
                conjugate=conjugate,
                transpose=transpose,
            )
            measurements = tuple(
                measurement.reduce_ensemble()
                if hasattr(measurement, "reduce_ensemble")
                else measurement
                for measurement in measurements
            )

        return (
            measurements[0] if len(measurements) == 1 else ComputableList(measurements)
        )

    def show(self, **kwargs):
        """
        Show the wave function intensities.

        kwargs :
            Keyword arguments for the `abtem.measure.Images.show` method.
        """
        return self.intensity().show(**kwargs)


class WavesBuilder(WavesLikeMixin):
    def __init__(self, transforms):

        if transforms is None:
            transforms = []

        if isinstance(transforms, list):
            transforms = CompositeWaveTransform(transforms)

        self._transforms = transforms

    @property
    @abstractmethod
    def named_transforms(self):
        pass

    @property
    def extra_transforms(self):
        return [
            transform
            for transform in self.transforms
            if not transform in self.named_transforms
        ]

    @abstractmethod
    def metadata(self):
        pass

    @abstractmethod
    def base_waves_partial(self):
        pass

    def insert_transform(self, transform, index=-1):
        self._transforms.insert_transform(index, transform)
        return self

    @property
    def transforms(self):
        return self._transforms

    @property
    def ensemble_axes_metadata(self):
        return self.transforms.ensemble_axes_metadata

    @property
    def ensemble_shape(self):
        return self.transforms.ensemble_shape

    @staticmethod
    def _build_waves_multislice_detect(*args, partials, multislice_func, detectors):

        waves = partials["base"][0]()

        transform = partials["transforms"][0](
            *[args[i] for i in partials["transforms"][1]]
        ).item()

        if "potential" in partials.keys():
            potential = partials["potential"][0](
                *[args[i] for i in partials["potential"][1]]
            ).item()
        else:
            potential = None

        waves = transform.apply(waves)

        waves = waves.ensure_real_space()

        if potential is not None:
            measurements = multislice_func(
                waves, potential=potential, detectors=detectors
            )

        else:
            measurements = tuple(detector.detect(waves) for detector in detectors)

        arr = np.zeros((1,) * (len(args) + 1), dtype=object)
        arr.itemset(measurements)
        return arr

    def lazy_build_multislice_detect(
        self,
        detectors,
        max_batch=None,
        potential=None,
        multislice_func=multislice_and_detect,
    ):

        args = ()
        symbols = ()
        adjust_chunks = {}
        extra_ensemble_axes_metadata = []
        new_axes = {}
        partials = {"base": (self.base_waves_partial(), ())}
        max_arg_index = 0
        max_symbol = 0

        if potential is not None:
            # add potential args
            potential_symbols = tuple(range(0, max(potential.ensemble_dims, 1)))
            partials["potential"] = potential.wrapped_from_partitioned_args(), (0,)
            max_arg_index += 1
            max_symbol += 1

            args += potential.partition_args()[0], potential_symbols

            if potential.ensemble_shape:
                symbols += potential_symbols
                adjust_chunks[0] = potential.default_ensemble_chunks[0]
                extra_ensemble_axes_metadata += potential.ensemble_axes_metadata

            # add exit plane args
            if len(potential.exit_planes) > 1:
                symbols += tuple(
                    range(max(potential_symbols) + 1, max(potential_symbols) + 2)
                )
                new_axes[1] = (len(potential.exit_planes),)
                extra_ensemble_axes_metadata += [potential.exit_planes_axes_metadata]
                max_symbol += 1

        # add transform args
        transform_arg_indices = tuple(
            range(max_arg_index, max_arg_index + self.transforms.ensemble_dims)
        )
        partials["transforms"] = (
            self.transforms.wrapped_from_partitioned_args(),
            transform_arg_indices,
        )
        transform_symbols = tuple(
            range(max_symbol, max_symbol + self.transforms.ensemble_dims)
        )
        transform_chunks = self.transforms.ensemble_chunks(
            max_batch, base_shape=self.gpts
        )

        args += tuple(
            itertools.chain(
                *tuple(
                    (block, (i,))
                    for i, block in zip(
                        transform_symbols,
                        self.transforms.partition_args(transform_chunks),
                    )
                )
            )
        )

        symbols += transform_symbols
        adjust_chunks = {
            **adjust_chunks,
            **{i: c for i, c in zip(transform_symbols, transform_chunks)},
        }

        partial_build = partial(
            self._build_waves_multislice_detect,
            partials=partials,
            multislice_func=multislice_func,
            detectors=detectors,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Increasing number of chunks")
            array = da.blockwise(
                partial_build,
                symbols,
                *args,
                adjust_chunks=adjust_chunks,
                new_axes=new_axes,
                concatenate=True,
                meta=np.array((), dtype=object)
            )

        return finalize_lazy_measurements(
            array,
            waves=self,
            detectors=detectors,
            extra_ensemble_axes_metadata=extra_ensemble_axes_metadata,
        )


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
        Small angle beam tilt [mrad]. Implemented by shifting the wave function at every slice. Default is (0., 0.).
    device : str, optional
        The wave function data is stored on this device. The default is determined by the user configuration.
    """

    def __init__(
        self,
        extent: Union[float, Tuple[float, float]] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        energy: float = None,
        normalize: bool = False,
        tilt: Tuple[float, float] = (0.0, 0.0),
        device: str = None,
        extra_transforms=None,
    ):

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)
        self._beam_tilt = validate_tilt(tilt=tilt)
        self._normalize = normalize
        self._device = validate_device(device)

        super().__init__(transforms=extra_transforms)

    @property
    def named_transforms(self):
        return CompositeWaveTransform([])

    @property
    def metadata(self):
        return {"energy": self.energy}

    @property
    def shape(self):
        return self.gpts

    @property
    def normalize(self):
        return self._normalize

    def base_waves_partial(self):
        def base_plane_wave(gpts, extent, energy, normalize, device):
            xp = get_array_module(device)

            if normalize:
                array = xp.full(
                    gpts, (1 / np.prod(gpts)).astype(xp.complex64), dtype=xp.complex64
                )
            else:
                array = xp.ones(gpts, dtype=xp.complex64)

            return Waves(array=array, energy=energy, extent=extent, fourier_space=False)

        return partial(
            base_plane_wave,
            gpts=self.gpts,
            extent=self.extent,
            energy=self.energy,
            normalize=self.normalize,
            device=self.device,
        )

    def build(
        self, lazy: bool = None, max_batch="auto", keep_ensemble_dims: bool = False
    ) -> Waves:
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

        if lazy:
            detectors = [WavesDetector()]
            return self.lazy_build_multislice_detect(
                detectors=detectors, max_batch=max_batch
            )

        waves = self.base_waves_partial()()
        return waves

    def multislice(
        self,
        potential: Union[AbstractPotential, Atoms],
        detectors: AbstractDetector = None,
        lazy: bool = None,
        max_batch: Union[int, str] = "auto",
        ctf: CTF = None,
        transition_potentials=None,
    ) -> Waves:
        """
        Build plane wave function and propagate it through the potential. The grid of the two will be matched.

        Parameters
        ----------
        potential : AbstractPotential, Atoms
            A potential as an AbstractPotential. The potential may also be provided as `ase.Atoms`,
        detectors : detector, list of detectors, optional
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm. See abtem.measure.detect for a list of implemented detectors.
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

        if transition_potentials:
            multislice_func = partial(
                transition_potential_multislice_and_detect,
                transition_potentials=transition_potentials,
                ctf=ctf,
            )
        else:
            multislice_func = multislice_and_detect

        if lazy:
            measurements = self.lazy_build_multislice_detect(
                detectors=detectors,
                max_batch=max_batch,
                potential=potential,
                multislice_func=multislice_func,
            )

        else:
            waves = self.build(lazy=False)
            measurements = waves.multislice(
                potential=potential, detectors=detectors
            )  ##multislice_func(waves, )

        return measurements  # [0] if len(measurements) == 1 else measurements


class Probe(WavesBuilder):
    """
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
    device : str, optional
        The probe wave functions will be build and stored on this device.
    ctf : CTF
        Contrast transfer function object. Note that this can be specified
    kwargs :
        Provide the parameters of the contrast transfer function as keyword arguments. See the documentation for the
        CTF object.
    """

    def __init__(
        self,
        extent: Union[float, Tuple[float, float]] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        energy: float = None,
        source_offset=None,
        tilt: Tuple[Union[float, Distribution], Union[float, Distribution]] = (
            0.0,
            0.0,
        ),
        device: str = None,
        semiangle_cutoff: float = 30.0,
        taper: float = 2.0,
        aperture: Aperture = None,
        aberrations: Union[Aberrations, dict] = None,
        extra_transforms=None,
        **kwargs
    ):

        self._accelerator = Accelerator(energy=energy)

        if aperture is None:
            aperture = Aperture(semiangle_cutoff=semiangle_cutoff, taper=taper)

        aperture._accelerator = self._accelerator

        if aberrations is None:
            aberrations = {}

        if isinstance(aberrations, dict):
            aberrations = Aberrations(**aberrations, **kwargs)

        aberrations._accelerator = self._accelerator

        self._aperture = aperture
        self._aberrations = aberrations
        self._source_offset = source_offset
        self._tilt = validate_tilt(tilt=tilt)

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self._device = validate_device(device)

        if extra_transforms is None:
            extra_transforms = []

        transforms = extra_transforms + [self._tilt, self.aperture, self.aberrations]

        super().__init__(transforms=transforms)

        self.accelerator.match(self.aperture)

    @classmethod
    def from_ctf(cls, ctf, **kwargs):
        return cls(aperture=ctf.aperture, aberrations=ctf.aberrations, **kwargs)

    @property
    def ctf(self):
        return CTF(
            semiangle_cutoff=self.aperture.semiangle_cutoff,
            aberrations=self.aberrations,
            energy=self.energy,
        )

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
        """Probe contrast transfer function."""
        return self._aberrations

    @aberrations.setter
    def aberrations(self, aberrations: Aberrations):
        """Probe contrast transfer function."""
        self._aberrations = aberrations

    @property
    def metadata(self):
        return {"energy": self.energy, **self.aperture.metadata, **self._tilt.metadata}

    @property
    def base_shape(self):
        """Shape of Waves."""
        return self.gpts

    def base_waves_partial(self):
        def base_probe(gpts, extent, energy, device, metadata):
            xp = get_array_module(device)

            array = xp.ones(gpts, dtype=xp.complex64)

            return Waves(
                array=array,
                energy=energy,
                extent=extent,
                fourier_space=True,
                metadata=metadata,
            )

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        return partial(
            base_probe,
            gpts=self.gpts,
            extent=self.extent,
            energy=self.energy,
            device=self.device,
            metadata=self.metadata,
        )

    def build(
        self,
        scan: Union[tuple, str, AbstractScan] = None,
        max_batch: Union[int, str] = None,
        lazy: bool = None,
    ) -> Waves:

        """
        Build probe wave functions at the provided positions.

        Parameters
        ----------
        scan : scan object or array of xy-positions
            Positions of the probe wave functions
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array. If None, the number of chunks are
            automatically estimated based on "dask.chunk-size" in the user configuration.
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

        if not isinstance(scan, AbstractScan):
            squeeze = (-3,)
        else:
            squeeze = ()

        if scan is None:
            scan = (self.extent[0] / 2, self.extent[1] / 2)

        scan = validate_scan(scan, self)

        probe = self.copy()

        if scan is not None:
            probe = probe.insert_transform(scan, len(probe.transforms))

        probe.insert_transform(WaveRenormalization(), 0)

        if lazy:
            detectors = [WavesDetector()]
            waves = probe.lazy_build_multislice_detect(
                detectors=detectors, max_batch=max_batch
            )

        else:
            waves = probe.base_waves_partial()()

            waves = probe.transforms.apply(waves)

            waves = waves.ensure_real_space()

        waves = waves.squeeze(squeeze)

        return waves

    def multislice(
        self,
        potential: Union[AbstractPotential, Atoms],
        scan: AbstractScan = None,
        detectors: AbstractDetector = None,
        max_batch: Union[int, str] = "auto",
        transition_potentials=None,
        lazy: bool = None,
    ) -> Union[AbstractMeasurement, Waves, List[Union[AbstractMeasurement, Waves]]]:
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
        chunks : int, optional
            Specifices the number of wave functions in each chunk of the created dask array. If None, the number
            of chunks are automatically estimated based on the "dask.chunk-size" parameter in the configuration.
        lazy : boolean, optional
            If True, create the measurements lazily, otherwise, calculate instantly. If None, this defaults to the value
            set in the configuration file.

        Returns
        -------
        measurements : AbstractMeasurement or Waves or list of AbstractMeasurement
        """
        initialize()

        potential = validate_potential(potential)
        self.grid.match(potential)

        if not isinstance(scan, AbstractScan):
            squeeze = (-3,)
        else:
            squeeze = ()

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

        if scan is not None:
            probe.insert_transform(scan, len(self.transforms))

        probe.insert_transform(WaveRenormalization(), 0)

        if transition_potentials:
            multislice_func = partial(
                transition_potential_multislice_and_detect,
                transition_potentials=transition_potentials,
            )
        else:
            multislice_func = multislice_and_detect

        measurements = probe.lazy_build_multislice_detect(
            detectors=detectors,
            max_batch=max_batch,
            potential=potential,
            multislice_func=multislice_func,
        )

        if squeeze:
            measurements = measurements.squeeze(squeeze)

        if not lazy:
            measurements.compute()

        return measurements

    def scan(
        self,
        potential: Union[Atoms, AbstractPotential],
        scan: Union[AbstractScan, np.ndarray, Sequence] = None,
        detectors: Union[AbstractDetector, Sequence[AbstractDetector]] = None,
        max_batch: Union[int, str] = "auto",
        transition_potentials=None,
        lazy: bool = None,
    ) -> Union[AbstractMeasurement, Waves, List[Union[AbstractMeasurement, Waves]]]:
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
            The number of wave functions in each chunk of the Dask array. If None, the number of chunks are
            automatically estimated based on "dask.chunk-size" in the user configuration.
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

        return self.multislice(
            potential,
            scan,
            detectors,
            max_batch=max_batch,
            lazy=lazy,
            transition_potentials=transition_potentials,
        )

    def profile(self, angle: float = 0.0):
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

        point2 = point1 + np.array(
            [np.cos(np.pi * angle / 180), np.sin(np.pi * angle / 180)]
        )
        point1, point2 = _line_intersect_rectangle(
            point1, point2, (0.0, 0.0), self.extent
        )
        return measurement.interpolate_line(point1, point2)

    def complex_images(self, lazy: bool = False):
        return self.build(lazy=lazy).complex_images()

    def show(self, **kwargs):
        """
        Show the probe wave function.

        Parameters
        ----------
        kwargs : Additional keyword arguments for the abtem.plot.show_image function.
        """
        return (
            self.build((self.extent[0] / 2, self.extent[1] / 2))
            .intensity()
            .show(**kwargs)
        )
