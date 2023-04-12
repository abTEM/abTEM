"""Module for describing wave functions of the incoming electron beam and the exit wave."""
import itertools
import numbers
import warnings
from abc import abstractmethod
from copy import copy
from functools import partial
from typing import Sequence, Dict
from typing import Union, Tuple, List

import dask.array as da
import numpy as np
from ase import Atoms

from abtem.core.array import HasArray, validate_lazy, ComputableList, expand_dims
from abtem.core.axes import HasAxes
from abtem.core.axes import RealSpaceAxis, ReciprocalSpaceAxis, AxisMetadata
from abtem.core.backend import HasDevice
from abtem.core.backend import get_array_module, validate_device
from abtem.core.chunks import validate_chunks
from abtem.core.complex import abs2
from abtem.core.energy import Accelerator
from abtem.core.energy import HasAcceleratorMixin
from abtem.core.ensemble import EmptyEnsemble
from abtem.core.fft import fft2, ifft2, fft_crop, fft_interpolate
from abtem.core.grid import Grid, validate_gpts, polar_spatial_frequencies
from abtem.core.grid import HasGridMixin
from abtem.core.transform import CompositeWaveTransform, WaveTransform
from abtem.core.utils import safe_floor_int, CopyMixin, EqualityMixin
from abtem.detectors import (
    BaseDetector,
    _validate_detectors,
    WavesDetector,
    FlexibleAnnularDetector,
)
from abtem.distributions import BaseDistribution
from abtem.inelastic.core_loss import transition_potential_multislice_and_detect
from abtem.measurements import (
    DiffractionPatterns,
    Images,
    BaseMeasurement,
    RealSpaceLineProfiles,
)
from abtem.multislice import multislice_and_detect
from abtem.potentials.iam import BasePotential, _validate_potential
from abtem.scan import BaseScan, GridScan, _validate_scan
from abtem.tilt import validate_tilt
from abtem.transfer import Aberrations, CTF, Aperture


def _extract_measurement(array, index):
    if array.size == 0:
        return array

    array = array.item()[index].array
    return array


def _wrap_measurements(measurements):
    return measurements[0] if len(measurements) == 1 else ComputableList(measurements)


def _finalize_lazy_measurements(
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

    return measurements


def _ensure_parity(n, even, v=1):
    assert (v == 1) or (v == -1)
    assert isinstance(even, bool)

    if n % 2 == 0 and not even:
        return n + v
    elif not n % 2 == 0 and even:
        return n + v
    return n


def _ensure_parity_of_gpts(new_gpts, old_gpts, parity):
    if parity == "same":
        return (
            _ensure_parity(new_gpts[0], old_gpts[0] % 2 == 0),
            _ensure_parity(new_gpts[1], old_gpts[1] % 2 == 0),
        )
    elif parity == "odd":
        return (
            _ensure_parity(new_gpts[0], even=False),
            _ensure_parity(new_gpts[1], even=False),
        )
    elif parity == "even":
        return (
            _ensure_parity(new_gpts[0], even=True),
            _ensure_parity(new_gpts[1], even=True),
        )
    elif parity != "none":
        raise ValueError()


def _antialias_cutoff_gpts(gpts, sampling):
    kcut = 2.0 / 3.0 / max(sampling)
    extent = gpts[0] * sampling[0], gpts[1] * sampling[1]
    new_gpts = safe_floor_int(kcut * extent[0]), safe_floor_int(kcut * extent[1])
    return _ensure_parity_of_gpts(new_gpts, gpts, parity="same")


class BaseWaves(
    HasGridMixin, HasAcceleratorMixin, HasAxes, HasDevice, CopyMixin, EqualityMixin
):
    """Base class of all wave functions. Documented in the subclasses."""

    _base_axes = (-2, -1)

    @property
    @abstractmethod
    def ensemble_shape(self):
        pass

    @property
    @abstractmethod
    def ensemble_axes_metadata(self):
        pass

    @property
    @abstractmethod
    def metadata(self) -> dict:
        """Metadata stored as a dictionary."""
        pass

    @property
    def base_shape(self) -> Tuple[int, int]:
        return self.gpts

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        self.grid.check_is_defined()
        return [
            RealSpaceAxis(
                label="x", sampling=self.sampling[0], units="Å", endpoint=False
            ),
            RealSpaceAxis(
                label="y", sampling=self.sampling[1], units="Å", endpoint=False
            ),
        ]

    @property
    def reciprocal_space_axes_metadata(self) -> List[AxisMetadata]:

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        return [
            ReciprocalSpaceAxis(
                label="scattering angle x",
                sampling=self.angular_sampling[0],
                units="mrad",
            ),
            ReciprocalSpaceAxis(
                label="scattering angle y",
                sampling=self.angular_sampling[1],
                units="mrad",
            ),
        ]

    @property
    def antialias_cutoff_gpts(self) -> Tuple[int, int]:
        """
        The number of grid points along the x and y direction in the simulation grid at the antialiasing cutoff
        scattering angle.
        """
        if "adjusted_antialias_cutoff_gpts" in self.metadata:
            return tuple(
                min(n, m)
                for n, m in zip(
                    self.metadata["adjusted_antialias_cutoff_gpts"], self.gpts
                )
            )

        self.grid.check_is_defined()
        return _antialias_cutoff_gpts(self.gpts, self.sampling)

    @property
    def antialias_valid_gpts(self) -> Tuple[int, int]:
        """
        The number of grid points along the x and y direction in the simulation grid for the largest rectangle that fits
        within antialiasing cutoff scattering angle.
        """
        cutoff_gpts = self.antialias_cutoff_gpts
        valid_gpts = (
            safe_floor_int(cutoff_gpts[0] / np.sqrt(2)),
            safe_floor_int(cutoff_gpts[1] / np.sqrt(2)),
        )
        # print(cutoff_gpts[1] / np.sqrt(2), safe_floor_int(cutoff_gpts[1] / np.sqrt(2)))

        valid_gpts = _ensure_parity_of_gpts(valid_gpts, self.gpts, parity="same")

        if "adjusted_antialias_cutoff_gpts" in self.metadata:
            return tuple(
                min(n, m)
                for n, m in zip(
                    self.metadata["adjusted_antialias_cutoff_gpts"], valid_gpts
                )
            )

        return valid_gpts

    def _gpts_within_angle(
        self, angle: Union[None, float, str], parity: str = "same"
    ) -> Tuple[int, int]:

        if angle is None or angle == "full":
            return self.gpts

        elif isinstance(angle, (numbers.Number, float)):
            gpts = (
                int(2 * np.ceil(angle / self.angular_sampling[0])) + 1,
                int(2 * np.ceil(angle / self.angular_sampling[1])) + 1,
            )

        elif angle == "cutoff":
            gpts = self.antialias_cutoff_gpts

        elif angle == "valid":
            gpts = self.antialias_valid_gpts

        else:
            raise ValueError(
                "Angle must be a number or one of 'cutoff', 'valid' or 'full'"
            )

        return _ensure_parity_of_gpts(gpts, self.gpts, parity=parity)

    @property
    def cutoff_angles(self) -> Tuple[float, float]:
        """Scattering angles at the antialias cutoff [mrad]."""
        return (
            self.antialias_cutoff_gpts[0] // 2 * self.angular_sampling[0],
            self.antialias_cutoff_gpts[1] // 2 * self.angular_sampling[1],
        )

    @property
    def rectangle_cutoff_angles(self) -> Tuple[float, float]:
        """Scattering angles corresponding to the sides of the largest rectangle within the antialias cutoff [mrad]."""
        return (
            self.antialias_valid_gpts[0] // 2 * self.angular_sampling[0],
            self.antialias_valid_gpts[1] // 2 * self.angular_sampling[1],
        )

    @property
    def full_cutoff_angles(self) -> Tuple[float, float]:
        """Scattering angles corresponding to the full wave function size [mrad]."""
        return (
            self.gpts[0] // 2 * self.angular_sampling[0],
            self.gpts[1] // 2 * self.angular_sampling[1],
        )

    @property
    def angular_sampling(self) -> Tuple[float, float]:
        """Reciprocal-space sampling in units of scattering angles [mrad]."""
        self.accelerator.check_is_defined()
        fourier_space_sampling = self.reciprocal_space_sampling
        return (
            fourier_space_sampling[0] * self.wavelength * 1e3,
            fourier_space_sampling[1] * self.wavelength * 1e3,
        )

    def _angular_grid(self):
        xp = get_array_module(self._device)
        alpha, phi = polar_spatial_frequencies(self.gpts, self.sampling, xp=xp)
        alpha *= self.wavelength
        return alpha, phi


class _WaveRenormalization(EmptyEnsemble, WaveTransform):
    def apply(self, waves, overwrite_x: bool = False):
        return waves.normalize(overwrite_x=overwrite_x)


class Waves(HasArray, BaseWaves):
    """
    Waves define a batch of arbitrary 2D wave functions defined by a complex array.

    Parameters
    ----------
    array : array
        Complex array defining one or more 2D wave functions. The second-to-last and last dimensions are the wave
        function `y`- and `x`-axes, respectively.
    energy : float
        Electron energy [eV].
    extent : one or two float
        Extent of wave functions in `x` and `y` [Å].
    sampling : one or two float
        Sampling of wave functions in `x` and `y` [1 / Å].
    reciprocal_space : bool, optional
        If True, the wave functions are assumed to be represented in reciprocal space instead of real space (default is False).
    ensemble_axes_metadata : list of AxesMetadata
        Axis metadata for each ensemble axis. The axis metadata must be compatible with the shape of the array.
    metadata : dict
        A dictionary defining wave function metadata. All items will be added to the metadata of measurements derived
        from the waves.
    """

    _base_dims = 2  # The dimension of waves is assumed to be 2D.

    def __init__(
        self,
        array: np.ndarray,
        energy: float,
        extent: Union[float, Tuple[float, float]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        reciprocal_space: bool = False,
        ensemble_axes_metadata: List[AxisMetadata] = None,
        metadata: Dict = None,
    ):

        if len(array.shape) < 2:
            raise RuntimeError(
                "Wave-function array should have two or more dimensions."
            )

        self._array = array
        self._grid = Grid(
            extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True
        )
        self._accelerator = Accelerator(energy=energy)
        self._ensemble_axes_metadata = (
            [] if ensemble_axes_metadata is None else ensemble_axes_metadata
        )
        self._metadata = {} if metadata is None else metadata

        self._reciprocal_space = reciprocal_space
        self._check_axes_metadata()

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
            if hasattr(axis, "tilt")
        )

    @property
    def tilt_axes_metadata(self):
        return [self.ensemble_axes_metadata[i] for i in self.tilt_axes]

    @property
    def ensemble_axes_metadata(self):
        return self._ensemble_axes_metadata

    @property
    def reciprocal_space(self):
        return self._reciprocal_space

    @property
    def metadata(self) -> Dict:
        self._metadata["energy"] = self.energy
        return self._metadata

    def from_partitioned_args(self):
        d = self._copy_kwargs(exclude=("array", "extent", "ensemble_axes_metadata"))
        cls = self.__class__
        return partial(lambda *args, **kwargs: cls(args[0], **kwargs), **d)

    @classmethod
    def from_array_and_metadata(
        cls, array: np.ndarray, axes_metadata: List[AxisMetadata], metadata: dict = None
    ) -> "Waves":

        """
        Creates wave functions from a given array and metadata.

        Parameters
        ----------
        array : array
            Complex array defining one or more 2D wave functions. The second-to-last and last dimensions are the wave
            function `y`- and `x`-axis, respectively.
        axes_metadata : list of AxesMetadata
            Axis metadata for each axis. The axis metadata must be compatible with the shape of the array. The last two
            axes must be RealSpaceAxis.
        metadata :
            A dictionary defining wave function metadata. All items will be added to the metadata of measurements derived from
            the waves. The metadata must contain the electron energy [eV].

        Returns
        -------
        wave_functions : Waves
            The created wave functions.
        """
        energy = metadata["energy"]

        x_axis, y_axis = axes_metadata[-2], axes_metadata[-1]

        if isinstance(x_axis, RealSpaceAxis) and isinstance(y_axis, RealSpaceAxis):
            sampling = x_axis.sampling, y_axis.sampling
        else:
            raise ValueError()

        return cls(
            array,
            sampling=sampling,
            energy=energy,
            ensemble_axes_metadata=axes_metadata[:-2],
            metadata=metadata,
        )

    def convolve(
        self,
        kernel: np.ndarray,
        axes_metadata: List[AxisMetadata] = None,
        out_space: str = "in_space",
        overwrite_x: bool = False,
    ):
        """
        Convolve the wave-function array with a given array.

        Parameters
        ----------
        kernel : np.ndarray
            Array to be convolved with.
        axes_metadata : list of AxisMetadata, optional
            Metadata for the resulting convolved array. Needed only if the given array has more than two dimensions.
        out_space : str, optional
            Space in which the convolved array is represented. Options are 'reciprocal_space' and 'real_space' (default is the space of the wave functions).

        Returns
        -------
        convolved : Waves
            The convolved wave functions.
        """

        if out_space == "in_space":
            fourier_space_out = self.reciprocal_space
        elif out_space in ("reciprocal_space", "real_space"):
            fourier_space_out = out_space == "reciprocal_space"
        else:
            raise ValueError

        if axes_metadata is None:
            axes_metadata = []

        if (len(kernel.shape) - 2) != len(axes_metadata):
            raise ValueError("provide axes metadata for each ensemble axis")

        waves = self.ensure_reciprocal_space(overwrite_x=overwrite_x)
        waves_dims = tuple(range(len(kernel.shape) - 2))
        kernel_dims = tuple(
            range(
                len(kernel.shape) - 2,
                len(waves.array.shape) - 2 + len(kernel.shape) - 2,
            )
        )

        kernel = expand_dims(kernel, axis=kernel_dims)
        array = expand_dims(waves._array, axis=waves_dims)

        xp = get_array_module(self.device)

        kernel = xp.array(kernel)

        if overwrite_x and (array.shape == kernel.shape):
            array *= kernel
        else:
            array = array * kernel

        if not fourier_space_out:
            array = ifft2(array, overwrite_x=overwrite_x)

        d = waves._copy_kwargs(exclude=("array",))
        d["reciprocal_space"] = fourier_space_out
        d["array"] = array
        d["ensemble_axes_metadata"] = axes_metadata + d["ensemble_axes_metadata"]
        return waves.__class__(**d)

    def normalize(self, space: str = "reciprocal", overwrite_x: bool = False):
        """
        Normalize the wave functions in real or reciprocal space.

        Parameters
        ----------
        space : str
            Should be one of 'real' or 'reciprocal' (default is 'reciprocal'). Defines whether the wave function should
            be normalized such that the intensity sums to one in real or reciprocal space.

        Returns
        -------
        normalized_waves : Waves
            The normalized wave functions.
        """

        if self.is_lazy:
            return self.apply_transform(_WaveRenormalization())

        xp = get_array_module(self.device)

        reciprocal_space = self.reciprocal_space

        if space == "reciprocal":
            waves = self.ensure_reciprocal_space(overwrite_x=overwrite_x)
            f = xp.sqrt(abs2(waves.array).sum((-2, -1), keepdims=True))
            if overwrite_x:
                waves._array /= f
            else:
                waves._array = waves._array / f

            if not reciprocal_space:
                waves = waves.ensure_real_space(overwrite_x=overwrite_x)

        elif space == "real":
            raise NotImplementedError
        else:
            raise ValueError()

        return waves

    def tile(self, repetitions: Tuple[int, int], renormalize: bool = False) -> "Waves":
        """
        Tile the wave functions. Can only be applied in real space.

        Parameters
        ----------
        repetitions : two int
            The number of repetitions of the wave functions along the `x`- and `y`-axes.
        renormalize : bool, optional
            If True, preserve the total intensity of the wave function (default is False).

        Returns
        -------
        tiled_wave_functions : Waves
            The tiled wave functions.
        """

        xp = get_array_module(self.device)

        if self.reciprocal_space:
            raise NotImplementedError

        if self.is_lazy:
            tile_func = da.tile
        else:
            tile_func = xp.tile

        array = tile_func(self.array, (1,) * len(self.ensemble_shape) + repetitions)

        if hasattr(array, "rechunk"):
            array = array.rechunk(array.chunks[:-2] + (-1, -1))

        kwargs = self._copy_kwargs(exclude=("array", "extent"))
        kwargs["array"] = array

        if renormalize:
            kwargs["array"] /= xp.asarray(np.prod(repetitions))

        return self.__class__(**kwargs)

    def ensure_reciprocal_space(self, overwrite_x: bool = False):
        """
        Transform to reciprocal space if the wave functions are represented in real space.

        Parameters
        ----------
        in_place : bool, optional
            If True, modify the array in place; otherwise a copy is created (default is False).

        Returns
        -------
        waves_in_reciprocal_space : Waves
            The wave functions in reciprocal space.
        """

        if self.reciprocal_space:
            return self

        d = self._copy_kwargs(exclude=("array",))
        d["array"] = fft2(self.array, overwrite_x=overwrite_x)
        d["reciprocal_space"] = True
        return self.__class__(**d)

    def ensure_real_space(self, overwrite_x: bool = False):
        """
        Transform to real space if the wave functions are represented in reciprocal space.

        Parameters
        ----------
        in_place : bool, optional
            If True, modify the array in place; otherwise a copy is created (default is False).

        Returns
        -------
        waves_in_real_space : Waves
            The wave functions in real space.
        """

        if not self.reciprocal_space:
            return self

        d = self._copy_kwargs(exclude=("array",))
        d["array"] = ifft2(self.array, overwrite_x=overwrite_x)
        d["reciprocal_space"] = False
        return self.__class__(**d)

    def phase_shift(self, amount: float):
        """
        Shift the phase of the wave functions.

        Parameters
        ----------
        amount : float
            Amount of phase shift [rad].

        Returns
        -------
        phase_shifted_waves : Waves
            The shifted wave functions.
        """

        def phase_shift(array):
            xp = get_array_module(self.array)
            return xp.exp(1.0j * amount) * array

        d = self._copy_kwargs(exclude=("array",))
        d["array"] = phase_shift(self.array)
        d["reciprocal_space"] = False
        return self.__class__(**d)

    def intensity(self) -> Images:
        """
        Calculate the intensity of the wave functions.

        Returns
        -------
        intensity_images : Images
            The intensity of the wave functions.
        """

        def intensity(array):
            return abs2(array)

        metadata = copy(self.metadata)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"

        xp = get_array_module(self.array)

        if self.is_lazy:
            array = self.array.map_blocks(intensity, dtype=xp.float32)
        else:
            array = intensity(self.array)

        return Images(
            array,
            sampling=self.sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=metadata,
        )

    def complex_images(self):
        """
        The complex array of the wave functions at the image plane.

        Returns
        -------
        complex_images : Images
            The wave functions as a complex image.
        """

        array = self.array.copy()
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
        Downsample the wave functions to a lower maximum scattering angle.

        Parameters
        ----------
        max_angle : {'cutoff', 'valid'} or float, optional
            Controls the downsampling of the wave functions.

                ``cutoff`` :
                    Downsample to the antialias cutoff scattering angle (default).

                ``valid`` :
                    Downsample to the largest rectangle that fits inside the circle with a radius defined by the antialias
                    cutoff scattering angle.

                float :
                    Downsample to a maximum scattering angle specified by a float [mrad].

        gpts : two int, optional
            Number of grid points of the wave functions after downsampling. If given, `max_angle` is not used.

        normalization : {'values', 'amplitude'}
            The normalization parameter determines the preserved quantity after normalization.

                ``values`` :
                    The pixel-wise values of the wave function are preserved (default).

                ``amplitude`` :
                    The total amplitude of the wave function is preserved.

        Returns
        -------
        downsampled_waves : Waves
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

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        kwargs["sampling"] = (self.extent[0] / gpts[0], self.extent[1] / gpts[1])
        kwargs["metadata"][
            "adjusted_antialias_cutoff_gpts"
        ] = self.antialias_cutoff_gpts
        return self.__class__(**kwargs)

    def diffraction_patterns(
        self,
        max_angle: Union[str, float, None] = "cutoff",
        block_direct: Union[bool, float] = False,
        fftshift: bool = True,
        parity: str = "odd",
        return_complex: bool = False,
        ensure_reciprocal_space_normalization: bool = True,
    ) -> DiffractionPatterns:
        """
        Calculate the intensity of the wave functions at the diffraction plane.

        Parameters
        ----------
        max_angle : {'cutoff', 'valid', 'full'} or float
            Control the maximum scattering angle of the diffraction patterns.

                ``cutoff`` :
                    Downsample to the antialias cutoff scattering angle (default).

                ``valid`` :
                    Downsample to the largest rectangle that fits inside the circle with a radius defined by the antialias
                    cutoff scattering angle.

                ``full`` :
                    The diffraction patterns are not cropped, and hence the antialiased region is included.

                float :
                    Downsample to a maximum scattering angle specified by a float [mrad].

        block_direct : bool or float, optional
            If True the direct beam is masked (default is False). If float, masks up to that scattering angle [mrad].
        fftshift : bool, optional
            If False, do not shift the direct beam to the center of the diffraction patterns (default is True).
        parity : {'same', 'even', 'odd', 'none'}
            The parity of the shape of the diffraction patterns. Default is 'odd', so that the shape of the diffraction
            pattern is odd with the zero at the middle.
        return_complex : bool
            If True, return complex-valued diffraction patterns (i.e. the wave function in reciprocal space)
            (default is False).

        Returns
        -------
        diffraction_patterns : DiffractionPatterns
            The diffraction pattern(s).
        """

        def _diffraction_pattern(array, new_gpts, return_complex, fftshift, normalize):
            xp = get_array_module(array)

            if normalize:
                array = array / np.prod(array.shape[-2:])

            array = fft2(array, overwrite_x=False)

            if array.shape[-2:] != new_gpts:
                array = fft_crop(array, new_shape=array.shape[:-2] + new_gpts)

            if not return_complex:
                array = abs2(array)

            if fftshift:
                return xp.fft.fftshift(array, axes=(-1, -2))

            return array

        xp = get_array_module(self.array)
        new_gpts = self._gpts_within_angle(max_angle, parity=parity)

        metadata = copy(self.metadata)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"

        if (
            ensure_reciprocal_space_normalization
            and "normalization" in metadata
            and metadata["normalization"] == "values"
        ):
            metadata["normalization"] = "reciprocal_space"
            normalize = True
        else:
            normalize = False

        validate_gpts(new_gpts)

        if self.is_lazy:
            dtype = xp.complex64 if return_complex else xp.float32

            pattern = self.array.map_blocks(
                _diffraction_pattern,
                new_gpts=new_gpts,
                fftshift=fftshift,
                return_complex=return_complex,
                normalize=normalize,
                chunks=self.array.chunks[:-2] + ((new_gpts[0],), (new_gpts[1],)),
                meta=xp.array((), dtype=dtype),
            )
        else:
            pattern = _diffraction_pattern(
                self.array,
                new_gpts=new_gpts,
                return_complex=return_complex,
                fftshift=fftshift,
                normalize=normalize,
            )

        diffraction_patterns = DiffractionPatterns(
            pattern,
            sampling=self.reciprocal_space_sampling,
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
    def _apply_wave_transform(
        *args, waves_partial, transform_partial, transform_ensemble_shape
    ):
        transform = transform_partial(
            *(arg.item() for arg in args[: len(transform_ensemble_shape)])
        )
        ensemble_axes_metadata = [
            axis.item() for axis in args[len(transform_ensemble_shape) : -1]
        ]
        array = args[-1]

        waves = waves_partial(array, ensemble_axes_metadata=ensemble_axes_metadata)

        waves = transform.apply(waves)
        return waves.array

    def apply_transform(
        self, transform: WaveTransform, max_batch: Union[int, str] = "auto"
    ) -> "Waves":
        """
        Transform the wave functions by a given transformation.

        Parameters
        ----------
        transform : WaveTransform
            The wave-function transformation to apply.
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array. If 'auto' (default), the batch size is
            automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".

        Returns
        -------
        transformed_waves : Waves
            The transformed waves.
        """

        if self.is_lazy:
            if isinstance(max_batch, int):
                max_batch = max_batch * self.gpts[0] * self.gpts[1]

            chunks = validate_chunks(
                transform.ensemble_shape + self.shape,
                transform._default_ensemble_chunks
                + tuple(max(chunk) for chunk in self.array.chunks),
                limit=max_batch,
                dtype=np.dtype("complex64"),
            )

            transform_blocks = tuple(
                (arg, (i,))
                for i, arg in enumerate(
                    transform._partition_args(chunks[: -len(self.shape)])
                )
            )
            transform_blocks = tuple(itertools.chain(*transform_blocks))

            axes_blocks = ()
            for i, (axis, c) in enumerate(
                zip(self.ensemble_axes_metadata, self.array.chunks)
            ):
                axes_blocks += (
                    axis._to_blocks(
                        (c,),
                    ),
                    (len(transform.ensemble_shape) + i,),
                )

            symbols = tuple(range(len(transform.ensemble_shape) + len(self.shape)))

            array_blocks = (
                self.array,
                tuple(
                    range(
                        len(transform.ensemble_shape),
                        len(transform.ensemble_shape) + len(self.shape),
                    )
                ),
            )
            xp = get_array_module(self.device)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Increasing number of chunks")
                array = da.blockwise(
                    self._apply_wave_transform,
                    symbols,
                    *transform_blocks,
                    *axes_blocks,
                    *array_blocks,
                    adjust_chunks={i: chunk for i, chunk in enumerate(chunks)},
                    transform_partial=transform._from_partitioned_args(),
                    transform_ensemble_shape=transform.ensemble_shape,
                    waves_partial=self.from_partitioned_args(),  # noqa
                    meta=xp.array((), dtype=np.complex64),
                    align_arrays=False
                )

            kwargs = self._copy_kwargs(exclude=("array",))
            kwargs["array"] = array
            kwargs["ensemble_axes_metadata"] = (
                transform.ensemble_axes_metadata + kwargs["ensemble_axes_metadata"]
            )
            return self.__class__(**kwargs)
        else:
            return transform.apply(self)

    def apply_ctf(
        self, ctf: CTF = None, max_batch: Union[int, str] = "auto", **kwargs
    ) -> "Waves":
        """
        Apply the aberrations and apertures of a contrast transfer function to the wave functions.

        Parameters
        ----------
        ctf : CTF, optional
            Contrast transfer function to be applied.
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array. If 'auto' (default), the batch size is
            automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        kwargs :
            Provide the parameters of the contrast transfer function as keyword arguments (see :class:`.CTF`).

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
    def _lazy_multislice(*args, potential_partial, waves_partial, detectors, **kwargs):
        potential = potential_partial(*(arg.item() for arg in args[:1]))
        ensemble_axes_metadata = [axis.item() for axis in args[1:-1]]
        waves = waves_partial(*args[-1:], ensemble_axes_metadata=ensemble_axes_metadata)

        measurements = waves.multislice(potential, detectors=detectors, **kwargs)
        measurements = (
            (measurements,) if hasattr(measurements, "array") else measurements
        )
        arr = np.zeros((1,) * (len(args) - 1), dtype=object)
        arr.itemset(measurements)
        return arr

    def multislice(
        self,
        potential: BasePotential,
        detectors: Union[BaseDetector, List[BaseDetector]] = None,
        conjugate: bool = False,
        transpose: bool = False,
    ) -> "Waves":
        """
        Propagate and transmit wave function through the provided potential using the multislice algorithm. When detector(s)
        are given, output will be the corresponding measurement.

        Parameters
        ----------
        potential : BasePotential or ASE.Atoms
            The potential through which to propagate the wave function. Optionally atoms can be directly given.
        detectors : BaseDetector or list of BaseDetector, optional
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm. See `abtem.measurements.detect` for a list of implemented detectors. If not
            given, returns the wave functions themselves.
        conjugate : bool, optional
            If True, use the conjugate of the transmission function (default is False).
        transpose : bool, optional
            If True, reverse the order of propagation and transmission (default is False).

        Returns
        -------
        detected_waves : BaseMeasurement or list of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential (if no detector(s) given).
        """

        potential = _validate_potential(potential, self)
        detectors = _validate_detectors(detectors)

        if self.is_lazy:
            potential_blocks = potential._partition_args()
            potential_blocks = tuple(
                (block, (i,)) for i, block in enumerate(potential_blocks)
            )

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

            axes_blocks = ()
            for i, (axis, chunks) in enumerate(
                zip(self.ensemble_axes_metadata, self.array.chunks)
            ):
                axes_blocks += (
                    axis._to_blocks(
                        (chunks),
                    ),
                    (num_new_symbols + i,),
                )

            arrays = da.blockwise(
                self._lazy_multislice,
                tuple(range(num_new_symbols + len(self.shape) - 2)),
                *tuple(itertools.chain(*potential_blocks)),
                *axes_blocks,
                self.array,
                tuple(range(num_new_symbols, num_new_symbols + len(self.shape))),
                potential_partial=potential._from_partitioned_args(),
                new_axes=new_axes,
                detectors=detectors,  # noqa
                conjugate=conjugate,  # noqa
                transpose=transpose,  # noqa
                waves_partial=self.from_partitioned_args(),  # noqa
                concatenate=True,
                meta=np.array((), dtype=np.complex64)
            )

            measurements = _finalize_lazy_measurements(
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

        return _wrap_measurements(measurements)

    def show(self, **kwargs):
        """
        Show the wave-function intensities.

        kwargs :
            Keyword arguments for `abtem.measurements.Images.show`.
        """
        return self.intensity().show(**kwargs)

    def interact(self, **kwargs):
        return self.complex_images().interact(**kwargs)


class _WavesFactory(BaseWaves):
    def __init__(self, transforms: List[WaveTransform]):

        if transforms is None:
            transforms = []

        self._transforms = transforms

    @property
    def tilt(self):
        return self._tilt

    @tilt.setter
    def tilt(self, value):
        old_tilt = self.tilt
        new_tilt = validate_tilt(value)
        for i, transform in enumerate(self._transforms):
            if transform is old_tilt:
                self._transforms[i] = new_tilt

        self._tilt = new_tilt

    @abstractmethod
    def metadata(self):
        pass

    @abstractmethod
    def _base_waves_partial(self):
        pass

    def insert_transform(self, transform, index=None):

        if index is None:
            index = len(self._transforms)

        self._transforms.insert(index, transform)
        return self

    @property
    def transforms(self):
        return self._transforms

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return CompositeWaveTransform(self.transforms).ensemble_axes_metadata

    @property
    def ensemble_shape(self):
        return CompositeWaveTransform(self.transforms).ensemble_shape

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

        waves = waves.ensure_real_space(overwrite_x=True)

        if potential is not None:
            measurements = multislice_func(
                waves, potential=potential, detectors=detectors
            )

        else:
            measurements = tuple(detector.detect(waves) for detector in detectors)

        arr = np.zeros((1,) * (len(args) + 1), dtype=object)
        arr.itemset(measurements)
        return arr

    def _lazy_build_multislice_detect(
        self,
        detectors: List[BaseDetector],
        max_batch: int = None,
        potential: BasePotential = None,
        multislice_func=multislice_and_detect,
    ):

        args = ()
        symbols = ()
        adjust_chunks = {}
        extra_ensemble_axes_metadata = []
        new_axes = {}
        partials = {"base": (self._base_waves_partial(), ())}
        max_arg_index = 0
        max_symbol = 0

        if potential is not None:
            # add potential args
            potential_symbols = tuple(range(0, max(len(potential.ensemble_shape), 1)))
            partials["potential"] = potential._wrapped_from_partitioned_args(), (0,)
            max_arg_index += 1
            max_symbol += 1

            args += potential._partition_args()[0], potential_symbols

            if potential.ensemble_shape:
                symbols += potential_symbols
                adjust_chunks[0] = potential._default_ensemble_chunks[0]
                extra_ensemble_axes_metadata += potential.ensemble_axes_metadata

            # add exit plane args
            if len(potential.exit_planes) > 1:
                symbols += tuple(
                    range(max(potential_symbols) + 1, max(potential_symbols) + 2)
                )
                new_axes[1] = (len(potential.exit_planes),)
                extra_ensemble_axes_metadata += [potential.exit_planes_axes_metadata]
                max_symbol += 1

        transforms = CompositeWaveTransform(self.transforms)

        # add transform args
        transform_arg_indices = tuple(
            range(max_arg_index, max_arg_index + len(transforms.ensemble_shape))
        )
        partials["transforms"] = (
            transforms._wrapped_from_partitioned_args(),
            transform_arg_indices,
        )
        transform_symbols = tuple(
            range(max_symbol, max_symbol + len(transforms.ensemble_shape))
        )
        transform_chunks = transforms._ensemble_chunks(max_batch, base_shape=self.gpts)

        args += tuple(
            itertools.chain(
                *tuple(
                    (block, (i,))
                    for i, block in zip(
                        transform_symbols,
                        transforms._partition_args(transform_chunks),
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
            warnings.filterwarnings("ignore", message="Increasing number of chunks.")
            array = da.blockwise(
                partial_build,
                symbols,
                *args,
                adjust_chunks=adjust_chunks,
                new_axes=new_axes,
                concatenate=True,
                meta=np.array((), dtype=object)
            )

        measurements = _finalize_lazy_measurements(
            array,
            waves=self,
            detectors=detectors,
            extra_ensemble_axes_metadata=extra_ensemble_axes_metadata,
        )

        return measurements


class PlaneWave(_WavesFactory):
    """
    Represents electron probe wave functions for simulating experiments with a plane-wave probe, such as HRTEM and SAED.

    Parameters
    ----------
    extent : two float, optional
        Lateral extent of the wave function [Å].
    gpts : two int, optional
        Number of grid points describing the wave function.
    sampling : two float, optional
        Lateral sampling of the wave functions [1 / Å]. If 'gpts' is also given, will be ignored.
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    tilt : two float, optional
        Small-angle beam tilt [mrad] (default is (0., 0.)). Implemented by shifting the wave functions at every slice.
    device : str, optional
        The wave functions are stored on this device ('cpu' or 'gpu'). The default is determined by the user configuration.
    transforms : list of WaveTransform, optional
        Can apply any transformation to the wave functions (e.g. to describe a phase plate).
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
        transforms: List[WaveTransform] = None,
    ):

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)
        self._tilt = validate_tilt(tilt=tilt)
        self._normalize = normalize
        self._device = validate_device(device)

        transforms = [] if transforms is None else transforms

        transforms = transforms + [self._tilt]

        super().__init__(transforms=transforms)

    @property
    def metadata(self):
        metadata = {
            "energy": self.energy,
            **self._tilt.metadata,
            "normalization": ("reciprocal_space" if self._normalize else "values"),
        }
        return metadata

    @property
    def normalize(self):
        return self._normalize

    def _base_waves_partial(self):
        def base_plane_wave(gpts, extent, energy, normalize, device):
            xp = get_array_module(device)

            if normalize:
                array = xp.full(
                    gpts, (1 / np.prod(gpts)).astype(xp.complex64), dtype=xp.complex64
                )
            else:
                array = xp.ones(gpts, dtype=xp.complex64)

            return Waves(
                array=array, energy=energy, extent=extent, reciprocal_space=False
            )

        return partial(
            base_plane_wave,
            gpts=self.gpts,
            extent=self.extent,
            energy=self.energy,
            normalize=self.normalize,
            device=self.device,
        )

    def build(
        self,
        lazy: bool = None,
        max_batch="auto",
    ) -> Waves:
        """
        Build plane-wave wave functions.

        Parameters
        ----------
        lazy : bool, optional
            If True, create the wave functions lazily, otherwise, calculate instantly. If not given, defaults to the
            setting in the user configuration file.
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array. If 'auto' (default), the batch size is
            automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".

        Returns
        -------
        plane_waves : Waves
            The wave functions.
        """

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        lazy = validate_lazy(lazy)

        if lazy:
            detectors = [WavesDetector()]

            measurements = self._lazy_build_multislice_detect(
                detectors=detectors, max_batch=max_batch
            )

            return _wrap_measurements(measurements)

        waves = self._base_waves_partial()()
        return waves

    def multislice(
        self,
        potential: Union[BasePotential, Atoms],
        detectors: BaseDetector = None,
        max_batch: Union[int, str] = "auto",
        lazy: bool = None,
        ctf: CTF = None,
        transition_potentials=None,
    ) -> Waves:
        """
        Run the multislice algorithm, after building the plane-wave wave function as needed. The grid of the wave
        functions will be set to the grid of the potential.

        Parameters
        ----------
        potential : BasePotential, Atoms
            The potential through which to propagate the wave function. Optionally atoms can be directly given.
        detectors : Detector, list of detectors, optional
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm.
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array. If 'auto' (default), the batch size is
            automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        lazy : bool, optional
            If True, create the wave functions lazily, otherwise, calculate instantly. If None, this defaults to the
            setting in the user configuration file.
        ctf : CTF, optional
            A contrast transfer function may be applied before detecting to save memory.
        transition_potentials : BaseTransitionPotential, optional
            Used to describe inelastic core losses.

        Returns
        -------
        detected_waves : BaseMeasurement or list of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential (if no detector(s) given).
        """
        potential = _validate_potential(potential)
        lazy = validate_lazy(lazy)
        detectors = _validate_detectors(detectors)

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
            measurements = self._lazy_build_multislice_detect(
                detectors=detectors,
                max_batch=max_batch,
                potential=potential,
                multislice_func=multislice_func,
            )
            measurements = _wrap_measurements(measurements)
        else:
            waves = self.build(lazy=False)
            measurements = waves.multislice(
                potential=potential, detectors=detectors
            )  ##multislice_func(waves, )

        return measurements


class Probe(_WavesFactory):
    """
    Represents electron-probe wave functions for simulating experiments with a convergent beam,
    such as CBED and STEM.

    Parameters
    ----------
    semiangle_cutoff : float, optional
        The cutoff semiangle of the aperture [mrad]. Ignored if a custom aperture is given.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be ignored.
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    soft : float, optional
        Taper the edge of the default aperture [mrad] (default is 2.0). Ignored if a custom aperture is given.
    tilt : two float, two 1D :class:`.BaseDistribution`, 2D :class:`.BaseDistribution`, optional
        Small-angle beam tilt [mrad]. This value should generally not exceed one degree.
    device : str, optional
        The probe wave functions will be build and stored on this device ('cpu' or 'gpu'). The default is determined by the user configuration.
    aperture : BaseAperture, optional
        An optional custom aperture. The provided aperture should be a subtype of :class:`.BaseAperture`.
    aberrations : dict or Aberrations
        The phase aberrations as a dictionary.
    transforms : list of :class:`.WaveTransform`
        A list of additional wave function transforms which will be applied after creation of the probe wave functions.
    kwargs :
        Provide the aberrations as keyword arguments, forwarded to the :class:`.Aberrations`.
    """

    def __init__(
        self,
        semiangle_cutoff: float = None,
        extent: Union[float, Tuple[float, float]] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        energy: float = None,
        soft: bool = True,
        tilt: Union[
            Tuple[Union[float, BaseDistribution], Union[float, BaseDistribution]],
            BaseDistribution,
        ] = (
            0.0,
            0.0,
        ),
        device: str = None,
        aperture: Aperture = None,
        aberrations: Union[Aberrations, dict] = None,
        ctf=None,
        transforms: List[WaveTransform] = None,
        metadata: dict = None,
        **kwargs
    ):
        if ctf is not None:
            aperture = ctf._aperture
            aberrations = ctf._aberrations
            energy = ctf.energy

        self._accelerator = Accelerator(energy=energy)

        if semiangle_cutoff is None and aperture is None:
            raise ValueError()
        elif semiangle_cutoff is None:
            semiangle_cutoff = 30.0

        if aperture is None:
            aperture = Aperture(semiangle_cutoff=semiangle_cutoff, soft=soft)

        aperture._accelerator = self._accelerator

        if aberrations is None:
            aberrations = {}

        if isinstance(aberrations, dict):
            aberrations = Aberrations(
                semiangle_cutoff=semiangle_cutoff,
                energy=energy,
                **aberrations,
                **kwargs
            )

        aberrations._accelerator = self._accelerator

        self._aperture = aperture
        self._aberrations = aberrations
        self._tilt = validate_tilt(tilt=tilt)

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self._device = validate_device(device)
        self._metadata = {} if metadata is None else metadata

        transforms = [] if transforms is None else transforms
        transforms = transforms + [self.tilt] + [self.aperture] + [self.aberrations]

        super().__init__(transforms=transforms)

        self.accelerator.match(self.aperture)

    @classmethod
    def _from_ctf(cls, ctf, **kwargs):
        return cls(
            semiangle_cutoff=ctf.semiangle_cutoff,
            soft=ctf.soft,
            aberrations=ctf.aberration_coefficients,
            **kwargs
        )

    @property
    def ctf(self):
        return CTF(aberration_coefficients=self.aberrations.aberration_coefficients,
                   semiangle_cutoff=self.semiangle_cutoff,
                   energy=self.energy)

    @property
    def semiangle_cutoff(self):
        return self.aperture.semiangle_cutoff

    @property
    def aperture(self) -> Aperture:
        """Condenser or probe-forming aperture."""
        return self._aperture

    @aperture.setter
    def aperture(self, aperture: Aperture):
        self._aperture = aperture

    @property
    def aberrations(self) -> Aberrations:
        """Phase aberrations of the probe wave functions."""
        return self._aberrations

    @aberrations.setter
    def aberrations(self, aberrations: Aberrations):
        self._aberrations = aberrations

    @property
    def metadata(self) -> dict:
        """Metadata describing the probe wave functions."""
        return {
            **self._metadata,
            "energy": self.energy,
            **self.aperture.metadata,
            **self._tilt.metadata,
        }

    def _base_waves_partial(self):
        def base_probe(gpts, extent, energy, device, metadata):
            xp = get_array_module(device)

            array = xp.ones(gpts, dtype=xp.complex64)

            return Waves(
                array=array,
                energy=energy,
                extent=extent,
                reciprocal_space=True,
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
        scan: Union[tuple, BaseScan] = None,
        max_batch: Union[int, str] = "auto",
        lazy: bool = None,
    ) -> Waves:
        """
        Build probe wave functions at the provided positions.

        Parameters
        ----------
        scan : array of `xy`-positions or BaseScan, optional
            Positions of the probe wave functions. If not given, scans across the entire potential at Nyquist sampling.
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array. If 'auto' (default), the batch size is
            automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        lazy : bool, optional
            If True, create the wave functions lazily, otherwise, calculate instantly. If not given, defaults to the
            setting in the user configuration file.

        Returns
        -------
        probe_wave_functions : Waves
            The built probe wave functions.
        """

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        lazy = validate_lazy(lazy)

        if not isinstance(scan, BaseScan):
            squeeze = (-3,)
        else:
            squeeze = ()

        if scan is None:
            scan = (self.extent[0] / 2, self.extent[1] / 2)

        scan = _validate_scan(scan, self)

        probe = self.copy()

        if scan is not None:
            probe = probe.insert_transform(scan, len(probe.transforms))

        probe.insert_transform(_WaveRenormalization(), 0)

        if lazy:
            detectors = [WavesDetector()]

            waves = probe._lazy_build_multislice_detect(
                detectors=detectors, max_batch=max_batch
            )

            waves = _wrap_measurements(waves)

        else:
            waves = probe._base_waves_partial()()

            waves = CompositeWaveTransform(probe.transforms).apply(
                waves, overwrite_x=True
            )

            waves = waves.ensure_real_space(overwrite_x=True)

        waves = waves.squeeze(squeeze)

        return waves

    def multislice(
        self,
        potential: Union[BasePotential, Atoms],
        scan: Union[tuple, BaseScan] = None,
        detectors: BaseDetector = None,
        max_batch: Union[int, str] = "auto",
        lazy: bool = None,
        transition_potentials=None,
    ) -> Union[BaseMeasurement, Waves, List[Union[BaseMeasurement, Waves]]]:
        """
        Run the multislice algorithm for probe wave functions at the provided positions.

        Parameters
        ----------
        potential : BasePotential or Atoms
            The scattering potential. Optionally atoms can be directly given.
        scan : array of xy-positions or BaseScan, optional
            Positions of the probe wave functions. If not given, scans across the entire potential at Nyquist sampling.
        detectors : BaseDetector or list of BaseDetector, optional
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm. If not given, defaults to the flexible annular detector.
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array. If 'auto' (default), the batch size is
            automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        lazy : bool, optional
            If True, create the wave functions lazily, otherwise, calculate instantly. If None, this defaults to the
            setting in the user configuration file.
        transition_potentials : BaseTransitionPotential, optional
            Used to describe inelastic core losses.

        Returns
        -------
        measurements : BaseMeasurement or Waves or list of BaseMeasurement
        """
        potential = _validate_potential(potential)
        self.grid.match(potential)

        if not isinstance(scan, BaseScan):
            squeeze = (-3,)
        else:
            squeeze = ()

        if scan is None:
            scan = self.extent[0] / 2, self.extent[1] / 2

        scan = _validate_scan(scan, self)

        if detectors is None:
            detectors = [WavesDetector()]

        lazy = validate_lazy(lazy)

        detectors = _validate_detectors(detectors)

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        probe = self.copy()

        if scan is not None:
            probe.insert_transform(scan, len(self.transforms))

        probe.insert_transform(_WaveRenormalization(), 0)

        if transition_potentials:
            multislice_func = partial(
                transition_potential_multislice_and_detect,
                transition_potentials=transition_potentials,
            )
        else:
            multislice_func = multislice_and_detect

        measurements = probe._lazy_build_multislice_detect(
            detectors=detectors,
            max_batch=max_batch,
            potential=potential,
            multislice_func=multislice_func,
        )

        for i, measurement in enumerate(measurements):
            if squeeze:
                measurements[i] = measurement.squeeze(squeeze)
            measurement.metadata.update(scan.metadata)

        measurements = _wrap_measurements(measurements)

        if not lazy:
            measurements.compute()

        return measurements

    def scan(
        self,
        potential: Union[Atoms, BasePotential],
        scan: Union[BaseScan, np.ndarray, Sequence] = None,
        detectors: Union[BaseDetector, Sequence[BaseDetector]] = None,
        max_batch: Union[int, str] = "auto",
        transition_potentials=None,
        lazy: bool = None,
    ) -> Union[BaseMeasurement, Waves, List[Union[BaseMeasurement, Waves]]]:
        """
        Run the multislice algorithm from probe wave functions over the provided scan.

        Parameters
        ----------
        potential : BasePotential or Atoms
            The scattering potential.
        scan : BaseScan
            Positions of the probe wave functions. If not given, scans across the entire potential at Nyquist sampling.
        detectors : BaseDetector, list of BaseDetector, optional
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm. See abtem.measurements.detect for a list of implemented detectors.
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array. If 'auto' (default), the batch size is
            automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        lazy : bool, optional
            If True, create the measurements lazily, otherwise, calculate instantly. If None, this defaults to the value
            set in the configuration file.

        Returns
        -------
        detected_waves : BaseMeasurement or list of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential (if no detector(s) given).
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

    def profiles(self, angle: float = 0.0) -> RealSpaceLineProfiles:
        """
        Create a line profile through the center of the probe.

        Parameters
        ----------
        angle : float, optional
            Angle with respect to the `x`-axis of the line profile [degree].
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

    def show(self, **kwargs):
        """
        Show the intensity of the probe wave function.

        Parameters
        ----------
        kwargs : Keyword arguments for the :func:`.Images.show` function.
        """
        return (
            self.build((self.extent[0] / 2, self.extent[1] / 2))
            .intensity()
            .show(**kwargs)
        )
