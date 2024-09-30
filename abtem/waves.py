"""Module for describing wave functions of the electron beam and the exit wave."""

from __future__ import annotations

import itertools
from abc import abstractmethod
from copy import copy
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Optional, Sequence

import dask.array as da
import numpy as np
from ase import Atoms

import abtem
from abtem.array import ArrayObject, ComputableList, _expand_dims, _validate_lazy
from abtem.core.axes import (
    AxesMetadataList,
    AxisMetadata,
    OrdinalAxis,
    RealSpaceAxis,
    ReciprocalSpaceAxis,
)
from abtem.core.backend import (
    device_name_from_array_module,
    get_array_module,
    validate_device,
)
from abtem.core.chunks import validate_chunks
from abtem.core.complex import abs2
from abtem.core.energy import Accelerator, HasAcceleratorMixin
from abtem.core.ensemble import (
    EmptyEnsemble,
    Ensemble,
    _wrap_with_array,
    unpack_blockwise_args,
)
from abtem.core.fft import fft2, fft_crop, fft_interpolate, ifft2
from abtem.core.grid import Grid, HasGrid2DMixin, polar_spatial_frequencies
from abtem.core.utils import (
    CopyMixin,
    EqualityMixin,
    get_dtype,
    safe_floor_int,
    tuple_range,
)
from abtem.detectors import BaseDetector, FlexibleAnnularDetector
from abtem.distributions import BaseDistribution
from abtem.inelastic.core_loss import (
    BaseTransitionPotential,
    _validate_transition_potentials,
)
from abtem.measurements import (
    BaseMeasurements,
    DiffractionPatterns,
    Images,
    RealSpaceLineProfiles,
)
from abtem.multislice import (
    MultisliceTransform,
    transition_potential_multislice_and_detect,
)
from abtem.potentials.iam import BasePotential, validate_potential
from abtem.scan import BaseScan, CustomScan, GridScan, validate_scan
from abtem.slicing import SliceIndexedAtoms
from abtem.tilt import _validate_tilt
from abtem.transfer import CTF, Aberrations, Aperture, BaseAperture
from abtem.transform import WavesToWavesTransform

if TYPE_CHECKING:
    from abtem.visualize import Visualization


def _ensure_parity(n: int, even: bool, v: int = 1) -> int:
    assert (v == 1) or (v == -1)
    assert isinstance(even, bool)

    if n % 2 == 0 and not even:
        return n + v
    elif not n % 2 == 0 and even:
        return n + v
    return n


def _ensure_parity_of_gpts(
    new_gpts: tuple[int, int], old_gpts: tuple[int, int], parity: str
) -> tuple[int, int]:
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
    else:
        raise ValueError("parity must be one of 'same', 'odd', 'even', 'none'")


def _antialias_cutoff_gpts(
    gpts: tuple[int, int], sampling: tuple[float, float]
) -> tuple[int, int]:
    kcut = 2.0 / 3.0 / max(sampling)
    extent = gpts[0] * sampling[0], gpts[1] * sampling[1]
    new_gpts = safe_floor_int(kcut * extent[0]), safe_floor_int(kcut * extent[1])
    return _ensure_parity_of_gpts(new_gpts, gpts, parity="same")


class BaseWaves(HasGrid2DMixin, HasAcceleratorMixin):
    """Base class of all wave functions. Documented in the subclasses."""

    @property
    @abstractmethod
    def device(self) -> str:
        """The device where the waves are built or stored."""
        pass

    @property
    def dtype(self) -> np.dtype:
        """The datatype of waves."""
        return get_dtype(complex=True)

    @property
    @abstractmethod
    def metadata(self) -> dict:
        """Metadata stored as a dictionary."""
        pass

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        """List of AxisMetadata for the base axes in real space."""
        self.grid.check_is_defined()
        assert self.sampling is not None
        return [
            RealSpaceAxis(
                label="x", sampling=self.sampling[0], units="Å", endpoint=False
            ),
            RealSpaceAxis(
                label="y", sampling=self.sampling[1], units="Å", endpoint=False
            ),
        ]

    @property
    def reciprocal_space_axes_metadata(self) -> list[AxisMetadata]:
        """List of AxisMetadata for base axes in reciprocal space."""
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
    def antialias_cutoff_gpts(self) -> tuple[int, int]:
        """
        The number of grid points along the x and y direction in the simulation grid at
        the antialiasing cutoff scattering angle.
        """
        if "adjusted_antialias_cutoff_gpts" in self.metadata:
            n = min(
                self.metadata["adjusted_antialias_cutoff_gpts"][0], self._valid_gpts[0]
            )
            m = min(
                self.metadata["adjusted_antialias_cutoff_gpts"][1], self._valid_gpts[1]
            )
            return n, m
        return _antialias_cutoff_gpts(self._valid_gpts, self._valid_sampling)

    @property
    def antialias_valid_gpts(self) -> tuple[int, int]:
        """
        The number of grid points along the x and y direction in the simulation grid for
        the largest rectangle that fits within antialiasing cutoff scattering angle.
        """
        cutoff_gpts = self.antialias_cutoff_gpts
        valid_gpts = (
            safe_floor_int(cutoff_gpts[0] / np.sqrt(2)),
            safe_floor_int(cutoff_gpts[1] / np.sqrt(2)),
        )

        valid_gpts = _ensure_parity_of_gpts(valid_gpts, self._valid_gpts, parity="same")

        if "adjusted_antialias_cutoff_gpts" in self.metadata:
            n = min(self.metadata["adjusted_antialias_cutoff_gpts"][0], valid_gpts[0])
            m = min(self.metadata["adjusted_antialias_cutoff_gpts"][1], valid_gpts[1])
            return n, m

        return valid_gpts

    def _gpts_within_angle(
        self, angle: float | str, parity: str = "same"
    ) -> tuple[int, int]:
        if angle is None or angle == "full":
            return self._valid_gpts

        elif isinstance(angle, (Number, float)):
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

        return _ensure_parity_of_gpts(gpts, self._valid_gpts, parity=parity)

    @property
    def cutoff_angles(self) -> tuple[float, float]:
        """Scattering angles at the antialias cutoff [mrad]."""
        return (
            self.antialias_cutoff_gpts[0] // 2 * self.angular_sampling[0],
            self.antialias_cutoff_gpts[1] // 2 * self.angular_sampling[1],
        )

    @property
    def rectangle_cutoff_angles(self) -> tuple[float, float]:
        """Scattering angles corresponding to the sides of the largest rectangle within
        the antialias cutoff [mrad]."""
        return (
            self.antialias_valid_gpts[0] // 2 * self.angular_sampling[0],
            self.antialias_valid_gpts[1] // 2 * self.angular_sampling[1],
        )

    @property
    def full_cutoff_angles(self) -> tuple[float, float]:
        """Scattering angles corresponding to the full wave function size [mrad]."""
        return (
            self._valid_gpts[0] // 2 * self.angular_sampling[0],
            self._valid_gpts[1] // 2 * self.angular_sampling[1],
        )

    @property
    def cutoff_frequencies(self) -> tuple[float, float]:
        """Spatial frequencies at the antialias cutoff [1/Å]."""
        return (
            self.antialias_cutoff_gpts[0] // 2 * self.reciprocal_space_sampling[0],
            self.antialias_cutoff_gpts[1] // 2 * self.reciprocal_space_sampling[1],
        )

    @property
    def angular_sampling(self) -> tuple[float, float]:
        """Reciprocal-space sampling in units of scattering angles [mrad]."""
        self.accelerator.check_is_defined()
        return (
            self.reciprocal_space_sampling[0] * self.wavelength * 1e3,
            self.reciprocal_space_sampling[1] * self.wavelength * 1e3,
        )

    def _angular_grid(self) -> tuple[np.ndarray, np.ndarray]:
        xp = get_array_module(self.device)
        alpha, phi = polar_spatial_frequencies(
            self._valid_gpts, self._valid_sampling, xp=xp
        )
        alpha *= self.wavelength
        return alpha, phi


def _reduce_ensemble(
    ensemble: ArrayObject | list[ArrayObject],
) -> ArrayObject | list[ArrayObject]:
    if isinstance(ensemble, (ComputableList, list, tuple)):
        outputs = [_reduce_ensemble(x) for x in ensemble]

        if isinstance(ensemble, ComputableList):
            outputs = ComputableList(outputs)

        return outputs

    squeeze = tuple(
        i
        for i, axes_metadata in enumerate(ensemble.ensemble_axes_metadata)
        if axes_metadata._squeeze
    )

    output = ensemble.squeeze(squeeze)

    if isinstance(output, BaseMeasurements):
        output = output.reduce_ensemble()

    return output


class _WavesNormalization(WavesToWavesTransform):
    def __init__(self, space: str, in_place: bool):
        self._space = space
        self._in_place = in_place

    def _calculate_new_array(self, waves: Waves) -> np.ndarray:
        array = waves._eager_array

        xp = get_array_module(array)

        if self._space == "reciprocal":
            if not waves._reciprocal_space:
                array = fft2(array, overwrite_x=self._in_place)

            # waves = self.ensure_reciprocal_space(overwrite_x=in_place)
            f = xp.sqrt(abs2(array).sum((-2, -1), keepdims=True))
            if self._in_place:
                array /= f
            else:
                array = array / f

            if not waves._reciprocal_space:
                array = ifft2(array, overwrite_x=self._in_place)

        elif self._space == "real":
            raise NotImplementedError
        else:
            raise ValueError()

        return array


class Waves(BaseWaves, ArrayObject):
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
        Sampling of wave functions in `x` and `y` [Å].
    reciprocal_space : bool, optional
        If True, the wave functions are assumed to be represented in reciprocal space instead of real space (default is
        False).
    ensemble_axes_metadata : list of AxesMetadata
        Axis metadata for each ensemble axis. The axis metadata must be compatible with the shape of the array.
    metadata : dict
        A dictionary defining wave function metadata. All items will be added to the metadata of measurements derived
        from the waves.
    """

    _base_dims = 2

    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        energy: Optional[float] = None,
        extent: Optional[float | tuple[float, float]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        reciprocal_space: bool = False,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        metadata: Optional[dict] = None,
    ):
        if sampling is not None and extent is not None:
            extent = None

        self._grid = Grid(
            extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True
        )
        self._accelerator = Accelerator(energy=energy)
        self._reciprocal_space = reciprocal_space

        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    def device(self) -> str:
        """The device where the array is stored."""
        return device_name_from_array_module(get_array_module(self.array))

    @property
    def base_tilt(self) -> tuple[float, float]:
        """
        The base small-angle beam tilt (i.e. the beam tilt not associated with an ensemble axis) applied to the Fresnel
        propagator [mrad].
        """
        return (
            self.metadata.get("base_tilt_x", 0.0),
            self.metadata.get("base_tilt_y", 0.0),
        )

    @property
    def reciprocal_space(self) -> bool:
        """True if the waves are represented in reciprocal space."""
        return self._reciprocal_space

    @property
    def metadata(self) -> dict:
        self._metadata["energy"] = self.energy
        self._metadata["reciprocal_space"] = self.reciprocal_space
        return self._metadata

    @classmethod
    def from_array_and_metadata(
        cls,
        array: np.ndarray | da.core.Array,
        axes_metadata: list[AxisMetadata],
        metadata: Optional[dict] = None,
    ) -> Waves:
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
            A dictionary defining wave function metadata. All items will be added to the metadata of measurements
            derived from the waves. The metadata must contain the electron energy [eV].

        Returns
        -------
        wave_functions : Waves
            The created wave functions.
        """
        if metadata is None:
            raise ValueError("metadata must be provided to create Waves")

        energy = metadata["energy"]
        reciprocal_space = metadata.get("reciprocal_space", False)

        x_axis, y_axis = axes_metadata[-2], axes_metadata[-1]

        if isinstance(x_axis, RealSpaceAxis) and isinstance(y_axis, RealSpaceAxis):
            sampling = x_axis.sampling, y_axis.sampling
        else:
            raise ValueError()

        return cls(
            array,
            sampling=sampling,
            energy=energy,
            reciprocal_space=reciprocal_space,
            ensemble_axes_metadata=axes_metadata[:-2],
            metadata=metadata,
        )

    def convolve(
        self,
        kernel: np.ndarray,
        axes_metadata: Optional[list[AxisMetadata]] = None,
        out_space: str = "in_space",
        in_place: bool = False,
    ) -> Waves:
        """
        Convolve the wave-function array with a given array.

        Parameters
        ----------
        kernel : np.ndarray
            Array to be convolved with.
        axes_metadata : list of AxisMetadata, optional
            Metadata for the resulting convolved array. Needed only if the given array has more than two dimensions.
        out_space : str, optional
            Space in which the convolved array is represented. Options are 'reciprocal_space' and 'real_space' (default
            is the space of the wave functions).
        in_place : bool, optional
            If True, the array representing the waves may be modified in-place.

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

        waves = self.ensure_reciprocal_space(overwrite_x=in_place)
        waves_dims = tuple(range(len(kernel.shape) - 2))
        kernel_dims = tuple(
            range(
                len(kernel.shape) - 2,
                len(waves.array.shape) - 2 + len(kernel.shape) - 2,
            )
        )

        kernel = _expand_dims(kernel, axis=kernel_dims)
        array = _expand_dims(waves._array, axis=waves_dims)

        xp = get_array_module(self.device)

        kernel = xp.array(kernel)

        if in_place and (array.shape == kernel.shape):
            array *= kernel
        else:
            array = array * kernel

        if not fourier_space_out:
            array = ifft2(array, overwrite_x=in_place)

        d = waves._copy_kwargs(exclude=("array",))
        d["reciprocal_space"] = fourier_space_out
        d["array"] = array
        d["ensemble_axes_metadata"] = axes_metadata + d["ensemble_axes_metadata"]
        return waves.__class__(**d)

    def normalize(self, space: str = "reciprocal", in_place: bool = False) -> Waves:
        """
        Normalize the wave functions in real or reciprocal space.

        Parameters
        ----------
        space : str
            Should be one of 'real' or 'reciprocal' (default is 'reciprocal'). Defines whether the wave function should
            be normalized such that the intensity sums to one in real or reciprocal space.
        in_place : bool, optional
            If True, the array representing the waves may be modified in-place.

        Returns
        -------
        normalized_waves : Waves
            The normalized wave functions.
        """
        transform = _WavesNormalization(space=space, in_place=in_place)
        return transform.apply(self)

    def tile(self, repetitions: tuple[int, int], renormalize: bool = False) -> Waves:
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

    def ensure_reciprocal_space(self, overwrite_x: bool = False) -> Waves:
        """
        Transform to reciprocal space if the wave functions are represented in real space.

        Parameters
        ----------
        overwrite_x : bool, optional
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

    def ensure_real_space(self, overwrite_x: bool = False) -> Waves:
        """
        Transform to real space if the wave functions are represented in reciprocal space.

        Parameters
        ----------
        overwrite_x : bool, optional
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
        waves = self.__class__(**d)
        return waves

    def phase_shift(self, amount: float) -> Waves:
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

        def _phase_shift(array):
            xp = get_array_module(self.array)
            return xp.exp(1.0j * amount) * array

        d = self._copy_kwargs(exclude=("array",))
        d["array"] = _phase_shift(self.array)
        d["reciprocal_space"] = False
        return self.__class__(**d)

    def to_images(self, convert_complex: Optional[str] = None) -> Images:
        """
        The complex array of the wave functions at the image plane.

        Returns
        -------
        images : Images
            The wave functions as an image.
        """
        array = self.array.copy()
        metadata = copy(self.metadata)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"

        images = Images(
            array,
            sampling=self._valid_sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=metadata,
        )

        if not convert_complex:
            return images

        if convert_complex in ("intensity", "phase", "real", "imag"):
            return getattr(images, convert_complex)()
        else:
            raise ValueError(
                "convert_complex must be one of 'intensity', 'phase', 'real', 'imag'"
            )

    def intensity(self) -> Images:
        """
        Calculate the intensity of the wave functions.

        Returns
        -------
        intensity_images : Images
            The intensity of the wave functions.
        """
        return self.to_images(convert_complex="intensity")

    def phase(self) -> Images:
        """
        Calculate the phase of the wave functions.

        Returns
        -------
        phase_images : Images
            The phase of the wave functions.
        """
        return self.to_images(convert_complex="phase")

    def real(self) -> Images:
        """
        Calculate the real part of the wave functions.

        Returns
        -------
        real_images : Images
            The real part of the wave functions.
        """
        return self.to_images(convert_complex="real")

    def imag(self) -> Images:
        """
        Calculate the imaginary part of the wave functions.

        Returns
        -------
        imaginary_images : Images
            The imaginary part of the wave functions.
        """
        return self.to_images(convert_complex="imag")

    def downsample(
        self,
        max_angle: str | float = "cutoff",
        gpts: Optional[tuple[int, int]] = None,
        normalization: str = "values",
    ) -> Waves:
        """
        Downsample the wave functions to a lower maximum scattering angle.

        Parameters
        ----------
        max_angle : {'cutoff', 'valid'} or float, optional
            Controls the downsampling of the wave functions.

                ``cutoff`` :
                    Downsample to the antialias cutoff scattering angle (default).

                ``valid`` :
                    Downsample to the largest rectangle that fits inside the circle with a radius defined by the
                    antialias cutoff scattering angle.

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
            array = da.map_blocks(
                fft_interpolate,
                self.array,
                new_shape=gpts,
                normalization=normalization,
                chunks=self._lazy_array.chunks[:-2] + gpts,
                meta=xp.array((), dtype=get_dtype(complex=True)),
            )
        else:
            array = fft_interpolate(
                self._eager_array, new_shape=gpts, normalization=normalization
            )

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        kwargs["sampling"] = (
            self._valid_extent[0] / gpts[0],
            self._valid_extent[1] / gpts[1],
        )
        kwargs["metadata"]["adjusted_antialias_cutoff_gpts"] = (
            self.antialias_cutoff_gpts
        )
        return self.__class__(**kwargs)

    @staticmethod
    def _diffraction_pattern(array, new_gpts, return_complex, fftshift, normalize):
        xp = get_array_module(array)

        if normalize:
            array = array / float(np.prod(array.shape[-2:]))

        array = fft2(array, overwrite_x=False)

        if array.shape[-2:] != new_gpts:
            array = fft_crop(array, new_shape=array.shape[:-2] + new_gpts)

        if not return_complex:
            array = abs2(array)

        if fftshift:
            return xp.fft.fftshift(array, axes=(-1, -2))

        return array

    def diffraction_patterns(
        self,
        max_angle: str | float = "cutoff",
        # max_frequency: str | float = None,
        block_direct: bool | float = False,
        fftshift: bool = True,
        parity: str = "odd",
        return_complex: bool = False,
        renormalize: bool = True,
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
                    Downsample to the largest rectangle that fits inside the circle with a radius defined by the
                    antialias cutoff scattering angle.

                ``full`` :
                    The diffraction patterns are not cropped, and hence the antialiased region is included.

                float :
                    Downsample to a maximum scattering angle specified by a float [mrad].

        block_direct : bool or float, optional
            If True the direct beam is masked (default is False). If given as a float, masks up to that scattering
            angle [mrad].
        fftshift : bool, optional
            If False, do not shift the direct beam to the center of the diffraction patterns (default is True).
        parity : {'same', 'even', 'odd', 'none'}
            The parity of the shape of the diffraction patterns. Default is 'odd', so that the shape of the diffraction
            pattern is odd with the zero at the middle.
        renormalize : bool, optional
            If true and the wave function intensities were normalized to sum to the number of pixels in real space, i.e.
            the default normalization of a plane wave, the intensities are to sum to one in reciprocal space.
        return_complex : bool
            If True, return complex-valued diffraction patterns (i.e. the wave function in reciprocal space)
            (default is False).

        Returns
        -------
        diffraction_patterns : DiffractionPatterns
            The diffraction pattern(s).
        """
        xp = get_array_module(self.array)

        if max_angle is None:
            max_angle = "full"

        new_gpts = self._gpts_within_angle(max_angle, parity=parity)

        metadata = copy(self.metadata)
        metadata["label"] = "intensity"
        metadata["units"] = "arb. unit"

        normalize = False
        if renormalize and "normalization" in metadata:
            if metadata["normalization"] == "values":
                normalize = True
            elif metadata["normalization"] != "reciprocal_space":
                raise RuntimeError(
                    f"normalization {metadata['normalization']} not recognized"
                )

        if self.is_lazy:
            dtype = get_dtype(complex=return_complex)

            pattern = da.map_blocks(
                self._diffraction_pattern,
                self.array,
                new_gpts=new_gpts,
                fftshift=fftshift,
                return_complex=return_complex,
                normalize=normalize,
                chunks=self._lazy_array.chunks[:-2] + ((new_gpts[0],), (new_gpts[1],)),
                meta=xp.array((), dtype=dtype),
            )
        else:
            pattern = self._diffraction_pattern(
                self.array,
                new_gpts=new_gpts,
                return_complex=return_complex,
                fftshift=fftshift,
                normalize=normalize,
            )

        diffraction_patterns = DiffractionPatterns(
            pattern,
            sampling=(
                self.reciprocal_space_sampling[0],
                self.reciprocal_space_sampling[1],
            ),
            fftshift=fftshift,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=metadata,
        )

        if block_direct:
            diffraction_patterns = diffraction_patterns.block_direct(
                radius=block_direct
            )

        return diffraction_patterns

    def apply_ctf(
        self, ctf: Optional[CTF] = None, max_batch: int | str = "auto", **kwargs: Any
    ) -> Waves:
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

        waves = self.apply_transform(ctf, max_batch=max_batch)
        assert isinstance(waves, Waves)  # Type narrowing for MyPy
        return waves

    def transition_potential_multislice(
        self,
        potential: BasePotential,
        transition_potentials: BaseTransitionPotential | list[BaseTransitionPotential],
        detectors: Optional[BaseDetector | list[BaseDetector]] = None,
        sites: Optional[SliceIndexedAtoms | Atoms] = None,
    ) -> Waves | BaseMeasurements | ComputableList[Waves | BaseMeasurements]:
        transition_potentials = _validate_transition_potentials(transition_potentials)

        potential = validate_potential(potential, self)

        measurements: list[Waves | BaseMeasurements] = []
        for transition_potential in transition_potentials:
            multislice_transform = MultisliceTransform(
                potential=potential,
                detectors=detectors,
                multislice_func=transition_potential_multislice_and_detect,
                transition_potential=transition_potential,
                sites=sites,
            )
            new_measurements = self.apply_transform(multislice_transform)
            assert isinstance(
                new_measurements, (Waves, BaseMeasurements)
            )  # Type narrowing for MyPy
            measurements.append(new_measurements)

        if len(measurements) > 1:
            axis_metadata = OrdinalAxis(
                label="Z, n, l",
                values=tuple(
                    ",".join(
                        (
                            str(transition_potential.metadata["Z"]),
                            str(transition_potential.metadata["n"]),
                            str(transition_potential.metadata["l"]),
                        )
                    )
                    for transition_potential in transition_potentials
                ),
                tex_label=r"$Z, n, \ell$",
            )

            assert isinstance(measurements, list)
            measurements = abtem.stack(
                measurements,
                axis_metadata,
            )
        else:
            measurements = measurements[0]

        return _reduce_ensemble(measurements)

    def multislice(
        self,
        potential: Atoms | BasePotential,
        detectors: Optional[BaseDetector | Sequence[BaseDetector]] = None,
    ) -> Waves | BaseMeasurements | list[Waves | BaseMeasurements]:
        """
        Propagate and transmit wave function through the provided potential using the
        multislice algorithm. When detector(s) are given, output will be the
        corresponding measurement.

        Parameters
        ----------
        potential : BasePotential or ASE.Atoms
            The potential through which to propagate the wave function. Optionally atoms
            can be directly given.
        detectors : BaseDetector or list of BaseDetector, optional
            A detector or a list of detectors defining how the wave functions should be
            converted to measurements after running the multislice algorithm.
            See `abtem.measurements.detect` for a list of implemented detectors. If
            not given, returns the wave functions themselves.


        Returns
        -------
        detected_waves : BaseMeasurements or list of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential
            (if no detector(s) given).
        """
        potential = validate_potential(potential, self)

        multislice_transform = MultisliceTransform(
            potential=potential, detectors=detectors
        )

        waves = self.apply_transform(transform=multislice_transform)

        return _reduce_ensemble(waves)

    def scan(
        self,
        scan: BaseScan | np.ndarray,
        potential: Optional[Atoms | BasePotential] = None,
        detectors: Optional[BaseDetector | Sequence[BaseDetector]] = None,
        max_batch: int | str = "auto",
    ) -> Waves | BaseMeasurements | ComputableList[Waves | BaseMeasurements]:
        """
        Run the multislice algorithm from probe wave functions over the provided scan.

        Parameters
        ----------
        potential : BasePotential or Atoms
            The scattering potential.
        scan : BaseScan
            Positions of the probe wave functions. If not given, scans across the entire
            potential at Nyquist sampling.
        detectors : BaseDetector, list of BaseDetector, optional
            A detector or a list of detectors defining how the wave functions should be
            converted to measurements after running the multislice algorithm.
            See abtem.measurements.detect for a list of implemented detectors.
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array.
            If 'auto' (default), the batch size is automatically chosen based on the
            abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".

        Returns
        -------
        detected_waves : BaseMeasurements or list of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential
            (if no detector(s) given).
        """
        scan = validate_scan(scan)

        waves = scan.apply(self, max_batch=max_batch)

        if potential is None:
            return waves

        measurements = waves.multislice(
            potential=potential,
            detectors=detectors,
        )

        return measurements

    def show(self, convert_complex: str = "intensity", **kwargs) -> Visualization:
        """
        Show the wave-function intensities.

        kwargs :
            Keyword arguments for `abtem.measurements.Images.show`.
        """
        return self.to_images(convert_complex=convert_complex).show(**kwargs)


class _WavesBuilder(BaseWaves, Ensemble, CopyMixin, EqualityMixin):
    def __init__(self, ensemble_names: tuple[str, ...], device: str, tilt=(0, 0)):
        self._ensemble_names = ensemble_names
        self._device = device
        self.tilt = tilt
        super().__init__()

    @property
    def tilt(self):
        """The small-angle tilt of applied to the Fresnel propagator [mrad]."""
        return self._tilt

    @tilt.setter
    def tilt(self, value):
        self._tilt = _validate_tilt(value)

    @abstractmethod
    def build(self, lazy) -> Waves:
        pass

    def apply_transform(
        self, transform, max_batch: int | str = "auto", lazy: bool = True
    ):
        return self.build(lazy=lazy).apply_transform(transform, max_batch=max_batch)

    def check_can_build(self, potential: Optional[BasePotential] = None):
        """Check whether the wave functions can be built."""
        if potential is not None:
            self.grid.match(potential)
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

    @property
    def _ensembles(self):
        return {name: getattr(self, name) for name in self._ensemble_names}

    @property
    def _ensemble_shapes(self):
        return tuple(ensemble.ensemble_shape for ensemble in self._ensembles.values())

    @property
    def ensemble_shape(self):
        """Shape of the ensemble axes of the waves."""
        return tuple(itertools.chain(*self._ensemble_shapes))

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        """List of AxisMetadata of the ensemble axes."""
        return list(
            itertools.chain(
                *tuple(
                    ensemble.ensemble_axes_metadata
                    for ensemble in self._ensembles.values()
                )
            )
        )

    def _chunk_splits(self):
        shapes = (0,) + tuple(
            len(ensemble_shape) for ensemble_shape in self._ensemble_shapes
        )
        cumulative_shapes = np.cumsum(shapes)
        return [
            (cumulative_shapes[i], cumulative_shapes[i + 1])
            for i in range(len(cumulative_shapes) - 1)
        ]

    def _arg_splits(self):
        shapes = (0,)
        for arg_split, ensemble in zip(self._chunk_splits(), self._ensembles.values()):
            shapes += (len(ensemble._partition_args(1, lazy=True)),)
        cumulative_shapes = np.cumsum(shapes)
        return [
            (cumulative_shapes[i], cumulative_shapes[i + 1])
            for i in range(len(cumulative_shapes) - 1)
        ]

    def _partition_args(self, chunks=(1,), lazy: bool = True):
        if chunks is None:
            chunks = self._default_ensemble_chunks
            chunks = validate_chunks(
                self.ensemble_shape,
                chunks,
                max_elements="auto",
                dtype=get_dtype(complex=True),
            )

        chunks = validate_chunks(self.ensemble_shape, chunks)

        args = ()
        for arg_split, ensemble in zip(self._chunk_splits(), self._ensembles.values()):
            arg_chunks = chunks[slice(*arg_split)]
            args += ensemble._partition_args(arg_chunks, lazy=lazy)

        return args

    @classmethod
    def _from_partitioned_args_func(
        cls,
        *args,
        partials,
        arg_splits,
        **kwargs,
    ):
        args = unpack_blockwise_args(args)
        for arg_split, (name, partial_item) in zip(arg_splits, partials.items()):
            kwargs[name] = partial_item(*args[slice(*arg_split)]).item()

        if "semiangle_cutoff" in kwargs and "aperture" in kwargs:
            del kwargs["semiangle_cutoff"]

        new_probe = cls(
            **kwargs,
        )
        new_probe = _wrap_with_array(new_probe)
        return new_probe

    def _from_partitioned_args(self, *args, **kwargs):
        partials = {
            name: ensemble._from_partitioned_args()
            for name, ensemble in self._ensembles.items()
        }

        kwargs = self._copy_kwargs(exclude=tuple(self._ensembles.keys()))
        return partial(
            self._from_partitioned_args_func,
            partials=partials,
            arg_splits=self._arg_splits(),
            **kwargs,
        )

    @property
    def _default_ensemble_chunks(self):
        return ("auto",) * len(self.ensemble_shape)

    @property
    def device(self):
        """The device where the waves are created."""
        return self._device

    @property
    def shape(self):
        """Shape of the waves."""
        return self.ensemble_shape + self.base_shape

    @property
    def base_shape(self) -> tuple[int, int]:
        """Shape of the base axes of the waves."""
        return self._valid_gpts

    @property
    def axes_metadata(self) -> AxesMetadataList:
        """List of AxisMetadata."""
        return AxesMetadataList(
            self.ensemble_axes_metadata + self.base_axes_metadata, self.shape
        )

    @staticmethod
    @abstractmethod
    def _build_waves(waves_builder: _WavesBuilder, wrapped: bool = True):
        pass

    @staticmethod
    def _lazy_build_waves(waves_builder: _WavesBuilder, max_batch: int | str) -> Waves:
        if isinstance(max_batch, int):
            max_batch = int(max_batch * np.prod(waves_builder._valid_gpts))

        chunks = waves_builder._default_ensemble_chunks + waves_builder.gpts

        chunks = validate_chunks(
            shape=waves_builder.ensemble_shape + waves_builder.gpts,
            chunks=chunks,
            max_elements=max_batch,
            dtype=waves_builder.dtype,
        )

        blocks = waves_builder.ensemble_blocks(chunks=chunks[:-2])

        xp = get_array_module(waves_builder.device)

        array = da.map_blocks(
            waves_builder._build_waves,
            blocks,
            meta=xp.array((), dtype=get_dtype(complex=True)),
            new_axis=tuple_range(2, len(waves_builder.ensemble_shape)),
            chunks=blocks.chunks + waves_builder.gpts,
            wrapped=False,
        )

        return Waves(
            array,
            energy=waves_builder.energy,
            extent=waves_builder.extent,
            reciprocal_space=False,
            metadata=waves_builder.metadata,
            ensemble_axes_metadata=waves_builder.ensemble_axes_metadata,
        )


class PlaneWave(_WavesBuilder):
    """
    Represents electron probe wave functions for simulating experiments with a plane-wave probe, such as HRTEM and SAED.

    Parameters
    ----------
    extent : two float, optional
        Lateral extent of the wave function [Å].
    gpts : two int, optional
        Number of grid points describing the wave function.
    sampling : two float, optional
        Lateral sampling of the wave functions [Å]. If 'gpts' is also given, will be ignored.
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    normalize : bool, optional
        If true, normalizes the wave function such that its reciprocal space intensity sums to one. If false, the
        wave function takes a value of one everywhere.
    tilt : two float, optional
        Small-angle beam tilt [mrad] (default is (0., 0.)). Implemented by shifting the wave functions at every slice.
    device : str, optional
        The wave functions are stored on this device ('cpu' or 'gpu'). The default is determined by the user
        configuration.
    """

    def __init__(
        self,
        extent: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        energy: Optional[float] = None,
        normalize: bool = False,
        tilt: tuple[float, float] = (0.0, 0.0),
        device: Optional[str] = None,
    ):
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._accelerator = Accelerator(energy=energy)

        self._normalize = normalize
        device = validate_device(device)

        super().__init__(ensemble_names=("tilt",), device=device, tilt=tilt)

    @property
    def tilt(self):
        """The small-angle tilt of applied to the Fresnel propagator [mrad]."""
        return self._tilt

    @tilt.setter
    def tilt(self, value):
        self._tilt = _validate_tilt(value)

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
        """True if the created waves are normalized in reciprocal space."""
        return self._normalize

    @staticmethod
    def _build_waves(waves_builder, wrapped: bool = True):
        if hasattr(waves_builder, "item"):
            waves_builder = waves_builder.item()

        xp = get_array_module(waves_builder.device)

        if waves_builder.normalize:
            array = xp.full(
                waves_builder.gpts,
                1 / np.prod(waves_builder.gpts),
                dtype=get_dtype(complex=True),
            )

        else:
            array = xp.ones(waves_builder.gpts, dtype=get_dtype(complex=True))

        waves = Waves(
            array,
            energy=waves_builder.energy,
            extent=waves_builder.extent,
            metadata=waves_builder.metadata,
            reciprocal_space=False,
        )

        waves = waves.apply_transform(waves_builder.tilt)

        if not wrapped:
            waves = waves.array

        return waves

    def build(
        self,
        lazy: Optional[bool] = None,
        max_batch: int | str = "auto",
    ) -> Waves:
        """
        Build plane-wave wave functions.

        Parameters
        ----------
        lazy : bool, optional
            If True, create the wave functions lazily, otherwise, calculate instantly. If not given, defaults to the
            setting in the user configuration file.
        max_batch : int or str, optional
            The number of wave functions in each chunk of the Dask array. If 'auto' (default), the batch size is
            automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".

        Returns
        -------
        plane_waves : Waves
            The wave functions.
        """
        self.check_can_build()

        lazy = _validate_lazy(lazy)

        if not lazy:
            probes = self._build_waves(self)
        else:
            probes = self._lazy_build_waves(self, max_batch)

        return _reduce_ensemble(probes)

    def multislice(
        self,
        potential: BasePotential | Atoms,
        detectors: Optional[BaseDetector] = None,
        max_batch: int | str = "auto",
        lazy: Optional[bool] = None,
    ) -> BaseMeasurements | Waves | ComputableList[BaseMeasurements | Waves]:
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

        Returns
        -------
        measurements : BaseMeasurements or ComputableList of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential (if no detector(s) given).
        """
        potential = validate_potential(potential)
        lazy = _validate_lazy(lazy)

        self.check_can_build(potential)

        if not lazy:
            probes = self._build_waves(self)
        else:
            probes = self._lazy_build_waves(self, max_batch)

        multislice = MultisliceTransform(potential, detectors)

        measurements = probes.apply_transform(multislice)

        return _reduce_ensemble(measurements)


class Probe(_WavesBuilder):
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
        Lateral sampling of wave functions [Å]. If 'gpts' is also given, will be ignored.
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    soft : float, optional
        Taper the edge of the default aperture [mrad] (default is 2.0). Ignored if a custom aperture is given.
    tilt : two float, two 1D :class:`.BaseDistribution`, 2D :class:`.BaseDistribution`, optional
        Small-angle beam tilt [mrad]. This value should generally not exceed one degree.
    device : str, optional
        The probe wave functions will be build and stored on this device ('cpu' or 'gpu'). The default is determined by
        the user configuration.
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
        semiangle_cutoff: Optional[float] = None,
        extent: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        energy: Optional[float] = None,
        soft: bool = True,
        tilt: (
            tuple[float | BaseDistribution, float | BaseDistribution] | BaseDistribution
        ) = (
            0.0,
            0.0,
        ),
        device: Optional[str] = None,
        aperture: Optional[BaseAperture] = None,
        aberrations: Optional[Aberrations | dict] = None,
        positions: Optional[BaseScan] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ):
        self._accelerator = Accelerator(energy=energy)

        if (semiangle_cutoff is not None) and (aperture is not None):
            if not np.allclose(aperture.semiangle_cutoff, semiangle_cutoff):
                raise ValueError(
                    "provide only one of `semiangle_cutoff` or `aperture`",
                    aperture.semiangle_cutoff,
                    semiangle_cutoff,
                )

        if semiangle_cutoff is None:
            semiangle_cutoff = 30.0

        if aperture is None:
            aperture = Aperture(semiangle_cutoff=semiangle_cutoff, soft=soft)

        aperture._accelerator = self._accelerator

        if aberrations is None:
            aberrations = {}

        if isinstance(aberrations, dict):
            aberrations = Aberrations(energy=energy, **aberrations, **kwargs)

        aberrations._accelerator = self._accelerator
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self._aperture = aperture
        self._aberrations = aberrations

        self._metadata = {} if metadata is None else metadata

        if positions is None:
            positions = abtem.CustomScan(np.zeros((0, 2)), squeeze=True)

        self._positions = positions

        self.accelerator.match(self.aperture)

        ensemble_names = (
            "tilt",
            "aberrations",
            "aperture",
            "positions",
        )

        super().__init__(ensemble_names=ensemble_names, device=device, tilt=tilt)

    @property
    def positions(self) -> BaseScan:
        """The position(s) of the probe."""
        return self._positions

    @property
    def soft(self):
        """True if the aperture has a soft edge."""
        return self.aperture.soft

    @classmethod
    def _from_ctf(cls, ctf, **kwargs):
        if (ctf.angular_spread != 0.0) or (ctf.focal_spread != 0.0):
            raise ValueError("The CTF should have a zero focal or angular spread.")

        return cls(
            semiangle_cutoff=ctf.semiangle_cutoff,
            soft=ctf.soft,
            aberrations=ctf.aberration_coefficients,
            **kwargs,
        )

    @property
    def ctf(self):
        """Contrast transfer function describing the probe."""
        return CTF(
            aberration_coefficients=self.aberrations.aberration_coefficients,
            semiangle_cutoff=self.semiangle_cutoff,
            energy=self.energy,
        )

    @property
    def semiangle_cutoff(self):
        """The semiangle cutoff [mrad]."""
        return self.aperture.semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value):
        self.aperture.semiangle_cutoff = value

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

    @staticmethod
    def _build_waves(waves_builder, wrapped: bool = True):
        if hasattr(waves_builder, "item"):
            waves_builder = waves_builder.item()

        array = waves_builder.positions._evaluate_kernel(waves_builder)

        waves = Waves(
            array,
            energy=waves_builder.energy,
            extent=waves_builder.extent,
            metadata=waves_builder.metadata,
            reciprocal_space=True,
            ensemble_axes_metadata=waves_builder.positions.ensemble_axes_metadata,
        )

        waves = waves.apply_transform(waves_builder.aperture)

        waves = waves.apply_transform(waves_builder.tilt)

        waves = waves.apply_transform(waves_builder.aberrations)

        waves = waves.normalize()

        waves = waves.ensure_real_space()

        if not wrapped:
            waves = waves.array

        return waves

    def _validate_and_build(
        self,
        scan: Optional[Sequence | BaseScan] = None,
        max_batch: int | str = "auto",
        lazy: Optional[bool] = None,
        potential=None,
    ):
        self.check_can_build(potential)
        lazy = _validate_lazy(lazy)

        probe = self.copy()

        if potential is not None:
            probe.grid.match(potential)

        scan = validate_scan(scan, probe)

        if isinstance(scan, CustomScan):
            squeeze = True
        else:
            squeeze = False

        probe._positions = scan

        if not lazy:
            probes = self._build_waves(probe)
        else:
            probes = self._lazy_build_waves(probe, max_batch)

        if squeeze:
            probes = probes.squeeze(axis=(-3,))

        return probes

    def build(
        self,
        scan: Optional[Sequence | BaseScan] = None,
        max_batch: int | str = "auto",
        lazy: Optional[bool] = None,
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
        return self._validate_and_build(scan=scan, max_batch=max_batch, lazy=lazy)

    def multislice(
        self,
        potential: BasePotential | Atoms,
        scan: Optional[tuple | BaseScan] = None,
        detectors: Optional[BaseDetector] = None,
        max_batch: int | str = "auto",
        lazy: Optional[bool] = None,
    ) -> Waves | BaseMeasurements | list[Waves | BaseMeasurements]:
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

        Returns
        -------
        measurements : BaseMeasurements or Waves or list of BaseMeasurement
        """

        potential = validate_potential(potential)

        probes = self._validate_and_build(
            scan=scan, max_batch=max_batch, lazy=lazy, potential=potential
        )
        multislice = MultisliceTransform(potential, detectors)
        measurements = probes.apply_transform(multislice)

        return _reduce_ensemble(measurements)

    def transition_potential_scan(
        self,
        potential: BasePotential | Atoms,
        transition_potentials: BaseTransitionPotential | list[BaseTransitionPotential],
        scan: Optional[BaseScan | Sequence] = None,
        detectors: Optional[BaseDetector | list[BaseDetector]] = None,
        sites: Optional[SliceIndexedAtoms | Atoms] = None,
        # detectors_elastic: BaseDetector | list[BaseDetector] = None,
        double_channel: bool = True,
        threshold: float = 1.0,
        max_batch: int | str = "auto",
        lazy: Optional[bool] = None,
    ) -> Waves | BaseMeasurements | list[Waves | BaseMeasurements]:
        """
        Parameters
        ----------
        potential : BasePotential | Atoms
            The potential to be used for calculating the transition potentials.
            It can be an instance of `BasePotential` or an `Atoms` object.
        transition_potentials : BaseTransitionPotential | list[BaseTransitionPotential]
            The transition potentials to be used for multislice calculations.
            It can be an instance of `BaseTransitionPotential` or a list of `BaseTransitionPotential` objects.
        scan : tuple | BaseScan, optional
            The scan parameters. It can be a tuple or an instance of `BaseScan`.
            Defaults to None.
        detectors : BaseDetector | list[BaseDetector], optional
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm. See abtem.measurements.detect for a list of implemented detectors.
            Defaults to None, which
        sites : SliceIndexedAtoms | Atoms, optional
            The slice indexed atoms to be used for multislice calculations.
            It can be an instance of `SliceIndexedAtoms` or an `Atoms` object.
            Defaults to None.
        detectors_elastic : BaseDetector | list[BaseDetector], optional
            The elastic detectors to be used for recording the measurements.
            It can be an instance of `BaseDetector` or a list of `BaseDetector` objects.
            Defaults to None.
        double_channel : bool, optional
            A boolean indicating whether to use double channel for recording the measurements.
            Defaults to True.
        max_batch : int | str, optional
            The maximum batch size for parallel processing.
            It can be an integer or the string "auto".
            Defaults to "auto".
        lazy : bool, optional
            A boolean indicating whether to use lazy evaluation for the calculations.
            Defaults to None.

        Returns
        -------
        Waves | BaseMeasurements
            The calculated waves or measurements, depending on the value of `lazy`.

        """
        if scan is None:
            scan = GridScan()

        if detectors is None:
            detectors = FlexibleAnnularDetector()

        potential = validate_potential(potential)

        probes = self._validate_and_build(
            scan=scan, max_batch=max_batch, lazy=lazy, potential=potential
        )

        multislice = MultisliceTransform(
            potential,
            detectors,
            multislice_func=transition_potential_multislice_and_detect,
            transition_potential=transition_potentials,
            sites=sites,
            double_channel=double_channel,
            threshold=threshold,
        )

        measurements = probes.apply_transform(multislice)

        return _reduce_ensemble(measurements)

    def scan(
        self,
        potential: Atoms | BasePotential,
        scan: Optional[BaseScan | Sequence] = None,
        detectors: Optional[BaseDetector | Sequence[BaseDetector]] = None,
        max_batch: int | str = "auto",
        lazy: Optional[bool] = None,
    ) -> BaseMeasurements | Waves | list[BaseMeasurements | Waves]:
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
        detected_waves : BaseMeasurements or list of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential (if no detector(s) given).
        """

        if scan is None:
            scan = GridScan()

        if detectors is None:
            detectors = FlexibleAnnularDetector()

        measurements = self.multislice(
            scan=scan,
            potential=potential,
            detectors=detectors,
            lazy=lazy,
            max_batch=max_batch,
        )

        return measurements

    @staticmethod
    def _line_intersect_rectangle(point0, point1, lower_corner, upper_corner):
        if point0[0] == point1[0]:
            return (point0[0], lower_corner[1]), (point0[0], upper_corner[1])

        m = (point1[1] - point0[1]) / (point1[0] - point0[0])

        def _y(x):
            return m * (x - point0[0]) + point0[1]

        def _x(y):
            return (y - point0[1]) / m + point0[0]

        if _y(0) < lower_corner[1]:
            intersect0 = (_x(lower_corner[1]), _y(_x(lower_corner[1])))
        else:
            intersect0 = (0, _y(lower_corner[0]))

        if _y(upper_corner[0]) > upper_corner[1]:
            intersect1 = (_x(upper_corner[1]), _y(_x(upper_corner[1])))
        else:
            intersect1 = (upper_corner[0], _y(upper_corner[0]))

        return intersect0, intersect1

    def profiles(self, angle: float = 0.0) -> RealSpaceLineProfiles:
        """
        Create a line profile through the center of the probe.

        Parameters
        ----------
        angle : float, optional
            Angle with respect to the `x`-axis of the line profile [degree].
        """

        point1 = (self._valid_extent[0] / 2, self._valid_extent[1] / 2)

        measurement = self.build(point1).intensity()

        point2 = point1 + np.array(
            [np.cos(np.pi * angle / 180), np.sin(np.pi * angle / 180)]
        )
        point1, point2 = self._line_intersect_rectangle(
            point1, point2, (0.0, 0.0), self.extent
        )
        return measurement.interpolate_line(point1, point2)

    def show(self, convert_complex: str = "intensity", **kwargs) -> Visualization:
        """
        Show the intensity of the probe wave function.

        Parameters
        ----------
        complex_images : bool
            If true shows complex images using domain-coloring instead of the intensity.
        kwargs : Keyword arguments for the :func:`.Images.show` function.
        """
        self.grid.check_is_defined()
        wave = self.build((self._valid_extent[0] / 2, self._valid_extent[1] / 2))
        return wave.to_images(convert_complex=convert_complex).show(**kwargs)
