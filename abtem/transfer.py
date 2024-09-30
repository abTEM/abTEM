"""Module to describe the contrast transfer function (CTF) and the related apertures."""

from __future__ import annotations

import copy
from abc import abstractmethod
from collections import defaultdict
from functools import reduce
from typing import TYPE_CHECKING, Any, Mapping, Optional, SupportsFloat

import numpy as np

from abtem.core.axes import AxisMetadata, OrdinalAxis, ParameterAxis
from abtem.core.backend import cp, get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.energy import (
    Accelerator,
    HasAcceleratorMixin,
    energy2wavelength,
    reciprocal_space_sampling_to_angular_sampling,
)
from abtem.core.fft import fft_crop
from abtem.core.grid import Grid, HasGrid2DMixin, polar_spatial_frequencies
from abtem.core.utils import expand_dims_to_broadcast, get_dtype
from abtem.distributions import (
    BaseDistribution,
    _unpack_distributions,
    validate_distribution,
)
from abtem.measurements import ReciprocalSpaceLineProfiles
from abtem.transform import ReciprocalSpaceMultiplication

if TYPE_CHECKING:
    from abtem.measurements import DiffractionPatterns, Images
    from abtem.visualize import Visualization
    from abtem.waves import BaseWaves, Waves


class BaseTransferFunction(
    ReciprocalSpaceMultiplication, HasAcceleratorMixin, HasGrid2DMixin
):
    """Base class for transfer functions."""

    def __init__(
        self,
        energy: Optional[float] = None,
        extent: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        distributions: tuple[str, ...] = (),
    ):
        self._accelerator = Accelerator(energy=energy)
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        super().__init__(distributions=distributions)

    @abstractmethod
    def _evaluate_from_angular_grid(
        self, alpha: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        pass

    @property
    def angular_sampling(self) -> tuple[float, float]:
        """The sampling in scattering angles of the transfer function [mrad]."""
        return reciprocal_space_sampling_to_angular_sampling(
            self.reciprocal_space_sampling, self._valid_energy
        )

    def _angular_grid(self, device: str) -> tuple[np.ndarray, np.ndarray]:
        xp = get_array_module(device)
        alpha, phi = polar_spatial_frequencies(
            self._valid_gpts, self._valid_sampling, xp=xp
        )
        alpha *= self.wavelength
        return alpha, phi

    def _evaluate_kernel(self, waves: Optional[BaseWaves] = None) -> np.ndarray:
        """
        Evaluate the array to be multiplied with the waves in reciprocal space.

        Parameters
        ----------
        waves : BaseWaves, optional
            If given, the array will be evaluated to match the provided waves.

        Returns
        -------
        kernel : np.ndarray or dask.array.Array
        """

        if waves is None:
            device = "cpu"
        else:
            self.accelerator.match(waves)
            self.grid.match(waves)
            device = waves.device

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        alpha, phi = self._angular_grid(device)
        return self._evaluate_from_angular_grid(alpha, phi)

    def to_diffraction_patterns(
        self,
        max_angle: Optional[float] = None,
        gpts: Optional[int | tuple[int, int]] = None,
    ) -> DiffractionPatterns:
        """Converts the transfer function instance to DiffractionPatterns.

        Parameters
        ----------
        max_angle : float, optional
            The maximum diffraction angle in radians. If not provided, the maximum angle
            will be determined based on the `self._max_semiangle_cutoff` attribute of
            the instance. If neither `max_angle` nor `self._max_semiangle_cutoff` is
            available, a `RuntimeError` will be raised.
        gpts : int | tuple[int, int], optional
            The number of grid points in reciprocal space for performing Fourier
            Transform. If not provided, a default value of 128 will be used.

        Returns
        -------
        abtem.measurements.DiffractionPatterns
            The diffraction patterns obtained from the conversion.

        """
        from abtem.measurements import DiffractionPatterns

        if self.sampling is None or max_angle is not None:
            if max_angle is None and hasattr(self, "_max_semiangle_cutoff"):
                max_angle = self._max_semiangle_cutoff

            elif max_angle is None:
                raise RuntimeError()

            sampling = 1 / (max_angle * 1e-3) / 2 * self.wavelength
        else:
            sampling = self.sampling

        if self.gpts is None and gpts is None:
            gpts = 128

        ctf = self.copy()
        ctf.sampling = sampling
        ctf.gpts = gpts

        array = ctf._evaluate_kernel()
        xp = get_array_module(array)
        diffraction_patterns = DiffractionPatterns(
            xp.fft.fftshift(array, axes=(-2, -1)),
            sampling=ctf.reciprocal_space_sampling,
            ensemble_axes_metadata=ctf.ensemble_axes_metadata,
            fftshift=False,
            metadata={"energy": self.energy},
        )
        return diffraction_patterns

    def show(self, max_angle: float, **kwargs: Any) -> Visualization:
        return self.to_diffraction_patterns(max_angle=max_angle).show(**kwargs)


class BaseAperture(BaseTransferFunction):
    """Base class for apertures. Documented in the subclasses."""

    def __init__(
        self,
        semiangle_cutoff: float | BaseDistribution = np.inf,
        energy: Optional[float] = None,
        extent: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        distributions: tuple[str, ...] = (),
    ):
        self._semiangle_cutoff = semiangle_cutoff
        super().__init__(
            energy=energy,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
            distributions=distributions,
        )

    @property
    def metadata(self) -> dict:
        metadata = {}
        if not isinstance(self.semiangle_cutoff, BaseDistribution):
            metadata["semiangle_cutoff"] = self.semiangle_cutoff
        return metadata

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return []

    @property
    def _max_semiangle_cutoff(self) -> float:
        if isinstance(self.semiangle_cutoff, BaseDistribution):
            return max(self.semiangle_cutoff.values)
        else:
            return self.semiangle_cutoff

    @property
    def nyquist_sampling(self) -> float:
        """Nyquist sampling corresponding to the semiangle cutoff of the
        aperture [Å]."""
        return 1 / (4 * self._max_semiangle_cutoff / self.wavelength * 1e-3)

    @property
    def semiangle_cutoff(self) -> float | BaseDistribution:
        """Semiangle cutoff of the aperture [mrad]."""
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, semiangle_cutoff: float | BaseDistribution) -> None:
        self._semiangle_cutoff = semiangle_cutoff

    def _cropped_aperture(self) -> BaseAperture:
        if self._max_semiangle_cutoff == np.inf:
            return self

        gpts = (
            int(2 * np.ceil(self._max_semiangle_cutoff / self.angular_sampling[0] + 1)),
            int(2 * np.ceil(self._max_semiangle_cutoff / self.angular_sampling[1] + 1)),
        )

        cropped_aperture = self.copy()
        cropped_aperture.gpts = gpts
        return cropped_aperture

    def _evaluate_from_cropped(self, waves: Waves) -> np.ndarray:
        cropped = self._cropped_aperture()
        array = cropped._evaluate_kernel(waves)
        array = fft_crop(array, waves._valid_gpts)
        return array


def soft_aperture(
    alpha: np.ndarray,
    phi: np.ndarray,
    semiangle_cutoff: float | np.ndarray,
    angular_sampling: tuple[float, float],
) -> np.ndarray:
    """
    Calculates an array with a disk of ones and a soft edge.

    Parameters
    ----------
    alpha : 2D array
        Array of radial angles [mrad].
    phi : 2D array
        Array of azimuthal angles [rad].
    semiangle_cutoff : float or 1D array
        Semiangle cutoff(s) of the aperture(s). If given as an array, a 3D array is
        returned where the first dimension represents a different aperture for each
        item in the array of semiangle cutoffs.
    angular_sampling : tuple of float
        Reciprocal-space sampling in units of scattering angles [mrad].

    Returns
    -------
    soft_aperture_array : 2D or 3D np.ndarray
    """
    xp = get_array_module(alpha)

    semiangle_cutoff_array = xp.array(semiangle_cutoff, dtype=get_dtype(complex=False))

    base_ndims = len(alpha.shape)

    semiangle_cutoff_array, alpha = expand_dims_to_broadcast(
        semiangle_cutoff_array, alpha
    )

    semiangle_cutoff, phi = expand_dims_to_broadcast(
        semiangle_cutoff_array, phi, match_dims=((-2, -1), (-2, -1))
    )

    angular_sampling = xp.array(angular_sampling, dtype=get_dtype(complex=False)) * 1e-3

    denominator = xp.sqrt(
        (xp.cos(phi) * angular_sampling[0]) ** 2
        + (xp.sin(phi) * angular_sampling[1]) ** 2
    )

    ndims = len(alpha.shape)

    zeros = (slice(None),) * (ndims - base_ndims) + (0,) * base_ndims

    denominator[zeros] = 1.0

    array = xp.clip(
        (semiangle_cutoff - alpha) / denominator + 0.5, a_min=0.0, a_max=1.0
    )

    array[zeros] = 1.0
    return array


def hard_aperture(
    alpha: np.ndarray, semiangle_cutoff: float | BaseDistribution
) -> np.ndarray:
    """
    Calculates an array with a disk of ones and a soft edge.

    Parameters
    ----------
    alpha : 2D array
        Array of radial angles [mrad].
    semiangle_cutoff : float or 1D array
        Semiangle cutoff(s) of the aperture(s). If given as an array, a 3D array is
        returned where the first dimension represents a different aperture for each
        item in the array of semiangle cutoffs.

    Returns
    -------
    hard_aperture_array : 2D or 3D np.ndarray
    """
    xp = get_array_module(alpha)
    return xp.array(alpha <= semiangle_cutoff).astype(get_dtype(complex=False))


class Aperture(BaseAperture):
    """
    A circular aperture cutting off the wave function at a specified angle, employed in
    both STEM and HRTEM. The abrupt cutoff may be softened by tapering it.

    Parameters
    ----------
    semiangle_cutoff : float or BaseDistribution
        The cutoff semiangle of the aperture [mrad]. Alternatively, a distribution of
        angles may be provided.
    soft : bool, optional
        If True, the edge of the aperture is softened (default is True).
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single
        float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be
        ignored.
    """

    def __init__(
        self,
        semiangle_cutoff: float | BaseDistribution,
        soft: bool = True,
        energy: Optional[float] = None,
        extent: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
    ):
        validated_semiangle_cutoff = validate_distribution(semiangle_cutoff)
        self._soft = soft

        super().__init__(
            distributions=("semiangle_cutoff",),
            energy=energy,
            semiangle_cutoff=validated_semiangle_cutoff,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
        )

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        ensemble_axes_metadata: list[AxisMetadata] = []
        if isinstance(self.semiangle_cutoff, BaseDistribution):
            ensemble_axes_metadata = [
                ParameterAxis(
                    label="semiangle_cutoff",
                    values=tuple(self.semiangle_cutoff),
                    units="mrad",
                    tex_label="$\\alpha_{cut}$",
                    _ensemble_mean=self.semiangle_cutoff.ensemble_mean,
                )
            ]
        return ensemble_axes_metadata

    @property
    def soft(self) -> bool:
        """True if the aperture has a soft edge."""
        return self._soft

    def _evaluate_from_angular_grid(
        self, alpha: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        xp = get_array_module(alpha)

        if self.semiangle_cutoff == xp.inf:
            return xp.ones_like(alpha)

        semiangle_cutoff = xp.array(self.semiangle_cutoff) * 1e-3
        
        if (
            self.soft
            and self.grid.check_is_defined(False)
            and not np.isscalar(alpha)
            and not np.isscalar(phi)
        ):
            aperture = soft_aperture(
                alpha, phi, semiangle_cutoff, self.angular_sampling
            )
            return aperture
        else:
            return hard_aperture(alpha, semiangle_cutoff)


class Bullseye(BaseAperture):
    """
    Bullseye aperture.

    Parameters
    ----------
    num_spokes : int
        Number of spokes.
    spoke_width : float
        Width of spokes [deg].
    num_rings : int
        Number of rings.
    ring_width : float
        Width of rings [mrad].
    semiangle_cutoff : float
        The cutoff semiangle of the aperture [mrad].
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single
        float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be
        ignored.
    """

    def __init__(
        self,
        num_spokes: int,
        spoke_width: float,
        num_rings: int,
        ring_width: float,
        semiangle_cutoff: float,
        energy: Optional[float] = None,
        extent: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
    ):
        self._spoke_num = num_spokes
        self._spoke_width = spoke_width
        self._num_rings = num_rings
        self._ring_width = ring_width
        super().__init__(
            energy=energy,
            semiangle_cutoff=semiangle_cutoff,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
        )

    @property
    def soft(self) -> bool:
        """True if the aperture has a soft edge."""
        return False

    @property
    def num_spokes(self) -> int:
        """Number of spokes."""
        return self._spoke_num

    @property
    def spoke_width(self) -> float:
        """Width of spokes [deg]."""
        return self._spoke_width

    @property
    def num_rings(self) -> int:
        """Number of rings."""
        return self._num_rings

    @property
    def ring_width(self) -> float:
        """Width of rings [mrad]."""
        return self._ring_width

    def _evaluate_from_angular_grid(
        self, alpha: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        xp = get_array_module(alpha)
        alpha = xp.array(alpha)

        semiangle_cutoff = self.semiangle_cutoff
        assert isinstance(semiangle_cutoff, SupportsFloat)

        semiangle_cutoff = semiangle_cutoff / 1e3

        array = alpha < semiangle_cutoff

        # add crossbars
        array = array * (
            ((phi + np.pi * self.spoke_width / (180 * 2)) * self.num_spokes)
            % (2 * np.pi)
            > (np.pi * self.spoke_width / 180 * self.num_spokes)
        )

        # add ring bars
        end_edges = np.linspace(
            semiangle_cutoff / self.num_rings, semiangle_cutoff, self.num_rings
        )
        start_edges = end_edges - self.ring_width / 1e3

        for start_edge, end_edge in zip(start_edges, end_edges):
            array[(alpha > start_edge) * (alpha < end_edge)] = 0.0

        return array


class Vortex(BaseAperture):
    """
    Vortex-beam aperture.

    Parameters
    ----------
    quantum_number : int
        Quantum number of vortex beam.
    semiangle_cutoff : float
        The cutoff semiangle of the aperture [mrad].
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single
        float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be
        ignored.
    """

    def __init__(
        self,
        quantum_number: int,
        semiangle_cutoff: float,
        energy: Optional[float] = None,
        extent: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
    ):
        self._quantum_number = quantum_number
        super().__init__(
            energy=energy,
            semiangle_cutoff=semiangle_cutoff,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
        )

    @property
    def soft(self) -> bool:
        """True if the aperture has a soft edge."""
        return False

    @property
    def quantum_number(self) -> int:
        """Quantum number of vortex beam."""
        return self._quantum_number

    def _evaluate_from_angular_grid(
        self, alpha: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        xp = get_array_module(alpha)
        alpha = xp.array(alpha)

        semiangle_cutoff = self.semiangle_cutoff
        assert isinstance(semiangle_cutoff, SupportsFloat)
        semiangle_cutoff = semiangle_cutoff / 1e3

        array = alpha < semiangle_cutoff
        array = array * np.exp(1j * phi * self.quantum_number)
        return array


class Zernike(BaseAperture):
    """
    Zernike aperture.

    Parameters
    ----------
    center_hole_cutoff : float
        Cutoff semiangle of aperture hole [mrad].
    phase_shift: float
        Phase shift of Zernike film [rad]
    semiangle_cutoff : float
        The cutoff semiangle of the aperture [mrad].
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single
        float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be
        ignored.
    """

    def __init__(
        self,
        center_hole_cutoff: float,
        phase_shift: float,
        semiangle_cutoff: float,
        energy: Optional[float] = None,
        extent: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
    ):
        self._center_hole_cutoff = center_hole_cutoff
        self._phase_shift = phase_shift
        super().__init__(
            energy=energy,
            semiangle_cutoff=semiangle_cutoff,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
        )

    @property
    def center_hole_cutoff(self) -> float:
        """Cutoff semiangle of aperture hole."""
        return self._center_hole_cutoff

    @property
    def soft(self) -> bool:
        """True if the aperture has a soft edge."""
        return False

    @property
    def phase_shift(self) -> float:
        """Phase shift of Zernike film."""
        return self._phase_shift

    def _evaluate_from_angular_grid(
        self, alpha: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        xp = get_array_module(alpha)
        alpha = xp.array(alpha)
        semiangle_cutoff = self.semiangle_cutoff
        assert isinstance(semiangle_cutoff, SupportsFloat)

        semiangle_cutoff = semiangle_cutoff / 1e3
        center_hole_cutoff = self.center_hole_cutoff / 1e3
        phase_shift = self.phase_shift

        amplitude = xp.asarray(alpha < semiangle_cutoff, dtype=get_dtype(complex=False))
        phase_array = xp.asarray(
            xp.logical_and(alpha > center_hole_cutoff, alpha < semiangle_cutoff),
            dtype=get_dtype(complex=False),
        )
        phase = xp.exp(1.0j * phase_shift * phase_array)
        array = amplitude * phase

        return array


class TemporalEnvelope(BaseTransferFunction):
    """
    Envelope function for simulating partial temporal coherence in the quasi-coherent
    approximation.

    Parameters
    ----------
    focal_spread: float or 1D array or BaseDistribution
        The standard deviation of the focal spread due to chromatic aberration and lens
        current instability [Å].
        Alternatively, a distribution of values may be provided.
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single
        float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be
        ignored.
    """

    def __init__(
        self,
        focal_spread: float | BaseDistribution,
        energy: Optional[float] = None,
        extent: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
    ):
        self._accelerator = Accelerator(energy=energy)
        self._focal_spread = validate_distribution(focal_spread)
        super().__init__(
            distributions=("focal_spread",),
            energy=energy,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
        )

    @property
    def focal_spread(self) -> float | BaseDistribution:
        """The standard deviation of the focal spread [Å]."""
        return self._focal_spread

    @focal_spread.setter
    def focal_spread(self, value: float | BaseDistribution) -> None:
        self._focal_spread = validate_distribution(value)

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return self._get_axes_metadata_from_distributions(
            focal_spread={"units": "mrad"}
        )

    def _evaluate_from_angular_grid(
        self, alpha: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        xp = get_array_module(alpha)

        unpacked, _ = _unpack_distributions(self.focal_spread, shape=alpha.shape, xp=xp)
        (focal_spread,) = unpacked

        alpha = xp.array(alpha)
        alpha = xp.expand_dims(alpha, axis=tuple(range(0, self._num_ensemble_axes)))

        array = xp.exp(
            -((0.5 * xp.pi / self.wavelength * focal_spread * alpha**2) ** 2)
        ).astype(get_dtype(complex=False))

        return array


def symbol_to_tex_symbol(symbol: str) -> str:
    return symbol.replace("C", "C_{").replace("phi", "\\phi_{") + "}"


polar_aliases = {
    "defocus": "C10",
    "Cs": "C30",
    "C5": "C50",
    "astigmatism": "C12",
    "astigmatism_angle": "phi12",
    "astigmatism3": "C32",
    "astigmatism3_angle": "phi32",
    "astigmatism5": "C52",
    "astigmatism5_angle": "phi52",
    "coma": "C21",
    "coma_angle": "phi21",
    "coma4": "C41",
    "coma4_angle": "phi41",
    "trefoil": "C23",
    "trefoil_angle": "phi23",
    "trefoil4": "C43",
    "trefoil4_angle": "phi43",
    "quadrafoil": "C34",
    "quadrafoil_angle": "phi34",
    "quadrafoil5": "C54",
    "quadrafoil5_angle": "phi54",
    "pentafoil": "C45",
    "pentafoil_angle": "phi45",
    "hexafoil": "C56",
    "hexafoil_angle": "phi56",
}

polar_symbols = {value: key for key, value in polar_aliases.items()}


class _HasAberrations(HasAcceleratorMixin):
    C10: float | BaseDistribution
    C12: float | BaseDistribution
    phi12: float | BaseDistribution
    C21: float | BaseDistribution
    phi21: float | BaseDistribution
    C23: float | BaseDistribution
    phi23: float | BaseDistribution
    C30: float | BaseDistribution
    C32: float | BaseDistribution
    phi32: float | BaseDistribution
    C34: float | BaseDistribution
    phi34: float | BaseDistribution
    C41: float | BaseDistribution
    phi41: float | BaseDistribution
    C43: float | BaseDistribution
    phi43: float | BaseDistribution
    C45: float | BaseDistribution
    phi45: float | BaseDistribution
    C50: float | BaseDistribution
    C52: float | BaseDistribution
    phi52: float | BaseDistribution
    C54: float | BaseDistribution
    phi54: float | BaseDistribution
    C56: float | BaseDistribution
    phi56: float | BaseDistribution
    Cs: float | BaseDistribution
    C5: float | BaseDistribution
    astigmatism: float | BaseDistribution
    astigmatism_angle: float | BaseDistribution
    astigmatism3: float | BaseDistribution
    astigmatism3_angle: float | BaseDistribution
    astigmatism5: float | BaseDistribution
    astigmatism5_angle: float | BaseDistribution
    coma: float | BaseDistribution
    coma_angle: float | BaseDistribution
    coma4: float | BaseDistribution
    coma4_angle: float | BaseDistribution
    trefoil: float | BaseDistribution
    trefoil_angle: float | BaseDistribution
    trefoil4: float | BaseDistribution
    trefoil4_angle: float | BaseDistribution
    quadrafoil: float | BaseDistribution
    quadrafoil_angle: float | BaseDistribution
    quadrafoil5: float | BaseDistribution
    quadrafoil5_angle: float | BaseDistribution
    pentafoil: float | BaseDistribution
    pentafoil_angle: float | BaseDistribution
    hexafoil: float | BaseDistribution
    hexafoil_angle: float | BaseDistribution

    def __init__(self, *args, **kwargs):
        self._aberration_coefficients = {
            symbol: 0.0 for symbol in polar_symbols.keys()
        }
        super().__init__(*args, **kwargs)

    def __getattr__(self, name: str) -> float | BaseDistribution:
        name = polar_aliases.get(name, name)

        if name not in polar_symbols:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        return self._aberration_coefficients.get(name, 0.0)

    def __setattr__(self, name: str, value: float | BaseDistribution) -> None:
        if name == "defocus":
            super().__setattr__(name, value)
            return

        name = polar_aliases.get(name, name)

        if name in polar_symbols:
            self._aberration_coefficients[name] = validate_distribution(value)
        else:
            super().__setattr__(name, value)

    @property
    def defocus(self) -> float | BaseDistribution:
        """Defocus equivalent to negative C10."""
        return -self.C10

    @defocus.setter
    def defocus(self, value: float | BaseDistribution) -> None:
        self.C10 = -value

    def _nonzero_coefficients(self, symbols: tuple[str, ...]) -> bool:
        for symbol in symbols:
            if not np.isscalar(self._aberration_coefficients[symbol]):
                return True

            if not self._aberration_coefficients[symbol] == 0.0:
                return True

        return False

    @property
    def _phase_aberrations_ensemble_axes_metadata(self) -> list[AxisMetadata]:
        axes_metadata: list[AxisMetadata] = []
        for parameter_name, value in self._aberration_coefficients.items():
            if isinstance(value, BaseDistribution):
                axes_metadata += [
                    ParameterAxis(
                        label=parameter_name,
                        values=tuple(value.values),
                        units="Å",
                        _ensemble_mean=value.ensemble_mean,
                        tex_label=symbol_to_tex_symbol(parameter_name),
                    )
                ]
        return axes_metadata

    @property
    def aberration_coefficients(self) -> Mapping[str, float | BaseDistribution]:
        """The aberration coefficients as a dictionary."""
        return copy.deepcopy(self._aberration_coefficients)

    @property
    def _has_aberrations(self) -> bool:
        if np.all(
            [np.all(value == 0.0) for value in self._aberration_coefficients.values()]
        ):
            return False
        else:
            return True

    def set_aberrations(
        self, aberration_coefficients: Mapping[str, str | float | BaseDistribution]
    ) -> None:
        """
        Set the phase of the phase aberration.

        Parameters
        ----------
        aberration_coefficients : dict
            Mapping from aberration symbols to their corresponding values.
        """
        for symbol, value in aberration_coefficients.items():
            if symbol in ("defocus", "C10"):
                if isinstance(value, str) and value.lower() == "scherzer":
                    if self.energy is None:
                        raise RuntimeError(
                            "energy undefined, Scherzer defocus cannot be evaluated"
                        )
                    C30 = self._aberration_coefficients["C30"]
                    assert isinstance(C30, SupportsFloat)
                    value = scherzer_defocus(float(C30), self._valid_energy)

            if isinstance(value, str):
                raise ValueError("string values only allowed for defocus")

            value = validate_distribution(value)

            setattr(self, symbol, value)

        #     if not isinstance(value, str):
        #         value = validate_distribution(value)

        #     if symbol in polar_symbols:
        #         self._aberration_coefficients[symbol] = value

        #     elif symbol in polar_aliases:
        #         self._aberration_coefficients[self._aliases()[symbol]] = value

        #     else:
        #         raise ValueError("{} not a recognized parameter".format(symbol))

        # for symbol, value in aberration_coefficients.items():
        #     if symbol in ("defocus", "C10"):
        #         if isinstance(value, str) and value.lower() == "scherzer":
        #             if self._valid_energy is None:
        #                 raise RuntimeError(
        #                     "energy undefined, Scherzer defocus cannot be evaluated"
        #                 )

        #             value = scherzer_defocus(
        #                 self._aberration_coefficients["C30"], self._valid_energy
        #             )
        #         elif isinstance(value, str):
        #             raise ValueError(
        #                 f"String values for defocus must be 'Scherzer', got {value}"
        #             )

        #         value = validate_distribution(value)

        #         if symbol == "defocus":
        #             value = -value

        #         self._aberration_coefficients["C10"] = value


class SpatialEnvelope(BaseTransferFunction, _HasAberrations):
    """
    Envelope function for simulating partial spatial coherence in the quasi-coherent
    approximation.

    Parameters
    ----------
    angular_spread: float or 1D array or BaseDistribution
        The standard deviation of the angular deviations due to source size [mrad].
        Alternatively, a distribution of standard deviations may be provided.
    aberration_coefficients: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration
        magnitudes should be given in [Å] and angles should be given in [radian].
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single
        float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be
        ignored.
    kwargs : dict, optional
        Optionally provide the aberration coefficients as keyword arguments.
    """

    def __init__(
        self,
        angular_spread: float | BaseDistribution,
        aberration_coefficients: Optional[
            Mapping[str, str | float | BaseDistribution]
        ] = None,
        energy: Optional[float] = None,
        extent: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        **kwargs: str | float | BaseDistribution,
    ):
        distributions = tuple(polar_symbols.keys()) + ("angular_spread",)
        super().__init__(
            distributions=distributions,
            energy=energy,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
        )

        self._angular_spread = validate_distribution(angular_spread)

        aberration_coefficients = (
            {} if aberration_coefficients is None else aberration_coefficients
        )
        aberration_coefficients = {**aberration_coefficients, **kwargs}
        self.set_aberrations(aberration_coefficients)

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        ensemble_axes_metadata = [
            *self._phase_aberrations_ensemble_axes_metadata,
            *self._get_axes_metadata_from_distributions(
                angular_spread={"units": "mrad"}
            ),
        ]
        return ensemble_axes_metadata

    @property
    def angular_spread(self) -> float | BaseDistribution:
        """The standard deviation of the angular deviations due to source size
        [mrad]."""
        return self._angular_spread

    @angular_spread.setter
    def angular_spread(self, value: float | BaseDistribution) -> None:
        self._angular_spread = value

    def _evaluate_from_angular_grid(
        self, alpha: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        xp = get_array_module(alpha)

        args = tuple(self.aberration_coefficients.values()) + (self.angular_spread,)

        unpacked, _ = _unpack_distributions(*args, shape=alpha.shape, xp=xp)
        angular_spread = unpacked[-1] / 1e3
        parameters = dict(zip(polar_symbols, unpacked[:-1]))

        alpha = xp.array(alpha)
        alpha = xp.expand_dims(alpha, axis=tuple(range(0, self._num_ensemble_axes)))

        xp = get_array_module(alpha)

        dchi_dk = (
            2
            * xp.pi
            / self.wavelength
            * (
                (
                    parameters["C12"] * xp.cos(2.0 * (phi - parameters["phi12"]))
                    + parameters["C10"]
                )
                * alpha
                + (
                    parameters["C23"] * xp.cos(3.0 * (phi - parameters["phi23"]))
                    + parameters["C21"] * xp.cos(1.0 * (phi - parameters["phi21"]))
                )
                * alpha**2
                + (
                    parameters["C34"] * xp.cos(4.0 * (phi - parameters["phi34"]))
                    + parameters["C32"] * xp.cos(2.0 * (phi - parameters["phi32"]))
                    + parameters["C30"]
                )
                * alpha**3
                + (
                    parameters["C45"] * xp.cos(5.0 * (phi - parameters["phi45"]))
                    + parameters["C43"] * xp.cos(3.0 * (phi - parameters["phi43"]))
                    + parameters["C41"] * xp.cos(1.0 * (phi - parameters["phi41"]))
                )
                * alpha**4
                + (
                    parameters["C56"] * xp.cos(6.0 * (phi - parameters["phi56"]))
                    + parameters["C54"] * xp.cos(4.0 * (phi - parameters["phi54"]))
                    + parameters["C52"] * xp.cos(2.0 * (phi - parameters["phi52"]))
                    + parameters["C50"]
                )
                * alpha**5
            )
        )

        dchi_dphi = (
            -2
            * xp.pi
            / self.wavelength
            * (
                1
                / 2.0
                * (2.0 * parameters["C12"] * xp.sin(2.0 * (phi - parameters["phi12"])))
                * alpha
                + 1
                / 3.0
                * (
                    3.0 * parameters["C23"] * xp.sin(3.0 * (phi - parameters["phi23"]))
                    + 1.0
                    * parameters["C21"]
                    * xp.sin(1.0 * (phi - parameters["phi21"]))
                )
                * alpha**2
                + 1
                / 4.0
                * (
                    4.0 * parameters["C34"] * xp.sin(4.0 * (phi - parameters["phi34"]))
                    + 2.0
                    * parameters["C32"]
                    * xp.sin(2.0 * (phi - parameters["phi32"]))
                )
                * alpha**3
                + 1
                / 5.0
                * (
                    5.0 * parameters["C45"] * xp.sin(5.0 * (phi - parameters["phi45"]))
                    + 3.0
                    * parameters["C43"]
                    * xp.sin(3.0 * (phi - parameters["phi43"]))
                    + 1.0
                    * parameters["C41"]
                    * xp.sin(1.0 * (phi - parameters["phi41"]))
                )
                * alpha**4
                + (1 / 6.0)
                * (
                    6.0 * parameters["C56"] * xp.sin(6.0 * (phi - parameters["phi56"]))
                    + 4.0
                    * parameters["C54"]
                    * xp.sin(4.0 * (phi - parameters["phi54"]))
                    + 2.0
                    * parameters["C52"]
                    * xp.sin(2.0 * (phi - parameters["phi52"]))
                )
                * alpha**5
            )
        )

        array = xp.exp(
            -xp.sign(angular_spread)
            * (angular_spread / 2) ** 2
            * (dchi_dk**2 + dchi_dphi**2)
        )

        return array


class Aberrations(BaseTransferFunction, _HasAberrations):
    """
    Phase aberrations.

    Parameters
    ----------
    aberration_coefficients: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration
        magnitudes should be given in [Å] and angles should be given in [radian].
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single
        float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [Å]. If 'gpts' is also given, will be
        ignored.
    kwargs : dict, optional
        Optionally provide the aberration coefficients as keyword arguments.
    """

    def __init__(
        self,
        aberration_coefficients: Optional[
            Mapping[str, str | float | BaseDistribution]
        ] = None,
        energy: Optional[float] = None,
        extent: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            distributions=tuple(polar_symbols.keys()),
            energy=energy,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
        )

        aberration_coefficients = (
            {} if aberration_coefficients is None else aberration_coefficients
        )

        aberration_coefficients = {**aberration_coefficients, **kwargs}
        self.set_aberrations(aberration_coefficients)

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return self._phase_aberrations_ensemble_axes_metadata

    @property
    def defocus(self) -> float | BaseDistribution:
        """The defocus [Å]."""
        return -self._aberration_coefficients["C10"]

    @defocus.setter
    def defocus(self, value: float | BaseDistribution) -> None:
        self.C10 = -value

    def _evaluate_from_angular_grid(
        self, alpha: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        xp = get_array_module(alpha)

        if not self._has_aberrations:
            return xp.ones(
                self.ensemble_shape + alpha.shape, dtype=get_dtype(complex=True)
            )

        parameter_values, weights = _unpack_distributions(
            *tuple(self.aberration_coefficients.values()), shape=alpha.shape, xp=xp
        )

        parameters = {
            symbol: value for symbol, value in zip(polar_symbols, parameter_values)
        }

        axis = tuple(range(0, len(self.ensemble_shape)))
        alpha = xp.expand_dims(alpha, axis=axis)
        phi = xp.expand_dims(phi, axis=axis).astype(get_dtype(complex=False))

        array = xp.zeros(alpha.shape, dtype=get_dtype(complex=False))
        if self._nonzero_coefficients(("C10", "C12", "phi12")):
            array = array + (
                1
                / 2
                * alpha**2
                * (
                    parameters["C10"]
                    + parameters["C12"] * xp.cos(2 * (phi - parameters["phi12"]))
                )
            )

        if self._nonzero_coefficients(("C21", "phi21", "C23", "phi23")):
            array = array + (
                1
                / 3
                * alpha**3
                * (
                    parameters["C21"] * xp.cos(phi - parameters["phi21"])
                    + parameters["C23"] * xp.cos(3 * (phi - parameters["phi23"]))
                )
            )

        if self._nonzero_coefficients(("C30", "C32", "phi32", "C34", "phi34")):
            array = array + (
                1
                / 4
                * alpha**4
                * (
                    parameters["C30"]
                    + parameters["C32"] * xp.cos(2 * (phi - parameters["phi32"]))
                    + parameters["C34"] * xp.cos(4 * (phi - parameters["phi34"]))
                )
            )

        if self._nonzero_coefficients(("C41", "phi41", "C43", "phi43", "C45", "phi45")):
            array = array + (
                1
                / 5
                * alpha**5
                * (
                    parameters["C41"] * xp.cos((phi - parameters["phi41"]))
                    + parameters["C43"] * xp.cos(3 * (phi - parameters["phi43"]))
                    + parameters["C45"] * xp.cos(5 * (phi - parameters["phi45"]))
                )
            )

        if self._nonzero_coefficients(
            ("C50", "C52", "phi52", "C54", "phi54", "C56", "phi56")
        ):
            array = array + (
                1
                / 6
                * alpha**6
                * (
                    parameters["C50"]
                    + parameters["C52"] * xp.cos(2 * (phi - parameters["phi52"]))
                    + parameters["C54"] * xp.cos(4 * (phi - parameters["phi54"]))
                    + parameters["C56"] * xp.cos(6 * (phi - parameters["phi56"]))
                )
            )

        dtype = get_dtype(complex=False)
        array *= np.array(2 * xp.pi / self.wavelength, dtype=dtype)
        array = complex_exponential(-array)

        if cp is not None:
            weights = cp.asnumpy(weights)

        if weights is not None:
            array = xp.asarray(weights, dtype=dtype) * array

        return array


class CTF(_HasAberrations, BaseAperture):
    """
    The contrast transfer function (CTF) describes the aberrations of the objective lens
    in HRTEM and specifies how the condenser system shapes the probe in STEM.

    abTEM implements phase aberrations up to 5th order using polar coefficients.
    See Eq. 2.22 in the reference [1]_.

    Cartesian coefficients can be converted to polar using the utility function
    `abtem.transfer.cartesian2polar`.

    Partial coherence is included as envelopes in the quasi-coherent approximation.
    See Chapter 3.2 in reference [1]_.

    Parameters
    ----------
    semiangle_cutoff: float, optional
        The semiangle cutoff describes the sharp reciprocal-space cutoff due to the
        objective aperture [mrad] (default is no cutoff).
    soft : bool, optional
        If True, the edge of the aperture is softened (default is True).
    focal_spread: float, optional
        The standard deviation of the focal spread due to chromatic aberration and lens
        current instability [Å] (default is 0).
    angular_spread: float, optional
        The standard deviation of the angular deviations due to source size [Å]
        (default is 0).
    aberration_coefficients: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration
        magnitudes should be given in [Å] and angles should be given in [radian].
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single
        float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be
        ignored.
    flip_phase : bool, optional
        Changes the sign of all negative parts of the CTF to positive
        (following doi:10.1016/j.ultramic.2008.03.004) (default is False).
    wiener_snr : float, optional
        Applies a Wiener filter to the CTF(following doi:10.1016/j.ultramic.2008.03.004)
        with a given SNR value. If no value is given, the default value of 0.0 means
        that no filter is applied.
    kwargs : dict, optional
        Optionally provide the aberration coefficients as keyword arguments.

    References
    ----------
    .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy (2nd ed.).
       Springer.

    """

    def __init__(
        self,
        semiangle_cutoff: float | BaseDistribution = np.inf,
        soft: bool = True,
        focal_spread: float | BaseDistribution = 0.0,
        angular_spread: float | BaseDistribution = 0.0,
        aberration_coefficients: Optional[
            Mapping[str, float | BaseDistribution]
        ] = None,
        energy: Optional[float] = None,
        extent: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        flip_phase: bool = False,
        wiener_snr: float = 0.0,
        **kwargs: Any,
    ):
        distributions = (
            *tuple(polar_symbols.keys()),
            "angular_spread",
            "focal_spread",
            "semiangle_cutoff",
        )

        semiangle_cutoff = validate_distribution(semiangle_cutoff)

        super().__init__(
            distributions=distributions,
            energy=energy,
            semiangle_cutoff=semiangle_cutoff,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
        )
        aberration_coefficients = (
            {} if aberration_coefficients is None else aberration_coefficients
        )
        aberration_coefficients = {**aberration_coefficients, **kwargs}
        self.set_aberrations(aberration_coefficients)

        self._angular_spread = validate_distribution(angular_spread)
        self._focal_spread = validate_distribution(focal_spread)

        self._soft = soft
        self._flip_phase = flip_phase
        self._wiener_snr = wiener_snr

    @property
    def scherzer_defocus(self) -> float:
        """The Scherzer defocus [Å]."""

        if self.Cs == 0.0:
            raise ValueError("Cs must be defined to calculate Scherzer defocus")

        Cs = self.Cs
        assert isinstance(Cs, SupportsFloat)
        return scherzer_defocus(Cs, self._valid_energy)

    @property
    def crossover_angle(self) -> float:
        """The first zero-crossing of the phase at Scherzer defocus [mrad]."""
        return 1e3 * energy2wavelength(self._valid_energy) / self.point_resolution

    @property
    def point_resolution(self) -> float:
        """The Scherzer point resolution [Å]."""
        Cs = self.Cs
        assert isinstance(Cs, SupportsFloat)
        return point_resolution(Cs, self._valid_energy)

    @property
    def _aberrations(self) -> Aberrations:
        return Aberrations(
            aberration_coefficients=self.aberration_coefficients,
            energy=self.energy,
            extent=self.extent,
            gpts=self.gpts,
        )

    @property
    def _aperture(self) -> Aperture:
        return Aperture(
            semiangle_cutoff=self.semiangle_cutoff,
            soft=self._soft,
            energy=self.energy,
            extent=self.extent,
            gpts=self.gpts,
        )

    @property
    def _spatial_envelope(self) -> SpatialEnvelope:
        return SpatialEnvelope(
            aberration_coefficients=self.aberration_coefficients,
            angular_spread=self.angular_spread,
            energy=self.energy,
        )

    @property
    def _temporal_envelope(self) -> TemporalEnvelope:
        return TemporalEnvelope(
            focal_spread=self.focal_spread,
            energy=self.energy,
        )

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return (
            self._spatial_envelope.ensemble_axes_metadata
            + self._temporal_envelope.ensemble_axes_metadata
            + self._aperture.ensemble_axes_metadata
        )

    @property
    def soft(self) -> float:
        """True if the aperture has a soft edge."""
        return self._soft

    @property
    def semiangle_cutoff(self) -> float | BaseDistribution:
        """The semiangle cutoff [mrad]."""
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value: float) -> None:
        self._semiangle_cutoff = value

    @property
    def focal_spread(self) -> float | BaseDistribution:
        """The standard deviation of the focal spread [Å]."""
        return self._focal_spread

    @focal_spread.setter
    def focal_spread(self, value: float) -> None:
        self._focal_spread = value

    @property
    def angular_spread(self) -> float | BaseDistribution:
        """The standard deviation of the angular deviations due to source size
        [mrad]."""
        return self._angular_spread

    @angular_spread.setter
    def angular_spread(self, value: float) -> None:
        self._angular_spread = value

    @property
    def flip_phase(self) -> bool:
        """If true the signs of all negative parts of the CTF are changed to
        positive."""
        return self._flip_phase

    @flip_phase.setter
    def flip_phase(self, value: bool) -> None:
        self._flip_phase = value

    @property
    def wiener_snr(self) -> float:
        """If true a Wiener filter is applied to the CTF."""
        return self._wiener_snr

    @wiener_snr.setter
    def wiener_snr(self, value: float) -> None:
        self._wiener_snr = value

    def _evaluate_to_match(
        self,
        component: Aberrations | Aperture | SpatialEnvelope | TemporalEnvelope,
        alpha: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        expanded_axes: tuple[int, ...] = ()
        for i, axis_metadata in enumerate(self.ensemble_axes_metadata):
            expand = all([a != axis_metadata for a in component.ensemble_axes_metadata])
            if expand:
                expanded_axes += (i,)

        array = component._evaluate_from_angular_grid(alpha, phi)

        return np.expand_dims(array, expanded_axes)

    def _evaluate_from_angular_grid(
        self, alpha: np.ndarray, phi: np.ndarray, keep_all: bool = False
    ) -> np.ndarray:
        match_dims = tuple(range(-len(alpha.shape), 0))

        array = self._aberrations._evaluate_from_angular_grid(alpha, phi)

        if self._spatial_envelope.angular_spread != 0.0:
            new_aberrations_dims = tuple(range(len(self._aberrations.ensemble_shape)))
            old_match_dims = new_aberrations_dims + match_dims

            added_dims = int(hasattr(self._spatial_envelope.angular_spread, "values"))
            new_match_dims = (
                tuple(range(len(self._spatial_envelope.ensemble_shape) - added_dims))
                + match_dims
            )

            new_array = self._spatial_envelope._evaluate_from_angular_grid(alpha, phi)
            array, new_array = expand_dims_to_broadcast(
                array, new_array, match_dims=(old_match_dims, new_match_dims)
            )

            array = array * new_array

        if self._temporal_envelope.focal_spread != 0.0:
            new_array = self._temporal_envelope._evaluate_from_angular_grid(alpha, phi)
            array, new_array = expand_dims_to_broadcast(
                array, new_array, match_dims=(match_dims, match_dims)
            )
            array = array * new_array

        if self._aperture.semiangle_cutoff != np.inf:
            new_array = self._aperture._evaluate_from_angular_grid(alpha, phi)
            array, new_array = expand_dims_to_broadcast(
                array, new_array, match_dims=(match_dims, match_dims)
            )
            array = array * new_array

        if self._wiener_snr != 0.0:
            return (
                (1 + 1 / self._wiener_snr)
                * array**2
                / (array**2 + 1 / self._wiener_snr)
            )

        elif self._flip_phase:
            return array.real - 1j * np.abs(array.imag)

        else:
            return array

    def to_point_spread_functions(
        self, gpts: int | tuple[int, int], extent: float | tuple[float, float]
    ) -> Images:
        from abtem.waves import Probe

        return (
            Probe(gpts=gpts, extent=extent, energy=self.energy, aperture=self)
            .build()
            .to_images()
        )

    def profiles(
        self,
        gpts: int = 1000,
        max_angle: Optional[float] = None,
        phi: float | np.ndarray = 0.0,
    ) -> ReciprocalSpaceLineProfiles:
        """
        Calculate radial line profiles for each included component (phase aberrations,
        aperture, temporal and spatial envelopes) of the contrast transfer function.

        Parameters
        ----------
        gpts: int
            Number of grid points along the line profiles.
        max_angle : float
            The maximum scattering angle included in the radial line profiles [mrad].
            The default is 1.5 times the semiangle cutoff or 50 mrad if no semiangle
            cutoff is set.
        phi : float
            The azimuthal angle of the radial line profiles [rad]. Default is 0.

        Returns
        -------
        ctf_profiles : ReciprocalSpaceLineProfiles
            Ensemble of reciprocal space line profiles. The first ensemble dimension
            represents the different
        """
        if max_angle is None:
            if self.semiangle_cutoff == np.inf:
                max_angle = 50.0
            else:
                max_angle = self._max_semiangle_cutoff * 1.6

        self.accelerator.check_is_defined()

        sampling = max_angle / (gpts - 1) / (self.wavelength * 1e3)
        alpha = np.linspace(0, max_angle * 1e-3, gpts).astype(get_dtype(complex=False))

        phi = np.array(phi)

        components = dict()
        components["ctf"] = self._evaluate_to_match(self._aberrations, alpha, phi).imag

        if self._spatial_envelope.angular_spread != 0.0:
            components["spatial envelope"] = self._evaluate_to_match(
                self._spatial_envelope, alpha, phi
            )

        if self._temporal_envelope.focal_spread != 0.0:
            components["temporal envelope"] = self._evaluate_to_match(
                self._temporal_envelope, alpha, phi
            )

        if self._aperture.semiangle_cutoff != np.inf:
            components["aperture"] = self._evaluate_to_match(self._aperture, alpha, phi)

        components["ctf"] = reduce(lambda x, y: x * y, tuple(components.values()))

        ensemble_axes_metadata: list[AxisMetadata] = self.ensemble_axes_metadata
        if len(components) > 1:
            profiles = np.stack(
                np.broadcast_arrays(*list(components.values())),
                axis=-2,
            )

            component_metadata: list[AxisMetadata] = [
                OrdinalAxis(
                    label="",
                    values=tuple(components.keys()),
                )
            ]

            ensemble_axes_metadata = ensemble_axes_metadata + component_metadata
        else:
            profiles = components["ctf"]

        metadata = {"energy": self.energy}

        profiles = ReciprocalSpaceLineProfiles(
            profiles,
            sampling=sampling,
            metadata=metadata,
            ensemble_axes_metadata=ensemble_axes_metadata,
        )

        return profiles


def nyquist_sampling(semiangle_cutoff: float, energy: float) -> float:
    """
    Calculate the Nyquist sampling.

    Parameters
    ----------
    semiangle_cutoff: float
        Semiangle cutoff [mrad].
    energy: float
        Electron energy [eV].
    """
    wavelength = energy2wavelength(energy)
    return 1 / (4 * semiangle_cutoff / wavelength * 1e-3)


def scherzer_defocus(Cs: float, energy: float) -> float:
    """
    Calculate the Scherzer defocus.

    Parameters
    ----------
    Cs: float
        Spherical aberration [Å].
    energy: float
        Electron energy [eV].
    """
    return np.sign(Cs) * np.sqrt(3 / 2 * np.abs(Cs) * energy2wavelength(energy))


def point_resolution(Cs: float, energy: float) -> float:
    """
    Calculate the Scherzer point resolution.

    Parameters
    ----------
    Cs: float
        Spherical aberration [Å].
    energy: float
        Electron energy [eV].
    """
    return (energy2wavelength(energy) ** 3 * np.abs(Cs) / 6) ** (1 / 4)


def polar2cartesian(polar: dict) -> dict:
    """
    Convert between polar and Cartesian aberration coefficients.

    Parameters
    ----------
    polar : dict
        Mapping from polar aberration symbols to their corresponding values.

    Returns
    -------
    cartesian : dict
        Mapping from Cartesian aberration symbols to their corresponding values.
    """

    polar = defaultdict(lambda: 0, polar)

    cartesian = dict()
    cartesian["C10"] = polar["C10"]
    cartesian["C12a"] = -polar["C12"] * np.cos(2 * polar["phi12"])
    cartesian["C12b"] = polar["C12"] * np.sin(2 * polar["phi12"])
    cartesian["C21a"] = polar["C21"] * np.sin(polar["phi21"])
    cartesian["C21b"] = polar["C21"] * np.cos(polar["phi21"])
    cartesian["C23a"] = -polar["C23"] * np.sin(3 * polar["phi23"])
    cartesian["C23b"] = polar["C23"] * np.cos(3 * polar["phi23"])
    cartesian["C30"] = polar["C30"]
    cartesian["C32a"] = -polar["C32"] * np.cos(2 * polar["phi32"])
    cartesian["C32b"] = polar["C32"] * np.cos(np.pi / 2 - 2 * polar["phi32"])
    cartesian["C34a"] = polar["C34"] * np.cos(-4 * polar["phi34"])
    k = np.sqrt(3 + np.sqrt(8.0))
    cartesian["C34b"] = (
        1
        / 4.0
        * (1 + k**2) ** 2
        / (k**3 - k)
        * polar["C34"]
        * np.cos(4 * np.arctan(1 / k) - 4 * polar["phi34"])
    )

    return cartesian


def cartesian2polar(cartesian: dict) -> dict:
    """
    Convert between Cartesian and polar aberration coefficients.

    Parameters
    ----------
    cartesian : dict
        Mapping from Cartesian aberration symbols to their corresponding values.

    Returns
    -------
    polar : dict
        Mapping from polar aberration symbols to their corresponding values.
    """

    cartesian = defaultdict(lambda: 0, cartesian)

    polar = dict()
    polar["C10"] = cartesian["C10"]
    polar["C12"] = -np.sqrt(cartesian["C12a"] ** 2 + cartesian["C12b"] ** 2)
    polar["phi12"] = -np.arctan2(cartesian["C12b"], cartesian["C12a"]) / 2.0
    polar["C21"] = np.sqrt(cartesian["C21a"] ** 2 + cartesian["C21b"] ** 2)
    polar["phi21"] = np.arctan2(cartesian["C21a"], cartesian["C21b"])
    polar["C23"] = np.sqrt(cartesian["C23a"] ** 2 + cartesian["C23b"] ** 2)
    polar["phi23"] = -np.arctan2(cartesian["C23a"], cartesian["C23b"]) / 3.0
    polar["C30"] = cartesian["C30"]
    polar["C32"] = -np.sqrt(cartesian["C32a"] ** 2 + cartesian["C32b"] ** 2)
    polar["phi32"] = -np.arctan2(cartesian["C32b"], cartesian["C32a"]) / 2.0
    polar["C34"] = np.sqrt(cartesian["C34a"] ** 2 + cartesian["C34b"] ** 2)
    polar["phi34"] = np.arctan2(cartesian["C34b"], cartesian["C34a"]) / 4

    return polar
