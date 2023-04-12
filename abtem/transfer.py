"""Module to describe the contrast transfer function (CTF) and the related apertures."""
import copy
import re
from collections import defaultdict
from typing import Dict, Tuple, TYPE_CHECKING, Union, List

import dask
import dask.array as da
import numpy as np

from abtem import stack
from abtem.core.axes import AxisMetadata, ParameterAxis
from abtem.core.axes import OrdinalAxis
from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.energy import Accelerator, energy2wavelength
from abtem.core.fft import fft_crop
from abtem.core.grid import HasGridMixin, GridUndefinedError
from abtem.core.transform import FourierSpaceConvolution
from abtem.core.utils import expand_dims_to_match
from abtem.distributions import (
    _EnsembleFromDistributionsMixin,
    BaseDistribution,
    _unpack_distributions,
    _validate_distribution,
)
from abtem.measurements import ReciprocalSpaceLineProfiles

if TYPE_CHECKING:
    from abtem.waves import Waves


class BaseAperture(FourierSpaceConvolution, HasGridMixin):
    """Base class for apertures. Documented in the subclasses."""

    def __init__(
        self,
        semiangle_cutoff: float = None,
        energy: float = None,
        extent: Union[float, Tuple[float, float]] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        *args,
        **kwargs,
    ):
        self._semiangle_cutoff = semiangle_cutoff
        super().__init__(
            energy=energy, extent=extent, gpts=gpts, sampling=sampling, **kwargs
        )

    @property
    def metadata(self):
        metadata = {}
        if not isinstance(self._semiangle_cutoff, BaseDistribution):
            metadata["semiangle_cutoff"] = self._semiangle_cutoff
        return metadata

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return []

    @property
    def _max_semiangle_cutoff(self):
        if isinstance(self._semiangle_cutoff, BaseDistribution):
            return max(self._semiangle_cutoff.values)
        else:
            return self._semiangle_cutoff

    @property
    def nyquist_sampling(self) -> float:
        return 1 / (4 * self._max_semiangle_cutoff / self.wavelength * 1e-3)

    @property
    def semiangle_cutoff(self):
        return self._semiangle_cutoff

    def _cropped_aperture(self):
        if self._max_semiangle_cutoff == np.inf:
            return self

        gpts = (
            int(2 * np.ceil(self._max_semiangle_cutoff / self.angular_sampling[0])) + 3,
            int(2 * np.ceil(self._max_semiangle_cutoff / self.angular_sampling[1])) + 3,
        )

        cropped_aperture = self.copy()
        cropped_aperture.gpts = gpts
        return cropped_aperture

    def _evaluate_from_cropped(self, gpts):
        cropped = self._cropped_aperture()
        array = cropped._evaluate()
        array = fft_crop(array, gpts)
        return array

    def evaluate(self, waves: "Waves" = None, lazy: bool = False) -> np.ndarray:
        if waves is not None:
            self.accelerator.match(waves)
            self.grid.match(waves)

        if lazy:
            array = dask.delayed(self._evaluate_from_cropped)(gpts=self.gpts)
            array = da.from_delayed(
                array, dtype=np.complex64, shape=self.ensemble_shape + self.gpts
            )
            return array
        else:
            return self._evaluate_from_cropped(gpts=self.gpts)


def soft_aperture(alpha, phi, semiangle_cutoff, angular_sampling):
    xp = get_array_module(alpha)

    if np.isscalar(semiangle_cutoff):
        num_ensemble_axes = 0
    else:
        num_ensemble_axes = len(semiangle_cutoff.shape) - len(alpha.shape)

    angular_sampling = xp.array(angular_sampling, dtype=xp.float32) * 1e-3

    alpha = xp.expand_dims(alpha, axis=tuple(range(0, num_ensemble_axes)))
    phi = xp.expand_dims(phi, axis=tuple(range(0, num_ensemble_axes)))

    denominator = xp.sqrt(
        (xp.cos(phi) * angular_sampling[0]) ** 2
        + (xp.sin(phi) * angular_sampling[1]) ** 2
    )

    zeros = (slice(None),) * num_ensemble_axes + (0,) * (
        len(denominator.shape) - num_ensemble_axes
    )

    denominator[zeros] = 1.0

    array = xp.clip(
        (semiangle_cutoff - alpha) / denominator + 0.5, a_min=0.0, a_max=1.0
    )
    array[zeros] = 1.0

    return array


class Aperture(_EnsembleFromDistributionsMixin, BaseAperture):
    """
    A circular aperture cutting off the wave function at a specified angle, employed in both STEM and HRTEM.
    The abrupt cutoff may be softened by tapering it.

    Parameters
    ----------
    semiangle_cutoff : float or BaseDistribution
        The cutoff semiangle of the aperture [mrad]. Alternatively, a distribution of angles may be provided.
    soft : bool, optional
        If True, the edge of the aperture is softened (default is True).
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be ignored.
    """

    def __init__(
        self,
        semiangle_cutoff: Union[float, BaseDistribution],
        soft: bool = True,
        energy: float = None,
        extent: Union[float, Tuple[float, float]] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
    ):
        semiangle_cutoff = _validate_distribution(semiangle_cutoff)
        self._soft = soft

        super().__init__(
            distributions=("semiangle_cutoff",),
            energy=energy,
            semiangle_cutoff=semiangle_cutoff,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
        )

    @property
    def semiangle_cutoff(self) -> Union[float, BaseDistribution]:
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value: Union[float, BaseDistribution]):
        self._semiangle_cutoff = _validate_distribution(value)

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        if isinstance(self.semiangle_cutoff, BaseDistribution):
            return [
                ParameterAxis(
                    label="semiangle cutoff",
                    values=tuple(self.semiangle_cutoff.values),
                    units="mrad",
                    _ensemble_mean=self.semiangle_cutoff.ensemble_mean,
                )
            ]
        else:
            return []

    @property
    def soft(self):
        return self._soft

    def _evaluate_with_alpha_and_phi(
        self, alpha: Union[float, np.ndarray], phi
    ) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        unpacked, _ = _unpack_distributions(
            self.semiangle_cutoff, shape=alpha.shape, xp=xp
        )

        (semiangle_cutoff,) = unpacked
        semiangle_cutoff = semiangle_cutoff * 1e-3

        alpha = xp.expand_dims(alpha, axis=tuple(range(0, self._num_ensemble_axes)))

        if self.semiangle_cutoff == xp.inf:
            return xp.ones_like(alpha)

        if not self.soft or not self.grid.check_is_defined(raise_error=False):
            return xp.array(alpha <= semiangle_cutoff).astype(xp.float32)

        angular_sampling = xp.array(self.angular_sampling, dtype=xp.float32) * 1e-3

        phi = xp.expand_dims(phi, axis=tuple(range(0, self._num_ensemble_axes)))

        denominator = xp.sqrt(
            (xp.cos(phi) * angular_sampling[0]) ** 2
            + (xp.sin(phi) * angular_sampling[1]) ** 2
        )

        zeros = (slice(None),) * len(self.ensemble_shape) + (0,) * (
            len(denominator.shape) - len(self.ensemble_shape)
        )

        denominator[zeros] = 1.0

        array = xp.clip(
            (semiangle_cutoff - alpha) / denominator + 0.5, a_min=0.0, a_max=1.0
        )
        array[zeros] = 1.0

        return array


class Bullseye(_EnsembleFromDistributionsMixin, BaseAperture):
    """
    Bullseye aperture.

    Parameters
    ----------
    num_spokes : int
        Number of spokes.
    spoke_width : float
        Width of spokes [degree].
    num_rings : int
        Number of rings.
    ring_width : float
        Width of rings [mrad].
    semiangle_cutoff : float
        The cutoff semiangle of the aperture [mrad].
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be ignored.
    """

    def __init__(
        self,
        num_spokes: int,
        spoke_width: float,
        num_rings: int,
        ring_width: float,
        semiangle_cutoff: float,
        energy: float = None,
        extent: Union[float, Tuple[float, float]] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
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
    def num_spokes(self) -> int:
        return self._spoke_num

    @property
    def spoke_width(self) -> float:
        return self._spoke_width

    @property
    def num_rings(self) -> int:
        return self._num_rings

    @property
    def ring_width(self) -> float:
        return self._ring_width

    @property
    def semiangle_cutoff(self) -> float:
        return self._semiangle_cutoff

    def _evaluate_with_alpha_and_phi(
        self, alpha: Union[float, np.ndarray], phi
    ) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)
        alpha = xp.array(alpha)

        semiangle_cutoff = self.semiangle_cutoff / 1e3

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


class Vortex(_EnsembleFromDistributionsMixin, BaseAperture):
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
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be ignored.
    """

    def __init__(
        self,
        quantum_number: int,
        semiangle_cutoff: float,
        energy: float = None,
        extent: Union[float, Tuple[float, float]] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
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
    def quantum_number(self):
        return self._quantum_number

    @property
    def semiangle_cutoff(self) -> float:
        return self._semiangle_cutoff

    def _evaluate_with_alpha_and_phi(
        self, alpha: Union[float, np.ndarray], phi
    ) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)
        alpha = xp.array(alpha)

        semiangle_cutoff = self.semiangle_cutoff / 1e3

        array = alpha < semiangle_cutoff
        array = array * np.exp(1j * phi * self.quantum_number)

        return array


class TemporalEnvelope(_EnsembleFromDistributionsMixin, FourierSpaceConvolution):
    """
    Envelope function for simulating partial temporal coherence in the quasi-coherent approximation.

    Parameters
    ----------
    focal_spread: float or BaseDistribution
        The standard deviation of the focal spread due to chromatic aberration and lens current instability [Å].
        Alternatively, a distribution of values may be provided.
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be ignored.
    """

    def __init__(
        self,
        focal_spread: Union[float, BaseDistribution],
        energy: float = None,
        extent: Union[float, Tuple[float, float]] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
    ):

        self._accelerator = Accelerator(energy=energy)
        self._focal_spread = _validate_distribution(focal_spread)
        super().__init__(
            distributions=("focal_spread",),
            energy=energy,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
        )

    @property
    def focal_spread(self):
        return self._focal_spread

    @focal_spread.setter
    def focal_spread(self, value):
        self._focal_spread = _validate_distribution(value)

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        if isinstance(self.focal_spread, BaseDistribution):
            return [
                ParameterAxis(
                    label="focal spread",
                    values=tuple(self.focal_spread.values),
                    units="mrad",
                    _ensemble_mean=self.focal_spread.ensemble_mean,
                )
            ]
        else:
            return []

    def _evaluate_with_alpha_and_phi(
        self, alpha: Union[float, np.ndarray], phi
    ) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        unpacked, _ = _unpack_distributions(self.focal_spread, shape=alpha.shape, xp=xp)
        (focal_spread,) = unpacked

        alpha = xp.array(alpha)
        alpha = xp.expand_dims(alpha, axis=tuple(range(0, self._num_ensemble_axes)))

        array = xp.exp(
            -((0.5 * xp.pi / self.wavelength * focal_spread * alpha**2) ** 2)
        ).astype(xp.float32)

        return array


def _aberration_property(name, key):
    def getter(self):
        try:
            return getattr(self, name)[key]
        except KeyError:
            return 0.0

    def setter(self, value):
        value = _validate_distribution(value)
        getattr(self, name)[key] = value

    return property(getter, setter)


class _HasAberrations:
    _aberration_coefficients: dict
    energy: float

    C10: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C10"
    )
    C12: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C12"
    )
    phi12: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi12"
    )
    C21: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C21"
    )
    phi21: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi21"
    )
    C23: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C23"
    )
    phi23: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi23"
    )
    C30: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C30"
    )
    C32: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C32"
    )
    phi32: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi32"
    )
    C34: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C34"
    )
    phi34: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi34"
    )
    C41: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C41"
    )
    phi41: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi41"
    )
    C43: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C43"
    )
    phi43: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi43"
    )
    C45: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C45"
    )
    phi45: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi45"
    )
    C50: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C50"
    )
    C52: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C52"
    )
    phi52: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi52"
    )
    C54: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C54"
    )
    phi54: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi54"
    )
    C56: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C56"
    )
    phi56: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi56"
    )
    astigmatism: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C12"
    )
    astigmatism_angle: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi12"
    )
    coma: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C21"
    )
    coma_angle: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "phi21"
    )
    Cs: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C30"
    )
    C5: Union[float, BaseDistribution] = _aberration_property(
        "_aberration_coefficients", "C5"
    )

    @property
    def defocus(self) -> Union[float, BaseDistribution]:
        return -self.C10

    @defocus.setter
    def defocus(self, value: Union[float, BaseDistribution]):
        self.C10 = -value

    @classmethod
    def _coefficient_symbols(cls):
        return tuple(var for var in dir(cls) if re.fullmatch("C[0-9][0-9]", var))

    @classmethod
    def _angular_symbols(cls):
        return tuple(
            var for var in dir(_HasAberrations) if re.fullmatch("phi[0-9][0-9]", var)
        )

    def _nonzero_coefficients(self, symbols):
        for symbol in symbols:
            if not np.isscalar(self._aberration_coefficients[symbol]):
                return True

            if not self._aberration_coefficients[symbol] == 0.0:
                return True

        return False

    @classmethod
    def _symbols(cls):
        return cls._coefficient_symbols() + cls._angular_symbols()

    @classmethod
    def _aliases(self):
        return {
            "defocus": "C10",
            "astigmatism": "C12",
            "astigmatism_angle": "phi12",
            "coma": "C21",
            "coma_angle": "phi21",
            "Cs": "C30",
            "C5": "C50",
        }

    def _default_aberration_coefficients(self):
        return {symbol: 0.0 for symbol in self._symbols()}

    def _check_is_valid_aberrations(self, aberrations):
        for key in aberrations.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()):
                raise ValueError("{} not a recognized phase aberration".format(key))

    @property
    def _phase_aberrations_ensemble_axes_metadata(self) -> List[AxisMetadata]:
        axes_metadata = []
        for parameter_name, value in self._aberration_coefficients.items():
            if isinstance(value, BaseDistribution):
                m = re.search(r"\d", parameter_name).start()
                _tex_label = f"${parameter_name[:m]}_{{{parameter_name[m:]}}}$"
                axes_metadata += [
                    ParameterAxis(
                        label=parameter_name,
                        values=tuple(value.values),
                        units="Å",
                        _ensemble_mean=value.ensemble_mean,
                        _tex_label=_tex_label
                    )
                ]
        return axes_metadata

    @property
    def aberration_coefficients(self) -> Dict[str, Union[float, BaseDistribution]]:
        """The parameters."""
        return copy.deepcopy(self._aberration_coefficients)

    @property
    def has_aberrations(self):
        if np.all(np.array(list(self._aberration_coefficients.values())) == 0.0):
            return False
        else:
            return True

    def set_aberrations(self, parameters: Dict[str, Union[float, BaseDistribution]]):
        """
        Set the phase of the phase aberration.

        Parameters
        ----------
        parameters : dict
            Mapping from aberration symbols to their corresponding values.
        """

        for symbol, value in parameters.items():
            value = _validate_distribution(value)

            if symbol in self._symbols():
                self._aberration_coefficients[symbol] = value

            elif symbol in self._aliases().keys():
                self._aberration_coefficients[self._aliases()[symbol]] = value

            else:
                raise ValueError("{} not a recognized parameter".format(symbol))

        for symbol, value in parameters.items():
            if symbol in ("defocus", "C10"):
                value = _validate_distribution(value)

                if isinstance(value, str) and value.lower() == "scherzer":
                    if self.energy is None:
                        raise RuntimeError(
                            "energy undefined, Scherzer defocus cannot be evaluated"
                        )

                    value = scherzer_defocus(
                        self._aberration_coefficients["C30"], self.energy
                    )

                if symbol == "defocus":
                    value = -value

                self._aberration_coefficients["C10"] = value


polar_symbols = _HasAberrations._symbols()

polar_aliases = _HasAberrations._aliases()


class SpatialEnvelope(
    _HasAberrations, _EnsembleFromDistributionsMixin, FourierSpaceConvolution
):
    """
    Envelope function for simulating partial spatial coherence in the quasi-coherent approximation.

    Parameters
    ----------
    angular_spread: float or BaseDistribution
        The standard deviation of the angular deviations due to source size [Å]. Alternatively, a distribution of angles
        may be provided.
    aberration_coefficients: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration magnitudes should be given in
        [Å] and angles should be given in [radian].
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be ignored.
    kwargs : dict, optional
        Optionally provide the aberration coefficients as keyword arguments.
    """

    def __init__(
        self,
        angular_spread: Union[float, BaseDistribution],
        aberration_coefficients: dict = None,
        energy: float = None,
        extent: Union[float, Tuple[float, float]] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        **kwargs,
    ):
        super().__init__(
            distributions=polar_symbols + ("angular_spread",),
            energy=energy,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
        )

        self._angular_spread = _validate_distribution(angular_spread)
        self._aberration_coefficients = self._default_aberration_coefficients()

        aberration_coefficients = (
            {} if aberration_coefficients is None else aberration_coefficients
        )
        aberration_coefficients = {**aberration_coefficients, **kwargs}
        self.set_aberrations(aberration_coefficients)

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        axes_metadata = self._phase_aberrations_ensemble_axes_metadata
        if isinstance(self.angular_spread, BaseDistribution):
            axes_metadata = axes_metadata + [
                ParameterAxis(
                    label="angular spread",
                    values=tuple(self.angular_spread.values),
                    units="mrad",
                    _ensemble_mean=self.angular_spread.ensemble_mean,
                )
            ]
        return axes_metadata

    @property
    def angular_spread(self):
        return self._angular_spread

    @angular_spread.setter
    def angular_spread(self, value):
        self._angular_spread = value

    def _evaluate_with_alpha_and_phi(
        self, alpha: Union[float, np.ndarray], phi
    ) -> Union[float, np.ndarray]:
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


class Aberrations(_EnsembleFromDistributionsMixin, BaseAperture, _HasAberrations):
    """
    Phase aberrations.

    Parameters
    ----------
    aberration_coefficients: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration magnitudes should be given in
        [Å] and angles should be given in [radian].
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be ignored.
    kwargs : dict, optional
        Optionally provide the aberration coefficients as keyword arguments.
    """

    def __init__(
        self,
        aberration_coefficients: Dict[str, Union[float, BaseDistribution]] = None,
        energy: float = None,
        extent: Union[float, Tuple[float, float]] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        semiangle_cutoff: float = np.inf,
        **kwargs,
    ):

        super().__init__(
            distributions=polar_symbols,
            energy=energy,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
            semiangle_cutoff=semiangle_cutoff,
        )

        aberration_coefficients = (
            {} if aberration_coefficients is None else aberration_coefficients
        )

        aberration_coefficients = {**aberration_coefficients, **kwargs}
        self._aberration_coefficients = self._default_aberration_coefficients()
        self.set_aberrations(aberration_coefficients)

    @property
    def ensemble_axes_metadata(self):
        return self._phase_aberrations_ensemble_axes_metadata

    @property
    def defocus(self) -> Union[float, BaseDistribution]:
        """The defocus [Å]."""
        return -self._aberration_coefficients["C10"]

    @defocus.setter
    def defocus(self, value: Union[float, BaseDistribution]):
        self.C10 = -value

    def _evaluate_with_alpha_and_phi(
        self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        if not self.has_aberrations:
            return xp.ones(self.ensemble_shape + alpha.shape, dtype=xp.complex64)

        parameters, weights = _unpack_distributions(
            *tuple(self.aberration_coefficients.values()), shape=alpha.shape, xp=xp
        )

        parameters = dict(zip(polar_symbols, parameters))

        axis = tuple(range(0, len(self.ensemble_shape)))
        alpha = xp.expand_dims(alpha, axis=axis)
        phi = xp.expand_dims(phi, axis=axis)

        array = xp.zeros(alpha.shape, dtype=np.float32)
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

        if self._nonzero_coefficients(("C41", "phi41", "C43", "phi43", "C45", "phi41")):
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

        array *= np.float32(2 * xp.pi / self.wavelength)
        array = complex_exponential(-array)

        if weights is not None and not np.all(weights == 1.0):
            array = xp.asarray(weights, dtype=xp.float32) * array

        return array

    def apply(self, waves, overwrite_x: bool = False):
        if self.has_aberrations:
            return super().apply(waves, overwrite_x=overwrite_x)
        else:
            return waves


class CTF(_HasAberrations, _EnsembleFromDistributionsMixin, BaseAperture):
    """
    The contrast transfer function (CTF) describes the aberrations of the objective lens in HRTEM and specifies how
    the condenser system shapes the probe in STEM.

    abTEM implements phase aberrations up to 5th order using polar coefficients.
    See Eq. 2.22 in the reference [1]_.

    Cartesian coefficients can be converted to polar using the utility function `abtem.transfer.cartesian2polar`.

    Partial coherence is included as envelopes in the quasi-coherent approximation.
    See Chapter 3.2 in reference [1]_.

    Parameters
    ----------
    semiangle_cutoff: float, optional
        The semiangle cutoff describes the sharp reciprocal-space cutoff due to the objective aperture [mrad]
        (default is no cutoff).
    soft : bool, optional
        If True, the edge of the aperture is softened (default is True).
    focal_spread: float, optional
        The standard deviation of the focal spread due to chromatic aberration and lens current instability [Å]
        (default is 0).
    angular_spread: float, optional
        The standard deviation of the angular deviations due to source size [Å] (default is 0).
    aberration_coefficients: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration magnitudes should be given in
        [Å] and angles should be given in [radian].
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1 / Å]. If 'gpts' is also given, will be ignored.
    kwargs : dict, optional
        Optionally provide the aberration coefficients as keyword arguments.

    References
    ----------
    .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy (2nd ed.). Springer.

    """

    def __init__(
        self,
        semiangle_cutoff: float = np.inf,
        soft: bool = True,
        focal_spread: float = 0.0,
        angular_spread: float = 0.0,
        aberration_coefficients: dict = None,
        energy: float = None,
        extent: Union[float, Tuple[float, float]] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        **kwargs,
    ):

        super().__init__(
            distributions=polar_symbols
            + (
                "angular_spread",
                "focal_spread",
                "semiangle_cutoff",
            ),
            energy=energy,
            semiangle_cutoff=semiangle_cutoff,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
        )

        self._aberration_coefficients = self._default_aberration_coefficients()

        aberration_coefficients = (
            {} if aberration_coefficients is None else aberration_coefficients
        )
        aberration_coefficients = {**aberration_coefficients, **kwargs}
        self.set_aberrations(aberration_coefficients)

        self._angular_spread = angular_spread
        self._focal_spread = focal_spread
        self._semiangle_cutoff = semiangle_cutoff
        self._soft = soft

    @property
    def aberration_coefficients(self):
        return self._aberration_coefficients

    @property
    def scherzer_defocus(self):
        self.accelerator.check_is_defined()

        if self.Cs == 0.0:
            raise ValueError()

        return scherzer_defocus(self.Cs, self.energy)

    @property
    def crossover_angle(self):
        return 1e3 * energy2wavelength(self.energy) / self.point_resolution

    @property
    def point_resolution(self):
        return point_resolution(self.Cs, self.energy)

    @property
    def _aberrations(self):
        return Aberrations(
            aberration_coefficients=self.aberration_coefficients,
            energy=self.energy,
            extent=self.extent,
            gpts=self.gpts,
        )

    @property
    def _aperture(self):
        return Aperture(
            semiangle_cutoff=self.semiangle_cutoff,
            soft=self._soft,
            energy=self.energy,
            extent=self.extent,
            gpts=self.gpts,
        )

    @property
    def _spatial_envelope(self):
        return SpatialEnvelope(
            aberration_coefficients=self.aberration_coefficients,
            angular_spread=self.angular_spread,
            energy=self.energy,
        )

    @property
    def _temporal_envelope(self):
        return TemporalEnvelope(
            focal_spread=self.focal_spread,
            energy=self.energy,
        )

    @property
    def ensemble_axes_metadata(self):
        return (
            self._spatial_envelope.ensemble_axes_metadata
            + self._temporal_envelope.ensemble_axes_metadata
            + self._aperture.ensemble_axes_metadata
        )

    @property
    def soft(self) -> float:
        return self._soft

    @property
    def semiangle_cutoff(self) -> float:
        """The semiangle cutoff [mrad]."""
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value: float):
        self._semiangle_cutoff = value

    @property
    def focal_spread(self) -> float:
        """The focal spread [Å]."""
        return self._focal_spread

    @focal_spread.setter
    def focal_spread(self, value: float):
        """The angular spread [mrad]."""
        self._focal_spread = value

    @property
    def angular_spread(self) -> float:
        return self._angular_spread

    @angular_spread.setter
    def angular_spread(self, value: float):
        self._angular_spread = value

    def _evaluate_with_alpha_and_phi(self, alpha, phi):
        array = self._aberrations._evaluate_with_alpha_and_phi(alpha, phi)

        if self._spatial_envelope.angular_spread != 0.0:
            new_aberrations_dims = tuple(range(len(self._aberrations.ensemble_shape)))
            old_match_dims = new_aberrations_dims + (-2, -1)

            added_dims = int(hasattr(self._spatial_envelope.angular_spread, "values"))
            new_match_dims = tuple(
                range(len(self._spatial_envelope.ensemble_shape) - added_dims)
            ) + (-2, -1)

            new_array = self._spatial_envelope._evaluate_with_alpha_and_phi(alpha, phi)
            array, new_array = expand_dims_to_match(
                array, new_array, match_dims=[old_match_dims, new_match_dims]
            )

            array = array * new_array

        if self._temporal_envelope.focal_spread != 0.0:
            new_array = self._temporal_envelope._evaluate_with_alpha_and_phi(alpha, phi)
            array, new_array = expand_dims_to_match(
                array, new_array, match_dims=[(-2, -1), (-2, -1)]
            )
            array = array * new_array

        if self._aperture.semiangle_cutoff != np.inf:
            new_array = self._aperture._evaluate_with_alpha_and_phi(alpha, phi)
            array, new_array = expand_dims_to_match(
                array, new_array, match_dims=[(-2, -1), (-2, -1)]
            )
            array = array * new_array

        return array

    def profiles(self, max_angle: float = None, phi: float = 0.0):
        if max_angle is None:
            if self.semiangle_cutoff == np.inf:
                max_angle = 50
            else:
                max_angle = self.semiangle_cutoff * 1.6

        sampling = max_angle / 1000.0 / 1000.0
        alpha = np.arange(0, max_angle / 1000.0, sampling)

        aberrations = self._aberrations._evaluate_with_alpha_and_phi(alpha, phi)
        spatial_envelope = self._spatial_envelope._evaluate_with_alpha_and_phi(
            alpha, phi
        )
        temporal_envelope = self._temporal_envelope._evaluate_with_alpha_and_phi(
            alpha, phi
        )
        aperture = self._aperture._evaluate_with_alpha_and_phi(alpha, phi)
        envelope = aperture * temporal_envelope * spatial_envelope

        sampling = alpha[1] / energy2wavelength(self.energy)

        axis_metadata = ["ctf"]
        metadata = {"energy": self.energy}
        profiles = [
            ReciprocalSpaceLineProfiles(
                -aberrations.imag * envelope,
                sampling=sampling,
                metadata=metadata,
                ensemble_axes_metadata=self._aberrations.ensemble_axes_metadata,
            )
        ]

        if self._aperture.semiangle_cutoff != np.inf:
            profiles += [
                ReciprocalSpaceLineProfiles(
                    aperture, sampling=sampling, metadata=metadata
                )
            ]
            axis_metadata += ["aperture"]

        if (
            self._temporal_envelope.focal_spread > 0.0
            and self._spatial_envelope.angular_spread > 0.0
        ):
            profiles += [
                ReciprocalSpaceLineProfiles(
                    envelope, sampling=sampling, metadata=metadata
                )
            ]
            axis_metadata += ["envelope"]

        if self._temporal_envelope.focal_spread > 0.0:
            profiles += [
                ReciprocalSpaceLineProfiles(
                    temporal_envelope, sampling=sampling, metadata=metadata
                )
            ]
            axis_metadata += ["temporal"]

        if self._spatial_envelope.angular_spread > 0.0:
            profiles += [
                ReciprocalSpaceLineProfiles(
                    spatial_envelope,
                    sampling=sampling,
                    metadata=metadata,
                    ensemble_axes_metadata=self._aberrations.ensemble_axes_metadata,
                )
            ]
            axis_metadata += ["spatial"]

        if len(profiles) > 1:
            return stack(
                profiles, axis_metadata=OrdinalAxis(values=tuple(axis_metadata))
            )
        else:
            return profiles[0]


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


def polar2cartesian(polar):
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
    K = np.sqrt(3 + np.sqrt(8.0))
    cartesian["C34b"] = (
        1
        / 4.0
        * (1 + K**2) ** 2
        / (K**3 - K)
        * polar["C34"]
        * np.cos(4 * np.arctan(1 / K) - 4 * polar["phi34"])
    )

    return cartesian


def cartesian2polar(cartesian):
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
