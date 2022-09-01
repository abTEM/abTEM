"""Module to describe the contrast transfer function."""
import copy
import itertools
from abc import abstractmethod
from collections import defaultdict
from functools import partial, reduce
from typing import Mapping, Union, TYPE_CHECKING, Dict, List

import numpy as np
from matplotlib.axes import Axes

from abtem import stack
from abtem.core.axes import ParameterAxis, OrdinalAxis, AxisMetadata
from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.distributions import Distribution
from abtem.core.energy import Accelerator, HasAcceleratorMixin, energy2wavelength
from abtem.core.ensemble import Ensemble, EmptyEnsemble
from abtem.core.fft import ifft2
from abtem.core.grid import Grid, polar_spatial_frequencies
from abtem.core.utils import (
    expand_dims_to_match,
    CopyMixin,
    EqualityMixin,
    dictionary_property,
    delegate_property,
)
from abtem.measure.measure import FourierSpaceLineProfiles, DiffractionPatterns, Images

if TYPE_CHECKING:
    from abtem.waves.waves import Waves, WavesLikeMixin

#: Symbols for the polar representation of all optical aberrations up to the fifth order.
polar_symbols = (
    "C10",
    "C12",
    "phi12",
    "C21",
    "phi21",
    "C23",
    "phi23",
    "C30",
    "C32",
    "phi32",
    "C34",
    "phi34",
    "C41",
    "phi41",
    "C43",
    "phi43",
    "C45",
    "phi45",
    "C50",
    "C52",
    "phi52",
    "C54",
    "phi54",
    "C56",
    "phi56",
)

#: Aliases for the most commonly used optical aberrations.
polar_aliases = {
    "defocus": "C10",
    "astigmatism": "C12",
    "astigmatism_angle": "phi12",
    "coma": "C21",
    "coma_angle": "phi21",
    "Cs": "C30",
    "C5": "C50",
}


class WaveTransform(Ensemble, EqualityMixin, CopyMixin):
    @property
    def metadata(self):
        return {}

    def __add__(self, other: "WaveTransform") -> "CompositeWaveTransform":
        wave_transforms = []

        for wave_transform in (self, other):

            if hasattr(wave_transform, "wave_transforms"):
                wave_transforms += wave_transform.wave_transforms
            else:
                wave_transforms += [wave_transform]

        return CompositeWaveTransform(wave_transforms)

    @abstractmethod
    def apply(self, waves: "Waves") -> "Waves":
        pass


class WaveRenormalization(EmptyEnsemble, WaveTransform):
    def apply(self, waves):
        return waves.renormalize()


class CompositeWaveTransform(WaveTransform):
    def __init__(self, wave_transforms: List[WaveTransform] = None):

        if wave_transforms is None:
            wave_transforms = []

        self._wave_transforms = wave_transforms
        super().__init__()

    def insert_transform(self, transform, index):
        self._wave_transforms.insert(transform, index)

    def __len__(self):
        return len(self.wave_transforms)

    def __iter__(self):
        return iter(self.wave_transforms)

    @property
    def metadata(self):
        metadata = [transform.metadata for transform in self.wave_transforms]
        return reduce(lambda a, b: {**a, **b}, metadata)

    @property
    def wave_transforms(self):
        return self._wave_transforms

    @property
    def ensemble_axes_metadata(self):
        ensemble_axes_metadata = [
            wave_transform.ensemble_axes_metadata
            for wave_transform in self.wave_transforms
        ]
        return list(itertools.chain(*ensemble_axes_metadata))

    @property
    def default_ensemble_chunks(self):
        default_ensemble_chunks = [
            wave_transform.default_ensemble_chunks
            for wave_transform in self.wave_transforms
        ]
        return tuple(itertools.chain(*default_ensemble_chunks))

    @property
    def ensemble_shape(self):
        ensemble_shape = [
            wave_transform.ensemble_shape for wave_transform in self.wave_transforms
        ]
        return tuple(itertools.chain(*ensemble_shape))

    def apply(self, waves: "WavesLikeMixin"):
        waves.grid.check_is_defined()

        for wave_transform in reversed(self.wave_transforms):
            waves = wave_transform.apply(waves)

        return waves

    def partition_args(self, chunks=None, lazy: bool = True):
        if chunks is None:
            chunks = self.default_ensemble_chunks

        chunks = self.validate_chunks(chunks)

        blocks = ()
        start = 0
        for wave_transform in self.wave_transforms:
            stop = start + wave_transform.ensemble_dims
            blocks += wave_transform.partition_args(chunks[start:stop], lazy=lazy)
            start = stop

        return blocks

    @staticmethod
    def ctf(*args, partials):
        wave_transfer_functions = []
        for p in partials:
            wave_transfer_functions += [p[0](*[args[i] for i in p[1]])]

        return CompositeWaveTransform(wave_transfer_functions)

    def from_partitioned_args(self):
        partials = ()
        i = 0
        for wave_transform in self.wave_transforms:
            arg_indices = tuple(range(i, i + len(wave_transform.ensemble_shape)))
            partials += ((wave_transform.from_partitioned_args(), arg_indices),)
            i += len(arg_indices)

        return partial(self.ctf, partials=partials)


def unpack_distributions(*args: Union[float, Distribution], shape: tuple, xp=np):
    if len(args) == 0:
        return (), 1.0

    num_new_axes = sum(len(arg.shape) for arg in args if hasattr(arg, "shape"))

    unpacked = ()
    weights = 1.0
    i = 0
    for arg in args:
        if not isinstance(arg, Distribution):
            unpacked += (arg,)
            continue

        axis = list(range(num_new_axes))
        del axis[i]
        i += 1

        axis = tuple(axis) + tuple(range(num_new_axes, num_new_axes + len(shape)))
        values = xp.asarray(np.expand_dims(arg.values, axis=axis), dtype=xp.float32)
        unpacked += (values,)

        new_weights = xp.asarray(
            np.expand_dims(arg.weights, axis=axis), dtype=xp.float32
        )
        weights = new_weights if weights is None else weights * new_weights

    return unpacked, weights


class EnsembleFromDistributionsMixin:
    _distributions: tuple

    @property
    def _num_ensemble_axes(self):
        return sum(
            len(distribution.shape) for distribution in self._distribution_properties
        )

    @property
    def _distribution_properties(self):
        ensemble_parameters = {}
        for parameter in self._distributions:
            value = getattr(self, parameter)
            if hasattr(value, "values"):
                ensemble_parameters[parameter] = value
        return ensemble_parameters

    @property
    def ensemble_shape(self):
        return tuple(
            map(
                sum,
                tuple(
                    distribution.shape
                    for distribution in self._distribution_properties.values()
                ),
            )
        )

    def partition_args(self, chunks: int = 1, lazy: bool = True):
        distributions = self._distribution_properties
        chunks = self.validate_chunks(chunks)
        blocks = ()
        for distribution, n in zip(distributions.values(), chunks):
            blocks += (distribution.divide(n, lazy=lazy),)
        return blocks

    @classmethod
    def _partial_wave_transform(cls, *args, keys, **kwargs):
        assert len(args) == len(keys)
        kwargs.update({key: arg for key, arg in zip(keys, args)})
        transform = cls(**kwargs)  # noqa
        return transform

    def from_partitioned_args(self):
        kwargs = self.copy_kwargs()
        keys = tuple(self._distribution_properties.keys())
        return partial(self._partial_wave_transform, keys=keys, **kwargs)

    @property
    def default_ensemble_chunks(self):
        return ("auto",) * len(self.ensemble_shape)


class AbstractTransfer(WaveTransform, HasAcceleratorMixin):
    def __init__(self, energy):
        self._accelerator = Accelerator(energy=energy)

    @abstractmethod
    def evaluate_with_alpha_and_phi(self, alpha, phi):
        pass

    @property
    def ensemble_axes_metadata(self):
        return []

    def evaluate(self, waves: "Waves") -> np.ndarray:
        self.accelerator.match(waves)
        waves.grid.check_is_defined()
        alpha, phi = waves._angular_grid()
        return self.evaluate_with_alpha_and_phi(alpha, phi)

    def apply(self, waves: "Waves") -> "Waves":
        axes_metadata = self.ensemble_axes_metadata
        array = self.evaluate(waves)
        return waves.convolve(array, axes_metadata)


class AbstractAperture(AbstractTransfer):
    def __init__(self, semiangle_cutoff: float, energy: float, *args, **kwargs):
        self._semiangle_cutoff = semiangle_cutoff
        super().__init__(energy=energy)

    @property
    def nyquist_sampling(self) -> float:
        if isinstance(self._semiangle_cutoff, Distribution):
            semiangle_cutoff = max(self._semiangle_cutoff.values)
        else:
            semiangle_cutoff = self._semiangle_cutoff

        return 1 / (4 * semiangle_cutoff / self.wavelength * 1e-3)


class Aperture(EnsembleFromDistributionsMixin, AbstractAperture):
    def __init__(
        self,
        semiangle_cutoff: Union[float, Distribution],
        energy: float = None,
        taper: float = 1.0,
    ):
        """
        A circular aperture cutting off the wave function at a specified angle, employed in both STEM and HRTEM.
        The hard cutoff may be softened by tapering it.

        Parameters
        ----------
        semiangle_cutoff : float
            The cutoff semiangle of the aperture [mrad].
        energy : float, optional
            Electron energy [eV].
        taper : float

        """

        self._taper = taper
        self._accelerator = Accelerator(energy=energy)
        self._distributions = ("semiangle_cutoff",)

        if isinstance(semiangle_cutoff, Distribution):
            semiangle_cutoff = np.max(semiangle_cutoff.values)

        super().__init__(
            energy=energy,
            semiangle_cutoff=semiangle_cutoff,
        )

    @property
    def semiangle_cutoff(self) -> Union[float, Distribution]:
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value: Union[float, Distribution]):
        self._semiangle_cutoff = value

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        if isinstance(self.semiangle_cutoff, Distribution):
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
    def metadata(self):
        metadata = {"semiangle_cutoff": self.semiangle_cutoff}
        return metadata

    @property
    def taper(self):
        return self._taper

    def evaluate_with_alpha_and_phi(
        self, alpha: Union[float, np.ndarray], phi
    ) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        unpacked, _ = unpack_distributions(
            self.semiangle_cutoff, shape=alpha.shape, xp=xp
        )

        (semiangle_cutoff,) = unpacked
        semiangle_cutoff = semiangle_cutoff / 1000

        alpha = xp.array(alpha)
        alpha = xp.expand_dims(alpha, axis=tuple(range(0, self._num_ensemble_axes)))

        if self.semiangle_cutoff == xp.inf:
            return xp.ones_like(alpha)

        if self.taper > 0.0:
            taper = self.taper / 1000.0
            array = 0.5 * (
                1 + xp.cos(np.pi * (alpha - semiangle_cutoff + taper) / taper)
            )
            array[alpha > semiangle_cutoff] = 0.0
            array = xp.where(
                alpha > semiangle_cutoff - taper,
                array,
                xp.ones_like(alpha, dtype=xp.float32),
            )
        else:
            array = xp.array(alpha < semiangle_cutoff).astype(xp.float32)

        return array


class TemporalEnvelope(EnsembleFromDistributionsMixin, AbstractTransfer):
    def __init__(
        self,
        focal_spread: Union[float, Distribution],
        energy: float = None,
    ):
        self._accelerator = Accelerator(energy=energy)
        self._focal_spread = focal_spread
        self._distributions = ("focal_spread",)
        super().__init__(energy=energy)

    @property
    def focal_spread(self):
        return self._focal_spread

    @focal_spread.setter
    def focal_spread(self, value):
        self._focal_spread = value

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        if isinstance(self.focal_spread, Distribution):
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

    def evaluate_with_alpha_and_phi(
        self, alpha: Union[float, np.ndarray], phi
    ) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        unpacked, _ = unpack_distributions(self.focal_spread, shape=alpha.shape, xp=xp)
        (focal_spread,) = unpacked

        alpha = xp.array(alpha)
        alpha = xp.expand_dims(alpha, axis=tuple(range(0, self._num_ensemble_axes)))

        array = xp.exp(
            -((0.5 * xp.pi / self.wavelength * focal_spread * alpha ** 2) ** 2)
        ).astype(xp.float32)

        return array


class SpatialEnvelope(EnsembleFromDistributionsMixin, AbstractTransfer):
    def __init__(
        self,
        angular_spread: Union[float, Distribution],
        energy: float = None,
        aberrations: "Aberrations" = None,
    ):

        self._angular_spread = angular_spread

        if aberrations is None or isinstance(aberrations, dict):
            aberrations = Aberrations(parameters=aberrations)

        super().__init__(energy=energy)

        self._aberrations = aberrations
        self._aberrations._accelerator = self._accelerator

        for symbol in polar_symbols:
            setattr(self.__class__, symbol, delegate_property("_aberrations", symbol))

        self._distributions = ("angular_spread",) + polar_symbols

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        axes_metadata = []
        if isinstance(self.angular_spread, Distribution):
            axes_metadata += [
                ParameterAxis(
                    label="angular spread",
                    values=tuple(self.angular_spread.values),
                    units="mrad",
                    _ensemble_mean=self.angular_spread.ensemble_mean,
                )
            ]

        return axes_metadata + self.aberrations.ensemble_axes_metadata

    @property
    def aberrations(self):
        return self._aberrations

    @property
    def angular_spread(self):
        return self._angular_spread

    @angular_spread.setter
    def angular_spread(self, value):
        self._angular_spread = value

    def evaluate_with_alpha_and_phi(
        self, alpha: Union[float, np.ndarray], phi
    ) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        args = (self.angular_spread,) + tuple(self.aberrations.parameters.values())

        unpacked, _ = unpack_distributions(*args, shape=alpha.shape, xp=xp)
        angular_spread = unpacked[0] / 1e3
        parameters = dict(zip(polar_symbols, unpacked[1:]))

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
                * alpha ** 2
                + (
                    parameters["C34"] * xp.cos(4.0 * (phi - parameters["phi34"]))
                    + parameters["C32"] * xp.cos(2.0 * (phi - parameters["phi32"]))
                    + parameters["C30"]
                )
                * alpha ** 3
                + (
                    parameters["C45"] * xp.cos(5.0 * (phi - parameters["phi45"]))
                    + parameters["C43"] * xp.cos(3.0 * (phi - parameters["phi43"]))
                    + parameters["C41"] * xp.cos(1.0 * (phi - parameters["phi41"]))
                )
                * alpha ** 4
                + (
                    parameters["C56"] * xp.cos(6.0 * (phi - parameters["phi56"]))
                    + parameters["C54"] * xp.cos(4.0 * (phi - parameters["phi54"]))
                    + parameters["C52"] * xp.cos(2.0 * (phi - parameters["phi52"]))
                    + parameters["C50"]
                )
                * alpha ** 5
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
                * alpha ** 2
                + 1
                / 4.0
                * (
                    4.0 * parameters["C34"] * xp.sin(4.0 * (phi - parameters["phi34"]))
                    + 2.0
                    * parameters["C32"]
                    * xp.sin(2.0 * (phi - parameters["phi32"]))
                )
                * alpha ** 3
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
                * alpha ** 4
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
                * alpha ** 5
            )
        )

        array = xp.exp(
            -xp.sign(angular_spread)
            * (angular_spread / 2) ** 2
            * (dchi_dk ** 2 + dchi_dphi ** 2)
        )

        return array


class Aberrations(EnsembleFromDistributionsMixin, AbstractTransfer):
    def __init__(
        self,
        energy: float = None,
        parameters: Dict[str, Union[float, Distribution]] = None,
        **kwargs
    ):

        for key in kwargs.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()):
                raise ValueError("{} not a recognized parameter".format(key))

        parameters = {} if parameters is None else parameters

        parameters.update(kwargs)

        self._parameters = dict(zip(polar_symbols, [0.0] * len(polar_symbols)))
        self.set_parameters(parameters)

        for symbol in polar_symbols:
            setattr(self.__class__, symbol, dictionary_property("_parameters", symbol))

        for alias, symbol in polar_aliases.items():
            if alias != "defocus":
                setattr(
                    self.__class__, alias, dictionary_property("_parameters", symbol)
                )

        self._distributions = polar_symbols

        super().__init__(energy=energy)

    @property
    def parameters(self) -> Dict[str, Union[float, Distribution]]:
        """The parameters."""
        return self._parameters

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        axes_metadata = []
        for parameter_name, value in self._parameters.items():
            if isinstance(value, Distribution):
                axes_metadata += [
                    ParameterAxis(
                        label=parameter_name,
                        values=tuple(value.values),
                        units="Å",
                        _ensemble_mean=value.ensemble_mean,
                    )
                ]
        return axes_metadata

    @property
    def defocus(self) -> Union[float, Distribution]:
        """The defocus [Å]."""
        return -self._parameters["C10"]

    @defocus.setter
    def defocus(self, value: Union[float, Distribution]):
        self.C10 = -value

    def evaluate_with_alpha_and_phi(
        self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        parameters, weights = unpack_distributions(
            *tuple(self.parameters.values()), shape=alpha.shape, xp=xp
        )

        parameters = dict(zip(polar_symbols, parameters))

        axis = tuple(range(0, len(self.ensemble_shape)))
        alpha = xp.array(alpha)
        alpha = xp.expand_dims(alpha, axis=axis)
        phi = xp.expand_dims(phi, axis=axis)
        alpha2 = alpha ** 2

        array = xp.zeros(alpha.shape, dtype=np.float32)

        if any(np.any(parameters[symbol] != 0.0) for symbol in ("C10", "C12", "phi12")):
            array = array + (
                1
                / 2
                * alpha2
                * (
                    parameters["C10"]
                    + parameters["C12"] * xp.cos(2 * (phi - parameters["phi12"]))
                )
            )

        if any(
            np.any(parameters[symbol] != 0.0)
            for symbol in ("C21", "phi21", "C23", "phi23")
        ):
            array = array + (
                1
                / 3
                * alpha2
                * alpha
                * (
                    parameters["C21"] * xp.cos(phi - parameters["phi21"])
                    + parameters["C23"] * xp.cos(3 * (phi - parameters["phi23"]))
                )
            )

        if any(
            np.any(parameters[symbol] != 0.0)
            for symbol in ("C30", "C32", "phi32", "C34", "phi34")
        ):
            array = array + (
                1
                / 4
                * alpha2 ** 2
                * (
                    parameters["C30"]
                    + parameters["C32"] * xp.cos(2 * (phi - parameters["phi32"]))
                    + parameters["C34"] * xp.cos(4 * (phi - parameters["phi34"]))
                )
            )

        if any(
            np.any(parameters[symbol] != 0.0)
            for symbol in ("C41", "phi41", "C43", "phi43", "C45", "phi41")
        ):
            array = array + (
                1
                / 5
                * alpha2 ** 2
                * alpha
                * (
                    parameters["C41"] * xp.cos((phi - parameters["phi41"]))
                    + parameters["C43"] * xp.cos(3 * (phi - parameters["phi43"]))
                    + parameters["C45"] * xp.cos(5 * (phi - parameters["phi45"]))
                )
            )

        if any(
            np.any(parameters[symbol] != 0.0)
            for symbol in ("C50", "C52", "phi52", "C54", "phi54", "C56", "phi56")
        ):
            array = array + (
                1
                / 6
                * alpha2 ** 3
                * (
                    parameters["C50"]
                    + parameters["C52"] * xp.cos(2 * (phi - parameters["phi52"]))
                    + parameters["C54"] * xp.cos(4 * (phi - parameters["phi54"]))
                    + parameters["C56"] * xp.cos(6 * (phi - parameters["phi56"]))
                )
            )

        array = np.float32(2 * xp.pi / self.wavelength) * array

        array = complex_exponential(-array)

        if weights is not None:
            array = xp.asarray(weights) * array

        return array

    def set_parameters(self, parameters: Dict[str, Union[float, Distribution]]):
        """
        Set the phase of the phase aberration.

        Parameters
        ----------
        parameters: dict
            Mapping from aberration symbols to their corresponding values.
        """

        for symbol, value in parameters.items():
            if symbol in self._parameters.keys():
                self._parameters[symbol] = value

            elif symbol == "defocus":
                self._parameters[polar_aliases[symbol]] = -value

            elif symbol in polar_aliases.keys():
                self._parameters[polar_aliases[symbol]] = value

            else:
                raise ValueError("{} not a recognized parameter".format(symbol))


class CTF(EnsembleFromDistributionsMixin, AbstractAperture):
    def __init__(
        self,
        energy: float = None,
        semiangle_cutoff: float = np.inf,
        taper: float = 0.0,
        focal_spread: float = 0.0,
        angular_spread: float = 0.0,
        aberrations: Union[dict, Aberrations] = None,
        **kwargs
    ):

        """
        Contrast transfer function object

        The Contrast Transfer Function (CTF) describes the aberrations of the objective lens in HRTEM and specifies how the
        condenser system shapes the probe in STEM.

        abTEM implements phase aberrations up to 5th order using polar coefficients. See Eq. 2.22 in the reference [1]_.
        Cartesian coefficients can be converted to polar using the utility function abtem.transfer.cartesian2polar.

        Partial coherence is included as an envelope in the quasi-coherent approximation. See Chapter 3.2 in reference [1]_.

        For a more detailed discussion with examples, see our `walkthrough
        <https://abtem.readthedocs.io/en/latest/walkthrough/05_contrast_transfer_function.html>`_.

        Parameters
        ----------
        energy: float
            The electron energy of the wave functions this contrast transfer function will be applied to [eV].
        semiangle_cutoff: float
            The semiangle cutoff describes the sharp Fourier space cutoff due to the objective aperture [mrad].
        taper: float
            Tapers the cutoff edge over the given angular range [mrad].
        focal_spread: float
            The 1 / e width of the focal spread due to chromatic aberration and lens current instability [Å].
        angular_spread: float
            The 1 / e width of the angular deviations due to source size [Å].
        aberrations: dict
            Mapping from aberration symbols to their corresponding values. All aberration magnitudes should be given in
            Å and angles should be given in radians.
        kwargs:
            Provide the aberration coefficients as keyword arguments.

        References
        ----------
        .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy (2nd ed.). Springer.

        """

        if aberrations is None:
            aberrations = {}
        elif not hasattr(aberrations, "update"):
            aberrations = copy.deepcopy(aberrations.parameters)

        aberrations.update(kwargs)

        self._aberrations = Aberrations(energy=energy, parameters=aberrations)
        self._aperture = Aperture(
            energy=energy, semiangle_cutoff=semiangle_cutoff, taper=taper
        )
        self._spatial_envelope = SpatialEnvelope(
            angular_spread=angular_spread, aberrations=self._aberrations
        )
        self._temporal_envelope = TemporalEnvelope(focal_spread=focal_spread)

        super().__init__(energy=energy, semiangle_cutoff=semiangle_cutoff)

        self._aberrations._accelerator = self._accelerator
        self._aperture._accelerator = self._accelerator
        self._spatial_envelope._accelerator = self._accelerator
        self._temporal_envelope._accelerator = self._accelerator

        self._distributions = (
            "semiangle_cutoff",
            "focal_spread",
            "angular_spread",
        ) + polar_symbols

        setattr(
            self.__class__,
            "semiangle_cutoff",
            delegate_property("_aperture", "semiangle_cutoff"),
        )
        setattr(
            self.__class__,
            "taper",
            delegate_property("_aperture", "taper"),
        )
        setattr(
            self.__class__,
            "focal_spread",
            delegate_property("_temporal_envelope", "focal_spread"),
        )
        setattr(
            self.__class__,
            "angular_spread",
            delegate_property("_spatial_envelope", "angular_spread"),
        )

        for symbol in polar_symbols:
            setattr(self.__class__, symbol, delegate_property("_aberrations", symbol))

        for alias in polar_aliases.keys():
            setattr(self.__class__, alias, delegate_property("_aberrations", alias))

    @property
    def scherzer_defocus(self):
        self.accelerator.check_is_defined()

        if self.aberrations.Cs == 0.0:  # noqa
            raise ValueError()

        return scherzer_defocus(self.aberrations.Cs, self.energy)  # noqa

    @property
    def crossover_angle(self):
        return 1e3 * energy2wavelength(self.energy) / self.point_resolution

    @property
    def point_resolution(self):
        return point_resolution(self.aberrations.Cs, self.energy)  # noqa

    @property
    def aberrations(self):
        return self._aberrations

    @property
    def aperture(self):
        return self._aperture

    @property
    def spatial_envelope(self):
        return self._spatial_envelope

    @property
    def temporal_envelope(self):
        return self._temporal_envelope

    @property
    def ensemble_axes_metadata(self):
        return (
            self.aperture.ensemble_axes_metadata
            + self.spatial_envelope.ensemble_axes_metadata
            + self.temporal_envelope.ensemble_axes_metadata
        )

    # @property
    # def defocus(self) -> float:
    #     """The defocus [Å]."""
    #     return self.aberrations.defocus
    #
    # @defocus.setter
    # def defocus(self, value: float):
    #     self.aberrations.defocus = value
    #
    # @property
    # def taper(self) -> float:
    #     return self.aperture.taper
    #
    # @property
    # def semiangle_cutoff(self) -> float:
    #     """The semi-angle cutoff [mrad]."""
    #     return self.aperture.semiangle_cutoff
    #
    # @semiangle_cutoff.setter
    # def semiangle_cutoff(self, value: float):
    #     self.aperture.semiangle_cutoff = value
    #
    # @property
    # def focal_spread(self) -> float:
    #     """The focal spread [Å]."""
    #     return self.temporal_envelope.focal_spread
    #
    # @focal_spread.setter
    # def focal_spread(self, value: float):
    #     """The angular spread [mrad]."""
    #     self.temporal_envelope.focal_spread = value
    #
    # @property
    # def angular_spread(self) -> float:
    #     return self.spatial_envelope.angular_spread
    #
    # @angular_spread.setter
    # def angular_spread(self, value: float):
    #     self.spatial_envelope.angular_spread = value

    def evaluate_with_alpha_and_phi(self, alpha, phi):
        array = self.aberrations.evaluate_with_alpha_and_phi(alpha, phi)

        if self.spatial_envelope.angular_spread != 0.0:
            new_aberrations_dims = tuple(range(self.aberrations.ensemble_dims))
            old_match_dims = new_aberrations_dims + (-2, -1)

            added_dims = int(hasattr(self.spatial_envelope.angular_spread, "values"))
            new_match_dims = tuple(
                range(self.spatial_envelope.ensemble_dims - added_dims)
            ) + (-2, -1)

            new_array = self.spatial_envelope.evaluate_with_alpha_and_phi(alpha, phi)
            array, new_array = expand_dims_to_match(
                array, new_array, match_dims=[old_match_dims, new_match_dims]
            )
            array = array * new_array

        if self.temporal_envelope.focal_spread != 0.0:
            new_array = self.temporal_envelope.evaluate_with_alpha_and_phi(alpha, phi)
            array, new_array = expand_dims_to_match(
                array, new_array, match_dims=[(-2, -1), (-2, -1)]
            )
            array = array * new_array

        if self.aperture.semiangle_cutoff != np.inf:
            new_array = self.aperture.evaluate_with_alpha_and_phi(alpha, phi)
            array, new_array = expand_dims_to_match(
                array, new_array, match_dims=[(-2, -1), (-2, -1)]
            )
            array = array * new_array

        return array

    def image(self, gpts, max_angle):

        angular_sampling = 2 * max_angle / gpts[0], 2 * max_angle / gpts[1]

        fourier_space_sampling = (
            angular_sampling[0] / (self.wavelength * 1e3),
            angular_sampling[1] / (self.wavelength * 1e3),
        )

        sampling = 1 / (fourier_space_sampling[0] * gpts[0]), 1 / (
            fourier_space_sampling[1] * gpts[1]
        )

        alpha, phi = self._polar_spatial_frequencies_from_grid(
            gpts=gpts, sampling=sampling, wavelength=self.wavelength, xp=np
        )

        array = np.fft.fftshift(self.evaluate_with_alpha_and_phi(alpha, phi))

        # array = np.fft.fftshift(self.evaluate(waves))
        return DiffractionPatterns(
            array, sampling=fourier_space_sampling, metadata={"energy": self.energy}
        )

    # def point_spread_function(self, waves):
    #     alpha, phi = self._polar_spatial_frequencies_from_waves(waves)
    #     xp = get_array_module(waves.device)
    #     array = xp.fft.fftshift(ifft2(self.evaluate_with_alpha_and_phi(alpha, phi)))
    #     return Images(array, sampling=waves.sampling, metadata={'energy': self.energy})

    def profiles(self, max_angle: float = None, phi: float = 0.0):
        if max_angle is None:
            if self.aperture.semiangle_cutoff == np.inf:
                max_angle = 50
            else:
                max_angle = self.aperture.semiangle_cutoff * 1.6

        sampling = max_angle / 1000.0 / 1000.0
        alpha = np.arange(0, max_angle / 1000.0, sampling)

        aberrations = self.aberrations.evaluate_with_alpha_and_phi(alpha, phi)
        spatial_envelope = self.spatial_envelope.evaluate_with_alpha_and_phi(alpha, phi)
        temporal_envelope = self.temporal_envelope.evaluate_with_alpha_and_phi(
            alpha, phi
        )
        aperture = self.aperture.evaluate_with_alpha_and_phi(alpha, phi)
        envelope = aperture * temporal_envelope * spatial_envelope

        sampling = alpha[1] / energy2wavelength(self.energy)

        axis_metadata = ["ctf"]
        metadata = {"energy": self.energy}
        profiles = [
            FourierSpaceLineProfiles(
                -aberrations.imag * envelope, sampling=sampling, metadata=metadata
            )
        ]

        if self.aperture.semiangle_cutoff != np.inf:
            profiles += [
                FourierSpaceLineProfiles(aperture, sampling=sampling, metadata=metadata)
            ]
            axis_metadata += ["aperture"]

        if (
            self.temporal_envelope.focal_spread > 0.0
            and self.spatial_envelope.angular_spread > 0.0
        ):
            profiles += [
                FourierSpaceLineProfiles(envelope, sampling=sampling, metadata=metadata)
            ]
            axis_metadata += ["envelope"]

        if self.temporal_envelope.focal_spread > 0.0:
            profiles += [
                FourierSpaceLineProfiles(
                    temporal_envelope, sampling=sampling, metadata=metadata
                )
            ]
            axis_metadata += ["temporal"]

        if self.spatial_envelope.angular_spread > 0.0:
            profiles += [
                FourierSpaceLineProfiles(
                    spatial_envelope, sampling=sampling, metadata=metadata
                )
            ]
            axis_metadata += ["spatial"]

        return stack(profiles, axis_metadata=OrdinalAxis(values=tuple(axis_metadata)))


def nyquist_sampling(cutoff, wavelength):
    return 1 / (4 * cutoff / wavelength * 1e-3)


def scherzer_defocus(Cs, energy):
    """
    Calculate the Scherzer defocus.

    Parameters
    ----------
    Cs: float
        Spherical aberration [Å].
    energy: float
        Electron energy [eV].

    Returns
    -------
    float
        The Scherzer defocus.
    """

    return np.sign(Cs) * np.sqrt(3 / 2 * np.abs(Cs) * energy2wavelength(energy))


def point_resolution(Cs: float, energy: float):
    """
    Calculate the point resolution.

    Parameters
    ----------
    Cs: float
        Spherical aberration [Å].
    energy: float
        Electron energy [eV].

    Returns
    -------
    float
        The point resolution.
    """

    return (energy2wavelength(energy) ** 3 * np.abs(Cs) / 6) ** (1 / 4)


def polar2cartesian(polar):
    """
    Convert between polar and Cartesian aberration coefficients.

    Parameters
    ----------
    polar: dict
        Mapping from polar aberration symbols to their corresponding values.

    Returns
    -------
    dict
        Mapping from cartesian aberration symbols to their corresponding values.
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
        * (1 + K ** 2) ** 2
        / (K ** 3 - K)
        * polar["C34"]
        * np.cos(4 * np.arctan(1 / K) - 4 * polar["phi34"])
    )

    return cartesian


def cartesian2polar(cartesian):
    """
    Convert between Cartesian and polar aberration coefficients.

    Parameters
    ----------
    cartesian: dict
        Mapping from Cartesian aberration symbols to their corresponding values.

    Returns
    -------
    dict
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
