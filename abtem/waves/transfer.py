"""Module to describe the contrast transfer function."""
import copy
import itertools
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import Mapping, Union, TYPE_CHECKING, Dict, List

import numpy as np
from matplotlib.axes import Axes

from abtem import stack
from abtem.core.axes import ParameterSeriesAxis, OrdinalAxis
from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.distributions import Distribution
from abtem.core.energy import Accelerator, HasAcceleratorMixin, energy2wavelength
from abtem.core.ensemble import Ensemble, EmptyEnsemble
from abtem.core.fft import ifft2
from abtem.core.grid import Grid, polar_spatial_frequencies
from abtem.core.utils import expand_dims_to_match, CopyMixin, EqualityMixin
from abtem.measure.measure import FourierSpaceLineProfiles, DiffractionPatterns, Images

if TYPE_CHECKING:
    from abtem.waves.waves import Waves, WavesLikeMixin

#: Symbols for the polar representation of all optical aberrations up to the fifth order.
polar_symbols = ('C10', 'C12', 'phi12',
                 'C21', 'phi21', 'C23', 'phi23',
                 'C30', 'C32', 'phi32', 'C34', 'phi34',
                 'C41', 'phi41', 'C43', 'phi43', 'C45', 'phi45',
                 'C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')

#: Aliases for the most commonly used optical aberrations.
polar_aliases = {'defocus': 'C10', 'astigmatism': 'C12', 'astigmatism_angle': 'phi12',
                 'coma': 'C21', 'coma_angle': 'phi21',
                 'Cs': 'C30',
                 'C5': 'C50'}


class WaveTransform(Ensemble, EqualityMixin, CopyMixin):

    def __add__(self, other: 'WaveTransform') -> 'CompositeWaveTransform':
        wave_transforms = []

        for wave_transform in (self, other):

            if hasattr(wave_transform, 'wave_transforms'):
                wave_transforms += wave_transform.wave_transforms
            else:
                wave_transforms += [wave_transform]

        return CompositeWaveTransform(wave_transforms)

    @abstractmethod
    def apply(self, waves: 'Waves'):
        pass


class WaveRenormalization(EmptyEnsemble, WaveTransform):

    def apply(self, waves):
        return waves.renormalize()


class ArrayWaveTransform(WaveTransform):

    def _polar_spatial_frequencies_from_grid(self, gpts, sampling, wavelength, xp):
        grid = Grid(gpts=gpts, sampling=sampling)
        alpha, phi = polar_spatial_frequencies(grid.gpts, grid.sampling, xp=xp)
        alpha *= wavelength
        return alpha, phi

    def _polar_spatial_frequencies_from_waves(self, waves):
        xp = get_array_module(waves.device)
        return self._polar_spatial_frequencies_from_grid(waves.gpts, waves.sampling, waves.wavelength, xp)

    def evaluate_with_alpha_and_phi(self, alpha, phi):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, waves: 'WavesLikeMixin'):
        raise NotImplementedError

    def apply(self, waves: 'Waves', out_space: 'str' = 'in_space') -> 'Waves':
        if out_space == 'in_space':
            fourier_space_out = waves.fourier_space
        elif out_space in ('fourier_space', 'real_space'):
            fourier_space_out = out_space == 'fourier_space'
        else:
            raise ValueError

        xp = get_array_module(waves.device)

        kernel = self.evaluate(waves)

        waves = waves.ensure_fourier_space(in_place=False)

        waves_dims = tuple(range(len(kernel.shape) - 2))
        kernel_dims = tuple(range(len(kernel.shape) - 2, len(waves.array.shape) - 2 + len(kernel.shape) - 2))

        array = xp.expand_dims(waves.array, axis=waves_dims) * xp.expand_dims(kernel, axis=kernel_dims)

        if not fourier_space_out:
            array = ifft2(array, overwrite_x=False)

        d = waves.copy_kwargs(exclude=('array',))
        d['fourier_space'] = fourier_space_out
        d['array'] = array
        d['ensemble_axes_metadata'] = self.ensemble_axes_metadata + d['ensemble_axes_metadata']
        return waves.__class__(**d)


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
    def wave_transforms(self):
        return self._wave_transforms

    @property
    def ensemble_axes_metadata(self):
        ensemble_axes_metadata = [wave_transform.ensemble_axes_metadata for wave_transform in self.wave_transforms]
        return list(itertools.chain(*ensemble_axes_metadata))

    @property
    def default_ensemble_chunks(self):
        default_ensemble_chunks = [wave_transform.default_ensemble_chunks for wave_transform in self.wave_transforms]
        return tuple(itertools.chain(*default_ensemble_chunks))

    @property
    def ensemble_shape(self):
        ensemble_shape = [wave_transform.ensemble_shape for wave_transform in self.wave_transforms]
        return tuple(itertools.chain(*ensemble_shape))

    def apply(self, waves: 'WavesLikeMixin'):
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


class HasParameters(Ensemble):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def parameters(self):
        pass

    @property
    def ensemble_axes_metadata(self):
        axes_metadata = []
        for parameter_name, parameter in self.ensemble_parameters.items():
            axes_metadata += [ParameterSeriesAxis(label=parameter_name,
                                                  values=tuple(parameter.values),
                                                  units='Å',
                                                  _ensemble_mean=parameter.ensemble_mean)]
        return axes_metadata

    @property
    def ensemble_shape(self):
        return tuple(map(sum, tuple(parameter.shape for parameter in self.ensemble_parameters.values())))

    def partition_args(self, chunks=1, lazy: bool = True):
        parameters = self.ensemble_parameters
        chunks = self.validate_chunks(chunks)
        blocks = ()
        for parameter, n in zip(parameters.values(), chunks):
            blocks += (parameter.divide(n, lazy=lazy),)

        return blocks

    @classmethod
    def ctf(cls, *args, keys, **kwargs):
        assert len(args) == len(keys)
        kwargs.update({key: arg for key, arg in zip(keys, args)})
        return cls(**kwargs)

    def from_partitioned_args(self):
        kwargs = self.copy_kwargs()
        keys = tuple(self.ensemble_parameters.keys())
        return partial(self.ctf, keys=keys, **kwargs)

    @property
    def default_ensemble_chunks(self):
        return ('auto',) * len(self.ensemble_shape)

    @property
    def num_new_axes(self):
        num_new_axes = 0
        for key, parameter in self.ensemble_parameters.items():
            num_new_axes += len(parameter.values.shape)
        return num_new_axes

    def _reshaped_parameters(self, shape, xp=np):
        ensemble_parameters = self.ensemble_parameters
        num_new_axes = self.num_new_axes

        weights = None
        for i, (key, parameter) in enumerate(self.ensemble_parameters.items()):
            axis = list(range(num_new_axes))
            del axis[i]
            axis = tuple(axis) + tuple(range(num_new_axes, num_new_axes + len(shape)))
            ensemble_parameters[key] = np.expand_dims(parameter.values, axis=axis)
            ensemble_parameters[key] = xp.asarray(ensemble_parameters[key], dtype=xp.float32)

            new_weights = np.expand_dims(parameter.weights, axis=axis)
            new_weights = xp.asarray(new_weights, dtype=xp.float32)
            weights = new_weights if weights is None else weights * new_weights

        parameters = {key: value for key, value in self.parameters.items()}
        parameters.update(ensemble_parameters)
        return parameters, weights

    @property
    def ensemble_parameters(self) -> Dict:
        ensemble_parameters = {}
        for parameter_name, parameter in self.parameters.items():
            if hasattr(parameter, 'values'):
                ensemble_parameters[parameter_name] = parameter
        return ensemble_parameters


class Aperture(HasParameters, ArrayWaveTransform, HasAcceleratorMixin):

    def __init__(self,
                 semiangle_cutoff: Union[float, Distribution],
                 energy: float = None,
                 taper: float = 0.,
                 normalize: bool = True):

        self._taper = taper
        self._normalize = normalize
        self._accelerator = Accelerator(energy=energy)
        self._parameters = {'semiangle_cutoff': semiangle_cutoff}

    @property
    def metadata(self):
        metadata = {}
        if not 'semiangle_cutoff' in self.ensemble_parameters:
            metadata['semiangle_cutoff'] = self.semiangle_cutoff
        return metadata

    @property
    def normalize(self):
        return self._normalize

    @property
    def parameters(self):
        return self._parameters

    @property
    def nyquist_sampling(self) -> float:
        return 1 / (4 * self.semiangle_cutoff / self.wavelength * 1e-3)

    @property
    def semiangle_cutoff(self):
        return self._parameters['semiangle_cutoff']

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value):
        self._parameters['semiangle_cutoff'] = value

    @property
    def taper(self):
        return self._taper

    def evaluate_with_alpha_and_phi(self, alpha: Union[float, np.ndarray], phi) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        parameters, _ = self._reshaped_parameters(alpha.shape, xp)

        alpha = xp.array(alpha)
        alpha = xp.expand_dims(alpha, axis=tuple(range(0, self.num_new_axes)))

        semiangle_cutoff = parameters['semiangle_cutoff'] / 1000

        if self.semiangle_cutoff == xp.inf:
            return xp.ones_like(alpha)

        if self.taper > 0.:
            taper = self.taper / 1000.
            array = .5 * (1 + xp.cos(np.pi * (alpha - semiangle_cutoff + taper) / taper))
            array[alpha > semiangle_cutoff] = 0.
            array = xp.where(alpha > semiangle_cutoff - taper, array, xp.ones_like(alpha, dtype=xp.float32))
        else:

            array = xp.array(alpha < semiangle_cutoff).astype(xp.float32)

        return array

    def evaluate(self, waves):
        self.accelerator.match(waves)
        waves.grid.check_is_defined()
        alpha, phi = self._polar_spatial_frequencies_from_waves(waves)
        return self.evaluate_with_alpha_and_phi(alpha, phi)


class TemporalEnvelope(HasParameters, ArrayWaveTransform, HasAcceleratorMixin):

    def __init__(self,
                 focal_spread: Union[float, Distribution],
                 energy: float = None,
                 normalize: bool = False):
        self._normalize = normalize
        self._accelerator = Accelerator(energy=energy)
        self._parameters = {'focal_spread': focal_spread}

    @property
    def normalize(self):
        return self._normalize

    @property
    def parameters(self):
        return self._parameters

    @property
    def focal_spread(self):
        return self._parameters['focal_spread']

    @focal_spread.setter
    def focal_spread(self, value):
        self._parameters['focal_spread'] = value

    def evaluate_with_alpha_and_phi(self, alpha: Union[float, np.ndarray], phi) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        parameters, _ = self._reshaped_parameters(alpha.shape, xp)

        alpha = xp.array(alpha)
        alpha = xp.expand_dims(alpha, axis=tuple(range(0, self.num_new_axes)))

        array = xp.exp(- (.5 * xp.pi / self.wavelength * parameters['focal_spread'] * alpha ** 2) ** 2).astype(
            xp.float32)

        if self._normalize:
            array = array / xp.sqrt(array.sum((-2, -1), keepdims=True))

        return array

    def evaluate(self, waves):
        self.accelerator.match(waves)
        waves.grid.check_is_defined()
        alpha, phi = self._polar_spatial_frequencies_from_waves(waves)
        return self.evaluate_with_alpha_and_phi(alpha, phi)


class SpatialEnvelope(HasParameters, ArrayWaveTransform, HasAcceleratorMixin):

    def __init__(self,
                 angular_spread: Union[float, Distribution],
                 energy: float = None,
                 normalize: bool = False,
                 aberrations: 'Aberrations' = None,
                 **kwargs):

        self._normalize = normalize
        self._accelerator = Accelerator(energy=energy)
        self._parameters = {}

        if aberrations is not None:
            self._parameters.update(aberrations.parameters)

        self._parameters.update(kwargs)
        self._parameters['angular_spread'] = angular_spread

        self._aberrations = aberrations
        self._aberrations._accelerator = self._accelerator

    @property
    def aberrations(self):
        return self._aberrations

    @property
    def normalize(self):
        return self._normalize

    @property
    def parameters(self):
        return self._parameters

    @property
    def angular_spread(self):
        return self._parameters['angular_spread']

    @angular_spread.setter
    def angular_spread(self, value):
        self._parameters['angular_spread'] = value

    def evaluate_with_alpha_and_phi(self, alpha: Union[float, np.ndarray], phi) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        p, _ = self._reshaped_parameters(alpha.shape, xp)

        alpha = xp.array(alpha)
        alpha = xp.expand_dims(alpha, axis=tuple(range(0, self.num_new_axes)))

        xp = get_array_module(alpha)

        dchi_dk = 2 * xp.pi / self.wavelength * (
                (p['C12'] * xp.cos(2. * (phi - p['phi12'])) + p['C10']) * alpha +
                (p['C23'] * xp.cos(3. * (phi - p['phi23'])) +
                 p['C21'] * xp.cos(1. * (phi - p['phi21']))) * alpha ** 2 +
                (p['C34'] * xp.cos(4. * (phi - p['phi34'])) +
                 p['C32'] * xp.cos(2. * (phi - p['phi32'])) + p['C30']) * alpha ** 3 +
                (p['C45'] * xp.cos(5. * (phi - p['phi45'])) +
                 p['C43'] * xp.cos(3. * (phi - p['phi43'])) +
                 p['C41'] * xp.cos(1. * (phi - p['phi41']))) * alpha ** 4 +
                (p['C56'] * xp.cos(6. * (phi - p['phi56'])) +
                 p['C54'] * xp.cos(4. * (phi - p['phi54'])) +
                 p['C52'] * xp.cos(2. * (phi - p['phi52'])) + p['C50']) * alpha ** 5)

        dchi_dphi = -2 * xp.pi / self.wavelength * (
                1 / 2. * (2. * p['C12'] * xp.sin(2. * (phi - p['phi12']))) * alpha +
                1 / 3. * (3. * p['C23'] * xp.sin(3. * (phi - p['phi23'])) +
                          1. * p['C21'] * xp.sin(1. * (phi - p['phi21']))) * alpha ** 2 +
                1 / 4. * (4. * p['C34'] * xp.sin(4. * (phi - p['phi34'])) +
                          2. * p['C32'] * xp.sin(2. * (phi - p['phi32']))) * alpha ** 3 +
                1 / 5. * (5. * p['C45'] * xp.sin(5. * (phi - p['phi45'])) +
                          3. * p['C43'] * xp.sin(3. * (phi - p['phi43'])) +
                          1. * p['C41'] * xp.sin(1. * (phi - p['phi41']))) * alpha ** 4 +
                1 / 6. * (6. * p['C56'] * xp.sin(6. * (phi - p['phi56'])) +
                          4. * p['C54'] * xp.sin(4. * (phi - p['phi54'])) +
                          2. * p['C52'] * xp.sin(2. * (phi - p['phi52']))) * alpha ** 5)

        array = xp.exp(-xp.sign(p['angular_spread']) * (p['angular_spread'] / 2 / 1000) ** 2 *
                       (dchi_dk ** 2 + dchi_dphi ** 2))

        if self._normalize:
            array = array / xp.sqrt(array.sum((-2, -1), keepdims=True))

        return array

    def evaluate(self, waves):
        self.accelerator.match(waves)
        waves.grid.check_is_defined()
        alpha, phi = self._polar_spatial_frequencies_from_waves(waves)
        return self.evaluate_with_alpha_and_phi(alpha, phi)


class Aberrations(HasParameters, ArrayWaveTransform, HasAcceleratorMixin):

    def __init__(self,
                 energy: float = None,
                 parameters: Union[Mapping[str, float], Mapping[str, Distribution]] = None,
                 **kwargs):

        for key in kwargs.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()):
                raise ValueError('{} not a recognized parameter'.format(key))

        self._parameters = dict(zip(polar_symbols, [0.] * len(polar_symbols)))

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)

        self.set_parameters(parameters)

        def parametrization_property(key):

            def getter(self):
                return self._parameters[key]

            def setter(self, value):
                self._parameters[key] = value

            return property(getter, setter)

        for symbol in polar_symbols:
            setattr(self.__class__, symbol, parametrization_property(symbol))

        for key, value in polar_aliases.items():
            if key != 'defocus':
                setattr(self.__class__, key, parametrization_property(value))

        self._accelerator = Accelerator(energy=energy)

    @property
    def parameters(self) -> Dict[str, Union[float, Distribution]]:
        """The parameters."""
        return self._parameters

    @property
    def defocus(self) -> float:
        """The defocus [Å]."""
        return - self._parameters['C10']

    @defocus.setter
    def defocus(self, value: float):
        self.C10 = -value

    def evaluate_with_alpha_and_phi(self,
                                    alpha: Union[float, np.ndarray],
                                    phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        p, weights = self._reshaped_parameters(alpha.shape, xp)

        axis = tuple(range(0, self.num_new_axes))
        alpha = xp.array(alpha)
        alpha = xp.expand_dims(alpha, axis=axis)
        phi = xp.expand_dims(phi, axis=axis)
        alpha2 = alpha ** 2

        array = xp.zeros(alpha.shape, dtype=np.float32)

        if any(np.any(p[symbol] != 0.) for symbol in ('C10', 'C12', 'phi12')):
            array = array + (1 / 2 * alpha2 *
                             (p['C10'] +
                              p['C12'] * xp.cos(2 * (phi - p['phi12']))))

        if any(np.any(p[symbol] != 0.) for symbol in ('C21', 'phi21', 'C23', 'phi23')):
            array = array + (1 / 3 * alpha2 * alpha *
                             (p['C21'] * xp.cos(phi - p['phi21']) +
                              p['C23'] * xp.cos(3 * (phi - p['phi23']))))

        if any(np.any(p[symbol] != 0.) for symbol in ('C30', 'C32', 'phi32', 'C34', 'phi34')):
            array = array + (1 / 4 * alpha2 ** 2 *
                             (p['C30'] +
                              p['C32'] * xp.cos(2 * (phi - p['phi32'])) +
                              p['C34'] * xp.cos(4 * (phi - p['phi34']))))

        if any(np.any(p[symbol] != 0.) for symbol in ('C41', 'phi41', 'C43', 'phi43', 'C45', 'phi41')):
            array = array + (1 / 5 * alpha2 ** 2 * alpha *
                             (p['C41'] * xp.cos((phi - p['phi41'])) +
                              p['C43'] * xp.cos(3 * (phi - p['phi43'])) +
                              p['C45'] * xp.cos(5 * (phi - p['phi45']))))

        if any(np.any(p[symbol] != 0.) for symbol in ('C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')):
            array = array + (1 / 6 * alpha2 ** 3 *
                             (p['C50'] +
                              p['C52'] * xp.cos(2 * (phi - p['phi52'])) +
                              p['C54'] * xp.cos(4 * (phi - p['phi54'])) +
                              p['C56'] * xp.cos(6 * (phi - p['phi56']))))

        array = np.float32(2 * xp.pi / self.wavelength) * array

        array = complex_exponential(-array)

        if weights is not None:
            array = xp.asarray(weights) * array

        return array

    def evaluate(self, waves):
        self.accelerator.match(waves)
        waves.grid.check_is_defined()
        alpha, phi = self._polar_spatial_frequencies_from_waves(waves)
        return self.evaluate_with_alpha_and_phi(alpha, phi)

    def set_parameters(self, parameters: Dict[str, float]):
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

            elif symbol == 'defocus':
                self._parameters[polar_aliases[symbol]] = -value

            elif symbol in polar_aliases.keys():
                self._parameters[polar_aliases[symbol]] = value

            else:
                raise ValueError('{} not a recognized parameter'.format(symbol))

        return parameters


class CTF(HasParameters, ArrayWaveTransform, HasAcceleratorMixin):

    def __init__(self,
                 energy: float = None,
                 semiangle_cutoff: float = np.inf,
                 taper: float = 0.,
                 focal_spread: float = 0.,
                 angular_spread: float = 0.,
                 aberrations: Union[dict, Aberrations] = None,
                 **kwargs):

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
        elif isinstance(aberrations, Aberrations):
            aberrations = copy.deepcopy(aberrations.parameters)

        aberrations.update(kwargs)

        aberrations = Aberrations(energy=energy, parameters=aberrations)
        aperture = Aperture(energy=energy, semiangle_cutoff=semiangle_cutoff, taper=taper)
        spatial_envelope = SpatialEnvelope(angular_spread=angular_spread, aberrations=aberrations)
        temporal_envelope = TemporalEnvelope(focal_spread=focal_spread)

        self._set_parts(aberrations, aperture, temporal_envelope, spatial_envelope)

    def _set_parts(self, aberrations, aperture, temporal_envelope, spatial_envelope):
        self._aberrations = aberrations
        self._aperture = aperture
        self._temporal_envelope = temporal_envelope
        self._spatial_envelope = spatial_envelope

        self._transforms = [aberrations, spatial_envelope, temporal_envelope, aperture]

        self._accelerator = Accelerator(energy=aperture.energy)

        self._aberrations._accelerator = self._accelerator
        self._aperture._accelerator = self._accelerator
        self._spatial_envelope._accelerator = self._accelerator
        self._temporal_envelope._accelerator = self._accelerator

    @classmethod
    def from_parts(cls, aberrations, aperture, temporal_envelope, spatial_envelope):
        ctf = cls()
        ctf._set_parts(aberrations, aperture, temporal_envelope, spatial_envelope)
        return ctf

    @property
    def scherzer_defocus(self):
        self.accelerator.check_is_defined()

        if self.aberrations.Cs == 0.:  # noqa
            raise ValueError()

        return scherzer_defocus(self.aberrations.Cs, self.energy)  # noqa

    @property
    def crossover_angle(self):
        return 1e3 * energy2wavelength(self.energy) / self.point_resolution

    @property
    def point_resolution(self):
        return point_resolution(self.aberrations.Cs, self.energy)  # noqa

    @property
    def parameters(self):
        parameters = {**self.aberrations.parameters,
                      **self.spatial_envelope.parameters,
                      **self.temporal_envelope.parameters,
                      **self.aperture.parameters,
                      }
        return parameters

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
    def nyquist_sampling(self) -> float:
        return 1 / (4 * self.semiangle_cutoff / self.wavelength * 1e-3)

    @property
    def defocus(self) -> float:
        """The defocus [Å]."""
        return self.aberrations.defocus

    @defocus.setter
    def defocus(self, value: float):
        self.aberrations.defocus = value

    @property
    def taper(self) -> float:
        return self.aperture.taper

    @property
    def semiangle_cutoff(self) -> float:
        """The semi-angle cutoff [mrad]."""
        return self.aperture.semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value: float):
        self.aperture.semiangle_cutoff = value

    @property
    def focal_spread(self) -> float:
        """The focal spread [Å]."""
        return self.temporal_envelope.focal_spread

    @focal_spread.setter
    def focal_spread(self, value: float):
        """The angular spread [mrad]."""
        self.temporal_envelope.focal_spread = value

    @property
    def angular_spread(self) -> float:
        return self.spatial_envelope.angular_spread

    @angular_spread.setter
    def angular_spread(self, value: float):
        self.spatial_envelope.angular_spread = value

    def evaluate_with_alpha_and_phi(self, alpha, phi):
        array = self.aberrations.evaluate_with_alpha_and_phi(alpha, phi)

        if self.angular_spread != 0.:
            new_aberrations_dims = tuple(range(self.aberrations.ensemble_dims))
            old_match_dims = new_aberrations_dims + (-2, -1)

            added_dims = int(hasattr(self.spatial_envelope.angular_spread, 'values'))
            new_match_dims = tuple(range(self.spatial_envelope.ensemble_dims - added_dims)) + (-2, -1)

            new_array = self.spatial_envelope.evaluate_with_alpha_and_phi(alpha, phi)
            array, new_array = expand_dims_to_match(array, new_array, match_dims=[old_match_dims, new_match_dims])
            array = array * new_array

        if self.focal_spread != 0.:
            new_array = self.temporal_envelope.evaluate_with_alpha_and_phi(alpha, phi)
            array, new_array = expand_dims_to_match(array, new_array, match_dims=[(-2, -1), (-2, -1)])
            array = array * new_array

        if self.semiangle_cutoff != np.inf:
            new_array = self.aperture.evaluate_with_alpha_and_phi(alpha, phi)
            array, new_array = expand_dims_to_match(array, new_array, match_dims=[(-2, -1), (-2, -1)])
            array = array * new_array

        return array

    def evaluate(self, waves):
        self.accelerator.match(waves)
        waves.grid.check_is_defined()
        alpha, phi = self._polar_spatial_frequencies_from_waves(waves)
        return self.evaluate_with_alpha_and_phi(alpha, phi)

    def image(self, gpts, max_angle):

        angular_sampling = 2 * max_angle / gpts[0], 2 * max_angle / gpts[1]

        fourier_space_sampling = (angular_sampling[0] / (self.wavelength * 1e3),
                                  angular_sampling[1] / (self.wavelength * 1e3))

        sampling = 1 / (fourier_space_sampling[0] * gpts[0]), 1 / (fourier_space_sampling[1] * gpts[1])

        alpha, phi = self._polar_spatial_frequencies_from_grid(gpts=gpts, sampling=sampling, wavelength=self.wavelength,
                                                               xp=np)

        array = np.fft.fftshift(self.evaluate_with_alpha_and_phi(alpha, phi))

        # array = np.fft.fftshift(self.evaluate(waves))
        return DiffractionPatterns(array, sampling=fourier_space_sampling, metadata={'energy': self.energy})

    # def point_spread_function(self, waves):
    #     alpha, phi = self._polar_spatial_frequencies_from_waves(waves)
    #     xp = get_array_module(waves.device)
    #     array = xp.fft.fftshift(ifft2(self.evaluate_with_alpha_and_phi(alpha, phi)))
    #     return Images(array, sampling=waves.sampling, metadata={'energy': self.energy})

    def profiles(self, max_angle: float = None, phi: float = 0.):
        if max_angle is None:
            if self.semiangle_cutoff == np.inf:
                max_angle = 50
            else:
                max_angle = self.semiangle_cutoff * 1.6

        sampling = max_angle / 1000. / 1000.
        alpha = np.arange(0, max_angle / 1000., sampling)

        aberrations = self.aberrations.evaluate_with_alpha_and_phi(alpha, phi)
        spatial_envelope = self.spatial_envelope.evaluate_with_alpha_and_phi(alpha, phi)
        temporal_envelope = self.temporal_envelope.evaluate_with_alpha_and_phi(alpha, phi)
        aperture = self.aperture.evaluate_with_alpha_and_phi(alpha, phi)
        envelope = aperture * temporal_envelope * spatial_envelope

        sampling = alpha[1] / energy2wavelength(self.energy)

        axis_metadata = ['ctf']
        metadata = {'energy': self.energy}
        profiles = [FourierSpaceLineProfiles(-aberrations.imag * envelope, sampling=sampling, metadata=metadata)]

        if self.semiangle_cutoff != np.inf:
            profiles += [FourierSpaceLineProfiles(aperture, sampling=sampling, metadata=metadata)]
            axis_metadata += ['aperture']

        if self.focal_spread > 0. and self.angular_spread > 0.:
            profiles += [FourierSpaceLineProfiles(envelope, sampling=sampling, metadata=metadata)]
            axis_metadata += ['envelope']

        if self.focal_spread > 0.:
            profiles += [FourierSpaceLineProfiles(temporal_envelope, sampling=sampling, metadata=metadata)]
            axis_metadata += ['temporal']

        if self.angular_spread > 0.:
            profiles += [FourierSpaceLineProfiles(spatial_envelope, sampling=sampling, metadata=metadata)]
            axis_metadata += ['spatial']

        return stack(profiles, axis_metadata=OrdinalAxis(values=tuple(axis_metadata)))


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
    cartesian['C10'] = polar['C10']
    cartesian['C12a'] = - polar['C12'] * np.cos(2 * polar['phi12'])
    cartesian['C12b'] = polar['C12'] * np.sin(2 * polar['phi12'])
    cartesian['C21a'] = polar['C21'] * np.sin(polar['phi21'])
    cartesian['C21b'] = polar['C21'] * np.cos(polar['phi21'])
    cartesian['C23a'] = - polar['C23'] * np.sin(3 * polar['phi23'])
    cartesian['C23b'] = polar['C23'] * np.cos(3 * polar['phi23'])
    cartesian['C30'] = polar['C30']
    cartesian['C32a'] = - polar['C32'] * np.cos(2 * polar['phi32'])
    cartesian['C32b'] = polar['C32'] * np.cos(np.pi / 2 - 2 * polar['phi32'])
    cartesian['C34a'] = polar['C34'] * np.cos(-4 * polar['phi34'])
    K = np.sqrt(3 + np.sqrt(8.))
    cartesian['C34b'] = 1 / 4. * (1 + K ** 2) ** 2 / (K ** 3 - K) * polar['C34'] * np.cos(
        4 * np.arctan(1 / K) - 4 * polar['phi34'])

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
    polar['C10'] = cartesian['C10']
    polar['C12'] = - np.sqrt(cartesian['C12a'] ** 2 + cartesian['C12b'] ** 2)
    polar['phi12'] = - np.arctan2(cartesian['C12b'], cartesian['C12a']) / 2.
    polar['C21'] = np.sqrt(cartesian['C21a'] ** 2 + cartesian['C21b'] ** 2)
    polar['phi21'] = np.arctan2(cartesian['C21a'], cartesian['C21b'])
    polar['C23'] = np.sqrt(cartesian['C23a'] ** 2 + cartesian['C23b'] ** 2)
    polar['phi23'] = -np.arctan2(cartesian['C23a'], cartesian['C23b']) / 3.
    polar['C30'] = cartesian['C30']
    polar['C32'] = -np.sqrt(cartesian['C32a'] ** 2 + cartesian['C32b'] ** 2)
    polar['phi32'] = -np.arctan2(cartesian['C32b'], cartesian['C32a']) / 2.
    polar['C34'] = np.sqrt(cartesian['C34a'] ** 2 + cartesian['C34b'] ** 2)
    polar['phi34'] = np.arctan2(cartesian['C34b'], cartesian['C34a']) / 4

    return polar
