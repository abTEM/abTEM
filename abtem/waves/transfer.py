"""Module to describe the contrast transfer function."""
import copy
import itertools
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import Mapping, Union, TYPE_CHECKING, Dict

import numpy as np

from abtem.core.axes import ParameterSeriesAxis
from abtem.core.backend import get_array_module
from abtem.core.blockwise import Ensemble
from abtem.core.complex import complex_exponential
from abtem.core.dask import validate_chunks
from abtem.core.distributions import ParameterSeries, Distribution
from abtem.core.energy import Accelerator, HasAcceleratorMixin, energy2wavelength
from abtem.core.fft import ifft2
from abtem.core.grid import Grid, polar_spatial_frequencies

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


class WaveTransform(Ensemble):

    def __add__(self, other):
        wave_transforms = []

        for wave_transform in (self, other):

            if hasattr(wave_transform, 'wave_transforms'):
                wave_transforms += wave_transform.wave_transforms
            else:
                wave_transforms += [wave_transform]

        return CompositeWaveTransform(wave_transforms)

    @abstractmethod
    def apply(self, waves):
        pass


class ArrayWaveTransform(WaveTransform):

    def _get_polar_spatial_frequencies(self, waves):
        gpts = waves.gpts
        extent = waves.extent
        device = waves.device

        xp = get_array_module(device)
        grid = Grid(gpts=gpts, extent=extent)
        alpha, phi = polar_spatial_frequencies(grid.gpts, grid.sampling, xp=xp)
        alpha *= waves.wavelength
        return alpha, phi

    def evaluate_with_alpha_and_phi(self, alpha, phi):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, waves: 'WavesLikeMixin'):
        raise NotImplementedError

    def apply(self, waves: 'Waves', out_space: 'str' = 'in_space'):
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

        d = waves._copy_as_dict(copy_array=False)
        d['fourier_space'] = fourier_space_out
        d['array'] = array
        d['ensemble_axes_metadata'] = self.ensemble_axes_metadata + d['ensemble_axes_metadata']
        return waves.__class__(**d)


class CompositeWaveTransform(WaveTransform):

    def __init__(self, wave_transforms=None):

        if wave_transforms is None:
            wave_transforms = []

        self._wave_transforms = wave_transforms
        super().__init__()

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

    def ensemble_blocks(self, chunks=None):
        if chunks is None:
            chunks = self.default_ensemble_chunks

        chunks = validate_chunks(self.ensemble_shape, chunks, limit=None)

        blocks = ()
        start = 0
        for wave_transform in self.wave_transforms:
            stop = start + wave_transform.ensemble_dims
            blocks += wave_transform.ensemble_blocks(chunks[start:stop])
            start = stop

        return blocks

    def apply(self, waves: 'WavesLikeMixin'):
        waves.grid.check_is_defined()

        for wave_transform in reversed(self.wave_transforms):
            waves = wave_transform.apply(waves)

        return waves

    def ensemble_partial(self):
        def ctf(*args, partials):
            wave_transfer_functions = []
            for p in partials:
                wave_transfer_functions += [p[0](*[args[i] for i in p[1]]).item()]

            arr = np.zeros((1,) * len(args), dtype=object)
            arr.itemset(CompositeWaveTransform(wave_transfer_functions))
            return arr

        partials = ()
        i = 0
        for wave_transform in self.wave_transforms:
            indices = tuple(range(i, i + len(wave_transform.ensemble_shape)))
            partials += ((wave_transform.ensemble_partial(), indices),)
            i += len(indices)

        return partial(ctf, partials=partials)

    def copy(self):
        wave_transforms = [wave_transform.copy() for wave_transform in self.wave_transforms]
        return self.__class__(wave_transforms)


def ensemble_axes_metadata_from_parameters(parameters):
    axes_metadata = []
    for parameter_name, parameter in parameters.items():
        axes_metadata += [ParameterSeriesAxis(label=parameter_name,
                                              values=tuple(parameter.values),
                                              units='Å',
                                              _ensemble_mean=parameter.ensemble_mean)]
    return axes_metadata


def ensemble_shape_from_parameters(parameters):
    return tuple(map(sum, tuple(parameter.shape for parameter in parameters.values())))


def ensemble_blocks_from_parameters(parameters, chunks):
    blocks = ()
    for parameter, n in zip(parameters.values(), chunks):
        blocks += (parameter.divide(n, lazy=True),)
    return blocks


class Aperture(ArrayWaveTransform, HasAcceleratorMixin):

    def __init__(self,
                 semiangle_cutoff: Union[float, Distribution],
                 energy: float = None,
                 normalize: bool = False,
                 taper: float = 0.):

        self._semiangle_cutoff = semiangle_cutoff
        self._taper = taper
        self._normalize = normalize
        self._accelerator = Accelerator(energy=energy)

    @property
    def ensemble_parameters(self):
        ensemble_parameters = {}
        if hasattr(self.semiangle_cutoff, 'values'):
            ensemble_parameters['semiangle_cutoff'] = self.semiangle_cutoff
        return ensemble_parameters

    @property
    def ensemble_shape(self):
        return ensemble_shape_from_parameters(self.ensemble_parameters)

    @property
    def ensemble_axes_metadata(self):
        return ensemble_axes_metadata_from_parameters(self.ensemble_parameters)

    def ensemble_blocks(self, chunks):
        return ensemble_blocks_from_parameters(self.ensemble_parameters, chunks)

    @property
    def default_ensemble_chunks(self):
        return ('auto',) * len(self.ensemble_shape)

    def ensemble_partial(self):
        def ctf(*args, keys, **kwargs):
            assert len(args) == len(keys)
            kwargs.update({key: arg.item() for key, arg in zip(keys, args)})
            arr = np.zeros((1,) * len(args), dtype=object)
            arr.itemset(Aperture(**kwargs))
            return arr

        kwargs = self._copy_as_dict()
        return partial(ctf, keys=tuple(self.ensemble_parameters.keys()), **kwargs)

    @property
    def nyquist_sampling(self) -> float:
        return 1 / (4 * self.semiangle_cutoff / self.wavelength * 1e-3)

    @property
    def semiangle_cutoff(self):
        return self._semiangle_cutoff

    @property
    def taper(self):
        return self._taper

    def evaluate_with_alpha_and_phi(self, alpha: Union[float, np.ndarray], phi) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        parameters = {'semiangle_cutoff': self.semiangle_cutoff}

        num_new_axes = 0
        for key, parameter in self.ensemble_parameters.items():
            num_new_axes += len(parameter.values.shape)

        for i, (key, parameter) in enumerate(self.ensemble_parameters.items()):
            axis = list(range(num_new_axes))
            del axis[i]
            axis = tuple(axis) + tuple(range(num_new_axes, num_new_axes + len(alpha.shape)))
            parameters[key] = xp.expand_dims(parameter.values, axis=axis)

        axis = tuple(range(0, num_new_axes))
        alpha = xp.array(alpha)
        alpha = xp.expand_dims(alpha, axis=axis)

        parameters['semiangle_cutoff'] = parameters['semiangle_cutoff'] / 1000

        if self.semiangle_cutoff == xp.inf:
            return xp.ones_like(alpha)

        if self.taper > 0.:
            rolloff = self.taper / 1000.
            array = .5 * (1 + xp.cos(np.pi * (alpha - parameters['semiangle_cutoff'] + rolloff) / rolloff))
            array[alpha > parameters['semiangle_cutoff']] = 0.
            array = xp.where(alpha > parameters['semiangle_cutoff'] - rolloff, array,
                             xp.ones_like(alpha, dtype=xp.float32))
        else:

            array = xp.array(alpha <= parameters['semiangle_cutoff']).astype(xp.float32)

        if self._normalize:
            array = array / xp.sqrt(array.sum((-2, -1), keepdims=True))

        return array

    def evaluate(self, waves):
        self.accelerator.match(waves)
        waves.grid.check_is_defined()
        alpha, phi = self._get_polar_spatial_frequencies(waves)
        return self.evaluate_with_alpha_and_phi(alpha, phi)

    def _copy_as_dict(self):
        d = {'energy': self.energy,
             'semiangle_cutoff': copy.copy(self.semiangle_cutoff),
             'normalize': self._normalize,
             'taper': copy.copy(self.taper),
             }
        return d

    def copy(self):
        return self.__class__(**self._copy_as_dict())


class Aberrations(ArrayWaveTransform, HasAcceleratorMixin):

    def __init__(self,
                 energy: float = None,
                 parameters: Union[Mapping[str, float], Mapping[str, ParameterSeries]] = None,
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
    def parameters(self) -> Dict[str, Union[float, ParameterSeries]]:
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

        p = {key: value for key, value in self.parameters.items()}

        num_new_axes = 0
        for key, parameter in self.ensemble_parameters.items():
            num_new_axes += len(parameter.values.shape)

        weights = None
        for i, (key, parameter) in enumerate(self.ensemble_parameters.items()):
            axis = list(range(num_new_axes))
            del axis[i]
            axis = tuple(axis) + tuple(range(num_new_axes, num_new_axes + len(alpha.shape)))
            p[key] = xp.expand_dims(parameter.values, axis=axis)

            new_weights = xp.expand_dims(parameter.weights, axis=axis)
            weights = new_weights if weights is None else weights * new_weights

        axis = tuple(range(0, num_new_axes))
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
            array = array * weights

        return array

    def evaluate(self, waves):
        self.accelerator.match(waves)
        waves.grid.check_is_defined()
        alpha, phi = self._get_polar_spatial_frequencies(waves)
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

    def ensemble_partial(self):
        def ctf(*args, keys, **kwargs):
            assert len(args) == len(keys)
            kwargs.update({key: arg.item() for key, arg in zip(keys, args)})
            arr = np.zeros((1,) * len(args), dtype=object)
            arr.itemset(Aberrations(**kwargs))
            return arr

        kwargs = self._copy_as_dict()
        parameter_names = ()
        for parameter_name in self.ensemble_parameters.keys():
            del kwargs['parameters'][parameter_name]
            parameter_names += (parameter_name,)

        return partial(ctf, keys=parameter_names, **kwargs)

    @property
    def ensemble_parameters(self) -> Dict:
        ensemble_parameters = {}
        for parameter_name, parameter in self.parameters.items():
            if hasattr(parameter, 'values'):
                ensemble_parameters[parameter_name] = parameter
        return ensemble_parameters

    @property
    def ensemble_shape(self):
        return ensemble_shape_from_parameters(self.ensemble_parameters)

    @property
    def ensemble_axes_metadata(self):
        return ensemble_axes_metadata_from_parameters(self.ensemble_parameters)

    def ensemble_blocks(self, chunks):
        return ensemble_blocks_from_parameters(self.ensemble_parameters, chunks)

    @property
    def default_ensemble_chunks(self):
        return ('auto',) * len(self.ensemble_shape)

    def _copy_as_dict(self):
        d = {'energy': self.energy,
             'parameters': copy.copy(self._parameters)}
        return d

    def copy(self):
        return self.__class__(**self._copy_as_dict())


# def CTF(energy: float = None, aperture: float = np.inf, taper: float = 0., parameters: dict = None, **kwargs):
#     return Aperture(energy=energy, semiangle_cutoff=aperture, taper=taper) * \
#            Aberrations(parameters=parameters, **kwargs)


class CTF(CompositeWaveTransform, HasAcceleratorMixin):

    def __init__(self, energy=None, semiangle_cutoff=np.inf, taper=0., parameters=None, aperture=None, **kwargs):
        self._aberrations = Aberrations(energy=energy, parameters=parameters, **kwargs)

        if aperture is None:
            aperture = Aperture(energy=energy, semiangle_cutoff=semiangle_cutoff, taper=taper)

        self._aperture = aperture

        transforms = [self._aberrations, self._aperture]

        super().__init__(wave_transforms=transforms)

        self._accelerator = self._aberrations._accelerator = self._aperture._accelerator


#
#     @property
#     def compound_wave_transfer_function(self):
#         return self._compund_wave_transfer_function
#
#     @property
#     def aberrations(self):
#         return self._aberrations
#
#     @property
#     def aperture(self):
#         return self._aperture
#
#     def ensemble_blocks(self, chunks=None):
#         return self.compound_wave_transfer_function.ensemble_blocks(chunks)
#
#     def ensemble_partial(self):
#         return self.compound_wave_transfer_function.ensemble_partial()
#
#     @property
#     def ensemble_shape(self):
#         return self.compound_wave_transfer_function.ensemble_shape
#
#     @property
#     def default_ensemble_chunks(self):
#         return self._aberrations.default_ensemble_chunks
#
#     @property
#     def ensemble_axes_metadata(self):
#         return self._aberrations.ensemble_axes_metadata
#
#     def evaluate_for_waves(self, waves):
#         return self.compound_wave_transfer_function.evaluate_for_waves(waves)
#
#     def _copy_as_dict(self):
#         d = {'energy': self.energy,
#              'aperture': self._aperture.copy(),
#              'parameters': copy.copy(self._aberrations._parameters),
#              }
#         return d
#
#     def copy(self):
#         return self.__class__(**self._copy_as_dict())


#
# class CTF(HasAcceleratorMixin, Ensemble):
#     """
#     Contrast transfer function object
#
#     The Contrast Transfer Function (CTF) describes the aberrations of the objective lens in HRTEM and specifies how the
#     condenser system shapes the probe in STEM.
#
#     abTEM implements phase aberrations up to 5th order using polar coefficients. See Eq. 2.22 in the reference [1]_.
#     Cartesian coefficients can be converted to polar using the utility function abtem.transfer.cartesian2polar.
#
#     Partial coherence is included as an envelope in the quasi-coherent approximation. See Chapter 3.2 in reference [1]_.
#
#     For a more detailed discussion with examples, see our `walkthrough
#     <https://abtem.readthedocs.io/en/latest/walkthrough/05_contrast_transfer_function.html>`_.
#
#     Parameters
#     ----------
#     semiangle_cutoff: float
#         The semiangle cutoff describes the sharp Fourier space cutoff due to the objective aperture [mrad].
#     rolloff: float
#         Tapers the cutoff edge over the given angular range [mrad].
#     focal_spread: float
#         The 1/e width of the focal spread due to chromatic aberration and lens current instability [Å].
#     angular_spread: float
#         The 1/e width of the angular deviations due to source size [Å].
#     gaussian_spread:
#         The 1/e width image deflections due to vibrations and thermal magnetic noise [Å].
#     energy: float
#         The electron energy of the wave functions this contrast transfer function will be applied to [eV].
#     parameters: dict
#         Mapping from aberration symbols to their corresponding values. All aberration magnitudes should be given in Å
#         and angles should be given in radians.
#     normalize : {'values', 'amplitude', 'intensity'}
#     weight : float
#     kwargs:
#         Provide the aberration coefficients as keyword arguments.
#
#     References
#     ----------
#     .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy (2nd ed.). Springer.
#
#     """
#
#     def __init__(self,
#                  semiangle_cutoff: float = np.inf,
#                  focal_spread: float = 0.,
#                  angular_spread: float = 0.,
#                  gaussian_spread: float = 0.,
#                  energy: float = None,
#                  parameters: Union[Mapping[str, float], Mapping[str, ParameterSeries]] = None,
#                  aperture=None,
#                  normalize: str = 'values',
#                  weight: float = 1.,
#                  **kwargs):
#
#         for key in kwargs.keys():
#             if (key not in polar_symbols) and (key not in polar_aliases.keys()):
#                 raise ValueError('{} not a recognized parameter'.format(key))
#
#         self._accelerator = Accelerator(energy=energy)
#
#         if aperture is not None:
#             self._aperture = aperture
#
#             if semiangle_cutoff < np.inf:
#                 raise RuntimeError()
#
#         elif semiangle_cutoff < np.inf:
#             self._aperture = Aperture(semiangle_cutoff=semiangle_cutoff)
#
#         if self._aperture is not None:
#             self._accelerator = self._aperture.accelerator
#
#         self._semiangle_cutoff = semiangle_cutoff
#         self._focal_spread = focal_spread
#         self._angular_spread = angular_spread
#         self._gaussian_spread = gaussian_spread
#         self._parameters = dict(zip(polar_symbols, [0.] * len(polar_symbols)))
#         self._normalize = normalize
#         self._weight = weight
#
#         if parameters is None:
#             parameters = {}
#
#         parameters.update(kwargs)
#         self.set_parameters(parameters)
#
#         def parametrization_property(key):
#
#             def getter(self):
#                 return self._parameters[key]
#
#             def setter(self, value):
#                 self._parameters[key] = value
#
#             return property(getter, setter)
#
#         for symbol in polar_symbols:
#             setattr(self.__class__, symbol, parametrization_property(symbol))
#
#         for key, value in polar_aliases.items():
#             if key != 'defocus':
#                 setattr(self.__class__, key, parametrization_property(value))
#
#     @property
#     def normalize(self):
#         return self._normalize
#
#     @property
#     def weight(self):
#         return self._weight
#
#     @property
#     def nyquist_sampling(self) -> float:
#         return 1 / (4 * self.semiangle_cutoff / self.wavelength * 1e-3)
#
#     @property
#     def parameters(self) -> Dict[str, Union[float, ParameterSeries]]:
#         """The parameters."""
#         return self._parameters
#
#     @property
#     def defocus(self) -> float:
#         """The defocus [Å]."""
#         return - self._parameters['C10']
#
#     @defocus.setter
#     def defocus(self, value: float):
#         self.C10 = -value
#
#     @property
#     def semiangle_cutoff(self) -> float:
#         """The semi-angle cutoff [mrad]."""
#         return self._semiangle_cutoff
#
#     @semiangle_cutoff.setter
#     def semiangle_cutoff(self, value: float):
#         self._semiangle_cutoff = value
#
#     @property
#     def rolloff(self) -> float:
#         """The fraction of soft tapering of the cutoff."""
#         return self._rolloff
#
#     @rolloff.setter
#     def rolloff(self, value: float):
#         self._rolloff = value
#
#     @property
#     def focal_spread(self) -> float:
#         """The focal spread [Å]."""
#         return self._focal_spread
#
#     @focal_spread.setter
#     def focal_spread(self, value: float):
#         """The angular spread [mrad]."""
#         self._focal_spread = value
#
#     @property
#     def angular_spread(self) -> float:
#         return self._angular_spread
#
#     @angular_spread.setter
#     def angular_spread(self, value: float):
#         self._angular_spread = value
#
#     @property
#     def gaussian_spread(self) -> float:
#         """The Gaussian spread [Å]."""
#         return self._gaussian_spread
#
#     @gaussian_spread.setter
#     def gaussian_spread(self, value: float):
#         self._gaussian_spread = value
#
#     def set_parameters(self, parameters: Dict[str, float]):
#         """
#         Set the phase of the phase aberration.
#
#         Parameters
#         ----------
#         parameters: dict
#             Mapping from aberration symbols to their corresponding values.
#         """
#
#         for symbol, value in parameters.items():
#             if symbol in self._parameters.keys():
#                 self._parameters[symbol] = value
#
#             elif symbol == 'defocus':
#                 self._parameters[polar_aliases[symbol]] = -value
#
#             elif symbol in polar_aliases.keys():
#                 self._parameters[polar_aliases[symbol]] = value
#
#             else:
#                 raise ValueError('{} not a recognized parameter'.format(symbol))
#
#         return parameters
#
#     def evaluate_temporal_envelope(self, alpha: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         xp = get_array_module(alpha)
#         return xp.exp(- (.5 * xp.pi / self.wavelength * self.focal_spread * alpha ** 2) ** 2).astype(xp.float32)
#
#     def evaluate_gaussian_envelope(self, alpha: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         xp = get_array_module(alpha)
#         return xp.exp(- .5 * self.gaussian_spread ** 2 * alpha ** 2 / self.wavelength ** 2)
#
#     def evaluate_spatial_envelope(self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> \
#             Union[float, np.ndarray]:
#         xp = get_array_module(alpha)
#         p = self.parameters
#         dchi_dk = 2 * xp.pi / self.wavelength * (
#                 (p['C12'] * xp.cos(2. * (phi - p['phi12'])) + p['C10']) * alpha +
#                 (p['C23'] * xp.cos(3. * (phi - p['phi23'])) +
#                  p['C21'] * xp.cos(1. * (phi - p['phi21']))) * alpha ** 2 +
#                 (p['C34'] * xp.cos(4. * (phi - p['phi34'])) +
#                  p['C32'] * xp.cos(2. * (phi - p['phi32'])) + p['C30']) * alpha ** 3 +
#                 (p['C45'] * xp.cos(5. * (phi - p['phi45'])) +
#                  p['C43'] * xp.cos(3. * (phi - p['phi43'])) +
#                  p['C41'] * xp.cos(1. * (phi - p['phi41']))) * alpha ** 4 +
#                 (p['C56'] * xp.cos(6. * (phi - p['phi56'])) +
#                  p['C54'] * xp.cos(4. * (phi - p['phi54'])) +
#                  p['C52'] * xp.cos(2. * (phi - p['phi52'])) + p['C50']) * alpha ** 5)
#
#         dchi_dphi = -2 * xp.pi / self.wavelength * (
#                 1 / 2. * (2. * p['C12'] * xp.sin(2. * (phi - p['phi12']))) * alpha +
#                 1 / 3. * (3. * p['C23'] * xp.sin(3. * (phi - p['phi23'])) +
#                           1. * p['C21'] * xp.sin(1. * (phi - p['phi21']))) * alpha ** 2 +
#                 1 / 4. * (4. * p['C34'] * xp.sin(4. * (phi - p['phi34'])) +
#                           2. * p['C32'] * xp.sin(2. * (phi - p['phi32']))) * alpha ** 3 +
#                 1 / 5. * (5. * p['C45'] * xp.sin(5. * (phi - p['phi45'])) +
#                           3. * p['C43'] * xp.sin(3. * (phi - p['phi43'])) +
#                           1. * p['C41'] * xp.sin(1. * (phi - p['phi41']))) * alpha ** 4 +
#                 1 / 6. * (6. * p['C56'] * xp.sin(6. * (phi - p['phi56'])) +
#                           4. * p['C54'] * xp.sin(4. * (phi - p['phi54'])) +
#                           2. * p['C52'] * xp.sin(2. * (phi - p['phi52']))) * alpha ** 5)
#
#         return xp.exp(-xp.sign(self.angular_spread) * (self.angular_spread / 2 / 1000) ** 2 *
#                       (dchi_dk ** 2 + dchi_dphi ** 2))
#
#     def evaluate_chi(self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         xp = get_array_module(alpha)
#
#         p = self.parameters
#
#         alpha2 = alpha ** 2
#         alpha = xp.array(alpha)
#
#         array = xp.zeros(alpha.shape, dtype=np.float32)
#         if any([p[symbol] != 0. for symbol in ('C10', 'C12', 'phi12')]):
#             array += (1 / 2 * alpha2 *
#                       (p['C10'] +
#                        p['C12'] * xp.cos(2 * (phi - p['phi12']))))
#
#         if any([p[symbol] != 0. for symbol in ('C21', 'phi21', 'C23', 'phi23')]):
#             array += (1 / 3 * alpha2 * alpha *
#                       (p['C21'] * xp.cos(phi - p['phi21']) +
#                        p['C23'] * xp.cos(3 * (phi - p['phi23']))))
#
#         if any([p[symbol] != 0. for symbol in ('C30', 'C32', 'phi32', 'C34', 'phi34')]):
#             array += (1 / 4 * alpha2 ** 2 *
#                       (p['C30'] +
#                        p['C32'] * xp.cos(2 * (phi - p['phi32'])) +
#                        p['C34'] * xp.cos(4 * (phi - p['phi34']))))
#
#         if any([p[symbol] != 0. for symbol in ('C41', 'phi41', 'C43', 'phi43', 'C45', 'phi41')]):
#             array += (1 / 5 * alpha2 ** 2 * alpha *
#                       (p['C41'] * xp.cos((phi - p['phi41'])) +
#                        p['C43'] * xp.cos(3 * (phi - p['phi43'])) +
#                        p['C45'] * xp.cos(5 * (phi - p['phi45']))))
#
#         if any([p[symbol] != 0. for symbol in ('C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')]):
#             array += (1 / 6 * alpha2 ** 3 *
#                       (p['C50'] +
#                        p['C52'] * xp.cos(2 * (phi - p['phi52'])) +
#                        p['C54'] * xp.cos(4 * (phi - p['phi54'])) +
#                        p['C56'] * xp.cos(6 * (phi - p['phi56']))))
#
#         array = np.float32(2 * xp.pi / self.wavelength) * array
#         return array
#
#     def evaluate_aberrations(self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> \
#             Union[float, np.ndarray]:
#
#         return complex_exponential(-self.evaluate_chi(alpha, phi))
#
#     def evaluate(self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         array = self.evaluate_aberrations(alpha, phi)
#
#         if self.semiangle_cutoff < np.inf:
#             array *= self.evaluate_aperture(alpha)
#
#         if self.focal_spread > 0.:
#             array *= self.evaluate_temporal_envelope(alpha)
#
#         if self.angular_spread > 0.:
#             array *= self.evaluate_spatial_envelope(alpha, phi)
#
#         if self.gaussian_spread > 0.:
#             array *= self.evaluate_gaussian_envelope(alpha)
#
#         xp = get_array_module(array)
#
#         if self.normalize == 'intensity':
#             array /= xp.sqrt(abs2(array).sum((-2, -1), keepdims=True))
#
#         elif self.normalize == 'amplitude':
#             array /= xp.abs(array).sum((-2, -1), keepdims=True)
#
#         elif not self.normalize == 'values':
#             raise RuntimeError()
#
#         return array
#
#     def evaluate_on_grid(self,
#                          gpts: Union[int, Tuple[int, int]] = None,
#                          extent: Union[float, Tuple[float, float]] = None,
#                          sampling: Union[float, Tuple[float, float]] = None,
#                          device: str = 'cpu') -> np.ndarray:
#
#         xp = get_array_module(device)
#         grid = Grid(gpts=gpts, extent=extent, sampling=sampling)
#         alpha, phi = polar_spatial_frequencies(grid.gpts, grid.sampling, xp=xp)
#
#         if self.ensemble_shape:
#             array = xp.zeros(self.ensemble_shape + grid.gpts, dtype=xp.complex64)
#             for i, (weight, ctf) in enumerate(self._generate_ctf_distribution()):
#                 array[np.unravel_index(i, self.ensemble_shape)] = weight * ctf.evaluate(alpha * self.wavelength, phi)
#         else:
#             array = self.evaluate(alpha * self.wavelength, phi)
#
#         return array
#
#     @property
#     def _distribution_parameters(self) -> Dict:
#         parameter_series = {}
#         for parameter_name, parameter in self.parameters.items():
#             if hasattr(parameter, 'divide'):
#                 parameter_series[parameter_name] = parameter
#         return parameter_series
#
#     def _generate_ctf_distribution(self):
#         values = tuple(distribution.values for distribution in self._distribution_parameters.values())
#         weights = tuple(distribution.weights for distribution in self._distribution_parameters.values())
#         xp = get_array_module(weights[0])
#
#         keys = self._distribution_parameters.keys()
#         for value, weight in zip(itertools.product(*values), itertools.product(*weights)):
#             d = self._copy_as_dict()
#             d['parameters'].update(dict(zip(keys, value)))
#
#             weight = weight if len(weight) > 1 else weight[0]
#             yield xp.prod(weight), self.__class__(**d)
#
#     @property
#     def ensemble_axes_metadata(self):
#         axes_metadata = []
#         for parameter_name, parameter in self._distribution_parameters.items():
#             axes_metadata += [ParameterSeriesAxis(label=parameter_name,
#                                                   values=tuple(parameter.values),
#                                                   units='Å',
#                                                   _ensemble_mean=parameter.ensemble_mean)]
#         return axes_metadata
#
#     @property
#     def default_ensemble_chunks(self):
#         return ('auto',) * len(self.ensemble_shape)
#
#     @property
#     def ensemble_shape(self):
#         return tuple(map(sum, tuple(block.shape for block in self._distribution_parameters.values())))
#
#     def ensemble_partial(self):
#         def ctf(*args, keys, **kwargs):
#
#             assert len(args) == len(keys)
#
#             for key, arg in zip(keys, args):
#                 kwargs['parameters'][key] = arg.item()
#
#             arr = np.zeros((1,) * len(keys), dtype=object)
#             arr[0] = CTF(**kwargs)
#             return arr
#
#         kwargs = self._copy_as_dict()
#         parameter_names = ()
#         for parameter_name in self._distribution_parameters.keys():
#             del kwargs['parameters'][parameter_name]
#             parameter_names += (parameter_name,)
#
#         return partial(ctf, keys=parameter_names, **kwargs)
#
#     def ensemble_blocks(self, max_batch: int = None, chunks=None):
#         if chunks is None:
#             chunks = self.default_ensemble_chunks
#
#         chunks = validate_chunks(self.ensemble_shape, chunks, limit=max_batch)
#
#         blocks = ()
#         for parameter, n in zip(self._distribution_parameters.values(), chunks):
#             blocks += (parameter.divide(n, lazy=True),)
#
#         return blocks
#
#     def profiles(self, max_semiangle: float = None, phi: float = 0., units='mrad'):
#         if max_semiangle is None:
#             if self._semiangle_cutoff == np.inf:
#                 max_semiangle = 50
#             else:
#                 max_semiangle = self._semiangle_cutoff * 1.6
#
#         sampling = max_semiangle / 1000. / 1000.
#         alpha = np.arange(0, max_semiangle / 1000., sampling)
#
#         aberrations = self.evaluate_aberrations(alpha, phi)
#         aperture = self.evaluate_aperture(alpha)
#         temporal_envelope = self.evaluate_temporal_envelope(alpha)
#         spatial_envelope = self.evaluate_spatial_envelope(alpha, phi)
#         gaussian_envelope = self.evaluate_gaussian_envelope(alpha)
#         envelope = aperture * temporal_envelope * spatial_envelope * gaussian_envelope
#
#         sampling = alpha[1] / energy2wavelength(self.energy)
#
#         profiles = {}
#         profiles['ctf'] = RadialFourierSpaceLineProfiles(-aberrations.imag * envelope,
#                                                          sampling=sampling,
#                                                          energy=self.energy)
#         profiles['aperture'] = RadialFourierSpaceLineProfiles(aperture, sampling=sampling, energy=self.energy)
#         profiles['envelope'] = RadialFourierSpaceLineProfiles(envelope, sampling=sampling, energy=self.energy)
#         profiles['temporal_envelope'] = RadialFourierSpaceLineProfiles(temporal_envelope, sampling=sampling,
#                                                                        energy=self.energy)
#         profiles['spatial_envelope'] = RadialFourierSpaceLineProfiles(spatial_envelope, sampling=sampling,
#                                                                       energy=self.energy)
#         profiles['gaussian_envelope'] = RadialFourierSpaceLineProfiles(gaussian_envelope, sampling=sampling,
#                                                                        energy=self.energy)
#         return profiles
#
#     def apply(self, waves: 'Waves', fourier_space_out: bool = False):
#         kernel = self.evaluate_on_grid(extent=waves.extent,
#                                        gpts=waves.gpts,
#                                        sampling=waves.sampling,
#                                        device=waves.device)
#
#         waves = waves.ensure_fourier_space()
#         array = waves.array[(None,) * self.ensemble_dims] * kernel
#
#         if not fourier_space_out:
#             array = fft2(array, overwrite_x=False)
#
#         d = waves._copy_as_dict(copy_array=False)
#         d['fourier_space'] = fourier_space_out
#         d['array'] = array
#         d['extra_axes_metadata'] = self.ensemble_axes_metadata + d['extra_axes_metadata']
#         return waves.__class__(**d)
#
#     def show(self,
#              max_semiangle: float = None,
#              phi: float = 0,
#              ax: Axes = None,
#              angular_units: bool = True,
#              legend: bool = True, **kwargs):
#         """
#         Show the contrast transfer function.
#
#         Parameters
#         ----------
#         max_semiangle: float
#             Maximum semiangle to display in the plot.
#         ax: matplotlib Axes, optional
#             If given, the plot will be added to this matplotlib axes.
#         phi: float, optional
#             The contrast transfer function will be plotted along this angle. Default is 0.
#         n: int, optional
#             Number of evaluation points to use in the plot. Default is 1000.
#         title: str, optional
#             The title of the plot. Default is 'None'.
#         kwargs:
#             Additional keyword arguments for the line plots.
#         """
#         import matplotlib.pyplot as plt
#
#         if ax is None:
#             ax = plt.subplot()
#
#         for key, profile in self.profiles(max_semiangle, phi).items():
#             if not np.all(profile.array == 1.):
#                 ax, lines = profile.show(ax=ax, label=key, angular_units=angular_units, **kwargs)
#
#         if legend:
#             ax.legend()
#
#         return ax
#
#     def _copy_as_dict(self):
#         d = {'semiangle_cutoff': self.semiangle_cutoff,
#              'focal_spread': self.focal_spread,
#              'angular_spread': self.angular_spread,
#              'gaussian_spread': self.gaussian_spread,
#              'weight': self.weight,
#              'energy': self.energy,
#              'normalize': self.normalize,
#              'parameters': copy.copy(self._parameters)
#              }
#         return d
#
#     def copy(self):
#         new_dict = self._copy_as_dict()
#         return self.__class__(**new_dict)


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
