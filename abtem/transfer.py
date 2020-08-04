from collections import defaultdict
from typing import Mapping

import numpy as np

from abtem.bases import HasAcceleratorMixin, Accelerator, watched_method, watched_property, Event

from abtem.device import get_array_module, get_device_function
from abtem.utils import energy2wavelength

polar_symbols = ('C10', 'C12', 'phi12',
                 'C21', 'phi21', 'C23', 'phi23',
                 'C30', 'C32', 'phi32', 'C34', 'phi34',
                 'C41', 'phi41', 'C43', 'phi43', 'C45', 'phi45',
                 'C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')

polar_aliases = {'defocus': 'C10', 'astigmatism': 'C12', 'astigmatism_angle': 'phi12',
                 'coma': 'C21', 'coma_angle': 'phi21',
                 'Cs': 'C30',
                 'C5': 'C50'}


class CTF(HasAcceleratorMixin):

    def __init__(self, semiangle_cutoff: float = np.inf, rolloff: float = 0., focal_spread: float = 0.,
                 angular_spread: float = 0., gaussian_spread: float = 0., energy: float = None,
                 parameters: Mapping[str, float] = None, **kwargs):

        for key in kwargs.keys():
            if ((key not in polar_symbols) and (key not in polar_aliases.keys())):
                raise ValueError('{} not a recognized parameter'.format(key))

        self.changed = Event()

        self._accelerator = Accelerator(energy=energy)
        self._semiangle_cutoff = semiangle_cutoff
        self._rolloff = rolloff
        self._focal_spread = focal_spread
        self._angular_spread = angular_spread
        self._gaussian_spread = gaussian_spread
        self._parameters = dict(zip(polar_symbols, [0.] * len(polar_symbols)))

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)
        self.set_parameters(parameters)

        def parametrization_property(key):

            def getter(self):
                return self._parameters[key]

            def setter(self, value):
                old = getattr(self, key)
                self._parameters[key] = value
                self.changed.notify(**{'notifier': self, 'property_name': key, 'change': old != value})

            return property(getter, setter)

        for symbol in polar_symbols:
            setattr(self.__class__, symbol, parametrization_property(symbol))

        for key, value in polar_aliases.items():
            if key != 'defocus':
                setattr(self.__class__, key, parametrization_property(value))

    @property
    def parameters(self):
        return self._parameters

    @property
    def defocus(self) -> float:
        return - self._parameters['C10']

    @defocus.setter
    def defocus(self, value: float):
        self.C10 = -value

    @property
    def semiangle_cutoff(self) -> float:
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    @watched_property('changed')
    def semiangle_cutoff(self, value: float):
        self._semiangle_cutoff = value

    @property
    def rolloff(self) -> float:
        return self._rolloff

    @rolloff.setter
    @watched_property('changed')
    def rolloff(self, value: float):
        self._rolloff = value

    @property
    def focal_spread(self) -> float:
        return self._focal_spread

    @focal_spread.setter
    @watched_property('changed')
    def focal_spread(self, value: float):
        self._focal_spread = value

    @property
    def angular_spread(self) -> float:
        return self._angular_spread

    @angular_spread.setter
    @watched_property('changed')
    def angular_spread(self, value: float):
        self._angular_spread = value

    @property
    def gaussian_spread(self) -> float:
        return self._gaussian_spread

    @gaussian_spread.setter
    @watched_property('changed')
    def gaussian_spread(self, value: float):
        self._gaussian_spread = value

    @watched_method('changed')
    def set_parameters(self, parameters):
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

    def evaluate_aperture(self, alpha) -> np.ndarray:
        xp = get_array_module(alpha)
        semiangle_cutoff = self.semiangle_cutoff / 1000.
        if self.rolloff > 0.:
            rolloff = self.rolloff * semiangle_cutoff
            array = .5 * (1 + xp.cos(np.pi * (alpha - semiangle_cutoff + rolloff) / rolloff))
            array[alpha > semiangle_cutoff] = 0.
            array = xp.where(alpha > semiangle_cutoff - rolloff, array, xp.ones_like(alpha, dtype=xp.float32))
        else:
            array = xp.array(alpha < semiangle_cutoff).astype(xp.float32)
        return array

    def evaluate_temporal_envelope(self, alpha: np.ndarray) -> np.ndarray:
        xp = get_array_module(alpha)
        return xp.exp(- (.5 * xp.pi / self.wavelength * self.focal_spread * alpha ** 2) ** 2).astype(xp.float32)

    def evaluate_gaussian_envelope(self, alpha: np.ndarray) -> np.ndarray:
        xp = get_array_module(alpha)
        return xp.exp(- .5 * self.gaussian_spread ** 2 * alpha ** 2 / self.wavelength ** 2)

    def evaluate_spatial_envelope(self, alpha, phi):
        xp = get_array_module(alpha)
        p = self.parameters
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

        return xp.exp(-xp.sign(self.angular_spread) * (self.angular_spread / 1000. / 2) ** 2 *
                      (dchi_dk ** 2 + dchi_dphi ** 2))

    def evaluate_chi(self, alpha, phi) -> np.ndarray:
        """
        Calculates the polar expansion of the phase error up to 5th order.

        See Eq. 2.22 in ref [1].

        Parameters
        ----------
        alpha : numpy.ndarray
            Angle between the scattered electrons and the optical axis [mrad].
        phi : numpy.ndarray
            Angle around the optical axis of the scattered electrons [mrad].
        wavelength : float
            Relativistic wavelength of wavefunction [Å].
        parameters : Mapping[str, float]
            Mapping from Cnn, phinn coefficients to their corresponding values. See parameter `parameters` in class CTFBase.

        Returns
        -------

        References
        ----------
        .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy (2nd ed.). Springer.

        """
        xp = get_array_module(alpha)
        p = self.parameters

        alpha2 = alpha ** 2

        array = xp.zeros(alpha.shape, dtype=np.float32)
        if any([p[symbol] != 0. for symbol in ('C10', 'C12', 'phi12')]):
            array += (1 / 2 * alpha2 *
                      (p['C10'] +
                       p['C12'] * xp.cos(2 * (phi - p['phi12']))))

        if any([p[symbol] != 0. for symbol in ('C21', 'phi21', 'C23', 'phi23')]):
            array += (1 / 3 * alpha2 * alpha *
                      (p['C21'] * xp.cos(phi - p['phi21']) +
                       p['C23'] * xp.cos(3 * (phi - p['phi23']))))

        if any([p[symbol] != 0. for symbol in ('C30', 'C32', 'phi32', 'C34', 'phi34')]):
            array += (1 / 4 * alpha2 ** 2 *
                      (p['C30'] +
                       p['C32'] * xp.cos(2 * (phi - p['phi32'])) +
                       p['C34'] * xp.cos(4 * (phi - p['phi34']))))

        if any([p[symbol] != 0. for symbol in ('C41', 'phi41', 'C43', 'phi43', 'C45', 'phi41')]):
            array += (1 / 5 * alpha2 ** 2 * alpha *
                      (p['C41'] * xp.cos((phi - p['phi41'])) +
                       p['C43'] * xp.cos(3 * (phi - p['phi43'])) +
                       p['C45'] * xp.cos(5 * (phi - p['phi45']))))

        if any([p[symbol] != 0. for symbol in ('C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')]):
            array += (1 / 6 * alpha2 ** 3 *
                      (p['C50'] +
                       p['C52'] * xp.cos(2 * (phi - p['phi52'])) +
                       p['C54'] * xp.cos(4 * (phi - p['phi54'])) +
                       p['C56'] * xp.cos(6 * (phi - p['phi56']))))

        array = 2 * xp.pi / self.wavelength * array
        return array

    def evaluate_aberrations(self, alpha, phi) -> np.ndarray:
        xp = get_array_module(alpha)
        complex_exponential = get_device_function(xp, 'complex_exponential')
        return complex_exponential(-self.evaluate_chi(alpha, phi))

    def evaluate(self, alpha, phi):
        alpha = np.array(alpha)
        phi = np.array(phi)

        array = self.evaluate_aberrations(alpha, phi)

        if self.semiangle_cutoff < np.inf:
            array *= self.evaluate_aperture(alpha)

        if self.focal_spread > 0.:
            array *= self.evaluate_temporal_envelope(alpha)

        if self.angular_spread > 0.:
            array *= self.evaluate_spatial_envelope(alpha, phi)

        if self.gaussian_spread > 0.:
            array *= self.evaluate_gaussian_envelope(alpha)

        return array

    def show(self, semiangle_cutoff: float, ax=None, phi=0, n=1000, title=None, **kwargs):
        import matplotlib.pyplot as plt

        alpha = np.linspace(0, semiangle_cutoff, n)

        aberrations = self.evaluate_aberrations(alpha, phi)
        aperture = self.evaluate_aperture(alpha)
        temporal_envelope = self.evaluate_temporal_envelope(alpha)
        spatial_envelope = self.evaluate_spatial_envelope(alpha, phi)
        gaussian_envelope = self.evaluate_gaussian_envelope(alpha)
        envelope = aperture * temporal_envelope * spatial_envelope * gaussian_envelope

        if ax is None:
            ax = plt.subplot()

        ax.plot(alpha * 1000, aberrations.imag * envelope, label='CTF', **kwargs)

        if self.semiangle_cutoff < np.inf:
            ax.plot(alpha * 1000, aperture, label='Aperture', **kwargs)

        if self.focal_spread > 0.:
            ax.plot(alpha * 1000, temporal_envelope, label='Temporal envelope', **kwargs)

        if self.angular_spread > 0.:
            ax.plot(alpha * 1000, spatial_envelope, label='Spatial envelope', **kwargs)

        if self.gaussian_spread > 0.:
            ax.plot(alpha * 1000, gaussian_envelope, label='Gaussian envelope', **kwargs)

        if not np.allclose(envelope, 1.):
            ax.plot(alpha * 1000, envelope, label='Product envelope', **kwargs)

        ax.set_xlabel('alpha [mrad.]')
        if title is not None:
            ax.set_title(title)
        ax.legend()

    def copy(self):
        parameters = self.parameters.copy()
        return self.__class__(semiangle_cutoff=self.semiangle_cutoff,
                              rolloff=self.rolloff,
                              focal_spread=self.focal_spread,
                              angular_spread=self.angular_spread,
                              gaussian_spread=self.gaussian_spread,
                              energy=self.energy,
                              parameters=parameters)


def scherzer_defocus(Cs, energy):
    return np.sign(Cs) * np.sqrt(3 / 2 * np.abs(Cs) * energy2wavelength(energy))


def point_resolution(Cs, energy):
    return (energy2wavelength(energy) ** 3 * np.abs(Cs) / 6) ** (1 / 4)


def polar2cartesian(polar):
    polar = defaultdict(lambda: 0, polar)

    cartesian = {}
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
    cartesian = defaultdict(lambda: 0, cartesian)

    polar = {}
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
