import numpy as np

from abtem.bases import Energy, HasCache, notifying_property, Grid, SelfObservable, cached_method
from abtem.utils import complex_exponential, squared_norm, semiangles

polar_symbols = ('C10', 'C12', 'phi12',
                 'C21', 'phi21', 'C23', 'phi23',
                 'C30', 'C32', 'phi32', 'C34', 'phi34',
                 'C41', 'phi41', 'C43', 'phi43', 'C45', 'phi45',
                 'C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')

polar_aliases = {'defocus': 'C10', 'astigmatism': 'C12', 'astigmatism_angle': 'phi12',
                 'coma': 'C21', 'coma_angle': 'phi21',
                 'Cs': 'C30',
                 'C5': 'C50'}


def calculate_polar_chi(alpha, phi, parameters):
    array = np.zeros(alpha.shape)

    alpha2 = alpha ** 2

    if any([parameters[symbol] != 0. for symbol in ('C10', 'C12', 'phi12')]):
        array += (1 / 2. * alpha2 *
                  (parameters['C10'] +
                   parameters['C12'] * np.cos(2. * (phi - parameters['phi12']))))

    if any([parameters[symbol] != 0. for symbol in ('C21', 'phi21', 'C23', 'phi23')]):
        array += (1 / 3. * alpha2 * alpha *
                  (parameters['C21'] * np.cos(phi - parameters['phi21']) +
                   parameters['C23'] * np.cos(3. * (phi - parameters['phi23']))))

    if any([parameters[symbol] != 0. for symbol in ('C30', 'C32', 'phi32', 'C34', 'phi34')]):
        array += (1 / 4. * alpha2 ** 2 *
                  (parameters['C30'] +
                   parameters['C32'] * np.cos(2. * (phi - parameters['phi32'])) +
                   parameters['C34'] * np.cos(4. * (phi - parameters['phi34']))))

    if any([parameters[symbol] != 0. for symbol in ('C41', 'phi41', 'C43', 'phi43', 'C45', 'phi41')]):
        array += (1 / 5. * alpha2 ** 2 * alpha *
                  (parameters['C41'] * np.cos((phi - parameters['phi41'])) +
                   parameters['C43'] * np.cos(3. * (phi - parameters['phi43'])) +
                   parameters['C45'] * np.cos(5. * (phi - parameters['phi45']))))

    if any([parameters[symbol] != 0. for symbol in ('C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')]):
        array += (1 / 6. * alpha2 ** 3 *
                  (parameters['C50'] +
                   parameters['C52'] * np.cos(2. * (phi - parameters['phi52'])) +
                   parameters['C54'] * np.cos(4. * (phi - parameters['phi54'])) +
                   parameters['C56'] * np.cos(6. * (phi - parameters['phi56']))))

    return array


def calculate_polar_aberrations(alpha, phi, wavelength, parameters):
    tensor = complex_exponential(2 * np.pi / wavelength * calculate_polar_chi(alpha, phi, parameters))
    return tensor


def calculate_aperture(alpha, cutoff, rolloff):
    if rolloff > 0.:
        array = .5 * (1 + np.cos(np.pi * (alpha - cutoff) / rolloff))
        array *= alpha < (cutoff + rolloff)
        array = np.where(alpha > cutoff, array, np.ones_like(alpha))
    else:
        array = np.array(alpha < cutoff).astype(np.float)
    return array


def calculate_temporal_envelope(alpha, wavelength, focal_spread):
    array = np.exp(- (.5 * np.pi / wavelength * focal_spread * alpha ** 2) ** 2)
    return array


def parametrization_property(key):
    def getter(self):
        return self._parameters[key]

    def setter(self, value):
        old = getattr(self, key)
        self._parameters[key] = value
        change = old != value
        self.notify_observers({'name': key, 'old': old, 'new': value, 'change': change})

    return property(getter, setter)


class CTF(Energy, Grid, HasCache, SelfObservable):

    def __init__(self, cutoff=np.inf, rolloff=0., focal_spread=0., extent=None, gpts=None, sampling=None, energy=None,
                 parameters=None, **kwargs):

        self._cutoff = cutoff
        self._rolloff = rolloff
        self._focal_spread = focal_spread

        self._parameters = dict(zip(polar_symbols, [0.] * len(polar_symbols)))

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)

        self.set_parameters(parameters)

        for symbol in polar_symbols:
            setattr(self.__class__, symbol, parametrization_property(symbol))

        for key, value in polar_aliases.items():
            setattr(self.__class__, key, parametrization_property(value))

        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy)

    cutoff = notifying_property('_cutoff')
    rolloff = notifying_property('_rolloff')
    focal_spread = notifying_property('_focal_spread')

    def set_parameters(self, parameters, parametrization='polar'):
        if parametrization != 'polar':
            raise NotImplementedError()

        for symbol, value in parameters.items():
            if symbol in self._parameters.keys():
                self._parameters[symbol] = value

            elif symbol in polar_aliases.keys():
                self._parameters[polar_aliases[symbol]] = value

            else:
                raise RuntimeError('{}'.format(symbol))

        return parameters

    @cached_method(('extent', 'gpts', 'sampling', 'energy'))
    def get_alpha(self):
        self.check_is_grid_defined()
        self.check_is_energy_defined()
        return np.sqrt(squared_norm(*semiangles(self)))

    @cached_method(('extent', 'gpts', 'sampling', 'energy'))
    def get_phi(self):
        self.check_is_grid_defined()
        self.check_is_energy_defined()
        alpha_x, alpha_y = semiangles(self)
        phi = np.arctan2(alpha_x.reshape((-1, 1)), alpha_y.reshape((1, -1)))
        return phi

    @cached_method(('extent', 'gpts', 'sampling', 'energy', '_cutoff', '_rolloff'))
    def get_aperture(self):
        alpha = self.get_alpha()
        return calculate_aperture(alpha, self.cutoff, self.rolloff)

    @cached_method(('extent', 'gpts', 'sampling', 'energy', '_focal_spread'))
    def get_temporal_envelope(self):
        alpha = self.get_alpha()
        return calculate_temporal_envelope(alpha, self.wavelength, self.focal_spread)

    @cached_method(('extent', 'gpts', 'sampling', 'energy') + polar_symbols)
    def get_aberrations(self):
        alpha = self.get_alpha()
        phi = self.get_phi()
        return calculate_polar_aberrations(alpha, phi, self.wavelength, self._parameters)

    @cached_method('any')
    def get_ctf(self):
        array = self.get_aberrations()

        if self.cutoff < np.inf:
            array = array * self.get_aperture()

        if self.focal_spread > 0.:
            array = array * self.get_temporal_envelope()

        return array

    def get_array(self):
        return self.get_ctf()
