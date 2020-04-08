from collections import defaultdict
from typing import Mapping, Union, Sequence

import numpy as np
from abtem.bases import Energy, Cache, Grid, cached_method, ArrayWithGridAndEnergy, notify, semiangles
from abtem.config import DTYPE
from abtem.utils import complex_exponential, squared_norm, energy2wavelength

polar_symbols = ('C10', 'C12', 'phi12',
                 'C21', 'phi21', 'C23', 'phi23',
                 'C30', 'C32', 'phi32', 'C34', 'phi34',
                 'C41', 'phi41', 'C43', 'phi43', 'C45', 'phi45',
                 'C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')

polar_aliases = {'defocus': 'C10', 'astigmatism': 'C12', 'astigmatism_angle': 'phi12',
                 'coma': 'C21', 'coma_angle': 'phi21',
                 'Cs': 'C30',
                 'C5': 'C50'}


def calculate_symmetric_chi(alpha: np.ndarray, wavelength: float, parameters: Mapping[str, float]) -> np.ndarray:
    """
    Calculates the first three symmetric terms in the phase error expansion.

    See Eq. 2.6 in ref [1].

    Parameters
    ----------
    alpha : numpy.ndarray
        Angle between the scattered electrons and the optical axis.
    wavelength : float
        Relativistic wavelength of wavefunction.
    parameters : Mapping[str, float]
        Mapping from Cn0 coefficients to its corresponding value.
    Returns
    -------

    References
    ----------
    .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy (2nd ed.). Springer.

    """
    alpha2 = alpha ** 2
    return 2 * np.pi / wavelength * (1 / 2. * alpha2 * parameters['C10'] +
                                     1 / 4. * alpha2 ** 2 * parameters['C30'] +
                                     1 / 6. * alpha2 ** 3 * parameters['C50'])


def calculate_polar_chi(alpha: np.ndarray, phi: np.ndarray, wavelength: float,
                        parameters: Mapping[str, float]) -> np.ndarray:
    """
    Calculates the polar expansion of the phase error up to 5th order.

    See Eq. 2.22 in ref [1].

    Parameters
    ----------
    alpha : numpy.ndarray
        Angle between the scattered electrons and the optical axis.
    phi : numpy.ndarray
        Angle around the optical axis of the scattered electrons.
    wavelength : float
        Relativistic wavelength of wavefunction.
    parameters : Mapping[str, float]
        Mapping from Cnn, phinn coefficients to their corresponding values. See parameter `parameters` in class CTFBase.

    Returns
    -------

    References
    ----------
    .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy (2nd ed.). Springer.

    """
    alpha2 = alpha ** 2

    array = np.zeros(alpha.shape, dtype=DTYPE)
    if any([parameters[symbol] != 0. for symbol in ('C10', 'C12', 'phi12')]):
        array += (1 / 2 * alpha2 *
                  (parameters['C10'] +
                   parameters['C12'] * np.cos(2 * (phi - parameters['phi12']))))

    if any([parameters[symbol] != 0. for symbol in ('C21', 'phi21', 'C23', 'phi23')]):
        array += (1 / 3 * alpha2 * alpha *
                  (parameters['C21'] * np.cos(phi - parameters['phi21']) +
                   parameters['C23'] * np.cos(3 * (phi - parameters['phi23']))))

    if any([parameters[symbol] != 0. for symbol in ('C30', 'C32', 'phi32', 'C34', 'phi34')]):
        array += (1 / 4 * alpha2 ** 2 *
                  (parameters['C30'] +
                   parameters['C32'] * np.cos(2 * (phi - parameters['phi32'])) +
                   parameters['C34'] * np.cos(4 * (phi - parameters['phi34']))))

    if any([parameters[symbol] != 0. for symbol in ('C41', 'phi41', 'C43', 'phi43', 'C45', 'phi41')]):
        array += (1 / 5 * alpha2 ** 2 * alpha *
                  (parameters['C41'] * np.cos((phi - parameters['phi41'])) +
                   parameters['C43'] * np.cos(3 * (phi - parameters['phi43'])) +
                   parameters['C45'] * np.cos(5 * (phi - parameters['phi45']))))

    if any([parameters[symbol] != 0. for symbol in ('C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')]):
        array += (1 / 6 * alpha2 ** 3 *
                  (parameters['C50'] +
                   parameters['C52'] * np.cos(2 * (phi - parameters['phi52'])) +
                   parameters['C54'] * np.cos(4 * (phi - parameters['phi54'])) +
                   parameters['C56'] * np.cos(6 * (phi - parameters['phi56']))))

    array = 2 * np.pi / wavelength * array
    return array


def calculate_symmetric_aberrations(alpha: np.ndarray, wavelength: float,
                                    parameters: Mapping[str, float]) -> np.ndarray:
    return complex_exponential(-calculate_symmetric_chi(alpha, wavelength, parameters))


def calculate_polar_aberrations(alpha: np.ndarray, phi: np.ndarray, wavelength: float,
                                parameters: Mapping[str, float]) -> np.ndarray:
    return complex_exponential(-calculate_polar_chi(alpha, phi, wavelength, parameters))


def calculate_aperture(alpha: np.ndarray, cutoff: float, rolloff: float) -> np.ndarray:
    if rolloff > 0.:
        rolloff *= cutoff
        array = .5 * (1 + np.cos(np.pi * (alpha - cutoff + rolloff) / rolloff))
        array[alpha > cutoff] = 0.
        array = np.where(alpha > cutoff - rolloff, array, np.ones_like(alpha, dtype=DTYPE))
    else:
        array = np.array(alpha < cutoff).astype(DTYPE)
    return array


def calculate_temporal_envelope(alpha: np.ndarray, wavelength: float, focal_spread: float) -> np.ndarray:
    return DTYPE(np.exp(- (.5 * np.pi / wavelength * focal_spread * alpha ** 2) ** 2))


def calculate_gaussian_envelope(alpha: np.ndarray, wavelength: float, gaussian_spread: float) -> np.ndarray:
    return DTYPE(np.exp(- .5 * gaussian_spread ** 2 * alpha ** 2 / wavelength ** 2))


def calculate_spatial_envelope(alpha, phi, wavelength, angular_spread, parameters):
    dchi_dk = 2 * np.pi / wavelength * (
            (parameters['C12'] * np.cos(2. * (phi - parameters['phi12'])) + parameters['C10']) * alpha +
            (parameters['C23'] * np.cos(3. * (phi - parameters['phi23'])) +
             parameters['C21'] * np.cos(1. * (phi - parameters['phi21']))) * alpha ** 2 +
            (parameters['C34'] * np.cos(4. * (phi - parameters['phi34'])) +
             parameters['C32'] * np.cos(2. * (phi - parameters['phi32'])) + parameters['C30']) * alpha ** 3 +
            (parameters['C45'] * np.cos(5. * (phi - parameters['phi45'])) +
             parameters['C43'] * np.cos(3. * (phi - parameters['phi43'])) +
             parameters['C41'] * np.cos(1. * (phi - parameters['phi41']))) * alpha ** 4 +
            (parameters['C56'] * np.cos(6. * (phi - parameters['phi56'])) +
             parameters['C54'] * np.cos(4. * (phi - parameters['phi54'])) +
             parameters['C52'] * np.cos(2. * (phi - parameters['phi52'])) + parameters['C50']) * alpha ** 5)

    dchi_dphi = -2 * np.pi / wavelength * (
            1 / 2. * (2. * parameters['C12'] * np.sin(2. * (phi - parameters['phi12']))) * alpha +
            1 / 3. * (3. * parameters['C23'] * np.sin(3. * (phi - parameters['phi23'])) +
                      1. * parameters['C21'] * np.sin(1. * (phi - parameters['phi21']))) * alpha ** 2 +
            1 / 4. * (4. * parameters['C34'] * np.sin(4. * (phi - parameters['phi34'])) +
                      2. * parameters['C32'] * np.sin(2. * (phi - parameters['phi32']))) * alpha ** 3 +
            1 / 5. * (5. * parameters['C45'] * np.sin(5. * (phi - parameters['phi45'])) +
                      3. * parameters['C43'] * np.sin(3. * (phi - parameters['phi43'])) +
                      1. * parameters['C41'] * np.sin(1. * (phi - parameters['phi41']))) * alpha ** 4 +
            1 / 6. * (6. * parameters['C56'] * np.sin(6. * (phi - parameters['phi56'])) +
                      4. * parameters['C54'] * np.sin(4. * (phi - parameters['phi54'])) +
                      2. * parameters['C52'] * np.sin(2. * (phi - parameters['phi52']))) * alpha ** 5)

    return np.exp(-np.sign(angular_spread) * (angular_spread / 2) ** 2 * (dchi_dk ** 2 + dchi_dphi ** 2))


def _parametrization_property(key):
    def getter(self):
        return self._parameters[key]

    def setter(self, value):
        old = getattr(self, key)
        self._parameters[key] = value
        self.notify_observers({'notifier': key, 'change': old != value})

    return property(getter, setter)


class CTFBase(Energy):

    def __init__(self, semiangle_cutoff: float = np.inf, rolloff: float = 0., focal_spread: float = 0.,
                 angular_spread: float = 0., gaussian_spread: float = 0., energy: float = None,
                 parameters: Mapping[str, float] = None, **kwargs):

        self._semiangle_cutoff = DTYPE(semiangle_cutoff)
        self._rolloff = DTYPE(rolloff)
        self._focal_spread = DTYPE(focal_spread)
        self._angular_spread = DTYPE(angular_spread)
        self._gaussian_spread = DTYPE(gaussian_spread)
        self._parameters = dict(zip(polar_symbols, [0.] * len(polar_symbols)))

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)

        self.set_parameters(parameters)

        for symbol in polar_symbols:
            setattr(self.__class__, symbol, _parametrization_property(symbol))
            kwargs.pop(symbol, None)

        for key, value in polar_aliases.items():
            if key != 'defocus':
                setattr(self.__class__, key, _parametrization_property(value))
            kwargs.pop(key, None)

        super().__init__(energy=energy, **kwargs)

    @property
    def parameters(self):
        return self._parameters

    @property
    def defocus(self) -> float:
        return - self._parameters['C10']

    @defocus.setter
    @notify
    def defocus(self, value: float):
        self._parameters['C10'] = DTYPE(-value)

    @property
    def semiangle_cutoff(self) -> float:
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    @notify
    def semiangle_cutoff(self, value: float):
        self._semiangle_cutoff = DTYPE(value)

    @property
    def rolloff(self) -> float:
        return self._rolloff

    @rolloff.setter
    @notify
    def rolloff(self, value: float):
        self._rolloff = DTYPE(value)

    @property
    def focal_spread(self) -> float:
        return self._focal_spread

    @focal_spread.setter
    @notify
    def focal_spread(self, value: float):
        self._focal_spread = DTYPE(value)

    @property
    def angular_spread(self) -> float:
        return self._angular_spread

    @angular_spread.setter
    @notify
    def angular_spread(self, value: float):
        self._angular_spread = DTYPE(value)

    @property
    def gaussian_spread(self) -> float:
        return self._gaussian_spread

    @gaussian_spread.setter
    @notify
    def gaussian_spread(self, value: float):
        self._gaussian_spread = DTYPE(value)

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

    def get_alpha(self):
        raise NotImplementedError()

    def get_phi(self):
        raise NotImplementedError()

    def get_aperture(self):
        return calculate_aperture(self.get_alpha(), self.semiangle_cutoff, self.rolloff)

    def get_temporal_envelope(self):
        return calculate_temporal_envelope(self.get_alpha(), self.wavelength, self.focal_spread)

    def get_spatial_envelope(self):
        return calculate_spatial_envelope(self.get_alpha(), self.get_phi(), self.wavelength, self.angular_spread,
                                          self.parameters)

    def get_gaussian_envelope(self):
        return calculate_gaussian_envelope(self.get_alpha(), self.wavelength, self.gaussian_spread)

    def get_aberrations(self):
        return calculate_polar_aberrations(self.get_alpha(), self.get_phi(), self.wavelength, self._parameters)

    def get_array(self):

        array = self.get_aberrations()

        if self.semiangle_cutoff < np.inf:
            array = array * self.get_aperture()

        if self.focal_spread > 0.:
            array = array * self.get_temporal_envelope()

        if self.angular_spread > 0.:
            array = array * self.get_spatial_envelope()

        if self.gaussian_spread > 0.:
            array = array * self.get_gaussian_envelope()

        return array[None]

    # def copy(self):
    #     parameters = self._parameters
    #
    #     self.__class__()


def scherzer_defocus(Cs, energy):
    return 1.2 * np.sign(Cs) * np.sqrt(np.abs(Cs) * energy2wavelength(energy))


class CTF(Grid, Cache, CTFBase):

    def __init__(self, semiangle_cutoff: float = np.inf, rolloff: float = 0., focal_spread: float = 0.,
                 angular_spread: float = 0., gaussian_spread: float = 0., parameters: Mapping[str, float] = None,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None,
                 **kwargs):
        """
        Contrast Transfer Function object.

        Parameters
        ----------
        cutoff : float
            Default is infinite.
        rolloff : float
            Softens the cutoff. A value of 0 gives a hard cutoff, while 1 gives the softest possible cutoff.
        focal_spread : float
            The spread

            due to, among other factors, chromatic aberrations and lens current instabilities [rad.].
            Default is 0.
        angular_spread :
            Default is 0.
        parameters :
            `C10`, `C12`, `phi12`,
            `C21`, `phi21`, `C23`, `phi23`,
            `C30`, `C32`, `phi32`, `C34`, `phi34`,
            `C41`, `phi41`, `C43`, `phi43`, `C45`, `phi45`,
            `C50`, `C52`, `phi52`, `C54`, `phi54`, `C56`, `phi56`
        extent : sequence of float, float, optional
            Lateral extent of wavefunctions [Å].
        gpts : sequence of int, int, optional
            Number of grid points describing the wavefunctions
        sampling : sequence of float, float, optional
            Lateral sampling of wavefunctions [1 / Å].
        energy : float, optional
            Waves energy [eV].
        kwargs :
            Provide the aberration coefficients as keyword arguments.
        """

        super().__init__(semiangle_cutoff=semiangle_cutoff, rolloff=rolloff, focal_spread=focal_spread,
                         angular_spread=angular_spread, gaussian_spread=gaussian_spread, extent=extent, gpts=gpts,
                         sampling=sampling,
                         energy=energy, parameters=parameters, **kwargs)
        self.register_observer(self)

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

    @cached_method(('extent', 'gpts', 'sampling', 'energy', 'semiangle_cutoff', 'rolloff'))
    def get_aperture(self):
        return super().get_aperture()

    @cached_method(('extent', 'gpts', 'sampling', 'energy', 'focal_spread'))
    def get_temporal_envelope(self):
        return super().get_temporal_envelope()

    @cached_method(('extent', 'gpts', 'sampling', 'energy', 'angular_spread'))
    def get_spatial_envelope(self):
        return super().get_spatial_envelope()

    @cached_method(('extent', 'gpts', 'sampling', 'energy', 'gaussian_spread'))
    def get_gaussian_envelope(self):
        return super().get_spatial_envelope()

    @cached_method(('extent', 'gpts', 'sampling', 'energy', 'defocus') + polar_symbols)
    def get_aberrations(self):
        return super().get_aberrations()

    @cached_method()
    def get_array(self):
        return super().get_array()

    def build(self):
        return ArrayWithGridAndEnergy(np.fft.fftshift(self.get_array(), axes=(1, 2)), spatial_dimensions=2,
                                      extent=self.extent, energy=self.energy)


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
