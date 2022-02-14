"""Module to describe the contrast transfer function."""
from collections import defaultdict
from typing import Mapping, Union, TYPE_CHECKING, Dict

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.energy import Accelerator, HasAcceleratorMixin, energy2wavelength
from abtem.core.grid import Grid, polar_spatial_frequencies
from abtem.measure.measure import LineProfiles

if TYPE_CHECKING:
    from abtem.waves.waves import Waves

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


class RadialFourierSpaceLineProfiles(LineProfiles):

    def __init__(self,
                 array,
                 sampling,
                 energy,
                 extra_axes_metadata=None,
                 metadata=None):
        self._energy = energy

        super().__init__(array=array, start=(0., 0.), end=(0., array.shape[-1] * sampling), sampling=sampling,
                         extra_axes_metadata=extra_axes_metadata, metadata=metadata)

    def show(self, ax=None, title='', angular_units=True, **kwargs):
        if ax is None:
            ax = plt.subplot()

        if angular_units:
            x = np.linspace(0., len(self.array) * self.sampling * 1000. * energy2wavelength(self._energy),
                            len(self.array))
        else:
            x = np.linspace(0., len(self.array) * self.sampling, len(self.array))

        p = ax.plot(x, self.array, **kwargs)
        ax.set_xlabel('Scattering angle [mrad]')
        ax.set_title(title)
        return ax, p


class CTF(HasAcceleratorMixin):
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
    semiangle_cutoff: float
        The semiangle cutoff describes the sharp Fourier space cutoff due to the objective aperture [mrad].
    rolloff: float
        Tapers the cutoff edge over the given angular range [mrad].
    focal_spread: float
        The 1/e width of the focal spread due to chromatic aberration and lens current instability [Å].
    angular_spread: float
        The 1/e width of the angular deviations due to source size [Å].
    gaussian_spread:
        The 1/e width image deflections due to vibrations and thermal magnetic noise [Å].
    energy: float
        The electron energy of the wave functions this contrast transfer function will be applied to [eV].
    parameters: dict
        Mapping from aberration symbols to their corresponding values. All aberration magnitudes should be given in Å
        and angles should be given in radians.
    kwargs:
        Provide the aberration coefficients as keyword arguments.

    References
    ----------
    .. [1] Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy (2nd ed.). Springer.

    """

    def __init__(self,
                 semiangle_cutoff: float = 30.,
                 rolloff: float = 0.,
                 focal_spread: float = 0.,
                 angular_spread: float = 0.,
                 gaussian_spread: float = 0.,
                 energy: float = None,
                 parameters: Mapping[str, float] = None,
                 **kwargs):

        for key in kwargs.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()):
                raise ValueError('{} not a recognized parameter'.format(key))

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

            return property(getter, setter)

        for symbol in polar_symbols:
            setattr(self.__class__, symbol, parametrization_property(symbol))

        for key, value in polar_aliases.items():
            if key != 'defocus':
                setattr(self.__class__, key, parametrization_property(value))

    @property
    def nyquist_sampling(self) -> float:
        return 1 / (4 * self.semiangle_cutoff / self.wavelength * 1e-3)

    @property
    def parameters(self) -> Dict[str, float]:
        """The parameters."""
        return self._parameters

    @property
    def defocus(self) -> float:
        """The defocus [Å]."""
        return - self._parameters['C10']

    @defocus.setter
    def defocus(self, value: float):
        self.C10 = -value

    @property
    def semiangle_cutoff(self) -> float:
        """The semi-angle cutoff [mrad]."""
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value: float):
        self._semiangle_cutoff = value

    @property
    def rolloff(self) -> float:
        """The fraction of soft tapering of the cutoff."""
        return self._rolloff

    @rolloff.setter
    def rolloff(self, value: float):
        self._rolloff = value

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

    @property
    def gaussian_spread(self) -> float:
        """The Gaussian spread [Å]."""
        return self._gaussian_spread

    @gaussian_spread.setter
    def gaussian_spread(self, value: float):
        self._gaussian_spread = value

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

    def evaluate_aperture(self, alpha: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)
        semiangle_cutoff = self.semiangle_cutoff / 1000

        if self.semiangle_cutoff == xp.inf:
            return xp.ones_like(alpha)

        if self.rolloff > 0.:
            rolloff = self.rolloff / 1000.
            array = .5 * (1 + xp.cos(np.pi * (alpha - semiangle_cutoff + rolloff) / rolloff))
            array[alpha > semiangle_cutoff] = 0.
            array = xp.where(alpha > semiangle_cutoff - rolloff, array, xp.ones_like(alpha, dtype=xp.float32))
        else:
            array = xp.array(alpha <= semiangle_cutoff).astype(xp.float32)

        return array

    def evaluate_temporal_envelope(self, alpha: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)
        return xp.exp(- (.5 * xp.pi / self.wavelength * self.focal_spread * alpha ** 2) ** 2).astype(xp.float32)

    def evaluate_gaussian_envelope(self, alpha: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)
        return xp.exp(- .5 * self.gaussian_spread ** 2 * alpha ** 2 / self.wavelength ** 2)

    def evaluate_spatial_envelope(self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> \
            Union[float, np.ndarray]:
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

        return xp.exp(-xp.sign(self.angular_spread) * (self.angular_spread / 2 / 1000) ** 2 *
                      (dchi_dk ** 2 + dchi_dphi ** 2))

    def evaluate_chi(self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)

        p = self.parameters

        alpha2 = alpha ** 2
        alpha = xp.array(alpha)

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

        array = np.float32(2 * xp.pi / self.wavelength) * array
        return array

    def evaluate_aberrations(self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> \
            Union[float, np.ndarray]:

        return complex_exponential(-self.evaluate_chi(alpha, phi))

    def evaluate(self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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

    def evaluate_on_grid(self, gpts=None, extent=None, sampling=None, xp=np):
        grid = Grid(gpts=gpts, extent=extent, sampling=sampling)
        alpha, phi = polar_spatial_frequencies(grid.gpts, grid.sampling, xp=xp)
        return self.evaluate(alpha * self.wavelength, phi)

    def profiles(self, max_semiangle: float = None, phi: float = 0., units='mrad'):
        if max_semiangle is None:
            if self._semiangle_cutoff == np.inf:
                max_semiangle = 50
            else:
                max_semiangle = self._semiangle_cutoff * 1.6

        sampling = max_semiangle / 1000. / 1000.
        alpha = np.arange(0, max_semiangle / 1000., sampling)

        aberrations = self.evaluate_aberrations(alpha, phi)
        aperture = self.evaluate_aperture(alpha)
        temporal_envelope = self.evaluate_temporal_envelope(alpha)
        spatial_envelope = self.evaluate_spatial_envelope(alpha, phi)
        gaussian_envelope = self.evaluate_gaussian_envelope(alpha)
        envelope = aperture * temporal_envelope * spatial_envelope * gaussian_envelope

        sampling = alpha[1] / energy2wavelength(self.energy)

        profiles = {}
        profiles['ctf'] = RadialFourierSpaceLineProfiles(-aberrations.imag * envelope,
                                                         sampling=sampling,
                                                         energy=self.energy)
        profiles['aperture'] = RadialFourierSpaceLineProfiles(aperture, sampling=sampling, energy=self.energy)
        profiles['envelope'] = RadialFourierSpaceLineProfiles(envelope, sampling=sampling, energy=self.energy)
        profiles['temporal_envelope'] = RadialFourierSpaceLineProfiles(temporal_envelope, sampling=sampling,
                                                                       energy=self.energy)
        profiles['spatial_envelope'] = RadialFourierSpaceLineProfiles(spatial_envelope, sampling=sampling,
                                                                      energy=self.energy)
        profiles['gaussian_envelope'] = RadialFourierSpaceLineProfiles(gaussian_envelope, sampling=sampling,
                                                                       energy=self.energy)
        return profiles

    def apply(self, waves: 'Waves'):
        return waves.apply_ctf(self)

    def show(self,
             max_semiangle: float = None,
             phi: float = 0,
             ax: Axes = None,
             angular_units: bool = True,
             legend: bool = True, **kwargs):
        """
        Show the contrast transfer function.

        Parameters
        ----------
        max_semiangle: float
            Maximum semiangle to display in the plot.
        ax: matplotlib Axes, optional
            If given, the plot will be added to this matplotlib axes.
        phi: float, optional
            The contrast transfer function will be plotted along this angle. Default is 0.
        n: int, optional
            Number of evaluation points to use in the plot. Default is 1000.
        title: str, optional
            The title of the plot. Default is 'None'.
        kwargs:
            Additional keyword arguments for the line plots.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.subplot()

        for key, profile in self.profiles(max_semiangle, phi).items():
            if not np.all(profile.array == 1.):
                ax, lines = profile.show(ax=ax, label=key, angular_units=angular_units, **kwargs)

        if legend:
            ax.legend()

        return ax

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
