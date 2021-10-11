"""Module to describe the contrast transfer function."""
from collections import defaultdict
from typing import Mapping, Union

import numpy as np
from abtem.base_classes import HasAcceleratorMixin, HasEventMixin, Accelerator, watched_method, watched_property, Event, \
    Grid
from abtem.device import get_array_module, get_device_function
from abtem.measure import Measurement, Calibration
from abtem.utils import energy2wavelength, spatial_frequencies, polar_coordinates

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


class CTF(HasAcceleratorMixin, HasEventMixin):
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
                 semiangle_cutoff: float = np.inf,
                 rolloff: float = 2,
                 focal_spread: float = 0.,
                 angular_spread: float = 0.,
                 gaussian_spread: float = 0.,
                 energy: float = None,
                 parameters: Mapping[str, float] = None,
                 aperture=None,
                 **kwargs):

        for key in kwargs.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()):
                raise ValueError('{} not a recognized parameter'.format(key))

        self._event = Event()

        self._accelerator = Accelerator(energy=energy)
        self._accelerator.observe(self.event.notify)

        self._semiangle_cutoff = semiangle_cutoff
        self._rolloff = rolloff
        self._focal_spread = focal_spread
        self._angular_spread = angular_spread
        self._gaussian_spread = gaussian_spread

        self._parameters = dict(zip(polar_symbols, [0.] * len(polar_symbols)))

        self._aperture = aperture

        if self._aperture is not None:
            self._aperture.accelerator.match(self)

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
                self.event.notify({'notifier': self, 'name': key, 'change': old != value})

            return property(getter, setter)

        for symbol in polar_symbols:
            setattr(self.__class__, symbol, parametrization_property(symbol))

        for key, value in polar_aliases.items():
            if key != 'defocus':
                setattr(self.__class__, key, parametrization_property(value))

    @property
    def nyquist_sampling(self):
        return 1 / (4 * self.semiangle_cutoff / self.wavelength * 1e-3)

    @property
    def parameters(self):
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
    @watched_property('_event')
    def semiangle_cutoff(self, value: float):
        self._semiangle_cutoff = value

    @property
    def rolloff(self) -> float:
        """The fraction of soft tapering of the cutoff."""
        return self._rolloff

    @rolloff.setter
    @watched_property('_event')
    def rolloff(self, value: float):
        self._rolloff = value

    @property
    def focal_spread(self) -> float:
        """The focal spread [Å]."""
        return self._focal_spread

    @focal_spread.setter
    @watched_property('_event')
    def focal_spread(self, value: float):
        """The angular spread [mrad]."""
        self._focal_spread = value

    @property
    def angular_spread(self) -> float:
        return self._angular_spread

    @angular_spread.setter
    @watched_property('_event')
    def angular_spread(self, value: float):
        self._angular_spread = value

    @property
    def gaussian_spread(self) -> float:
        """The Gaussian spread [Å]."""
        return self._gaussian_spread

    @gaussian_spread.setter
    @watched_property('_event')
    def gaussian_spread(self, value: float):
        self._gaussian_spread = value

    @watched_method('_event')
    def set_parameters(self, parameters: dict):
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

    def evaluate_aperture(self,
                          alpha: Union[float, np.ndarray],
                          phi: Union[float, np.ndarray] = None) -> Union[float, np.ndarray]:

        if self._aperture is not None:
            return self._aperture.evaluate(alpha, phi)

        xp = get_array_module(alpha)
        semiangle_cutoff = self.semiangle_cutoff / 1000

        if self.semiangle_cutoff == xp.inf:
            return xp.ones_like(alpha)

        if self.rolloff > 0.:
            rolloff = self.rolloff / 1000.  # * semiangle_cutoff
            array = .5 * (1 + xp.cos(np.pi * (alpha - semiangle_cutoff + rolloff) / rolloff))
            array[alpha > semiangle_cutoff] = 0.
            array = xp.where(alpha > semiangle_cutoff - rolloff, array, xp.ones_like(alpha, dtype=xp.float32))
        else:
            array = xp.array(alpha < semiangle_cutoff).astype(xp.float32)
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

        array = 2 * xp.pi / self.wavelength * array
        return array

    def evaluate_aberrations(self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> \
            Union[float, np.ndarray]:
        xp = get_array_module(alpha)
        complex_exponential = get_device_function(xp, 'complex_exponential')
        return complex_exponential(-self.evaluate_chi(alpha, phi))

    def evaluate(self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        array = self.evaluate_aberrations(alpha, phi)

        if (self.semiangle_cutoff < np.inf) or (self._aperture is not None):
            array *= self.evaluate_aperture(alpha, phi)

        if self.focal_spread > 0.:
            array *= self.evaluate_temporal_envelope(alpha)

        if self.angular_spread > 0.:
            array *= self.evaluate_spatial_envelope(alpha, phi)

        if self.gaussian_spread > 0.:
            array *= self.evaluate_gaussian_envelope(alpha)

        return array

    def _polar_coordinates(self, gpts=None, extent=None, sampling=None, xp=np):
        grid = Grid(gpts=gpts, extent=extent, sampling=sampling)

        gpts = grid.gpts
        sampling = grid.sampling

        kx, ky = spatial_frequencies(gpts, sampling)
        kx = kx.reshape((1, -1, 1))
        ky = ky.reshape((1, 1, -1))
        kx = xp.asarray(kx)
        ky = xp.asarray(ky)
        return polar_coordinates(xp.asarray(kx * self.wavelength), xp.asarray(ky * self.wavelength))

    def evaluate_on_grid(self, gpts=None, extent=None, sampling=None, xp=np):
        return self.evaluate(*self._polar_coordinates(gpts, extent, sampling, xp))

    def profiles(self, max_semiangle: float = None, phi: float = 0.):
        if max_semiangle is None:
            if self._semiangle_cutoff == np.inf:
                max_semiangle = 50
            else:
                max_semiangle = self._semiangle_cutoff * 1.6

        alpha = np.linspace(0, max_semiangle / 1000., 500)

        aberrations = self.evaluate_aberrations(alpha, phi)
        aperture = self.evaluate_aperture(alpha)
        temporal_envelope = self.evaluate_temporal_envelope(alpha)
        spatial_envelope = self.evaluate_spatial_envelope(alpha, phi)
        gaussian_envelope = self.evaluate_gaussian_envelope(alpha)
        envelope = aperture * temporal_envelope * spatial_envelope * gaussian_envelope

        calibration = Calibration(offset=0., sampling=(alpha[1] - alpha[0]) * 1000., units='mrad', name='alpha')

        profiles = {}
        profiles['ctf'] = Measurement(aberrations.imag * envelope, calibrations=[calibration], name='CTF')
        profiles['aperture'] = Measurement(aperture, calibrations=[calibration], name='Aperture')
        profiles['temporal_envelope'] = Measurement(temporal_envelope,
                                                    calibrations=[calibration],
                                                    name='Temporal')
        profiles['spatial_envelope'] = Measurement(spatial_envelope, calibrations=[calibration],
                                                   name='Spatial')
        profiles['gaussian_envelope'] = Measurement(gaussian_envelope, calibrations=[calibration],
                                    name='Gaussian')
        profiles['envelope'] = Measurement(envelope, calibrations=[calibration], name='Envelope')
        return profiles

    def apply(self, waves, interact=False, sliders=None, throttling=0.):
        if interact:
            from abtem.visualize.interactive import Canvas, MeasurementArtist2d
            from abtem.visualize.widgets import quick_sliders, throttle
            import ipywidgets as widgets

            image_waves = waves.copy()
            canvas = Canvas()
            artist = MeasurementArtist2d()
            canvas.artists = {'artist': artist}

            def update(*args):
                image_waves.array[:] = waves.apply_ctf(self).array
                artist.measurement = image_waves.intensity()[0]
                canvas.adjust_limits_to_artists()
                canvas.adjust_labels_to_artists()

            if throttling:
                update = throttle(throttling)(update)

            self.observe(update)
            update()

            if sliders:
                sliders = quick_sliders(self, **sliders)
                figure = widgets.HBox([canvas.figure, widgets.VBox(sliders)])
            else:
                figure = canvas.figure

            return image_waves, figure
        else:
            if sliders:
                raise RuntimeError()

            return waves.apply_ctf(self)

    def interact(self, max_semiangle: float = None, phi: float = 0., sliders=None, throttling=False):
        from abtem.visualize.interactive.utils import quick_sliders, throttle
        from abtem.visualize.interactive import Canvas, MeasurementArtist1d
        import ipywidgets as widgets

        canvas = Canvas(lock_scale=False)
        ctf_artist = MeasurementArtist1d()
        envelope_artist = MeasurementArtist1d()
        canvas.artists = {'ctf': ctf_artist, 'envelope': envelope_artist}
        canvas.y_scale.min = -1.1
        canvas.y_scale.max = 1.1

        def callback(*args):
            profiles = self.profiles(max_semiangle, phi)

            for name, artist in canvas.artists.items():
                artist.measurement = profiles[name]

        if throttling:
            callback = throttle(throttling)(callback)

        self.observe(callback)

        callback()
        canvas.adjust_limits_to_artists(adjust_y=False)
        canvas.adjust_labels_to_artists()

        if sliders:
            sliders = quick_sliders(self, **sliders)
            return widgets.HBox([canvas.figure, widgets.VBox(sliders)])
        else:
            return canvas.figure

    def show(self, max_semiangle: float = None, phi: float = 0, ax=None, **kwargs):
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
                ax, lines = profile.show(legend=True, ax=ax, **kwargs)

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
