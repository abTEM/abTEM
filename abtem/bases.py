import numpy as np
from ase import units


def named_property(name):
    def getter(self):
        return getattr(self, name)

    def setter(self, value):
        setattr(self, name, value)

    return property(getter, setter)


def referenced_property(reference_name, property_name):
    def getter(self):
        return getattr(getattr(self, reference_name), property_name)

    def setter(self, value):
        setattr(getattr(self, reference_name), property_name, value)

    return property(getter, setter)


def notifying_property(name):
    def getter(self):
        return getattr(self, name)

    def setter(self, value):
        old = getattr(self, name)
        setattr(self, name, value)
        change = np.all(old != value)
        self.notify_observers({'name': name, 'old': old, 'new': value, 'change': change})

    return property(getter, setter)


def xy_property(component, name):
    def getter(self):
        return getattr(self, name)[component]

    def setter(self, value):
        new = getattr(self, name).copy()
        new[component] = value
        setattr(self, name, new)

    return property(getter, setter)


class Observable(object):
    def __init__(self):
        self._observers = []

    def register_observer(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def notify_observers(self, message):
        for observer in self._observers:
            observer.notify(self, message)


class Observer(object):
    def __init__(self, observable=None):
        self._observing = []

        if observable:
            self.observe(observable)

    def observe(self, observable):
        observable.register_observer(self)

    def notify(self, observable, message):
        raise NotImplementedError()


def cached_method(func):
    def new_func(*args):
        self = args[0]
        try:
            return self._cached[func.__name__]
        except:
            self._cached[func.__name__] = func(*args)
            return self._cached[func.__name__]

    return new_func


def cached_method_with_args(func):
    def new_func(*args):
        self = args[0]
        try:
            return self._cached[func.__name__][args[1:]]
        except:
            value = func(*args)

            try:
                self._cached[func.__name__][args[1:]] = value
            except:
                self._cached[func.__name__] = {}
                self._cached[func.__name__][args[1:]] = value

            return self._cached[func.__name__][args[1:]]

    return new_func


class HasCache(Observer):

    def __init__(self):
        Observer.__init__(self)
        self._cached = {}

    def notify(self, observable, message):
        if message['change']:
            self.clear_cache()

    def clear_cache(self):
        self._cached = {}


class GridProperty(object):

    def __init__(self, value, dtype, locked=False, dimensions=2):
        self._dtype = dtype
        self._locked = locked
        self._dimensions = dimensions
        self._value = self._validate(value)

    @property
    def locked(self):
        return self._locked

    @property
    def value(self):
        if self._locked:
            return self._validate(self._value())
        else:
            return self._value

    def _validate(self, value):
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) != self._dimensions:
                raise RuntimeError('grid value length of {} != {}'.format(len(value), self._dimensions))
            value = np.array(value).astype(self._dtype)

        elif isinstance(value, (int, float, complex)):
            value = np.full(self._dimensions, value, dtype=self._dtype)

        elif callable(value):
            if not self._locked:
                raise RuntimeError('lock gridproperty to assign callable')

        elif value is None:
            pass

        else:
            raise RuntimeError('invalid grid property ({})'.format(value))

        return value

    @value.setter
    def value(self, value):
        if self._locked:
            raise RuntimeError('grid property locked')
        self._value = self._validate(value)

    def copy(self):
        return self.__class__(value=self._value, dtype=self._dtype, locked=self._locked, dimensions=self._dimensions)


class Grid(Observable):

    def __init__(self, extent=None, gpts=None, sampling=None, dimensions=2):

        Observable.__init__(self)

        self._dimensions = dimensions

        if isinstance(extent, GridProperty):
            self._extent = extent
        else:
            self._extent = GridProperty(extent, np.float32, locked=False, dimensions=dimensions)

        if isinstance(gpts, GridProperty):
            self._gpts = gpts
        else:
            self._gpts = GridProperty(gpts, np.int32, locked=False, dimensions=dimensions)

        if isinstance(sampling, GridProperty):
            self._sampling = sampling
        else:
            self._sampling = GridProperty(sampling, np.float32, locked=False, dimensions=dimensions)

        if self.extent is None:
            if not ((self.gpts is None) | (self.sampling is None)):
                self._extent.value = self._adjusted_extent()

        if self.gpts is None:
            if not ((self.extent is None) | (self.sampling is None)):
                self._gpts.value = self._adjusted_gpts()

        if self.sampling is None:
            if not ((self.extent is None) | (self.gpts is None)):
                self._sampling.value = self._adjusted_sampling()

        if (extent is not None) & (self.gpts is not None):
            self._sampling.value = self._adjusted_sampling()

        if (gpts is not None) & (self.extent is not None):
            self._sampling.value = self._adjusted_sampling()

    @property
    def extent(self):
        if self._gpts.locked & self._sampling.locked:
            return self._adjusted_extent()

        return self._extent.value

    @extent.setter
    def extent(self, value):
        old = self._extent.value
        self._extent.value = value

        if self._gpts.locked & self._sampling.locked:
            raise RuntimeError()

        if not (self._gpts.locked | (self.extent is None) | (self.sampling is None)):
            self._gpts.value = self._adjusted_gpts()
            self._sampling.value = self._adjusted_sampling()

        elif not (self._sampling.locked | (self.extent is None) | (self.gpts is None)):
            self._sampling.value = self._adjusted_sampling()

        self.notify_observers({'name': 'extent', 'old': old, 'new': value, 'change': np.any(old != value)})

    @property
    def gpts(self):
        if self._extent.locked & self._sampling.locked:
            return self._adjusted_sampling()

        return self._gpts.value

    @gpts.setter
    def gpts(self, value):
        old = self._gpts.value
        self._gpts.value = value

        if self._extent.locked & self._sampling.locked:
            raise RuntimeError()

        if not (self._sampling.locked | (self.extent is None) | (self.gpts is None)):
            self._sampling.value = self._adjusted_sampling()

        elif not (self._extent.locked | (self.gpts is None) | (self.sampling is None)):
            self._extent.value = self._adjusted_extent()

        self.notify_observers({'name': 'gpts', 'old': old, 'new': value, 'change': np.any(old != value)})

    @property
    def sampling(self):
        if self._gpts.locked & self._extent.locked:
            return self._adjusted_sampling()

        return self._sampling.value

    @sampling.setter
    def sampling(self, value):
        old = self._sampling.value
        self._sampling.value = value

        if self._gpts.locked & self._extent.locked:
            raise RuntimeError()

        if not (self._gpts.locked | (self.extent is None) | (self.sampling is None)):
            self._gpts.value = self._adjusted_gpts()
            self._extent.value = self._adjusted_extent()

        elif not (self._extent.locked | (self.gpts is None) | (self.sampling is None)):
            self._extent.value = self._adjusted_extent()

        self.notify_observers({'name': 'sampling', 'old': old, 'new': value, 'change': np.any(old != value)})

    def _adjusted_extent(self):
        return np.float32(self.gpts) * self.sampling

    def _adjusted_gpts(self):
        return np.ceil(self.extent / self.sampling).astype(np.int32)

    def _adjusted_sampling(self):
        return self.extent / np.float32(self.gpts)

    # def linspace(self):
    #     return tuple([np.linspace(0., self.extent[i], self.gpts[i], endpoint=False) for i in range(self._dimensions)])
    #
    # def fftfreq(self):
    #     return np.fft.fftfreq(self.gpts[0], self.sampling[0]), np.fft.fftfreq(self.gpts[1], self.sampling[1])

    def check_is_grid_defined(self):
        if (self.extent is None) | (self.gpts is None) | (self.sampling is None):
            raise RuntimeError('grid is not defined')

    def clear_grid(self):
        self._extent.value = None
        self._gpts.value = None
        self._sampling.value = None
        self.notify_observers({'change': True})

    def match_grid(self, other):
        if self.extent is None:
            self.extent = other.extent

        elif other.extent is None:
            other.extent = self.extent

        elif np.any(self.extent != other.extent):
            raise RuntimeError('inconsistent grids')

        if self.gpts is None:
            self.gpts = other.gpts

        elif other.gpts is None:
            other.gpts = self.gpts

        elif np.any(self.gpts != other.gpts):
            raise RuntimeError('inconsistent grids')

        if self.sampling is None:
            self.sampling = other.sampling

        elif other.sampling is None:
            other.sampling = self.sampling

        elif np.any(self.sampling != other.sampling):
            raise RuntimeError('inconsistent grids')

    def copy(self):
        return self.__class__(extent=self._extent.copy(), gpts=self._gpts.copy(), sampling=self._sampling.copy(),
                              dimensions=self._dimensions)


def energy2mass(energy):
    """
    Calculate relativistic mass from energy.
    :param energy: Energy in electron volt
    :type energy: float
    :return: Relativistic mass in kg
    :rtype: float
    """
    return (1 + units._e * energy / (units._me * units._c ** 2)) * units._me


def energy2wavelength(energy):
    """
    Calculate relativistic de Broglie wavelength from energy.
    :param energy: Energy in electron volt
    :type energy: float
    :return: Relativistic de Broglie wavelength in Angstrom.
    :rtype: float
    """
    return units._hplanck * units._c / np.sqrt(
        energy * (2 * units._me * units._c ** 2 / units._e + energy)) / units._e * 1.e10


def energy2sigma(energy):
    """
    Calculate interaction parameter from energy.
    :param energy: Energy in electron volt.
    :type energy: float
    :return: Interaction parameter in 1 / (Angstrom * eV).
    :rtype: float
    """
    return (2 * np.pi * energy2mass(energy) * units.kg * units._e * units.C * energy2wavelength(energy) / (
            units._hplanck * units.s * units.J) ** 2)


class Energy(Observable):

    def __init__(self, energy=None):
        Observable.__init__(self)

        self._energy = energy

    energy = notifying_property('_energy')

    @property
    def wavelength(self):
        self.check_is_energy_defined()
        return energy2wavelength(self.energy)

    @property
    def sigma(self):
        self.check_is_energy_defined()
        return energy2sigma(self.energy)

    def check_is_energy_defined(self):
        if self.energy is None:
            raise RuntimeError('energy is not defined')

    def match_energy(self, other):
        if other.energy is None:
            other.energy = self.energy

        elif self.energy is None:
            self.energy = other.energy

        elif self.energy != other.energy:
            raise RuntimeError('inconsistent energies')

    def copy(self):
        return self.__class__(self.energy)


class ArrayWithGrid(Grid):
    def __init__(self, array, array_dimensions, spatial_dimensions, extent=None, sampling=None, space='direct'):

        if array_dimensions < spatial_dimensions:
            raise RuntimeError()

        if len(array.shape) != array_dimensions:
            raise RuntimeError('tensor shape {} not {}d'.format(array.shape, array_dimensions))

        self._array = array

        gpts = GridProperty(lambda: self.gpts, dtype=np.int32, locked=True, dimensions=spatial_dimensions)
        Grid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, dimensions=spatial_dimensions)

        self.space = space

    @property
    def gpts(self):
        shape = self.array.shape
        return np.array([dim for dim in shape[- self._dimensions:]])

    @property
    def array(self):
        return self._array

    def copy(self):
        new = self.__class__(array=self.array.copy(), extent=self.extent.copy(), space=self.space)
        return new
