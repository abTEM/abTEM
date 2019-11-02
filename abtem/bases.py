from typing import Optional, Union

import numpy as np
from ase import units


class Observable:

    def __init__(self, **kwargs):
        """
        Observable base class.

        Base class for creating an observable class in the classic observer design pattern.

        :param kwargs: dummy
        """
        self._observers = []
        super().__init__(**kwargs)

    @property
    def observers(self) -> list:
        return self._observers

    def register_observer(self, observer: 'Observer'):
        if observer not in self._observers:
            self._observers.append(observer)

    def notify_observers(self, message):
        for observer in self._observers:
            observer.notify(self, message)


class Observer:

    def __init__(self, **kwargs):
        """
        Observer base class.

        Base class for creating an observer class in the classic observer design pattern.

        :param kwargs: dummy
        """
        super().__init__(**kwargs)

    def observe(self, observable):
        observable.register_observer(self)

    def notify(self, observable, message):
        raise NotImplementedError()


class SelfObservable(Observable, Observer):

    def __init__(self, **kwargs):
        """
        SelfObserver base class.

        Base class for creating an observable observer that observes itself.
        The object notifies itself before other registered observers.

        :param kwargs: dummy
        """
        super().__init__(**kwargs)

    def notify_observers(self, message):
        self.notify(self, message)
        super().notify_observers(message)


class HasCache(Observer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._cached = {}
        self._clear_conditions = {}

    def notify(self, observable, message):
        pop = []
        if message['change']:
            for name, conditions in self._clear_conditions.items():
                if (conditions == 'any') | (message['name'] in conditions):
                    pop.append(name)

        for name in pop:
            self._cached.pop(name, None)
            self._clear_conditions.pop(name, None)

    def clear_cache(self):
        self._cached = {}


def cached_method(clear_conditions='any'):
    def wrapper(func):
        def new_func(*args):
            self = args[0]
            try:
                return self._cached[func.__name__]
            except KeyError:
                self._cached[func.__name__] = func(*args)
                self._clear_conditions[func.__name__] = clear_conditions
                return self._cached[func.__name__]

        return new_func

    return wrapper


def cached_method_with_args(clear_conditions='any'):
    def wrapper(func):
        def new_func(*args):
            self = args[0]
            try:
                return self._cached[func.__name__][args[1:]]
            except KeyError:
                value = func(*args)

                try:
                    self._cached[func.__name__][args[1:]] = value
                except KeyError:
                    self._cached[func.__name__] = {}
                    self._cached[func.__name__][args[1:]] = value
                    self._clear_conditions[func.__name__] = clear_conditions

                return self._cached[func.__name__][args[1:]]

        return new_func

    return wrapper


class GridProperty:

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
            return self._validate(self._value(self))
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


def fourier_extent(extent, n):
    return np.array([-n[0] / extent[0], n[1] / extent[1]]) / 2.


class Grid(Observable):

    def __init__(self,
                 extent: Optional[Union[np.ndarray, float, GridProperty]] = None,
                 gpts: Optional[Union[np.ndarray, int, GridProperty]] = None,
                 sampling: Optional[Union[np.ndarray, int, GridProperty]] = None,
                 dimensions: int = 2, endpoint: bool = False,
                 **kwargs):

        self._dimensions = dimensions
        self._endpoint = endpoint

        if isinstance(extent, GridProperty):
            self._extent = extent
        else:
            self._extent = GridProperty(extent, np.float, locked=False, dimensions=dimensions)

        if isinstance(gpts, GridProperty):
            self._gpts = gpts
        else:
            self._gpts = GridProperty(gpts, np.int, locked=False, dimensions=dimensions)

        if isinstance(sampling, GridProperty):
            self._sampling = sampling
        else:
            self._sampling = GridProperty(sampling, np.float, locked=False, dimensions=dimensions)

        if self.extent is None:
            if not ((self.gpts is None) | (self.sampling is None)):
                self._extent.value = self._adjusted_extent(self.gpts, self.sampling)

        if self.gpts is None:
            if not ((self.extent is None) | (self.sampling is None)):
                self._gpts.value = self._adjusted_gpts(self.extent, self.sampling)

        if self.sampling is None:
            if not ((self.extent is None) | (self.gpts is None)):
                self._sampling.value = self._adjusted_sampling(self.extent, self.gpts)

        if (extent is not None) & (self.gpts is not None):
            self._sampling.value = self._adjusted_sampling(self.extent, self.gpts)

        if (gpts is not None) & (self.extent is not None):
            self._sampling.value = self._adjusted_sampling(self.extent, self.gpts)

        super().__init__(**kwargs)

    @property
    def endpoint(self) -> bool:
        return self._endpoint

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def extent(self) -> np.ndarray:
        if self._gpts.locked & self._sampling.locked:
            return self._adjusted_extent(self.gpts, self.sampling)

        return self._extent.value

    @extent.setter
    def extent(self, value):
        old = self._extent.value

        if self._gpts.locked & self._sampling.locked:
            raise RuntimeError()

        if not (self._sampling.locked | (value is None) | (self.gpts is None)):
            self._sampling.value = self._adjusted_sampling(value, self.gpts)

        elif not (self._gpts.locked | (value is None) | (self.sampling is None)):
            self._gpts.value = self._adjusted_gpts(value, self.sampling)
            self._sampling.value = self._adjusted_sampling(value, self.gpts)

        self._extent.value = value

        self.notify_observers({'name': 'extent', 'old': old, 'new': value, 'change': np.any(old != value)})

    @property
    def gpts(self):
        if self._extent.locked & self._sampling.locked:
            return self._adjusted_sampling(self.extent, self.sampling)

        return self._gpts.value

    @gpts.setter
    def gpts(self, value):
        old = self._gpts.value

        if self._extent.locked & self._sampling.locked:
            raise RuntimeError()

        if not (self._sampling.locked | (self.extent is None) | (value is None)):
            self._sampling.value = self._adjusted_sampling(self.extent, value)

        elif not (self._extent.locked | (value is None) | (self.sampling is None)):
            self._extent.value = self._adjusted_extent(value, self.sampling)

        self._gpts.value = value

        self.notify_observers({'name': 'gpts', 'old': old, 'new': value, 'change': np.any(old != value)})

    @property
    def sampling(self):
        if self._extent.locked & self._gpts.locked:
            return self._adjusted_sampling(self.extent, self.gpts)

        return self._sampling.value

    @sampling.setter
    def sampling(self, value):
        old = self._sampling.value

        if self._gpts.locked & self._extent.locked:
            raise RuntimeError()

        if not (self._gpts.locked | (self.extent is None) | (value is None)):
            self._gpts.value = self._adjusted_gpts(self.extent, value)
            value = self._adjusted_sampling(self.extent, self.gpts)

        elif not (self._extent.locked | (self.gpts is None) | (value is None)):
            self._extent.value = self._adjusted_extent(self.gpts, value)

        self._sampling.value = value

        self.notify_observers({'name': 'sampling', 'old': old, 'new': value, 'change': np.any(old != value)})

    def _adjusted_extent(self, gpts, sampling):
        if self._endpoint:
            return (gpts - 1) * sampling
        else:
            return gpts * sampling

    def _adjusted_gpts(self, extent, sampling):
        if self._endpoint:
            return np.ceil(extent / sampling).astype(np.int) + 1
        else:
            return np.ceil(extent / sampling).astype(np.int)

    def _adjusted_sampling(self, extent, gpts):
        if self._endpoint:
            return extent / (gpts - 1)
        else:
            return extent / gpts

    def check_is_grid_defined(self):
        """ Throw error if the grid is not defined. """
        if (self.extent is None) | (self.gpts is None) | (self.sampling is None):
            raise RuntimeError('grid is not defined')

    @property
    def fourier_extent(self):
        return fourier_extent(self.extent, self.gpts)

    def match_grid(self, other):
        """ Throw error if the grid of another object is different from this object. """
        if np.any(self.extent != other.extent):
            raise RuntimeError('inconsistent grids')

        elif np.any(self.gpts != other.gpts):
            raise RuntimeError('inconsistent grids')

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


def notifying_property(name):
    def getter(self):
        return getattr(self, name)

    def setter(self, value):
        old = getattr(self, name)
        setattr(self, name, value)
        change = np.any(old != value)
        self.notify_observers({'name': name, 'old': old, 'new': value, 'change': change})

    return property(getter, setter)


def notify(func):
    name = func.__name__

    def wrapper(*args):
        obj, value = args
        old = getattr(obj, name)
        func(*args)
        change = np.any(old != value)
        obj.notify_observers({'name': name, 'old': old, 'new': value, 'change': change})

    return wrapper


class Energy(Observable):

    def __init__(self, energy: Optional[float] = None, **kwargs):
        """
        Energy base class

        Base class for describing the energy of wavefunctions and transfer functions.

        :param energy: energy
        :type energy: optional, float
        """
        self._energy = energy

        super().__init__(**kwargs)

    @property
    def energy(self) -> float:
        return self._energy

    @energy.setter
    @notify
    def energy(self, value: float):
        self._energy = value

    @property
    def wavelength(self) -> float:
        """
        Relativistic wavelength from energy.
        :return: wavelength
        :rtype: float
        """
        self.check_is_energy_defined()
        return energy2wavelength(self.energy)

    @property
    def sigma(self) -> float:
        """
        Interaction parameter from energy.
        """
        self.check_is_energy_defined()
        return energy2sigma(self.energy)

    def check_is_energy_defined(self):
        """ Throw error if the energy is not defined. """
        if self.energy is None:
            raise RuntimeError('energy is not defined')

    def check_same_energy(self, other: 'Energy'):
        """ Throw error if the energy of another object is different from this object. """
        if self.energy != other.energy:
            raise RuntimeError('inconsistent energies')

    def copy(self) -> 'Energy':
        """
        :return: A copy of itself
        :rtype: Energy
        """
        return self.__class__(self.energy)


class ArrayWithGrid(Grid):
    def __init__(self, array, spatial_dimensions, extent=None, sampling=None, space=None, **kwargs):
        array_dimensions = len(array.shape)

        if array_dimensions < spatial_dimensions:
            raise RuntimeError()

        self._array = array
        self._spatial_dimensions = spatial_dimensions
        self._space = space

        gpts = GridProperty(value=lambda obj: obj.gpts, dtype=np.int, locked=True, dimensions=spatial_dimensions)
        super().__init__(extent=extent, gpts=gpts, sampling=sampling, dimensions=spatial_dimensions, **kwargs)

    def fourier_transform(self, shift=True, in_place=True):
        axes = tuple(range(len(self.array.shape)))[-self.spatial_dimensions:]

        if (self.space is None) | (self.space == 'direct'):
            self._array = np.fft.fftn(self._array, axes=axes)
            self._array = np.fft.fftshift(self._array, axes=axes)

        elif self.space == 'fourier':
            if shift:
                self._array = np.fft.fftshift(self._array, axes=axes)
            self._array = np.fft.ifftn(self._array, axes=axes)

        return self

    @property
    def space(self):
        return self._space

    @property
    def spatial_dimensions(self):
        return self._spatial_dimensions

    @property
    def gpts(self):
        shape = self.array.shape
        return np.array([dim for dim in shape[- self._dimensions:]])

    @property
    def array(self):
        return self._array

    def copy(self):
        return self.__class__(array=self.array.copy(), extent=self.extent.copy())


class ArrayWithGridAndEnergy(ArrayWithGrid, Energy):

    def __init__(self, array, spatial_dimensions, extent=None, sampling=None, energy=None, **kwargs):
        super().__init__(array=array, spatial_dimensions=spatial_dimensions, extent=extent, sampling=sampling,
                         energy=energy, **kwargs)

    def copy(self):
        return self.__class__(array=self.array.copy(), spatial_dimensions=self.spatial_dimensions,
                              extent=self.extent.copy(), energy=self.energy)


class LineProfile(ArrayWithGrid):

    def __init__(self, array, extent=None, sampling=None, space='direct'):
        super().__init__(array, 1, extent=extent, sampling=sampling, space=space)

    # def __getitem__(self, i):
    #    new_array = self._array[i]
    #    return self.__class__(new_array, extent=self.sampling * new_array.shape)


class Image(ArrayWithGrid):
    def __init__(self, array, extent=None, sampling=None, space='direct'):
        super().__init__(array, 2, extent=extent, sampling=sampling, space=space)

    def get_profile(self, slice_position=None, axis=0):
        if slice_position is None:
            slice_position = self.gpts[int(not axis)] // 2

        array = np.take(self.array, slice_position, int(not axis))
        return LineProfile(array, extent=self.extent[axis], space=self.space)
