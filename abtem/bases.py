from collections import OrderedDict
from typing import Optional, Union, Sequence

import numpy as np

from abtem.config import DTYPE
from abtem.utils import energy2wavelength, energy2sigma


def watched_method(event):
    """
    Decorator for class methods that have to notify.
    """

    def wrapper(func):
        property_name = func.__name__

        def new_func(*args):
            instance, value = args
            # old = getattr(instance, property_name)
            func(*args)
            # change = old != value
            # if isinstance(change, Iterable):
            #    change = np.any(change)
            getattr(instance, event).notify(**{'notifier': instance, 'property_name': property_name, 'change': True})

        return new_func

    return wrapper


class Event(object):

    def __init__(self):
        self.callbacks = []
        self._notify_count = 0

    def notify(self, *args, **kwargs):
        self._notify_count += 1
        for callback in self.callbacks:
            callback(*args, **kwargs)

    def register(self, callbacks):
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        self.callbacks += callbacks

    @classmethod
    def watched_property(cls, event_name, key):
        actual_key = '_%s' % key

        def getter(obj):
            return getattr(obj, actual_key)

        def setter(obj, value):
            event = getattr(obj, event_name)
            setattr(obj, actual_key, value)
            event.notify(obj, key, value)

        return property(fget=getter, fset=setter)


def cache_clear_callback(target_cache):
    def callback(notifier, property_name, change):
        if change:
            target_cache.clear()

    return callback


def cached_method(target_cache, ignore_args=False):
    def wrapper(func):

        def new_func(*args):
            self = args[0]

            cache = getattr(self, target_cache)

            if ignore_args is True:
                key = (func,)
            else:
                key = (func,) + args[1:]

            if key in cache._cache:
                result = cache._cache[key]
                cache._hits += 1
            else:
                result = func(*args)
                cache.insert(key, result)
                cache._misses += 1
            return result

        return new_func

    return wrapper


class Cache:

    def __init__(self, max_size):
        self._max_size = max_size
        self._cache = OrderedDict()
        self._hits = 0
        self._misses = 0

    def __len__(self):
        return len(self._cache)

    def insert(self, key, value):
        self._cache[key] = value
        self._check_size()

    def _check_size(self):
        if self._max_size is not None:
            while len(self) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self):
        self._cache = OrderedDict()
        self._hits = 0
        self._misses = 0


class DelegatedAttribute:
    def __init__(self, delegate_name, attr_name):
        self.attr_name = attr_name
        self.delegate_name = delegate_name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            # return instance.delegate.attr
            return getattr(self.delegate(instance), self.attr_name)

    def __set__(self, instance, value):
        # instance.delegate.attr = value
        setattr(self.delegate(instance), self.attr_name, value)

    def __delete__(self, instance):
        delattr(self.delegate(instance), self.attr_name)

    def delegate(self, instance):
        return getattr(instance, self.delegate_name)


class Grid:

    def __init__(self,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 dimensions: int = 2,
                 endpoint: bool = False,
                 lock_extent=False,
                 lock_gpts=False,
                 lock_sampling=False,
                 flexible=False):

        """
        Grid object.

        The grid object represent the simulation grid on which the wave function and potential is discretized.

        Parameters
        ----------
        extent : sequence of float, float, optional
            Grid extent in each dimension [Å]
        gpts : sequence of int, int, optional
            Number of grid points in each dimension
        sampling : sequence of float, float, optional
            Grid sampling in each dimension [1 / Å]
        dimensions : int
            Number of dimensions represented by the grid.
        endpoint : bool, optional
            If true include the grid endpoint (the dafault is False). For periodic grids the endpoint should not be
            included.
        """

        self.changed = Event()
        self._dimensions = dimensions
        self._endpoint = endpoint

        if sum([lock_extent, lock_gpts, lock_sampling]) > 1:
            raise RuntimeError('at most one of extent, gpts, and sampling may be locked')

        self._lock_extent = lock_extent
        self._lock_gpts = lock_gpts
        self._lock_sampling = lock_sampling

        self._extent = self._validate(extent, dtype=DTYPE)
        self._gpts = self._validate(gpts, dtype=np.int)
        self._sampling = self._validate(sampling, dtype=DTYPE)

        if self.extent is None:
            self._adjust_extent(self.gpts, self.sampling)

        if self.gpts is None:
            self._adjust_gpts(self.extent, self.sampling)

        self._adjust_sampling(self.extent, self.gpts)

        self.cache = Cache(1)
        self.changed.register(cache_clear_callback(self.cache))
        self.changed.register(cache_clear_callback(self.cache))

    # def __str__(self):
    #     str(' x '.join(map(str, list(np.round(self.grid.extent, 2))))) + ' Å'

    # def

    def _validate(self, value, dtype):
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) != self._dimensions:
                raise RuntimeError('grid value length of {} != {}'.format(len(value), self._dimensions))
            return np.array(value).astype(dtype)

        if isinstance(value, (int, float, complex)):
            return np.full(self._dimensions, value, dtype=dtype)

        if value is None:
            return value

        raise RuntimeError('invalid grid property ({})'.format(value))

    def __len__(self):
        return self.dimensions

    @property
    def endpoint(self) -> bool:
        return self._endpoint

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def extent(self) -> np.ndarray:
        return self._extent

    @extent.setter
    @watched_method('changed')
    def extent(self, extent: Union[float, Sequence[float]]):
        if self._lock_extent:
            raise RuntimeError('extent cannot be modified')

        extent = self._validate(extent, dtype=DTYPE)

        if self._lock_sampling or (self.gpts is None):
            self._adjust_gpts(extent, self.sampling)
            self._adjust_sampling(extent, self.gpts)
        elif self.gpts is not None:
            self._adjust_sampling(extent, self.gpts)

        self._extent = extent

    @property
    def gpts(self) -> np.ndarray:
        return self._gpts

    @gpts.setter
    @watched_method('changed')
    def gpts(self, gpts: Union[int, Sequence[int]]):
        if self._lock_gpts:
            raise RuntimeError('gpts cannot be modified')

        gpts = self._validate(gpts, dtype=np.int)

        if self._lock_sampling:
            self._adjust_extent(gpts, self.sampling)
        elif self.extent is not None:
            self._adjust_sampling(self.extent, gpts)
        else:
            self._adjust_extent(gpts, self.sampling)

        self._gpts = gpts

    @property
    def sampling(self) -> np.ndarray:
        return self._sampling

    @sampling.setter
    @watched_method('changed')
    def sampling(self, sampling):
        if self._lock_sampling:
            raise RuntimeError('sampling cannot be modified')

        sampling = self._validate(sampling, dtype=DTYPE)

        if self._lock_gpts:
            self._adjust_extent(self.gpts, sampling)
        elif self.extent is not None:
            self._adjust_gpts(self.extent, sampling)
        else:
            self._adjust_extent(self.gpts, sampling)

        self._adjust_sampling(self.extent, self.gpts)

    def _adjust_extent(self, gpts, sampling):
        if (gpts is not None) & (sampling is not None):
            if self._endpoint:
                self._extent = (gpts - 1) * sampling
            else:
                self._extent = gpts * sampling

    def _adjust_gpts(self, extent, sampling):
        if (extent is not None) & (sampling is not None):
            if self._endpoint:
                self._gpts = np.ceil(extent / sampling).astype(np.int) + 1
            else:
                self._gpts = np.ceil(extent / sampling).astype(np.int)

    def _adjust_sampling(self, extent, gpts):
        if (extent is not None) & (gpts is not None):
            if self._endpoint:
                self._sampling = extent / (gpts - 1)
            else:
                self._sampling = extent / gpts

    def check_is_defined(self):
        """ Throw error if the grid is not defined. """
        if self.extent is None:
            raise RuntimeError('grid extent is not defined')

        elif self.gpts is None:
            raise RuntimeError('grid gpts is not defined')

    # def spatial_frequency_limits(self):
    #     return np.array([(-1 / (2 * d), 1 / (2 * d) - 1 / (d * p)) if (p % 2 == 0) else
    #                      (-1 / (2 * d) + 1 / (2 * d * p), 1 / (2 * d) - 1 / (2 * d * p)) for d, p in
    #                      zip(self.sampling, self.gpts)])
    #
    # @property
    # def spatial_frequency_extent(self):
    #     fourier_limits = self.spatial_frequency_limits
    #     return fourier_limits[:, 1] - fourier_limits[:, 0]

    def match(self, other):
        self.check_can_match(other)

        if (self.extent is None) & (other.extent is None):
            raise RuntimeError('grid extent cannot be inferred')

        elif self.extent is None:
            self.extent = other.extent

        elif other.extent is None:
            other.extent = self.extent

        if (self.gpts is None) & (other.gpts is None):
            raise RuntimeError('grid gpts cannot be inferred')

        elif self.gpts is None:
            self.gpts = other.gpts

        elif other.gpts is None:
            other.gpts = self.gpts

    def check_can_match(self, other):
        """ Throw error if the grid of another object is different from this object. """

        if (self.extent is not None) & (other.extent is not None) & np.any(self.extent != other.extent):
            raise RuntimeError('inconsistent grid extent ({} != {})'.format(self.extent, other.extent))

        elif (self.gpts is not None) & (other.gpts is not None) & np.any(self.gpts != other.gpts):
            raise RuntimeError('inconsistent grid gpts ({} != {})'.format(self.gpts, other.gpts))

    def snap_to_power(self, power: float = 2):
        self.gpts = [power ** np.ceil(np.log(n) / np.log(power)) for n in self.gpts]

    @cached_method('cache')
    def coordinates(self):
        self.check_is_defined()
        return [np.linspace(0, e, g, endpoint=self.endpoint, dtype=np.float32) for g, e in zip(self.gpts, self.extent)]

    @cached_method('cache')
    def spatial_frequencies(self):
        self.check_is_defined()
        return [np.fft.fftfreq(g, s).astype(np.float32) for g, s in zip(self.gpts, self.sampling)]

    def copy(self):
        return self.__class__(extent=self._extent.copy(),
                              gpts=self._gpts.copy(),
                              sampling=self._sampling.copy(),
                              dimensions=self._dimensions,
                              endpoint=self.endpoint,
                              lock_extent=self._lock_extent,
                              lock_gpts=self._lock_gpts,
                              lock_sampling=self._lock_sampling)


class HasGridMixin:
    _grid: Grid

    @property
    def grid(self) -> Grid:
        return self._grid

    @grid.setter
    def grid(self, new: Grid):
        changed_event = self._grid.changed
        self._accelerator = new
        self._accelerator.changed = changed_event

    extent = DelegatedAttribute('grid', 'extent')
    gpts = DelegatedAttribute('grid', 'gpts')
    sampling = DelegatedAttribute('grid', 'sampling')


class Accelerator:
    """
    Energy base class

    Base class for describing the energy of wavefunctions and transfer functions.

    :param energy: energy
    :type energy: optional, float
    """

    def __init__(self, energy: Optional[float] = None, lock_energy=False):
        """
        Energy base class.

        The Energy object is used to represent the acceleration energy of an inheriting waves object.

        Parameters
        ----------
        energy : float
            Acceleration energy [eV]
        kwargs :
        """
        if energy is not None:
            energy = DTYPE(energy)

        self.changed = Event()
        self._energy = energy
        self._lock_energy = lock_energy

    @property
    def energy(self) -> float:
        return self._energy

    @energy.setter
    @watched_method('changed')
    def energy(self, value: float):
        if self._lock_energy:
            raise RuntimeError('energy cannot be modified')

        if value is not None:
            value = DTYPE(value)
        self._energy = value

    @property
    def wavelength(self) -> float:
        """
        Relativistic wavelength from energy.
        :return: wavelength
        :rtype: float
        """
        self.check_is_defined()
        return DTYPE(energy2wavelength(self.energy))

    @property
    def sigma(self) -> float:
        """
        Interaction parameter from energy.
        """
        self.check_is_defined()
        return DTYPE(energy2sigma(self.energy))

    def check_is_defined(self):
        """ Throw error if the energy is not defined. """

        if self.energy is None:
            raise RuntimeError('energy is not defined')

    def check_energies_can_match(self, other: 'Accelerator'):
        if (self.energy is not None) & (other.energy is not None) & (self.energy != other.energy):
            raise RuntimeError('inconsistent energies')

    def match(self, other):
        self.check_energies_can_match(other)

        if (self.energy is None) & (other.energy is None):
            raise RuntimeError('energy cannot be inferred')

        elif self.energy is None:
            self.energy = other.energy

        elif other.energy is None:
            other.energy = self.energy

    def copy(self):
        return self.__class__(self.energy)


class HasAcceleratorMixin:
    _accelerator: Accelerator

    @property
    def accelerator(self) -> Accelerator:
        return self._accelerator

    @accelerator.setter
    def accelerator(self, new: Accelerator):
        self._accelerator = new
        self._accelerator.changed = new.changed

    energy = DelegatedAttribute('accelerator', 'energy')
    wavelength = DelegatedAttribute('accelerator', 'wavelength')
