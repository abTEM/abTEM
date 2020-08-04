from collections import OrderedDict
from typing import Optional, Union, Sequence

import numpy as np

from abtem.utils import energy2wavelength, energy2sigma


def watched_method(event):
    """
    Decorator for class methods that have to notify.
    """

    def wrapper(func):
        property_name = func.__name__

        def new_func(*args, **kwargs):
            instance = args[0]
            result = func(*args, **kwargs)
            getattr(instance, event).notify(**{'notifier': instance, 'property_name': property_name, 'change': True})
            return result

        return new_func

    return wrapper


def watched_property(event):
    """
    Decorator for class properties that have to notify.
    """

    def wrapper(func):
        property_name = func.__name__

        def new_func(*args):
            instance, value = args
            old = getattr(instance, property_name)
            result = func(*args)
            change = old != value
            change = np.any(change)
            getattr(instance, event).notify(**{'notifier': instance, 'property_name': property_name, 'change': change})
            return result

        return new_func

    return wrapper


class Event(object):

    def __init__(self):
        self.callbacks = []
        self._notify_count = 0

    @property
    def notify_count(self):
        return self._notify_count

    def notify(self, *args, **kwargs):
        self._notify_count += 1
        for callback in self.callbacks:
            callback(*args, **kwargs)

    def register(self, callbacks):
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        self.callbacks += callbacks


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

            if key in cache.cached:
                result = cache.retrieve(key)
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
        self._cached = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def cached(self):
        return self._cached

    @property
    def hits(self):
        return self._hits

    @property
    def misses(self):
        return self._hits

    def __len__(self):
        return len(self._cached)

    def insert(self, key, value):
        self._cached[key] = value
        self._check_size()

    def retrieve(self, key):
        return self._cached[key]

    def _check_size(self):
        if self._max_size is not None:
            while len(self) > self._max_size:
                self._cached.popitem(last=False)

    def clear(self):
        self._cached = OrderedDict()
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
                 lock_sampling=False):

        """
        Grid object.

        The grid object represent the simulation grid on which the wave functions and potential are discretized.

        Parameters
        ----------
        extent : sequence of float, float, optional
            Grid extent in each dimension [Å].
        gpts : sequence of int, int, optional
            Number of grid points in each dimension.
        sampling : sequence of float, float, optional
            Grid sampling in each dimension [1 / Å].
        dimensions : int
            Number of dimensions represented by the grid.
        endpoint : bool, optional
            If true include the grid endpoint (the default is False). For periodic grids the endpoint should not be
            included.
        """

        self.changed = Event()
        self._dimensions = dimensions
        self._endpoint = endpoint

        if sum([lock_extent, lock_gpts, lock_sampling]) > 1:
            raise RuntimeError('At most one of extent, gpts, and sampling may be locked')

        self._lock_extent = lock_extent
        self._lock_gpts = lock_gpts
        self._lock_sampling = lock_sampling

        self._extent = self._validate(extent, dtype=float)
        self._gpts = self._validate(gpts, dtype=int)
        self._sampling = self._validate(sampling, dtype=float)

        if self.extent is None:
            self._adjust_extent(self.gpts, self.sampling)

        if self.gpts is None:
            self._adjust_gpts(self.extent, self.sampling)

        self._adjust_sampling(self.extent, self.gpts)

    def _validate(self, value, dtype):
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) != self.dimensions:
                raise RuntimeError('Grid value length of {} != {}'.format(len(value), self._dimensions))
            return tuple((map(dtype, value)))

        if isinstance(value, (int, float, complex)):
            return (dtype(value),) * self.dimensions

        if value is None:
            return value

        raise RuntimeError('Invalid grid property ({})'.format(value))

    def __len__(self):
        return self.dimensions

    @property
    def endpoint(self) -> bool:
        return self._endpoint

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def extent(self) -> tuple:
        return self._extent

    @extent.setter
    @watched_method('changed')
    def extent(self, extent: Union[float, Sequence[float]]):
        if self._lock_extent:
            raise RuntimeError('Extent cannot be modified')

        extent = self._validate(extent, dtype=float)

        if self._lock_sampling or (self.gpts is None):
            self._adjust_gpts(extent, self.sampling)
            self._adjust_sampling(extent, self.gpts)
        elif self.gpts is not None:
            self._adjust_sampling(extent, self.gpts)

        self._extent = extent

    @property
    def gpts(self) -> tuple:
        return self._gpts

    @gpts.setter
    @watched_method('changed')
    def gpts(self, gpts: Union[int, Sequence[int]]):
        if self._lock_gpts:
            raise RuntimeError('Grid gpts cannot be modified')

        gpts = self._validate(gpts, dtype=int)

        if self._lock_sampling:
            self._adjust_extent(gpts, self.sampling)
        elif self.extent is not None:
            self._adjust_sampling(self.extent, gpts)
        else:
            self._adjust_extent(gpts, self.sampling)

        self._gpts = gpts

    @property
    def sampling(self) -> tuple:
        return self._sampling

    @sampling.setter
    @watched_method('changed')
    def sampling(self, sampling):
        if self._lock_sampling:
            raise RuntimeError('Sampling cannot be modified')

        sampling = self._validate(sampling, dtype=float)
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
                self._extent = tuple((n - 1) * d for n, d in zip(gpts, sampling))
            else:
                self._extent = tuple(n * d for n, d in zip(gpts, sampling))

    def _adjust_gpts(self, extent, sampling):
        if (extent is not None) & (sampling is not None):
            if self._endpoint:
                self._gpts = tuple(int(np.ceil(l / d)) + 1 for l, d in zip(extent, sampling))
            else:
                self._gpts = tuple(int(np.ceil(l / d)) for l, d in zip(extent, sampling))

    def _adjust_sampling(self, extent, gpts):
        if (extent is not None) & (gpts is not None):
            if self._endpoint:
                self._sampling = tuple(l / (n - 1) for l, n in zip(extent, gpts))
            else:
                self._sampling = tuple(l / n for l, n in zip(extent, gpts))

    def check_is_defined(self):
        """ Raise error if the grid is not defined. """
        if self.extent is None:
            raise RuntimeError('Grid extent is not defined')

        elif self.gpts is None:
            raise RuntimeError('Grid gpts are not defined')

    @property
    def antialiased_gpts(self):
        return tuple(n // 2 for n in self.gpts)

    @property
    def antialiased_sampling(self):
        return tuple(l / n for n, l in zip(self.antialiased_gpts, self.extent))

    def match(self, other):
        self.check_match(other)

        if (self.extent is None) & (other.extent is None):
            raise RuntimeError('Grid extent cannot be inferred')
        elif other.extent is None:
            other.extent = self.extent
        elif np.any(self.extent != other.extent):
            self.extent = other.extent

        if (self.gpts is None) & (other.gpts is None):
            raise RuntimeError('Grid gpts cannot be inferred')
        elif other.gpts is None:
            other.gpts = self.gpts
        elif np.any(self.gpts != other.gpts):
            self.gpts = other.gpts

    def check_match(self, other):
        """ Raise error if the grid of another object is different from this object. """

        if (self.extent is not None) & (other.extent is not None) & np.any(self.extent != other.extent):
            raise RuntimeError('Inconsistent grid extent ({} != {})'.format(self.extent, other.extent))

        elif (self.gpts is not None) & (other.gpts is not None) & np.any(self.gpts != other.gpts):
            raise RuntimeError('Inconsistent grid gpts ({} != {})'.format(self.gpts, other.gpts))

    def snap_to_power(self, power: float = 2):
        self.gpts = tuple(power ** np.ceil(np.log(n) / np.log(power)) for n in self.gpts)

    def __copy__(self):
        return self.__class__(extent=self.extent,
                              gpts=self.gpts,
                              sampling=self.sampling,
                              dimensions=self.dimensions,
                              endpoint=self.endpoint,
                              lock_extent=self._lock_extent,
                              lock_gpts=self._lock_gpts,
                              lock_sampling=self._lock_sampling)


class HasGridMixin:
    _grid: Grid

    @property
    def grid(self) -> Grid:
        return self._grid

    extent = DelegatedAttribute('grid', 'extent')
    gpts = DelegatedAttribute('grid', 'gpts')
    sampling = DelegatedAttribute('grid', 'sampling')


class Accelerator:
    """
    Accelerator object.

    The accelerator describes the energy of wave functions and transfer functions.

    :param energy: Acceleration energy [eV].
    """

    def __init__(self, energy: Optional[float] = None, lock_energy=False):
        if energy is not None:
            energy = float(energy)

        self.changed = Event()
        self._energy = energy
        self._lock_energy = lock_energy

    @property
    def energy(self) -> float:
        """
        :return: Acceleration energy [eV].
        """
        return self._energy

    @energy.setter
    @watched_method('changed')
    def energy(self, value: float):
        if self._lock_energy:
            raise RuntimeError('Energy cannot be modified')

        if value is not None:
            value = float(value)
        self._energy = value

    @property
    def wavelength(self) -> float:
        """
        :return: Relativistic wavelength [Å].
        """
        self.check_is_defined()
        return energy2wavelength(self.energy)

    @property
    def sigma(self) -> float:
        """
        :return: Interaction parameter.
        """
        self.check_is_defined()
        return energy2sigma(self.energy)

    def check_is_defined(self):
        """ Raise error if the energy is not defined. """

        if self.energy is None:
            raise RuntimeError('Energy is not defined')

    def check_match(self, other: 'Accelerator'):
        if (self.energy is not None) & (other.energy is not None) & (self.energy != other.energy):
            raise RuntimeError('Inconsistent energies')

    def match(self, other, check_match=True):
        if check_match:
            self.check_match(other)

        if (self.energy is None) & (other.energy is None):
            raise RuntimeError('Energy cannot be inferred')

        if other.energy is None:
            other.energy = self.energy

        else:
            self.energy = other.energy

    def __copy__(self):
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


class HasGridAndAcceleratorMixin(HasGridMixin, HasAcceleratorMixin):

    @property
    def max_scattering_angle(self):
        return 1 / np.max(self.grid.antialiased_sampling) * self.wavelength / 2 * 1000
