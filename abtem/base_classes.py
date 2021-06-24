"""Module for often-used base classes."""
from collections import OrderedDict
from typing import Union, Sequence, Any, Callable, Tuple

import numpy as np


class Event(object):
    """
    Event class for registering callbacks.
    """

    def __init__(self):
        self.callbacks = []
        self._notify_count = 0

    @property
    def notify_count(self):
        """
        Number of times this event has been notified.
        """
        return self._notify_count

    def notify(self, change):
        """
        Notify this event. All registered callbacks are called.
        """

        self._notify_count += 1
        for callback in self.callbacks:
            callback(change)

    def observe(self, callbacks: Union[Callable, Sequence[Callable]]):
        """
        Register new callbacks.

        Parameters
        ----------
        callbacks : callable
            The callbacks to register.
        """
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        self.callbacks += callbacks


class HasEventMixin:
    _event: Event

    @property
    def event(self):
        return self._event

    def observe(self, callback):
        self.event.observe(callback)


def watched_method(event: 'str'):
    """
    Decorator for class methods that have to notify.

    Parameters
    ----------
    event : str
        Name class property with target event.
    """

    def wrapper(func):
        property_name = func.__name__

        def new_func(*args, **kwargs):
            instance = args[0]
            result = func(*args, **kwargs)
            getattr(instance, event).notify({'owner': instance, 'name': property_name, 'change': True})
            return result

        return new_func

    return wrapper


def watched_property(event: 'str'):
    """
    Decorator for class properties that have to notify an event.

    Parameters
    ----------
    event : str
        Name class property with target event
    """

    def wrapper(func):
        property_name = func.__name__

        def new_func(*args):
            instance, value = args
            old = getattr(instance, property_name)
            result = func(*args)
            change = np.any(old != value)
            getattr(instance, event).notify({'notifier': instance, 'name': property_name, 'change': change,
                                             'old': old, 'new': value})
            return result

        return new_func

    return wrapper


def cache_clear_callback(target_cache: 'Cache'):
    """
    Helper function for creating a callback that clears a target cache object.

    Parameters
    ----------
    target_cache : Cache object
        The target cache object.
    """

    # noinspection PyUnusedLocal
    def callback(change):
        if change['change']:
            target_cache.clear()

    return callback


def cached_method(target_cache_property: str):
    """
    Decorator for cached methods. The method will store the output in the cache held by the target property.

    Parameters
    ----------
    target_cache_property : str
        The property holding the target cache.
    """

    def wrapper(func):

        def new_func(*args):
            cache = getattr(args[0], target_cache_property)
            key = (func,) + args[1:]

            if key in cache.cached:
                # The decorated method has been called once with the given args.
                # The calculation will be retrieved from cache.

                result = cache.retrieve(key)
                cache._hits += 1
            else:
                # The method will be called and its output will be cached.

                result = func(*args)
                cache.insert(key, result)
                cache._misses += 1

            return result

        return new_func

    return wrapper


def copy_docstring_from(source):
    def wrapper(func):
        func.__doc__ = source.__doc__
        return func

    return wrapper


class Cache:
    """
    Cache object.

    Class for handling a dictionary-based cache. When the cache is full, the first inserted item is deleted.

    Parameters
    ----------
    max_size : int
        The maximum number of values stored by this cache.
    """

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._cached = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def cached(self) -> dict:
        """
        Dictionary of cached data.
        """
        return self._cached

    @property
    def hits(self) -> int:
        """
        Number of times a previously calculated object was retrieved.
        """
        return self._hits

    @property
    def misses(self) -> int:
        """
        Number of times a new object had to be calculated.
        """
        return self._hits

    def __len__(self) -> int:
        """
        Number of objects cached.
        """
        return len(self._cached)

    def insert(self, key: Any, value: Any):
        """
        Insert new value into the cache.

        Parameters
        ----------
        key : Any
            The dictionary key of the cached object.
        value : Any
            The object to cache.
        """
        self._cached[key] = value
        self._check_size()

    def retrieve(self, key: Any) -> Any:
        """
        Retrieve object from cache.

        Parameters
        ----------
        key: Any
            The key of the cached item.

        Returns
        -------
        Any
            The cached object.
        """
        return self._cached[key]

    def _check_size(self):
        """
        Delete item from cache, if it is too large.
        """
        if self._max_size is not None:
            while len(self) > self._max_size:
                self._cached.popitem(last=False)

    def clear(self):
        """
        Clear the cache.
        """
        self._cached = OrderedDict()
        self._hits = 0
        self._misses = 0


class AntialiasAperture:

    def __init__(self, antialias_aperture=(2 / 3., 2 / 3.)):
        self._antialias_aperture = antialias_aperture

    @property
    def antialias_aperture(self) -> Tuple[float, float]:
        """Anti-aliasing aperture as a fraction of the Nyquist frequency."""
        return self._antialias_aperture

    @antialias_aperture.setter
    def antialias_aperture(self, value: Tuple[float, float]):
        self._antialias_aperture = value


class HasAntialiasAperture(HasEventMixin):
    _antialias_aperture: AntialiasAperture

    @property
    @copy_docstring_from(AntialiasAperture.antialias_aperture)
    def antialias_aperture(self) -> Tuple[float, float]:
        return self._antialias_aperture.antialias_aperture

    @antialias_aperture.setter
    def antialias_aperture(self, value: Tuple[float, float]):
        self._antialias_aperture.antialias_aperture = value
