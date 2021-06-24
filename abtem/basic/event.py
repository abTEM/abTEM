from typing import Union, Sequence, Callable

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
