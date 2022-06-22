from typing import Optional, Union, Sequence, Any, Callable, Tuple
from collections import defaultdict
import numpy as np

from abtem.core.utils import EqualityMixin


class Events(EqualityMixin):
    """
    Event class for registering callbacks.
    """

    def __init__(self):
        self._callbacks = defaultdict(list)

    def notify(self, message):
        """
        Notify this event. All registered callbacks are called.
        """
        if not len(self._callbacks):
            return

        for prop, callbacks in self._callbacks.items():
            if message['property'] not in prop:
                continue

            for callback in callbacks:
                callback(message)

    def observe(self, callback, props):
        """
        Register new callbacks.

        Parameters
        ----------
        callbacks : callable
            The callbacks to register.
        """
        if not isinstance(props, str):
            props = [props]

        for prop in props:
            self._callbacks[prop].append(callback)


class HasEventsMixin:
    _events: Events

    @property
    def events(self):
        return self._events

    def notify(self, message):
        return self.events.notify(message)

    def observe(self, callback, props):
        self.events.observe(callback, props)


def watch(func):
    name = func.__name__

    def new_func(*args, **kwargs):
        instance = args[0]

        old = getattr(instance, name)
        new = args[1]
        result = func(*args, **kwargs)

        try:
            if np.allclose(old, new):
                return result
        except TypeError:
            if np.all(old == new):
                return result

        message = {'instance': instance, 'property': name, 'old': old, 'new': new}
        getattr(instance, 'events').notify(message)
        return result

    return new_func
