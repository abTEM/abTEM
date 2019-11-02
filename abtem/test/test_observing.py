import pytest
import numpy as np
import mock
from ..bases import Observable, Observer, SelfObservable


def test_register_observer():
    observable = Observable()
    observer1 = Observer()
    observer2 = Observer()

    observable.register_observer(observer1)
    assert observable.observers == [observer1]

    observable.register_observer(observer1)
    assert observable.observers == [observer1]

    observable.register_observer(observer2)
    assert observable.observers == [observer1, observer2]


@mock.patch.object(Observable, 'notify_observers')
def test_notify_observers(mock_notify_observers):
    observable = Observable()
    observer = Observer()

    assert mock_notify_observers.call_count == 0

    observable.register_observer(observer)
    observable.notify_observers('notify')

    assert mock_notify_observers.call_count == 1


@mock.patch.object(Observer, 'notify')
def test_notified(mock_notify):
    observable = Observable()
    observer = Observer()

    assert mock_notify.call_count == 0

    observable.register_observer(observer)
    observable.notify_observers('notify')

    assert mock_notify.call_count == 1


@mock.patch.object(Observer, 'notify')
def test_self_observe(mock_notify):
    class Dummy(Observable, Observer):
        def __init__(self):
            self.register_observer(self)
            super().__init__()

    dummy = Dummy()

    assert mock_notify.call_count == 0

    dummy.notify_observers('notify')

    assert mock_notify.call_count == 1