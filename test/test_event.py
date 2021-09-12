from abtem.base_classes import Event, watched_method, watched_property


def test_register_event():
    event = Event()

    num_calls = {}

    def callback(*args):
        num_calls['a'] = 'a'

    event.observe(callback)
    event.notify(None)

    assert event.notify_count == 1
    assert num_calls['a'] == 'a'


def test_watched_method():
    class DummyWithWatchedMethod:

        def __init__(self):
            self._notify_count = 0

            def callback(*args):
                self._notify_count += 1

            self._notified_property = 0

            self.event = Event()
            self.event.observe(callback)

        @watched_method('event')
        def notified_method(self):
            pass

        @property
        def notified_property(self):
            return self._notified_property

        @notified_property.setter
        @watched_property('event')
        def notified_property(self, value):
            self._notified_property = value

    dummy = DummyWithWatchedMethod()
    dummy.notified_method()

    assert dummy._notify_count == 1

    dummy.notified_property = 5
    assert dummy._notify_count == 2
