from abtem.bases import Event, watched_method, watched_property


def test_register_event():
    event = Event()

    num_calls = {}

    def callback():
        num_calls['a'] = 'a'

    event.register(callback)
    event.notify()

    assert event.notify_count == 1
    assert num_calls['a'] == 'a'


def test_watched_method():
    class DummyWithWatchedMethod:

        def __init__(self):
            self._notify_count = 0

            def callback(notifier, property_name, change):
                self._notify_count += 1

            self._notified_property = 0

            self.event = Event()
            self.event.register(callback)

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
