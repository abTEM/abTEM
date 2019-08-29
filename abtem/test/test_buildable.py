import numpy as np

from ..bases import BuildableArray, Observable, notifying_property


def test_buildable_array():
    class Dummy(BuildableArray):

        def __init__(self, save_array=True):
            BuildableArray.__init__(self, save_array)
            self.num_build_calls = 0

        def check_is_defined(self):
            pass

        def _build(self):
            self.num_build_calls += 1
            return np.full((10, 10), 2, dtype=np.float32)

    buildable_array = Dummy()

    array = buildable_array.get_array()

    assert np.all(array == np.full((10, 10), 2, dtype=np.float32))

    buildable_array.get_array()

    assert buildable_array.num_build_calls == 1

    buildable_array.clear()

    buildable_array.get_array()

    assert buildable_array.num_build_calls == 2

    buildable_array = Dummy(save_array=False)

    array = buildable_array.get_array()

    assert np.all(array == np.full((10, 10), 2, dtype=np.float32))

    buildable_array.get_array()

    assert buildable_array.num_build_calls == 2


def test_observe_other():
    class DummyObserveable(Observable):
        def __init__(self, value):
            Observable.__init__(self)

            self._value = value

        value = notifying_property('_value')

    class Dummy(BuildableArray):

        def __init__(self, value):
            BuildableArray.__init__(self)

            self.dummy_observeable = DummyObserveable(value)

            self.dummy_observeable.register_observer(self)

            self.num_build_calls = 0

        def check_is_defined(self):
            pass

        def _build(self):
            self.num_build_calls += 1
            return np.full((10, 10), self.dummy_observeable.value, dtype=np.float32)

    buildable_array = Dummy(2)

    array = buildable_array.get_array()

    assert np.all(array == np.full((10, 10), 2, dtype=np.float32))

    buildable_array.get_array()

    assert buildable_array.num_build_calls == 1

    buildable_array.dummy_observeable.value = 3

    array = buildable_array.get_array()

    assert np.all(array == np.full((10, 10), 3, dtype=np.float32))
    assert buildable_array.num_build_calls == 2


def test_observe_self():
    class Dummy(BuildableArray, Observable):

        def __init__(self, value):
            BuildableArray.__init__(self)
            Observable.__init__(self)

            self._value = value
            self.register_observer(self)

            self.num_build_calls = 0

        value = notifying_property('_value')

        def check_is_defined(self):
            pass

        def _build(self):
            self.num_build_calls += 1
            return np.full((10, 10), self.value, dtype=np.float32)

    buildable_array = Dummy(2)

    array = buildable_array.get_array()

    assert np.all(array == np.full((10, 10), 2, dtype=np.float32))

    buildable_array.get_array()

    assert buildable_array.num_build_calls == 1

    buildable_array.value = 3

    array = buildable_array.get_array()

    assert np.all(array == np.full((10, 10), 3, dtype=np.float32))
    assert buildable_array.num_build_calls == 2
