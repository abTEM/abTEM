from ..bases import HasCache, Observable, notifying_property, cached_method, cached_method_with_args


def test_has_cache():
    class Dummy(HasCache):

        def __init__(self):
            HasCache.__init__(self)
            self.num_calculate1_calls = 0
            self.num_calculate2_calls = 0

        @cached_method
        def calculate1(self):
            self.num_calculate1_calls += 1
            return 1

        @cached_method
        def calculate2(self):
            self.num_calculate2_calls += 1
            return 2

    dummy = Dummy()

    result = dummy.calculate1()

    assert dummy._cached['calculate1'] == 1
    assert result == 1

    result = dummy.calculate1()

    assert result == 1

    dummy.calculate2()
    dummy.calculate2()

    assert dummy.num_calculate1_calls == 1
    assert dummy.num_calculate2_calls == 1

    dummy.clear_cache()

    dummy.calculate1()

    assert dummy.num_calculate1_calls == 2


def test_has_cache_with_args():
    class Dummy(HasCache):

        def __init__(self):
            HasCache.__init__(self)
            self.num_calculate_calls = 0

        @cached_method_with_args
        def calculate(self, x):
            self.num_calculate_calls += 1
            return 2 * x

    dummy = Dummy()
    result = dummy.calculate(2)

    assert dummy._cached['calculate'][(2,)] == 4
    assert result == 4

    result = dummy.calculate(2)

    assert result == 4
    assert dummy.num_calculate_calls == 1

    result = dummy.calculate(4)

    assert result == 8
    assert dummy.num_calculate_calls == 2

    assert dummy._cached['calculate'][(2,)] == 4
    assert dummy._cached['calculate'][(4,)] == 8

    dummy.calculate(2)
    dummy.clear_cache()
    dummy.calculate(2)

    assert dummy.num_calculate_calls == 3


def test_observe_other():
    class DummyObserveable(Observable):
        def __init__(self, value):
            Observable.__init__(self)

            self._value = value

        value = notifying_property('_value')

    class Dummy(HasCache):

        def __init__(self, value):
            HasCache.__init__(self)

            self.dummy_observeable = DummyObserveable(value)

            self.dummy_observeable.register_observer(self)

            self.num_calculate_calls = 0

        @cached_method
        def calculate(self):
            self.num_calculate_calls += 1
            return self.dummy_observeable.value

    has_cache = Dummy(2)

    result = has_cache.calculate()

    assert result == 2

    has_cache.calculate()

    assert has_cache.num_calculate_calls == 1

    assert has_cache._cached['calculate'] == 2

    has_cache.dummy_observeable.value = 3

    assert has_cache._cached == {}

    result = has_cache.calculate()

    assert result == 3
    assert has_cache.num_calculate_calls == 2


def test_observe_self():
    class Dummy(HasCache, Observable):

        def __init__(self, value):
            HasCache.__init__(self)
            Observable.__init__(self)

            self._value = value
            self.register_observer(self)

            self.num_build_calls = 0

        value = notifying_property('_value')

        @cached_method
        def calculate(self):
            self.num_build_calls += 1
            return self._value

    dummy = Dummy(2)

    result = dummy.calculate()

    assert result == 2

    dummy.calculate()

    assert dummy.num_build_calls == 1

    dummy.value = 3

    result = dummy.calculate()

    assert result == 3
    assert dummy.num_build_calls == 2
