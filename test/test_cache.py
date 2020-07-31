import numpy as np
import pytest

from abtem.bases import Cache, cached_method


def test_cache():
    cache = Cache(2)

    with pytest.raises(KeyError):
        cache.retrieve('a')

    cache.insert('a', 'a')
    assert cache.retrieve('a') == 'a'

    cache.insert('b', 'b')
    assert cache.retrieve('b') == 'b'
    assert cache.retrieve('a') == 'a'

    cache.insert('c', 'c')
    with pytest.raises(KeyError):
        cache.retrieve('a')

    cache.clear()
    with pytest.raises(KeyError):
        cache.retrieve('b')


def test_cached_method():
    class DummyClassWithCache:

        def __init__(self):
            self.cache = Cache(2)
            self.call_count = 0

        @cached_method('cache')
        def method1(self, x):
            self.call_count += 1
            return 2 * x

        @cached_method('cache')
        def method2(self, x):
            self.call_count += 1
            return 3 * x

    dummy = DummyClassWithCache()

    assert dummy.method1('a') == 'aa'
    assert dummy.call_count == 1

    dummy.method1('a')
    assert dummy.call_count == 1

    assert dummy.method1('b') == 'bb'
    assert dummy.call_count == 2

    assert dummy.method1('a') == 'aa'
    assert dummy.call_count == 2

    assert dummy.method2('c') == 'ccc'
    assert dummy.call_count == 3

    assert dummy.method1('a') == 'aa'
    assert dummy.call_count == 4

    dummy.cache.clear()
    dummy.method1('a')
    assert dummy.call_count == 5


def test_notify_cache():
    cache = Cache()
    observable = Observable()
    key = 'key'
    condition = 'condition'
    data = np.zeros((2, 2))

    cache.update_cache(key, data, (condition,))

    cache.notify(observable, {'notifier': 'not_condition', 'change': True})
    assert (cache.retrieve_from_cache(key) == data).all()

    cache.notify(observable, {'notifier': condition, 'change': False})
    assert (cache.retrieve_from_cache(key) == data).all()

    cache.notify(observable, {'notifier': condition, 'change': True})
    with pytest.raises(KeyError):
        cache.retrieve_from_cache(key)


def test_multiple_cached_methods():
    class Dummy(Cache):
        call_count1 = 0
        call_count2 = 0
        call_count3 = 0

        @cached_method()
        def get_data1(self):
            self.call_count1 += 1
            return np.zeros((2, 2))

        @cached_method()
        def get_data2(self):
            self.call_count2 += 1
            return np.ones((2, 2))

        @cached_method()
        def get_data3(self):
            self.call_count3 += 1
            return self.get_data1() + self.get_data2()

    dummy = Dummy()
    dummy.get_data1()
    assert dummy.cache.keys() == {'get_data1'}

    dummy.get_data3()
    assert dummy.cache.keys() == {'get_data1', 'get_data2', 'get_data3'}

    assert dummy.call_count1 == dummy.call_count2 == dummy.call_count3 == 1


def test_conditional_cached_method():
    class Dummy(Cache):
        call_count = 0

        @cached_method('condition')
        def get_data(self):
            self.call_count += 1
            return np.zeros((2, 2))

    dummy = Dummy()
    dummy.get_data()

    dummy.notify(Observable(), {'notifier': 'not_condition', 'change': True})
    dummy.get_data()
    assert dummy.call_count == 1

    dummy.notify(Observable(), {'notifier': 'condition', 'change': True})
    dummy.get_data()
    assert dummy.call_count == 2


def test_multiple_conditional_cached_methods():
    class Dummy(Cache):
        call_count1 = 0
        call_count2 = 0
        call_count3 = 0

        @cached_method('condition1')
        def get_data1(self):
            self.call_count1 += 1
            return np.zeros((2, 2))

        @cached_method('condition2')
        def get_data2(self):
            self.call_count2 += 1
            return np.ones((2, 2))

        @cached_method('condition3')
        def get_data3(self):
            self.call_count3 += 1
            return self.get_data1() + self.get_data2()

    dummy = Dummy()
    dummy.get_data3()
    assert dummy.cache.keys() == {'get_data1', 'get_data2', 'get_data3'}

    dummy.notify(Observable(), {'notifier': 'condition2', 'change': True})

    assert dummy.cache.keys() == {'get_data1', 'get_data3'}


def test_cached_method_with_args():
    class Dummy(Cache):
        call_count = 0

        @cached_method_with_args()
        def get_data(self, value):
            self.call_count += 1
            return np.full((2, 2), value)

    dummy = Dummy()

    assert (dummy.get_data(2) == np.full((2, 2), 2)).all()
    assert dummy.call_count == 1

    dummy.get_data(2)
    assert dummy.call_count == 1

    dummy.get_data(3)
    assert dummy.call_count == 2

    dummy.notify(Observable(), {'notifier': 'condition', 'change': True})
    with pytest.raises(KeyError):
        dummy.retrieve_from_cache('get_data')
