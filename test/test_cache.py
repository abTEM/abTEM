import pytest

from abtem.base_classes import Cache, cached_method


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
