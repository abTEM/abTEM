import dask.array as da


def computable(func):
    def wrapper(*args, compute=False, **kwargs):
        if compute:
            result = func(*args, **kwargs)
            if isinstance(result, tuple):
                return tuple(partial_result.compute() for partial_result in result)
            else:
                return result.compute()
        else:
            return func(*args, **kwargs)

    return wrapper


def requires_dask_array(func):
    def wrapper(*args, **kwargs):
        args[0].delay()
        return func(*args, **kwargs)

    return wrapper


class BuildsDaskArray:

    def build(self, *args, **kwargs):
        raise NotImplementedError

    def visualize_graph(self, **kwargs):
        self.build(compute=False).visualize_graph(**kwargs)


class HasDaskArray:

    def __init__(self, array):
        self._array = array

    @property
    def array(self):
        return self._array

    def __len__(self):
        return len(self.array)

    @property
    def shape(self):
        return self.array.shape

    def build(self, compute=True):
        pass

    @property
    def is_delayed(self):
        return isinstance(self.array, da.core.Array)

    def delay(self, chunks=None):
        if self.is_delayed:
            return self

        self._array = da.from_array(self._array)
        return self

    def compute(self, **kwargs):
        if not self.is_delayed:
            return self

        self._array = self.array.compute(**kwargs)
        return self

    def visualize_graph(self, **kwargs):
        return self.array.visualize(**kwargs)
