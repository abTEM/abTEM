import dask.array as da
from dask.diagnostics import ProgressBar
import dask


def compute_dask_array_wrappers(dask_array_wrappers):
    with ProgressBar():
        arrays = dask.compute([wrapper.array for wrapper in dask_array_wrappers])[0]

    for array, wrapper in zip(arrays, dask_array_wrappers):
        wrapper._array = array

    return dask_array_wrappers


def computable(func):
    def wrapper(*args, compute=False, **kwargs):
        result = func(*args, **kwargs)

        if isinstance(result, tuple) and compute:
            return compute_dask_array_wrappers(result)

        if compute:
            return result.compute()

        return result

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
    def is_lazy(self):
        return isinstance(self.array, da.core.Array)

    def delay(self, chunks=None):
        if self.is_lazy:
            return self

        self._array = da.from_array(self._array, chunks=-1)

        return self

    def compute(self, pbar=True, **kwargs):
        if not self.is_lazy:
            return self

        if pbar:
            with ProgressBar():
                self._array = self.array.compute(**kwargs)
        else:
            self._array = self.array.compute(**kwargs)

        return self

    def visualize_graph(self, **kwargs):
        return self.array.visualize(**kwargs)
