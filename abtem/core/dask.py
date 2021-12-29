from abc import abstractmethod

import dask.array as da
from dask.diagnostics import ProgressBar
import dask
from abtem.core import config
from typing import Union
import numpy as np


class ComputableList(list):

    def _get_computables(self):
        computables = []

        for computable in self:
            if hasattr(computable, 'array'):
                computables.append(computable.array)
            else:
                computables.append(computable)

        return computables

    def compute(self, **kwargs):

        with ProgressBar():
            arrays = dask.compute(self._get_computables(), **kwargs)[0]

        for array, wrapper in zip(arrays, self):
            wrapper._array = array

        return

    def visualize_graph(self, **kwargs):
        return dask.visualize(self._get_computables(), **kwargs)


def _compute(dask_array_wrappers, **kwargs):
    with ProgressBar():
        arrays = dask.compute([wrapper.array for wrapper in dask_array_wrappers], **kwargs)[0]

    for array, wrapper in zip(arrays, dask_array_wrappers):
        wrapper._array = array

    return dask_array_wrappers


def compute(dask_array_wrappers, **kwargs):
    return _compute(dask_array_wrappers, **kwargs)


def computable(func):
    def wrapper(*args, compute=False, **kwargs):
        result = func(*args, **kwargs)

        if isinstance(result, tuple) and compute:
            return _compute(result)

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


def _validate_lazy(lazy):
    if lazy is None:
        return config.get('lazy')

    return lazy


class HasDaskArray:

    def __init__(self, array, **kwargs):
        self._array = array

    @property
    def array(self):
        return self._array

    @property
    def shape(self):
        return self.array.shape

    @property
    def is_lazy(self):
        return isinstance(self.array, da.core.Array)

    @abstractmethod
    def _copy_as_dict(self, copy_array: bool = True) -> dict:
        pass

    # def _map_blocks(self, func, new_cls=None, new_cls_kwargs: dict = None, **kwargs):
    #
    #     def wrapped_func(array, cls, cls_kwargs, **kwargs):
    #         has_dask_array = cls(array=array, **cls_kwargs)
    #         has_dask_array = func(has_dask_array, **kwargs)
    #         return has_dask_array.array
    #
    #     cls_kwargs = self._copy_as_dict(copy_array=False)
    #
    #     array = self.array.map_blocks(wrapped_func,
    #                                   cls=self.__class__,
    #                                   cls_kwargs=cls_kwargs,
    #                                   **kwargs)
    #
    #     if new_cls is None:
    #         new_cls = self.__class__
    #
    #     if new_cls_kwargs is None:
    #         new_cls_kwargs = cls_kwargs
    #
    #     return new_cls(array=array, **new_cls_kwargs)

    def delay(self, chunks=None):
        if self.is_lazy:
            return self

        self._array = da.from_array(self._array, chunks=-1)
        return self

    def compute(self, pbar: bool = True, **kwargs):
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
