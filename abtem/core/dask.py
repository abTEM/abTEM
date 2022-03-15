from abc import abstractmethod

import dask.array as da
from dask.diagnostics import ProgressBar
import dask
from abtem.core import config
from typing import Union, TYPE_CHECKING, List
import numpy as np
from contextlib import nullcontext

from abtem.core.backend import get_array_module, device_name_from_array_module
from abtem.core.device import HasDeviceMixin

if TYPE_CHECKING:
    from abtem.waves.waves import Waves
    from abtem.measure.detect import AbstractDetector


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
        if config.get('progress_bar'):
            progress_bar = ProgressBar()
        else:
            progress_bar = nullcontext()

        with progress_bar:
            arrays = dask.compute(self._get_computables(), **kwargs)[0]

        for array, wrapper in zip(arrays, self):
            wrapper._array = array

        return

    def visualize_graph(self, **kwargs):
        return dask.visualize(self._get_computables(), **kwargs)


def _compute(dask_array_wrappers, progress_bar=None, **kwargs):
    if progress_bar is None:
        progress_bar = config.get('progress_bar')

    if progress_bar:
        progress_bar = ProgressBar()
    else:
        progress_bar = nullcontext()

    with progress_bar:
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


def validate_lazy(lazy):
    if lazy is None:
        return config.get('dask.lazy')

    return lazy


class HasDaskArray:

    def __init__(self, array, **kwargs):
        self._array = array

    def __len__(self) -> int:
        return len(self.array)

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

    def to_delayed(self):

        def wrap(array, cls, cls_kwargs):
            return cls(array, **cls_kwargs)

        return dask.delayed(wrap)(self.array, self.__class__, self._copy_as_dict(copy_array=False))


    def apply_gufunc(self,
                     func,
                     signature,
                     new_cls=None,
                     new_cls_kwargs=None,
                     axes=None,
                     output_sizes=None,
                     allow_rechunk=False,
                     meta=None,
                     **kwargs):

        if not self.is_lazy:
            return func(self, **kwargs)

        def wrapped_func(array, cls=None, cls_kwargs=None, **kwargs):
            has_dask_array = cls(array=array, **cls_kwargs)
            outputs = func(has_dask_array, **kwargs)

            if len(outputs) == 1:
                return outputs[0].array

            return [output.array for output in outputs]

        cls_kwargs = self._copy_as_dict(copy_array=False)

        arrays = da.apply_gufunc(
            wrapped_func,
            signature,
            self.array,
            output_sizes=output_sizes,
            meta=meta,
            axes=axes,
            allow_rechunk=allow_rechunk,
            cls=self.__class__,
            cls_kwargs=cls_kwargs,
            **kwargs,
        )

        if len(new_cls) > 1:
            new_cls_kwargs = [{**kwargs, 'array': array} for kwargs, array in zip(new_cls_kwargs, arrays)]
        else:
            new_cls_kwargs = [{**new_cls_kwargs[0], 'array': arrays}]

        return tuple(cls(**kwargs) for cls, kwargs in zip(new_cls, new_cls_kwargs))

    def map_blocks(self,
                   func,
                   new_cls=None,
                   new_cls_kwargs: dict = None,
                   dtype=None,
                   name=None,
                   token=None,
                   chunks=None,
                   drop_axis=None,
                   new_axis=None,
                   meta=None,
                   **kwargs):

        if not self.is_lazy:
            return func(self, **kwargs)

        def wrapped_func(array, cls, cls_kwargs, **kwargs):
            has_dask_array = cls(array=array, **cls_kwargs)
            has_dask_array = func(has_dask_array, **kwargs)
            return has_dask_array.array

        cls_kwargs = self._copy_as_dict(copy_array=False)

        array = self.array.map_blocks(wrapped_func,
                                      cls=self.__class__,
                                      cls_kwargs=cls_kwargs,
                                      name=name,
                                      token=token,
                                      dtype=dtype,
                                      chunks=chunks,
                                      drop_axis=drop_axis,
                                      new_axis=new_axis,
                                      meta=meta,
                                      **kwargs)

        if new_cls is None:
            new_cls = self.__class__

        if new_cls_kwargs is None:
            new_cls_kwargs = cls_kwargs

        return new_cls(array=array, **new_cls_kwargs)

    def delay_array(self, chunks=-1):
        if self.is_lazy:
            return self
        d = self._copy_as_dict(copy_array=False)
        d['array'] = da.from_array(self._array, chunks=chunks)
        return self.__class__(**d)

    def compute(self, progress_bar: bool = None, **kwargs):
        if not self.is_lazy:
            return self

        return _compute([self])[0]

    def visualize_graph(self, **kwargs):
        return self.array.visualize(**kwargs)
