from abc import abstractmethod
from contextlib import nullcontext
from functools import reduce
from operator import mul
from typing import TYPE_CHECKING, Tuple, Union

import dask
import dask.array as da
import numpy as np
from dask.array.core import Array, getitem
from dask.core import flatten
from dask.diagnostics import ProgressBar
from dask.highlevelgraph import HighLevelGraph
from dask.array.core import normalize_chunks as dask_normalize_chunks
from dask.utils import parse_bytes
from itertools import accumulate
from abtem.core import config

if TYPE_CHECKING:
    pass


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

        return self

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


def validate_chunks(shape: Tuple[int, ...],
                    chunks: Union[str, int, Tuple[Union[int, str], ...], Tuple[Tuple[int, ...]]],
                    limit: Union[int, str] = None,
                    dtype=None,
                    device='cpu'):
    if isinstance(chunks, int):
        assert limit is None
        limit = chunks
        chunks = ('auto',) * len(shape)
        return auto_chunks(shape, chunks, limit, dtype, device=device)

    if all(isinstance(c, tuple) for c in chunks):
        return chunks

    if any(isinstance(c, str) for c in chunks):
        return auto_chunks(shape, chunks, limit, dtype, device=device)

    validated_chunks = ()
    for s, c in zip(shape, chunks):

        if isinstance(c, tuple):
            if sum(c) != s:
                raise RuntimeError()

            validated_chunks += (c,)

        elif isinstance(c, int):
            if c == -1:
                validated_chunks += ((s,),)
            elif s % c:
                validated_chunks += ((c,) * (s // c) + (s - c * (s // c),),)
            else:
                validated_chunks += ((c,) * (s // c),)
        else:
            raise RuntimeError()

    return validated_chunks


def chunk_range(chunks):
    return tuple(tuple((cumchunks - cc, cumchunks) for cc, cumchunks in zip(c, accumulate(c))) for c in chunks)


def config_chunk_size(device):
    if device == 'gpu':
        return parse_bytes(config.get("dask.chunk-size-gpu"))

    if device != 'cpu':
        raise RuntimeError()

    return parse_bytes(config.get("dask.chunk-size"))


def auto_chunks(shape, chunks, limit=None, dtype=None, device='cpu'):
    if limit is None or limit == 'auto':
        if dtype is None:
            raise ValueError

        limit = int(np.floor(config_chunk_size(device)) / dtype.itemsize)

    elif isinstance(limit, str):
        limit = int(np.floor(parse_bytes(limit) / dtype.itemsize))

    elif not isinstance(limit, int):
        raise ValueError

    normalized_chunks = tuple(s if c == -1 else c for s, c in zip(shape, chunks))

    minimum_chunks = tuple(1 if c == 'auto' else c for s, c in zip(shape, normalized_chunks))
    maximum_chunks = tuple(s if c == 'auto' else c for s, c in zip(shape, normalized_chunks))

    current_chunks = list(minimum_chunks)

    auto = [i for i, c in enumerate(normalized_chunks) if c == 'auto']

    j = 0
    while len(auto):
        auto = [i for i in auto if current_chunks[i] != maximum_chunks[i]]
        if len(auto) == 0:
            break

        j = j % len(auto)

        current_chunks[auto[j]] += 1

        total = reduce(mul, current_chunks)

        if total > limit:
            current_chunks[auto[j]] -= 1
            break

        j += 1

    chunks = validate_chunks(shape, current_chunks, limit, dtype)
    return chunks


def equal_sized_chunks(num_items: int, num_chunks: int = None, chunks: int = None):
    """
    Split an n integer into m (almost) equal integers, such that the sum of smaller integers equals n.

    Parameters
    ----------
    n: int
        The integer to split.
    m: int
        The number integers n will be split into.

    Returns
    -------
    list of int
    """
    if num_items == 0:
        return 0, 0

    if (num_chunks is not None) & (chunks is not None):
        raise RuntimeError()

    if (num_chunks is None) & (chunks is not None):
        num_chunks = (num_items + (-num_items % chunks)) // chunks

    if num_items < num_chunks:
        raise RuntimeError('num_chunks may not be larger than num_items')

    elif num_items % num_chunks == 0:
        return tuple([num_items // num_chunks] * num_chunks)
    else:
        v = []
        zp = num_chunks - (num_items % num_chunks)
        pp = num_items // num_chunks
        for i in range(num_chunks):
            if i >= zp:
                v = [pp + 1] + v
            else:
                v = [pp] + v
        return tuple(v)


class HasDaskArray:

    def __init__(self, array, **kwargs):
        self._array = array

    def __len__(self) -> int:
        return len(self.array)

    @property
    def chunks(self):
        return self.array.chunks

    def rechunk(self, **kwargs):
        return self.array.rechunk(**kwargs)

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

        return _compute([self], **kwargs)[0]

    def visualize_graph(self, **kwargs):
        return self.array.visualize(**kwargs)
