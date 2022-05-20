from abc import abstractmethod
from contextlib import nullcontext
from functools import reduce
from operator import mul
from typing import TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
from dask.array.core import Array, getitem
from dask.core import flatten
from dask.diagnostics import ProgressBar
from dask.highlevelgraph import HighLevelGraph
from dask.array.core import normalize_chunks as dask_normalize_chunks
from dask.utils import parse_bytes

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


def validate_chunks(shape, chunks, limit=None, dtype=None):
    if isinstance(chunks, int):
        limit = chunks
        chunks = ('auto',) * len(shape)

    if any(isinstance(c, str) for c in chunks):
        return auto_chunks(shape, chunks, limit, dtype)

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


def auto_chunks(shape, chunks, limit=None, dtype=None):

    # chunks must be tuple of int

    if limit is None or limit == 'auto':
        if dtype is None:
            raise ValueError

        limit = int(np.floor(parse_bytes(config.get("dask.chunk-size")) / dtype.itemsize))

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


def normalize_chunks(chunks, shape, limit, dtype):
    if limit is None:
        limit = config.get("dask.chunk-size")

    return dask_normalize_chunks(chunks, shape, limit, dtype)


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


class Blocks:

    def __init__(self, shape, chunks, axes_metadata=None):
        self._array = np.empty(shape, dtype=object)
        self._chunks = chunks

        if axes_metadata is not None:
            assert len(axes_metadata) == len(shape)

        self._axes_metadata = axes_metadata

    @property
    def shape(self):
        return self._array.shape

    @property
    def array(self):
        return self._array

    @property
    def dask_array(self):
        return da.from_array(self._array, chunks=self._array.shape)

    @property
    def chunks(self):
        return self._chunks

    def __getitem__(self, item):
        return self._array[item]

    def __setitem__(self, key, value):
        self._array[key] = value


def normalize_axes(axes, num_axes):
    if isinstance(axes, int):
        return num_axes + axes if axes < 0 else axes

    return tuple(num_axes + axis if axis < 0 else axis for axis in axes)


def blockwise(func, arrays, loop_dims, new_dims, new_sizes, metas, **kwargs):
    assert len(new_dims) == len(new_sizes) == len(metas)
    nout = len(new_dims)

    if isinstance(metas, list):
        metas = tuple(metas)

    last_used_symbol = 0
    args = ()
    out_ind = ()
    for dims, array in zip(loop_dims, arrays):
        dims = normalize_axes(dims, len(array.shape))

        ind = tuple(j + last_used_symbol for j in range(len(array.shape)))
        args += (array, ind)
        out_ind += tuple(j + last_used_symbol for j in range(len(array.shape)) if j in dims)
        last_used_symbol += len(ind)

    tmp = da.blockwise(
        func,
        out_ind,
        *args,
        meta=metas,
        concatenate=True,
        **kwargs
    )

    loop_output_shape = tmp.shape
    loop_output_chunks = tmp.chunks
    keys = list(flatten(tmp.__dask_keys__()))
    name, token = keys[0][0].split("-")

    leaf_arrs = []
    for i, (nd, ns, meta) in enumerate(zip(new_dims, new_sizes, metas)):
        normalized_nd = normalize_axes(nd, len(loop_output_shape) + len(nd))

        leaf_name = "%s_%d-%s" % (name, i, token)
        leaf_dsk = {}
        for key in keys:
            indices = list(key[1:])
            output_shape = list(loop_output_shape)
            output_chunks = list(loop_output_chunks)

            for normalized_j, j in sorted(zip(normalized_nd, nd)):
                indices.insert(normalized_j, 0)
                output_shape.insert(normalized_j, ns[j])
                output_chunks.insert(normalized_j, (ns[j],))

            indices = tuple(indices)
            output_shape = tuple(output_shape)
            output_chunks = tuple(output_chunks)

            leaf_dsk[(leaf_name,) + indices] = (getitem, key, i) if nout > 1 else key

        graph = HighLevelGraph.from_collections(leaf_name, leaf_dsk, dependencies=[tmp])

        leaf_arr = Array(
            graph, leaf_name, chunks=output_chunks, shape=output_shape, meta=meta
        )

        leaf_arrs.append(leaf_arr)

    return leaf_arrs  # (*leaf_arrs,) if nout > 1 else leaf_arrs[0]
