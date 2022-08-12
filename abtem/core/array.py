import copy
from abc import abstractmethod
from contextlib import nullcontext
from functools import reduce
from numbers import Number
from operator import mul
from typing import TYPE_CHECKING, Tuple, Union, TypeVar, List, Sequence

import dask
import dask.array as da
import numpy as np
import zarr
from dask.diagnostics import ProgressBar, Profiler
from dask.utils import format_bytes
from tabulate import tabulate

from abtem.core import config
from abtem.core.axes import HasAxes, UnknownAxis, axis_to_dict, axis_from_dict, AxisMetadata, OrdinalAxis
from abtem.core.backend import get_array_module, copy_to_device, device_name_from_array_module
from abtem.core.chunks import Chunks
from abtem.core.utils import normalize_axes, CopyMixin


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
        if config.get('local_diagnostics.progress_bar'):
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


def _compute(dask_array_wrappers, progress_bar: bool = None, **kwargs):
    if progress_bar is None:
        progress_bar = config.get('local_diagnostics.progress_bar')

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


def validate_lazy(lazy):
    if lazy is None:
        return config.get('dask.lazy')

    return lazy


T = TypeVar('T', bound='HasArray')


def format_axes_metadata(axes_metadata, shape):
    data = []
    for axis, n in zip(axes_metadata, shape):
        data += [axis._tabular_repr_data(n)]

    return tabulate(data, headers=['type', 'label', 'coordinates'], tablefmt="simple")


def format_array(array):
    is_lazy = isinstance(array, da.core.Array)

    nbytes = format_bytes(array.nbytes)
    cbytes = format_bytes(np.prod(array.chunksize) * array.dtype.itemsize) if is_lazy else '-'
    chunksize = str(array.chunksize) if is_lazy else '-'
    nchunks = reduce(mul, (len(chunks) for chunks in array.chunks)) if is_lazy else '-'
    meta = array._meta if is_lazy else array
    array_type = f'{type(meta).__module__.split(".")[0]}.{type(meta).__name__}'

    ntasks = f'{len(array.dask)} tasks' if is_lazy else '-'

    data = [
        ['array', nbytes, str(array.shape), ntasks, array.dtype.name],
        ['chunks', cbytes, chunksize, f'{str(nchunks)} chunks', array_type]
    ]
    return tabulate(data, headers=['', 'bytes', 'shape', 'count', 'type'], tablefmt="simple")


def format_type(x):
    module = x.__module__
    qualname = x.__qualname__
    text = f"<{module}.{qualname} object at {hex(id(x))}>"
    return f'{text}\n{"-" * len(text)}'


class HasArray(HasAxes, CopyMixin):
    _array: Union[np.ndarray, da.core.Array]
    _base_dims: int

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __repr__(self):
        formatted_type = format_type(self.__class__)
        axes_table = format_axes_metadata(self.axes_metadata, self.shape)
        array_table = format_array(self.array)
        return '\n\n'.join([formatted_type, axes_table, array_table])

    @classmethod
    def from_array_and_metadata(cls, array, axes_metadata, metadata):
        raise NotImplementedError

    @property
    def metadata(self):
        raise NotImplementedError

    @property
    def base_shape(self) -> Tuple[int, ...]:
        return self.array.shape[-self._base_dims:]

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        return self.array.shape[:-self._base_dims]

    def __len__(self) -> int:
        return len(self.array)

    @property
    def chunks(self):
        return self.array.chunks

    def rechunk(self, *args, **kwargs):
        return self.array.rechunk(*args, **kwargs)

    @property
    def array(self) -> Union[np.ndarray, da.core.Array]:
        return self._array

    @property
    def dtype(self) -> np.dtype.base:
        return self._array.dtype

    @property
    def device(self):
        return device_name_from_array_module(get_array_module(self.array))

    @property
    def is_lazy(self):
        return isinstance(self.array, da.core.Array)

    @classmethod
    def _to_delayed_func(cls, array, **kwargs):
        kwargs['array'] = array

        return cls(**kwargs)

    @property
    def is_complex(self):
        return np.iscomplexobj(self.array)

    def check_is_compatible(self, other: 'HasArray'):
        if not isinstance(other, self.__class__):
            raise RuntimeError(f'incompatible types ({self.__class__} != {other.__class__})')

        if self.shape != other.shape:
            raise RuntimeError(f'incompatible shapes ({self.shape} != {other.shape})')

        for (key, value), (other_key, other_value) in zip(self.copy_kwargs(exclude=('array', 'metadata')).items(),
                                                          other.copy_kwargs(exclude=('array', 'metadata')).items()):
            if np.any(value != other_value):
                raise RuntimeError(f'incompatible values for {key} ({value} != {other_value})')

    def mean(self, axes=None, **kwargs) -> 'T':
        return self._reduction('mean', axes=axes, **kwargs)

    def sum(self, axes=None, **kwargs) -> 'T':
        return self._reduction('sum', axes=axes, **kwargs)

    def std(self, axes=None, **kwargs) -> 'T':
        return self._reduction('std', axes=axes, **kwargs)

    def min(self, axes=None, **kwargs) -> 'T':
        return self._reduction('min', axes=axes, **kwargs)

    def max(self, axes=None, **kwargs) -> 'T':
        return self._reduction('max', axes=axes, **kwargs)

    def _reduction(self, reduction_func, axes, split_every: int = 2) -> 'T':
        xp = get_array_module(self.array)

        if axes is None:
            if self.is_lazy:
                return getattr(da, reduction_func)(self.array)
            else:
                return getattr(xp, reduction_func)(self.array)

        if isinstance(axes, Number):
            axes = (axes,)

        axes = tuple(axis if axis >= 0 else len(self) + axis for axis in axes)

        if self._is_base_axis(axes):
            raise RuntimeError('base axes cannot be reduced')

        ensemble_axes_metadata = copy.deepcopy(self.ensemble_axes_metadata)
        ensemble_axes_metadata = [axis_metadata for axis_metadata, axis in
                                  zip(ensemble_axes_metadata, self.ensemble_axes)
                                  if axis not in axes]

        kwargs = self.copy_kwargs(exclude=('array',))
        if self.is_lazy:
            kwargs['array'] = getattr(da, reduction_func)(self.array, axes, split_every=split_every)
        else:

            kwargs['array'] = getattr(xp, reduction_func)(self.array, axes)

        kwargs['ensemble_axes_metadata'] = ensemble_axes_metadata
        return self.__class__(**kwargs)

    def _arithmetic(self, other, func) -> 'T':
        if hasattr(other, 'array'):
            self.check_is_compatible(other)
            other = other.array
        # else:
        #     try:
        #         other = other.item()
        #     except AttributeError:
        #         pass

        kwargs = self.copy_kwargs(exclude=('array',))
        kwargs['array'] = getattr(self.array, func)(other)
        return self.__class__(**kwargs)

    def _in_place_arithmetic(self, other, func) -> 'T':
        if self.is_lazy or other.is_lazy:
            raise RuntimeError('inplace arithmetic operation not implemented for lazy measurement')
        return self._arithmetic(other, func)

    def __mul__(self, other) -> 'T':
        return self._arithmetic(other, '__mul__')

    def __imul__(self, other) -> 'T':
        return self._in_place_arithmetic(other, '__imul__')

    def __truediv__(self, other) -> 'T':
        return self._arithmetic(other, '__truediv__')

    def __itruediv__(self, other) -> 'T':
        return self._arithmetic(other, '__itruediv__')

    def __sub__(self, other) -> 'T':
        return self._arithmetic(other, '__sub__')

    def __isub__(self, other) -> 'T':
        return self._in_place_arithmetic(other, '__isub__')

    def __add__(self, other) -> 'T':
        return self._arithmetic(other, '__add__')

    def __iadd__(self, other) -> 'T':
        return self._in_place_arithmetic(other, '__iadd__')

    def __pow__(self, other) -> 'T':
        return self._arithmetic(other, '__pow__')

    __rmul__ = __mul__
    __rtruediv__ = __truediv__

    def get_items(self, items, keep_dims: bool = False) -> 'T':
        if isinstance(items, (Number, slice)):
            items = (items,)
        elif not isinstance(items, tuple):
            raise NotImplementedError('indices must be integers or slices, or a tuple of integers or slices')

        if keep_dims:
            items = tuple(slice(item, item + 1) if isinstance(item, int) else item for item in items)

        if isinstance(items, tuple) and len(items) > len(self.ensemble_shape):
            raise RuntimeError('base axes cannot be indexed')

        if any(isinstance(item, (type(...), type(None))) for item in items):
            raise NotImplementedError

        if len(items) > len(self.ensemble_shape):
            raise RuntimeError('base axes cannot be indexed')

        metadata = {}
        axes_metadata = []
        last_indexed = 0
        for axes_metadata_item, item in zip(self.ensemble_axes_metadata, items):
            last_indexed += 1

            indexed_axes_metadata = axes_metadata_item[item]

            if isinstance(item, Number):
                metadata = {**metadata, **indexed_axes_metadata.item_metadata(0)}
            else:
                axes_metadata += [indexed_axes_metadata]

        axes_metadata += self.ensemble_axes_metadata[last_indexed:]

        d = self.copy_kwargs(exclude=('array', 'ensemble_axes_metadata', 'metadata'))
        d['array'] = self._array[items]
        d['ensemble_axes_metadata'] = axes_metadata
        d['metadata'] = {**self.metadata, **metadata}
        return self.__class__(**d)

    def __getitem__(self, items) -> 'T':
        return self.get_items(items)

    def to_delayed(self):
        return dask.delayed(self._to_delayed_func)(self.array, self.copy_kwargs(exclude=('array',)))

    def expand_dims(self, axis: Tuple[int, ...] = None, axis_metadata: List[AxisMetadata] = None) -> 'T':
        if axis is None:
            axis = (0,)

        if type(axis) not in (tuple, list):
            axis = (axis,)

        if axis_metadata is None:
            axis_metadata = [UnknownAxis()] * len(axis)

        axis = normalize_axes(axis, self.shape)

        if any(a >= (len(self.ensemble_shape) + len(axis)) for a in axis):
            raise RuntimeError()

        ensemble_axes_metadata = copy.deepcopy(self.ensemble_axes_metadata)

        for a, am in zip(axis, axis_metadata):
            ensemble_axes_metadata.insert(a, am)

        kwargs = self.copy_kwargs(exclude=('array', 'ensemble_axes_metadata'))
        kwargs['array'] = np.expand_dims(self.array, axis=axis)
        kwargs['ensemble_axes_metadata'] = ensemble_axes_metadata
        return self.__class__(**kwargs)

    def squeeze(self, axis: Tuple[int, ...] = None) -> 'T':
        if len(self.array.shape) < len(self.base_shape):
            return self

        if axis is None:
            axis = range(len(self.shape))
        else:
            axis = normalize_axes(axis, self.shape)

        shape = self.shape[:-len(self.base_shape)]

        squeezed = tuple(np.where([(n == 1) and (i in axis) for i, n in enumerate(shape)])[0])

        xp = get_array_module(self.array)

        kwargs = self.copy_kwargs(exclude=('array', 'ensemble_axes_metadata'))

        kwargs['array'] = xp.squeeze(self.array, axis=squeezed)
        kwargs['ensemble_axes_metadata'] = [element for i, element in enumerate(self.ensemble_axes_metadata) if
                                            i not in squeezed]

        return self.__class__(**kwargs)

    def ensure_lazy(self, chunks='auto') -> 'T':

        if self.is_lazy:
            return self

        chunks = ('auto',) * len(self.ensemble_shape) + (-1,) * len(self.base_shape)

        array = da.from_array(self.array, chunks=chunks)

        return self.__class__(array, **self.copy_kwargs(exclude=('array',)))

    def compute(self, progress_bar: bool = None, **kwargs):
        if not self.is_lazy:
            return self

        return _compute([self], **kwargs)[0]

    def visualize_graph(self, **kwargs):
        return self.array.visualize(**kwargs)

    def copy_to_device(self, device: str) -> 'T':
        """Copy array to specified device."""
        kwargs = self.copy_kwargs(exclude=('array',))
        kwargs['array'] = copy_to_device(self.array, device)
        return self.__class__(**kwargs)

    def to_cpu(self) -> 'T':
        return self.copy_to_device('cpu')

    def to_gpu(self) -> 'T':
        return self.copy_to_device('gpu')

    def to_zarr(self, url: str, compute: bool = True, overwrite: bool = False):
        """
        Write wave functions to a zarr file.

        Parameters
        ----------
        url : str
            Location of the data, typically a path to a local file. A URL can also include a protocol specifier like
            s3:// for remote data.
        overwrite : bool
            If given array already exists, overwrite=False will cause an error, where overwrite=True will replace the
            existing data.
        """

        with zarr.open(url, mode='w') as root:
            waves = self.ensure_lazy()

            array = waves.copy_to_device('cpu').array

            stored = array.to_zarr(url, compute=compute, component='array', overwrite=overwrite)
            for key, value in waves.copy_kwargs(exclude=('array',)).items():
                if key == 'ensemble_axes_metadata':
                    root.attrs[key] = [axis_to_dict(axis) for axis in value]
                else:
                    root.attrs[key] = value

            root.attrs['type'] = self.__class__.__name__

        return stored

    @classmethod
    def from_zarr(cls, url, chunks: int = 'auto') -> 'T':
        """
        Read wave functions from a hdf5 file.

        url : str
            Location of the data, typically a path to a local file. A URL can also include a protocol specifier like
            s3:// for remote data.
        chunks : int, optional
        """

        with zarr.open(url, mode='r') as f:
            kwargs = {}

            for key, value in f.attrs.items():
                if key == 'ensemble_axes_metadata':
                    ensemble_axes_metadata = [axis_from_dict(d) for d in value]
                elif key == 'type':
                    pass
                #    cls = globals()[value]
                else:

                    kwargs[key] = value

        if chunks == 'auto':
            chunks = ('auto',) * len(ensemble_axes_metadata) + (-1,) * cls._base_dims

        array = da.from_zarr(url, component='array', chunks=chunks)
        return cls(array, ensemble_axes_metadata=ensemble_axes_metadata, **kwargs)


def from_zarr(url: str, chunks: Chunks = None):
    import abtem

    with zarr.open(url, mode='r') as f:
        name = f.attrs['type']

    cls = getattr(abtem, name)
    return cls.from_zarr(url, chunks)


def stack(has_arrays: Sequence[HasArray], axis_metadata: AxisMetadata = None, axis: int = 0) -> 'T':
    if axis_metadata is None:
        axis_metadata = UnknownAxis()

    elif isinstance(axis_metadata, (tuple, list)):
        if not all(isinstance(element, str) for element in axis_metadata):
            raise ValueError()
        axis_metadata = OrdinalAxis(values=axis_metadata)

    xp = get_array_module(has_arrays[0].array)

    assert axis <= len(has_arrays[0].ensemble_shape)

    if has_arrays[0].is_lazy:
        array = da.stack([measurement.array for measurement in has_arrays], axis=axis)
    else:
        array = xp.stack([measurement.array for measurement in has_arrays], axis=axis)

    cls = has_arrays[0].__class__
    kwargs = has_arrays[0].copy_kwargs(exclude=('array',))

    kwargs['array'] = array
    kwargs['ensemble_axes_metadata'] = [axis_metadata] + kwargs['ensemble_axes_metadata']
    return cls(**kwargs)


def concatenate(has_arrays: Sequence[HasArray], axis: bool = 0) -> 'T':
    xp = get_array_module(has_arrays[0].array)

    if has_arrays[0].is_lazy:
        array = da.concatenate([has_array.array for has_array in has_arrays], axis=axis)
    else:
        array = xp.concatenate([has_array.array for has_array in has_arrays], axis=axis)

    cls = has_arrays[0].__class__

    concatenated_axes_metadata = has_arrays[0].axes_metadata[axis]
    for has_array in has_arrays[1:]:
        concatenated_axes_metadata = concatenated_axes_metadata.concatenate(has_array.axes_metadata[axis])

    axes_metadata = copy.deepcopy(has_arrays[0].axes_metadata)
    axes_metadata[axis] = concatenated_axes_metadata

    return cls.from_array_and_metadata(array=array,
                                       axes_metadata=axes_metadata,
                                       metadata=has_arrays[0].metadata)
