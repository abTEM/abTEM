"""Module for describing array objects."""

from __future__ import annotations

import copy
import json
import warnings
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager, nullcontext
from functools import partial
from numbers import Number
from types import ModuleType
from typing import (TYPE_CHECKING, Any, Callable, Generator, Optional, Self,
                    Sequence, TypeVar, Union)

import dask
import dask.array as da
import numpy as np
from dask.array.utils import validate_axis
from dask.diagnostics import Profiler, ProgressBar, ResourceProfiler
from tqdm.dask import TqdmCallback

from abtem._version import __version__
from abtem.core import config
from abtem.core.axes import (AxesMetadataList, AxisMetadata, LinearAxis,
                             OrdinalAxis, UnknownAxis, axis_from_dict,
                             axis_to_dict)
from abtem.core.backend import (check_cupy_is_installed, copy_to_device, cp,
                                device_name_from_array_module,
                                get_array_module)
from abtem.core.chunks import Chunks, iterate_chunk_ranges, validate_chunks
from abtem.core.ensemble import (Ensemble, _wrap_with_array,
                                 unpack_blockwise_args)
from abtem.core.utils import (CopyMixin, EqualityMixin, interleave, itemset,
                              normalize_axes, number_to_tuple, tuple_range)
from abtem.transform import TransformFromFunc

if TYPE_CHECKING:
    from abtem.transform import ArrayObjectTransform


tifffile: Optional[ModuleType] = None
try:
    import tifffile  # type: ignore
except ImportError:
    pass


hs: Optional[ModuleType] = None
try:
    import hyperspy.api as hs  # type: ignore
except ImportError:
    pass


xr: Optional[ModuleType] = None
try:
    import xarray as xr  # type: ignore
except ImportError:
    pass


ArrayObjectType = TypeVar("ArrayObjectType", bound="ArrayObject")
ArrayObjectTypeAlt = TypeVar("ArrayObjectTypeAlt", bound="ArrayObject")
ArrayItemType = Union[int, slice, list, np.ndarray, None]


def _to_hyperspy_axes_metadata(
    axes_metadata: list[AxisMetadata], shape: int | tuple[int, ...]
):
    hyperspy_axes = []

    if not isinstance(shape, (list, tuple)):
        shape = (shape,)

    for metadata, n in zip(axes_metadata, shape):
        hyperspy_axis = {"size": n, "name": metadata.label}

        if isinstance(metadata, LinearAxis):
            hyperspy_axis["scale"] = metadata.sampling
            hyperspy_axis["offset"] = metadata.offset
            hyperspy_axis["units"] = metadata.units
        elif isinstance(metadata, OrdinalAxis):
            if all(isinstance(value, Number) for value in metadata.values) and (
                all(
                    metadata.values[i] < metadata.values[i + 1]
                    for i in range(len(metadata.values) - 1)
                )
            ):
                hyperspy_axis["axis"] = metadata.values
                hyperspy_axis["units"] = metadata.units
            else:
                warnings.warn(
                    f"Axis ({metadata.label}) not supported by hyperspy, some metadata"
                    f"will be lost."
                )
        else:
            raise RuntimeError()

        hyperspy_axes.append(hyperspy_axis)

    return hyperspy_axes


def _extract_blockwise_multi_output(arr: np.ndarray, index: int) -> np.ndarray:
    arr = arr.item()[index]
    return arr


def multi_output_blockwise(
    func: Callable,
    array: da.core.Array,
    chunks: tuple[tuple[int, ...], ...],
    array_axes: tuple[da.core.Array, ...],
    new_axes: tuple[da.core.Array, ...],
    out_metas: tuple,
    drop_axes: tuple[tuple[int, ...], ...],
    new_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
) -> tuple[da.core.Array, ...]:
    assert len(out_metas) == len(drop_axes)

    base_ndim = array.ndim
    new_ndim = sum(len(axis.shape) for axis in new_axes)

    out_ndim = new_ndim + base_ndim
    old_chunks = array.chunks

    assert isinstance(array_axes, tuple)
    assert len(array_axes) == base_ndim
    assert all(len(base_axis.numblocks) == 1 for base_axis in array_axes)
    assert all(
        base_axis.numblocks[0] == len(c) for base_axis, c in zip(array_axes, old_chunks)
    )

    out_symbols = tuple(range(out_ndim))
    array_symbols = tuple(range(new_ndim, out_ndim))

    old_axes_symbols = tuple((i,) for i in array_symbols)
    old_axes_args = interleave(array_axes, old_axes_symbols)

    new_axes_args: tuple[da.core.Array | tuple[int, ...], ...] = ()
    current_symbol = 0
    for axis in new_axes:
        new_axes_args += (
            axis,
            tuple(range(current_symbol, current_symbol + len(axis.shape))),
        )
        current_symbol += len(axis.shape)

    out_array = da.blockwise(
        func,
        out_symbols,
        array,
        array_symbols,
        *old_axes_args,
        *new_axes_args,
        align_arrays=False,
        meta=np.array((), dtype=object),
        **kwargs,
    )

    outputs: tuple[da.core.Array, ...] = ()
    for i, (drop_axis, out_meta, new_shape) in enumerate(
        zip(drop_axes, out_metas, new_shapes)
    ):
        if not all(len(out_array.chunks[i]) == 1 for i in drop_axis):
            raise RuntimeError()

        drop_chunks = []
        for j, (item, ns) in enumerate(zip(chunks, new_shape)):
            if j not in drop_axis:
                if sum(item) != ns:
                    assert len(item) == 1
                    item = (ns,)
                drop_chunks.append(item)
        drop_chunks = tuple(drop_chunks)

        new_output = da.map_blocks(
            _extract_blockwise_multi_output,
            out_array,
            chunks=drop_chunks,
            drop_axis=drop_axis,
            index=i,
            meta=out_meta,
        )
        outputs += (new_output,)
    return outputs


class ComputableList(list):
    """A list with methods for conveniently computing its items."""

    def to_zarr(
        self,
        url: str,
        compute: bool = True,
        overwrite: bool = False,
        progress_bar: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Write data to a zarr file.
        Parameters
        ----------
        url : str
            Location of the data. For Zarr 3 zip stores, use a .zip extension.
            For directory stores, use .zarr extension or a directory path.
        compute : bool
            If true compute immediately; return dask.delayed.Delayed otherwise.
        overwrite : bool
            If given array already exists, overwrite=False will cause an error, where
            overwrite=True will replace the existing data.
        progress_bar : bool
            Display a progress bar in the terminal or notebook during computation. The
            progress bar is only displayed with a local scheduler.
        kwargs :
            Keyword arguments passed to `dask.array.to_zarr`.
        """
        import os

        import zarr

        # Helper functions for type preservation
        def encode_types(obj):
            """Recursively encode tuples for JSON serialization."""
            if isinstance(obj, tuple):
                return {
                    "_type": "tuple",
                    "_value": [encode_types(item) for item in obj],
                }
            elif isinstance(obj, list):
                return [encode_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: encode_types(value) for key, value in obj.items()}
            else:
                return obj

        # Determine if this is a zip file
        is_zip = url.endswith(".zip")

        arrays_to_write = []
        metadata_list = []

        for i, has_array in enumerate(self):
            has_array = has_array.ensure_lazy()
            array = has_array.copy_to_device("cpu").array

            packed_kwargs = has_array._pack_kwargs(
                has_array._copy_kwargs(exclude=("array",))
            )

            # Encode tuples in packed_kwargs
            packed_kwargs = encode_types(packed_kwargs)

            arrays_to_write.append((i, array))
            metadata_list.append(
                {f"kwargs{i}": packed_kwargs, f"type{i}": has_array.__class__.__name__}
            )

        if is_zip:
            # Use ZipStore for .zip files
            @dask.delayed
            def write_to_zipstore(computed_arrays, url, metadata_list, overwrite):

                if overwrite and os.path.exists(url):
                    os.remove(url)

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Duplicate name:.*zarr.json",
                        category=UserWarning,
                    )

                    store = zarr.storage.ZipStore(url, mode="w")

                    try:
                        root = zarr.group(store=store, overwrite=True)

                        for metadata_dict in metadata_list:
                            for key, value in metadata_dict.items():
                                root.attrs[key] = value

                        for i, computed_array in computed_arrays:
                            root.create_array(
                                name=f"array{i}",
                                data=computed_array,
                                chunks=computed_array.shape,
                                overwrite=True,
                            )
                    finally:
                        store.close()

                return url

            delayed_arrays = [
                (i, dask.delayed(array.compute)()) for i, array in arrays_to_write
            ]
            delayed_write = write_to_zipstore(
                delayed_arrays, url, metadata_list, overwrite
            )

        else:
            # Use directory store for non-.zip files
            @dask.delayed
            def write_to_directory(computed_arrays, url, metadata_list, overwrite):
                import shutil

                if overwrite and os.path.exists(url):
                    shutil.rmtree(url)

                root = zarr.open(url, mode="w")

                try:
                    for metadata_dict in metadata_list:
                        for key, value in metadata_dict.items():
                            root.attrs[key] = value

                    for i, computed_array in computed_arrays:
                        root.create_array(
                            name=f"array{i}",
                            data=computed_array,
                            chunks=computed_array.shape,
                            overwrite=True,
                        )
                finally:
                    store = getattr(root, "store", None)
                    if store is not None:
                        close_fn = getattr(store, "close", None)
                        if callable(close_fn):
                            close_fn()

                return url

            delayed_arrays = [
                (i, dask.delayed(array.compute)()) for i, array in arrays_to_write
            ]
            delayed_write = write_to_directory(
                delayed_arrays, url, metadata_list, overwrite
            )

        if not compute:
            return delayed_write

        with _compute_context(
            progress_bar, profiler=False, resource_profiler=False
        ) as (_, profiler, resource_profiler):
            output = dask.compute(delayed_write, **kwargs)[0]

        profilers = tuple(p for p in (profiler, resource_profiler) if p is not None)
        if profilers:
            return output, profilers
        else:
            return output

    def compute(self, **kwargs) -> list[ArrayObject] | tuple[list[ArrayObject], tuple]:
        """Turn a list of lazy ArrayObjects object into the in-memory equivalents.

        kwargs :
            Keyword arguments passed to `ArrayObject.compute`.
        """

        output, profilers = _compute(self, **kwargs)

        if profilers:
            return output, profilers

        return output


def _get_progress_bar(
    progress_bar: Optional[bool] = None,
) -> Union[ProgressBar, TqdmCallback, nullcontext]:
    if progress_bar is None:
        progress_bar = config.get("diagnostics.progress_bar")

    progress_bar_obj: Union[ProgressBar, TqdmCallback, nullcontext]

    if progress_bar:
        if progress_bar == "tqdm":
            progress_bar_obj = TqdmCallback(desc="tasks")
        else:
            progress_bar_obj = ProgressBar()
    else:
        progress_bar_obj = nullcontext()

    return progress_bar_obj


@contextmanager
def _compute_context(
    progress_bar: Optional[bool] = None,
    profiler: int = False,
    resource_profiler: int = False,
) -> Generator[tuple[Any, Any, Any], None, None]:
    progress_bar_ctx = _get_progress_bar(progress_bar)
    profiler_ctx: Union[Profiler, nullcontext]
    resource_profiler_ctx: Union[ResourceProfiler, nullcontext]

    if profiler:
        profiler_ctx = Profiler()
    else:
        profiler_ctx = nullcontext()

    if resource_profiler:
        resource_profiler_ctx = ResourceProfiler()
    else:
        resource_profiler_ctx = nullcontext()

    with (
        progress_bar_ctx as progress_bar_ctx1,
        profiler_ctx as profiler_ctx1,
        resource_profiler_ctx as resource_profiler_ctx1,
    ):
        yield progress_bar_ctx1, profiler_ctx1, resource_profiler_ctx1


def _compute(
    array_objects: list[ArrayObjectType],
    progress_bar: Optional[bool] = None,
    profiler: bool = False,
    resource_profiler: bool = False,
    **kwargs,
) -> tuple[list[ArrayObjectType], tuple]:
    if config.get("device") == "gpu":
        check_cupy_is_installed()

        if "num_workers" not in kwargs:
            kwargs["num_workers"] = cp.cuda.runtime.getDeviceCount()

        if "threads_per_worker" not in kwargs:
            kwargs["threads_per_worker"] = cp.cuda.runtime.getDeviceCount()

    with _compute_context(
        progress_bar, profiler=profiler, resource_profiler=resource_profiler
    ) as (_, profiler, resource_profiler):
        arrays = dask.compute([wrapper.array for wrapper in array_objects], **kwargs)[0]

    for array, wrapper in zip(arrays, array_objects):
        wrapper._array = array

    profilers = tuple(p for p in (profiler, resource_profiler) if p is not None)

    return array_objects, profilers


def validate_lazy(lazy: Optional[bool]) -> bool:
    """
    Validate the lazy argument.

    Parameters
    ----------
    lazy : bool or None
        If None, return the default value from the configuration. If a boolean, return
        the value.

    Returns
    -------
    lazy : bool
        The validated value of lazy.
    """
    if lazy is None:
        return config.get("dask.lazy")
    elif not isinstance(lazy, bool):
        raise ValueError("lazy must be a boolean")

    return lazy


def _validate_array_items(
    items: ArrayItemType | tuple[ArrayItemType, ...],
    shape: tuple[int, ...],
    keepdims: bool = False,
) -> tuple[ArrayItemType, ...]:
    if isinstance(items, (int, slice, type(None), list, np.ndarray)):
        items = (items,)

    elif not isinstance(items, tuple):
        raise NotImplementedError(
            (
                "Indices must be integers or slices or a tuple of integers or"
                f" slices or None, not {type(items).__name__}."
            )
        )

    if keepdims:
        items = tuple(
            slice(item, item + 1) if isinstance(item, int) else item for item in items
        )

    if any(isinstance(item, (type(...),)) for item in items):
        raise NotImplementedError

    if len(tuple(item for item in items if item is not None)) > len(shape):
        raise RuntimeError("too many indices for array")

    return items


class ArrayObject(Ensemble, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """A base class for simulation objects described by an array and associated
    metadata.

    Parameters
    ----------
    array : ndarray
        Array representing the array object.
    ensemble_axes_metadata : list of AxesMetadata
        Axis metadata for each ensemble axis. The axis metadata must be compatible with
        the shape of the array.
    metadata : dict
        A dictionary defining wave function metadata. All items will be added to the
        metadata of measurements derived from the waves.
    """

    _base_dims: int

    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        ensemble_axes_metadata: list[AxisMetadata] | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        if ensemble_axes_metadata is None:
            ensemble_axes_metadata = []

        if metadata is None:
            metadata = {}

        self._array = array
        self._ensemble_axes_metadata = ensemble_axes_metadata
        self._metadata = metadata

        if len(array.shape) < self._base_dims:
            raise RuntimeError(
                f"{self.__class__.__name__} must be {self._base_dims}D or greater, not "
                f"{len(array.shape)}D"
            )

        self._check_axes_metadata()

        super().__init__(**kwargs)

    @property
    def base_dims(self) -> int:
        """Number of base dimensions."""
        return self._base_dims

    @property
    def ensemble_dims(self) -> int:
        """Number of ensemble dimensions."""
        return len(self.shape) - self.base_dims

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        return [UnknownAxis() for _ in range(self._base_dims)]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying array."""
        return self.array.shape

    @property
    def base_shape(self) -> tuple[int, ...]:
        """Shape of the base axes of the underlying array."""
        return self.shape[self.ensemble_dims :]

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        """Shape of the ensemble axes of the underlying array."""
        return self.shape[: self.ensemble_dims]

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        """List of AxisMetadata of the ensemble axes."""
        return self._ensemble_axes_metadata

    @property
    def axes_metadata(self) -> AxesMetadataList:
        """List of AxisMetadata."""
        return AxesMetadataList(
            self.ensemble_axes_metadata + self.base_axes_metadata, self.shape
        )

    def _check_axes_metadata(self) -> None:
        if len(self.shape) != len(self.axes_metadata):
            raise RuntimeError(
                f"number of array dimensions ({len(self.shape)}) does not match number"
                f"of axis metadata items ({len(self.axes_metadata)})"
            )

        for n, axis in zip(self.shape, self.axes_metadata):
            if isinstance(axis, OrdinalAxis) and len(axis) != n:
                raise RuntimeError(
                    f"number of values for ordinal axis ({len(axis)}), does not match"
                    f"size of dimension ({n})"
                )

    def _is_base_axis(self, axis: int | tuple[int, ...]) -> bool:
        axis = number_to_tuple(axis)
        base_axes = tuple(range(len(self.ensemble_shape), len(self.shape)))
        return len(set(axis).intersection(base_axes)) > 0

    def apply_func(self, func: Callable, **kwargs) -> Self:
        """Apply a function to the array object. The function must take an array as its
        first argument, only the array is modified, the metadata is not changed. The
        function is applied lazily if the array object is lazy.

        Parameters
        ----------
        func : callable
            Function to apply to the array object.
        kwargs :
            Additional keyword arguments passed to the function.

        Returns
        -------
        array_object : ArrayObject or subclass of ArrayObject
            The array object with the function applied.
        """
        transform = TransformFromFunc(func, func_kwargs=kwargs)
        return transform.apply(self)

    def get_from_metadata(self, name: str, broadcastable: bool = False):
        axes_metadata_index = None
        data = None
        for i, (n, axis) in enumerate(zip(self.shape, self.ensemble_axes_metadata)):
            if axis.label == name:
                data = axis.coordinates(n)
                axes_metadata_index = i

        if axes_metadata_index is not None and broadcastable:
            return np.array(data)[
                (
                    *((None,) * axes_metadata_index),
                    slice(None),
                    *((None,) * (len(self.ensemble_shape) - 1 - axes_metadata_index)),
                )
            ]
        elif axes_metadata_index is not None:
            if name in self.metadata.keys():
                raise RuntimeError(
                    f"Could not resolve metadata for {name}, found in both ensemble"
                    "axes metadata and metadata"
                )

            return data
        else:
            try:
                return self.metadata[name]
            except KeyError as exc:
                raise RuntimeError(f"Could not resolve metadata for {name}") from exc

    @classmethod
    @abstractmethod
    def from_array_and_metadata(
        cls,
        array: np.ndarray | da.core.Array,
        axes_metadata: list[AxisMetadata],
        metadata: dict,
    ) -> Self:
        """Creates array object from a given array and metadata.

        Parameters
        ----------
        array : ndarray
            Array defining the array object.
        axes_metadata : list of AxesMetadata
            Axis metadata for each axis. The axis metadata must be compatible with the
            shape of the array.
        metadata :
            A dictionary defining the metadata of the array object.

        Returns
        -------
        array_object : ArrayObject or subclass of ArrayObject
            The array object.
        """

    def rechunk(self, chunks: Chunks, **kwargs) -> ArrayObject:
        """Rechunk dask array.

        chunks : int or tuple or str
            How to rechunk the array. See `dask.array.rechunk`.
        kwargs :
            Additional keyword arguments passes to `dask.array.rechunk`.
        """

        if not self.is_lazy:
            raise RuntimeError("cannot rechunk non-lazy array object")

        if isinstance(chunks, tuple) and len(chunks) < len(self.shape):
            chunks = chunks + ("auto",) * max((self.ensemble_dims - len(chunks), 0))
            chunks = chunks + (-1,) * max((len(self.shape) - len(chunks), 0))

        array = self._lazy_array.rechunk(chunks=chunks, **kwargs)
        kwargs = self._copy_kwargs(exclude=("array",))
        return self.__class__(array, **kwargs)

    @property
    def metadata(self) -> dict:
        """Metadata stored as a dictionary."""
        return self._metadata

    def __len__(self) -> int:
        return len(self.array)

    @property
    def array(self) -> np.ndarray | da.core.Array:
        """Underlying array describing the array object."""
        return self._array

    @array.setter
    def array(self, array: np.ndarray | da.core.Array):
        """Set underlying array describing the array object."""
        if not array.shape == self.shape:
            raise ValueError("Shape of array must match shape of object.")

        if not array.dtype == self.dtype:
            raise ValueError("Datatype of array must match datatype of object.")

        if self.is_lazy != isinstance(array, da.core.Array):
            raise ValueError("Type of array must match type of object.")

        self._array = array

    @property
    def _lazy_array(self) -> da.core.Array:
        """Underlying lazy array describing the array object."""
        if not self.is_lazy:
            raise RuntimeError("array object is not lazy")
        assert isinstance(self.array, da.core.Array)
        return self.array

    @property
    def _eager_array(self) -> np.ndarray:
        """Underlying eager array describing the array object."""
        if self.is_lazy:
            raise RuntimeError("array object is lazy")
        assert not isinstance(self.array, da.core.Array)
        return self.array

    @property
    def dtype(self) -> np.dtype:
        """Datatype of array."""
        return self._array.dtype

    @property
    def device(self) -> str:
        """The device where the array is stored."""
        return device_name_from_array_module(get_array_module(self.array))

    @property
    def is_lazy(self) -> bool:
        """True if array is lazy."""
        return isinstance(self.array, da.core.Array)

    @classmethod
    def _to_delayed_func(cls, array, **kwargs):
        kwargs["array"] = array
        return cls(**kwargs)

    @property
    def is_complex(self) -> bool:
        """True if array is complex."""
        return np.iscomplexobj(self.array)

    def _check_is_compatible(self, other: ArrayObject):
        if not isinstance(other, self.__class__):
            raise RuntimeError(
                f"incompatible types ({self.__class__} != {other.__class__})"
            )

    def mean(
        self,
        axis: Optional[int | tuple[int, ...]] = None,
        keepdims: bool = False,
        split_every: int = 2,
    ) -> Self:
        """Mean of array object over one or more axes. Only ensemble axes can be
        reduced.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which a means are calculated. The default is to compute
            the mean of the flattened array. If this is a tuple of ints, the mean is
            calculated over multiple axes. The indicated axes must be ensemble axes.
        keepdims : bool, optional
            If True, the reduced axes are left in the result as dimensions with size
            one. Default is False.
        split_every : int
            Only used for lazy arrays. See `dask.array.reductions`.

        Returns
        -------
        reduced_array : ArrayObject or subclass of ArrayObject
            The reduced array object.
        """
        return self._reduction(
            "mean", axes=axis, keepdims=keepdims, split_every=split_every
        )

    def sum(
        self,
        axis: Optional[int | tuple[int, ...]] = None,
        keepdims: bool = False,
        split_every: int = 2,
    ) -> ArrayObject:
        """Sum of array object over one or more axes. Only ensemble axes can be reduced.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which a sums are performed. The default is to compute the
            mean of the flattened array. If this is a tuple of ints, the sum is
            performed over multiple axes. The indicated axes must be ensemble axes.
        keepdims : bool, optional
            If True, the reduced axes are left in the result as dimensions with size
            one. Default is False.
        split_every : int
            Only used for lazy arrays. See `dask.array.reductions`.

        Returns
        -------
        reduced_array : ArrayObject or subclass of ArrayObject
            The reduced array object.
        """
        return self._reduction(
            "sum", axes=axis, keepdims=keepdims, split_every=split_every
        )

    def std(
        self,
        axis: Optional[int | tuple[int, ...]] = None,
        keepdims: bool = False,
        split_every: int = 2,
    ) -> ArrayObject:
        """Standard deviation of array object over one or more axes. Only ensemble axes
        can be reduced.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which a standard deviations are calculated. The default
            is to compute the mean of the flattened array. If this is a tuple of ints,
            the standard deviations are calculated over multiple axes.
            The indicated axes must be ensemble axes.
        keepdims : bool, optional
            If True, the reduced axes are left in the result as dimensions with size
            one. Default is False.
        split_every : int
            Only used for lazy arrays. See `dask.array.reductions`.

        Returns
        -------
        reduced_array : ArrayObject or subclass of ArrayObject
            The reduced array object.
        """
        return self._reduction(
            "std", axes=axis, keepdims=keepdims, split_every=split_every
        )

    def min(
        self,
        axis: Optional[int | tuple[int, ...]] = None,
        keepdims: bool = False,
        split_every: int = 2,
    ) -> ArrayObject:
        """Minmimum of array object over one or more axes. Only ensemble axes can be
        reduced.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which a minima are calculated. The default is to compute
            the mean of the flattened array. If this is a tuple of ints, the minima are
            calculated over multiple axes. The indicated axes must be ensemble axes.
        keepdims : bool, optional
            If True, the reduced axes are left in the result as dimensions with size
            one. Default is False.
        split_every : int
            Only used for lazy arrays. See `dask.array.reductions`.

        Returns
        -------
        reduced_array : ArrayObject or subclass of ArrayObject
            The reduced array object.
        """
        return self._reduction(
            "min", axes=axis, keepdims=keepdims, split_every=split_every
        )

    def max(
        self,
        axis: Optional[int | tuple[int, ...]] = None,
        keepdims: bool = False,
        split_every: int = 2,
    ) -> ArrayObject:
        """Maximum of array object over one or more axes. Only ensemble axes can be
        reduced.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which a maxima are calculated. The default is to compute
            the mean of the flattened array. If this is a tuple of ints, the maxima are
            calculated over multiple axes. The indicated axes must be ensemble axes.
        keepdims : bool, optional
            If True, the reduced axes are left in the result as dimensions with size
            one. Default is False.
        split_every : int
            Only used for lazy arrays. See `dask.array.reductions`.

        Returns
        -------
        reduced_array : ArrayObject or subclass of ArrayObject
            The reduced array object.
        """
        return self._reduction(
            "max", axes=axis, keepdims=keepdims, split_every=split_every
        )

    def _reduction(
        self,
        reduction_func: str,
        axes: Optional[int | tuple[int, ...]] = None,
        keepdims: bool = False,
        split_every: int = 2,
        **kwargs,
    ) -> Self:
        xp = get_array_module(self.array)

        if axes is None:
            if self.is_lazy:
                return getattr(da, reduction_func)(self.array)
            else:
                return getattr(xp, reduction_func)(self.array)

        axes = number_to_tuple(axes)

        axes = tuple(axis if axis >= 0 else len(self.shape) + axis for axis in axes)

        if self._is_base_axis(axes):
            raise RuntimeError("base axes cannot be reduced")

        ensemble_axes_metadata = copy.deepcopy(self.ensemble_axes_metadata)
        if not keepdims:
            ensemble_axes = tuple(range(len(self.ensemble_shape)))
            ensemble_axes_metadata = [
                axis_metadata
                for axis_metadata, axis in zip(ensemble_axes_metadata, ensemble_axes)
                if axis not in axes
            ]

        default_kwargs = self._copy_kwargs(exclude=("array",))

        kwargs = {**default_kwargs, **kwargs}

        if self.is_lazy:
            kwargs["array"] = getattr(da, reduction_func)(
                self.array, axes, split_every=split_every, keepdims=keepdims
            )
        else:
            kwargs["array"] = getattr(xp, reduction_func)(
                self.array, axes, keepdims=keepdims
            )

        kwargs["ensemble_axes_metadata"] = ensemble_axes_metadata
        return self.__class__(**kwargs)

    def _arithmetic(
        self, other: Self | np.ndarray | da.core.Array | Number, func: str
    ) -> Self:
        other_array: np.ndarray | da.core.Array | Number
        if isinstance(other, self.__class__):
            self._check_is_compatible(other)
            other_array = other.array
        elif isinstance(other, (np.ndarray, da.core.Array, Number)):
            other_array = other
        else:
            raise NotImplementedError(
                f"arithmetic operation not implemented for {type(other).__name__}"
            )

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = getattr(self.array, func)(other_array)
        return self.__class__(**kwargs)

    def _in_place_arithmetic(
        self, other: Self | np.ndarray | da.core.Array | Number, func: str
    ) -> Self:
        if (
            self.is_lazy
            or (hasattr(other, "is_lazy") and other.is_lazy)
            or isinstance(other, da.core.Array)
        ):
            raise RuntimeError(
                "inplace arithmetic operation not implemented for lazy measurement or"
                " arrays"
            )
        return self._arithmetic(other, func)

    def __mul__(self, other: Self) -> Self:
        return self._arithmetic(other, "__mul__")

    def __imul__(self, other: Self) -> Self:
        return self._in_place_arithmetic(other, "__imul__")

    def __truediv__(self, other: Self) -> Self:
        return self._arithmetic(other, "__truediv__")

    def __itruediv__(self, other: Self) -> Self:
        return self._arithmetic(other, "__itruediv__")

    def __sub__(self, other: Self) -> Self:
        return self._arithmetic(other, "__sub__")

    def __isub__(self, other: Self) -> Self:
        return self._in_place_arithmetic(other, "__isub__")

    def __add__(self, other: Self) -> Self:
        return self._arithmetic(other, "__add__")

    def __iadd__(self, other: Self) -> Self:
        return self._in_place_arithmetic(other, "__iadd__")

    def __pow__(self, other: Self) -> Self:
        return self._arithmetic(other, "__pow__")

    __rmul__ = __mul__
    __rtruediv__ = __truediv__

    def _get_ensemble_axes_metadata_items(self, items):
        expanded_axes_metadatas = [
            axis_metadata.copy() for axis_metadata in self.ensemble_axes_metadata
        ]
        for i, item in enumerate(items):
            if item is None:
                expanded_axes_metadatas.insert(i, UnknownAxis())

        metadata = {}
        axes_metadata = []
        last_indexed = 0
        for item, expanded_axes_metadata in zip(items, expanded_axes_metadatas):
            last_indexed += 1
            if isinstance(item, Number):
                metadata = {
                    **metadata,
                    **expanded_axes_metadata.item_metadata(item, self.metadata),
                }
            else:
                try:
                    axes_metadata += [expanded_axes_metadata[item].copy()]
                except TypeError:
                    axes_metadata += [expanded_axes_metadata.copy()]

        axes_metadata += expanded_axes_metadatas[last_indexed:]
        return axes_metadata, metadata

    def get_items(
        self,
        items: ArrayItemType | tuple[ArrayItemType, ...],
        keepdims: bool = False,
    ) -> dict:
        """Index the array and the corresponding axes metadata. Only ensemble axes can
        be indexed.

        Parameters
        ----------
        items : int or tuple of int or slice
            The array is indexed according to this.
        keepdims : bool, optional
            If True, all ensemble axes are left in the result as dimensions with size
            one. Default is False.

        Returns
        -------
        indexed_array : ArrayObject or subclass of ArrayObject
            The indexed array object.
        """
        items = _validate_array_items(
            items, shape=self.ensemble_shape, keepdims=keepdims
        )
        ensemble_axes_metadata, metadata = self._get_ensemble_axes_metadata_items(items)

        kwargs = self._copy_kwargs(
            exclude=("array", "ensemble_axes_metadata", "metadata")
        )
        kwargs["array"] = self._array[items]
        kwargs["ensemble_axes_metadata"] = ensemble_axes_metadata
        kwargs["metadata"] = {**self.metadata, **metadata}
        return kwargs

    def __getitem__(self: ArrayObjectType, items) -> ArrayObjectType:
        return self.__class__(**self.get_items(items))

    def expand_dims(
        self,
        axis: Optional[int | tuple[int, ...]] = None,
        axis_metadata: Optional[list[AxisMetadata]] = None,
    ) -> Self:
        """Expand the shape of the array object.

        Parameters
        ----------
        axis : int or tuple of ints
            Position in the expanded axes where the new axis (or axes) is placed.
        axis_metadata : AxisMetadata or List of AxisMetadata, optional
            The axis metadata describing the expanded axes. Default is UnknownAxis.

        Returns
        -------
        expanded : ArrayObject or subclass of ArrayObject
            View of array object with the number of dimensions increased.
        """
        if axis is None:
            axis = (0,)

        axis = number_to_tuple(axis)

        if axis_metadata is None:
            axis_metadata = [UnknownAxis()] * len(axis)

        axis = normalize_axes(axis, self.shape)

        if any(a >= (len(self.ensemble_shape) + len(axis)) for a in axis):
            raise RuntimeError()

        ensemble_axes_metadata = copy.deepcopy(self.ensemble_axes_metadata)

        for a, am in zip(axis, axis_metadata):
            ensemble_axes_metadata.insert(a, am)

        kwargs = self._copy_kwargs(exclude=("array", "ensemble_axes_metadata"))
        kwargs["array"] = _expand_dims(self.array, axis=axis)
        kwargs["ensemble_axes_metadata"] = ensemble_axes_metadata
        return self.__class__(**kwargs)

    def squeeze(self, axis: Optional[tuple[int, ...]] = None) -> Self:
        """Remove axes of length one from array object.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Selects a subset of the entries of length one in the shape.

        Returns
        -------
        squeezed : ArrayObject or subclass of ArrayObject
            The input array object, but with all or a subset of the dimensions of length
            1 removed.
        """
        if len(self.array.shape) < len(self.base_shape):
            return self

        if axis is None:
            axis = tuple(range(len(self.shape)))
        else:
            axis = normalize_axes(axis, self.shape)

        shape = self.shape[: -len(self.base_shape)]

        squeezed = tuple(
            np.where([(n == 1) and (i in axis) for i, n in enumerate(shape)])[0]
        )

        xp = get_array_module(self.array)

        kwargs = self._copy_kwargs(exclude=("array", "ensemble_axes_metadata"))

        kwargs["array"] = xp.squeeze(self.array, axis=squeezed)

        kwargs["ensemble_axes_metadata"] = [
            element
            for i, element in enumerate(self.ensemble_axes_metadata)
            if i not in squeezed
        ]

        return self.__class__(**kwargs)

    def ensure_lazy(self, chunks: Chunks = "auto") -> Self:
        """Creates an equivalent lazy version of the array object.

        Parameters
        ----------
        chunks : int or tuple or str
            How to chunk the array. See `dask.array.from_array`.

        Returns
        -------
        lazy_array_object : ArrayObject or subclass of ArrayObject
            Lazy version of the array object.
        """

        if self.is_lazy:
            return self

        if chunks == "auto":
            chunks = ("auto",) * len(self.ensemble_shape) + (-1,) * len(self.base_shape)

        array = da.from_array(self.array, chunks=chunks)

        return self.__class__(array, **self._copy_kwargs(exclude=("array",)))

    def lazy(self, chunks: str = "auto") -> Self:
        return self.ensure_lazy(chunks)

    def compute(
        self,
        progress_bar: bool | None = None,
        profiler: bool = False,
        resource_profiler: bool = False,
        **kwargs,
    ) -> Self | tuple[Self, tuple]:
        """Turn a lazy *ab*TEM object into its in-memory equivalent.

        Parameters
        ----------
        progress_bar : bool
            Display a progress bar in the terminal or notebook during computation. The
            progress bar is only displayed with a local scheduler.
        profiler : bool
            Return Profiler class used to profile Dask's execution at the task level.
            Only execution with a local scheduler is profiled.
        resource_profiler : bool
            Return ResourceProfiler class used to profile Dask’s execution at the
            resource level.
        kwargs :
            Additional keyword arguments passed to `dask.compute`.
        """
        if not self.is_lazy:
            return self

        output, profilers = _compute(
            [self],
            progress_bar=progress_bar,
            profiler=profiler,
            resource_profiler=resource_profiler,
            **kwargs,
        )

        output_value = output[0]

        if profilers:
            return output_value, profilers

        return output_value

    def copy_to_device(self, device: str) -> Self:
        """Copy array to specified device.

        Parameters
        ----------
        device : str

        Returns
        -------
        object_on_device : ArrayObject
        """
        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = copy_to_device(self.array, device)

        return self.__class__(**kwargs)

    def to_cpu(self) -> Self:
        """Move the array to the host memory from an arbitrary source array."""
        return self.copy_to_device("cpu")

    def to_gpu(self, device: str = "gpu") -> Self:
        """Move the array from the host memory to a gpu."""
        return self.copy_to_device(device)

    def to_zarr(
        self, url: str, compute: bool = True, overwrite: bool = False, **kwargs
    ):
        """Write data to a zarr file.

        Parameters
        ----------
        url : str
            Location of the data, typically a path to a local file. A URL can also
            include a protocol specifier like s3:// for remote data.
        compute : bool
            If true compute immediately; return dask.delayed.Delayed otherwise.
        overwrite : bool
            If given array already exists, overwrite=False will cause an error, where
            overwrite=True will replace the existing data.
        kwargs :
            Keyword arguments passed to `dask.array.to_zarr`.
        """

        return ComputableList([self]).to_zarr(
            url=url, compute=compute, overwrite=overwrite, **kwargs
        )

    @classmethod
    def _pack_kwargs(cls, kwargs):
        attrs = {}
        for key, value in kwargs.items():
            if key == "ensemble_axes_metadata":
                attrs[key] = [axis_to_dict(axis) for axis in value]
            else:
                attrs[key] = value
        return attrs

    @classmethod
    def _unpack_kwargs(cls, attrs):
        kwargs = dict()
        kwargs["ensemble_axes_metadata"] = []
        for key, value in attrs.items():
            if key == "ensemble_axes_metadata":
                kwargs["ensemble_axes_metadata"] = [axis_from_dict(d) for d in value]
            elif key == "type":
                pass
            else:
                kwargs[key] = value

        return kwargs

    def _metadata_to_dict(self):
        metadata = copy.copy(self.metadata)
        metadata["axes"] = {
            f"axis_{i}": axis_to_dict(axis) for i, axis in enumerate(self.axes_metadata)
        }
        metadata["data_origin"] = f"abTEM_v{__version__}"
        metadata["type"] = self.__class__.__name__
        return metadata

    def _metadata_to_json(self):
        metadata = copy.copy(self.metadata)
        metadata["axes"] = {
            f"axis_{i}": axis_to_dict(axis) for i, axis in enumerate(self.axes_metadata)
        }
        metadata["data_origin"] = f"abTEM_v{__version__}"
        metadata["type"] = self.__class__.__name__
        return json.dumps(metadata)

    def to_tiff(self, filename: str, **kwargs):
        """Write data to a tiff file.

        Parameters
        ----------
        filename : str
            The filename of the file to write.
        kwargs :
            Keyword arguments passed to `tifffile.imwrite`.
        """
        if tifffile is None:
            raise RuntimeError(
                "This functionality of abTEM requires tifffile, see https://github.com/cgohlke/tifffile."
            )

        array = self.array
        if self.is_lazy:
            warnings.warn("Lazy arrays are computed in memory before writing to tiff.")
            array = self._lazy_array.compute()

        return tifffile.imwrite(
            filename, array, description=self._metadata_to_json(), **kwargs
        )

    @classmethod
    def from_zarr(cls, url: str, chunks: Chunks = "auto") -> Self:
        """Read wave functions from a Zarr file.

        url : str
            Location of the data, typically a path to a local file. A URL can also
            include a protocol specifier like s3:// for remote data.
        chunks : tuple of ints or tuples of ints
            Passed to dask.array.from_array(), allows setting the chunks on
            initialisation, if the chunking scheme in the on-disc dataset is not optimal
            for the calculations to follow.
        """
        return from_zarr(url, chunks=chunks)

    @property
    def _has_base_chunks(self) -> bool:
        if not isinstance(self.array, da.core.Array):
            return False

        base_chunks = self.array.chunks[-len(self.base_shape) :]
        return any(len(c) > 1 for c in base_chunks)

    def no_base_chunks(self):
        """Rechunk to remove chunks across the base dimensions."""
        if not self._has_base_chunks:
            return self
        chunks = self.array.chunks[: -len(self.base_shape)] + (-1,) * len(
            self.base_shape
        )
        return self.rechunk(chunks)

    @staticmethod
    def _apply_transform(
        array: np.ndarray,
        *args: tuple[np.ndarray, ...],
        transform_partial: Callable,
        array_object_partial: Callable,
        base_ndims: int,
    ) -> np.ndarray:
        axes = unpack_blockwise_args(args)

        array_axes = axes[: len(array.shape)]
        ensemble_axes = array_axes[:-base_ndims]
        transform_axes = axes[len(array.shape) :]

        array_object = array_object_partial((array, list(ensemble_axes))).item()
        transform = transform_partial(*transform_axes).item()

        out_arrays = transform._calculate_new_array(array_object)

        if not isinstance(out_arrays, tuple):
            out_arrays = (out_arrays,)

        ndims = len(transform_axes) + len(array.shape)
        packing = np.zeros((1,) * ndims, dtype=object)
        itemset(packing, 0, out_arrays)
        return packing

    def apply_transform(
        self, transform: ArrayObjectTransform, max_batch: int | str = "auto"
    ) -> ArrayObject | list[ArrayObject]:
        """Transform the wave functions by a given transformation.

        Parameters
        ----------
        transform : ArrayObjectTransform
            The array object transformation to apply.
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array.
            If 'auto' (default), the batch size is automatically chosen based on the
            abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".

        Returns
        -------
        transformed_array_object : ArrayObject
            The transformed array object.
        """

        new_arrays: (
            da.core.Array
            | np.ndarray
            | tuple[da.core.Array, ...]
            | tuple[np.ndarray, ...]
        )
        if self.is_lazy:
            if isinstance(max_batch, int):
                max_batch = int(max_batch * np.prod(self.base_shape))

            chunks = transform._default_ensemble_chunks + self._lazy_array.chunks

            shape = transform.ensemble_shape + self.shape

            validated_chunks = validate_chunks(
                shape=shape,
                chunks=chunks,
                max_elements=max_batch,
                dtype=self.dtype,
            )

            assert chunks[len(transform.ensemble_shape) :] == self._lazy_array.chunks

            old_axes = tuple(
                axis._to_blocks((c,))
                for axis, c in zip(self.axes_metadata, self._lazy_array.chunks)
            )

            new_axes = transform._partition_args(
                validated_chunks[: len(transform.ensemble_shape)]
            )

            num_dropped_axes = tuple(
                len(shape) - len(out_shape) for out_shape in transform._out_shape(self)
            )

            # drop_axes = tuple(
            #     tuple(range(self.ensemble_dims, self.ensemble_dims + num_dropped_axis))
            #     for num_dropped_axis in num_dropped_axes
            # )

            n = len(transform.ensemble_shape)
            drop_axes = tuple(
                tuple(
                    range(
                        self.ensemble_dims + n,
                        self.ensemble_dims + num_dropped_axis + n,
                    )
                )
                for num_dropped_axis in num_dropped_axes
            )

            out_metas = transform._out_meta(self)

            array_object_partial = self._from_partitioned_args()
            transform_partial = transform._from_partitioned_args()

            new_shapes = tuple(
                tuple(out_shape) for out_shape in transform._out_shape(self)
            )

            new_arrays = multi_output_blockwise(
                self._apply_transform,
                array=self._lazy_array,
                chunks=validated_chunks,
                array_axes=old_axes,
                new_axes=new_axes,
                drop_axes=drop_axes,
                out_metas=out_metas,
                new_shapes=new_shapes,
                array_object_partial=array_object_partial,
                transform_partial=transform_partial,
                base_ndims=len(self.base_shape),
            )
        else:
            new_arrays = transform._calculate_new_array(self)
            if not isinstance(new_arrays, tuple):
                new_arrays = (new_arrays,)

        base_axes_metadatas = transform._out_base_axes_metadata(self)
        ensemble_axes_metadatas = transform._out_ensemble_axes_metadata(self)
        metadatas = transform._out_metadata(self)
        clss = transform._out_type(self)
        outputs = []
        for array, cls, base_axes_metadata, ensemble_axes_metadata, metadata in zip(
            new_arrays,
            clss,
            base_axes_metadatas,
            ensemble_axes_metadatas,
            metadatas,
        ):
            axes_metadata = ensemble_axes_metadata + base_axes_metadata
            output = cls.from_array_and_metadata(
                array, axes_metadata=axes_metadata, metadata=metadata
            )
            outputs.append(output)

        if len(outputs) > 1:
            if self.is_lazy:
                return ComputableList(outputs)
            else:
                return outputs
        else:
            return outputs[0]

    def set_ensemble_axes_metadata(
        self, axes_metadata: AxisMetadata, axis: int
    ) -> Self:
        """Sets the axes metadata of an ensemble axis.

        Parameters
        ----------
        axes_metadata : AxisMetadata
            The new axis metadata.
        axis : int
            The axis to set.
        """

        old_axes_metadata = copy.deepcopy(self.ensemble_axes_metadata)

        self.ensemble_axes_metadata[axis] = axes_metadata

        try:
            self._check_axes_metadata()
        except RuntimeError:
            self._ensemble_axes_metadata = old_axes_metadata

        return self

    def to_hyperspy(self, transpose: bool = True):
        """Convert ArrayObject to a Hyperspy signal.

        Parameters
        ----------
        transpose : bool, optional
            If True, transpose the base axes of the array before converting to a
            Hyperspy signal. Default is True.

        Returns
        -------
        signal : Hyperspy signal
            The converted Hyperspy signal.

        Raises
        ------
        ImportError
            If Hyperspy is not installed.

        RuntimeError
            If the number of base dimensions is not 1 or 2.

        Notes
        -----
        This method requires Hyperspy to be installed. You can find more information
        about Hyperspy at https://hyperspy.org.
        """
        if hs is None:
            raise ImportError(
                "This functionality of *ab*TEM requires Hyperspy, see https://hyperspy.org."
            )

        if self._base_dims == 1:
            signal_type = hs.signals.Signal1D
        elif self._base_dims == 2:
            signal_type = hs.signals.Signal2D
        else:
            raise RuntimeError()

        axes_base = _to_hyperspy_axes_metadata(
            self.base_axes_metadata,
            self.base_shape,
        )
        ensemble_axes_metadata = _to_hyperspy_axes_metadata(
            self.ensemble_axes_metadata,
            self.ensemble_shape,
        )

        xp = get_array_module(self.device)

        axes_base_indices = tuple_range(
            offset=len(self.ensemble_shape), length=len(self.base_shape)
        )

        ensemble_axes = tuple_range(
            offset=0,
            length=len(self.ensemble_shape),
        )

        source = ensemble_axes + axes_base_indices
        destination = ensemble_axes + axes_base_indices[::-1]

        if transpose:
            if self.is_lazy:
                array = da.moveaxis(self.array, source=source, destination=destination)
            else:
                array = xp.moveaxis(self.array, source=source, destination=destination)
        else:
            array = self.array

        s = signal_type(array, axes=ensemble_axes_metadata[::-1] + axes_base[::-1])

        if self.is_lazy:
            s = s.as_lazy()

        return s

    def to_data_array(self):
        """Convert ArrayObject to a xarray DataArray. Requires xarray to be installed.

        Returns
        -------
        xarray.DataArray
            The converted xarray DataArray.

        Raises
        ------
        ImportError
            If xarray is not installed.
        """
        if xr is None:
            raise ImportError(
                "This functionality of *ab*TEM requires xarray, see https://xarray.dev/."
            )

        coords = {}
        dims = []
        for n, axis in zip(self.shape, self.axes_metadata):
            x = np.array(axis.coordinates(n))
            if isinstance(x, np.ndarray) and len(x.shape) == 2:
                x = [f"{i}" for i in x]
            elif len(x.shape) == 1:
                pass
            else:
                raise ValueError("The shape of the coordinates is not supported.")

            dims.append(axis.label)

            dr = xr.DataArray(
                x, name=axis.label, dims=(axis.label,), attrs={"units": axis.units}
            )

            coords[axis.label] = dr

        attrs = self.metadata
        attrs["long_name"] = self.metadata["label"]

        return xr.DataArray(self.array, dims=dims, coords=coords, attrs=attrs)

    @classmethod
    def _stack(
        cls,
        array_objects: tuple[ArrayObject, ...],
        axis_metadata: AxisMetadata,
        axis: int,
    ) -> Self:
        xp = get_array_module(array_objects[0].device)

        if any(array.is_lazy for array in array_objects):
            array = da.stack(
                [array_object.array for array_object in array_objects], axis=axis
            )
        else:
            array = xp.stack(
                [array_object.array for array_object in array_objects], axis=axis
            )

        kwargs = array_objects[0]._copy_kwargs(exclude=("array",))

        kwargs["array"] = array
        ensemble_axes_metadata = [
            axis_metadata.copy() for axis_metadata in kwargs["ensemble_axes_metadata"]
        ]
        ensemble_axes_metadata.insert(axis, axis_metadata)
        kwargs["ensemble_axes_metadata"] = ensemble_axes_metadata
        return cls(**kwargs)

    def _partition_ensemble_axes_metadata(
        self, chunks: Optional[Chunks] = None, lazy: bool = True
    ):
        if len(self.ensemble_shape) == 0:
            ensemble_axes_metadata = _wrap_with_array([], 0)
        else:
            chunks = self._validate_ensemble_chunks(chunks)
            chunk_shape = tuple(len(c) for c in chunks)

            ensemble_axes_metadata = np.zeros(chunk_shape, dtype=object)
            for index, slic in iterate_chunk_ranges(chunks):
                new_ensemble_axes_metadata = [
                    axis[slic[i]] if hasattr(axis, "__getitem__") else axis.copy()
                    for i, axis in enumerate(self.ensemble_axes_metadata)
                ]

                itemset(ensemble_axes_metadata, index, new_ensemble_axes_metadata)

        if lazy:
            ensemble_axes_metadata = da.from_array(ensemble_axes_metadata, chunks=1)

        return ensemble_axes_metadata

    @property
    def _default_ensemble_chunks(self):
        if self.is_lazy:
            return self._lazy_array.chunks[: self.ensemble_dims]
        else:
            return -1

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True):
        if chunks is None and self.is_lazy:
            chunks = self._lazy_array.chunks[: -len(self.base_shape)]
        elif chunks is None:
            chunks = (1,) * len(self.ensemble_shape)

        ensemble_chunks = self._validate_ensemble_chunks(chunks)
        if lazy:
            xp = get_array_module(self.array)
            array = self.ensure_lazy()._lazy_array

            if ensemble_chunks != array.chunks:
                array = array.rechunk(
                    ensemble_chunks + array.chunks[len(ensemble_chunks) :]
                )

            ensemble_axes_metadata = self._partition_ensemble_axes_metadata(
                chunks=ensemble_chunks
            )

            def _combine_args(*args):
                combined = args[0], args[1].item()
                return _wrap_with_array(combined, 1)

            ndims = len(self.ensemble_shape)
            blocks = da.blockwise(
                _combine_args,
                tuple_range(ndims),
                array,
                tuple_range(len(array.shape)),
                ensemble_axes_metadata,
                tuple_range(ndims),
                align_arrays=False,
                concatenate=True,
                dtype=object,
                meta=xp.array((), object),
            )
        else:
            array = self.compute().array
            if len(self.ensemble_shape) == 0:
                blocks = np.zeros((), dtype=object)
            else:
                chunk_shape = tuple(len(c) for c in chunks)
                blocks = np.zeros(chunk_shape, dtype=object)

            ensemble_axes_metadata = self._partition_ensemble_axes_metadata(
                chunks, lazy=False
            )

            for block_indices, chunk_range in iterate_chunk_ranges(chunks):
                itemset(
                    blocks,
                    block_indices,
                    (array[chunk_range], ensemble_axes_metadata[block_indices]),
                )

        return (blocks,)

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        args = unpack_blockwise_args(args)

        array, ensemble_axes_metadata = args[0]
        assert isinstance(ensemble_axes_metadata, list)

        new_array_object = cls(
            array=array, ensemble_axes_metadata=ensemble_axes_metadata, **kwargs
        )
        ndims = new_array_object.ensemble_dims
        return _wrap_with_array(new_array_object, ndims)

    def _from_partitioned_args(self):
        return partial(
            self._from_partitioned_args_func,
            **self._copy_kwargs(exclude=("array", "ensemble_axes_metadata")),
        )


def _expand_dims(
    array: np.ndarray | da.core.Array, axis: int | tuple | list
) -> np.ndarray:
    if isinstance(axis, int):
        axis = (axis,)

    out_ndim = len(axis) + array.ndim
    axis = validate_axis(axis, out_ndim)

    assert not isinstance(axis, int)

    shape_it = iter(array.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]

    return array.reshape(shape)


def from_zarr(url: str, chunks: Optional[Chunks] = None):
    """Read abTEM data from zarr.

    Parameters
    ----------
    url : str
        Location of the data. A URL can include a protocol specifier like s3:// for
        remote data. For Zarr 3 zip stores, this should be a path to a .zip file.
    chunks :  tuple of ints or tuples of ints
        Passed to dask.array.from_array(), allows setting the chunks on initialisation,
        if the chunking scheme in the on-disc dataset is not optimal for the
        calculations to follow.

    Returns
    -------
    imported : ArrayObject
    """
    import zarr

    import abtem

    # Helper function for type restoration
    def decode_types(obj):
        """Recursively decode tuples from JSON."""
        if isinstance(obj, dict):
            if obj.get("_type") == "tuple":
                return tuple(decode_types(item) for item in obj["_value"])
            else:
                return {key: decode_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [decode_types(item) for item in obj]
        else:
            return obj

    # Determine if this is a zip file
    is_zip = url.endswith(".zip")

    if is_zip:
        store = zarr.storage.ZipStore(url, mode="r")
        f = zarr.open(store=store, mode="r")
    else:
        f = zarr.open(url, mode="r")

    # Read metadata
    i = 0
    types = []
    while True:
        try:
            types.append(f.attrs[f"type{i}"])
        except KeyError:
            break
        i += 1

    imported = []
    for i, t in enumerate(types):
        cls = getattr(abtem, t)

        # Decode types before unpacking
        packed_kwargs = decode_types(f.attrs[f"kwargs{i}"])
        kwargs = cls._unpack_kwargs(packed_kwargs)

        num_ensemble_axes = len(kwargs["ensemble_axes_metadata"])

        # Get the zarr array
        zarr_array = f[f"array{i}"]

        # Determine chunks
        if chunks == "auto":
            array_chunks = ("auto",) * num_ensemble_axes + (-1,) * cls._base_dims
        elif chunks is None:
            # Use the chunks from the zarr array itself
            array_chunks = zarr_array.chunks
        else:
            array_chunks = chunks

        # Create dask array from zarr array
        array = da.from_array(zarr_array, chunks=array_chunks)

        with config.set({"warnings.overspecified-grid": False}):
            imported.append(cls(array, **kwargs))

    if len(imported) == 1:
        imported = imported[0]

    return imported


def validate_axis_metadata(
    axis_metadata: Optional[AxisMetadata | Sequence[str] | dict],
) -> AxisMetadata:
    validated_axis_metadata: AxisMetadata
    if axis_metadata is None:
        validated_axis_metadata = UnknownAxis()

    elif isinstance(axis_metadata, (tuple, list)):
        if not all(isinstance(element, str) for element in axis_metadata):
            raise ValueError("All elements in the list must be strings.")

        validated_axis_metadata = OrdinalAxis(values=tuple(axis_metadata))

    elif isinstance(axis_metadata, dict):
        validated_axis_metadata = OrdinalAxis(**axis_metadata)

    elif not isinstance(axis_metadata, AxisMetadata):
        raise ValueError(
            "axis_metadata must be a dict, sequence of strings or an AxisMetadata"
            "object."
        )
    elif isinstance(axis_metadata, OrdinalAxis):
        validated_axis_metadata = axis_metadata
    else:
        raise ValueError(
            "axis_metadata must be a dict, sequence of strings or an AxisMetadata"
            f"object. Not {type(axis_metadata).__name__}."
        )

    return validated_axis_metadata


def stack(
    arrays: Sequence[ArrayObjectType],
    axis_metadata: Optional[AxisMetadata | Sequence[str] | dict] = None,
    axis: int = 0,
) -> ArrayObjectType:
    """Stack multiple array objects (e.g. Waves and BaseMeasurement) along a new
    ensemble axis.

    Parameters
    ----------
    arrays : sequence of array objects
        Each abTEM array object must have the same type and shape.
    axis_metadata : AxisMetadata
        The axis metadata describing the new axis.
    axis : int
        The ensemble axis in the resulting array object along which the input arrays are
        stacked.

    Returns
    -------
    array_object : ArrayObject
        The stacked array object of the same type as the input.
    """

    # if not all(isinstance(array, ArrayObject) for array in arrays):
    #    raise ValueError("arrays must be a sequence of array objects.")

    assert axis <= len(arrays[0].ensemble_shape)
    assert axis >= 0

    axis_metadata = validate_axis_metadata(axis_metadata)

    return arrays[0]._stack(arrays, axis_metadata, axis)


def concatenate(arrays: Sequence[ArrayObject], axis: int = 0) -> ArrayObject:
    """Join a sequence of abTEM array classes along an existing axis.

    Parameters
    ----------
    arrays : sequence of array objects
        Each abTEM array object must have the same type and shape, except in the
        dimension corresponding to axis. The axis metadata along the concatenated
        axis must be compatible for concatenation.
    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.

    Returns
    -------
    array_object : ArrayObject
        The concatenated array object of the same type as the input.
    """

    xp = get_array_module(arrays[0].array)

    if arrays[0].is_lazy:
        array = da.concatenate([has_array.array for has_array in arrays], axis=axis)
    else:
        array = xp.concatenate([has_array.array for has_array in arrays], axis=axis)

    cls = arrays[0].__class__

    concatenated_axes_metadata = arrays[0].axes_metadata[axis]
    for has_array in arrays[1:]:
        concatenated_axes_metadata = concatenated_axes_metadata.concatenate(
            has_array.axes_metadata[axis]
        )

    axes_metadata = copy.deepcopy(arrays[0].axes_metadata)
    axes_metadata[axis] = concatenated_axes_metadata

    return cls.from_array_and_metadata(
        array=array, axes_metadata=axes_metadata, metadata=arrays[0].metadata
    )


def swapaxes(array_object, axis1, axis2):
    xp = get_array_module(array_object.array)

    if array_object.is_lazy:
        array = da.swapaxes(array_object.array, axis1, axis2)
    else:
        array = xp.swapaxes(array_object.array, axis1, axis2)

    cls = array_object.__class__

    axes_metadata = copy.copy(array_object.axes_metadata)
    axes_metadata[axis2], axes_metadata[axis1] = (
        axes_metadata[axis1],
        axes_metadata[axis2],
    )

    return cls.from_array_and_metadata(
        array=array, axes_metadata=axes_metadata, metadata=array_object.metadata
    )


def moveaxis(
    array_object: ArrayObject,
    source: tuple[int, ...],
    destination: tuple[int, ...],
) -> ArrayObject:
    xp = get_array_module(array_object.array)

    if array_object.is_lazy:
        array = da.moveaxis(array_object.array, source, destination)
    else:
        array = xp.moveaxis(array_object.array, source, destination)

    axes_metadata = copy.copy(array_object.axes_metadata)

    for s, d in zip(reversed(source), reversed(destination)):
        element = axes_metadata.pop(s)
        axes_metadata.insert(d, element)

    return array_object.__class__.from_array_and_metadata(
        array=array, axes_metadata=axes_metadata, metadata=array_object.metadata
    )
