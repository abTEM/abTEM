"""Module for describing array objects."""
from __future__ import annotations

import copy
import json
import warnings
from abc import ABCMeta
from contextlib import nullcontext, contextmanager
from functools import partial
from numbers import Number
from typing import TypeVar, Sequence, TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
import zarr
from dask.array.utils import validate_axis
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler

from abtem._version import __version__
from abtem.core import config
from abtem.core.axes import (
    UnknownAxis,
    axis_to_dict,
    axis_from_dict,
    AxisMetadata,
    OrdinalAxis,
    AxesMetadataList,
    NonLinearAxis,
    LinearAxis,
)
from abtem.core.backend import (
    get_array_module,
    copy_to_device,
    cp,
    device_name_from_array_module,
    check_cupy_is_installed,
)
from abtem.core.chunks import Chunks, validate_chunks, chunk_shape, iterate_chunk_ranges
from abtem.core.ensemble import (
    Ensemble,
    _wrap_with_array,
    unpack_blockwise_args,
    concatenate_array_blocks,
)
from abtem.core.utils import (
    normalize_axes,
    CopyMixin,
    tuple_range,
    EqualityMixin,
    interleave,
)

if TYPE_CHECKING:
    from abtem.transform import ArrayObjectTransform

try:
    import tifffile
except ImportError:
    tifffile = None


def _to_hyperspy_axes_metadata(axes_metadata, shape):
    hyperspy_axes = []

    if not isinstance(shape, (list, tuple)):
        shape = (shape,)

    for metadata, n in zip(axes_metadata, shape):
        hyperspy_axes.append({"size": n})

        axes_mapping = {
            "sampling": "scale",
            "units": "units",
            "label": "name",
            "offset": "offset",
        }

        if isinstance(metadata, OrdinalAxis):
            # TODO : when hyperspy supports arbitrary (non-uniform) DataAxis this should be updated

            if len(metadata.values) > 1:
                sampling = metadata.values[1] - metadata.values[0]
            else:
                sampling = 1.0

            if metadata.units is None:
                units = ""
            else:
                units = metadata.units

            metadata = LinearAxis(
                label=metadata.label,
                units=units,
                sampling=sampling,
                offset=metadata.values[0],
            )

        for attr, mapped_attr in axes_mapping.items():
            if hasattr(metadata, attr):
                hyperspy_axes[-1][mapped_attr] = getattr(metadata, attr)

    return hyperspy_axes


class ComputableList(list):
    """
    A list with methods for conveniently computing its items.
    """

    def to_zarr(
        self,
        url: str,
        compute: bool = True,
        overwrite: bool = False,
        progress_bar: bool = None,
        **kwargs,
    ):
        """
        Write data to a zarr file.

        Parameters
        ----------
        url : str
            Location of the data, typically a path to a local file. A URL can also include a protocol specifier like
            s3:// for remote data.
        compute : bool
            If true compute immediately; return dask.delayed.Delayed otherwise.
        overwrite : bool
            If given array already exists, overwrite=False will cause an error, where overwrite=True will replace the
            existing data.
        progress_bar : bool
            Display a progress bar in the terminal or notebook during computation. The progress bar is only displayed
            with a local scheduler.
        kwargs :
            Keyword arguments passed to `dask.array.to_zarr`.
        """

        computables = []
        with zarr.open(url, mode="w") as root:
            for i, has_array in enumerate(self):
                has_array = has_array.ensure_lazy()

                array = has_array.copy_to_device("cpu").array

                computables.append(
                    array.to_zarr(
                        url, compute=False, component=f"array{i}", overwrite=overwrite
                    )
                )
                packed_kwargs = has_array._pack_kwargs(
                    has_array._copy_kwargs(exclude=("array",))
                )

                root.attrs[f"kwargs{i}"] = packed_kwargs
                root.attrs[f"type{i}"] = has_array.__class__.__name__

        if not compute:
            return computables

        with _compute_context(
            progress_bar, profiler=False, resource_profiler=False
        ) as (_, profiler, resource_profiler, _):
            output = dask.compute(computables, **kwargs)[0]

        profilers = ()
        if profiler is not None:
            profilers += (profiler,)

        if resource_profiler is not None:
            profilers += (resource_profiler,)

        if profilers:
            return output, profilers

        return output

    def compute(self, **kwargs) -> list[ArrayObject] | tuple[list[ArrayObject], tuple]:
        """
        Turn a list of lazy ArrayObjects object into the in-memory equivalents.

        kwargs :
            Keyword arguments passed to `ArrayObject.compute`.
        """

        output, profilers = _compute(self, **kwargs)

        if profilers:
            return output, profilers

        return output


@contextmanager
def _compute_context(
    progress_bar: bool = None, profiler=False, resource_profiler=False
):
    if progress_bar is None:
        progress_bar = config.get("local_diagnostics.progress_bar")

    if progress_bar:
        if progress_bar == "tqdm":
            from tqdm.dask import TqdmCallback

            progress_bar = TqdmCallback(desc="tasks")
        else:
            progress_bar = ProgressBar()
    else:
        progress_bar = nullcontext()

    if profiler:
        profiler = Profiler()
    else:
        profiler = nullcontext()

    if resource_profiler:
        resource_profiler = ResourceProfiler()
    else:
        resource_profiler = nullcontext()

    dask_configuration = {
        "optimization.fuse.active": config.get("dask.fuse"),
    }

    with (
        progress_bar as progress_bar,
        profiler as profiler,
        resource_profiler as resource_profiler,
        dask.config.set(dask_configuration) as dask_configuration,
    ):
        yield progress_bar, profiler, resource_profiler, dask_configuration


def _compute(
    dask_array_wrappers,
    progress_bar: bool = None,
    profiler: bool = False,
    resource_profiler: bool = False,
    **kwargs,
):
    if config.get("device") == "gpu":
        check_cupy_is_installed()

        if "num_workers" not in kwargs:
            kwargs["num_workers"] = cp.cuda.runtime.getDeviceCount()

        if "threads_per_worker" not in kwargs:
            kwargs["threads_per_worker"] = cp.cuda.runtime.getDeviceCount()

    with _compute_context(
        progress_bar, profiler=profiler, resource_profiler=resource_profiler
    ) as (_, profiler, resource_profiler, _):
        arrays = dask.compute(
            [wrapper.array for wrapper in dask_array_wrappers], **kwargs
        )[0]

    for array, wrapper in zip(arrays, dask_array_wrappers):
        wrapper._array = array

    profilers = ()
    if profiler is not None:
        profilers += (profiler,)

    if resource_profiler is not None:
        profilers += (resource_profiler,)

    return dask_array_wrappers, profilers


def _validate_lazy(lazy):
    if lazy is None:
        return config.get("dask.lazy")

    return lazy


T = TypeVar("T", bound="ArrayObject")


class ArrayObject(Ensemble, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """
    A base class for simulation objects described by an array and associated metadata.

    Parameters
    ----------
    array : ndarray
        Array representing the array object.
    ensemble_axes_metadata : list of AxesMetadata
        Axis metadata for each ensemble axis. The axis metadata must be compatible with the shape of the array.
    metadata : dict
        A dictionary defining wave function metadata. All items will be added to the metadata of measurements derived
        from the waves.
    """

    _base_dims: int

    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        metadata: dict = None,
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

    @property
    def base_dims(self):
        """Number of base dimensions."""
        return self._base_dims

    @property
    def ensemble_dims(self):
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
    def ensemble_axes_metadata(self):
        """List of AxisMetadata of the ensemble axes."""
        return self._ensemble_axes_metadata

    @property
    def axes_metadata(self) -> AxesMetadataList:
        """List of AxisMetadata."""
        return AxesMetadataList(
            self.ensemble_axes_metadata + self.base_axes_metadata, self.shape
        )

    def _check_axes_metadata(self):
        if len(self.shape) != len(self.axes_metadata):
            raise RuntimeError(
                f"number of array dimensions ({len(self.shape)}) does not match number of axis metadata items "
                f"({len(self.axes_metadata)})"
            )

        for n, axis in zip(self.shape, self.axes_metadata):
            if isinstance(axis, OrdinalAxis) and len(axis) != n:
                raise RuntimeError(
                    f"number of values for ordinal axis ({len(axis)}), does not match size of dimension "
                    f"({n})"
                )

    def _is_base_axis(self, axis: int | tuple[int, ...]) -> bool:
        if isinstance(axis, Number):
            axis = (axis,)

        base_axes = tuple(range(len(self.ensemble_shape), len(self.shape)))
        return len(set(axis).intersection(base_axes)) > 0

    @classmethod
    def from_array_and_metadata(
        cls,
        array: np.ndarray | da.core.Array,
        axes_metadata: list[AxisMetadata],
        metadata: dict,
    ) -> T:
        """
        Creates array object from a given array and metadata.

        Parameters
        ----------
        array : array
            Complex array defining one or more 2D wave functions. The second-to-last and last dimensions are the wave
            function `y`- and `x`-axis, respectively.
        axes_metadata : list of AxesMetadata
            Axis metadata for each axis. The axis metadata must be compatible with the shape of the array. The last two
            axes must be RealSpaceAxis.
        metadata :
            A dictionary defining wave function metadata. All items will be added to the metadata of measurements
            derived from the waves. The metadata must contain the electron energy [eV].

        Returns
        -------
        wave_functions : Waves
            The created wave functions.
        """
        raise NotImplementedError

    def rechunk(self, chunks: Chunks, **kwargs):
        """
        Rechunk dask array.

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

        array = self._array.rechunk(chunks=chunks, **kwargs)
        kwargs = self._copy_kwargs(exclude=("array",))
        return self.__class__(array, **kwargs)

    @property
    def metadata(self):
        """Metadata stored as a dictionary."""
        return self._metadata

    def __len__(self) -> int:
        return len(self.array)

    @property
    def array(self) -> np.ndarray | da.core.Array:
        """
        Underlying array describing the array object.
        """
        return self._array

    @property
    def dtype(self) -> np.dtype.base:
        """
        Datatype of array.
        """
        return self._array.dtype

    @property
    def device(self) -> str:
        """The device where the array is stored."""
        return device_name_from_array_module(get_array_module(self.array))

    @property
    def is_lazy(self) -> bool:
        """
        True if array is lazy.
        """
        return isinstance(self.array, da.core.Array)

    @classmethod
    def _to_delayed_func(cls, array, **kwargs):
        kwargs["array"] = array
        return cls(**kwargs)

    @property
    def is_complex(self) -> bool:
        """
        True if array is complex.
        """
        return np.iscomplexobj(self.array)

    def _check_is_compatible(self, other: ArrayObject):
        if not isinstance(other, self.__class__):
            raise RuntimeError(
                f"incompatible types ({self.__class__} != {other.__class__})"
            )

        # if self.shape != other.shape:
        #    raise RuntimeError(f"incompatible shapes ({self.shape} != {other.shape})")

        # for (key, value), (other_key, other_value) in zip(
        #     self._copy_kwargs(exclude=("array", "metadata")).items(),
        #     other._copy_kwargs(exclude=("array", "metadata")).items(),
        # ):
        #     if np.any(value != other_value):
        #         raise RuntimeError(
        #             f"incompatible values for {key} ({value} != {other_value})"
        #         )

    def generate_ensemble(self, keepdims: bool = False):
        """
        Generate every member of the ensemble.

        Parameters
        ----------
        keepdims : bool, opptional
            If True, all ensemble axes are left in the result as dimensions with size one. Default is False.

        Yields
        ------
        ArrayObject or subclass of ArrayObject
            Member of the ensemble.
        """
        for i in np.ndindex(*self.ensemble_shape):
            yield i, self.get_items(i, keepdims=keepdims)

    def mean(
        self,
        axis: int | tuple[int, ...] = None,
        keepdims: bool = False,
        split_every: int = 2,
    ) -> T:
        """
        Mean of array object over one or more axes. Only ensemble axes can be reduced.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which a means are calculated. The default is to compute the mean of the flattened array.
            If this is a tuple of ints, the mean is calculated over multiple axes. The indicated axes must be ensemble
            axes.
        keepdims : bool, optional
            If True, the reduced axes are left in the result as dimensions with size one. Default is False.
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
        axis: int | tuple[int, ...] = None,
        keepdims: bool = False,
        split_every: int = 2,
    ) -> T:
        """
        Sum of array object over one or more axes. Only ensemble axes can be reduced.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which a sums are performed. The default is to compute the mean of the flattened array.
            If this is a tuple of ints, the sum is performed over multiple axes. The indicated axes must be ensemble
            axes.
        keepdims : bool, optional
            If True, the reduced axes are left in the result as dimensions with size one. Default is False.
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
        axis: int | tuple[int, ...] = None,
        keepdims: bool = False,
        split_every: int = 2,
    ) -> T:
        """
        Standard deviation of array object over one or more axes. Only ensemble axes can be reduced.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which a standard deviations are calculated. The default is to compute the mean of the
            flattened array. If this is a tuple of ints, the standard deviations are calculated over multiple axes.
            The indicated axes must be ensemble axes.
        keepdims : bool, optional
            If True, the reduced axes are left in the result as dimensions with size one. Default is False.
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
        axis: int | tuple[int, ...] = None,
        keepdims: bool = False,
        split_every: int = 2,
    ) -> T:
        """
        Minmimum of array object over one or more axes. Only ensemble axes can be reduced.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which a minima are calculated. The default is to compute the mean of the flattened array.
            If this is a tuple of ints, the minima are calculated over multiple axes. The indicated axes must be
            ensemble axes.
        keepdims : bool, optional
            If True, the reduced axes are left in the result as dimensions with size one. Default is False.
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
        axis: int | tuple[int, ...] = None,
        keepdims: bool = False,
        split_every: int = 2,
    ) -> T:
        """
        Maximum of array object over one or more axes. Only ensemble axes can be reduced.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axis or axes along which a maxima are calculated. The default is to compute the mean of the flattened array.
            If this is a tuple of ints, the maxima are calculated over multiple axes. The indicated axes must be
            ensemble axes.
        keepdims : bool, optional
            If True, the reduced axes are left in the result as dimensions with size one. Default is False.
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
        self, reduction_func, axes, keepdims: bool = False, split_every: int = 2
    ) -> T:
        xp = get_array_module(self.array)

        if axes is None:
            if self.is_lazy:
                return getattr(da, reduction_func)(self.array)
            else:
                return getattr(xp, reduction_func)(self.array)

        if isinstance(axes, Number):
            axes = (axes,)

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

        kwargs = self._copy_kwargs(exclude=("array",))
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

    def _arithmetic(self, other, func) -> T:
        if hasattr(other, "array"):
            self._check_is_compatible(other)
            other = other.array

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = getattr(self.array, func)(other)
        return self.__class__(**kwargs)

    def _in_place_arithmetic(self, other, func) -> T:
        # if hasattr(other, 'array'):
        #    self.check_is_compatible(other)
        #    other = other.array

        if self.is_lazy or (hasattr(other, "is_lazy") and other.is_lazy):
            raise RuntimeError(
                "inplace arithmetic operation not implemented for lazy measurement"
            )
        return self._arithmetic(other, func)

    def __mul__(self, other) -> T:
        return self._arithmetic(other, "__mul__")

    def __imul__(self, other) -> T:
        return self._in_place_arithmetic(other, "__imul__")

    def __truediv__(self, other) -> T:
        return self._arithmetic(other, "__truediv__")

    def __itruediv__(self, other) -> T:
        return self._arithmetic(other, "__itruediv__")

    def __sub__(self, other) -> T:
        return self._arithmetic(other, "__sub__")

    def __isub__(self, other) -> T:
        return self._in_place_arithmetic(other, "__isub__")

    def __add__(self, other) -> T:
        return self._arithmetic(other, "__add__")

    def __iadd__(self, other) -> T:
        return self._in_place_arithmetic(other, "__iadd__")

    def __pow__(self, other) -> T:
        return self._arithmetic(other, "__pow__")

    __rmul__ = __mul__
    __rtruediv__ = __truediv__

    def get_items(
        self, items: int | tuple[int, ...] | slice, keepdims: bool = False
    ) -> T:
        """
        Index the array and the corresponding axes metadata. Only ensemble axes can be indexed.

        Parameters
        ----------
        items : int or tuple of int or slice
            The array is indexed according to this.
        keepdims : bool, optional
            If True, all ensemble axes are left in the result as dimensions with size one. Default is False.

        Returns
        -------
        indexed_array : ArrayObject or subclass of ArrayObject
            The indexed array object.
        """
        if isinstance(items, (Number, slice, type(None), list, np.ndarray)):
            items = (items,)

        elif not isinstance(items, tuple):
            raise NotImplementedError(
                "Indices must be integers or slices or a tuple of integers or slices or None."
            )

        if keepdims:
            items = tuple(
                slice(item, item + 1) if isinstance(item, int) else item
                for item in items
            )

        assert isinstance(items, tuple)

        if any(isinstance(item, (type(...),)) for item in items):
            raise NotImplementedError

        if len(tuple(item for item in items if item is not None)) > len(
            self.ensemble_shape
        ):
            raise RuntimeError("Base axes cannot be indexed.")

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
                metadata = {**metadata, **expanded_axes_metadata.item_metadata(item)}
            else:
                axes_metadata += [expanded_axes_metadata[item].copy()]

        axes_metadata += expanded_axes_metadatas[last_indexed:]

        d = self._copy_kwargs(exclude=("array", "ensemble_axes_metadata", "metadata"))
        d["array"] = self._array[items]
        d["ensemble_axes_metadata"] = axes_metadata
        d["metadata"] = {**self.metadata, **metadata}
        return self.__class__(**d)

    def __getitem__(self, items) -> T:
        return self.get_items(items)

    def expand_dims(
        self, axis: tuple[int, ...] = None, axis_metadata: list[AxisMetadata] = None
    ) -> T:
        """
        Expand the shape of the array object.

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

        kwargs = self._copy_kwargs(exclude=("array", "ensemble_axes_metadata"))
        kwargs["array"] = _expand_dims(self.array, axis=axis)
        kwargs["ensemble_axes_metadata"] = ensemble_axes_metadata
        return self.__class__(**kwargs)

    def squeeze(self, axis: tuple[int, ...] = None) -> T:
        """
        Remove axes of length one from array object.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Selects a subset of the entries of length one in the shape.

        Returns
        -------
        squeezed : ArrayObject or subclass of ArrayObject
            The input array object, but with all or a subset of the dimensions of length 1 removed.
        """
        if len(self.array.shape) < len(self.base_shape):
            return self

        if axis is None:
            axis = range(len(self.shape))
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

    def ensure_lazy(self, chunks: str = "auto") -> T:
        """
        Creates an equivalent lazy version of the array object.

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

    def lazy(self, chunks: str = "auto") -> T:
        return self.ensure_lazy(chunks)

    def compute(
        self,
        progress_bar: bool = None,
        profiler: bool = False,
        resource_profiler: bool = False,
        **kwargs,
    ):
        """
        Turn a lazy *ab*\TEM object into its in-memory equivalent.

        Parameters
        ----------
        progress_bar : bool
            Display a progress bar in the terminal or notebook during computation. The progress bar is only displayed
            with a local scheduler.
        profiler : bool
            Return Profiler class used to profile Dask's execution at the task level. Only execution with a local
            is profiled.
        resource_profiler : bool
            Return ResourceProfiler class is used to profile Dask’s execution at the resource level.
        kwargs :
            Additional keyword arguments passes to `dask.compute`.
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

        output = output[0]

        if profilers:
            return output, profilers

        return output

    def copy_to_device(self, device: str) -> T:
        """
        Copy array to specified device.

        Parameters
        ----------
        device : str

        Returns
        -------
        object_on_device : T
        """
        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = copy_to_device(self.array, device)
        return self.__class__(**kwargs)

    def to_cpu(self) -> T:
        """
        Move the array to the host memory from an arbitrary source array.
        """
        return self.copy_to_device("cpu")

    def to_gpu(self, device: str = "gpu") -> T:
        """
        Move the array from the host memory to a gpu.
        """
        return self.copy_to_device(device)

    def to_zarr(
        self, url: str, compute: bool = True, overwrite: bool = False, **kwargs
    ):
        """
        Write data to a zarr file.

        Parameters
        ----------
        url : str
            Location of the data, typically a path to a local file. A URL can also include a protocol specifier like
            s3:// for remote data.
        compute : bool
            If true compute immediately; return dask.delayed.Delayed otherwise.
        overwrite : bool
            If given array already exists, overwrite=False will cause an error, where overwrite=True will replace the
            existing data.
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

    def _metadata_to_json_string(self):
        return json.dumps(self._metadata_to_dict())

    @staticmethod
    def _metadata_from_json_string(json_string):
        import abtem

        metadata = json.loads(json_string)
        cls = getattr(abtem, metadata["type"])
        del metadata["type"]

        axes_metadata = []
        for key, axis_metadata in metadata["axes"].items():
            axes_metadata.append(axis_from_dict(axis_metadata))

        del metadata["axes"]
        return cls, axes_metadata, metadata

    def _metadata_to_json(self):
        metadata = copy.copy(self.metadata)
        metadata["axes"] = {
            f"axis_{i}": axis_to_dict(axis) for i, axis in enumerate(self.axes_metadata)
        }
        metadata["data_origin"] = f"abTEM_v{__version__}"
        metadata["type"] = self.__class__.__name__
        return json.dumps(metadata)

    def to_tiff(self, filename: str, **kwargs):
        """
        Write data to a tiff file.

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
            array = array.compute()

        return tifffile.imwrite(
            filename, array, description=self._metadata_to_json(), **kwargs
        )

    @classmethod
    def from_zarr(cls, url, chunks: int = "auto") -> T:
        """
        Read wave functions from a hdf5 file.

        url : str
            Location of the data, typically a path to a local file. A URL can also include a protocol specifier like
            s3:// for remote data.
        chunks : tuple of ints or tuples of ints
            Passed to dask.array.from_array(), allows setting the chunks on initialisation, if the chunking scheme in
            the on-disc dataset is not optimal for the calculations to follow.
        """
        return from_zarr(url, chunks=chunks)

    @staticmethod
    def _apply_transform(
        *args,
        array_object_partial,
        transform_partial,
        num_transform_args,
    ):
        args = unpack_blockwise_args(args[:-1]) + (args[-1],)

        transform = transform_partial(*args[:num_transform_args]).item()

        ensemble_axes_metadata = [axis for axis in args[num_transform_args:-1]]

        array = args[-1]

        array_object = array_object_partial((array, ensemble_axes_metadata)).item()

        array = transform._calculate_new_array(array_object)

        ensemble_dims = (
            len(transform.ensemble_shape) + len(array_object.ensemble_shape) + 1
        )

        if transform._num_outputs > 1:
            arr = np.zeros((1,) * ensemble_dims, dtype=object)
            arr.itemset(array)
            return arr

        return array

    @property
    def _has_base_chunks(self):
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

    def apply_transform(
        self, transform: ArrayObjectTransform, max_batch: int | str = "auto"
    ) -> T | ComputableList[T, ...]:
        """
        Transform the wave functions by a given transformation.

        Parameters
        ----------
        transform : ArrayObjectTransform
            The array object transformation to apply.
        max_batch : int, optional
            The number of wave functions in each chunk of the Dask array. If 'auto' (default), the batch size is
            automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".

        Returns
        -------
        transformed_array_object : ArrayObjectTransform
            The transformed array object.
        """

        if self.is_lazy:
            if not transform._allow_base_chunks and self._has_base_chunks:
                raise RuntimeError(
                    f"Transform {transform.__class__} not implemented for array object with chunks along base axes, "
                    f"compute first or use the method `.no_base_chunks` to rechunk."
                )

            if isinstance(max_batch, int):
                max_batch = int(max_batch * np.prod(self.base_shape))

            chunks = transform._default_ensemble_chunks + self.array.chunks

            chunks = validate_chunks(
                transform.ensemble_shape + self.shape,
                chunks,
                limit=max_batch,
                dtype=self.dtype,
            )

            assert chunks[len(transform.ensemble_shape) :] == self.array.chunks

            transform_chunks = chunks[: len(transform.ensemble_shape)]
            array_ensemble_chunks = self.array.chunks[: len(self.ensemble_shape)]

            transform_args, transform_symbols = transform._get_blockwise_args(
                transform_chunks
            )

            axes_args = tuple(
                axis._to_blocks(
                    (c,),
                )
                for axis, c in zip(self.ensemble_axes_metadata, array_ensemble_chunks)
            )

            axes_symbols = tuple(
                tuple_range(length=1, offset=i + len(transform.ensemble_shape))
                for i, args in enumerate(axes_args)
            )

            array_symbols = tuple_range(len(self.shape), len(transform.ensemble_shape))

            num_ensemble_dims = len(transform._out_ensemble_shape(self))

            # if transform._num_outputs > 4:
            #     chunks = chunks[:num_ensemble_dims]
            #     symbols = tuple_range(num_ensemble_dims)
            #     meta = np.array((), dtype=object)
            # else:

            base_shape = transform._out_base_shape(self)
            symbols = tuple_range(num_ensemble_dims + len(base_shape))
            chunks = chunks[: -len(base_shape)] + base_shape
            meta = transform._out_meta(self)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Increasing number of chunks."
                )
                new_array = da.blockwise(
                    self._apply_transform,
                    symbols,
                    *interleave(transform_args, transform_symbols),
                    *interleave(axes_args, axes_symbols),
                    self.array,
                    array_symbols,
                    adjust_chunks={i: chunk for i, chunk in enumerate(chunks)},
                    transform_partial=transform._from_partitioned_args(),
                    num_transform_args=len(transform_args),  # noqa
                    array_object_partial=self._from_partitioned_args(),  # noqa
                    meta=meta,
                    align_arrays=False,
                    concatenate=True,
                )

            if transform._num_outputs > 1:
                outputs = transform._pack_multiple_outputs(self, new_array)
                outputs = ComputableList(outputs)

                return outputs
            else:
                return transform._pack_single_output(self, new_array)
        else:
            return transform.apply(self)

    def set_ensemble_axes_metadata(self, axes_metadata: AxisMetadata, axis: int):
        """
        Sets the axes metadata of an ensemble axis.

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

    def to_hyperspy(self):
        """Convert measurement to a Hyperspy signal."""

        try:
            import hyperspy.api as hs
        except ImportError:
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

        array = xp.transpose(self.array, ensemble_axes + axes_base_indices[::-1])

        s = signal_type(array, axes=ensemble_axes_metadata + axes_base[::-1])

        if self.is_lazy:
            s = s.as_lazy()

        return s

    def _stack(self, arrays, axis_metadata, axis):
        xp = get_array_module(arrays[0].array)

        if arrays[0].is_lazy:
            array = da.stack([measurement.array for measurement in arrays], axis=axis)
        else:
            array = xp.stack([measurement.array for measurement in arrays], axis=axis)

        cls = arrays[0].__class__
        kwargs = arrays[0]._copy_kwargs(exclude=("array",))

        kwargs["array"] = array
        ensemble_axes_metadata = [
            axis_metadata.copy() for axis_metadata in kwargs["ensemble_axes_metadata"]
        ]
        ensemble_axes_metadata.insert(axis, axis_metadata)
        kwargs["ensemble_axes_metadata"] = ensemble_axes_metadata
        return cls(**kwargs)

    def _partition_ensemble_axes_metadata(
        self, chunks: Chunks = None, lazy: bool = True
    ):
        if len(self.ensemble_shape) == 0:
            ensemble_axes_metadata = _wrap_with_array([], 1)
        else:
            chunks = self._validate_ensemble_chunks(chunks)

            ensemble_axes_metadata = np.zeros(chunk_shape(chunks), dtype=object)
            for index, slic in iterate_chunk_ranges(chunks):
                ensemble_axes_metadata.itemset(
                    index,
                    [
                        self.ensemble_axes_metadata[i][slic[i]]
                        for i, axis in enumerate(self.ensemble_axes_metadata)
                    ],
                )

        if lazy:
            ensemble_axes_metadata = da.from_array(ensemble_axes_metadata, chunks=1)

        return ensemble_axes_metadata

    @property
    def _default_ensemble_chunks(self):
        if self.is_lazy:
            return self.array.chunks[: self.ensemble_dims]
        else:
            raise NotImplementedError

    def _partition_args(self, chunks: int = None, lazy: bool = True):
        if chunks is None and self.is_lazy:
            chunks = self.array.chunks[: -len(self.base_shape)]
        elif chunks is None:
            chunks = (1,) * len(self.ensemble_shape)

        chunks = self._validate_ensemble_chunks(chunks)

        if lazy:
            xp = get_array_module(self.array)
            array = self.ensure_lazy().array

            if chunks != array.chunks:
                array = array.rechunk(chunks + array.chunks[len(chunks) :])

            ensemble_axes_metadata = self._partition_ensemble_axes_metadata(
                chunks=chunks
            )

            def _combine_args(*args):
                return args[0], args[1].item()

            ndims = max(len(self.ensemble_shape), 1)
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
                meta=xp.array((), self.dtype),
            )
        else:
            array = self.compute().array
            if len(self.ensemble_shape) == 0:
                blocks = np.zeros((1,), dtype=object)
            else:
                blocks = np.zeros(chunk_shape(chunks), dtype=object)

            ensemble_axes_metadata = self._partition_ensemble_axes_metadata(
                chunks, lazy=False
            )

            for block_indices, chunk_range in iterate_chunk_ranges(chunks):
                if len(block_indices) == 0:
                    block_indices = 0
                # blocks.itemset(
                #     block_indices,
                #     _wrap_with_array(
                #         (array[chunk_range], ensemble_axes_metadata[block_indices]), 1
                #     ),
                # )

                blocks.itemset(
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
        ndims = max(new_array_object.ensemble_dims, 1)
        return _wrap_with_array(new_array_object, ndims)

    def _from_partitioned_args(self):
        return partial(
            self._from_partitioned_args_func,
            **self._copy_kwargs(exclude=("array", "ensemble_axes_metadata")),
        )


def _expand_dims(array: np.ndarray, axis: int | tuple | list) -> np.ndarray:
    if type(axis) not in (tuple, list):
        axis = (axis,)

    out_ndim = len(axis) + array.ndim
    axis = validate_axis(axis, out_ndim)

    shape_it = iter(array.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]

    return array.reshape(shape)


def from_zarr(url: str, chunks: Chunks = None):
    """
    Read abTEM data from zarr.

    Parameters
    ----------
    url : str
        Location of the data. A URL can include a protocol specifier like s3:// for remote data.
    chunks :  tuple of ints or tuples of ints
        Passed to dask.array.from_array(), allows setting the chunks on initialisation, if the chunking scheme in the
        on-disc dataset is not optimal for the calculations to follow.

    Returns
    -------
    imported : ArrayObject
    """
    import abtem

    imported = []
    with zarr.open(url, mode="r") as f:
        i = 0
        types = []
        while True:
            try:
                types.append(f.attrs[f"type{i}"])
            except KeyError:
                break
            i += 1

        for i, t in enumerate(types):
            cls = getattr(abtem, t)

            kwargs = cls._unpack_kwargs(f.attrs[f"kwargs{i}"])
            num_ensemble_axes = len(kwargs["ensemble_axes_metadata"])

            if chunks == "auto":
                chunks = ("auto",) * num_ensemble_axes + (-1,) * cls._base_dims

            array = da.from_zarr(url, component=f"array{i}", chunks=chunks)

            with config.set({"warnings.overspecified-grid": False}):
                imported.append(cls(array, **kwargs))

    if len(imported) == 1:
        imported = imported[0]

    return imported


def stack(
    arrays: Sequence[ArrayObject],
    axis_metadata: AxisMetadata | Sequence[str] = None,
    axis: int = 0,
) -> T:
    """
    Join multiple array objects (e.g. Waves and BaseMeasurement) along a new ensemble axis.

    Parameters
    ----------
    arrays : sequence of array objects
        Each abTEM array object must have the same type and shape.
    axis_metadata : AxisMetadata
        The axis metadata describing the new axis.
    axis : int
        The ensemble axis in the resulting array object along which the input arrays are stacked.
    Returns
    -------
    array_object : ArrayObject
        The stacked array object of the same type as the input.
    """

    assert axis <= len(arrays[0].ensemble_shape)
    assert axis >= 0

    if axis_metadata is None:
        axis_metadata = UnknownAxis()

    elif isinstance(axis_metadata, (tuple, list)):
        if not all(isinstance(element, str) for element in axis_metadata):
            raise ValueError()
        axis_metadata = OrdinalAxis(values=axis_metadata)
    elif not isinstance(axis_metadata, AxisMetadata):
        raise ValueError()

    return arrays[0]._stack(arrays, axis_metadata, axis)


def concatenate(arrays: Sequence[ArrayObject], axis: int = 0) -> T:
    """
    Join a sequence of abTEM array classes along an existing axis.

    Parameters
    ----------
    arrays : sequence of array objects
        Each abTEM array object must have the same type and shape, except in the dimension corresponding to axis. The
        axis metadata along the concatenated axis must be compatible for concatenation.
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


def move_item(lst, from_index, to_index):
    element = lst.pop(from_index)
    lst.insert(to_index, element)
    return lst


def moveaxis(array_object, source, destination):
    xp = get_array_module(array_object.array)

    if array_object.is_lazy:
        array = da.moveaxis(array_object.array, source, destination)
    else:
        array = xp.moveaxis(array_object.array, source, destination)

    cls = array_object.__class__

    axes_metadata = copy.copy(array_object.axes_metadata)

    for s, d in zip(reversed(source), reversed(destination)):
        axes_metadata = move_item(axes_metadata, s, d)

    return cls.from_array_and_metadata(
        array=array, axes_metadata=axes_metadata, metadata=array_object.metadata
    )


def _unpack_array_object_blocks(blocks):
    new_blocks = np.empty(blocks.shape, dtype=object)
    for indices in np.ndindex(blocks.shape):
        new_blocks[indices] = blocks[indices].array
    return new_blocks


def _concatenate_axes_metadata(axes_metadata):
    if len(axes_metadata) == 0:
        raise RuntimeError()

    while len(axes_metadata) > 1:
        axes_metadata = [
            *axes_metadata[:-2],
            axes_metadata[-2].concatenate(axes_metadata[-1]),
        ]
    return axes_metadata[0]


def _axes_metadata_from_array_object_blocks(blocks):
    if blocks.ravel()[0].ensemble_dims == 0:
        return []

    axes_metadata = []
    for i, n in enumerate(blocks.shape):
        index = tuple(slice(None) if j == i else 0 for j in range(len(blocks.shape)))

        axes_metadata.append(
            _concatenate_axes_metadata(
                [
                    block.ensemble_axes_metadata[i]
                    for block in blocks[index]
                    if len(block.ensemble_axes_metadata)
                ]
            )
        )
    return axes_metadata


def _concat_array_object_ensemble_blocks(blocks):
    array_blocks = _unpack_array_object_blocks(blocks)
    concat_array = concatenate_array_blocks(array_blocks)
    concat_axes_metadata = _axes_metadata_from_array_object_blocks(blocks)

    concat_array_object = ArrayObject(
        array=concat_array,
        ensemble_axes_metadata=concat_axes_metadata,
    )
    return concat_array_object
