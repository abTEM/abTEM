import copy
import json
import warnings
from abc import abstractmethod
from contextlib import nullcontext, contextmanager
from functools import reduce
from numbers import Number
from operator import mul
from typing import Tuple, Union, TypeVar, List, Sequence

import dask
import dask.array as da
import numpy as np
import zarr
from dask.array.utils import validate_axis
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler
from dask.utils import format_bytes
from dask.distributed import get_client
from tabulate import tabulate

from abtem.core import config
from abtem.core.axes import (
    HasAxes,
    UnknownAxis,
    axis_to_dict,
    axis_from_dict,
    AxisMetadata,
    OrdinalAxis,
    format_axes_metadata,
)
from abtem.core.backend import (
    get_array_module,
    copy_to_device,
    cp,
    device_name_from_array_module,
    check_cupy_is_installed,
)
from abtem.core.chunks import Chunks
from abtem.core.utils import normalize_axes, CopyMixin
from abtem._version import __version__

try:
    import tifffile
except ImportError:
    tifffile = None


class ComputableList(list):
    def to_zarr(
        self,
        urls: str,
        compute: bool = True,
        overwrite: bool = False,
        progress_bar: bool = None,
        **kwargs,
    ):

        if isinstance(urls, str):
            urls = [urls]

        if not len(urls) == len(self):
            raise RuntimeError("Provide a file name for each measurement.")

        computables = [
            m.to_zarr(url, compute=False, overwrite=overwrite)
            for m, url in zip(self, urls)
        ]

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

    def compute(self, **kwargs) -> Union[List, Tuple[List, tuple]]:
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

    # try:
    #     client = get_client()
    #     client.run(config.set, *config.config)
    #     worker_saturation = config.get("dask.worker-saturation")
    #     client.run(
    #         dask.config.set(
    #             {"distributed.scheduler.worker-saturation": worker_saturation}
    #         )
    #     )
    #
    # except ValueError:
    #     pass

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
    # if cp is not None:
    #     cache = cp.fft.config.get_plan_cache()
    #     cache_size = parse_bytes(config.get('cupy.fft-cache-size'))
    #     cache.set_size(cache_size)

    if config.get("device") == "gpu":
        check_cupy_is_installed()

        if not "num_workers" in kwargs:
            kwargs["num_workers"] = cp.cuda.runtime.getDeviceCount()

        if not "threads_per_worker" in kwargs:
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
        return config.get("dask.lazy")

    return lazy


T = TypeVar("T", bound="HasArray")


def format_array(array):
    is_lazy = isinstance(array, da.core.Array)

    nbytes = format_bytes(array.nbytes)
    cbytes = (
        format_bytes(np.prod(array.chunksize) * array.dtype.itemsize)
        if is_lazy
        else "-"
    )
    chunksize = str(array.chunksize) if is_lazy else "-"
    nchunks = reduce(mul, (len(chunks) for chunks in array.chunks)) if is_lazy else "-"
    meta = array._meta if is_lazy else array
    array_type = f'{type(meta).__module__.split(".")[0]}.{type(meta).__name__}'

    ntasks = f"{len(array.dask)} tasks" if is_lazy else "-"

    data = [
        ["array", nbytes, str(array.shape), ntasks, array.dtype.name],
        ["chunks", cbytes, chunksize, f"{str(nchunks)} chunks", array_type],
    ]
    return tabulate(
        data, headers=["", "bytes", "shape", "count", "type"], tablefmt="simple"
    )


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
        if config.get("extended_repr"):
            formatted_type = format_type(self.__class__)
            axes_table = format_axes_metadata(self.axes_metadata, self.shape)
            array_table = format_array(self.array)
            return "\n\n".join([formatted_type, axes_table, array_table])
        else:
            return super().__repr__()

    @classmethod
    def from_array_and_metadata(cls, array, axes_metadata, metadata):
        raise NotImplementedError

    @property
    def metadata(self):
        raise NotImplementedError

    @property
    def base_shape(self) -> Tuple[int, ...]:
        return self.array.shape[-self._base_dims :]

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        return self.array.shape[: -self._base_dims]

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
    def device(self) -> str:
        return device_name_from_array_module(get_array_module(self.array))

    @property
    def is_lazy(self) -> bool:
        return isinstance(self.array, da.core.Array)

    @classmethod
    def _to_delayed_func(cls, array, **kwargs):
        kwargs["array"] = array

        return cls(**kwargs)

    @property
    def is_complex(self) -> str:
        return np.iscomplexobj(self.array)

    def check_is_compatible(self, other: "HasArray"):
        if not isinstance(other, self.__class__):
            raise RuntimeError(
                f"incompatible types ({self.__class__} != {other.__class__})"
            )

        if self.shape != other.shape:
            raise RuntimeError(f"incompatible shapes ({self.shape} != {other.shape})")

        for (key, value), (other_key, other_value) in zip(
            self._copy_kwargs(exclude=("array", "metadata")).items(),
            other._copy_kwargs(exclude=("array", "metadata")).items(),
        ):
            if np.any(value != other_value):
                raise RuntimeError(
                    f"incompatible values for {key} ({value} != {other_value})"
                )

    def mean(self, axes=None, **kwargs) -> "T":
        return self._reduction("mean", axes=axes, **kwargs)

    def sum(self, axes=None, **kwargs) -> "T":
        return self._reduction("sum", axes=axes, **kwargs)

    def std(self, axes=None, **kwargs) -> "T":
        return self._reduction("std", axes=axes, **kwargs)

    def min(self, axes=None, **kwargs) -> "T":
        return self._reduction("min", axes=axes, **kwargs)

    def max(self, axes=None, **kwargs) -> "T":
        return self._reduction("max", axes=axes, **kwargs)

    def _reduction(self, reduction_func, axes, split_every: int = 2) -> "T":
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
            raise RuntimeError("base axes cannot be reduced")

        ensemble_axes_metadata = copy.deepcopy(self.ensemble_axes_metadata)
        ensemble_axes_metadata = [
            axis_metadata
            for axis_metadata, axis in zip(ensemble_axes_metadata, self.ensemble_axes)
            if axis not in axes
        ]

        kwargs = self._copy_kwargs(exclude=("array",))
        if self.is_lazy:
            kwargs["array"] = getattr(da, reduction_func)(
                self.array, axes, split_every=split_every
            )
        else:
            kwargs["array"] = getattr(xp, reduction_func)(self.array, axes)

        kwargs["ensemble_axes_metadata"] = ensemble_axes_metadata
        return self.__class__(**kwargs)

    def _arithmetic(self, other, func) -> "T":
        if hasattr(other, "array"):
            self.check_is_compatible(other)
            other = other.array
        # else:
        #     try:
        #         other = other.item()
        #     except AttributeError:
        #         pass

        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = getattr(self.array, func)(other)
        return self.__class__(**kwargs)

    def _in_place_arithmetic(self, other, func) -> "T":
        # if hasattr(other, 'array'):
        #    self.check_is_compatible(other)
        #    other = other.array

        if self.is_lazy or (hasattr(other, "is_lazy") and other.is_lazy):
            raise RuntimeError(
                "inplace arithmetic operation not implemented for lazy measurement"
            )
        return self._arithmetic(other, func)

    def __mul__(self, other) -> "T":
        return self._arithmetic(other, "__mul__")

    def __imul__(self, other) -> "T":
        return self._in_place_arithmetic(other, "__imul__")

    def __truediv__(self, other) -> "T":
        return self._arithmetic(other, "__truediv__")

    def __itruediv__(self, other) -> "T":
        return self._arithmetic(other, "__itruediv__")

    def __sub__(self, other) -> "T":
        return self._arithmetic(other, "__sub__")

    def __isub__(self, other) -> "T":
        return self._in_place_arithmetic(other, "__isub__")

    def __add__(self, other) -> "T":
        return self._arithmetic(other, "__add__")

    def __iadd__(self, other) -> "T":
        return self._in_place_arithmetic(other, "__iadd__")

    def __pow__(self, other) -> "T":
        return self._arithmetic(other, "__pow__")

    __rmul__ = __mul__
    __rtruediv__ = __truediv__

    def get_items(self, items, keep_dims: bool = False) -> "T":
        if isinstance(items, (Number, slice, type(None))):
            items = (items,)

        elif not isinstance(items, tuple):
            raise NotImplementedError(
                "indices must be integers or slices or a tuple of integers or slices or None"
            )

        if keep_dims:
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
            raise RuntimeError("base axes cannot be indexed")

        expanded_axes_metadata = [
            axis_metadata.copy() for axis_metadata in self.ensemble_axes_metadata
        ]
        for i, item in enumerate(items):
            if item is None:
                expanded_axes_metadata.insert(i, UnknownAxis())

        metadata = {}
        axes_metadata = []
        last_indexed = 0
        for i, item in enumerate(items):
            last_indexed += 1

            if isinstance(item, Number):
                metadata = {**metadata, **expanded_axes_metadata[i].item_metadata(0)}
            else:
                axes_metadata += [expanded_axes_metadata[i][item].copy()]

        axes_metadata += expanded_axes_metadata[last_indexed:]

        d = self._copy_kwargs(exclude=("array", "ensemble_axes_metadata", "metadata"))
        d["array"] = self._array[items]
        d["ensemble_axes_metadata"] = axes_metadata
        d["metadata"] = {**self.metadata, **metadata}
        return self.__class__(**d)

    def __getitem__(self, items) -> "T":
        return self.get_items(items)

    def to_delayed(self):
        return dask.delayed(self._to_delayed_func)(
            self.array, self._copy_kwargs(exclude=("array",))
        )

    def expand_dims(
        self, axis: Tuple[int, ...] = None, axis_metadata: List[AxisMetadata] = None
    ) -> "T":
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
        kwargs["array"] = expand_dims(self.array, axis=axis)
        kwargs["ensemble_axes_metadata"] = ensemble_axes_metadata
        return self.__class__(**kwargs)

    def squeeze(self, axis: Tuple[int, ...] = None) -> "T":
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

    def ensure_lazy(self, chunks="auto") -> "T":

        if self.is_lazy:
            return self

        if chunks == "auto":
            chunks = ("auto",) * len(self.ensemble_shape) + (-1,) * len(self.base_shape)

        array = da.from_array(self.array, chunks=chunks)

        print(array)

        return self.__class__(array, **self._copy_kwargs(exclude=("array",)))

    def compute(
        self,
        progress_bar: bool = None,
        profiler: bool = False,
        resource_profiler=False,
        **kwargs,
    ):
        """
        This turns a lazy abTEM object into its in-memory equivalent.

        Parameters
        ----------
        progress_bar : bool
            Display a progress bar in the terminal or notebook during computation. The progress bar is only displayed
            with a local scheduler.
        profiler : bool
            Return Profiler class used to profile Dask’s execution at the task level. Only execution with a local
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

    def visualize_graph(self, **kwargs):
        return self.array.visualize(**kwargs)

    def copy_to_device(self, device: str) -> "T":
        """Copy array to specified device."""
        kwargs = self._copy_kwargs(exclude=("array",))
        kwargs["array"] = copy_to_device(self.array, device)
        return self.__class__(**kwargs)

    def to_cpu(self) -> "T":
        return self.copy_to_device("cpu")

    def to_gpu(self) -> "T":
        return self.copy_to_device("gpu")

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
            Keyword arguments passed to dask.array.to_zarr.
        """

        with zarr.open(url, mode="w") as root:
            has_array = self.ensure_lazy()

            array = has_array.copy_to_device("cpu").array

            stored = array.to_zarr(
                url, compute=False, component="array", overwrite=overwrite, **kwargs
            )

            kwargs = has_array._copy_kwargs(exclude=("array",))

            self._pack_kwargs(root.attrs, kwargs)

            root.attrs["type"] = self.__class__.__name__

        if compute:
            with _compute_context():
                stored = stored.compute()

        return stored

    @classmethod
    def _pack_kwargs(cls, attrs, kwargs):
        for key, value in kwargs.items():
            if key == "ensemble_axes_metadata":
                attrs[key] = [axis_to_dict(axis) for axis in value]
            else:
                attrs[key] = value

    @classmethod
    def _unpack_kwargs(cls, attrs):
        kwargs = {}
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

    @classmethod
    def _metadata_from_json_string(self, json_string):
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
            Keyword arguments passed to tifffile.imwrite.
        """
        if tifffile is None:
            raise RuntimeError(
                "This functionality of abTEM requires tifffile, see https://github.com/cgohlke/tifffile/"
            )

        array = self.array
        if self.is_lazy:
            warnings.warn("lazy arrays are computed in memory before writing to tiff")
            array = array.compute()

        return tifffile.imwrite(
            filename, array, description=self._metadata_to_json(), **kwargs
        )


    @classmethod
    def from_zarr(cls, url, chunks: int = "auto") -> "T":
        """
        Read wave functions from a hdf5 file.

        url : str
            Location of the data, typically a path to a local file. A URL can also include a protocol specifier like
            s3:// for remote data.
        chunks : int, optional
            aaaa
        """
        with zarr.open(url, mode="r") as f:
            kwargs = cls._unpack_kwargs(f.attrs)

        num_ensemble_axes = len(kwargs["ensemble_axes_metadata"])

        if chunks == "auto":
            chunks = ("auto",) * num_ensemble_axes + (-1,) * cls._base_dims

        array = da.from_zarr(url, component="array", chunks=chunks)

        with config.set({"warnings.overspecified-grid": False}):
            return cls(array, **kwargs)


def expand_dims(a, axis):
    if type(axis) not in (tuple, list):
        axis = (axis,)

    out_ndim = len(axis) + a.ndim
    axis = validate_axis(axis, out_ndim)

    shape_it = iter(a.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]

    return a.reshape(shape)


def from_zarr(url: str, chunks: Chunks = None):
    import abtem

    with zarr.open(url, mode="r") as f:
        name = f.attrs["type"]

    cls = getattr(abtem, name)
    return cls.from_zarr(url, chunks)


def stack(
    has_arrays: Sequence[HasArray], axis_metadata: AxisMetadata = None, axis: int = 0
) -> "T":
    if axis_metadata is None:
        axis_metadata = UnknownAxis()

    elif isinstance(axis_metadata, (tuple, list)):
        if not all(isinstance(element, str) for element in axis_metadata):
            raise ValueError()
        axis_metadata = OrdinalAxis(values=axis_metadata)

    xp = get_array_module(has_arrays[0].array)

    print(axis)

    assert axis <= len(has_arrays[0].ensemble_shape)

    if has_arrays[0].is_lazy:
        array = da.stack([measurement.array for measurement in has_arrays], axis=axis)
    else:
        array = xp.stack([measurement.array for measurement in has_arrays], axis=axis)

    cls = has_arrays[0].__class__
    kwargs = has_arrays[0]._copy_kwargs(exclude=("array",))

    kwargs["array"] = array
    kwargs["ensemble_axes_metadata"] = [axis_metadata] + kwargs[
        "ensemble_axes_metadata"
    ]
    return cls(**kwargs)


def concatenate(has_arrays: Sequence[HasArray], axis: bool = 0) -> "T":
    """
    Join a sequence of abTEM array classes along an existing axis.

    Parameters
    ----------
    has_arrays : list of
    axis :

    Returns
    -------

    """

    xp = get_array_module(has_arrays[0].array)

    if has_arrays[0].is_lazy:
        array = da.concatenate([has_array.array for has_array in has_arrays], axis=axis)
    else:
        array = xp.concatenate([has_array.array for has_array in has_arrays], axis=axis)

    cls = has_arrays[0].__class__

    concatenated_axes_metadata = has_arrays[0].axes_metadata[axis]
    for has_array in has_arrays[1:]:
        concatenated_axes_metadata = concatenated_axes_metadata.concatenate(
            has_array.axes_metadata[axis]
        )

    axes_metadata = copy.deepcopy(has_arrays[0].axes_metadata)
    axes_metadata[axis] = concatenated_axes_metadata

    return cls.from_array_and_metadata(
        array=array, axes_metadata=axes_metadata, metadata=has_arrays[0].metadata
    )
