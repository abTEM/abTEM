"""Module to describe wave function transformations."""

from __future__ import annotations

import itertools
from abc import ABCMeta, abstractmethod
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Generic, Iterator, Mapping, Optional, TypeVar

import dask.array as da
import numpy as np

from abtem.core.axes import AxisMetadata, ParameterAxis
from abtem.core.backend import get_array_module
from abtem.core.chunks import Chunks, validate_chunks
from abtem.core.ensemble import (
    EmptyEnsemble,
    Ensemble,
    _wrap_with_array,
    unpack_blockwise_args,
)
from abtem.core.fft import ifft2
from abtem.core.utils import (
    CopyMixin,
    EqualityMixin,
    expand_dims_to_broadcast,
)
from abtem.distributions import (
    BaseDistribution,
    EnsembleFromDistributions,
    validate_distribution,
)

if TYPE_CHECKING:
    from abtem.array import ArrayObject, ArrayObjectType, ArrayObjectTypeAlt
    from abtem.waves import Waves
else:
    ArrayObject = object
    ArrayObjectType = TypeVar("ArrayObjectType", bound="ArrayObject")
    ArrayObjectTypeAlt = TypeVar("ArrayObjectTypeAlt", bound="ArrayObject")
    Waves = object

# if TYPE_CHECKING:
#     from abtem.array import ArrayObject, ArrayObjectType, ArrayObjectTypeAlt
#    from abtem.waves import Waves
WavesType = TypeVar("WavesType", bound=Waves)


class ArrayObjectTransform(
    Generic[ArrayObjectType, ArrayObjectTypeAlt],
    Ensemble,
    EqualityMixin,
    CopyMixin,
    metaclass=ABCMeta,
):
    _allow_base_chunks: bool = False

    @property
    def _num_outputs(self) -> int:
        return 1

    @property
    def metadata(self) -> dict:
        """Metadata added to the waves when applying the transform."""
        return {}

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        """The shape of the ensemble axes added to the waves when applying the
        transform."""
        return ()

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        """Axes metadata describing the ensemble axes added to the waves when applying
        the transform."""
        return []

    def _out_meta(self, array_object: ArrayObjectType) -> tuple[np.ndarray, ...]:
        """
        The meta describing the measurement array created when detecting the given
        waves.

        Parameters
        ----------
        array_object : ArrayObject
            The array object to derive the measurement meta from.

        Returns
        -------
        meta : array-like
            Empty array.
        """
        xp = get_array_module(array_object.device)
        return (xp.array((), dtype=self._out_dtype(array_object)[0]),)

    def _out_metadata(self, array_object: ArrayObjectType) -> tuple[dict, ...]:
        """
        Metadata added to the measurements created when detecting the given waves.

        Parameters
        ----------
        array_object : ArrayObject
            The array object to derive the metadata from.

        Returns
        -------
        metadata : dict
        """
        return ({**array_object.metadata, **self.metadata},)

    def _out_dtype(self, array_object: ArrayObjectType) -> tuple[np.dtype, ...]:
        """Datatype of the output array."""
        return (array_object.dtype,)

    def _out_type(
        self, array_object: ArrayObjectType
    ) -> tuple[type[ArrayObjectType], ...] | tuple[type[ArrayObjectTypeAlt], ...]:
        """
        The subtype of the created array object after applying the transform.

        Parameters
        ----------
        array_object : ArrayObject
            The waves to derive the measurement shape from.

        Returns
        -------
        measurement_type : type of :class:`BaseMeasurements`
        """
        return (array_object.__class__,)

    def _out_ensemble_shape(
        self, array_object: ArrayObjectType
    ) -> tuple[tuple[int, ...], ...]:
        """
        Shape of the measurements created when detecting the given waves.

        Parameters
        ----------
        array_object : ArrayObject
            The array object to derive the shape of the output array object from.

        Returns
        -------
        measurement_shape : tuple of int
        """
        return (self.ensemble_shape + array_object.ensemble_shape,)

    def _out_base_shape(
        self, array_object: ArrayObjectType
    ) -> tuple[tuple[int, ...], ...]:
        """
        Shape of the array object created by the transformation.

        Parameters
        ----------
        array_object : ArrayObject
            The waves to derive the measurement shape from.

        Returns
        -------
        measurement_shape : tuple of int
        """
        return (array_object.base_shape,)

    def _out_shape(self, array_object: ArrayObjectType) -> tuple[tuple[int, ...], ...]:
        ensemble_shapes = self._out_ensemble_shape(array_object)

        base_shapes = self._out_base_shape(array_object)

        return tuple(
            ensemble_shape + base_shape
            for ensemble_shape, base_shape in zip(ensemble_shapes, base_shapes)
        )

    def _out_dims(self, array_object: ArrayObjectType) -> tuple[int, ...]:
        return tuple(len(dims) for dims in self._out_shape(array_object))

    def _out_base_axes_metadata(
        self, array_object: ArrayObjectType
    ) -> tuple[list[AxisMetadata], ...]:
        """
        Axes metadata of the created measurements when detecting the given waves.

        Parameters
        ----------
        array_object: ArrayObject
            The waves to derive the measurement shape from.

        Returns
        -------
        axes_metadata : list of :class:`AxisMetadata`
        """

        return (array_object.base_axes_metadata,)

    def _out_ensemble_axes_metadata(
        self, array_object: ArrayObjectType
    ) -> tuple[list[AxisMetadata], ...]:
        return ([*self.ensemble_axes_metadata, *array_object.ensemble_axes_metadata],)

    def __add__(self, other: ArrayObjectTransform) -> CompositeArrayObjectTransform:
        transforms = []

        for transform in (self, other):
            if hasattr(transform, "transforms"):
                transforms += transform.transforms
            else:
                transforms += [transform]

        return CompositeArrayObjectTransform(transforms)

    def _out_axes_metadata(
        self, array_object: ArrayObjectType
    ) -> tuple[list[AxisMetadata], ...]:
        return (
            [
                *self._out_ensemble_axes_metadata(array_object)[0],
                *self._out_base_axes_metadata(array_object)[0],
            ],
        )

    def _get_blockwise_args(self, chunks):
        def _tuple_range(length, offset=0):
            return tuple(range(offset, offset + length))

        def _arrays_to_symbols(arrays):
            offset = 0
            symbols = ()
            for array in arrays:
                length = len(array.shape)
                symbols += (_tuple_range(length=length, offset=offset),)
                offset += length
            return symbols

        transform_args = self._partition_args(chunks=chunks)
        transform_symbols = _arrays_to_symbols(transform_args)

        assert sum(len(args.shape) for args in transform_args) == sum(
            len(symbols) for symbols in transform_symbols
        )
        return transform_args, transform_symbols

    @staticmethod
    def _extract(array: np.ndarray, index: int) -> np.ndarray:
        try:
            array = array.item()[index]
        except AttributeError:
            array = array[index]
        return array

    def _pack_multiple_outputs(
        self, array_object: ArrayObjectType, new_arrays: tuple[np.ndarray, ...]
    ) -> tuple[ArrayObjectType | ArrayObjectTypeAlt, ...]:
        base_shapes = self._out_base_shape(array_object)
        ensemble_shapes = self._out_ensemble_shape(array_object)
        metas = self._out_meta(array_object)
        clss = self._out_type(array_object)
        metadatas = self._out_metadata(array_object)
        base_axes_metadatas = self._out_base_axes_metadata(array_object)
        ensemble_axes_metadatas = self._out_ensemble_axes_metadata(array_object)

        assert len(base_shapes) == self._num_outputs
        assert len(ensemble_shapes) == self._num_outputs
        assert len(metas) == self._num_outputs
        assert len(clss) == self._num_outputs
        assert len(metadatas) == self._num_outputs
        assert len(base_axes_metadatas) == self._num_outputs
        assert len(ensemble_axes_metadatas) == self._num_outputs

        outputs: tuple[ArrayObjectType | ArrayObjectTypeAlt, ...] = ()
        for i in range(self._num_outputs):
            base_shape = base_shapes[i]
            ensemble_shape = ensemble_shapes[i]
            meta = metas[i]
            cls = clss[i]
            metadata = metadatas[i]
            base_axes_metadata = base_axes_metadatas[i]
            ensemble_axes_metadata = ensemble_axes_metadatas[i]

            if isinstance(new_arrays, da.core.Array):
                shape = ensemble_shape + base_shape
                consumed_ensemble_axes = len(new_arrays.shape) - len(ensemble_shape)
                new_axis = tuple(range(len(ensemble_shape), len(shape)))
                chunks = new_arrays.chunks
                chunks = chunks + tuple((n,) for n in base_shape)
                chunks = chunks[: len(chunks) - consumed_ensemble_axes]
                new_axis = new_axis[: len(new_axis) - consumed_ensemble_axes]

                new_array = da.map_blocks(
                    self._extract,
                    new_arrays,
                    index=i,
                    chunks=chunks,
                    new_axis=new_axis,
                    meta=meta,
                )
            else:
                new_array = new_arrays[i]

            axes_metadata = ensemble_axes_metadata + base_axes_metadata

            output = cls.from_array_and_metadata(
                new_array, axes_metadata=axes_metadata, metadata=metadata
            )

            outputs += (output,)

        return outputs

    def _pack_single_output(
        self, array_object: ArrayObjectType, new_array: np.ndarray
    ) -> ArrayObjectType | ArrayObjectTypeAlt:
        ensemble_axes_metadata = self._out_ensemble_axes_metadata(array_object)

        base_axes_metadata = self._out_base_axes_metadata(array_object)

        if len(ensemble_axes_metadata) > 1 or len(base_axes_metadata) > 1:
            raise NotImplementedError

        axes_metadata = ensemble_axes_metadata[0] + base_axes_metadata[0]

        cls = self._out_type(array_object)[0]
        metadata = self._out_metadata(array_object)[0]

        new_array_object = cls.from_array_and_metadata(
            new_array, axes_metadata=axes_metadata, metadata=metadata
        )

        return new_array_object

    @abstractmethod
    def _calculate_new_array(
        self, array_object: ArrayObjectType
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        pass

    def _apply(
        self, array_object: ArrayObjectType
    ) -> (
        ArrayObjectType
        | ArrayObjectTypeAlt
        | tuple[ArrayObjectType | ArrayObjectTypeAlt, ...]
    ):
        new_array = self._calculate_new_array(array_object)
        if self._num_outputs > 1:
            assert isinstance(new_array, tuple)
            return self._pack_multiple_outputs(array_object, new_array)
        else:
            assert isinstance(new_array, np.ndarray)
            return self._pack_single_output(array_object, new_array)


class EmptyTransform(
    EmptyEnsemble, ArrayObjectTransform[ArrayObjectType, ArrayObjectTypeAlt]
):
    def apply(self, array_object: ArrayObject) -> ArrayObject | tuple[ArrayObject, ...]:
        return array_object


class EnsembleTransform(
    EnsembleFromDistributions, ArrayObjectTransform[ArrayObjectType, ArrayObjectTypeAlt]
):
    def __init__(self, distributions: tuple[str, ...] = ()):
        super().__init__(distributions=distributions)

    @staticmethod
    def _validate_distribution(distribution):
        return validate_distribution(distribution)

    def _validate_ensemble_axes_metadata(
        self, ensemble_axes_metadata: list[AxisMetadata]
    ) -> list[AxisMetadata]:
        if isinstance(ensemble_axes_metadata, AxisMetadata):
            ensemble_axes_metadata = [ensemble_axes_metadata]

        assert len(ensemble_axes_metadata) == len(self.ensemble_shape)
        return ensemble_axes_metadata

    def _get_axes_metadata_from_distributions(
        self, **kwargs: Mapping[str, Any]
    ) -> list[AxisMetadata]:
        ensemble_axes_metadata: list[AxisMetadata] = []
        for name, value in kwargs.items():
            assert name in self._distributions
            distribution = getattr(self, name)
            if isinstance(distribution, BaseDistribution):
                ensemble_axes_metadata += [
                    ParameterAxis(
                        values=tuple(distribution),
                        _ensemble_mean=distribution.ensemble_mean,
                        **value,
                    )
                ]

        return ensemble_axes_metadata


class WavesTransform(EnsembleTransform[WavesType, ArrayObjectType]):
    def __init__(self, distributions: tuple[str, ...] = ()):
        super().__init__(distributions=distributions)

    @property
    def distributions(self) -> tuple[str, ...]:
        return self._distributions

    @abstractmethod
    def _calculate_new_array(
        self, waves: WavesType
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        pass

    def apply(self, waves: WavesType) -> ArrayObject | list[ArrayObject]:
        transformed_waves = waves.apply_transform(self)
        return transformed_waves


class WavesToWavesTransform(WavesTransform[Waves, Waves]):
    def __init__(self, distributions: tuple[str, ...] = ()):
        super().__init__(distributions=distributions)

    @abstractmethod
    def _calculate_new_array(self, waves: Waves) -> np.ndarray:
        pass

    def _out_type(self, array_object: Waves) -> tuple[type[Waves], ...]:
        return (array_object.__class__,)

    def apply(self, waves: Waves, max_batch: int | str = "auto") -> Waves:
        transformed_waves = waves.apply_transform(self, max_batch=max_batch)
        if TYPE_CHECKING:
            assert isinstance(transformed_waves, Waves)
        return transformed_waves


class TransformFromFunc(WavesTransform):
    def __init__(self, func, func_kwargs):
        self._func = func
        self._func_kwargs = func_kwargs
        super().__init__()

    @property
    def func(self):
        return self._func

    @property
    def func_kwargs(self):
        return self._func_kwargs

    def _calculate_new_array(self, array_object):
        return self.func(array_object, **self.func_kwargs)


def join_tuples(tuples: tuple[tuple[Any, ...], ...]) -> tuple[Any, ...]:
    return tuple(item for subtuple in tuples for item in subtuple)


class CompositeArrayObjectTransform(ArrayObjectTransform):
    """
    Combines multiple array object transformations into a single transformation.

    Parameters
    ----------
    transforms : ArrayObject
        The array object to transform.
    """

    def __init__(
        self,
        transforms: Optional[list[ArrayObjectTransform]] = None,
    ):
        if transforms is None:
            transforms = []

        self._transforms = transforms

        self._base_shapes: Optional[tuple] = None
        self._ensemble_shapes: Optional[tuple] = None
        self._base_axes_metadata: Optional[tuple] = None
        self._ensemble_axes_metadata: Optional[tuple] = None
        self._types: Optional[tuple] = None
        self._metas: Optional[tuple] = None
        self._metadata: Optional[tuple] = None
        super().__init__()

    @property
    def _num_outputs(self) -> int:
        return self._transforms[0]._num_outputs

    def set_output_specification(self, array_object):
        for transform in reversed(self.transforms):
            array_object = array_object.apply_transform(transform)

        if self._num_outputs == 1:
            output = [array_object]
        else:
            output = array_object

        self._base_shapes = ()
        self._ensemble_shapes = ()
        self._base_axes_metadata = ()
        self._ensemble_axes_metadata = ()
        self._metas = ()
        self._types = ()
        self._metadata = ()
        for i in range(self._num_outputs):
            self._base_shapes += (output[i].base_shape,)
            self._ensemble_shapes += (output[i].ensemble_shape,)
            self._base_axes_metadata += (output[i].base_axes_metadata,)
            self._ensemble_axes_metadata += (output[i].ensemble_axes_metadata,)
            xp = get_array_module(output[i].array)
            self._metas += (xp.array((), dtype=output[i].dtype),)
            self._types += (output[i].__class__,)
            self._metadata += (output[i].metadata,)

        return self

    def insert(
        self, transform: ArrayObjectTransform, index: int
    ) -> CompositeArrayObjectTransform:
        """
        Inserts an array object transform to the sequence of transforms before the
        specified index.

        Parameters
        ----------
        transform : ArrayObjectTransform
            Array object transform to insert.
        index : int
            The array object transform is inserted before this index.

        Returns
        -------
        composite_array_transform : CompositeArrayObjectTransform
        """
        self._transforms.insert(index, transform)
        return self

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self) -> Iterator[ArrayObjectTransform]:
        return iter(self.transforms)

    @property
    def metadata(self):
        metadata = [transform.metadata for transform in self.transforms]
        return reduce(lambda a, b: {**a, **b}, metadata)

    def _out_metadata(self, array_object):
        if self._metadata is not None:
            return (self._metadata,)

        metadata = [
            transform._out_metadata(array_object)[0] for transform in self.transforms
        ]
        out_metadata = reduce(lambda a, b: {**a, **b}, metadata)
        return (out_metadata,)

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        ensemble_axes_metadata = list(
            itertools.chain(
                *(transform.ensemble_axes_metadata for transform in self.transforms)
            )
        )
        return ensemble_axes_metadata

    def _out_ensemble_axes_metadata(
        self, array_object
    ) -> tuple[list[AxisMetadata], ...]:
        if self._ensemble_axes_metadata is not None:
            return self._ensemble_axes_metadata
        return (self.ensemble_axes_metadata + array_object.ensemble_axes_metadata,)

    def _out_base_axes_metadata(self, array_object) -> tuple[list[AxisMetadata], ...]:
        if self._base_axes_metadata is not None:
            return self._base_axes_metadata

        return self.transforms[0]._out_base_axes_metadata(array_object)

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        ensemble_shape = [transform.ensemble_shape for transform in self.transforms]
        return tuple(itertools.chain(*ensemble_shape))

    def _out_ensemble_shape(self, array_object) -> tuple[tuple[int, ...], ...]:
        if self._ensemble_shapes is not None:
            return self._ensemble_shapes

        ensemble_shape = self.ensemble_shape + array_object.ensemble_shape
        return (ensemble_shape,)

    def _out_base_shape(self, array_object):
        if self._base_shapes is not None:
            return self._base_shapes

        return self.transforms[0]._out_base_shape(array_object)

    def _out_meta(self, array_object):
        if self._metas is not None:
            return self._metas
        return self.transforms[0]._out_meta(array_object)

    def _out_dtype(self, array_object):
        if self._metas is not None:
            return tuple(meta.dtype for meta in self._metas)

        return tuple(meta.dtype for meta in self._out_meta(array_object))

    def _out_type(self, array_object):
        if self._types is not None:
            return self._types

        return self.transforms[0]._out_type(array_object)

    @property
    def transforms(self) -> list[ArrayObjectTransform]:
        """The list of transforms in the composite."""
        return self._transforms

    @property
    def _default_ensemble_chunks(self) -> Chunks:
        chunks = tuple(
            transform._default_ensemble_chunks for transform in self.transforms
        )
        chunks_tuples = tuple(c if isinstance(c, tuple) else (c,) for c in chunks)
        return join_tuples(chunks_tuples)

    def apply(self, array_object: ArrayObject) -> ArrayObject | list[ArrayObject]:
        if len(self) > 0:
            return array_object.apply_transform(self)
        else:
            return array_object

    def _calculate_new_array(
        self, array_object: ArrayObject
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        for transform in reversed(self.transforms):
            new_array_object = transform._apply(array_object)
            assert isinstance(new_array_object, ArrayObject)
            array_object = new_array_object

        if self._num_outputs > 1:
            return tuple(array_object[i]._eager_array for i in range(self._num_outputs))
        else:
            return array_object._eager_array

    @staticmethod
    def _partial(*args, partials):
        args = unpack_blockwise_args(args)

        transforms = []
        for partial_func, arg_indices in partials:
            partial_args = tuple(args[i] for i in arg_indices)
            transforms += [partial_func(*partial_args).item()]

        new_transform = CompositeArrayObjectTransform(transforms)
        return _wrap_with_array(new_transform)

    def _from_partitioned_args(self):
        partials = ()
        i = 0
        for transform in self.transforms:
            num_args = len(transform._partition_args(1))
            arg_indices = tuple(range(i, i + num_args))
            partials += ((transform._from_partitioned_args(), arg_indices),)
            i += num_args
        return partial(self._partial, partials=partials)

    def _partition_args(self, chunks=None, lazy: bool = True):
        if chunks is None:
            chunks = self._default_ensemble_chunks

        chunks = validate_chunks(self.ensemble_shape, chunks, max_elements="auto")

        chunks = self._validate_ensemble_chunks(chunks)

        blocks = ()
        start = 0
        for transform in self.transforms:
            stop = start + len(transform.ensemble_shape)
            blocks += transform._partition_args(chunks[start:stop], lazy=lazy)
            start = stop

        return blocks


class ReciprocalSpaceMultiplication(WavesToWavesTransform):
    """
    Wave function transformation for multiplying each member of an ensemble of wave
    functions with an array.

    Parameters
    ----------
    in_place: bool, optional
        If True, the array representing the waves may be modified in-place.
    distributions : tuple of str, optional
        Names of properties that may be described by a distribution.
    """

    def __init__(
        self,
        in_place: bool = False,
        distributions: tuple[str, ...] = (),
    ):
        self._in_place = in_place
        super().__init__(distributions=distributions)

    @property
    def in_place(self) -> bool:
        """The array representing the waves may be modified in-place."""
        return self._in_place

    @abstractmethod
    def _evaluate_kernel(self, waves: Waves) -> np.ndarray:
        pass

    def _calculate_new_array(self, waves: Waves) -> np.ndarray:
        real_space_in = not waves.reciprocal_space

        waves = waves.ensure_reciprocal_space(overwrite_x=self.in_place)
        kernel = self._evaluate_kernel(waves)

        array = waves._eager_array

        kernel, new_array = expand_dims_to_broadcast(
            kernel, array, match_dims=((-2, -1), (-2, -1))
        )

        xp = get_array_module(array)

        kernel = xp.array(kernel)

        if self.in_place:
            new_array *= kernel
        else:
            new_array = new_array * kernel

        if real_space_in:
            new_array = ifft2(new_array, overwrite_x=self.in_place)

        return new_array
