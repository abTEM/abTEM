"""Module to describe wave function transformations."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Generic, Mapping, Optional, TypeVar

import numpy as np

from abtem.core.axes import AxisMetadata, ParameterAxis
from abtem.core.backend import get_array_module
from abtem.core.chunks import Chunks
from abtem.core.ensemble import (
    EmptyEnsemble,
    Ensemble,
    _wrap_with_array,
)
from abtem.core.fft import ifft2
from abtem.core.utils import CopyMixin, EqualityMixin, expand_dims_to_broadcast
from abtem.distributions import (
    BaseDistribution,
    EnsembleFromDistributions,
    validate_distribution,
)

if TYPE_CHECKING:
    from abtem.array import ArrayObject, ArrayObjectType, ArrayObjectTypeAlt
    from abtem.measurements import BaseMeasurements
    from abtem.waves import Waves
else:
    ArrayObjectType = TypeVar("ArrayObjectType", bound="ArrayObject")
    ArrayObjectTypeAlt = TypeVar("ArrayObjectTypeAlt", bound="ArrayObject")
    ArrayObject = object
    Waves = object
    BaseMeasurements = object

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

    def _out_axes_metadata(
        self, array_object: ArrayObjectType
    ) -> tuple[list[AxisMetadata], ...]:
        return (
            [
                *self._out_ensemble_axes_metadata(array_object)[0],
                *self._out_base_axes_metadata(array_object)[0],
            ],
        )

    @abstractmethod
    def _calculate_new_array(
        self, array_object: ArrayObjectType
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        pass

    @abstractmethod
    def apply(
        self, array_object: ArrayObjectType, max_batch: int | str = "auto"
    ) -> (
        ArrayObjectType
        | ArrayObjectTypeAlt
        | list[ArrayObjectType | ArrayObjectTypeAlt]
    ):
        pass


class EmptyTransform(EmptyEnsemble, ArrayObjectTransform[ArrayObject, ArrayObject]):
    def apply(
        self, array_object: ArrayObject, max_batch: int | str = "auto"
    ) -> ArrayObject:
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


class WavesTransform(EnsembleTransform[Waves, ArrayObjectType]):
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

    @abstractmethod
    def apply(
        self, waves: Waves, max_batch: int | str = "auto"
    ) -> Waves | ArrayObjectType | list[Waves | ArrayObjectType]:
        pass


class WavesToMeasurementTransform(WavesTransform[BaseMeasurements]):
    def apply(
        self, waves: Waves, max_batch: int | str = "auto"
    ) -> Waves | BaseMeasurements | list[Waves | BaseMeasurements]:
        transformed_waves = waves.apply_transform(self, max_batch=max_batch)
        if TYPE_CHECKING:
            assert isinstance(transformed_waves, BaseMeasurements)
        return transformed_waves


class WavesToWavesTransform(WavesTransform):
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


class TransformFromFunc(ArrayObjectTransform[ArrayObject, ArrayObject]):
    def __init__(self, func, func_kwargs):
        self._func = func
        self._func_kwargs = func_kwargs
        super().__init__()

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def _default_ensemble_chunks(self) -> Chunks:
        return ()

    @property
    def func(self):
        return self._func

    @property
    def func_kwargs(self):
        return self._func_kwargs

    def _out_type(
        self, array_object: ArrayObject
    ) -> tuple[type[ArrayObject], ...]:
        return (array_object.__class__,)

    def _calculate_new_array(self, array_object: ArrayObject) -> np.ndarray:
        return self.func(array_object, **self.func_kwargs)

    def _partition_args(self, chunks: Optional[Chunks] = 1, lazy: bool = True) -> tuple:
        return ()

    @classmethod
    def _partial_transform(cls, *args, **kwargs) -> np.ndarray:
        new_transform = _wrap_with_array(cls(**kwargs), ndims=0)
        return new_transform

    def _from_partitioned_args(self) -> Callable[..., np.ndarray]:
        kwargs = self._copy_kwargs()
        return partial(self._partial_transform, **kwargs)

    def apply(
        self, array_object: ArrayObjectType, max_batch: int | str = "auto"
    ) -> ArrayObjectType:
        new_array_object = array_object.apply_transform(self, max_batch=max_batch)
        assert isinstance(new_array_object, array_object.__class__)
        return new_array_object


def join_tuples(tuples: tuple[tuple[Any, ...], ...]) -> tuple[Any, ...]:
    return tuple(item for subtuple in tuples for item in subtuple)


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
