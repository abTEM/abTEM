"""Module to describe wave function transformations."""
from __future__ import annotations

import itertools
from abc import abstractmethod
from functools import partial, reduce
from typing import TYPE_CHECKING, Iterator, TypeVar

import dask
import dask.array as da
import numpy as np

from abtem.core.axes import AxisMetadata, ParameterAxis
from abtem.core.backend import get_array_module
from abtem.core.chunks import Chunks
from abtem.core.energy import (
    HasAcceleratorMixin,
    Accelerator,
    reciprocal_space_sampling_to_angular_sampling,
)
from abtem.core.ensemble import Ensemble
from abtem.core.fft import ifft2
from abtem.core.grid import HasGridMixin, polar_spatial_frequencies, Grid
from abtem.core.utils import (
    CopyMixin,
    EqualityMixin,
    expand_dims_to_broadcast,
)
from abtem.distributions import (
    EnsembleFromDistributions,
    _validate_distribution,
    BaseDistribution,
)

if TYPE_CHECKING:
    from abtem.waves import Waves, BaseWaves
    from abtem.array import ArrayObject


class ArrayObjectTransform(Ensemble, EqualityMixin, CopyMixin):
    @property
    def metadata(self):
        """Metadata added to the waves when applying the transform."""
        return {}

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        """The shape of the ensemble axes added to the waves when applying the transform."""
        return ()

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        """Axes metadata describing the ensemble axes added to the waves when applying the transform."""
        return []

    def _out_meta(self, array_object: ArrayObject) -> np.ndarray:
        """
        The meta describing the measurement array created when detecting the given waves.

        Parameters
        ----------
        waves : Waves
            The waves to derive the measurement meta from.

        Returns
        -------
        meta : array-like
            Empty array.
        """
        xp = get_array_module(array_object.device)
        return xp.array((), dtype=self._out_dtype(array_object))

    def _out_metadata(self, array_object: ArrayObject) -> dict:
        """
        Metadata added to the measurements created when detecting the given waves.

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the metadata from.

        Returns
        -------
        metadata : dict
        """
        return array_object.metadata

    def _out_dtype(self, array_object: ArrayObject) -> type[np.dtype]:
        """Datatype of the output array."""
        return array_object.dtype

    def _out_shape(self, array_object: ArrayObject) -> tuple[int, ...]:
        """
        Shape of the measurements created when detecting the given waves.

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the measurement shape from.

        Returns
        -------
        measurement_shape : tuple of int
        """
        return array_object.shape

    def _out_type(self, array_object: ArrayObject) -> type[ArrayObject]:
        """
        The type of the created measurements when detecting the given waves.

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the measurement shape from.

        Returns
        -------
        measurement_type : type of :class:`BaseMeasurements`
        """
        return array_object.__class__

    def _out_base_axes_metadata(self, array_object: ArrayObject) -> list[AxisMetadata]:
        """
        Axes metadata of the created measurements when detecting the given waves.

        Parameters
        ----------
        waves : BaseWaves
            The waves to derive the measurement shape from.

        Returns
        -------
        axes_metadata : list of :class:`AxisMetadata`
        """
        return array_object.base_axes_metadata

    def __add__(self, other: ArrayObjectTransform) -> CompositeArrayObjectTransform:
        transforms = []

        for transform in (self, other):

            if hasattr(transform, "transforms"):
                transforms += transform.transforms
            else:
                transforms += [transform]

        return CompositeArrayObjectTransform(transforms)

    @abstractmethod
    def apply(self, array_object: ArrayObject) -> ArrayObject:
        """
        Apply the transform to the given waves.

        Parameters
        ----------
        array_object : ArrayObject
            The array object to transform.

        Returns
        -------
        transformed_array_object : ArrayObject
        """
        pass


T = TypeVar("T", bound="ArrayObject")


class EnsembleTransform(EnsembleFromDistributions, ArrayObjectTransform):
    def __init__(self, distributions: tuple[str, ...] = ()):
        super().__init__(distributions=distributions)

    @staticmethod
    def _validate_distribution(distribution):
        return _validate_distribution(distribution)

    @property
    def ensemble_axes_metadata(self):
        return []

    def _axes_metadata_from_distributions(self, **kwargs):
        ensemble_axes_metadata = []
        for distribution_name in self._distributions:
            distribution = getattr(self, distribution_name)
            if isinstance(distribution, BaseDistribution):
                axis_kwargs = kwargs[distribution_name]
                ensemble_axes_metadata += [
                    ParameterAxis(
                        values=distribution,
                        _ensemble_mean=distribution.ensemble_mean,
                        **axis_kwargs,
                    )
                ]

        return ensemble_axes_metadata

    def _pack_array(
        self,
        array_object: ArrayObject,
        new_array: np.ndarray,
        exclude: tuple[str, ...] = (),
    ):

        kwargs = array_object._copy_kwargs(exclude=("array",) + exclude)
        kwargs["ensemble_axes_metadata"] = (
            self.ensemble_axes_metadata + kwargs["ensemble_axes_metadata"]
        )
        kwargs["metadata"].update(self.metadata)
        return array_object.__class__(new_array, **kwargs)

    @abstractmethod
    def _calculate_new_array(self, array_object: ArrayObject) -> np.ndarray:
        pass

    def apply(self, array_object: ArrayObject) -> ArrayObject | T:
        new_array = self._calculate_new_array(array_object)
        new_array_object = self._pack_array(array_object, new_array)
        return new_array_object


class WavesTransform(EnsembleTransform):


    def apply(self, waves: Waves) -> Waves:
        waves = super().apply(waves)
        return waves


class CompositeArrayObjectTransform(ArrayObjectTransform):
    """
    Combines multiple array object transformations into a single transformation.

    Parameters
    ----------
    transforms : ArrayObject
        The array object to transform.
    """

    def __init__(self, transforms: list[ArrayObjectTransform] = None):
        if transforms is None:
            transforms = []

        self._transforms = transforms
        super().__init__()

    def insert(
        self, transform: ArrayObjectTransform, index: int
    ) -> CompositeArrayObjectTransform:
        """
        Inserts an array object transform to the sequence of transforms before the specified index.

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

    @property
    def transforms(self) -> list[ArrayObjectTransform]:
        """The list of transforms in the composite."""
        return self._transforms

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        ensemble_axes_metadata = [
            wave_transform.ensemble_axes_metadata for wave_transform in self.transforms
        ]
        return list(itertools.chain(*ensemble_axes_metadata))

    @property
    def _default_ensemble_chunks(self) -> Chunks:
        default_ensemble_chunks = [
            transform._default_ensemble_chunks for transform in self.transforms
        ]
        return tuple(itertools.chain(*default_ensemble_chunks))

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        ensemble_shape = [
            wave_transform.ensemble_shape for wave_transform in self.transforms
        ]
        return tuple(itertools.chain(*ensemble_shape))

    def apply(self, array_object: ArrayObject):
        for transform in reversed(self.transforms):
            array_object = transform.apply(array_object)

        return array_object

    def _partition_args(self, chunks=None, lazy: bool = True):
        if chunks is None:
            chunks = self._default_ensemble_chunks

        chunks = self._validate_chunks(chunks)

        blocks = ()
        start = 0
        for wave_transform in self.transforms:
            stop = start + len(wave_transform.ensemble_shape)
            blocks += wave_transform._partition_args(chunks[start:stop], lazy=lazy)
            start = stop

        return blocks

    @staticmethod
    def _partial(*args, partials):
        wave_transfer_functions = []
        for p in partials:
            wave_transfer_functions += [p[0](*[args[i] for i in p[1]])]

        return CompositeArrayObjectTransform(wave_transfer_functions)

    def _from_partitioned_args(self):
        partials = ()
        i = 0
        for wave_transform in self.transforms:
            arg_indices = tuple(range(i, i + len(wave_transform.ensemble_shape)))
            partials += ((wave_transform._from_partitioned_args(), arg_indices),)
            i += len(arg_indices)

        return partial(self._partial, partials=partials)


class ReciprocalSpaceMultiplication(WavesTransform, HasAcceleratorMixin, HasGridMixin):
    """


    Parameters
    ----------
    energy : float, optional
        Electron energy [eV]. If not provided, inferred from the wave functions.
    extent : float or two float, optional
        Lateral extent of wave functions [Å] in `x` and `y` directions. If a single float is given, both are set equal.
    gpts : two ints, optional
        Number of grid points describing the wave functions.
    sampling : two float, optional
        Lateral sampling of wave functions [1/Å]. If 'gpts' is also given, will be ignored.
    in_place: bool, optional
        If True, the array representing the waves may be modified in-place.
    device : str, optional
        The probe wave functions will be build and stored on this device ('cpu' or 'gpu'). The default is determined by
        the user configuration.
    distributions : tuple of str, optional
        Names of properties that may be described by a distribution.
    """

    def __init__(
        self,
        energy: float = None,
        extent: float | tuple[float, float] = None,
        gpts: int | tuple[int, int] = None,
        sampling: float | tuple[float, float] = None,
        in_place: bool = False,
        device: str = "cpu",
        distributions: tuple[str, ...] = (),
        **kwargs,
    ):
        self._accelerator = Accelerator(energy=energy, **kwargs)
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        self._in_place = in_place
        self._device = device
        super().__init__(distributions=distributions)

    @property
    def in_place(self) -> bool:
        return self._in_place

    @property
    def device(self) -> str:
        return self._device

    @abstractmethod
    def _evaluate_from_angular_grid(self, alpha, phi):
        pass

    @property
    def angular_sampling(self) -> tuple[float, float]:
        return reciprocal_space_sampling_to_angular_sampling(
            self.reciprocal_space_sampling, self.energy
        )

    def _angular_grid(self) -> tuple[np.ndarray, np.ndarray]:
        xp = get_array_module(self._device)
        alpha, phi = polar_spatial_frequencies(self.gpts, self.sampling, xp=xp)
        alpha *= self.wavelength
        return alpha, phi

    def _evaluate(self) -> np.ndarray:
        alpha, phi = self._angular_grid()
        return self._evaluate_from_angular_grid(alpha, phi)

    def evaluate(self, waves: BaseWaves = None, lazy: bool = False) -> np.ndarray:
        """
        Evaluate the array to be multiplied with the waves in reciprocal space.

        Parameters
        ----------
        waves : BaseWaves, optional
            If given, the array will be evaluated to match the provided waves.
        lazy : bool, optional
            If True, the array is lazily evaluated, a Dask array is returned.

        Returns
        -------
        kernel : np.ndarray or dask.array.Array
        """

        if waves is not None:
            self.accelerator.match(waves)
            self.grid.match(waves)

        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        if lazy:
            array = dask.delayed(self._evaluate)()
            array = da.from_delayed(
                array, dtype=np.complex64, shape=self.ensemble_shape + self.gpts
            )
            return array
        else:
            return self._evaluate()

    def to_diffraction_patterns(self):
        from abtem.measurements import DiffractionPatterns

        array = self.evaluate()
        diffraction_patterns = DiffractionPatterns(
            array,
            sampling=self.reciprocal_space_sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            fftshift=False,
        )
        diffraction_patterns = diffraction_patterns.shift_spectrum("center")
        return diffraction_patterns

    def show(self, **kwargs):
        return self.to_diffraction_patterns().show(**kwargs)

    def _calculate_new_array(self, waves: Waves) -> np.ndarray:
        real_space_in = not waves.reciprocal_space

        waves = waves.ensure_reciprocal_space(overwrite_x=self.in_place)
        kernel = self.evaluate(waves, lazy=False)

        kernel, new_array = expand_dims_to_broadcast(
            kernel, waves.array, match_dims=[(-2, -1), (-2, -1)]
        )

        xp = get_array_module(self.device)

        kernel = xp.array(kernel)

        if self.in_place:
            new_array *= kernel
        else:
            new_array = new_array * kernel

        if real_space_in:
            new_array = ifft2(new_array, overwrite_x=self.in_place)

        return new_array
