"""Module for describing distributions of simulation parameters."""
from __future__ import annotations

from abc import abstractmethod, ABCMeta
from functools import partial
from numbers import Number
from typing import Sequence, Iterable, Iterator

import dask.array as da
import numpy as np

from abtem.core.backend import get_array_module, ArrayModule
from abtem.core.chunks import Chunks, equal_sized_chunks
from abtem.core.ensemble import Ensemble, _wrap_with_array, unpack_blockwise_args
from abtem.core.utils import EqualityMixin, CopyMixin


class BaseDistribution(EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """
    Base object for defining distributions of simulation parameters.
    """

    def __array__(self):
        return self.values

    def __iter__(self) -> Iterator[float]:
        return iter(self.values)

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """The number of dimensions in the distribution."""
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the distribution parameters."""
        pass

    @abstractmethod
    def divide(self, chunks: Chunks, lazy: bool = True):
        """Divide the distribution into chunks."""
        pass

    @property
    @abstractmethod
    def ensemble_mean(self) -> bool:
        """Calculate the mean of the ensemble."""
        pass

    @property
    @abstractmethod
    def values(self) -> np.ndarray:
        """Scalar values representing the distribution."""
        pass

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        """Weight of each of distribution value."""
        pass


class DistributionFromValues(BaseDistribution):
    def __init__(
        self,
        values: np.ndarray,
        weights: np.ndarray = None,
        ensemble_mean: bool = False,
    ):
        self._values = values

        if weights is None:
            weights = np.ones(len(values))

        self._weights = weights

        self._ensemble_mean = ensemble_mean

    def __neg__(self) -> DistributionFromValues:
        return self.__class__(
            values=-self.values, weights=self.weights, ensemble_mean=self.ensemble_mean
        )

    @property
    def dimensions(self) -> int:
        if len(self.shape) > 1:
            return self.shape[1]

        return 1

    @property
    def shape(self) -> tuple[int]:
        return (self.values.shape[0],)

    def divide(self, chunks: int | tuple[int, ...] = 1, lazy: bool = True):
        if isinstance(chunks, int):
            chunks = equal_sized_chunks(len(self), chunks=chunks)
        elif isinstance(chunks, tuple):
            assert sum(chunks) == len(self)
        else:
            raise ValueError

        blocks = np.empty(len(chunks), dtype=object)
        for i, (start, stop) in enumerate(
            zip(np.cumsum((0,) + chunks), np.cumsum(chunks))
        ):
            blocks[i] = self.__class__(
                self.values[start:stop].copy(),
                weights=self.weights[start:stop].copy(),
                ensemble_mean=self.ensemble_mean,
            )

        if lazy:
            blocks = da.from_array(blocks, chunks=1)

        return blocks

    def __len__(self) -> int:
        return len(self._values)

    @property
    def ensemble_mean(self) -> bool:
        return self._ensemble_mean

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def combine(self, other: DistributionFromValues) -> MultidimensionalDistribution:
        """
        Combine distribution with another distribution to produce a higher-dimensional distribution.

        Parameters
        ----------
        other : DistributionFromValues
            The distribution to combine this distribution with.

        Returns
        -------
        combined_distribution : MultidimensionalDistribution
            Higher-dimensional combined distribution.
        """
        return MultidimensionalDistribution([self, other])


class MultidimensionalDistribution(BaseDistribution):
    """
    A multidimensional distribution composed of multiple lower-dimensional distributions.

    Parameters
    ----------
    distributions : list of BaseDistribution
        The lower-dimensional distributions composed into a higher-dimensional distribution.
    """

    def __init__(self, distributions: list[BaseDistribution]):
        for distribution in distributions:
            assert distribution.dimensions == 1

        self._distributions = distributions

    @property
    def distributions(self):
        """The lower dimensional distributions making up this distribution."""
        return self._distributions

    def _apply_to_distributions(self, method_name):
        return self.__class__(
            [
                getattr(distribution, method_name)()
                for distribution in self.distributions
            ]
        )

    def __neg__(self):
        return self._apply_to_distributions("__neg__")

    def divide(self, chunks: Chunks, lazy: bool = True):
        if self.dimensions == 1:
            return self._distributions[0].divide(chunks, lazy)

        raise NotImplementedError

    @property
    def shape(self):
        return tuple(
            map(sum, tuple(distribution.shape for distribution in self._distributions))
        )

    @property
    def dimensions(self):
        return len(self._distributions)

    @property
    def values(self):
        if self.dimensions == 1:
            return self._distributions[0].values
        values = [distribution.values for distribution in self._distributions]
        xp = get_array_module(values[0])
        return xp.stack(xp.meshgrid(*values, indexing="ij"), axis=-1)

    @property
    def ensemble_mean(self):
        return tuple(distribution.ensemble_mean for distribution in self._distributions)

    @property
    def weights(self):
        if self.dimensions == 1:
            return self._distributions[0].weights

        xp = get_array_module(self._distributions[0].weights)

        weights = xp.outer(
            self._distributions[0].weights, self._distributions[1].weights
        )
        for i in range(2, len(self._distributions)):
            weights = xp.outer(weights, self._distributions[i].weights)

        return weights


def from_values(
    values: Sequence[Number], weights: np.ndarray = None, ensemble_mean: bool = False
) -> DistributionFromValues:
    """
    Return a distribution from user-defined values and weights.

    Parameters
    ----------
    values : sequence of int or float
        The scalar values of the parameters.
    weights : sequence of float, optional
        The scalar values of the weights (default is None).
    ensemble_mean : bool, optional
        If True, the mean of an ensemble of measurements defined by the distribution is calculated, otherwise the full
        ensemble is kept.
    """
    if weights is None:
        weights = np.ones(len(values))
    values = np.array(values)
    return DistributionFromValues(
        values=values, weights=weights, ensemble_mean=ensemble_mean
    )


def uniform(
    low: float,
    high: float,
    num_samples: int,
    endpoint: bool = True,
    ensemble_mean: bool = False,
) -> DistributionFromValues:
    """
    Return a distribution with uniformly weighted values evenly spaced over a specified interval.
    As an example, this distribution may be used for simulating a focal series.

    Parameters
    ----------
    low : float
        The lowest value of the distribution.
    high : float
        The highest value of the distribution. If endpoint is set to False, the sequence consists of
        all but the last of `num_samples + 1` evenly spaced samples so that the high value is excluded.
    num_samples : int
        Number of samples in the distribution.
    endpoint : bool

    ensemble_mean : bool, optional
        If True, the mean of an ensemble of measurements defined by the distribution is calculated, otherwise the full
        ensemble is kept.
    """

    values = np.linspace(start=low, stop=high, num=num_samples, endpoint=endpoint)
    weights = np.ones(len(values))
    values = np.array(values)
    return DistributionFromValues(
        values=values, weights=weights, ensemble_mean=ensemble_mean
    )


def gaussian(
    standard_deviation: float | tuple[float, ...],
    num_samples: int | tuple[int, ...],
    dimension: int = 1,
    center: float | tuple[float, ...] = 0.0,
    ensemble_mean: bool | tuple[bool, ...] = True,
    sampling_limit: float | tuple[float, ...] = 3.0,
    normalize: str = "intensity",
) -> MultidimensionalDistribution:
    """
    Return a distribution with values weighted according to a (multidimensional) Gaussian distribution.
    The values are evenly spaced within a given truncation of the Gaussian distribution. As an example, this
    distribution may be used for simulating focal spread.

    Parameters
    ----------
    standard_deviation : float or tuple of float
        The standard deviation of the distribution. The standard deviations may be given for each axis as a tuple,
        or as a single number, in which case it is equal for all axes.
    num_samples : int
        Number of samples uniformly spaced samples. The samples may be given for each axis as a tuple, or as a
        single number, in which case it is equal for all axes.
    center : float or tuple of float
        The center of the Gaussian distribution (default is 0.0). The center may be given for each axis as a tuple, or
        as a single number, in which case it is equal for all axes.
    dimension : int, optional
        Number of dimensions of the Gaussian distribution.
    ensemble_mean : bool, optional
        If True, the mean of ensemble of measurements defined by the distribution is calculated, otherwise the full
        ensemble is kept. Default is True.
    sampling_limit : float, optional
        Truncate the distribution at this many standard deviations (default is 3.0).
    normalize : str, optional
        Specifies whether to normalize the 'intensity' (default) or 'amplitude'.
    """
    if np.isscalar(center):
        center = (center,) * dimension

    if np.isscalar(standard_deviation):
        standard_deviation = (standard_deviation,) * dimension

    if np.isscalar(num_samples):
        num_samples = (num_samples,) * dimension

    if np.isscalar(ensemble_mean):
        ensemble_mean = (ensemble_mean,) * dimension

    if np.isscalar(sampling_limit):
        sampling_limit = (sampling_limit,) * dimension

    distributions = []
    for i in range(dimension):
        values = np.linspace(
            -standard_deviation[i] * sampling_limit[i] + center[i],
            standard_deviation[i] * sampling_limit[i] + center[i],
            num_samples[i],
        )

        weights = np.exp(-0.5 * (values - center[i]) ** 2 / standard_deviation[i] ** 2)

        if normalize == "intensity":
            weights /= np.sqrt((weights**2).sum())
        elif normalize == "amplitude":
            weights /= weights.sum()
        else:
            raise RuntimeError()

        distributions.append(
            DistributionFromValues(
                values=values, weights=weights, ensemble_mean=ensemble_mean[i]
            )
        )

    return MultidimensionalDistribution(distributions=distributions)


def _validate_distribution(
    distribution: BaseDistribution | Iterable | Number,
) -> BaseDistribution | Number:
    """
    Parameters
    ----------
    distribution : BaseDistribution or Iterable or Number
        The input distribution to be validated.

    Returns
    -------
    BaseDistribution or Number
        The validated distribution. If the input distribution is already a
        valid distribution, it is returned as is. If the input distribution is
        a single number, it is returned unchanged. If the input distribution is
        an ndarray with shape (0,), its single element is returned. If the input
        distribution is a tuple, list, or ndarray, it is converted to an ndarray
        and wrapped into a DistributionFromValues object where each value has
        equal weight. Otherwise, a ValueError is raised.

    Raises
    ------
    ValueError
        If the input distribution is not a valid distribution or .
    """
    if isinstance(distribution, (BaseDistribution, Number, str)):
        return distribution

    if isinstance(distribution, np.ndarray) and len(distribution.shape) == 0:
        return distribution.item()

    if isinstance(distribution, (tuple, list, np.ndarray)):
        distribution = np.array(distribution)
        return DistributionFromValues(
            distribution, np.ones_like(distribution, dtype=np.float32)
        )

    raise ValueError(f"value {distribution} is not a single number or could not be converted to a valid distribution")


def _unpack_distributions(
    *args: float | BaseDistribution, shape: tuple, xp: ArrayModule = np
):
    if len(args) == 0:
        return (), 1.0

    num_new_axes = sum(len(arg.shape) for arg in args if hasattr(arg, "shape"))

    unpacked = ()
    weights = 1.0
    i = 0
    for arg in args:
        if not isinstance(arg, BaseDistribution):
            unpacked += (arg,)
            continue

        axis = list(range(num_new_axes))
        del axis[i]
        i += 1

        axis = tuple(axis) + tuple(range(num_new_axes, num_new_axes + len(shape)))
        values = xp.asarray(np.expand_dims(arg.values, axis=axis), dtype=xp.float32)
        unpacked += (values,)

        new_weights = xp.asarray(
            np.expand_dims(arg.weights, axis=axis), dtype=xp.float32
        )
        weights = new_weights if weights is None else weights * new_weights

    return unpacked, weights


class EnsembleFromDistributions(Ensemble, EqualityMixin, CopyMixin):
    """
    Base object for ensembles based on distributions.

    Parameters
    ----------
    distributions : tuple of str, optional
        Names of properties that may be described by a distribution.
    """

    def __init__(self, distributions: tuple[str, ...] = (), **kwargs):
        self._distributions = distributions
        super().__init__(**kwargs)

    @property
    def _num_ensemble_axes(self):
        return sum(
            len(distribution.shape)
            for distribution in self._distribution_properties.values()
        )

    @property
    def _distribution_properties(self):
        ensemble_parameters = {}
        for parameter in self._distributions:
            value = getattr(self, parameter)
            if hasattr(value, "values"):
                ensemble_parameters[parameter] = value
        return ensemble_parameters

    @property
    def ensemble_shape(self):
        return tuple(
            map(
                sum,
                tuple(
                    distribution.shape
                    for distribution in self._distribution_properties.values()
                ),
            )
        )

    def _partition_args(self, chunks: int = 1, lazy: bool = True):
        distributions = self._distribution_properties
        chunks = self._validate_ensemble_chunks(chunks)
        blocks = ()
        for distribution, n in zip(distributions.values(), chunks):
            blocks += (distribution.divide(n, lazy=lazy),)

        return blocks

    @classmethod
    def _partial_transform(cls, *args, keys, **kwargs):
        assert len(args) == len(keys)

        args = unpack_blockwise_args(args)
        kwargs = {**kwargs, **{key: arg for key, arg in zip(keys, args)}}

        new_transform = cls(**kwargs)
        new_transform = _wrap_with_array(new_transform, len(keys))

        return new_transform

    def _from_partitioned_args(self):
        keys = tuple(self._distribution_properties.keys())
        kwargs = self._copy_kwargs()
        return partial(self._partial_transform, keys=keys, **kwargs)

    @property
    def _default_ensemble_chunks(self):
        return ("auto",) * len(self.ensemble_shape)
