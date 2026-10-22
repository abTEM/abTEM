"""Module for describing distributions of simulation parameters."""

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Number
from typing import Callable, Iterator, Optional, Sequence, SupportsFloat, overload

import dask.array as da
import numpy as np

from abtem.core.backend import ArrayModule, get_array_module
from abtem.core.chunks import Chunks, equal_sized_chunks, is_tuple_of_ints
from abtem.core.ensemble import Ensemble, _wrap_with_array, unpack_blockwise_args
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype, number_to_tuple


class BaseDistribution(EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """
    Base object for defining distributions of simulation parameters.
    """

    @abstractmethod
    def __neg__(self) -> BaseDistribution:
        """Return the negated distribution."""

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self) -> np.ndarray:
        return self.values

    def __iter__(self) -> Iterator[float]:
        return iter(self.values)

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """The number of dimensions in the distribution."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the distribution parameters."""

    @abstractmethod
    def divide(
        self, chunks: int | tuple[int, ...] = 1, lazy: bool = True
    ) -> np.ndarray | da.Array:
        """Divide the distribution into chunks."""

    @property
    @abstractmethod
    def ensemble_mean(self) -> bool:
        """Calculate the mean of the ensemble."""

    @property
    @abstractmethod
    def values(self) -> np.ndarray:
        """Scalar values representing the distribution."""

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        """Weight of each of distribution value."""


class DistributionFromValues(BaseDistribution):
    """
    Distribution defined by user-defined values and weights.

    Parameters
    ----------
    values : np.ndarray
        The values of the distribution.
    weights : np.ndarray, optional
        The values of the weights. If None, all weights are set to 1.
    ensemble_mean : bool, optional
        If True, the mean of an ensemble of measurements defined by the distribution is
        calculated, otherwise the full ensemble is kept.
    """

    def __init__(
        self,
        values: np.ndarray,
        weights: np.ndarray | None = None,
        ensemble_mean: bool = False,
    ):
        self._values = np.array(values)

        if weights is None:
            weights = np.ones(len(self._values))

        weights = np.asarray(weights)

        if weights.shape != (len(self._values),):
            raise ValueError(
                f"expected one weight per value ({len(self._values)}), got weights of "
                f"shape {weights.shape}"
            )

        if np.any(weights < 0):
            raise ValueError("distribution weights are probabilities: must be >= 0")

        # No all-zero check here: the blocks built by ``divide`` (and their
        # negations) may hold only zero-weight members of a valid distribution.
        # ``from_values`` rejects an all-zero distribution. An ensemble axis
        # whose weights are all zero (e.g. a slice of zero-weight members) has
        # equal weights and is reduced with the plain mean.
        self._weights = weights

        self._ensemble_mean = ensemble_mean

    def __neg__(self) -> DistributionFromValues:
        return self.__class__(
            values=-self.values, weights=self.weights, ensemble_mean=self.ensemble_mean
        )

    @property
    def dimensions(self) -> int:
        return 1

    @property
    def shape(self) -> tuple[int]:
        return (self.values.shape[0],)

    def divide(
        self, chunks: int | tuple[int, ...] = 1, lazy: bool = True
    ) -> np.ndarray | da.Array:
        if isinstance(chunks, int):
            chunks = equal_sized_chunks(len(self), num_chunks=chunks)
        elif is_tuple_of_ints(chunks):
            assert sum(chunks) == len(self)
        else:
            raise ValueError("chunks must be an int or a tuple of ints")

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
        Combine distribution with another distribution to produce a higher-dimensional
        distribution.

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
    A multidimensional distribution composed of multiple lower-dimensional
    distributions.

    Parameters
    ----------
    distributions : list of BaseDistribution
        The lower-dimensional distributions composed into a higher-dimensional
        distribution.
    """

    def __init__(self, distributions: Sequence[BaseDistribution]):
        for distribution in distributions:
            assert distribution.dimensions == 1

        self._distributions = distributions

    @property
    def distributions(self):
        """The lower dimensional distributions making up this distribution."""
        return self._distributions

    def _apply_to_distributions(self, method: str) -> MultidimensionalDistribution:
        return self.__class__(
            [getattr(distribution, method)() for distribution in self.distributions]
        )

    def __neg__(self) -> MultidimensionalDistribution:
        return self._apply_to_distributions("__neg__")

    def divide(
        self, chunks: int | tuple[int, ...] = 1, lazy: bool = True
    ) -> np.ndarray | da.Array:
        if self.dimensions == 1:
            return self._distributions[0].divide(chunks, lazy)
        else:
            raise NotImplementedError(
                "Dividing multidimensional distributions is not supported."
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            map(sum, tuple(distribution.shape for distribution in self._distributions))
        )

    @property
    def dimensions(self) -> int:
        return len(self._distributions)

    @property
    def values(self) -> np.ndarray:
        if self.dimensions == 1:
            return self._distributions[0].values
        values = [distribution.values for distribution in self._distributions]
        xp = get_array_module(values[0])
        return xp.stack(xp.meshgrid(*values, indexing="ij"), axis=-1)

    @property
    def ensemble_mean(self) -> bool:
        ensemble_means = tuple(
            distribution.ensemble_mean for distribution in self._distributions
        )
        assert all(
            ensemble_mean == ensemble_means[0] for ensemble_mean in ensemble_means
        )
        return ensemble_means[0]

    @property
    def weights(self) -> np.ndarray:
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
    values: Sequence[SupportsFloat] | np.ndarray,
    weights: np.ndarray | None = None,
    ensemble_mean: bool = False,
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
        If True, the mean of an ensemble of measurements defined by the distribution is
        calculated, otherwise the full ensemble is kept.
    """
    if weights is None:
        weights = np.ones(len(values))
    values_array = np.array(values)
    distribution = DistributionFromValues(
        values=values_array, weights=weights, ensemble_mean=ensemble_mean
    )
    if len(distribution.weights) and not np.sum(distribution.weights) > 0:
        raise ValueError("distribution weights must not all be zero")
    return distribution


def uniform(
    low: float,
    high: float,
    num_samples: int,
    endpoint: bool = True,
    ensemble_mean: bool = False,
) -> DistributionFromValues:
    """
    Return a distribution with uniformly weighted values evenly spaced over a specified
    interval. As an example, this distribution may be used for simulating a focal
    series.

    Parameters
    ----------
    low : float
        The lowest value of the distribution.
    high : float
        The highest value of the distribution. If endpoint is set to False, the sequence
        consists of all but the last of `num_samples + 1` evenly spaced samples so that
        the high value is excluded.
    num_samples : int
        Number of samples in the distribution.
    endpoint : bool

    ensemble_mean : bool, optional
        If True, the mean of an ensemble of measurements defined by the distribution is
        calculated, otherwise the full ensemble is kept.
    """

    values = np.linspace(start=low, stop=high, num=num_samples, endpoint=endpoint)
    weights = np.ones(len(values))
    values = np.array(values)
    return DistributionFromValues(
        values=values, weights=weights, ensemble_mean=ensemble_mean
    )


def _validate_normalize(normalize: str) -> str:
    if normalize in ("intensity", "amplitude"):
        warnings.warn(
            f"normalize='{normalize}' is deprecated and now equivalent to "
            "normalize='probability'. Distribution weights are probabilities "
            "applied when the ensemble is reduced (the weighted mean "
            "Σ p_i I_i / Σ p_i), not factors on the wave amplitude, so the weights "
            "are normalized to sum to one. Use the default normalize='probability'.",
            FutureWarning,
            stacklevel=3,
        )
    elif normalize != "probability":
        raise ValueError(
            f"Unknown normalization method: {normalize!r}; use 'probability'"
        )
    return "probability"


def gaussian(
    standard_deviation: float | tuple[float, ...],
    num_samples: int | tuple[int, ...],
    dimension: int = 1,
    center: float | tuple[float, ...] = 0.0,
    ensemble_mean: bool | tuple[bool, ...] = True,
    sampling_limit: float | tuple[float, ...] = 3.0,
    normalize: str = "probability",
) -> MultidimensionalDistribution:
    """
    Return a distribution with values weighted according to a (multidimensional)
    Gaussian distribution. The values are evenly spaced within a given truncation of the
    Gaussian distribution. As an example, this distribution may be used for simulating
    focal spread.

    Parameters
    ----------
    standard_deviation : float or tuple of float
        The standard deviation of the distribution. The standard deviations may be given
        for each axis as a tuple, or as a single number, in which case it is equal for
        all axes.
    num_samples : int
        Number of samples uniformly spaced samples. The samples may be given for each
        axis as a tuple, or as a single number, in which case it is equal for all axes.
    center : float or tuple of float
        The center of the Gaussian distribution (default is 0.0). The center may be
        given for each axis as a tuple, or as a single number, in which case it is equal
        for all axes.
    dimension : int, optional
        Number of dimensions of the Gaussian distribution.
    ensemble_mean : bool, optional
        If True, the mean of ensemble of measurements defined by the distribution is
        calculated, otherwise the full ensemble is kept. Default is True.
    sampling_limit : float, optional
        Truncate the distribution at this many standard deviations (default is 3.0).
    normalize : str, optional
        Scaling of the stored probability weights: 'probability' (default)
        normalizes them to sum to one. The weights are probabilities applied when
        the ensemble is reduced (Σ p_i I_i / Σ p_i), so the scaling never changes a
        reduced result. 'intensity' and 'amplitude' are deprecated remnants of the
        former convention in which the weights multiplied the wave amplitude; both
        now emit a FutureWarning and behave as 'probability'.

    Notes
    -----
    The weights are the probabilities p_i ∝ exp(-x_i² / 2σ²) of the sampled values:
    each ensemble member is simulated unweighted and the reduced (incoherently
    averaged) measurement is Σ p_i I(x_i) / Σ p_i, so the parameter seen by the
    intensity has standard deviation σ (up to the truncation at ``sampling_limit``).
    For a defocus spread this corresponds to Kirkland's 1/e focal-spread width
    Δ = √2·σ used by :class:`abtem.transfer.TemporalEnvelope`.
    """
    normalize = _validate_normalize(normalize)
    center = number_to_tuple(center, dimension)
    standard_deviation = number_to_tuple(standard_deviation, dimension)
    ensemble_mean = number_to_tuple(ensemble_mean, dimension)
    sampling_limit = number_to_tuple(sampling_limit, dimension)
    num_samples = number_to_tuple(num_samples, dimension)

    distributions: list[BaseDistribution] = []
    for i in range(dimension):
        values = np.linspace(
            -standard_deviation[i] * sampling_limit[i] + center[i],
            standard_deviation[i] * sampling_limit[i] + center[i],
            num_samples[i],
        )

        weights = np.exp(-0.5 * (values - center[i]) ** 2 / standard_deviation[i] ** 2)

        weights /= weights.sum()

        distributions.append(
            DistributionFromValues(
                values=values, weights=weights, ensemble_mean=ensemble_mean[i]
            )
        )

    return MultidimensionalDistribution(distributions=distributions)


@overload
def validate_distribution(
    distribution: BaseDistribution | tuple | list | np.ndarray,
) -> BaseDistribution: ...


@overload
def validate_distribution(distribution: int) -> int: ...


@overload
def validate_distribution(distribution: float) -> float: ...


def validate_distribution(
    distribution: BaseDistribution | tuple | list | np.ndarray | SupportsFloat,
) -> BaseDistribution | float | int:
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

    elif isinstance(distribution, np.ndarray) and len(distribution.shape) == 0:
        return distribution.item()

    elif isinstance(distribution, (tuple, list, np.ndarray)):
        distribution = np.array(distribution)

        # One weight per value (not ``ones_like``: for an (N, 2) array of tilts
        # that would give (N, 2) weights, misaligned with the N values).
        return DistributionFromValues(
            distribution, np.ones(len(distribution), dtype=get_dtype(complex=False))
        )
    else:
        raise ValueError(
            f"value {distribution} is not a single number or could not be converted to",
            "a valid distribution",
        )


def tuple_range_except(n, i):
    return tuple(x for x in range(n) if x != i)


def _unpack_distributions(
    *args: float | BaseDistribution, shape: tuple[int, ...], xp: ArrayModule = np
) -> tuple[float | np.ndarray, ...]:
    """Broadcastable parameter values of the given (distributions of) parameters.

    The distribution weights are not returned: they are probabilities carried by
    the ensemble axis metadata and applied when the ensemble is reduced.
    """
    if len(args) == 0:
        return ()

    xp = get_array_module(xp)
    dtype = get_dtype(complex=False)

    num_new_axes = sum(len(arg.shape) for arg in args if hasattr(arg, "shape"))
    base_axes = tuple(range(num_new_axes, num_new_axes + len(shape)))

    unpacked = []
    i = 0
    for arg in args:
        if not isinstance(arg, BaseDistribution):
            unpacked.append(arg)
        else:
            axis = tuple_range_except(num_new_axes, i) + base_axes
            values = xp.asarray(np.expand_dims(arg.values, axis=axis), dtype=dtype)
            unpacked.append(values)
            i += 1

    return tuple(unpacked)


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
    def _num_ensemble_axes(self) -> int:
        return sum(
            len(distribution.shape)
            for distribution in self._distribution_properties.values()
        )

    @property
    def _distribution_properties(self) -> dict[str, BaseDistribution]:
        ensemble_parameters = {}
        for parameter in self._distributions:
            value = getattr(self, parameter)
            if hasattr(value, "values"):
                ensemble_parameters[parameter] = value
        return ensemble_parameters

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return tuple(
            sum(distribution.shape)
            for distribution in self._distribution_properties.values()
        )

    @property
    def _default_ensemble_chunks(self) -> Chunks:
        return ("auto",) * len(self.ensemble_shape)

    def _partition_args(self, chunks: Optional[Chunks] = 1, lazy: bool = True) -> tuple:
        distributions = self._distribution_properties

        chunks = self._validate_ensemble_chunks(chunks)
        blocks = tuple(
            distribution.divide(n, lazy=lazy)
            for distribution, n in zip(distributions.values(), chunks)
        )
        return blocks

    @classmethod
    def _partial_transform(cls, *args, keys, **kwargs) -> np.ndarray:
        assert len(args) == len(keys)

        args = unpack_blockwise_args(args)
        kwargs = {**kwargs, **{key: arg for key, arg in zip(keys, args)}}

        new_transform = _wrap_with_array(cls(**kwargs), len(keys))

        return new_transform

    def _from_partitioned_args(self) -> Callable[..., np.ndarray]:
        keys = tuple(self._distribution_properties.keys())
        kwargs = self._copy_kwargs()
        return partial(self._partial_transform, keys=keys, **kwargs)
