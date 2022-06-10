from abc import abstractmethod, ABCMeta
from numbers import Number
from typing import Sequence, Union, Tuple

import dask.array as da
import numpy as np

from abtem.core.axes import LinearAxis
from abtem.core.backend import validate_device, get_array_module
from abtem.core.utils import subdivide_into_chunks


class Distribution(metaclass=ABCMeta):

    @property
    @abstractmethod
    def shape(self):
        pass

    @abstractmethod
    def divide(self, chunks):
        pass

    @property
    @abstractmethod
    def ensemble_mean(self):
        pass

    @property
    @abstractmethod
    def values(self):
        pass

    @property
    @abstractmethod
    def weights(self):
        pass


class OneDimensionalDistributionFromValues(Distribution):

    def __init__(self,
                 values: np.ndarray,
                 weights: np.ndarray,
                 ensemble_mean: bool = True):
        self._values = values
        self._weights = weights
        self._ensemble_mean = ensemble_mean

    @property
    def shape(self):
        return self.values.shape

    def divide(self, chunks):
        pass

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(zip(self._values, self._weights))

    @property
    def ensemble_mean(self):
        return self._ensemble_mean

    @property
    def values(self):
        return self._values

    @property
    def weights(self):
        return self._weights


class MultidimensionalAxisAlignedDistribution(Distribution):

    def __init__(self, distributions):
        self._distributions = distributions

    def divide(self, chunks):
        raise NotImplementedError

    @property
    def shape(self):
        return tuple(map(sum, tuple(distribution.shape for distribution in self._distributions)))

    def create_axes_metadata(self, labels):

        if isinstance(labels, str):
            labels = (labels,) * self.dimension

        axes = []
        for i in range(self.dimension):
            axes.append(LinearAxis(label=labels[i], _ensemble_mean=self.ensemble_mean[i]))

        return axes

    @property
    def dimension(self):
        return len(self._distributions)

    @property
    def values(self):
        if self.dimension == 1:
            return self._distributions[0].values
        values = [distribution.values for distribution in self._distributions]
        xp = get_array_module(values[0])
        return xp.stack(xp.meshgrid(*values, indexing='ij'), axis=-1)

    @property
    def ensemble_mean(self):
        return tuple(distribution.ensemble_mean for distribution in self._distributions)

    @property
    def weights(self):
        if self.dimension == 1:
            return self._distributions[0].weights

        xp = get_array_module(self._distributions[0].weights)

        weights = xp.outer(self._distributions[0].weights, self._distributions[1].weights)
        for i in range(2, len(self._distributions)):
            weights = xp.outer(weights, self._distributions[i].weights)

        return weights

    @property
    def factors(self):
        return self._distributions

    @classmethod
    def product(cls, distributions):
        return cls(distributions)


class ParameterSeries(OneDimensionalDistributionFromValues):

    def __init__(self, values: Sequence[Number], ensemble_mean: bool = False):
        weights = np.ones(len(values))
        super().__init__(values=values, weights=weights, ensemble_mean=ensemble_mean)

    def __neg__(self):
        return self.__class__(-self.values, ensemble_mean=self.ensemble_mean)

    def divide(self, chunks: Union[int, Tuple[int, ...]] = 1, lazy: bool = False):
        if isinstance(chunks, int):
            chunks = subdivide_into_chunks(len(self), chunks=chunks)
        elif isinstance(chunks, tuple):
            assert sum(chunks) == len(self)
        else:
            raise ValueError

        blocks = np.empty(len(chunks), dtype=object)
        for i, (start, stop) in enumerate(zip(np.cumsum((0,) + chunks), np.cumsum(chunks))):
            blocks[i] = self.__class__(self.values[start:stop], ensemble_mean=self.ensemble_mean)

        if lazy:
            blocks = da.from_array(blocks, chunks=1)

        return blocks


class OneDimensionalGaussianDistribution(OneDimensionalDistributionFromValues):

    def __init__(self,
                 center: float,
                 standard_deviation: float,
                 num_samples: int,
                 ensemble_mean: bool = True,
                 sampling_limit: float = 3.,
                 normalize: str = 'intensity',
                 device: str = None):

        device = validate_device(device)
        xp = get_array_module(device)
        values = xp.linspace(-standard_deviation * sampling_limit,
                             standard_deviation * sampling_limit, num_samples) + center
        weights = xp.exp(-.5 * (values - center) ** 2 / standard_deviation ** 2)

        if normalize == 'intensity':
            weights /= np.sqrt((weights ** 2).sum())
        elif normalize == 'amplitude':
            weights /= weights.sum()
        else:
            raise ValueError()

        self._center = center
        self._standard_deviation = standard_deviation
        self._num_samples = num_samples
        self._ensemble_mean = ensemble_mean
        self._sampling_limit = sampling_limit
        super().__init__(values=values, weights=weights, ensemble_mean=ensemble_mean)

    @property
    def center(self):
        return self._center

    @property
    def standard_deviation(self):
        return self._standard_deviation

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def ensemble_mean(self):
        return self._ensemble_mean

    @property
    def sampling_limit(self):
        return self._sampling_limit

    def divide(self, chunks, lazy: bool = True):
        blocks = np.empty(len(chunks), dtype=object)
        for i, (start, stop) in enumerate(zip(np.cumsum((0,) + chunks), np.cumsum(chunks))):
            blocks[i] = OneDimensionalDistributionFromValues(self.values[start:stop],
                                                             weights=self.weights[start:stop],
                                                             ensemble_mean=self.ensemble_mean)

        if lazy:
            blocks = da.from_array(blocks, chunks=1)

        return blocks


class GaussianDistribution(Distribution):

    def __init__(self,
                 standard_deviation: Union[float, Tuple[float, ...]],
                 num_samples: Union[int, Tuple[int, ...]],
                 dimension: int,
                 center: Union[float, Tuple[float, ...]] = 0.,
                 ensemble_mean: Union[bool, Tuple[bool, ...]] = True,
                 sampling_limit: Union[float, Tuple[float, ...]] = 3.,
                 normalize: str = 'intensity',
                 device: str = None):

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

        self._distributions = tuple(OneDimensionalGaussianDistribution(center[i],
                                                                       standard_deviation[i],
                                                                       num_samples[i],
                                                                       ensemble_mean[i],
                                                                       sampling_limit=sampling_limit[i],
                                                                       normalize=normalize,
                                                                       device=device)
                                    for i in range(dimension))

        self._dimension = dimension
        super().__init__()

    @property
    def shape(self):
        return tuple(map(sum, tuple(distribution.shape for distribution in self._distributions)))

    def create_axes_metadata(self, labels):

        if isinstance(labels, str):
            labels = (labels,) * self.dimension

        axes = []
        for i in range(self.dimension):
            axes.append(LinearAxis(label=labels[i], _ensemble_mean=self.ensemble_mean[i]))

        return axes

    @property
    def dimension(self):
        return self._dimension

    @property
    def values(self):
        return MultidimensionalAxisAlignedDistribution(self._distributions).values

    @property
    def ensemble_mean(self):
        if self.dimension == 1:
            return self._distributions[0].ensemble_mean

        return tuple(distribution.ensemble_mean for distribution in self._distributions)

    @property
    def weights(self):
        return MultidimensionalAxisAlignedDistribution(self._distributions).weights

    @property
    def factors(self):
        return self._distributions

    @classmethod
    def product(cls, distributions):
        center = tuple(distribution.center for distribution in distributions)
        standard_deviation = tuple(distribution.sigma for distribution in distributions)
        num_samples = tuple(distribution.num_samples for distribution in distributions)
        ensemble_mean = tuple(distribution.ensemble_mean for distribution in distributions)
        sampling_limit = tuple(distribution.sampling_limit for distribution in distributions)
        dimension = len(distributions)
        return cls(center=center, standard_deviation=standard_deviation, num_samples=num_samples,
                   ensemble_mean=ensemble_mean, dimension=dimension, sampling_limit=sampling_limit)

    def divide(self, chunks, lazy: bool = True):
        if self.dimension == 1:
            return self._distributions[0].divide(chunks, lazy)

        raise NotImplementedError
