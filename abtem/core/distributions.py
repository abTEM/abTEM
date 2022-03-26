from abc import abstractmethod
from numbers import Number
from typing import Sequence

import numpy as np

from abtem.core.utils import generate_chunks


class Distribution:

    def __init__(self,
                 values: Sequence[Number],
                 weights: Sequence[Number],
                 ensemble_mean: bool = True,
                 chunks: int = 1):
        self._values = np.array(values)
        self._weights = np.array(weights)
        self._ensemble_mean = ensemble_mean
        self._chunks = chunks

    @abstractmethod
    def divide(self):
        pass

    @property
    def chunks(self):
        return self._chunks
        #return tuple(stop - start for start, stop in generate_chunks(len(self), chunks=self._chunks))

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


class GaussianDistribution:

    def __init__(self, center, sigma, num_samples, sampling_limit=4):
        self.center = center
        self.sigma = sigma
        self.sampling_limit = sampling_limit
        self.num_samples = num_samples

    def __iter__(self):
        samples = np.linspace(-self.sigma * self.sampling_limit, self.sigma * self.sampling_limit, self.num_samples)
        values = 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(-.5 * samples ** 2 / self.sigma ** 2)
        values /= values.sum()
        for sample, value in zip(samples, values):
            yield sample + self.center, value


class ParameterSeries(Distribution):

    def __init__(self, values: Sequence[Number], ensemble_mean: bool = False, chunks: int = 1):
        weights = np.ones(len(values))
        super().__init__(values=values, weights=weights, ensemble_mean=ensemble_mean, chunks=chunks)

    def __neg__(self):
        return self.__class__(-self.values, ensemble_mean=self.ensemble_mean, chunks=self._chunks)

    def divide(self):
        chunks = [self.values[start:stop] for start, stop in generate_chunks(len(self), chunks=self._chunks)]
        return [self.__class__(chunk, ensemble_mean=self.ensemble_mean, chunks=1) for chunk in chunks]
