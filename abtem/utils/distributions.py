import numpy as np


class GaussianDistribution:

    def __init__(self, center, sigma, num_samples, sampling_limit=4):
        self.center = center
        self.sigma = sigma
        self.sampling_limit = sampling_limit
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        samples = np.linspace(-self.sigma * self.sampling_limit, self.sigma * self.sampling_limit, self.num_samples)
        values = 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(-.5 * samples ** 2 / self.sigma ** 2)
        values /= values.sum()
        for sample, value in zip(samples, values):
            yield sample + self.center, value
