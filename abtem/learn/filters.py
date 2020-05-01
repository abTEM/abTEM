import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableFilter(nn.Module):

    def __init__(self, kernel):
        super().__init__()
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        x = F.pad(x, list((len(self.kernel) // 2,) * 4))
        return F.conv2d(F.conv2d(x, self.kernel.reshape((1, 1, 1, -1))), self.kernel.reshape((1, 1, -1, 1)))


class GaussianFilter2d(SeparableFilter):
    def __init__(self, sigma):
        kernel = self.get_kernel(sigma)
        super().__init__(kernel)

    def get_kernel(self, sigma):
        kernel_size = int(np.ceil(sigma) ** 2) * 4 + 1
        A = 1 / (sigma * np.sqrt(2 * np.pi))
        return A * torch.exp(-(torch.arange(kernel_size) - (kernel_size - 1) / 2) ** 2 / (2 * sigma ** 2))

    def set_sigma(self, sigma):
        new_kernel = self.get_kernel(sigma)
        if self.kernel.is_cuda:
            new_kernel = new_kernel.to(self.kernel.get_device())
        self.kernel = new_kernel


class SumFilter2d(SeparableFilter):
    def __init__(self, kernel_size):
        kernel = torch.ones(kernel_size)
        super().__init__(kernel)


class PeakEnhancementFilter:

    def __init__(self, alpha, sigma, iterations, epsilon=1e-7):
        self._base_filter = GaussianFilter2d(sigma)
        self._alpha = alpha
        self._iterations = iterations
        self._epsilon = epsilon

    def iterate(self, tensor):
        temp = tensor ** self._alpha
        return temp * self._base_filter(tensor) / (self._base_filter(temp) + self._epsilon)

    def to(self, device):
        self._base_filter.to(device)
        return self

    def __call__(self, tensor):
        for i in range(self._iterations):
            tensor = self.iterate(tensor)
        return tensor


class GaussianEnhancementFilter:

    def __init__(self, sigma, alpha, enhancement_sigma, iterations):
        self._peak_enhancement_filter = PeakEnhancementFilter(alpha, enhancement_sigma, iterations)
        print(sigma * np.sqrt(1 - 1 / alpha ** iterations))
        self._gaussian_filter = GaussianFilter2d(sigma * np.sqrt(1 - 1 / alpha ** iterations))

    def to(self, device):
        self._peak_enhancement_filter.to(device)
        self._gaussian_filter.to(device)
        return self

    def __call__(self, tensor):
        markers = self._peak_enhancement_filter(tensor)

        return self._gaussian_filter()
