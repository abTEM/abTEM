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
        kernel_size = int(np.ceil(sigma) ** 2) * 2 + 1
        A = 1 / (sigma * np.sqrt(2 * np.pi))
        kernel = A * torch.exp(-(torch.arange(kernel_size) - (kernel_size - 1) / 2) ** 2 / (2 * sigma ** 2))
        super().__init__(kernel)


class SumFilter2d(SeparableFilter):
    def __init__(self, kernel_size):
        kernel = torch.ones(kernel_size)
        super().__init__(kernel)


class PeakEnhancementFilter(nn.Module):

    def __init__(self, alpha, sigmas, iterations, epsilon=1e-7):
        super().__init__()
        self._filters = nn.ModuleList([GaussianFilter2d(sigma) for sigma in sigmas])
        self._alpha = alpha
        self._iterations = iterations
        self._epsilon = epsilon

    def forward(self, tensor):
        temp = tensor.clone()
        for i in range(self._iterations):
            temp = temp ** self._alpha
            for filt in self._filters:
                temp = temp * filt(tensor) / (filt(temp) + self._epsilon)
        return temp


class GaussianEnhancementFilter(nn.Module):

    def __init__(self, sigma, alpha, enhancement_sigmas, iterations):
        super().__init__()
        self._peak_enhancement_filter = PeakEnhancementFilter(alpha, enhancement_sigmas, iterations)
        self._gaussian_filter = GaussianFilter2d(sigma * np.sqrt(1 - 1 / alpha ** (iterations)))

    def forward(self, tensor, return_markers=False):
        markers = self._peak_enhancement_filter(tensor)
        if return_markers:
            return self._gaussian_filter(markers), markers
        else:
            return self._gaussian_filter(markers)
