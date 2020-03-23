import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import convolve


def gaussian_kernel(sigma, normalize=True):
    kernel_size = int(np.ceil(sigma) ** 2) * 4 + 1
    kernel = cp.exp(-(cp.arange(kernel_size) - (kernel_size - 1) / 2) ** 2 / (2 * sigma ** 2))
    if normalize:
        kernel /= kernel.sum()
    return kernel


def separable_filter_2d(image, kernel):
    return convolve(convolve(image, kernel[None]), kernel[:, None])


def average_filter_2d(image, width, normalize=True):
    kernel_size = 1 + 2 * width
    if normalize:
        kernel = cp.full(kernel_size, 1 / kernel_size)
    else:
        kernel = cp.ones(1 + 2 * width)
    return convolve(convolve(image, kernel[None]), kernel[:, None])


def gaussian_filter_2d(image, sigma, normalize=True):
    kernel = gaussian_kernel(sigma, normalize=normalize)
    return convolve(convolve(image, kernel[None]), kernel[:, None])
