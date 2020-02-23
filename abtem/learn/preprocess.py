import numpy as np
import torch
import torch.nn.functional as F


def closest_multiple_ceil(n, m):
    return int(np.ceil(n / m) * m)


def pad_to_size(images, height, width, n=None):
    if n is not None:
        height = closest_multiple_ceil(height, n)
        width = closest_multiple_ceil(width, n)

    shape = images.shape[-2:]

    up = (height - shape[0]) // 2
    down = height - shape[0] - up
    left = (width - shape[1]) // 2
    right = width - shape[1] - left
    images = F.pad(images, pad=[up, down, left, right])
    return images

#def add_margin(images):







def normalize_global(images):
    return (images - torch.mean(images, dim=(-2, -1), keepdim=True)) / torch.std(images, dim=(-2, -1), keepdim=True)


def weighted_normalization(images, mask=None):
    if mask is None:
        return normalize_global(images)

    weighted_means = torch.sum(images * mask, dim=(-1, -2), keepdim=True) / torch.sum(mask, dim=(-1, -2), keepdim=True)
    weighted_stds = torch.sqrt(
        torch.sum(mask * (images - weighted_means) ** 2, dim=(-1, -2), keepdim=True) /
        torch.sum(mask, dim=(-1, -2), keepdim=True))
    return (images - weighted_means) / weighted_stds
