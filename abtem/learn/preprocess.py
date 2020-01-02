import numpy as np


def closest_multiple_ceil(n, m):
    return int(np.ceil(n / m) * m)


def pad_to_size(images, height, width):
    shape = images.shape[-2:]

    up = (height - shape[0]) // 2
    down = height - shape[0] - up
    left = (width - shape[1]) // 2
    right = width - shape[1] - left

    if len(images.shape) == 3:
        images = np.pad(images, pad_width=((0, 0), (up, down), (left, right)), mode='constant',
                        constant_values=((0., 0.), (0., 0.), (0., 0.)))
    else:
        images = np.pad(images, pad_width=((up, down), (left, right)), mode='constant',
                        constant_values=((0., 0.), (0., 0.)))

    return images


def normalize_global(images):
    return (images - np.mean(images, axis=(-2, -1), keepdims=True)) / np.std(images, axis=(-2, -1), keepdims=True)
