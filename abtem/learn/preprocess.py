import numpy as np


def closest_multiple_ceil(n, m):
    return int(np.ceil(n / m) * m)


def pad_to_size(image, height, width):
    up = (height - image.shape[0]) // 2
    down = height - image.shape[0] - up
    left = (width - image.shape[1]) // 2
    right = width - image.shape[1] - left
    image = np.pad(image, pad_width=((up, down), (left, right)), mode='constant', constant_values=((0., 0.), (0., 0.)))
    return image


def normalize_global(image):
    return (image - np.mean(image)) / np.std(image)
