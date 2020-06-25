import bisect
import functools
import os

import cupy as cp
import numpy as np
import torch
from abtem.cuda_kernels import superpose_gaussians
from abtem.utils import BatchGenerator
from torch.utils import dlpack
import torch.nn.functional as F

def walk_dir(path, ending):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file[-len(ending):] == ending:
                files.append(os.path.join(r, file))

    return files


class Example:

    def __init__(self, identifier, image, positions, labels, num_classes, augment=True):
        self.identifier = identifier
        self.image = image
        self.positions = positions
        self.labels = labels
        self.num_classes = num_classes
        self.augment = augment

    @property
    def shape(self):
        return self.image.shape

    def get_density(self, sigma):
        array = cupy_to_pytorch(superpose_gaussians(self.positions, self.shape, sigma))
        return torch.clamp(array, 0, 1.05)[None, None]

    @functools.lru_cache(1)
    def get_segmentation(self, sigma):
        array = cp.zeros((self.num_classes,) + self.shape, dtype=cp.float32)
        for label in range(array.shape[0]):
            positions = self.positions[self.labels == label]
            if len(positions) > 0:
                array[label] = superpose_gaussians(positions, tuple(array.shape[1:]), sigma)

        array /= array.sum(axis=0, keepdims=True) + 1e-3
        array = cupy_to_pytorch(array)
        return array[None]

    def get_image(self):
        return torch.from_numpy(self.image)[None, None].to('cuda:0')

    def get_class_weights(self, class_weights, sigma):
        array = torch.zeros((1, 1,) + self.shape, device='cuda:0')
        for i, value in enumerate(class_weights):
            array[0, 0] += self.get_segmentation(sigma)[0, i] * value
        return array

    def copy(self):
        self.get_segmentation.cache_clear()
        return self.__class__(identifier=self.identifier, image=self.image.copy(), positions=self.positions.copy(),
                              labels=self.labels.copy(), num_classes=self.num_classes, augment=self.augment)


class DataGenerator:

    def __init__(self, data, batch_size=8, augmentation=None, preprocessing_func=None, preprocessing_kwargs=None,
                 shuffle=True):

        self._data = data
        self._preprocessing_func = preprocessing_func

        if preprocessing_kwargs is None:
            preprocessing_kwargs = {}

        self._preprocessing_kwargs = preprocessing_kwargs
        self.shuffle = shuffle
        self._batch_size = batch_size
        self._augmentation = augmentation
        self._epoch_data = self._data

    def __len__(self):
        return len(self._data)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_examples(self):
        return len(self._data)

    def reset_indices(self):
        self._indices = np.arange(len(self), dtype=np.int)
        if self.shuffle:
            np.random.shuffle(self._indices)

    def reset(self):
        self.reset_indices()

    def __iter__(self):
        self.reset()

        data = self._epoch_data
        batch_generator = BatchGenerator(len(self._indices), max_batch_size=self._batch_size)

        for start, size in batch_generator.generate():
            batch = []

            batch_indices = self._indices[start: start + size]

            for i in batch_indices:
                batch.append(self._augmentation(data[i]))

            yield batch


class DataGeneratorWithHardExamples(DataGenerator):

    def __init__(self, data, batch_size, augmentation=None, preprocessing_func=None, preprocessing_kwargs=None,
                 num_hard_examples=0, max_repeat_hard_examples=1, shuffle=True):
        self._num_hard_examples = num_hard_examples
        self._max_repeat_hard_examples = max_repeat_hard_examples
        self._hard_examples = []

        super().__init__(data, batch_size=batch_size, augmentation=augmentation, preprocessing_func=preprocessing_func,
                         preprocessing_kwargs=preprocessing_kwargs, shuffle=shuffle)

    @property
    def hard_examples(self):
        return self._hard_examples

    def reset(self):
        self._epoch_data = self._data.copy()
        for example in self._hard_examples:
            example = example[2].copy()
            example.augment = False
            self._epoch_data += [example]

        self._hard_examples = []
        self._k = 0

        super().reset()

    def collect_hard_example(self, metrics, examples, extra_data=None):
        for i, (metric, example) in enumerate(zip(metrics, examples)):

            if len(self._hard_examples) < self._num_hard_examples:
                max_hard_metric = np.inf
            else:
                max_hard_metric = self._hard_examples[-1][0]

            if metric < max_hard_metric:
                if len(self._hard_examples) == self._num_hard_examples:
                    self._hard_examples.pop(-1)

                if extra_data:
                    bisect.insort(self._hard_examples, (metric, self._k, example, extra_data[i]))
                else:
                    bisect.insort(self._hard_examples, (metric, self._k, example))

                self._k += 1


def cupy_to_pytorch(array):
    return dlpack.from_dlpack(array.toDlpack())


def pytorch_to_cupy(tensor):
    return cp.fromDlpack(dlpack.to_dlpack(tensor))


def concatenate_mask(x, n):
    mask = cp.zeros((x.shape[0],) + (1,) + x.shape[2:], dtype=cp.float32)
    mask[:, :, :n] = 1.
    mask[:, :, -n:] = 1.
    mask[:, :, :, :n] = 1.
    mask[:, :, :, -n:] = 1.
    means = x.mean((2, 3), keepdims=True)
    x[:, :, :n] = means
    x[:, :, -n:] = means
    x[:, :, :, :n] = means
    x[:, :, :, -n:] = means
    return cp.concatenate((x, mask), axis=1)


def closest_multiple_ceil(n, m):
    return int(np.ceil(n / m) * m)


def pad_to_size(images, height, width, n=16, **kwargs):
    xp = cp.get_array_module(images)
    shape = images.shape[-2:]

    if n is not None:
        height = closest_multiple_ceil(height, n)
        width = closest_multiple_ceil(width, n)

    up = max((height - shape[0]) // 2, 0)
    down = max(height - shape[0] - up, 0)
    left = max((width - shape[1]) // 2, 0)
    right = max(width - shape[1] - left, 0)

    #padding = [up, down, left, right]
    images = xp.pad(images, pad_width=[(up, down), (left, right)], **kwargs)
    return images, [up, down, left, right]
    #return F.pad(images, padding, mode=mode), padding

# def pad_to_size(images, height, width, n=None, **kwargs):
#     xp = cp.get_array_module(images)
#
#     if n is not None:
#         height = closest_multiple_ceil(height, n)
#         width = closest_multiple_ceil(width, n)
#
#     shape = images.shape[-2:]
#
#     up = max((height - shape[0]) // 2, 0)
#     down = max(height - shape[0] - up, 0)
#     left = max((width - shape[1]) // 2, 0)
#     right = max(width - shape[1] - left, 0)
#     images = xp.pad(images, pad_width=[(up, down), (left, right)], **kwargs)
#
#     return images, [up, down, left, right]


def weighted_normalization(image, mask=None):
    if mask is None:
        return (image - torch.mean(image)) / torch.std(image)

    weighted_means = (torch.sum(image * mask, dim=(1, 2, 3), keepdims=True) /
                      torch.sum(mask, dim=(1, 2, 3), keepdims=True))
    weighted_stds = torch.sqrt(
        torch.sum(mask * (image - weighted_means) ** 2, dim=(1, 2, 3), keepdims=True) /
        torch.sum(mask ** 2, dim=(1, 2, 3), keepdims=True))
    return (image - weighted_means) / weighted_stds


class BatchGenerator:

    def __init__(self, n_items, max_batch_size):
        self._n_items = n_items
        self._n_batches = (n_items + (-n_items % max_batch_size)) // max_batch_size
        self._batch_size = (n_items + (-n_items % self.n_batches)) // self.n_batches

    @property
    def n_batches(self):
        return self._n_batches

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def n_items(self):
        return self._n_items

    def generate(self):
        batch_start = 0
        for i in range(self.n_batches):
            batch_end = batch_start + self.batch_size
            if i == self.n_batches - 1:
                yield batch_start, self.n_items - batch_end + self.batch_size
            else:
                yield batch_start, self.batch_size

            batch_start = batch_end