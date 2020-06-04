import bisect
import functools
import os

import cupy as cp
import numpy as np
import torch

from abtem.cudakernels import superpose_gaussians
from abtem.learn.augment import SequentialAugmentations
from abtem.learn.utils import cupy_to_pytorch
from abtem.utils import BatchGenerator


class Example:

    def __init__(self, image, points, labels, sampling=None, filename=None):
        self.image = image
        self.points = points
        self.labels = labels
        self.sampling = sampling
        self.filename = filename

    @property
    def shape(self):
        return self.image.shape

    def get_density(self, sigma):
        array = cupy_to_pytorch(superpose_gaussians(self.points, self.shape, sigma))
        return torch.clamp(array, 0, 1.05)[None, None]

    @functools.lru_cache(1)
    def get_segmentation(self, sigma, num_classes):
        array = cp.zeros((num_classes,) + self.shape, dtype=cp.float32)
        array[0] = 1e-7
        for label in range(array.shape[0]):
            positions = self.points[self.labels == label]
            if len(positions) > 0:
                array[label + 1] += superpose_gaussians(positions, tuple(array.shape[1:]), sigma)
        array /= array.sum(axis=0, keepdims=True)  # + 1e-5
        array = cupy_to_pytorch(array)
        return array[None]

    def get_image(self):
        return torch.from_numpy(self.image)[None, None].to('cuda:0')

    def get_class_weights(self, class_weights, sigma):
        array = torch.zeros((1, 1,) + self.shape, device='cuda:0')
        for i, value in enumerate(class_weights):
            array[0, 0] += self.get_segmentation(sigma, len(class_weights))[0, i] * value
        return array

    def write(self, path):
        np.savez(path, image=self.image, points=self.points, labels=self.labels, sampling=self.sampling)

    @classmethod
    def read(cls, path):
        with np.load(path) as f:
            filename = os.path.split(path)[-1]
            return cls(image=f['image'], sampling=f['sampling'], points=f['points'], labels=f['labels'],
                       filename=filename)

    def copy(self):
        self.get_segmentation.cache_clear()
        return self.__class__(image=self.image.copy(), sampling=self.sampling, points=self.points.copy(),
                              labels=self.labels.copy(), filename=self.filename)


class Dataset:

    def __init__(self, examples, batch_size=1, augmentations=None, shuffle=True):
        self._examples = examples
        self.shuffle = shuffle
        self._batch_size = batch_size
        self._augmentations = augmentations
        self._epoch_data = self._examples

    def __len__(self):
        return len(self._examples)

    @property
    def examples(self):
        return self._examples

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_examples(self):
        return len(self._examples)

    @property
    def augmentations(self):
        return self._augmentations

    @augmentations.setter
    def augmentations(self, augmentations):
        if isinstance(augmentations, list):
            augmentations = SequentialAugmentations(augmentations)
        self._augmentations = augmentations

    def append(self, example):
        self._examples.append(example)

    def write(self, base_path):
        for i, example in enumerate(self._examples):
            path = base_path + '_{}'.format(str(i).zfill(len(str(len(self._examples)))))
            np.savez(path, image=example.image, sampling=example.sampling, positions=example.positions,
                     labels=example.labels)

    @classmethod
    def read(cls, base_path):
        dataset = cls([])
        for filename in os.listdir(base_path):
            _, file_extension = os.path.splitext(filename)
            if file_extension != '.npz':
                continue
            f = os.path.join(base_path, filename)
            dataset.append(Example.read(f))
        return dataset

    def __iter__(self):
        indices = np.arange(len(self), dtype=np.int)
        if self.shuffle:
            np.random.shuffle(indices)

        batch_generator = BatchGenerator(len(indices), max_batch_size=self._batch_size)

        for start, size in batch_generator.generate():
            batch = []
            batch_indices = indices[start: start + size]

            for i in batch_indices:
                example = self._examples[i].copy()
                if self._augmentations:
                    self._augmentations(example)
                batch.append(example)

            yield batch


class HardExamples:

    def __init__(self, max_examples, limiting_value, order='ascending'):
        self._max_examples = max_examples
        self._limiting_value = limiting_value
        self._examples = []
        self._k = 0

    def __len__(self):
        return len(self._examples)

    def __iter__(self):
        for example in self._examples:
            yield example

    def clear(self):
        self._examples = []
        self._k = 0

    def append(self, metrics, *args):
        for i, metric in enumerate(metrics):

            if len(self._examples) < self._max_examples:
                limiting_value = self._limiting_value
            else:
                limiting_value = self._examples[-1][0]

            if metric < limiting_value:
                if len(self._examples) == self._max_examples:
                    self._examples.pop(-1)

                bisect.insort(self._examples, (metric, self._k) + args)

                self._k += 1
