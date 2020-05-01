import bisect
import functools
import random
import os

import cupy as cp
import numpy as np
import torch
from abtem.cudakernels import superpose_gaussians
from abtem.learn.utils import cupy_to_pytorch
from abtem.utils import BatchGenerator
from abtem.learn.augment import SequentialAugmentations

class Example:

    def __init__(self, image, sampling, positions, labels):
        self.sampling = sampling
        self.image = image
        self.positions = positions
        self.labels = labels

    @property
    def shape(self):
        return self.image.shape

    def get_density(self, sigma):
        array = cupy_to_pytorch(superpose_gaussians(self.positions, self.shape, sigma))
        return torch.clamp(array, 0, 1.05)[None, None]

    @functools.lru_cache(1)
    def get_segmentation(self, sigma, num_classes):
        array = cp.zeros((num_classes,) + self.shape, dtype=cp.float32)
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
            array[0, 0] += self.get_segmentation(sigma, len(class_weights))[0, i] * value
        return array

    def write(self, path):
        np.savez(path, image=self.image, positions=self.positions, labels=self.labels)

    @classmethod
    def read(cls, path):
        with np.load(path) as f:
            return cls(image=f['image'], sampling=f['sampling'], positions=f['postions'], labels=f['labels'])

    def copy(self):
        self.get_segmentation.cache_clear()
        return self.__class__(image=self.image.copy(), sampling=self.sampling, positions=self.positions.copy(),
                              labels=self.labels.copy())


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

            with np.load(os.path.join(base_path, filename)) as f:
                dataset.append(
                    Example(image=f['image'], sampling=f['sampling'], positions=f['positions'], labels=f['labels']))
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
                example = self._examples[i]
                if self._augmentations:
                    self._augmentations(example.copy())
                batch.append(example)

            yield batch


class DatasetWithHardExamples(Dataset):

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
