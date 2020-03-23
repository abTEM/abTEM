from abtem.utils import BatchGenerator
import numpy as np
import cupy as cp
from torch.utils import dlpack
import os


def walk_dir(path, ending):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file[-len(ending):] == ending:
                files.append(os.path.join(r, file))

    return files


class DataGenerator:

    def __init__(self, data, batch_size=8, preprocessing=None, preprocessing_kwargs=None, shuffle=True):
        self._data = data
        self._preprocessing = preprocessing

        if preprocessing_kwargs is None:
            preprocessing_kwargs = {}

        self._preprocessing_kwargs = preprocessing_kwargs
        self.shuffle = shuffle

        self._indices = np.arange(len(self._data))
        self._batch_size = batch_size

        self._batch_generator = BatchGenerator(len(data), max_batch_size=batch_size)

    def __len__(self):
        return len(self._data)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_examples(self):
        return len(self._data)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self._indices)

        for start, size in self._batch_generator.generate():
            batch = [self._data[i] for i in self._indices[start: start + size]]
            if self._preprocessing is not None:
                batch = self._preprocessing(batch, **self._preprocessing_kwargs)
            yield batch


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
