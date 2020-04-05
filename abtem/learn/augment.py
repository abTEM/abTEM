import functools

import cupy as cp
import cupyx.scipy.ndimage
import numpy as np

from abtem.learn.filters import gaussian_filter_2d
from abtem.noise import bandpass_noise
from abtem.learn.utils import pad_to_size


class RandomNumberGenerator:

    def __init__(self):
        self._value = None

    def randomize(self):
        raise NotImplementedError()

    @property
    def last_value(self):
        return self._value

    @property
    def new_value(self):
        self.randomize()
        return self._value


class RandomUniform(RandomNumberGenerator):

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__()

    def randomize(self):
        self._value = np.random.uniform(self.min_value, self.max_value)


class RandomNormal(RandomNumberGenerator):

    def __init__(self, mean, std, min_value=-np.inf, max_value=np.inf):
        self.mean = mean
        self.std = std
        self.min_value = min_value
        self.max_value = max_value
        super().__init__()

    def randomize(self):
        self._value = max((np.random.randn() * self.std + self.mean, self.min_value))


class DataModifier:

    def apply(self, image, label):
        raise NotImplementedError()

    def randomize(self, random_number_generator):
        try:
            return random_number_generator.new_value
        except AttributeError:
            return random_number_generator

    def __call__(self, image, label):
        # assert len(images) == len(labels)

        # for i in range(len(images)):
        #     images[i], labels[i] = self.apply(images[i], labels[i])

        return self.apply(image, label)


class SequentialDataModifiers(DataModifier):

    def __init__(self, modifiers):
        super().__init__()
        self.modifiers = modifiers

    def __call__(self, image, label):
        for modifier in self.modifiers:
            image, label = modifier(image, label)

        return image, label


class RandomCrop(DataModifier):

    def __init__(self, new_shape):
        self.new_shape = new_shape
        super().__init__()

    def apply(self, image, label):
        shift_x = np.random.rand()
        shift_y = np.random.rand()

        old_shape = image.shape

        if (old_shape[0] < self.new_shape[0]) | (old_shape[1] < self.new_shape[1]):
            raise RuntimeError()

        shift_x = np.round(shift_x * (old_shape[0] - self.new_shape[0])).astype(np.int)
        shift_y = np.round(shift_y * (old_shape[1] - self.new_shape[1])).astype(np.int)

        shape = image.shape

        image = image[shift_x:shift_x + self.new_shape[0], shift_y:shift_y + self.new_shape[1]]

        if label:
            sampling = np.linalg.norm(label.cell, axis=0) / np.array(shape)
            label.positions -= np.array((shift_x * sampling[0], shift_y * sampling[1]))
            label.cell *= np.asarray(self.new_shape) / np.asarray(old_shape)

        return image, label


class PoissonNoise(DataModifier):

    def __init__(self, mean, background):
        self._mean = mean
        self._background = background

    @property
    def mean(self):
        return self.randomize(self._mean)

    @property
    def background(self):
        return self.randomize(self._background)

    def apply(self, image, label):
        b = - image.min() + self.background
        image = image + b
        a = image.shape[0] * image.shape[1] / image.sum() * self.mean
        image = cp.random.poisson(image * a)
        return (image / a - b).astype(cp.float32), label


class ScanNoise(DataModifier):

    def __init__(self, periodicity, amount):
        self._periodicity = periodicity
        self._amount = amount
        super().__init__()

    @property
    def periodicity(self):
        return self.randomize(self._periodicity)

    @property
    def amount(self):
        return self.randomize(self._amount)

    @staticmethod
    def independent_roll(array, shifts):
        shifts[shifts < 0] += array.shape[1]
        x = cp.arange(array.shape[0])[:, None]
        y = cp.arange(array.shape[1])[None] - shifts[:, None]
        result = array[x, y]
        return result

    def apply(self, image, label):
        noise = bandpass_noise(0, image.shape[1], (image.shape[1],), xp=cp.get_array_module(image))
        if self.periodicity > 1:
            noise *= bandpass_noise(0, self.periodicity, (image.shape[1],), xp=cp.get_array_module(image))
        noise *= self.amount / cp.std(noise)
        image = self.independent_roll(image, noise.astype(cp.int32))
        return image, label


class Warp(DataModifier):

    def __init__(self, periodicity, amount, axis=0):
        self._periodicity = periodicity
        self._amount = amount
        self.axis = axis
        super().__init__()

    @property
    def periodicity(self):
        return self.randomize(self._periodicity)

    @property
    def amount(self):
        return self.randomize(self._amount)

    @functools.lru_cache(1)
    def get_coordinates(self, shape):
        x = cp.linspace(0, shape[0], shape[0], endpoint=False)
        y = cp.linspace(0, shape[1], shape[1], endpoint=False)
        x, y = cp.meshgrid(x, y)
        return cp.vstack([x.ravel(), y.ravel()])

    def apply(self, image, label):
        noise = bandpass_noise(0, self.periodicity, image.shape, xp=cp.get_array_module(image))

        coordinates = self.get_coordinates(image.shape).copy()[::-1]
        coordinates[self.axis] += noise.ravel() * self.amount

        shape = image.shape

        coordinates[0] = cp.clip(coordinates[0], 0, image.shape[0] - 1).astype(cp.int)
        coordinates[1] = cp.clip(coordinates[1], 0, image.shape[1] - 1).astype(cp.int)
        image = cupyx.scipy.ndimage.map_coordinates(image, coordinates, order=1)

        image = image.reshape(shape)

        if label:
            positions = label.positions
            sampling = np.linalg.norm(label.cell, axis=0) / np.array(noise.shape)

            rounded = np.around(positions / sampling).astype(np.int)
            rounded[:, 0] = np.clip(rounded[:, 0], 0, shape[0] - 1)
            rounded[:, 1] = np.clip(rounded[:, 1], 0, shape[1] - 1)

            positions[:, self.axis] -= cp.asnumpy(noise)[rounded[:, 0], rounded[:, 1]] * sampling[self.axis] * self.amount
        return image, label


class PadToSize(DataModifier):

    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def apply(self, image, label):
        if label:
            raise RuntimeError()

        return pad_to_size(image, self.shape[0], self.shape[1]), label


class Flip(DataModifier):

    def __init__(self):
        super().__init__()

    def apply(self, image, label):
        if label:
            sampling = np.linalg.norm(label.cell, axis=0)[0] / np.array(image.shape)

        if np.random.rand() < .5:
            image = image[::-1, :]
            if label:
                label.positions[:, 0] = image.shape[0] * sampling[0] - label.positions[:, 0]

        if np.random.rand() < .5:
            image = image[:, ::-1]
            if label:
                label.positions[:, 1] = image.shape[1] * sampling[1] - label.positions[:, 1]

        return image, label


class Rotate90(DataModifier):

    def __init__(self):
        super().__init__()

    def apply(self, image, label):

        k = np.random.randint(0, 3)

        if k:
            image = cp.rot90(image, k=k)
            if label:
                label.rotate(k * 90)

        return image, label


class Border(DataModifier):

    def __init__(self, amount):
        self.amount = amount

    def apply(self, image, label):
        image[:self.amount] = 0.
        image[-self.amount:] = 0.
        image[:, :self.amount] = 0.
        image[:, -self.amount:] = 0.
        return image, label


class Stain(DataModifier):

    def __init__(self, periodicity, amount):
        self._periodicity = periodicity
        self._amount = amount
        super().__init__()

    @property
    def periodicity(self):
        return self.randomize(self._periodicity)

    @property
    def amount(self):
        return self.randomize(self._amount)

    def apply(self, image, label):
        noise = bandpass_noise(0, self.periodicity, image.shape, xp=cp.get_array_module(image))
        noise *= image.std() / noise.std()
        return image + self.amount * noise, label


class Normalize(DataModifier):

    def __init__(self):
        super().__init__()

    def apply(self, image, label):
        return (image - cp.mean(image)) / cp.std(image), label


class Add(DataModifier):

    def __init__(self, amount):
        self._amount = amount
        super().__init__()

    @property
    def amount(self):
        return self.randomize(self._amount)

    def apply(self, image, label):
        return image + self.amount, label


class Multiply(DataModifier):

    def __init__(self, multiplier):
        self._multiplier = multiplier
        super().__init__()

    @property
    def multiplier(self):
        return self.randomize(self._multiplier)

    def __call__(self, image, label):
        return image * self.multiplier, label


class Blur(DataModifier):

    def __init__(self, sigma):
        self._sigma = sigma
        super().__init__()

    @property
    def sigma(self):
        return self.randomize(self._sigma)

    def __call__(self, image, label):
        return gaussian_filter_2d(image, self.sigma), label
