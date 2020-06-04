import cupy as cp
import cupyx.scipy.ndimage
import numpy as np
from abtem.learn.utils import pad_to_size
from abtem.noise import bandpass_noise
from scipy.ndimage import gaussian_filter


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


class Augmentation:

    def randomize(self, random_number_generator):
        try:
            return random_number_generator.new_value
        except AttributeError:
            return random_number_generator

    def apply(self, example):
        raise NotImplementedError()

    def __call__(self, example):
        self.apply(example)


class SequentialAugmentations(Augmentation):

    def __init__(self, modifiers):
        super().__init__()
        self.modifiers = modifiers

    def apply(self, example):
        for modifier in self.modifiers:
            modifier.apply(example)


class Sometimes(Augmentation):

    def __init__(self, p, augmentation):
        self._p = p
        self._augmentation = augmentation

    def apply(self, example):
        if np.random.rand() < self._p:
            return self._augmentation(example)
        else:
            return example


class RandomCrop(Augmentation):

    def __init__(self, new_shape):
        self.new_shape = new_shape
        super().__init__()

    def apply(self, example):
        shift_x = np.random.rand()
        shift_y = np.random.rand()

        old_shape = example.image.shape

        if (old_shape[0] < self.new_shape[0]) | (old_shape[1] < self.new_shape[1]):
            return example

        shift_x = np.round(shift_x * (old_shape[0] - self.new_shape[0])).astype(np.int)
        shift_y = np.round(shift_y * (old_shape[1] - self.new_shape[1])).astype(np.int)
        example.image = example.image[shift_x:shift_x + self.new_shape[0], shift_y:shift_y + self.new_shape[1]]

        example.points -= np.array((shift_x, shift_y))


class PoissonNoise(Augmentation):

    def __init__(self, mean, background):
        self._mean = mean
        self._background = background

    @property
    def mean(self):
        return self.randomize(self._mean)

    @property
    def background(self):
        return self.randomize(self._background)

    def apply(self, example):
        image = example.image
        xp = cp.get_array_module(image)
        b = - image.min() + self.background
        image = image + b
        a = image.shape[0] * image.shape[1] / image.sum() * self.mean
        image = xp.random.poisson(image * a)
        example.image[:] = (image / a - b).astype(xp.float32)


class ScanNoise(Augmentation):

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
        xp = cp.get_array_module(array)
        shifts[shifts < 0] += array.shape[1]
        x = xp.arange(array.shape[0])[:, None]
        y = xp.arange(array.shape[1])[None] - shifts[:, None]
        result = array[x, y]
        return result

    def apply(self, example):
        image = example.image
        xp = cp.get_array_module(image)
        noise = bandpass_noise(0, image.shape[1], (image.shape[1],), xp=xp)
        if self.periodicity > 1:
            noise *= bandpass_noise(0, self.periodicity, (image.shape[1],), xp=xp)
        noise *= self.amount / xp.std(noise)
        example.image[:] = self.independent_roll(image, noise.astype(np.int32))


class Warp(Augmentation):

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

    def apply(self, example):
        image = example.image
        xp = cp.get_array_module(image)
        scipy = cupyx.scipy.get_array_module(image)

        noise = bandpass_noise(0, self.periodicity, image.shape, xp=cp.get_array_module(image))

        x = xp.linspace(0, image.shape[0], image.shape[0], endpoint=False)
        y = xp.linspace(0, image.shape[1], image.shape[1], endpoint=False)
        x, y = xp.meshgrid(x, y)
        coordinates = xp.vstack([y.ravel(), x.ravel()])

        coordinates[self.axis] += noise.ravel() * self.amount

        shape = image.shape

        coordinates[0] = xp.clip(coordinates[0], 0, image.shape[0] - 1).astype(cp.int)
        coordinates[1] = xp.clip(coordinates[1], 0, image.shape[1] - 1).astype(cp.int)

        image = scipy.ndimage.map_coordinates(image, coordinates, order=1)

        example.image = image.reshape(shape)

        points = example.points
        rounded = np.around(points).astype(np.int)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, shape[0] - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, shape[1] - 1)

        points[:, self.axis] -= cp.asnumpy(noise)[rounded[:, 0], rounded[:, 1]] * self.amount


class PadToSize(Augmentation):

    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def apply(self, example):
        example.image, padding = pad_to_size(example.image, self.shape[0], self.shape[1], mode='reflect')
        example.points[:] += [padding[0], padding[2]]


class PadByAmount(Augmentation):

    def __init__(self, amount):
        self.amount = amount
        super().__init__()

    def apply(self, example):
        example.image, padding = pad_to_size(example.image, example.shape[0] + self.amount,
                                             example.shape[1] + self.amount, mode='reflect', n=16)

        example.points[:] += [padding[0], padding[2]]


class Flip(Augmentation):

    def __init__(self):
        super().__init__()

    def apply(self, example):
        if np.random.rand() < .5:
            example.image[:] = example.image[::-1, :]
            example.points[:, 0] = example.image.shape[0] - example.points[:, 0]

        if np.random.rand() < .5:
            example.image[:] = example.image[:, ::-1]
            example.points[:, 1] = example.image.shape[1] - example.points[:, 1]

        return example


class Rotate90(Augmentation):

    def __init__(self):
        super().__init__()

    def apply(self, example):
        xp = cp.get_array_module(example)
        k = np.random.randint(0, 3)

        if k:
            example.image = xp.rot90(example.image, k=k).copy()

            if k == 1:
                old_points = example.points.copy() - np.array(example.image.shape)[::-1] / 2
                example.points[:, 0] = - old_points[:, 1]
                example.points[:, 1] = old_points[:, 0]
            elif k == 2:
                old_points = example.points.copy() - np.array(example.image.shape) / 2
                example.points[:, 0] = - old_points[:, 0]
                example.points[:, 1] = - old_points[:, 1]
            else:
                old_points = example.points.copy() - np.array(example.image.shape)[::-1] / 2
                example.points[:, 0] = old_points[:, 1]
                example.points[:, 1] = - old_points[:, 0]

            example.points += np.array(example.image.shape) / 2


class Stain(Augmentation):

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

    def apply(self, example):
        image = example.image
        noise = bandpass_noise(0, self.periodicity, image.shape, xp=cp.get_array_module(image))
        noise *= image.std() / noise.std()
        example.image = example.image + self.amount * noise


class Normalize(Augmentation):

    def __init__(self):
        super().__init__()

    def apply(self, example):
        image = example.image
        xp = cp.get_array_module(image)
        example.image = (example.image - xp.mean(image)) / xp.std(image)


class Add(Augmentation):

    def __init__(self, amount):
        self._amount = amount
        super().__init__()

    @property
    def amount(self):
        return self.randomize(self._amount)

    def apply(self, example):
        example.image += self.amount


class Multiply(Augmentation):

    def __init__(self, multiplier):
        self._multiplier = multiplier
        super().__init__()

    @property
    def multiplier(self):
        return self.randomize(self._multiplier)

    def apply(self, example):
        example.image *= self.multiplier


class Blur(Augmentation):

    def __init__(self, sigma):
        self._sigma = sigma
        super().__init__()

    @property
    def sigma(self):
        return self.randomize(self._sigma)

    def apply(self, example):
        example.image = gaussian_filter(example.image, self.sigma)
