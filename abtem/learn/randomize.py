import functools

import cupy as cp
import cupyx.scipy.ndimage
import numpy as np
import scipy.spatial as spatial

from abtem.learn.filters import gaussian_filter_2d
from abtem.points import Points, fill_rectangle


def graphene_like(a=2.46, n=1, m=1, pointwise_properties=None):
    basis = [(0, 0), (2 / 3., 1 / 3.)]
    cell = [[a, 0], [-a / 2., a * 3 ** 0.5 / 2.]]
    positions = np.dot(np.array(basis), np.array(cell))

    points = Points(positions, cell=cell, pointwise_properties=pointwise_properties)
    points = points.repeat(n, m)
    return points





class SequentialStructureModifier:

    def __init__(self, modifiers):
        self.modifiers = modifiers

    def __call__(self, points):
        for modifier in self.modifiers:
            points = modifier(points)

        return points


class RandomRotation:

    def __init__(self, rotate_cell=True):
        self.rotate_cell = rotate_cell

    def __call__(self, points):
        return points.rotate(np.random.rand() * 360., rotate_cell=True)


class FillRectangle:

    def __init__(self, extent, margin):
        self.extent = extent
        self.margin = margin

    def __call__(self, points):
        return fill_rectangle(points, self.extent, margin=self.margin)


class RandomDelete:

    def __init__(self, fraction):
        self.fraction = fraction

    def __call__(self, points):
        n = int(np.round((1 - self.fraction) * len(points)))
        return points[np.random.choice(len(points), n, replace=False)]


class RandomSetProperties:

    def __init__(self, fraction, name, new_value, default_value=0):
        self.fraction = fraction
        self.name = name
        self.new_value = new_value
        self.default_value = default_value

    def __call__(self, points):
        n = int(np.round(self.fraction * len(points)))
        idx = np.random.choice(len(points), n, replace=False)

        try:
            properties = points.get_properties(self.name)
            properties[idx] = self.new_value
        except KeyError:
            properties = np.full(fill_value=self.default_value, shape=len(points))
            properties[idx] = self.new_value
            points.set_properties(self.name, properties)

        return points


class RandomRotateNeighbors:

    def __init__(self, fraction, cutoff):
        self.R = np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2)],
                           [np.sin(np.pi / 2), np.cos(np.pi / 2)]])

        self.fraction = fraction
        self.cutoff = cutoff

    def __call__(self, points):
        point_tree = spatial.cKDTree(points.positions)
        n = int(np.round(self.fraction * len(points)))
        first_indices = np.random.choice(len(points), n, replace=False)

        marked = np.zeros(len(points), dtype=np.bool)
        pairs = []
        for first_index in first_indices:
            marked[first_index] = True
            second_indices = point_tree.query_ball_point(points.positions[first_index], 1.5)

            if len(second_indices) < 2:
                continue

            np.random.shuffle(second_indices)

            for second_index in second_indices:
                if not marked[second_index]:
                    break

            marked[second_index] = True
            pairs += [(first_index, second_index)]

        for i, j in pairs:
            center = np.mean(points.positions[[i, j]], axis=0)
            points.positions[[i, j]] = np.dot(self.R, (points.positions[[i, j]] - center).T).T + center

        return points


class RandomStrain:

    def __init__(self):
        pass

    def random_strain(points, scale, amplitude):
        shape = np.array((128, 128))
        sampling = np.linalg.norm(points.cell, axis=1) / shape
        outer = 1 / scale * 2
        noise = bandpass_noise(inner=0, outer=outer, shape=shape, sampling=sampling)
        indices = np.floor((points.scaled_positions % 1.) * shape).astype(np.int)

        for i in [0, 1]:
            points.positions[:, i] += amplitude * noise[indices[:, 0], indices[:, 1]]

        return points


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

    def __call__(self, images, labels):
        assert len(images) == len(labels)

        for i in range(len(images)):
            images[i], labels[i] = self.apply(images[i], labels[i])

        return images, labels


class SequentialDataModifiers(DataModifier):

    def __init__(self, modifiers):
        super().__init__()
        self.modifiers = modifiers

    def __call__(self, images, labels):
        for modifier in self.modifiers:
            images, labels = modifier(images, labels)

        return images, labels


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

        sampling = cp.linalg.norm(label.cell, axis=0) / cp.array(image.shape)

        image = image[shift_x:shift_x + self.new_shape[0], shift_y:shift_y + self.new_shape[1]]
        label.positions -= cp.array((shift_x * sampling[0], shift_y * sampling[1]))



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
        image = cp.random.poisson(image * a).astype(cp.float32)
        return image / a - b, label


class ScanNoise(DataModifier):

    def __init__(self, periodicity, amount):
        self.periodicity = periodicity
        self.amount = amount
        super().__init__()

    @staticmethod
    def independent_roll(array, shifts):
        shifts[shifts < 0] += array.shape[1]
        x = cp.arange(array.shape[0])[:, None]
        y = cp.arange(array.shape[1])[None] - shifts[:, None]
        result = array[x, y]
        return result

    def apply(self, image, label):
        freqs = cp.fft.fftfreq(image.shape[0], 1 / image.shape[0])
        noise = bandpass_noise(0, self.periodicity, freqs)
        noise = noise / cp.std(noise) * self.amount
        image = self.independent_roll(image, noise.astype(cp.int32))
        return image, label


class Warp(DataModifier):

    def __init__(self, scale, amount, axis=0):
        self.scale = scale
        self.amount = amount
        self.axis = axis
        super().__init__()

    @functools.lru_cache(1)
    def get_coordinates(self, shape):
        x = cp.linspace(0, shape[0], shape[0], endpoint=False)
        y = cp.linspace(0, shape[1], shape[1], endpoint=False)
        x, y = cp.meshgrid(x, y)
        return cp.vstack([x.ravel(), y.ravel()])

    def apply(self, image, label):
        kx = cp.fft.fftfreq(image.shape[0], 1 / image.shape[0])
        ky = cp.fft.fftfreq(image.shape[1], 1 / image.shape[1])
        k = cp.sqrt(kx[:, None] ** 2 + ky[None] ** 2)
        noise = bandpass_noise(0, self.scale, k)

        coordinates = self.get_coordinates(image.shape).copy()[::-1]
        coordinates[self.axis] += noise.ravel() * self.amount
        shape = image.shape

        coordinates[0] = cp.clip(coordinates[0], 0, image.shape[0] - 1).astype(cp.int)
        coordinates[1] = cp.clip(coordinates[1], 0, image.shape[1] - 1).astype(cp.int)
        image = cupyx.scipy.ndimage.map_coordinates(image, coordinates, order=1)
        image = image.reshape(shape)

        positions = label.positions
        sampling = cp.linalg.norm(label.cell, axis=0) / cp.array(noise.shape)

        rounded = cp.around(positions / sampling).astype(cp.int)
        positions[:, self.axis] -= noise[rounded[:, 0], rounded[:, 1]] * sampling[self.axis] * self.amount
        return image, label


class Flip(DataModifier):

    def __init__(self):
        super().__init__()

    def apply(self, image, label):
        sampling = np.linalg.norm(label.cell, axis=0)[0] / cp.array(image.shape)

        if np.random.rand() < .5:
            image = image[::-1, :]
            label.positions[:, 0] = image.shape[0] * sampling[0] - label.positions[:, 0]

        if np.random.rand() < .5:
            image = image[:, ::-1]
            label.positions[:, 1] = image.shape[1] * sampling[1] - label.positions[:, 1]

        return image, label


class Rotate90(DataModifier):

    def __init__(self):
        super().__init__()

    def apply(self, image, label):
        k = np.random.randint(0, 3)

        if k:
            image = cp.rot90(image, k=k)
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

    def __init__(self, scale, amount):
        self.scale = scale
        self.amount = amount
        super().__init__()

    def apply(self, image, label):
        kx = cp.fft.fftfreq(image.shape[0], 1 / image.shape[0])
        ky = cp.fft.fftfreq(image.shape[1], 1 / image.shape[1])
        k = cp.sqrt(kx[:, None] ** 2 + ky[None] ** 2)
        noise = bandpass_noise(0, self.scale, k)
        noise /= noise.max()
        return image * (1 - self.amount * noise), label


class Normalize(DataModifier):

    def __init__(self):
        super().__init__()

    def apply(self, image, label):
        return (image - cp.mean(image)) / cp.std(image), label


class Add(DataModifier):

    def __init__(self, amount):
        self.amount = amount
        super().__init__()

    def apply(self, image, label):
        return image + self.amount, label


class Multiply(DataModifier):

    def __init__(self, multiplier):
        self.multiplier = multiplier
        super().__init__()

    def __call__(self, image, label):
        return image * self.multiplier, label


class Blur(DataModifier):

    def __init__(self, sigma):
        self.sigma = sigma
        super().__init__()

    def __call__(self, image, label):
        return gaussian_filter_2d(image, self.sigma), label
