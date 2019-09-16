from numbers import Number

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage.transform import rescale


class Augmentation(object):
    apply_to_label = True

    def __call__(self, image):
        raise NotImplementedError()

    def randomize(self):
        pass


class Sometimes(Augmentation):
    def __init__(self, probability, augmentation):
        self.probability = probability
        self.augmentation = augmentation

    def randomize(self):
        self._apply_augmentation = np.random.rand() < self.probability

    def __call__(self, image):
        if self._apply_augmentation:
            return self.augmentation(image)
        else:
            return image


class FlipAndRotate90(Augmentation):
    def randomize(self):
        self._fliplr = np.random.rand() < .5
        self._flipud = np.random.rand() < .5
        self._rot90 = np.random.rand() < .5

    def __call__(self, image):
        if self._fliplr:
            image = np.fliplr(image)

        if self._flipud:
            image = np.flipud(image)

        if self._rot90:
            image = np.rot90(image)

        return image


class RandomCrop(Augmentation):

    def __init__(self, out_shape):
        self.out_shape = out_shape

    def randomize(self):
        self._shift_x = np.random.rand()  # np.random.randint(0, old_size[0] - self.size[0])

    def __call__(self, image):
        old_size = image.shape[:2]

        shift_x = np.round(self._shift_x * (old_size[0] - self.out_shape[0])).astype(np.int)
        shift_y = np.round(self._shift_x * (old_size[0] - self.out_shape[0])).astype(np.int)

        image = image[shift_x:shift_x + self.out_shape[0], shift_y:shift_y + self.out_shape[1]]

        return image


class Zoom(Augmentation):

    def __init__(self, zoom):
        self.zoom = zoom

    def randomize(self):
        self._random_zoom = np.random.uniform(*self.zoom)

    def __call__(self, image):
        image = rescale(image, self._random_zoom, mode='reflect', multichannel=False, anti_aliasing=False,
                        preserve_range=True)
        return image


class ShiftValues(Augmentation):
    apply_to_label = False

    def __init__(self, shift):
        self.shift = shift

    def randomize(self):
        self._random_shift = np.random.uniform(*self.shift)

    def __call__(self, image):
        return image + self._random_shift


class ScaleValues(Augmentation):
    apply_to_label = False

    def __init__(self, scale):
        self.scale = scale

    def randomize(self):
        self._random_scale = np.random.uniform(*self.scale)

    def __call__(self, image):
        return image * self._random_scale


class NormalizeRange(Augmentation):
    apply_to_label = False

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())


class NormalizeVariance(Augmentation):
    apply_to_label = False

    def __call__(self, image):
        return (image - np.mean(image)) / np.std(image)


class NormalizeLocal(Augmentation):
    apply_to_label = False

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, image):
        mean = gaussian_filter(image, self.sigma)
        image = image - mean
        image = image / np.sqrt(gaussian_filter(image ** 2, self.sigma))
        return image


class GaussianBlur(Augmentation):
    apply_to_label = False

    def __init__(self, sigma):
        self.sigma = sigma

    def randomize(self):
        self._random_sigma = np.random.uniform(*self.sigma)

    def __call__(self, image):
        return gaussian_filter(image, self._random_sigma)


class Gamma(Augmentation):
    apply_to_label = False

    def __init__(self, gamma):
        self.gamma = gamma

    def randomize(self):
        self._random_gamma = np.random.uniform(*self.gamma)

    def internal_apply(self, image):
        return image ** self._random_gamma


class PoissonNoise(Augmentation):
    apply_to_label = False

    def __init__(self, mean, background):
        self.mean = mean
        self.background = background

    def randomize(self):
        self._random_mean = np.random.uniform(*self.mean)
        self._random_background = np.random.uniform(*self.background)

    def __call__(self, image):
        image = image - image.min() + self._random_background
        image = image / image.sum() * np.prod(image.shape[:2])
        image = np.random.poisson(image * self._random_mean).astype(np.float)
        return image


class GaussianNoise(Augmentation):
    apply_to_label = False

    def __init__(self, sigma):
        self.sigma = sigma

    def randomize(self):
        self._random_sigma = np.random.uniform(*self.sigma)

    def apply_image(self, image):
        image = image + np.random.randn(*image.shape) * self._random_sigma
        return image


def bandpass_noise(inner, outer, shape, sampling):
    k = np.sqrt(np.sum([ki ** 2 for ki in
                        np.meshgrid(*[np.fft.fftfreq(n, d) for (n, d) in zip(shape, sampling)])], axis=0))
    mask = (k > inner) & (k < outer)
    noise = np.fft.ifftn(mask * np.exp(-1.j * 2 * np.pi * np.random.rand(*k.shape)))
    noise = (noise.real + noise.imag) / 2
    return noise / np.std(noise)


class ScanNoise(Augmentation):
    apply_to_label = False

    def __init__(self, scale, amount):
        self.scale = scale
        self.amount = amount

    def randomize(self):
        self._random_scale = np.random.uniform(*self.scale)
        self._random_amount = np.random.uniform(*self.amount)

    def __call__(self, image):
        n = bandpass_noise(0, self._random_scale, (image.shape[1],), (1. / image.shape[1],))
        n *= bandpass_noise(0, np.max(image.shape), (image.shape[1],), (1. / image.shape[1],))
        n = n / np.std(n) * self._random_amount
        n = n.astype(np.int)

        def strided_indexing_roll(a, r):
            from skimage.util.shape import view_as_windows
            a_ext = np.concatenate((a, a[:, :-1]), axis=1)
            n = a.shape[1]
            return view_as_windows(a_ext, (1, n))[np.arange(len(r)), (n - r) % n, 0]

        image = strided_indexing_roll(np.ascontiguousarray(image), n)

        return image


class Border(Augmentation):
    apply_to_label = False

    def __init__(self, amount):
        self.amount = amount

    def randomize(self):
        self._random_amount = int(np.random.uniform(*self.amount))

    def __call__(self, image):
        new_image = np.zeros_like(image)
        new_image[self._random_amount:-self._random_amount,
        self._random_amount:-self._random_amount] = image[self._random_amount:-self._random_amount,
                                                    self._random_amount:-self._random_amount]

        return new_image
