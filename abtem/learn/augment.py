import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.transform import rescale
from skimage.exposure import equalize_adapthist


class Augmentation(object):

    def __init__(self, apply_to_label=True):
        self.apply_to_label = apply_to_label

    def __call__(self, image):
        raise NotImplementedError()

    def randomize(self):
        pass


class Sometimes(Augmentation):

    def __init__(self, augmentation, probability):
        self.probability = probability
        self._augmentation = augmentation

        super().__init__(apply_to_label=augmentation.apply_to_label)

    def randomize(self):
        self._apply_augmentation = np.random.rand() < self.probability
        self._augmentation.randomize()

    def __call__(self, image):
        if self._apply_augmentation:
            return self._augmentation(image)
        else:
            return image


class Flip(Augmentation):

    def __init__(self):
        super().__init__(apply_to_label=True)

    def randomize(self):
        self._fliplr = np.random.rand() < .5
        self._flipud = np.random.rand() < .5

    def __call__(self, image):
        if self._fliplr:
            image = image[:, :, ::-1]

        if self._flipud:
            image = image[:, ::-1, :]

        return image


class Rotate90(Augmentation):

    def __init__(self):
        super().__init__(apply_to_label=True)

    def randomize(self):
        self._rot90 = np.random.randint(0, 3)

    def __call__(self, image):
        if self._rot90:
            image = np.rot90(image, k=self._rot90)

        return image


class RandomCropStack(Augmentation):

    def __init__(self, out_shape):
        self.out_shape = out_shape
        super().__init__(apply_to_label=True)

    def randomize(self):
        self._shift_x = np.random.rand()
        self._shift_y = np.random.rand()

    def __call__(self, image):
        old_size = image.shape[2:]

        if (old_size[0] < self.out_shape[0]) | (old_size[1] < self.out_shape[1]):
            raise RuntimeError()

        shift_x = np.round(self._shift_x * (old_size[0] - self.out_shape[0])).astype(np.int)
        shift_y = np.round(self._shift_y * (old_size[1] - self.out_shape[1])).astype(np.int)

        image = image[:, :, shift_x:shift_x + self.out_shape[0], shift_y:shift_y + self.out_shape[1]]

        return image


class Zoom(Augmentation):

    def __init__(self, zoom):
        self.zoom = zoom
        super().__init__(apply_to_label=True)

    def randomize(self):
        self._random_zoom = np.random.uniform(*self.zoom)

    def __call__(self, image):
        image = rescale(image, self._random_zoom, mode='reflect', multichannel=False, anti_aliasing=False,
                        preserve_range=True)
        return image


class ShiftValues(Augmentation):

    def __init__(self, shift):
        self.shift = shift
        super().__init__(apply_to_label=False)

    def randomize(self):
        self._random_shift = np.random.uniform(*self.shift)

    def __call__(self, image):
        return image + self._random_shift


class ScaleValues(Augmentation):

    def __init__(self, scale):
        self.scale = scale
        super().__init__(apply_to_label=False)

    def randomize(self):
        self._random_scale = np.random.uniform(*self.scale)

    def __call__(self, image):
        return image * self._random_scale


class NormalizeRange(Augmentation):

    def __init__(self):
        super().__init__(apply_to_label=False)

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())


class NormalizeVariance(Augmentation):
    def __init__(self):
        super().__init__(apply_to_label=False)

    def __call__(self, image):
        return (image - np.mean(image)) / np.std(image)


class NormalizeLocal(Augmentation):

    def __init__(self, sigma):
        self.sigma = sigma
        super().__init__(apply_to_label=False)

    def __call__(self, image):
        mean = gaussian_filter(image, (0, self.sigma, self.sigma))
        image = image - mean
        image = image / np.sqrt(gaussian_filter(image ** 2, (0, self.sigma, self.sigma)))
        return image


class EqualizeAdaptive(Augmentation):

    def __init__(self, kernel_size=None, clip_limit=.01):
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size
        super().__init__(apply_to_label=False)

    def __call__(self, image):
        for i in range(len(image)):
            image[i] = equalize_adapthist(image[i], kernel_size=self.kernel_size, clip_limit=self.clip_limit)
        return image


class GaussianBlur(Augmentation):

    def __init__(self, sigma):
        self.sigma = sigma
        super().__init__(apply_to_label=False)

    def randomize(self):
        self._random_sigma = np.random.uniform(*self.sigma)

    def __call__(self, image):
        return gaussian_filter(image, (0, self._random_sigma, self._random_sigma))


class Gamma(Augmentation):

    def __init__(self, gamma):
        self.gamma = gamma
        super().__init__(apply_to_label=False)

    def randomize(self):
        self._random_gamma = np.random.uniform(*self.gamma)

    def internal_apply(self, image):
        return image ** self._random_gamma


class PoissonNoise(Augmentation):

    def __init__(self, mean, background):
        self.mean = mean
        self.background = background
        super().__init__(apply_to_label=False)

    def randomize(self):
        self._random_mean = np.random.uniform(*self.mean)
        self._random_background = np.random.uniform(*self.background)

    def __call__(self, image):
        image = image - image.min() + self._random_background
        image = image / image.sum() * np.prod(image.shape[1:])
        image = np.random.poisson(image * self._random_mean).astype(np.float)
        return image


class GaussianNoise(Augmentation):

    def __init__(self, sigma):
        self.sigma = sigma
        super().__init__(apply_to_label=False)

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


def independent_roll(array, shifts, axis=1):
    array = np.swapaxes(array, axis, -1)
    all_idcs = np.ogrid[[slice(0, n) for n in array.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += array.shape[-1]
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = array[tuple(all_idcs)]
    arr = np.swapaxes(result, -1, axis)
    return arr


class ScanNoise(Augmentation):

    def __init__(self, slow_scale, amount, fast_scale=None, apply_to_label=False):
        self.slow_scale = slow_scale
        self.amount = amount
        self.fast_scale = fast_scale
        super().__init__(apply_to_label=apply_to_label)

    def randomize(self):
        self._random_slow_scale = np.random.uniform(*self.slow_scale)
        self._random_amount = np.random.uniform(*self.amount)
        if self.fast_scale is None:
            self._random_fast_scale = None
        else:
            self._random_fast_scale = np.random.uniform(*self.fast_scale)
        self._noise = None

    def __call__(self, image):
        if self._random_fast_scale is None:
            fast_scale = np.max(image.shape)
        else:
            fast_scale = self._random_fast_scale

        if self._noise is None:
            self._noise = bandpass_noise(0, self._random_slow_scale, (image.shape[1],), (1. / image.shape[1],))
            self._noise = self._noise * bandpass_noise(0, fast_scale, (image.shape[1],), (1. / image.shape[1],))
            self._noise = self._noise / np.std(self._noise) * self._random_amount
            self._noise = self._noise.astype(np.int)

        image = independent_roll(image, self._noise, 2)
        return image


# def __init__(self, rms_power):
#     self.rms_power = rms_power
#     super().__init__(apply_to_label=False)

# def randomize(self):
#     self._random_rms_power = np.random.uniform(*self.rms_power)
#
#     def __call__(self, image):
#         flyback_time = 361e-6  # Flyback time [s]
#         dwell_time = 8e-6  # Dwell time [s]
#         max_frequency = 50  # [Hz]
#         print(self._random_rms_power)
#         return add_scan_noise(image, dwell_time, flyback_time, max_frequency, self._random_rms_power, num_components=300)

# def randomize(self):
#     self._random_rms_power = np.random.uniform(*self.rms_power)


class Blank(Augmentation):

    def __init__(self, width):
        self.width = width
        super().__init__(apply_to_label=False)

    def randomize(self):
        self._random_width = np.random.randint(*self.width)
        self._random_position = None

    def __call__(self, image):
        if self._random_position is None:
            self._random_position = np.random.randint(0, image.shape[1])

        image[:, self._random_position:self._random_position + self._random_width] = image.min()

        return image


class Glitch(Augmentation):

    def __init__(self, width):
        self.width = width
        super().__init__(apply_to_label=True)

    def randomize(self):
        self._random_width = np.random.randint(*self.width)
        self._random_position = None

    def __call__(self, image):
        if self._random_position is None:
            self._random_position = np.random.randint(self._random_width, image.shape[1])

        image[:, self._random_position:] = image[:, -(image.shape[1] - self._random_position +
                                                      self._random_width):-self._random_width]

        return image


class LineWarp(Augmentation):

    def __init__(self, scale, amount, apply_to_label=False):
        self.scale = scale
        self.amount = amount
        super().__init__(apply_to_label=apply_to_label)

    def randomize(self):
        self._random_scale = np.random.uniform(*self.scale)
        self._random_amount = np.random.uniform(*self.amount)
        self._noise = None

    def __call__(self, image):
        if self._noise is None:
            self._noise = bandpass_noise(0, self._random_scale, (10,), (1 / 10,))

        x = np.linspace(0, image.shape[1], 2)
        y = np.linspace(0, image.shape[2], 5)
        x, y = np.meshgrid(x, y)
        src = np.array([x.ravel(), y.ravel()]).T

        dst = src.copy()
        dst[:, 1] = dst[:, 1] + self._random_amount * self._noise
        dst[0][dst[0] < 0] = 0
        dst[0][dst[0] > image.shape[1] - 1] = image.shape[1] - 1

        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        warped = np.zeros_like(image)
        for i in range(len(image)):
            warped[i] = warp(image[i], tform, output_shape=image.shape[1:], order=0)
        return warped


class LineDarkening(Augmentation):

    def __init__(self, scale, amount):
        self.scale = scale
        self.amount = amount
        super().__init__(apply_to_label=False)

    def randomize(self):
        self._random_scale = np.random.uniform(*self.scale)
        self._random_amount = np.random.uniform(*self.amount)

    def __call__(self, image):
        self._noise = bandpass_noise(0, self._random_scale, (image.shape[1],), (1 / image.shape[1],))
        self._noise = self._noise / self._noise.max()
        return image * (1 - self._random_amount * self._noise[None, :, None])


class Border(Augmentation):

    def __init__(self, amount):
        self.amount = amount
        super().__init__(apply_to_label=False)

    def randomize(self):
        self._random_amount = int(np.random.uniform(*self.amount))

    def __call__(self, image):
        new_image = np.zeros_like(image)
        new_image[:,self._random_amount:-self._random_amount,
        self._random_amount:-self._random_amount] = image[:,self._random_amount:-self._random_amount,
                                                    self._random_amount:-self._random_amount]

        return new_image
