import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from abtem.noise import add_scan_noise


class Augmentation(object):

    def __init__(self, apply_to_label=True, channels=None):
        self.apply_to_label = apply_to_label

        if channels is not None:
            channels = np.array(channels).reshape((1,))

        self.channels = channels

    def __call__(self, image):
        raise NotImplementedError()

    def randomize(self):
        pass


class Sometimes(Augmentation):

    def __init__(self, augmentation, probability):
        self.probability = probability
        self._augmentation = augmentation

        super().__init__(apply_to_label=augmentation.apply_to_label, channels=augmentation.channels)

    def randomize(self):
        self._apply_augmentation = np.random.rand() < self.probability
        self._augmentation.randomize()

    def __call__(self, image):
        if self._apply_augmentation:
            return self._augmentation(image)
        else:
            return image


class FlipAndRotate90(Augmentation):

    def __init__(self):
        super().__init__(apply_to_label=True, channels=None)

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
        super().__init__(apply_to_label=True, channels=None)

    def randomize(self):
        self._shift_x = np.random.rand()
        self._shift_y = np.random.rand()

    def __call__(self, image):
        old_size = image.shape[:2]

        if (old_size[0] < self.out_shape[0]) | (old_size[1] < self.out_shape[1]):
            raise RuntimeError()

        shift_x = np.round(self._shift_x * (old_size[0] - self.out_shape[0])).astype(np.int)
        shift_y = np.round(self._shift_y * (old_size[1] - self.out_shape[1])).astype(np.int)

        image = image[shift_x:shift_x + self.out_shape[0], shift_y:shift_y + self.out_shape[1]]

        return image


class Zoom(Augmentation):

    def __init__(self, zoom):
        self.zoom = zoom
        super().__init__(apply_to_label=True, channels=None)

    def randomize(self):
        self._random_zoom = np.random.uniform(*self.zoom)

    def __call__(self, image):
        image = rescale(image, self._random_zoom, mode='reflect', multichannel=False, anti_aliasing=False,
                        preserve_range=True)
        return image


class ShiftValues(Augmentation):

    def __init__(self, shift):
        self.shift = shift
        super().__init__(apply_to_label=False, channels=None)

    def randomize(self):
        self._random_shift = np.random.uniform(*self.shift)

    def __call__(self, image):
        return image + self._random_shift


class ScaleValues(Augmentation):

    def __init__(self, scale):
        self.scale = scale
        super().__init__(apply_to_label=False, channels=None)

    def randomize(self):
        self._random_scale = np.random.uniform(*self.scale)

    def __call__(self, image):
        return image * self._random_scale


class NormalizeRange(Augmentation):

    def __init__(self):
        super().__init__(apply_to_label=False, channels=None)

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())


class NormalizeVariance(Augmentation):
    def __init__(self):
        super().__init__(apply_to_label=False, channels=None)

    def __call__(self, image):
        return (image - np.mean(image)) / np.std(image)


class NormalizeLocal(Augmentation):

    def __init__(self, sigma):
        self.sigma = sigma
        super().__init__(apply_to_label=False, channels=None)

    def __call__(self, image):
        mean = gaussian_filter(image, self.sigma)
        image = image - mean
        image = image / np.sqrt(gaussian_filter(image ** 2, self.sigma))
        return image


class GaussianBlur(Augmentation):

    def __init__(self, sigma):
        self.sigma = sigma
        super().__init__(apply_to_label=False, channels=None)

    def randomize(self):
        self._random_sigma = np.random.uniform(*self.sigma)

    def __call__(self, image):
        return gaussian_filter(image, self._random_sigma)


class Gamma(Augmentation):

    def __init__(self, gamma):
        self.gamma = gamma
        super().__init__(apply_to_label=False, channels=None)

    def randomize(self):
        self._random_gamma = np.random.uniform(*self.gamma)

    def internal_apply(self, image):
        return image ** self._random_gamma


class PoissonNoise(Augmentation):

    def __init__(self, mean, background):
        self.mean = mean
        self.background = background
        super().__init__(apply_to_label=False, channels=None)

    def randomize(self):
        self._random_mean = np.random.uniform(*self.mean)
        self._random_background = np.random.uniform(*self.background)

    def __call__(self, image):
        image = image - image.min() + self._random_background
        image = image / image.sum() * np.prod(image.shape[:2])
        image = np.random.poisson(image * self._random_mean).astype(np.float)
        return image


class GaussianNoise(Augmentation):

    def __init__(self, sigma):
        self.sigma = sigma
        super().__init__(apply_to_label=False, channels=None)

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

    def __init__(self, slow_scale, amount, fast_scale=None, apply_to_label=False):
        self.slow_scale = slow_scale
        self.amount = amount
        self.fast_scale = fast_scale
        super().__init__(apply_to_label=apply_to_label, channels=None)

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

        def strided_indexing_roll(a, r):
            from skimage.util.shape import view_as_windows
            a_ext = np.concatenate((a, a[:, :-1]), axis=1)
            n = a.shape[1]
            return view_as_windows(a_ext, (1, n))[np.arange(len(r)), (n - r) % n, 0]

        image = strided_indexing_roll(np.ascontiguousarray(image), self._noise)
        return image


# def __init__(self, rms_power):
#     self.rms_power = rms_power
#     super().__init__(apply_to_label=False, channels=None)

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
            self._random_position = np.random.randint(0, image.shape[0])

        image[self._random_position:self._random_position + self._random_width] = image.min()

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
            self._random_position = np.random.randint(self._random_width, image.shape[0])

        image[self._random_position:] = image[-(image.shape[0] - self._random_position +
                                                self._random_width):-self._random_width]

        return image


class LineWarp(Augmentation):

    def __init__(self, scale, amount, axis=0, channels=None, apply_to_label=False):
        self.scale = scale
        self.amount = amount
        self.axis = axis
        super().__init__(apply_to_label=apply_to_label, channels=channels)

    def randomize(self):
        self._random_scale = np.random.uniform(*self.scale)
        self._random_amount = np.random.uniform(*self.amount)
        self._noise = None

    def __call__(self, image):
        if self._noise is None:
            self._noise = bandpass_noise(0, self._random_scale, (image.shape[self.axis],),
                                         (1 / image.shape[self.axis],))

        x = np.arange(0, image.shape[0])
        y = np.arange(0, image.shape[1])

        indices = [x, y]

        interpolating_function = RegularGridInterpolator(indices, image)

        indices[self.axis] = indices[self.axis] + self._random_amount * self._noise
        indices[self.axis][indices[self.axis] < 0] = 0
        indices[self.axis][indices[self.axis] > image.shape[self.axis] - 1] = image.shape[self.axis] - 1

        x, y = np.meshgrid(indices[0], indices[1], indexing='ij')

        p = np.array([x.ravel(), y.ravel()]).T

        warped = interpolating_function(p)
        warped = warped.reshape(image.shape)
        return warped


class LineDarkening(Augmentation):

    def __init__(self, scale, amount, axis=0, channels=None):
        self.scale = scale
        self.amount = amount
        self.axis = axis
        super().__init__(apply_to_label=False, channels=channels)

    def randomize(self):
        self._random_scale = np.random.uniform(*self.scale)
        self._random_amount = np.random.uniform(*self.amount)

    def __call__(self, image):
        self._noise = bandpass_noise(0, self._random_scale, (image.shape[self.axis],),
                                     (1 / image.shape[self.axis],))

        self._noise = self._noise / self._noise.max()
        if self.axis == 0:
            return image * (1 - self._random_amount * self._noise[:, None])
        else:
            return image * (1 - self._random_amount * self._noise[None, :])


class Border(Augmentation):

    def __init__(self, amount):
        self.amount = amount
        super().__init__(apply_to_label=False, channels=None)

    def randomize(self):
        self._random_amount = int(np.random.uniform(*self.amount))

    def __call__(self, image):
        new_image = np.zeros_like(image)
        new_image[self._random_amount:-self._random_amount,
        self._random_amount:-self._random_amount] = image[self._random_amount:-self._random_amount,
                                                    self._random_amount:-self._random_amount]

        return new_image
