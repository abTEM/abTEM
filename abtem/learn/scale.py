try:
    import cupy as xp
    from cupyx.scipy.ndimage import map_coordinates, zoom
except:
    import numpy as xp
    from scipy.ndimage import map_coordinates, zoom

import numpy as np

from abtem.cudakernels import interpolate_radial_functions
from abtem.learn.postprocess import NonMaximumSuppression

def periodic_smooth_decomposition(image):
    u = image
    v = u2v(u)
    v_fft = xp.fft.fft2(v)
    s = v2s(v_fft)
    s_i = xp.fft.ifft2(s)
    s_f = xp.real(s_i)
    p = u - s_f  # u = p + s
    return p, s_f


def u2v(u):
    v = xp.zeros(u.shape, dtype=u.dtype)
    v[..., 0, :] = u[..., -1, :] - u[..., 0, :]
    v[..., -1, :] = u[..., 0, :] - u[..., -1, :]

    v[..., :, 0] += u[..., :, -1] - u[..., :, 0]
    v[..., :, -1] += u[..., :, 0] - u[..., :, -1]
    return v


def v2s(v_hat):
    M, N = v_hat.shape[-2:]

    q = xp.arange(M).reshape(M, 1).astype(v_hat.dtype)
    r = xp.arange(N).reshape(1, N).astype(v_hat.dtype)

    den = (2 * xp.cos(xp.divide((2 * np.pi * q), M)) + 2 * xp.cos(xp.divide((2 * xp.pi * r), N)) - 4)

    for i in range(len(v_hat.shape) - 2):
        den = xp.expand_dims(den, 0)

    s = v_hat / den
    s[..., 0, 0] = 0
    return s


def periodic_smooth_decomposed_fft(image):
    p, s = periodic_smooth_decomposition(image)
    return xp.fft.fft2(p)


def image_as_polar_representation(image, inner=1, outer=None, symmetry=1, bins_per_symmetry=32):
    center = xp.array(image.shape[-2:]) // 2

    if outer is None:
        outer = (xp.min(center) // 2).item()

    n_radial = outer - inner
    n_angular = (symmetry // 2) * bins_per_symmetry

    radials = xp.linspace(inner, outer, n_radial)
    angles = xp.linspace(0, xp.pi, n_angular)

    polar_coordinates = center[:, None, None] + radials[None, :, None] * xp.array([xp.cos(angles), xp.sin(angles)])[:,
                                                                         None]
    polar_coordinates = polar_coordinates.reshape((2, -1))

    unrolled = map_coordinates(image, polar_coordinates, order=1)
    unrolled = unrolled.reshape((n_radial, n_angular))

    if symmetry > 1:
        unrolled = unrolled.reshape((unrolled.shape[0], symmetry // 2, -1)).sum(1)

    return unrolled


def find_hexagonal_spots(image, lattice_constant=None, min_sampling=None, max_sampling=None, bins_per_symmetry=32,
                         return_cartesian=False):
    if image.shape[0] != image.shape[1]:
        raise RuntimeError('image is not square')

    n = image.shape[0]

    if (lattice_constant is None) & ((min_sampling is not None) | (max_sampling is not None)):
        raise RuntimeError()

    k = n / lattice_constant * 2 / xp.sqrt(3)
    if min_sampling is None:
        inner = 1
    else:
        inner = int(xp.ceil(max(1, k * min_sampling)))

    if max_sampling is None:
        outer = None
    else:
        outer = int(xp.floor(min(n // 2, k * max_sampling)))

    f = periodic_smooth_decomposed_fft(image)
    f = xp.abs(xp.fft.fftshift(f)) ** 2

    unrolled = image_as_polar_representation(f, inner=inner, outer=outer, symmetry=6,
                                             bins_per_symmetry=bins_per_symmetry)

    # unrolled = unrolled - unrolled.mean(1)[:, None]
    # unrolled = unrolled - unrolled.min(1)[:, None]
    normalized = unrolled / (unrolled.mean(1, keepdims=True) + 1)
    normalized = normalized - normalized.mean(0, keepdims=True)  # [:, None]

    nms = NonMaximumSuppression(4, 0., max_num_maxima=2)
    maxima = xp.asarray(nms.predict(normalized))

    #import matplotlib.pyplot as plt
    #plt.imshow(normalized.T)
    #plt.plot(*maxima.T, 'rx')
    #plt.show()

    spot_radial, spot_angle = maxima[xp.argmax(unrolled[maxima[:, 0], maxima[:, 1]])]
    # radial, angle = xp.unravel_index(xp.argmax(unrolled), unrolled.shape)
    spot_radial = spot_radial + inner + .5
    spot_angle = spot_angle / (bins_per_symmetry * 6) * 2 * np.pi

    if return_cartesian:
        angles = spot_angle + xp.linspace(0, 2 * np.pi, 6, endpoint=False)
        radial = xp.array(spot_radial)[None]
        return spot_radial, spot_angle, xp.array([xp.cos(angles) * radial + image.shape[0] // 2,
                                                  xp.sin(angles) * radial + image.shape[0] // 2]).T
    else:
        return spot_radial, spot_angle


def find_hexagonal_sampling(image, lattice_constant, min_sampling=None, max_sampling=None):
    if image.shape[0] != image.shape[1]:
        raise RuntimeError('image is not square')

    n = image.shape[0]

    radial, _ = find_hexagonal_spots(image, lattice_constant, min_sampling, max_sampling)
    return (radial * lattice_constant / n * np.sqrt(3) / 2).item()


class FourierSpaceScaleModel:

    def __init__(self, target_sampling):
        self._target_sampling = target_sampling
        self._spot_positions = None
        self._spot_radial = None
        self._sampling = None
        self._image = None

    def get_spots(self):
        if self._spot_positions is None:
            raise RuntimeError()
        return self._spot_positions

    def create_fourier_filter(self, function):
        return interpolate_radial_functions(function, self._spot_positions, self._image.shape[-2:],
                                            int(self._spot_radial))

    def rescale(self, images):
        return zoom(images, zoom=self._sampling / self._target_sampling, order=1)


class HexagonalFourierSpaceScaleModel(FourierSpaceScaleModel):

    def __init__(self, lattice_constant, target_sampling=None, min_sampling=None, max_sampling=None):
        self._lattice_constant = lattice_constant
        self._min_sampling = min_sampling
        self._max_sampling = max_sampling

        super().__init__(target_sampling=target_sampling)

    def _find_spots(self, image):
        self._spot_radial, _, self._spot_positions = find_hexagonal_spots(image, self._lattice_constant,
                                                                          min_sampling=self._min_sampling,
                                                                          max_sampling=self._max_sampling,
                                                                          return_cartesian=True)

    def predict(self, image):
        self._find_spots(image)
        self._image = image
        n = image.shape[0]
        self._sampling = (self._spot_radial * self._lattice_constant / n * np.sqrt(3) / 2).item()
        return self._sampling
