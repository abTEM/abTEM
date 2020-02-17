import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import map_coordinates


def periodic_smooth_decomposition(image):
    u = image
    v = u2v(u)
    v_fft = cp.fft.fft2(v)
    s = v2s(v_fft)
    s_i = cp.fft.ifft2(s)
    s_f = cp.real(s_i)
    p = u - s_f  # u = p + s
    return p, s_f


def u2v(u):
    v = cp.zeros(u.shape, dtype=u.dtype)
    v[..., 0, :] = u[..., -1, :] - u[..., 0, :]
    v[..., -1, :] = u[..., 0, :] - u[..., -1, :]

    v[..., :, 0] += u[..., :, -1] - u[..., :, 0]
    v[..., :, -1] += u[..., :, 0] - u[..., :, -1]
    return v


def v2s(v_hat):
    M, N = v_hat.shape[-2:]

    q = cp.arange(M).reshape(M, 1).astype(v_hat.dtype)
    r = cp.arange(N).reshape(1, N).astype(v_hat.dtype)

    den = (2 * cp.cos(cp.divide((2 * np.pi * q), M)) + 2 * cp.cos(cp.divide((2 * cp.pi * r), N)) - 4)

    for i in range(len(v_hat.shape) - 2):
        den = cp.expand_dims(den, 0)

    s = v_hat / den
    s[..., 0, 0] = 0
    return s


def periodic_smooth_decomposed_fft(image):
    p, s = periodic_smooth_decomposition(image)
    return cp.fft.fft2(p)


def image_as_polar_representation(image, inner=1, outer=None, symmetry=1, bins_per_symmetry=32):
    center = cp.array(image.shape[-2:]) // 2

    if outer is None:
        outer = (cp.min(center) // 2).item()

    n_radial = outer - inner
    n_angular = (symmetry // 2) * bins_per_symmetry

    radials = cp.linspace(inner, outer, n_radial)
    angles = cp.linspace(0, cp.pi, n_angular)

    polar_coordinates = center[:, None, None] + radials[None, :, None] * cp.array([cp.cos(angles), cp.sin(angles)])[:,
                                                                         None]
    polar_coordinates = polar_coordinates.reshape((2, -1))

    unrolled = map_coordinates(image, polar_coordinates, order=1)
    unrolled = unrolled.reshape((n_radial, n_angular))

    if symmetry > 1:
        unrolled = unrolled.reshape((unrolled.shape[0], symmetry // 2, -1)).sum(1)

    return unrolled


def find_hexagonal_spots(image, lattice_constant=None, min_sampling=None, max_sampling=None, bins_per_symmetry=32,
                         cartesian=False):
    if image.shape[0] != image.shape[1]:
        raise RuntimeError('image is not square')

    n = image.shape[0]

    if (lattice_constant is None) & ((min_sampling is not None) | (max_sampling is not None)):
        raise RuntimeError()

    k = n / lattice_constant * 2 / cp.sqrt(3)
    if min_sampling is None:
        inner = 1
    else:
        inner = int(cp.ceil(max(1, k * min_sampling)))

    if max_sampling is None:
        outer = None
    else:
        outer = int(cp.floor(min(n // 2, k * max_sampling)))

    f = periodic_smooth_decomposed_fft(image)
    f = cp.abs(cp.fft.fftshift(f))

    unrolled = image_as_polar_representation(f, inner=inner, outer=outer, symmetry=6,
                                             bins_per_symmetry=bins_per_symmetry)

    unrolled = unrolled - unrolled.mean(1)[:, None]
    unrolled = unrolled - unrolled.min(1)[:, None]

    radial, angle = cp.unravel_index(cp.argmax(unrolled), unrolled.shape)
    radial = radial + inner + .5
    angle = angle / (bins_per_symmetry * 6) * 2 * np.pi

    if cartesian:
        angles = angle + cp.linspace(0, 2 * np.pi, 6, endpoint=False)
        radial = cp.array(radial)[None]

        return cp.array([cp.cos(angles) * radial + image.shape[0] // 2,
                         cp.sin(angles) * radial + image.shape[0] // 2]).T
    else:
        return radial, angle


def find_hexagonal_sampling(image, lattice_constant, min_sampling=None, max_sampling=None):
    if image.shape[0] != image.shape[1]:
        raise RuntimeError('image is not square')

    n = image.shape[0]

    radial, _ = find_hexagonal_spots(image, lattice_constant, min_sampling, max_sampling)
    return (radial * lattice_constant / n * np.sqrt(3) / 2).item()


class ScaleModel:

    def __init__(self, target_sampling):
        self._target_sampling = target_sampling

    def rescale(self):
        pass

    def predict(self):
        pass


class FourierSpaceScaleModel:

    def __init__(self, crystal_system):
        pass
