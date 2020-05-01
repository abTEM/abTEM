import numpy as np
import cupy as cp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from abtem.cudakernels import interpolate_radial_functions
from abtem.utils import cosine_window, polar_coordinates, get_array_module, get_ndimage_module


def periodic_smooth_decomposition(image):
    xp = get_array_module(image)

    u = image
    v = u2v(u)
    v_fft = xp.fft.fft2(v)
    s = v2s(v_fft)
    s_i = xp.fft.ifft2(s)
    s_f = xp.real(s_i)
    p = u - s_f  # u = p + s
    return p, s_f


def u2v(u):
    xp = get_array_module(u)
    v = xp.zeros(u.shape, dtype=u.dtype)
    v[..., 0, :] = u[..., -1, :] - u[..., 0, :]
    v[..., -1, :] = u[..., 0, :] - u[..., -1, :]

    v[..., :, 0] += u[..., :, -1] - u[..., :, 0]
    v[..., :, -1] += u[..., :, 0] - u[..., :, -1]
    return v


def v2s(v_hat):
    xp = get_array_module(v_hat)
    M, N = v_hat.shape[-2:]

    q = xp.arange(M).reshape(M, 1).astype(v_hat.dtype)
    r = xp.arange(N).reshape(1, N).astype(v_hat.dtype)

    den = (2 * xp.cos(xp.divide((2 * np.pi * q), M)) + 2 * xp.cos(xp.divide((2 * xp.pi * r), N)) - 4)

    for i in range(len(v_hat.shape) - 2):
        den = xp.expand_dims(den, 0)

    s = v_hat / (den + 1e-12)
    s[..., 0, 0] = 0
    return s


def periodic_smooth_decomposed_fft(image):
    xp = get_array_module(image)
    p, s = periodic_smooth_decomposition(image)
    return xp.fft.fft2(p)


def image_as_polar_representation(image, inner=1, outer=None, symmetry=1, bins_per_symmetry=32):
    xp = get_array_module(image)
    ndimage = get_ndimage_module(image)

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
    unrolled = ndimage.map_coordinates(image, polar_coordinates, order=1)
    unrolled = unrolled.reshape((n_radial, n_angular))

    if symmetry > 1:
        unrolled = unrolled.reshape((unrolled.shape[0], symmetry // 2, -1)).sum(1)

    return unrolled


def _window_sum_2d(image, window_shape):
    xp = get_array_module()

    window_sum = xp.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1] - window_sum[:-window_shape[0] - 1])

    window_sum = xp.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1] - window_sum[:, :-window_shape[1] - 1])

    return window_sum


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def normalized_cross_correlation_with_2d_gaussian(image, kernel_size, std):
    xp = get_array_module(image)
    ndimage = get_ndimage_module(image)

    kernel_1d = xp.exp(-(xp.arange(kernel_size) - (kernel_size - 1) / 2) ** 2 / (2 * std ** 2))
    kernel = np.outer(kernel_1d, kernel_1d)
    kernel_mean = kernel.mean()
    kernel_ssd = xp.sum((kernel - kernel_mean) ** 2)

    xcorr = ndimage.convolve(ndimage.convolve(image, kernel_1d[None]), kernel_1d[None].T)

    image_window_sum = _window_sum_2d(image, kernel.shape)
    image_window_sum2 = _window_sum_2d(image ** 2, kernel.shape)

    xcorr = _centered(xcorr, image_window_sum.shape)
    numerator = xcorr - image_window_sum * kernel_mean

    denominator = image_window_sum2
    np.multiply(image_window_sum, image_window_sum, out=image_window_sum)
    np.divide(image_window_sum, np.prod(kernel.shape), out=image_window_sum)
    denominator -= image_window_sum
    denominator *= kernel_ssd
    np.maximum(denominator, 0, out=denominator)
    np.sqrt(denominator, out=denominator)

    response = np.zeros_like(xcorr, dtype=xp.float32)
    mask = denominator >= xp.finfo(xp.float32).eps
    response[mask] = numerator[mask] / (denominator[mask])
    return response #np.float32(mask)


def find_hexagonal_spots(image, lattice_constant=None, min_sampling=None, max_sampling=None, bins_per_symmetry=16,
                         return_cartesian=False):
    xp = get_array_module(image)

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
    f[0, 0] = 0
    f = xp.abs(xp.fft.fftshift(f)) ** 2
    f = (f - f.min()) / (f.max() - f.min())

    peaks = []
    for w in range(3, 11, 1):
        response = normalized_cross_correlation_with_2d_gaussian(f, w, w / 8)
        polar = image_as_polar_representation(response, inner=inner, outer=outer, symmetry=6,
                                              bins_per_symmetry=bins_per_symmetry)
        #polar = polar - polar.mean(axis=1, keepdims=True)
        peak = np.vstack(np.unravel_index((-polar).ravel().argsort()[:5], polar.shape)).T
        peaks.append(peak)

    #import matplotlib.pyplot as plt

    #plt.figure(figsize=(12,12))
    #plt.imshow(np.log(1+1e10*cp.asnumpy(f)))
    #plt.imshow(response[400:-400,400:-400])
    #plt.imshow(cp.asnumpy(polar)[:50])
    #plt.show()

    peaks = cp.asnumpy(xp.vstack(peaks))

    below_center = peaks[:, 1] > bins_per_symmetry / 2
    peaks = np.vstack([peaks,
                       np.array([peaks[below_center][:, 0], peaks[below_center][:, 1] - bins_per_symmetry]).T,
                       np.array([peaks[below_center == 0][:, 0],
                                 peaks[below_center == 0][:, 1] + bins_per_symmetry]).T])

    assignments = fcluster(linkage(pdist(peaks), method='single'), 2, 'distance')
    unique, counts = np.unique(assignments, return_counts=True)
    cluster_centers = np.array([np.mean(peaks[assignments == u], axis=0) for u in unique])

    valid = (cluster_centers[:, 1] > -1e-12) & (cluster_centers[:, 1] < bins_per_symmetry) & (counts > 3)
    cluster_centers = cluster_centers[valid][np.argsort(-counts[valid])][:2]

    cluster_centers[:, 0] += inner - .5
    cluster_centers[:, 1] = (cluster_centers[:, 1] / (bins_per_symmetry * 6) * 2 * np.pi) % (2 * np.pi / 6)

    if len(cluster_centers) > 1:
        radial_ratio = np.min(cluster_centers[:, 0]) / np.max(cluster_centers[:, 0])
        angle_diff = np.max(cluster_centers[:, 1]) - np.min(cluster_centers[:, 1])
        if (np.abs(radial_ratio * np.sqrt(3) - 1) < .1) & (np.abs(angle_diff - np.pi / 6) < np.pi / 10):
            spot_radial, spot_angle = cluster_centers[np.argmin(cluster_centers[:, 0])]
        else:
            spot_radial, spot_angle = cluster_centers[0]

    else:
        spot_radial, spot_angle = cluster_centers[0]

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

    def __init__(self):
        self._spot_positions = None
        self._spot_radial = None
        self._sampling = None
        self._image = None

    def get_spots(self):
        if self._spot_positions is None:
            raise RuntimeError()
        return self._spot_positions

    def predict(self, image):
        raise RuntimeError()

    def __call__(self, image):
        return self.predict(image)


class HexagonalFourierSpaceScaleModel(FourierSpaceScaleModel):

    def __init__(self, lattice_constant, min_sampling=None, max_sampling=None):
        self._lattice_constant = lattice_constant
        self._min_sampling = min_sampling
        self._max_sampling = max_sampling

        super().__init__()

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


def add_margin_to_image(image, margin):
    margin = int(np.ceil(margin))
    height = image.shape[-1] + 2 * margin
    width = image.shape[-2] + 2 * margin
    images, padding = pad_to_size(image, height, width, 16)
    return images, padding


def closest_multiple_ceil(n, m):
    return int(np.ceil(n / m) * m)


def pad_to_size(image, height, width, n=None):
    if n is not None:
        height = closest_multiple_ceil(height, n)
        width = closest_multiple_ceil(width, n)
    xp = get_array_module(image)
    shape = image.shape[-2:]
    up = (height - shape[0]) // 2
    down = height - shape[0] - up
    left = (width - shape[1]) // 2
    right = width - shape[1] - left
    padding = [(up, down), (left, right)]
    images = xp.pad(image, pad_width=padding)
    return images, padding


def rescale(image, scale_factor):
    ndimage = get_ndimage_module(image)
    return ndimage.zoom(image, zoom=scale_factor, order=0)


def threshold_otsu(image, nbins=256):
    xp = get_array_module(image)
    hist, bin_edges = xp.histogram(image.ravel(), nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    weight1 = xp.cumsum(hist)
    weight2 = xp.cumsum(hist[::-1])[::-1]
    mean1 = xp.cumsum(hist * bin_centers) / weight1
    mean2 = (xp.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


def create_band_pass_filter(image, center, width, rolloff=0):
    xp = get_array_module(image)

    width = width / 2

    inner = center - width
    outer = center + width
    r = polar_coordinates(image.shape, xp=xp)

    mask = cosine_window(r, inner, rolloff=rolloff, attenuate='high', xp=xp)
    mask *= cosine_window(r, outer, rolloff=rolloff, attenuate='low', xp=xp)
    mask *= image.sum() / mask.sum()
    return xp.fft.fftshift(mask)


def create_fourier_filter(shape, spots, cutoff, func=None):
    func = lambda x: np.exp(-x ** 2 / .5)
    return interpolate_radial_functions(func, spots, shape, cutoff)
