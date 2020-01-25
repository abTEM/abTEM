import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from skimage import exposure

from abtem.utils import ind2sub
import functools


# @functools.lru_cache(maxsize=32)
def polar_bins(shape, inner=1, outer=None, nbins_angular=32, nbins_radial=None):
    if outer is None:
        outer = min(shape) // 2
    if nbins_radial is None:
        nbins_radial = outer - inner
    sx, sy = shape
    X, Y = np.ogrid[0:sx, 0:sy]

    r = np.hypot(X - sx / 2, Y - sy / 2)
    radial_bins = -np.ones(shape, dtype=int)
    valid = (r > inner) & (r < outer)
    radial_bins[valid] = nbins_radial * (r[valid] - inner) / (outer - inner)

    angles = np.arctan2(X - sx // 2, Y - sy // 2) % (2 * np.pi)

    angular_bins = np.floor(nbins_angular * (angles / (2 * np.pi)))
    angular_bins = np.clip(angular_bins, 0, nbins_angular - 1).astype(np.int)

    bins = -np.ones(shape, dtype=int)
    bins[valid] = angular_bins[valid] * nbins_radial + radial_bins[valid]
    return bins


def unroll_powerspec(f, inner=1, outer=None, nbins_angular=64, nbins_radial=None):
    if f.shape[0] != f.shape[1]:
        raise RuntimeError()

    bins = polar_bins(f.shape, inner, outer, nbins_angular=nbins_angular, nbins_radial=nbins_radial)

    with np.errstate(divide='ignore', invalid='ignore'):
        unrolled = ndimage.mean(f, bins, range(0, bins.max() + 1))

    unrolled = unrolled.reshape((nbins_angular, -1))

    for i in range(unrolled.shape[1]):
        y = unrolled[:, i]
        nans = np.isnan(y)
        y[nans] = np.interp(nans.nonzero()[0], (~nans).nonzero()[0], y[~nans], period=len(y))
        unrolled[:, i] = y

    return unrolled


def top_n_2d(array, n, margin=0):
    top = np.argsort(array.ravel())[::-1]
    accepted = np.zeros((n, 2), dtype=np.int)
    marked = np.zeros((array.shape[0] + 2 * margin, array.shape[1] + 2 * margin), dtype=np.bool_)
    i = 0
    j = 0
    while j < n:
        idx = ind2sub(array.shape, top[i])

        if marked[idx[0] + margin, idx[1] + margin] == False:
            marked[idx[0]:idx[0] + 2 * margin, idx[1]:idx[1] + 2 * margin] = True
            marked[margin:2 * margin] += marked[-margin:]
            marked[-2 * margin:-margin] += marked[:margin]

            accepted[j] = idx
            j += 1

        i += 1
        if i >= array.size - 1:
            break

    return accepted


def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1


def fourier_padding(N, k):
    m = np.ones(N)
    m[:k] = np.sin(np.linspace(-np.pi / 2, np.pi / 2, k)) / 2 + .5
    m[-k:] = np.sin(-np.linspace(-np.pi / 2, np.pi / 2, k)) / 2 + .5
    return m


def fourier_padding_2d(shape, k):
    return fourier_padding(shape[0], k)[:, None] * fourier_padding(shape[1], k)[None]


def fixed_fft2d(image):
    # image = ((image - image.min()) / image.ptp() * 255).astype(np.uint16)
    # image = exposure.equalize_adapthist(image, clip_limit=.03)
    image = image * fourier_padding_2d(image.shape[1:], image.shape[1] // 4)[None]
    return np.fft.fftshift(np.abs(np.fft.fft2(image)) ** 2).sum(0)


def find_hexagonal_sampling(image, a, min_sampling, bins_per_spot=16):
    if len(image.shape) == 2:
        image = image[None]

    if image.shape[1] != image.shape[2]:
        raise RuntimeError('square image required')

    inner = max(1, int(np.ceil(min_sampling / a * float(min(image.shape[1:])) * 2. / np.sqrt(3.))) - 1)
    outer = min(image.shape[1:]) // 2

    if inner >= outer:
        raise RuntimeError('min. sampling too large')

    f = fixed_fft2d(image)

    nbins_angular = 6 * bins_per_spot

    unrolled = unroll_powerspec(f, inner, outer=None, nbins_angular=nbins_angular, nbins_radial=None)

    # unrolled = unrolled.reshape((6, bins_per_spot, unrolled.shape[1])).sum(0)
    #
    # normalized = unrolled / savgol_filter(unrolled.mean(0), 5, 1, mode='nearest')
    #
    # peaks = top_n_2d(normalized, 5, 2)
    # intensities = unrolled[peaks[:, 0], peaks[:, 1]]
    # angle, radial = peaks[np.argmax(intensities)]
    #
    # # angle = (angle + .5) / nbins_angular * 2 * np.pi
    # radial = radial + inner + .5
    #
    # return radial * a / float(min(f.shape)) * (np.sqrt(3.) / 2.)
