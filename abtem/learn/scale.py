import numpy as np
from scipy import ndimage
from abtem.utils import ind2sub

def polar_bins(shape, inner, outer, nbins_angular=32, nbins_radial=None):
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

    if outer is None:
        outer = min(f.shape) // 2

    if nbins_radial is None:
        nbins_radial = outer - inner

    bins = polar_bins(f.shape, inner, outer, nbins_angular=nbins_angular, nbins_radial=nbins_radial)

    with np.errstate(divide='ignore', invalid='ignore'):
        unrolled = ndimage.mean(f, bins, range(0, bins.max() + 1))

    unrolled = unrolled.reshape((nbins_angular, nbins_radial))

    for i in range(unrolled.shape[1]):
        y = unrolled[:, i]
        nans = np.isnan(y)
        y[nans] = np.interp(nans.nonzero()[0], (~nans).nonzero()[0], y[~nans], period=len(y))
        unrolled[:, i] = y

    return unrolled


def rotational_average(images):
    power_spec = np.abs(np.fft.fftshift(np.fft.fft2(images)))

    if len(power_spec.shape) == 3:
        power_spec = power_spec.sum(0)

    #import matplotlib.pyplot as plt
    #plt.imshow(np.log(1+power_spec)[200:-200,200:-200])
    #plt.show()


    unrolled = unroll_powerspec(power_spec, inner=1)

    return unrolled.mean(0)


def find_ring(images, a, min_sampling=0.):
    if len(images.shape) == 3:
        shape = images.shape[1:]
    else:
        shape = images.shape

    power_spec = rotational_average(images)

    #import matplotlib.pyplot as plt
    #plt.plot(power_spec)
    #plt.show()

    inner = int(np.floor(min_sampling / a * float(min(shape)) * 2. / np.sqrt(3.))) - 1
    inner = max(0, inner)

    scale = (np.argmax(power_spec[inner:]) + inner + 1 + .5) * a / float(min(shape)) * (np.sqrt(3.) / 2.)
    return scale

def top_n_2d(array, n, margin=0):
    top = np.argsort(array.ravel())[::-1]
    accepted = np.zeros((n, 2), dtype=np.int)
    values = np.zeros(n, dtype=np.int)
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
            values[j] = array[idx[0], idx[1]]
            j += 1

        i += 1
        if i >= array.size - 1:
            break

    return accepted, values
#
#
# def moving_average(x, w):
#     return np.convolve(x, np.ones(1 + 2 * w), 'valid') / w
#
#
def find_circular_spots(power_spec, n, m=1, inner=1, w=1, bins_per_spot=16):
    nbins_angular = n * bins_per_spot

    unrolled = unroll_powerspec(power_spec, inner=inner, nbins_angular=nbins_angular)
    unrolled /= unrolled.mean(axis=0, keepdims=True)
    unrolled /= unrolled.mean(axis=1, keepdims=True)
    unrolled = unrolled.reshape((n, bins_per_spot, unrolled.shape[1])).sum(0)

    #unrolled = unrolled[:, w:-w] / moving_average(unrolled.mean(axis=0), w)
    peaks, intensities = top_n_2d(unrolled, m, bins_per_spot // 4)
    radials, angles = peaks[:, 1], peaks[:, 0]

    angles = (angles + .5) / nbins_angular * 2 * np.pi
    radials = radials + inner + w + .5

    #x = radials[:, None] * np.cos(angles[:, None] + np.linspace(0, 2 * np.pi, n, endpoint=False)[None, :]) + w // 2
    #y = radials[:, None] * np.sin(angles[:, None] + np.linspace(0, 2 * np.pi, n, endpoint=False)[None, :]) + h // 2

    return radials, angles, intensities


def find_hexagonal_scale(image, a=2.46, ratio_tol=.0, angle_tol=0., limiting_regime='high'):
    angle_tol = angle_tol / 180. * np.pi

    power_spec = np.fft.fftshift(np.abs(np.fft.fft2(image)) ** 2)
    if len(power_spec.shape) == 3:
        power_spec = power_spec.sum(0)

    #inner = int(np.ceil(min_scale / a * float(min(power_spec.shape)) * 2. / np.sqrt(3.))) - 1
    #inner = max(1, inner)
    radials, angles, intensities = find_circular_spots(power_spec, 6, m=2, inner=1)

    ordered_angles = np.sort(angles)
    ordered_radials = np.sort(radials)
    ratio = ordered_radials[0] / ordered_radials[1]
    angle_diff = np.diff(ordered_angles)[0]

    if np.isclose(ratio, 1 / np.sqrt(3), atol=ratio_tol) & np.isclose(angle_diff, np.pi / 6, atol=angle_tol):
        scale = np.max(radials) * a / float(min(power_spec.shape)) / 2.
    elif limiting_regime == 'low':
        scale = radials[np.argmax(intensities)] * a / float(min(power_spec.shape)) / 2.
    elif limiting_regime == 'high':
        scale = radials[np.argmax(intensities)] * a / float(min(power_spec.shape)) * (np.sqrt(3.) / 2.)
    else:
        raise RuntimeError()

    return scale
