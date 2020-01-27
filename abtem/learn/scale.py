from bisect import bisect_left

import numpy as np
import torch
import functools


def polar_labels(shape, inner=1, outer=None, nbins_angular=32, nbins_radial=None):
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


def generate_indices(labels, first_label=0):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(first_label, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield np.sort(indices[l:h])


@functools.lru_cache(maxsize=1)
def polar_indices(shape, inner, outer, nbins_angular):
    labels = polar_labels(shape, inner=inner, outer=outer, nbins_angular=nbins_angular)

    indices = np.zeros((labels.max() + 1, nbins_angular), dtype=np.int)
    weights = np.zeros((labels.max() + 1, nbins_angular))
    lengths = np.zeros((labels.max() + 1,), dtype=np.int)

    for j, i in enumerate(generate_indices(labels, first_label=0)):
        if len(i) > 0:
            indices[j, :len(i)] = i
            weights[j, :len(i)] = 1 / len(i)
            lengths[j] = len(i)

    indices = indices.reshape((nbins_angular, -1, nbins_angular))
    weights = weights.reshape((nbins_angular, -1, nbins_angular))
    lengths = lengths.reshape((nbins_angular, -1))
    nans = lengths == 0

    for i in range(indices.shape[0]):
        k = np.where(nans[:, i] == 0)[0]
        for j in np.where(nans[:, i])[0]:
            idx = bisect_left(k, j)
            idx = idx % len(k)

            l1 = lengths[k[idx - 1], i]
            l2 = lengths[k[idx], i]

            indices[j, i, :l1] = indices[k[idx - 1], i, :l1]
            indices[j, i, l1:l1 + l2] = indices[k[idx], i, :l2]

            d1 = min(abs(k[idx - 1] - j), abs((nbins_angular - k[idx - 1] + j)))
            d2 = min(abs(k[idx] - j), abs((-nbins_angular - k[idx] + j)))

            weights[j, i, :l1] = 1 / d1
            weights[j, i, l1:l1 + l2] = 1 / d2
            weights[j, i, :l1 + l2] /= weights[j, i, :l1 + l2].sum()

    indices = indices[:, :, :np.max(lengths)]
    weights = weights[:, :, :np.max(lengths)]

    return indices, weights


def roll(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def fftshift2d(x):
    for dim in range(1, len(x.size())):
        n_shift = x.size(dim) // 2
        if x.size(dim) % 2 != 0:
            n_shift += 1
        x = roll(x, axis=dim, n=n_shift)
    return x


def soft_border(shape, k):
    def f(N, k):
        mask = torch.ones(N)
        # print(torch.linspace(-np.pi / 2, np.pi / 2, k))
        mask[:k] = torch.sin(torch.linspace(-np.pi / 2, np.pi / 2, k)) / 2 + .5
        mask[-k:] = torch.sin(-torch.linspace(-np.pi / 2, np.pi / 2, k)) / 2 + .5

        return mask

    return f(shape[0], k)[:, None] * f(shape[1], k)[None]


def nms(array, n, margin=0):
    top = torch.argsort(array.view(-1), descending=True)
    accepted = torch.zeros((n, 2), dtype=np.long)
    marked = torch.zeros((array.shape[0] + 2 * margin, array.shape[1] + 2 * margin), dtype=torch.bool)

    i = 0
    j = 0
    while j < n:
        idx = torch.tensor((top[i] // array.shape[1], top[i] % array.shape[1]))

        if marked[idx[0] + margin, idx[1] + margin] == False:
            marked[idx[0]:idx[0] + 2 * margin, idx[1]:idx[1] + 2 * margin] = True
            marked[margin:2 * margin] += marked[-margin:]
            marked[-2 * margin:-margin] += marked[:margin]

            accepted[j] = idx
            j += 1

        i += 1
        if i >= torch.numel(array) - 1:
            break

    return accepted


def find_hexagonal_sampling(image, a, min_sampling, bins_per_spot=16):
    if len(image.shape) == 2:
        image = image[None]

    if image.shape[1] != image.shape[2]:
        raise RuntimeError('square image required')

    N = image.shape[1]

    inner = max(1, int(np.ceil(min_sampling / a * float(N) * 2. / np.sqrt(3.))) - 1)
    outer = N // 2

    if inner >= outer:
        raise RuntimeError('min. sampling too large')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nbins_angular = 6 * bins_per_spot
    indices, weights = polar_indices(image.shape[1:], inner=inner, outer=outer, nbins_angular=nbins_angular)

    indices = torch.tensor(indices, dtype=torch.long).to(device)
    weights = torch.tensor(weights).to(device)

    complex_image = torch.zeros(tuple(image.shape) + (2,), dtype=torch.float32, device=device)
    complex_image[..., 0] = image * soft_border(image.shape[1:], N // 4).to(device)[None]

    f = torch.sum(torch.fft(complex_image, 2) ** 2, axis=-1)
    f = fftshift2d(f)
    unrolled = (f.view(-1)[indices] * weights).sum(-1)
    unrolled = unrolled.view((6, -1, unrolled.shape[1])).sum(0)

    normalized = unrolled / unrolled.mean(0)
    peaks = nms(normalized, 5, 3)

    intensities = unrolled[peaks[:, 0], peaks[:, 1]]
    angle, r = peaks[torch.argmax(intensities)]
    r = r + inner + .5

    return r * a / float(min(f.shape[1:])) * (np.sqrt(3.) / 2.)
