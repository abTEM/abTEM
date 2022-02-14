from typing import Tuple, Union

import numpy as np
from numba import prange, jit

from abtem.core.backend import get_array_module
from abtem.core.grid import polar_spatial_frequencies


def _label_to_index(labels, max_label=None):
    xp = get_array_module(labels)
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = xp.arange(0, len(labels) + 1)[labels_order]

    if max_label is None:
        max_label = np.max(labels)

    index = xp.arange(0, max_label + 1)
    lo = xp.searchsorted(sorted_labels, index, side='left')
    hi = xp.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield indices[l:h]


def polar_detector_bins(gpts: Tuple[int, int],
                        sampling: Tuple[float, float],
                        inner: float,
                        outer: float,
                        nbins_radial: int,
                        nbins_azimuthal: int,
                        rotation: float = 0.,
                        offset: Tuple[float, float] = (0., 0.),
                        fftshift=False,
                        return_indices=False,
                        ) -> np.ndarray:
    """
    Create an array of labels for the regions of a given detector geometry.

    Parameters
    ----------
    gpts : two int
        Number of grid points describing the detector regions.
    angular_sampling : two float
        Angular sampling of the discretized detector regions in radians.
    inner : float
        Inner boundary of the detector regions [rad].
    outer : float
        Outer boundary of the detector regions [rad].
    nbins_radial : int
        Number of radial detector bins.
    nbins_azimuthal
        Number of azimuthal detector bins.

    Returns
    -------
    2d array
        Array of integer labels representing the detector regions.
    """

    alpha, phi = polar_spatial_frequencies(gpts, (1 / sampling[0] / gpts[0], 1 / sampling[1] / gpts[1]))
    phi = (phi + rotation) % (2 * np.pi)

    radial_bins = -np.ones(gpts, dtype=int)
    valid = (alpha >= inner) & (alpha < outer)

    radial_bins[valid] = (nbins_radial * (alpha[valid] - inner) / (outer - inner))

    # import matplotlib.pyplot as plt
    # #print(inner)
    # #plt.imshow(radial_bins[:10,:10])
    # plt.imshow(valid[:10, :10])
    # plt.colorbar()
    # plt.show()

    angular_bins = np.floor(nbins_azimuthal * (phi / (2 * np.pi)))
    angular_bins = np.clip(angular_bins, 0, nbins_azimuthal - 1).astype(int)

    bins = -np.ones(gpts, dtype=int)
    bins[valid] = angular_bins[valid] + radial_bins[valid] * nbins_azimuthal

    if np.any(np.array(offset) != 0.):
        offset = (int(round(offset[0] / sampling[0])), int(round(offset[1] / sampling[1])))

        if (abs(offset[0]) > bins[0]) or (abs(offset[1]) > bins[1]):
            raise RuntimeError('Detector offset exceeds maximum detected angle.')

        bins = np.roll(bins, offset, (0, 1))

    if fftshift:
        bins = np.fft.fftshift(bins)

    if return_indices:
        indices = []
        for i in _label_to_index(bins, nbins_radial * nbins_azimuthal - 1):
            indices.append(i)
        return indices
    else:
        return bins


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def sum_run_length_encoded(array, result, separators):
    for x in prange(result.shape[1]):
        for i in range(result.shape[0]):
            for j in range(separators[x], separators[x + 1]):
                result[i, x] += array[i, j]
