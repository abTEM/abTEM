import time

import numpy as np
from ase import units

from abtem.device import get_array_module
from tqdm.auto import tqdm


def energy2mass(energy):
    """
    Calculate relativistic mass from energy.
    :param energy: Energy [eV].
    :type energy: float
    :return: Relativistic mass [kg].
    :rtype: float
    """
    return (1 + units._e * energy / (units._me * units._c ** 2)) * units._me


def energy2wavelength(energy):
    """
    Calculate relativistic de Broglie wavelength from energy.
    :param energy: Energy [eV].
    :type energy: float
    :return: Relativistic de Broglie wavelength [Å].
    :rtype: float
    """
    return units._hplanck * units._c / np.sqrt(
        energy * (2 * units._me * units._c ** 2 / units._e + energy)) / units._e * 1.e10


def energy2sigma(energy):
    """
    Calculate interaction parameter from energy.
    :param energy: Energy [Å].
    :type energy: float
    :return: Interaction parameter [1 / (Å * eV)].
    :rtype: float
    """
    return (2 * np.pi * energy2mass(energy) * units.kg * units._e * units.C * energy2wavelength(energy) / (
            units._hplanck * units.s * units.J) ** 2)


def spatial_frequencies(gpts, sampling):
    return tuple(np.fft.fftfreq(n, d).astype(np.float32) for n, d in zip(gpts, sampling))


def coordinates(gpts, extent, endpoint):
    return tuple(np.linspace(0, l, n, endpoint=endpoint, dtype=np.float32) for n, l in zip(gpts, extent))


def polargrid(x, y):
    xp = get_array_module(x)
    alpha = xp.sqrt(x.reshape((-1, 1)) ** 2 + y.reshape((1, -1)) ** 2)
    phi = xp.arctan2(x.reshape((-1, 1)), y.reshape((1, -1)))
    return alpha, phi


def cosine_window(x, cutoff, rolloff, attenuate='high'):
    xp = get_array_module(x)

    rolloff *= cutoff
    if attenuate == 'high':
        array = .5 * (1 + xp.cos(xp.pi * (x - cutoff - rolloff) / rolloff))
        array[x < cutoff] = 0.
        array = xp.where(x < cutoff + rolloff, array, xp.ones_like(x, dtype=xp.float32))
    elif attenuate == 'low':
        array = .5 * (1 + xp.cos(xp.pi * (x - cutoff + rolloff) / rolloff))
        array[x > cutoff] = 0.
        array = xp.where(x > cutoff - rolloff, array, xp.ones_like(x, dtype=xp.float32))
    else:
        raise RuntimeError('Attenuate must be either "high" or "low"')

    return array


def split_integer(n, m):
    if n < m:
        raise RuntimeError()

    elif n % m == 0:
        return [n // m] * m
    else:
        v = []
        zp = m - (n % m)
        pp = n // m
        for i in range(m):
            if i >= zp:
                v = [pp + 1] + v
            else:
                v = [pp] + v
        return v


def label_to_index_generator(labels, first_label=0):
    xp = get_array_module(labels)
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = xp.arange(0, len(labels) + 1)[labels_order]
    index = xp.arange(first_label, xp.max(labels) + 1)
    lo = xp.searchsorted(sorted_labels, index, side='left')
    hi = xp.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield indices[l:h]


class ProgressBar:
    def __init__(self, **kwargs):
        self._tqdm = tqdm(**kwargs)

    @property
    def tqdm(self):
        return self._tqdm

    @property
    def disable(self):
        return self.tqdm.disable

    def update(self, n):
        if not self.disable:
            self.tqdm.update(n)

    def reset(self):
        if not self.disable:
            self.tqdm.reset()

    def refresh(self):
        if not self.disable:
            self.tqdm.refresh()

    def close(self):
        self.tqdm.close()
