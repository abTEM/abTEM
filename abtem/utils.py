import time

import numpy as np
from ase import units

from abtem.device import get_array_module


def energy2mass(energy):
    """
    Calculate relativistic mass from energy.
    :param energy: Energy in electron volt
    :type energy: float
    :return: Relativistic mass in kg
    :rtype: float
    """
    return (1 + units._e * energy / (units._me * units._c ** 2)) * units._me


def energy2wavelength(energy):
    """
    Calculate relativistic de Broglie wavelength from energy.
    :param energy: Energy in electron volt
    :type energy: float
    :return: Relativistic de Broglie wavelength in Angstrom.
    :rtype: float
    """
    return units._hplanck * units._c / np.sqrt(
        energy * (2 * units._me * units._c ** 2 / units._e + energy)) / units._e * 1.e10


def energy2sigma(energy):
    """
    Calculate interaction parameter from energy.
    :param energy: Energy in electron volt.
    :type energy: float
    :return: Interaction parameter in 1 / (Angstrom * eV).
    :rtype: float
    """
    return (2 * np.pi * energy2mass(energy) * units.kg * units._e * units.C * energy2wavelength(energy) / (
            units._hplanck * units.s * units.J) ** 2)


def polargrid(x, y):
    xp = get_array_module(x)
    alpha = xp.sqrt(x.reshape((-1, 1)) ** 2 + y.reshape((1, -1)) ** 2)
    phi = xp.arctan2(x.reshape((-1, 1)), y.reshape((1, -1)))
    return alpha, phi


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
                v.append(pp + 1)
            else:
                v.append(pp)
        return v


def label_to_index_generator(labels, first_label=0):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(first_label, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield np.sort(indices[l:h])


class Bar:

    def __init__(self, name, total, enable=True, parent=None):
        self._name = name
        self._total = total
        self._current = 0
        self._last_t = time.time()
        self._elapsed = 0
        self._last_finished = 0
        self._output = f"{self._name}: 0 of {self._total} (0 %) [Elapsed N/A, ETA N/A, N/A]"
        self._enable = enable
        self._parent = parent

    def reset(self):
        self._current = 0
        self._last_t = time.time()
        self._elapsed = 0
        self._last_finished = 0

    def update(self, n):
        t = time.time()

        self._current += n
        dt = t - self._last_t

        if (dt >= .1) or (self._current == self._total):
            self._elapsed += dt
            percent = round(self._current / self._total * 100)
            iter_per_sec = (self._current - self._last_finished) / (dt + 1e-7)
            eta = (self._total - self._current) / iter_per_sec
            self._output = f"{self._name}: {self._current} of {self._total} ({percent} %) "
            self._output += f"[Elapsed {self._elapsed:.1f} s, ETA {eta:.1f} s, {iter_per_sec:.1f} / s]"
            self._last_t = time.time()
            self._last_finished = self._current

    @property
    def output(self):
        if self._parent:
            if self._parent._enable:
                return ' <- '.join((self._parent.output, self._output))

        return self._output

    def print_bar(self):
        if not self._enable:
            return

        print(f'{self.output:<{128}}\r', end="", flush=True)
