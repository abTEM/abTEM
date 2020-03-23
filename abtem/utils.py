import numbers

import cupy as cp
import numba as nb
import numpy as np
from ase import units

from abtem.config import DTYPE, COMPLEX_DTYPE


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


# def complex_exponential(x):
#     return ne.evaluate('exp(1.j * x)')


def cosine_window(x, cutoff, rolloff, attenuate='high', xp=np):
    rolloff *= cutoff
    if attenuate == 'high':
        array = .5 * (1 + xp.cos(xp.pi * (x - cutoff - rolloff) / rolloff))
        array[x < cutoff] = 0.
        array = xp.where(x < cutoff + rolloff, array, xp.ones_like(x, dtype=DTYPE))
    elif attenuate == 'low':
        array = .5 * (1 + xp.cos(xp.pi * (x - cutoff + rolloff) / rolloff))
        array[x > cutoff] = 0.
        array = xp.where(x > cutoff - rolloff, array, xp.ones_like(x, dtype=DTYPE))
    else:
        raise RuntimeError('attenuate must be "high" or "low"')

    return array


def complex_exponential(x):
    df_exp = np.empty(x.shape, dtype=COMPLEX_DTYPE)
    trig_buf = np.cos(x)
    df_exp.real[:] = trig_buf
    np.sin(x, out=trig_buf)
    df_exp.imag[:] = trig_buf
    return df_exp


@nb.vectorize([nb.float32(nb.complex64), nb.float64(nb.complex128)])
def abs2(x):
    return x.real ** 2 + x.imag ** 2


def coordinates_in_disc(radius, shape=None):
    cols = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.int32)
    cols[:] = np.linspace(0, 2 * radius, 2 * radius + 1) - radius
    rows = cols.copy().T
    r2 = rows ** 2 + cols ** 2
    inside = r2 < radius ** 2

    if shape is None:
        return rows[inside], cols[inside]
    else:
        return rows[inside] * shape[0] + cols[inside]


def polar_coordinates(shape, return_azimuth=False, xp=np):
    x = xp.arange(shape[0], dtype=xp.float32) - shape[0] // 2
    y = xp.arange(shape[1], dtype=xp.float32) - shape[1] // 2
    r = xp.sqrt(x[:, None] ** 2 + y[None] ** 2)
    if return_azimuth:
        # TODO : implement azimuthal coordinates
        raise NotImplementedError()
    return r


def squared_norm(x, y):
    return x.reshape((-1, 1)) ** 2 + y.reshape((1, -1)) ** 2


def sub2ind2d(rows, cols, shape):
    return rows * shape[0] + cols


def ind2sub2d(array_shape, ind):
    rows = ind // array_shape[1]
    cols = ind % array_shape[1]
    return (rows, cols)


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


class BatchGenerator:

    def __init__(self, n_items, max_batch_size):
        self._n_items = n_items
        self._n_batches = (n_items + (-n_items % max_batch_size)) // max_batch_size
        self._batch_size = (n_items + (-n_items % self.n_batches)) // self.n_batches

    @property
    def n_batches(self):
        return self._n_batches

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def n_items(self):
        return self._n_items

    def generate(self):
        batch_start = 0
        for i in range(self.n_batches):
            batch_end = batch_start + self.batch_size
            if i == self.n_batches - 1:
                yield batch_start, self.n_items - batch_end + self.batch_size
            else:
                yield batch_start, self.batch_size

            batch_start = batch_end


def view_as_windows(arr_in, window_shape, step):
    if not isinstance(arr_in, (np.ndarray, cp.ndarray)):
        raise TypeError("`arr_in` must be a numpy ndarray")

    xp = cp.get_array_module(arr_in)

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim

    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = xp.array(arr_in.shape)
    window_shape = xp.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = xp.array(arr_in.strides)

    indexing_strides = xp.asarray(arr_in[slices].strides)

    win_indices_shape = (((xp.array(arr_in.shape) - xp.array(window_shape)) // xp.array(step)) + 1)

    new_shape = tuple(xp.concatenate((win_indices_shape, window_shape)))
    strides = tuple(xp.concatenate((indexing_strides, window_strides)))

    arr_out = xp.lib.stride_tricks.as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out



