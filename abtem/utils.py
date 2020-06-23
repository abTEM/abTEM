import numbers

import numpy as np
from ase import units
from scipy import ndimage
import pyfftw
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







# @nb.cuda.jit([nb.float32(nb.complex64), nb.float64(nb.complex128)], target='cuda')
# def abs2_gpu(x):
#    return x.real ** 2 + x.imag ** 2


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


def create_fftw_objects(array, allow_new_plan=True):
    """
    Creates FFTW object for forward and backward Fourier transforms. The input array object will be transformed in place.
    The function tries to retrieve plans from wisdom only. If no plan exists for the input array, a new plan is created.

    Parameters
    ----------
    array : ndarray
        Array to be transformed. 2 dimensions or greater.
    allow_new_plan : bool
        If true allow creation of new plan, otherwise, raise an exception
    """

    try:
        fftw_forward = pyfftw.FFTW(array, array, axes=(-1, -2), threads=12, flags=('FFTW_WISDOM_ONLY',))
        fftw_backward = pyfftw.FFTW(array, array, axes=(-1, -2), direction='FFTW_BACKWARD', threads=12,
                                    flags=('FFTW_WISDOM_ONLY',))

        return fftw_forward, fftw_backward

    except RuntimeError as e:
        if ('No FFTW wisdom is known for this plan.' != str(e)) or (not allow_new_plan):
            raise

        dummy = np.zeros_like(array)  # this is necessary because FFTW overwrites input arrays
        pyfftw.FFTW(dummy, dummy, axes=(-1, -2), threads=12, flags=('FFTW_MEASURE',))
        pyfftw.FFTW(dummy, dummy, axes=(-1, -2), direction='FFTW_BACKWARD', threads=12, flags=('FFTW_MEASURE',))
        return create_fftw_objects(array, allow_new_plan=False)


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
