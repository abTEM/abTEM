from collections import defaultdict

#import numexpr as ne
import numpy as np
from ase import units
from numba import njit


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


def scherzer_defocus(Cs, energy):
    return 1.2 * np.sign(Cs) * np.sqrt(np.abs(Cs) * energy2wavelength(energy))


def polar2cartesian(polar):
    polar = defaultdict(lambda: 0, polar)

    cartesian = {}
    cartesian['C10'] = polar['C10']
    cartesian['C12a'] = - polar['C12'] * np.cos(2 * polar['phi12'])
    cartesian['C12b'] = polar['C12'] * np.sin(2 * polar['phi12'])
    cartesian['C21a'] = polar['C21'] * np.sin(polar['phi21'])
    cartesian['C21b'] = polar['C21'] * np.cos(polar['phi21'])
    cartesian['C23a'] = - polar['C23'] * np.sin(3 * polar['phi23'])
    cartesian['C23b'] = polar['C23'] * np.cos(3 * polar['phi23'])
    cartesian['C30'] = polar['C30']
    cartesian['C32a'] = - polar['C32'] * np.cos(2 * polar['phi32'])
    cartesian['C32b'] = polar['C32'] * np.cos(np.pi / 2 - 2 * polar['phi32'])
    cartesian['C34a'] = polar['C34'] * np.cos(-4 * polar['phi34'])
    K = np.sqrt(3 + np.sqrt(8.))
    cartesian['C34b'] = 1 / 4. * (1 + K ** 2) ** 2 / (K ** 3 - K) * polar['C34'] * np.cos(
        4 * np.arctan(1 / K) - 4 * polar['phi34'])

    return cartesian


def cartesian2polar(cartesian):
    cartesian = defaultdict(lambda: 0, cartesian)

    polar = {}
    polar['C10'] = cartesian['C10']
    polar['C12'] = - np.sqrt(cartesian['C12a'] ** 2 + cartesian['C12b'] ** 2)
    polar['phi12'] = - np.arctan2(cartesian['C12b'], cartesian['C12a']) / 2.
    polar['C21'] = np.sqrt(cartesian['C21a'] ** 2 + cartesian['C21b'] ** 2)
    polar['phi21'] = np.arctan2(cartesian['C21a'], cartesian['C21b'])
    polar['C23'] = np.sqrt(cartesian['C23a'] ** 2 + cartesian['C23b'] ** 2)
    polar['phi23'] = -np.arctan2(cartesian['C23a'], cartesian['C23b']) / 3.
    polar['C30'] = cartesian['C30']
    polar['C32'] = -np.sqrt(cartesian['C32a'] ** 2 + cartesian['C32b'] ** 2)
    polar['phi32'] = -np.arctan2(cartesian['C32b'], cartesian['C32a']) / 2.
    polar['C34'] = np.sqrt(cartesian['C34a'] ** 2 + cartesian['C34b'] ** 2)
    polar['phi34'] = np.arctan2(cartesian['C34b'], cartesian['C34a']) / 4

    return polar


#def complex_exponential(x):
#    return ne.evaluate('exp(1.j * x)')


def complex_exponential(x):
    df_exp = np.empty(x.shape, dtype=np.complex64)
    trig_buf = np.cos(x)
    df_exp.real[:] = trig_buf
    np.sin(x, out=trig_buf)
    df_exp.imag[:] = trig_buf
    return df_exp


def squared_norm(x, y):
    return x.reshape((-1, 1)) ** 2 + y.reshape((1, -1)) ** 2


def fftfreq(grid):
    grid.check_is_grid_defined()
    return tuple(np.fft.fftfreq(gpts, sampling) for gpts, sampling in zip(grid.gpts, grid.sampling))


def linspace(grid):
    grid.check_is_grid_defined()
    return tuple(np.linspace(0, extent, gpts, endpoint=grid.endpoint) for gpts, extent in zip(grid.gpts, grid.extent))


def semiangles(grid_and_energy):
    wavelength = grid_and_energy.wavelength
    return (np.fft.fftfreq(gpts, sampling) * wavelength for gpts, sampling in
            zip(grid_and_energy.gpts, grid_and_energy.sampling))


@njit
def sub2ind(rows, cols, array_shape):
    return rows * array_shape[1] + cols


@njit
def ind2sub(array_shape, ind):
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


def convert_complex(array, output):
    if output == 'intensity':
        array = np.abs(array) ** 2
    elif output == 'abs':
        array = np.abs(array)
    elif output == 'real':
        array = array.real
    elif output == 'imag':
        array = array.imag
    elif output == 'phase':
        array = np.angle(array)
    else:
        raise RuntimeError()

    return array


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
