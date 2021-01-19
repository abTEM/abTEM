"""Module for various convenient utilities."""
import numpy as np
from ase import units

from abtem.device import get_array_module, get_device_function
from tqdm.auto import tqdm
import numbers
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def _set_path(path):
    """Internal function to set the parametrization data directory."""
    return os.path.join(_ROOT, 'data', path)


def energy2mass(energy):
    """
    Calculate relativistic mass from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic mass [kg]̄
    """

    return (1 + units._e * energy / (units._me * units._c ** 2)) * units._me


def energy2wavelength(energy):
    """
    Calculate relativistic de Broglie wavelength from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic de Broglie wavelength [Å].
    """

    return units._hplanck * units._c / np.sqrt(
        energy * (2 * units._me * units._c ** 2 / units._e + energy)) / units._e * 1.e10


def energy2sigma(energy):
    """
    Calculate interaction parameter from energy.

    Parameters
    ----------
    energy: float
        Energy [ev].

    Returns
    -------
    float
        Interaction parameter [1 / (Å * eV)].
    """

    return (2 * np.pi * energy2mass(energy) * units.kg * units._e * units.C * energy2wavelength(energy) / (
            units._hplanck * units.s * units.J) ** 2)


def spatial_frequencies(gpts, sampling):
    """
    Calculate spatial frequencies of a grid.

    Parameters
    ----------
    gpts: tuple of int
        Number of grid points.
    sampling: tuple of float
        Sampling of the potential [1 / Å].

    Returns
    -------
    tuple of arrays
    """

    return tuple(np.fft.fftfreq(n, d).astype(np.float32) for n, d in zip(gpts, sampling))


def polar_coordinates(x, y):
    """Calculate a polar grid for a given Cartesian grid."""
    xp = get_array_module(x)
    alpha = xp.sqrt(x.reshape((-1, 1)) ** 2 + y.reshape((1, -1)) ** 2)
    phi = xp.arctan2(x.reshape((-1, 1)), y.reshape((1, -1)))
    return alpha, phi


def _disc_meshgrid(r):
    """Internal function to return all indices inside a disk with a given radius."""
    cols = np.zeros((2 * r + 1, 2 * r + 1)).astype(np.int32)
    cols[:] = np.linspace(0, 2 * r, 2 * r + 1) - r
    rows = cols.T
    inside = (rows ** 2 + cols ** 2) <= r ** 2
    return rows[inside], cols[inside]


def periodic_crop(array, corners, new_shape):
    xp = get_array_module(array)

    if ((corners[0] > 0) & (corners[1] > 0) & (corners[0] + new_shape[0] < array.shape[-2]) & (
            corners[1] + new_shape[1] < array.shape[-1])):
        array = array[..., corners[0]:corners[0] + new_shape[0], corners[1]:corners[1] + new_shape[1]]
        return array

    x = xp.arange(corners[0], corners[0] + new_shape[0], dtype=xp.int) % array.shape[-2]
    y = xp.arange(corners[1], corners[1] + new_shape[1], dtype=xp.int) % array.shape[-1]

    x, y = xp.meshgrid(x, y, indexing='ij')
    array = array[..., x.ravel(), y.ravel()].reshape(array.shape[:-2] + new_shape)
    return array


def fft_interpolation_masks(shape1, shape2, xp=np, epsilon=1e-7):
    kx1 = xp.fft.fftfreq(shape1[-2], 1 / shape1[-2])
    ky1 = xp.fft.fftfreq(shape1[-1], 1 / shape1[-1])

    kx2 = xp.fft.fftfreq(shape2[-2], 1 / shape2[-2])
    ky2 = xp.fft.fftfreq(shape2[-1], 1 / shape2[-1])

    kx_min = max(xp.min(kx1), xp.min(kx2)) - epsilon
    kx_max = min(xp.max(kx1), xp.max(kx2)) + epsilon
    ky_min = max(xp.min(ky1), xp.min(ky2)) - epsilon
    ky_max = min(xp.max(ky1), xp.max(ky2)) + epsilon

    kx1, ky1 = xp.meshgrid(kx1, ky1, indexing='ij')
    kx2, ky2 = xp.meshgrid(kx2, ky2, indexing='ij')

    mask1 = (kx1 <= kx_max) & (kx1 >= kx_min) & (ky1 <= ky_max) & (ky1 >= ky_min)
    mask2 = (kx2 <= kx_max) & (kx2 >= kx_min) & (ky2 <= ky_max) & (ky2 >= ky_min)

    return mask1, mask2


def fft_crop(array, new_shape):
    xp = get_array_module(array)

    mask_in, mask_out = fft_interpolation_masks(array.shape, new_shape, xp=xp)

    if len(new_shape) < len(array.shape):
        new_shape = array.shape[:-2] + new_shape

    new_array = xp.zeros(new_shape, dtype=array.dtype)

    out_indices = xp.where(mask_out)
    in_indices = xp.where(mask_in)

    new_array[..., out_indices[0], out_indices[1]] = array[..., in_indices[0], in_indices[1]]
    return new_array


def fft_interpolate_2d(array, new_shape, normalization='values', overwrite_x=False):
    xp = get_array_module(array)
    fft2 = get_device_function(xp, 'fft2')
    ifft2 = get_device_function(xp, 'ifft2')

    old_size = array.shape[-2] * array.shape[-1]

    if np.iscomplexobj(array):
        cropped = fft_crop(fft2(array), new_shape)
        print(cropped.shape)
        array = ifft2(cropped, overwrite_x=overwrite_x)
    else:
        array = xp.complex64(array)
        array = ifft2(fft_crop(fft2(array), new_shape), overwrite_x=overwrite_x).real

    if normalization == 'values':
        array *= array.shape[-1] * array.shape[-2] / old_size
    elif normalization == 'norm':
        array *= array.shape[-1] * array.shape[-2] / old_size
    elif (normalization != False) and (normalization != None):
        raise RuntimeError()

    return array


def fourier_translation_operator(positions, shape):
    positions_shape = positions.shape

    if len(positions_shape) == 1:
        positions = positions[None]

    xp = get_array_module(positions)
    complex_exponential = get_device_function(xp, 'complex_exponential')

    kx, ky = spatial_frequencies(shape, (1., 1.))
    kx = kx.reshape((1, -1, 1))
    ky = ky.reshape((1, 1, -1))
    kx = xp.asarray(kx)
    ky = xp.asarray(ky)
    positions = xp.asarray(positions)
    x = positions[:, 0].reshape((-1,) + (1, 1))
    y = positions[:, 1].reshape((-1,) + (1, 1))

    result = complex_exponential(-2 * np.pi * kx * x) * complex_exponential(-2 * np.pi * ky * y)

    if len(positions_shape) == 1:
        return result[0]
    else:
        return result


def subdivide_into_batches(num_items: int, num_batches: int = None, max_batch: int = None):
    """
    Split an n integer into m (almost) equal integers, such that the sum of smaller integers equals n.

    Parameters
    ----------
    n: int
        The integer to split.
    m: int
        The number integers n will be split into.

    Returns
    -------
    list of int
    """
    if (num_batches is not None) & (max_batch is not None):
        raise RuntimeError()

    if num_batches is None:
        if max_batch is not None:
            num_batches = (num_items + (-num_items % max_batch)) // max_batch
        else:
            raise RuntimeError()

    if num_items < num_batches:
        raise RuntimeError('num_batches may not be larger than num_items')

    elif num_items % num_batches == 0:
        return [num_items // num_batches] * num_batches
    else:
        v = []
        zp = num_batches - (num_items % num_batches)
        pp = num_items // num_batches
        for i in range(num_batches):
            if i >= zp:
                v = [pp + 1] + v
            else:
                v = [pp] + v
        return v


def generate_batches(num_items: int, num_batches: int = None, max_batch: int = None, start=0):
    for batch in subdivide_into_batches(num_items, num_batches, max_batch):
        end = start + batch
        yield start, end

        start = end


def tapered_cutoff(x, cutoff, rolloff=.1):
    xp = get_array_module(x)

    rolloff = rolloff * cutoff

    if rolloff > 0.:
        array = .5 * (1 + xp.cos(np.pi * (x - cutoff + rolloff) / rolloff))
        array[x > cutoff] = 0.
        array = xp.where(x > cutoff - rolloff, array, xp.ones_like(x, dtype=xp.float32))
    else:
        array = xp.array(x < cutoff).astype(xp.float32)

    return array


class ProgressBar:
    """Object to describe progress bar indicators for computations."""

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
