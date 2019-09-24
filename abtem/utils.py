import numexpr as ne
import numpy as np
from numba import njit
from tqdm.auto import tqdm


def complex_exponential(x):
    return ne.evaluate('exp(1.j * x)')


# def complex_exponential(x):
#     df_exp = np.empty(x.shape, dtype=np.complex64)
#     trig_buf = np.cos(x)
#     df_exp.real[:] = trig_buf
#     np.sin(x, out=trig_buf)
#     df_exp.imag[:] = trig_buf
#     return df_exp


def fourier_propagator(k, dz, wavelength):
    return complex_exponential(-k * np.pi * wavelength * dz)


def squared_norm(x, y):
    return x.reshape((-1, 1)) ** 2 + y.reshape((1, -1)) ** 2


def fftfreq(grid):
    grid.check_is_grid_defined()
    return tuple(np.fft.fftfreq(gpts, sampling) for gpts, sampling in zip(grid.gpts, grid.sampling))


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

    def generate(self, show_progress=False):
        batch_start = 0
        for i in tqdm(range(self.n_batches), disable=not show_progress):
            batch_end = batch_start + self.batch_size
            if i == self.n_batches - 1:
                yield batch_start, self.n_items - batch_end + self.batch_size
            else:
                yield batch_start, self.batch_size

            batch_start = batch_end
