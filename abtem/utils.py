# import numexpr as ne
import numpy as np


# def complex_exponential(x):
#    return ne.evaluate('exp(1.j * x)')


def complex_exponential(x):
    df_exp = np.empty(x.shape, dtype=np.complex64)
    trig_buf = np.cos(x)
    df_exp.real[:] = trig_buf
    np.sin(x, out=trig_buf)
    df_exp.imag[:] = trig_buf
    return df_exp


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


def sub2ind(rows, cols, array_shape):
    return rows * array_shape[1] + cols


def ind2sub(array_shape, ind):
    rows = ind // array_shape[1]
    cols = ind % array_shape[1]
    return (rows, cols)


class BatchGenerator(object):

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

    def generate(self, show_progress):
        batch_start = 0
        for i in range(self.n_batches):
            batch_end = batch_start + self.batch_size
            if i == self.n_batches - 1:
                yield batch_start, self.n_items - batch_end + self.batch_size
            else:
                yield batch_start, self.batch_size

            batch_start = batch_end
