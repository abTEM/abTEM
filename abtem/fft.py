import mkl_fft
import numpy as np
import dask.array as da


def fft2(x, **kwargs):
    if isinstance(x, np.ndarray):
        return mkl_fft.fft2(x, **kwargs)

    if isinstance(x, da.core.Array):
        return x.map_blocks(mkl_fft.fft2)


def ifft2(x, **kwargs):
    if isinstance(x, np.ndarray):
        return mkl_fft.ifft2(x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(mkl_fft.ifft2)


def fft2_convolve(x, kernel, **kwargs):
    if isinstance(x, np.ndarray):
        return mkl_fft.ifft2(mkl_fft.fft2(x, **kwargs) * kernel, **kwargs)

    if isinstance(x, da.core.Array):
        return (x.map_blocks(mkl_fft.fft2) * kernel).map_blocks(mkl_fft.ifft2)

        # return lambda x, overwrite_x: mkl_fft.ifft2(x, overwrite_x)
