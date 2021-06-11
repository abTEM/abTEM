import mkl_fft
import numpy as np
import dask.array as da


def fft2(x, overwrite_x=False, dask_key_name=None):
    if isinstance(x, np.ndarray):
        return mkl_fft.fft2(x, overwrite_x=overwrite_x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(mkl_fft.fft2, name=dask_key_name)


def ifft2(x, overwrite_x=False, dask_key_name=None):
    if isinstance(x, np.ndarray):
        return mkl_fft.ifft2(x, overwrite_x=overwrite_x)

    if isinstance(x, da.core.Array):
        return x.map_blocks(mkl_fft.ifft2, name=dask_key_name)


def fft2_convolve(x, kernel, overwrite_x=False, dask_key_name=None):
    if isinstance(x, np.ndarray):
        return mkl_fft.ifft2(mkl_fft.fft2(x, overwrite_x=overwrite_x) * kernel, overwrite_x=overwrite_x)

    if isinstance(x, da.core.Array):
        return (x.map_blocks(mkl_fft.fft2) * kernel).map_blocks(mkl_fft.ifft2)
