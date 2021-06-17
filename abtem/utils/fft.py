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


def fft2_convolve(x, kernel, overwrite_x=True, dask_key_name=None):
    if isinstance(x, np.ndarray):
        array = mkl_fft.fft2(x, overwrite_x=overwrite_x)
        array *= kernel
        return mkl_fft.ifft2(array, overwrite_x=overwrite_x)

    if isinstance(x, da.core.Array):
        return (x.map_blocks(fft2, overwrite_x=True) * kernel).map_blocks(ifft2, overwrite_x=True)
