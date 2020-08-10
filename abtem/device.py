from typing import Any

import numpy.fft as mkl_fft
import numpy as np

from abtem.cpu_kernels import abs2, complex_exponential, interpolate_radial_functions, scale_reduce, \
    windowed_scale_reduce

# TODO : This is a little ugly, change after mkl_fft is updated

try:  # This should be the only place to get cupy, to make it a non-essential dependency
    import cupy as cp
    import cupyx.scipy.fft
    from abtem.cuda_kernels import launch_interpolate_radial_functions, launch_scale_reduce, \
        launch_windowed_scale_reduce

    get_array_module = cp.get_array_module


    def fft2_convolve(array, kernel, overwrite_x=True):
        array = cupyx.scipy.fftpack.fft2(array, overwrite_x=overwrite_x)
        array *= kernel
        array = cupyx.scipy.fftpack.ifft2(array, overwrite_x=overwrite_x)
        return array


    gpu_functions = {'fft2': cupyx.scipy.fft.fft2,
                     'ifft2': cupyx.scipy.fft.ifft2,
                     'fft2_convolve': fft2_convolve,
                     'complex_exponential': lambda x: cp.exp(1.j * x),
                     'abs2': lambda x: cp.abs(x) ** 2,
                     'interpolate_radial_functions': launch_interpolate_radial_functions,
                     'scale_reduce': launch_scale_reduce,
                     'windowed_scale_reduce': launch_windowed_scale_reduce}

    asnumpy = cp.asnumpy

except ImportError:
    cp = None
    get_array_module = lambda *args, **kwargs: np
    fft2_gpu = None
    ifft2_gpu = None
    fft2_convolve_gpu = None
    gpu_functions = None
    asnumpy = np.asarray


def fft2_convolve(array, kernel):
    def _fft_convolve(array, kernel):
        mkl_fft.fft2(array)
        array *= kernel
        mkl_fft.ifft2(array)
        return array

    if len(array.shape) == 2:
        return _fft_convolve(array, kernel)
    elif (len(array.shape) == 3):
        for i in range(len(array)):
            _fft_convolve(array[i], kernel)
        return array
    else:
        raise ValueError()


def fft2(array, overwrite_x):
    if not overwrite_x:
        array = array.copy()

    if len(array.shape) == 2:
        return mkl_fft.fft2(array)
    elif (len(array.shape) == 3):
        for i in range(array.shape[0]):
            mkl_fft.fft2(array[i])
        return array
    else:
        shape = array.shape
        array = array.reshape((-1,) + shape[1:])
        for i in range(array.shape[0]):
            mkl_fft.fft2(array[i])

        array = array.reshape(shape)
        return array


def ifft2(array, overwrite_x):
    if not overwrite_x:
        array = array.copy()

    if len(array.shape) == 2:
        return mkl_fft.ifft2(array)
    elif len(array.shape) == 3:
        for i in range(array.shape[0]):
            array = mkl_fft.ifft2(array)
        return array
    else:
        raise NotImplementedError()


cpu_functions = {'fft2': fft2,
                 'ifft2': ifft2,
                 'fft2_convolve': fft2_convolve,
                 'abs2': abs2,
                 'complex_exponential': complex_exponential,
                 'interpolate_radial_functions': interpolate_radial_functions,
                 'scale_reduce': scale_reduce,
                 'windowed_scale_reduce': windowed_scale_reduce}


def get_device_function(xp, name):
    if xp is cp:
        return gpu_functions[name]
    elif xp is np:
        return cpu_functions[name]
    else:
        raise RuntimeError()


def get_array_module_from_device(device):
    if device == 'cpu':
        return np

    if device == 'gpu':
        if cp is None:
            raise RuntimeError('CuPy is not installed, only CPU calculations available')
        return cp

    return get_array_module(device)


def copy_to_device(array, device):
    if (device == 'cpu') or (device is np):
        return asnumpy(array)
    elif (device == 'gpu') or (device is cp):
        if cp is None:
            raise RuntimeError('CuPy is not installed, only CPU calculations available')
        return cp.asarray(array)
    else:
        raise RuntimeError()


class HasDeviceMixin:
    _device: Any

    @property
    def device(self):
        return self._device

    def set_device(self, device):
        self._device = device

    def get_array_module(self):
        if self.device == 'cpu':
            return np

        if self.device == 'gpu':
            if cp is None:
                raise RuntimeError('CuPy is not installed, only CPU calculations available')
            return cp

        return get_array_module(self.device)
