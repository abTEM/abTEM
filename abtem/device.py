import numpy as np
import mkl_fft
from abtem.cpu_kernels import abs2, complex_exponential

# TODO : This is a little ugly, change after mkl_fft is updated

try:  # This should be the only place to get cupy, to make it a non-essential dependency
    import cupy as cp
    import cupyx.scipy.fft

    get_array_module = cp.get_array_module


    def fft2_convolve(array, kernel, overwrite_x=True):
        array = cupyx.scipy.fft.fft2(array, overwrite_x=overwrite_x)
        array = cupyx.scipy.fft.ifft2(array * kernel, overwrite_x=overwrite_x)
        return array


    gpu_functions = {'fft2': cupyx.scipy.fft.fft2, 'ifft2': cupyx.scipy.fft.ifft2, 'fft2_convolve': fft2_convolve,
                     'complex_exponential': lambda x: cp.exp(1.j * x)}

    asnumpy = cp.asnumpy

except ImportError:
    cp = None
    get_array_module = lambda *args, **kwargs: np
    fft2_gpu = None
    ifft2_gpu = None
    fft2_convolve_gpu = None
    gpu_functions = {'fft2': None, 'ifft2': None, 'fft2_convolve': None}
    asnumpy = np.asarray


def fft2_convolve(array, kernel, overwrite_x=True):
    def _fft_convolve(array, kernel, overwrite_x):
        array = mkl_fft.fft2(array, overwrite_x=overwrite_x)
        array *= kernel
        array = mkl_fft.ifft2(array, overwrite_x=overwrite_x)
        return array

    if len(array.shape) == 2:
        return _fft_convolve(array, kernel, overwrite_x)
    elif (len(array.shape) == 3) & overwrite_x:
        for i in range(len(array)):
            _fft_convolve(array[i], kernel, overwrite_x=True)
        return array
    elif (len(array.shape) == 3) & (not overwrite_x):
        new_array = np.zeros_like(array)
        for i in range(len(array)):
            new_array[i] = _fft_convolve(array[i], kernel, overwrite_x=False)
        return new_array
    else:
        raise ValueError()


cpu_functions = {'fft2': mkl_fft.fft2, 'ifft2': mkl_fft.ifft2, 'fft2_convolve': fft2_convolve, 'abs2': abs2,
                 'complex_exponential': complex_exponential}


def get_device_function(xp, name):
    if xp is cp:
        return gpu_functions[name]
    elif xp is np:
        return cpu_functions[name]
    else:
        raise RuntimeError()


class Device:

    def __init__(self, device_definition):
        self._device_definition = device_definition

        if (device_definition == 'cpu') or isinstance(device_definition, np.ndarray):
            self._device_type = 'cpu'
            self._array_module = np
            self._cupy_device = None
        elif cp is None:
            raise RuntimeError('cupy is not installed, only cpu calculations available')

        if (device_definition == 'gpu') or isinstance(device_definition, cp.ndarray):
            self._device_type = 'gpu'
            self._array_module = np
            self._cupy_device = None

    @property
    def device_type(self):
        return self._device_type

    @property
    def array_module(self):
        return self._array_module

    @property
    def xp(self):
        return self.array_module
