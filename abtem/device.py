from typing import Union, Callable
import numpy as np
import psutil
import pyfftw

from abtem.cpu_kernels import abs2, complex_exponential, interpolate_radial_functions, scale_reduce, \
    windowed_scale_reduce, sum_run_length_encoded

FFTW_EFFORT = 'FFTW_MEASURE'
FFTW_THREADS = 12
FFTW_TIMELIMIT = 600

try:  # This should be the only place import cupy, to make it a non-essential dependency
    import cupy as cp
    import cupyx.scipy.fft
    import cupyx.scipy.ndimage as ndimage
    from abtem.cuda_kernels import launch_interpolate_radial_functions, launch_scale_reduce, \
        launch_windowed_scale_reduce, launch_superpose_deltas, launch_sum_run_length_encoded

    get_array_module = cp.get_array_module


    def fft2_convolve(array: cp.array, kernel: cp.array, overwrite_x: bool = True):
        """
        2d FFT convolution using GPU.
        """
        array = cupyx.scipy.fftpack.fft2(array, overwrite_x=overwrite_x)
        array *= kernel
        array = cupyx.scipy.fftpack.ifft2(array, overwrite_x=overwrite_x)
        return array


    def fft2(array, overwrite_x=True):
        return cupyx.scipy.fft.fft2(array, overwrite_x=overwrite_x)


    def ifft2(array, overwrite_x=True):
        return cupyx.scipy.fft.ifft2(array, overwrite_x=overwrite_x)


    def pin_array(array):
        print(array.dtype, array.shape)
        mem = cp.cuda.alloc_pinned_memory(array.nbytes)

        src = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
        src[...] = array
        return src


    gpu_functions = {'fft2': fft2,
                     'ifft2': ifft2,
                     'fft2_convolve': fft2_convolve,
                     'pin_array': pin_array,
                     'complex_exponential': lambda x: cp.exp(1.j * x),
                     'abs2': lambda x: cp.abs(x) ** 2,
                     'interpolate_radial_functions': launch_interpolate_radial_functions,
                     'scale_reduce': launch_scale_reduce,
                     'sum_run_length_encoded': launch_sum_run_length_encoded,
                     'windowed_scale_reduce': launch_windowed_scale_reduce,
                     'superpose_deltas': launch_superpose_deltas}

    asnumpy = cp.asnumpy

except ImportError:  # cupy is not available
    cp = None
    get_array_module = lambda *args, **kwargs: np
    gpu_functions = None
    asnumpy = np.asarray
    import scipy.ndimage as ndimage


def create_fftw_objects(array, allow_new_plan=True):
    """
    Creates FFTW object for forward and backward Fourier transforms. The input array will be
    transformed in place. The function tries to retrieve FFTW plans from wisdom only.
    If no plan exists for the input array, a new plan is cached and then retrieved.

    :param array: Numpy array to be transformed. 2 dimensions or more.
    :param allow_new_plan: If false raise an exception instead of caching a new plan.
    :return:
    """

    try:
        # try using cached FFTW plan
        fftw_forward = pyfftw.FFTW(array, array, axes=(-1, -2),
                                   threads=FFTW_THREADS,
                                   flags=(FFTW_EFFORT, 'FFTW_WISDOM_ONLY', 'FFTW_DESTROY_INPUT'))
        fftw_backward = pyfftw.FFTW(array, array, axes=(-1, -2),
                                    direction='FFTW_BACKWARD', threads=FFTW_THREADS,
                                    flags=(FFTW_EFFORT, 'FFTW_WISDOM_ONLY', 'FFTW_DESTROY_INPUT'))
        return fftw_forward, fftw_backward

    except RuntimeError as e:
        if (not allow_new_plan):
            fftw_forward = pyfftw.builders.fft2(array)
            fftw_backward = pyfftw.builders.ifft2(array)
            return fftw_forward, fftw_backward
        # if ('No FFTW wisdom is known for this plan.' != str(e)) or (not allow_new_plan):
        #    raise

        # create new FFTW object, not to be used, but the plan will remain in the cache
        dummy = pyfftw.byte_align(np.zeros_like(array))
        # this is necessary because FFTW overwrites the array during planning

        pyfftw.FFTW(dummy, dummy,
                    axes=(-1, -2),
                    threads=FFTW_THREADS,
                    flags=(FFTW_EFFORT, 'FFTW_DESTROY_INPUT'),
                    planning_timelimit=FFTW_TIMELIMIT)

        pyfftw.FFTW(dummy, dummy, axes=(-1, -2),
                    direction='FFTW_BACKWARD',
                    threads=FFTW_THREADS,
                    flags=(FFTW_EFFORT, 'FFTW_DESTROY_INPUT'),
                    planning_timelimit=FFTW_TIMELIMIT)

        return create_fftw_objects(array, False)


def fft2_convolve(array, kernel, overwrite_x=True):
    def _fft_convolve(array, kernel):
        fftw_forward, fftw_backward = create_fftw_objects(array)
        fftw_forward()
        array *= kernel
        fftw_backward()
        return array

    if not overwrite_x:
        array = array.copy()

    return _fft_convolve(array, kernel)


def fft2(array, overwrite_x=True):
    if not overwrite_x:
        array = array.copy()

    fftw_forward, fftw_backward = create_fftw_objects(array)
    return fftw_forward()


def ifft2(array, overwrite_x=True):
    if not overwrite_x:
        array = array.copy()

    fftw_forward, fftw_backward = create_fftw_objects(array)
    return fftw_backward()


cpu_functions = {'fft2': fft2,
                 'ifft2': ifft2,
                 'fft2_convolve': fft2_convolve,
                 'abs2': abs2,
                 'pin_array': lambda x: x,
                 'complex_exponential': complex_exponential,
                 'interpolate_radial_functions': interpolate_radial_functions,
                 'sum_run_length_encoded': sum_run_length_encoded,
                 'scale_reduce': scale_reduce,
                 'windowed_scale_reduce': windowed_scale_reduce}


def get_device_function(xp, name: str) -> Callable:
    """
    Return the function appropriate to the given array library.

    :param xp: The array library. Must numpy or cupy.
    :param name: Name of function.
    """
    if (xp is cp):
        return gpu_functions[name]
    elif xp is np:
        return cpu_functions[name]
    else:
        raise RuntimeError('The array library ')


def get_array_module_from_device(device):
    if device == 'cpu':
        return np

    if device == 'gpu':
        if cp is None:
            raise RuntimeError('CuPy is not installed, only CPU calculations available')
        return cp

    return get_array_module(device)


def copy_to_device(array, device):
    """
    Copy array to a device.

    :param array: Array to be copied.
    :param device:
    :return:
    """
    if (device == 'cpu') or (device is None) or (device is np):
        return asnumpy(array)

    elif (device == 'gpu') or (device is cp):
        if cp is None:
            raise RuntimeError('CuPy is not installed, only CPU calculations available')
        return cp.asarray(array)

    else:
        raise RuntimeError()


def get_available_memory(device: str) -> float:
    if device == 'cpu':
        return psutil.virtual_memory().available

    else:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

        if device == 'gpu':
            device = cp.cuda.Device(0)
        return device.mem_info[0]


class HasDeviceMixin:
    _device: str

    @property
    def device(self):
        return self._device

    @property
    def calculation_device(self):
        return self._device
