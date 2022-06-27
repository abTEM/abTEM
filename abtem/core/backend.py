from numbers import Number

import dask.array as da
import numpy as np
import scipy

from abtem.core.config import config

try:
    import cupy as cp
except:
    cp = None

try:
    import cupyx
except:
    cupyx = None


def check_cupy_is_installed():
    if cp is None:
        raise RuntimeError('CuPy is not installed, GPU calculations disabled')


def xp_to_str(xp):
    if xp is np:
        return 'numpy'

    check_cupy_is_installed()

    if xp is cp:
        return 'cupy'

    raise ValueError(f'array module must be NumPy or CuPy')


def validate_device(device):
    if device is None:
        return config.get('device')

    return device


def get_array_module(x):
    if x is None:
        return get_array_module(config.get('device'))

    if isinstance(x, da.core.Array):
        return get_array_module(x._meta)

    if isinstance(x, str):
        if x.lower() in ('numpy', 'cpu'):
            return np

        check_cupy_is_installed()

        if x.lower() in ('cupy', 'gpu'):
            return cp

    if isinstance(x, np.ndarray):
        return np

    if x is np:
        return np

    check_cupy_is_installed()

    if isinstance(x, cp.ndarray):
        return cp

    if x is cp:
        return cp

    if isinstance(x, Number):
        return np

    raise ValueError(f'array module specification {x} not recognized')


def device_name_from_array_module(xp):
    if xp is np:
        return 'cpu'

    if xp is cp:
        return 'gpu'

    assert False


def get_scipy_module(x):
    xp = get_array_module(x)

    if xp is np:
        return scipy

    if xp is cp:
        return cupyx.scipy


def get_ndimage_module(x):
    xp = get_array_module(x)

    if xp is np:
        import scipy.ndimage
        return scipy.ndimage

    if xp is cp:
        import cupyx.scipy.ndimage
        return cupyx.scipy.ndimage


def asnumpy(array):
    if cp is None:
        return array

    if isinstance(array, da.core.Array):
        return array.map_blocks(asnumpy)

    return cp.asnumpy(array)


def copy_to_device(array, device):
    old_xp = get_array_module(array)
    new_xp = get_array_module(device)

    if old_xp is new_xp:
        return array

    if isinstance(array, da.core.Array):
        return array.map_blocks(copy_to_device, meta=new_xp.array((), dtype=array.dtype), device=device)

    if new_xp is np:
        return cp.asnumpy(array)

    if new_xp is cp:
        return cp.asarray(array)

    raise RuntimeError()


class HasDevice:
    _device: str

    @property
    def device(self):
        return self._device
