import numpy as np
import dask.array as da

try:
    import cupy as cp
except:
    cp = None


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


def get_array_module(x):
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

    if isinstance(x, da.core.Array):
        return get_array_module(x._meta)

    raise ValueError(f'array module specification {x} not recognized')
