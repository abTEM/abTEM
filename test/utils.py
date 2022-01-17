from typing import Iterable

import dask.array as da

from abtem.core.backend import get_array_module
import numpy as np

def check_array_matches_device(array, device):
    assert get_array_module(array) is get_array_module(device)


def ensure_is_tuple(x, length: int = 1):
    if not isinstance(x, tuple):
        x = (x,) * length
    elif isinstance(x, Iterable):
        x = tuple(x)
    assert len(x) == length
    return x


def check_array_matches_laziness(array, lazy):
    if lazy:
        assert isinstance(array, da.core.Array)


def array_is_close(a1, a2, rel_tol=np.inf, abs_tol=np.inf, check_above_abs=0., check_above_rel=0.):
    if rel_tol < np.inf:
        element_is_checked = (a2 > check_above_abs) * (a2 > (a2.max() * check_above_rel))
        rel_error = (a1[element_is_checked] - a2[element_is_checked]) / a2[element_is_checked]
        if np.any(np.abs(rel_error) > rel_tol):
            return False

    if abs_tol < np.inf:
        if np.any(np.abs(a1 - a2) > abs_tol):
            return False

    return True
