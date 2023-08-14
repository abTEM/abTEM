from __future__ import annotations

import numpy as np
from numba import jit
from scipy.special import erf


@jit(nopython=True, nogil=True)
def scattering_factor(k, p):
    return (
        p[0, 0] * np.exp(-p[1, 0] * k**2.0)
        + p[0, 1] * np.exp(-p[1, 1] * k**2.0)
        + p[0, 2] * np.exp(-p[1, 2] * k**2.0)
        + p[0, 3] * np.exp(-p[1, 3] * k**2.0)
        + p[0, 4] * np.exp(-p[1, 4] * k**2.0)
    )


@jit(nopython=True, nogil=True)
def scattering_factor_k2(k2, p):
    return (
        p[0, 0] * np.exp(-p[1, 0] * k2)
        + p[0, 1] * np.exp(-p[1, 1] * k2)
        + p[0, 2] * np.exp(-p[1, 2] * k2)
        + p[0, 3] * np.exp(-p[1, 3] * k2)
        + p[0, 4] * np.exp(-p[1, 4] * k2)
    )


def finite_projected_scattering_factor(r, p, a, b):
    p = np.expand_dims(p, tuple(range(2, 2 + len(r.shape))))
    return (
        np.abs(erf(p[2] * b) - erf(p[2] * a))
        * p[0]
        * np.exp(-p[1] * r[None, ...] ** 2.0)
    ).sum(0) / 2
