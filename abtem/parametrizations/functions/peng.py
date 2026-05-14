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


def ionic_scattering_factor_k2(k2, p, coulomb_coeff, screening=0.0):
    """Peng 1999 ionic projected scattering factor: 5-Gaussian sum + screened ΔZ correction.

    The Coulomb correction term is ΔZ · M / (κ² + s²), where κ is an optional screening
    wavevector (in 1/Å, same units as s).  With κ=0 this reduces to the bare ΔZ · M / s²
    from Peng 1999 eq. (3).  In abTEM units (k = 2s), coulomb_coeff already absorbs the
    factor of 4 from 1/s² = 4/k², and screening is given as κ in the same k-space units.
    """
    denominator = screening ** 2 + k2
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.where(denominator > 0.0, coulomb_coeff / denominator, 0.0)
    return scattering_factor_k2(k2, p) + correction


def finite_projected_scattering_factor(r, p, a, b):
    p = np.expand_dims(p, tuple(range(2, 2 + len(r.shape))))
    return (
        np.abs(erf(p[2] * b) - erf(p[2] * a))
        * p[0]
        * np.exp(-p[1] * r[None, ...] ** 2.0)
    ).sum(0) / 2
