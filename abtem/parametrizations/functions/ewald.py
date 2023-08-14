from __future__ import annotations

import numpy as np
from scipy.special import erf

from abtem.core.constants import eps0


def gaussian_potential(r, Z, width):
    return Z / (4 * np.pi * eps0) * erf(r / (np.sqrt(2) * width)) / r


def gaussian_charge(r, Z, width):
    return (
        Z
        / (width**3 * np.sqrt(2 * np.pi) ** 3)
        * np.exp(-(r**2) / (2 * width**2))
    )


def point_charge_potential(r, Z):
    return Z / (4 * np.pi * eps0) / r


def potential(r, p):
    return point_charge_potential(r, p[1]) - gaussian_potential(r, p[1], p[0])
