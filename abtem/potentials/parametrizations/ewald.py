import numpy as np
from abtem.potentials.utils import eps0
from scipy.special import erf
from ase.data import chemical_symbols

ewald_sigma = 1


def load_parameters():
    return {chemical_symbols[i]: i for i in range(1, 100)}


def gaussian_potential(r, p):
    return p / (4 * np.pi * eps0) * erf(r / (np.sqrt(2) * ewald_sigma)) / r


def point_charge_potential(r, p):
    return p / (4 * np.pi * eps0) / r


def potential(r, p):
    return point_charge_potential(r, p) - gaussian_potential(r, p)
