import numpy as np
from ase import units
from ase.data import chemical_symbols

# Vacuum permitivity in ASE units
eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)

# Conversion factor from unitless potential parametrizations to ASE potential units
kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)


def validate_symbol(symbol):
    if not isinstance(symbol, str):
        symbol = chemical_symbols[symbol]

    return symbol