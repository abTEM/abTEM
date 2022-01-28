"""Module to handle ab initio electrostatic potentials from the DFT code GPAW."""
import warnings
from collections import defaultdict
from typing import Tuple, Union

from abtem.core.electron_configurations import electron_configurations, config_str_to_config_tuples
from abtem.potentials.poisson import ChargeDensityPotential
import os
import contextlib
from ase.data import chemical_symbols, atomic_numbers
from ase import units
import numpy as np

from abtem.potentials.utils import validate_symbol
from abtem.potentials.utils import kappa, eps0
from scipy.interpolate import interp1d

try:
    from gpaw import GPAW
    from gpaw.atom.aeatom import AllElectronAtom
except ImportError:
    warnings.warn('This functionality of abTEM requires GPAW, see https://wiki.fysik.dtu.dk/gpaw/.')


class GPAWPotential(ChargeDensityPotential):

    def __init__(self,
                 calc: GPAW,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 slice_thickness: float = .5,
                 plane: str = 'xy',
                 box: Tuple[float, float, float] = None,
                 origin: Tuple[float, float, float] = (0., 0., 0.),
                 chunks: int = 1,
                 fft_singularities: bool = False):

        charge_density = calc.get_all_electron_density(gridrefinement=4)

        atoms = calc.atoms

        super().__init__(charge_density=charge_density, atoms=atoms, gpts=gpts, sampling=sampling,
                         slice_thickness=slice_thickness, plane=plane, box=box, origin=origin, chunks=chunks,
                         fft_singularities=fft_singularities)


class GPAWParametrization:

    def __init__(self):
        self._potential_functions = {}

    def _get_added_electrons(self, symbol, charge):
        if not charge:
            return []

        charge = (np.sign(charge) * np.ceil(np.abs(charge)))

        number = atomic_numbers[symbol]
        config = config_str_to_config_tuples(electron_configurations[chemical_symbols[number]])
        ionic_config = config_str_to_config_tuples(electron_configurations[chemical_symbols[number - charge]])

        config = defaultdict(lambda: 0, {shell[:2]: shell[2] for shell in config})
        ionic_config = defaultdict(lambda: 0, {shell[:2]: shell[2] for shell in ionic_config})

        electrons = []
        for key in set(config.keys()).union(set(ionic_config.keys())):

            difference = config[key] - ionic_config[key]

            for i in range(np.abs(difference)):
                electrons.append(key + (np.sign(difference),))
        return electrons

    def _get_all_electron_atom(self, symbol, charge=0.):

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            ae = AllElectronAtom(symbol, spinpol=True, xc='PBE')

            added_electrons = self._get_added_electrons(symbol, charge)
        #     for added_electron in added_electrons:
        #         ae.add(*added_electron[:2], added_electron[-1])
        # # ae.run()
            #ae.run(mix=0.005, maxiter=5000, dnmax=1e-5)
            ae.run()
            ae.refine()

        return ae

        # vr_e = interp1d(radial_coord, electron_potential, fill_value='extrapolate', bounds_error=False)
        # vr = lambda r: atomic_numbers[symbol] / r / (4 * np.pi * eps0) + vr_e(r) / r * units.Hartree * units.Bohr

    def charge(self, symbol, charge=0.):
        ae = self._get_all_electron_atom(symbol, charge)
        r = ae.rgd.r_g * units.Bohr
        n = ae.n_sg.sum(0) / units.Bohr ** 3
        return interp1d(r, n, fill_value='extrapolate', bounds_error=False)

    def potential(self, symbol, charge=0.):
        ae = self._get_all_electron_atom(symbol, charge)
        r = ae.rgd.r_g * units.Bohr
        # n = ae.n_sg.sum(0) / units.Bohr ** 3

        ve = -ae.rgd.poisson(ae.n_sg.sum(0))
        ve = interp1d(r, ve, fill_value='extrapolate', bounds_error=False)
        # electron_potential = -ae.rgd.poisson(ae.n_sg.sum(0))

        vr = lambda r: atomic_numbers[symbol] / r / (4 * np.pi * eps0) + ve(r) / r * units.Hartree * units.Bohr
        return vr

    # def get_function(self, symbol, charge=0.):
    #     #if symbol in self._potential_functions.keys():
    #     #    return self._potential_functions[(symbol, charge)]
    #
    #
    #
    #     self._potential_functions[(symbol, charge)] = vr
    #     return self._potential_functions[(symbol, charge)]
    #
    # def potential(self, r, symbol, charge=0.):
    #     potential = self._calculate(symbol, charge)
    #     return potential(r)
