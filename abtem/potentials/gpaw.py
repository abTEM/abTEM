"""Module to handle ab initio electrostatic potentials from the DFT code GPAW."""
import contextlib
import os
import warnings
from collections import defaultdict
from typing import Tuple, Union, List

import dask
import dask.array as da
import numpy as np
from ase import units
from ase.data import chemical_symbols, atomic_numbers
from scipy.interpolate import interp1d

from abtem.core.electron_configurations import electron_configurations, config_str_to_config_tuples
from abtem.potentials.poisson import ChargeDensityPotential
from abtem.potentials.temperature import MDFrozenPhonons, LazyAtoms, DummyFrozenPhonons
from abtem.potentials.utils import eps0

try:
    from gpaw import GPAW
    from gpaw.atom.aeatom import AllElectronAtom
except ImportError:
    GPAW = None


class GPAWPotential(ChargeDensityPotential):

    def __init__(self,
                 calculators: Union[GPAW, List[GPAW], List[str], str],
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 slice_thickness: float = .5,
                 plane: str = 'xy',
                 box: Tuple[float, float, float] = None,
                 gridrefinement: int = 4,
                 exit_planes: int = None,
                 origin: Tuple[float, float, float] = (0., 0., 0.)):

        """
        GPAW potential

        Calculate electrostratic potential from a GPAW calculation.

        Parameters
        ----------
        calculators : gpaw.GPAW or list of gpaw.GPAW or str or list of str

        gpts : one or two int, optional
            Number of grid points describing each slice of the potential.
        sampling : one or two float, optional
            Lateral sampling of the potential [1 / Å].
        slice_thickness : float or sequence of float, optional
            Thickness of the potential slices in Å. If given as a float the number of slices are calculated by dividing the
            slice thickness into the z-height of supercell.
            The slice thickness may be as a sequence of values for each slice, in which case an error will be thrown is the
            sum of slice thicknesses is not equal to the z-height of the supercell.
            Default is 0.5 Å.
        plane :
        box :
        origin :
        """

        if GPAW is None:
            raise RuntimeError('This functionality of abTEM requires GPAW, see https://wiki.fysik.dtu.dk/gpaw/.')

        def load_gpw(path):
            calc = GPAW(path)
            n = calc.get_all_electron_density(gridrefinement=gridrefinement)
            atoms = calc.atoms
            return n.astype(np.float32), atoms

        if not isinstance(calculators, list):
            calculators = [calculators]

        if hasattr(calculators[0], 'get_all_electron_density'):
            frozen_phonons = []
            charge_densities = []
            for calculator in calculators:
                if not hasattr(calculator, 'get_all_electron_density'):
                    raise ValueError()

                shape = calculator.wfs.gd.N_c * gridrefinement
                frozen_phonons.append(calculator.atoms)
                charge_density = dask.delayed(calculator.get_all_electron_density)(gridrefinement=gridrefinement)
                charge_density = da.from_delayed(charge_density, shape=shape, meta=np.array((), dtype=float))
                charge_densities.append(charge_density)

            cell = None
            atomic_numbers = None

        else:
            paths = calculators
            calculator = GPAW(calculators[0])
            shape = calculator.wfs.gd.N_c * gridrefinement

            cell = calculator.atoms.cell
            atomic_numbers = np.unique(calculator.atoms.numbers)

            frozen_phonons = []
            charge_densities = []

            for path in paths:
                # if not isinstance(calculator, str):
                #    raise ValueError()

                charge_density, atoms = dask.delayed(load_gpw, nout=2)(path)
                # atoms = LazyAtoms(atoms, numbers=numbers, cell=cell)
                charge_density = da.from_delayed(charge_density, shape=shape, meta=np.array((), dtype=float))

                frozen_phonons.append(atoms)
                charge_densities.append(charge_density)

        if len(charge_densities) > 1:
            charge_density = da.stack(charge_densities)
            atoms = MDFrozenPhonons(frozen_phonons, atomic_numbers=atomic_numbers, cell=cell)
        else:
            charge_density = charge_densities[0]
            atoms = DummyFrozenPhonons(frozen_phonons[0], atomic_numbers=atomic_numbers, cell=cell)

        super().__init__(charge_density=charge_density, atoms=atoms, gpts=gpts, sampling=sampling,
                         slice_thickness=slice_thickness, plane=plane, box=box, origin=origin, exit_planes=exit_planes)


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
            # ae.run(mix=0.005, maxiter=5000, dnmax=1e-5)
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
