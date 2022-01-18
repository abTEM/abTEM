"""Module to handle ab initio electrostatic potentials from the DFT code GPAW."""
import warnings
from typing import Tuple, Union

from abtem.core.electron_configurations import electron_configurations, config_str_to_config_tuples
from abtem.potentials.poisson import ChargeDensityPotential
import os
import contextlib
from ase.data import chemical_symbols, atomic_numbers
from ase import units
import numpy as np

from abtem.potentials.utils import validate_symbol

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
        pass

    def _calculate(self, symbol, charge):
        qn = config_str_to_config_tuples(electron_configurations[self.symbol])[-1][:2]

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            ae = AllElectronAtom(self._symbol, spinpol=True)
            if self.charge:
                ae.add(2, 1, 1.7)
            ae.run()

        vr_e = interp1d(ae.rgd.r_g * units.Bohr, -ae.rgd.poisson(ae.n_sg.sum(0)),
                        fill_value='extrapolate', bounds_error=False)

        vr = lambda r: atomic_numbers[self.symbol] / r / (4 * np.pi * eps0) + vr_e(r) / r * units.Hartree * units.Bohr

        self._potential

    def potential(self, charge=None):
        pass
