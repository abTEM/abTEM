"""Module to handle ab initio electrostatic potentials from the DFT code GPAW."""
from dataclasses import dataclass
from typing import Any

import dask
import numpy as np
from ase import units, Atoms
from ase.io.trajectory import read_atoms
from ase.units import Bohr

try:
    from gpaw import GPAW
    from gpaw.atom.aeatom import AllElectronAtom
    from gpaw.io import Reader
    from gpaw.density import RealSpaceDensity
    from gpaw.mpi import SerialCommunicator
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.utilities import unpack_atomic_matrices
except ImportError:
    GPAW = None
    AllElectronAtom = None
    Reader = None
    SerialCommunicator = None
    GridDescriptor = None
    unpack_atomic_matrices = None


def safe_read_atoms(calculator):
    if isinstance(calculator, str):
        atoms = read_atoms(Reader(calculator).atoms)
    else:
        atoms = calculator.atoms

    return atoms.copy()


def get_gpaw_setups(atoms, mode, xc):
    gpaw = GPAW(txt=None, mode=mode, xc=xc)
    gpaw.initialize(atoms)
    return gpaw.setups


@dataclass
class DummyGPAW:
    setup_mode: str
    setup_xc: str
    nt_sG: np.ndarray
    gd: Any
    D_asp: np.ndarray
    atoms: Atoms

    @property
    def setups(self):
        gpaw = GPAW(txt=None, mode=self.setup_mode, xc=self.setup_xc)
        gpaw.initialize(self.atoms)
        return gpaw.setups

    @classmethod
    def from_gpaw(cls, gpaw, lazy: bool = True):
        if lazy:
            return dask.delayed(cls.from_gpaw)(gpaw, lazy=False)

        atoms = gpaw.atoms.copy()
        atoms.calc = None

        kwargs = {'setup_mode': gpaw.parameters['mode'],
                  'setup_xc': gpaw.parameters['xc'],
                  'nt_sG': gpaw.density.nt_sG.copy(),
                  'gd': gpaw.density.gd.new_descriptor(comm=SerialCommunicator()),
                  'D_asp': dict(gpaw.density.D_asp),
                  'atoms': atoms}
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str, lazy: bool = True):
        if lazy:
            return dask.delayed(cls.from_file)(path, lazy=False)

        reader = Reader(path)
        atoms = read_atoms(reader.atoms)

        parameters = GPAW.default_parameters
        parameters.update(reader.parameters.asdict())

        setup_mode = parameters['mode']
        setup_xc = parameters['xc']

        if isinstance(setup_xc, dict) and 'setup_name' in setup_xc:
            setup_xc = setup_xc['setup_name']

        assert isinstance(setup_xc, str)

        density = reader.density.density * units.Bohr ** 3
        gd = GridDescriptor(N_c=density.shape[-3:],
                            cell_cv=atoms.get_cell() / Bohr,
                            comm=SerialCommunicator())

        setups = get_gpaw_setups(atoms, setup_mode, setup_xc)

        D_asp = unpack_atomic_matrices(reader.density.atomic_density_matrices, setups)

        kwargs = {'setup_mode': setup_mode,
                  'setup_xc': setup_xc,
                  'nt_sG': density,
                  'gd': gd,
                  'D_asp': D_asp,
                  'atoms': atoms}
        return cls(**kwargs)

    @classmethod
    def from_generic(cls, calculator, lazy: bool = True):
        if isinstance(calculator, str):
            return cls.from_file(calculator, lazy=lazy)
        elif isinstance(calculator, GPAW):
            return cls.from_gpaw(calculator, lazy=lazy)
        elif isinstance(calculator, cls):
            return calculator
        else:
            raise RuntimeError()
