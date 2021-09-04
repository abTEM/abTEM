"""Module to handle ab initio electrostatic potentials from the DFT code GPAW."""
import warnings
from typing import Tuple, Union

from abtem.potentials.poisson import ChargeDensityPotential

try:
    from gpaw import GPAW
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