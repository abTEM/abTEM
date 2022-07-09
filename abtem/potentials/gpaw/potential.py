"""Module to handle ab initio electrostatic potentials from the DFT code GPAW."""
from functools import partial
from typing import Tuple, Union, List, TYPE_CHECKING

import dask
import dask.array as da
import numpy as np

from abtem.core.axes import AxisMetadata
from abtem.potentials.gpaw.density import get_all_electron_density
from abtem.potentials.gpaw.io import safe_read_atoms, DummyGPAW
from abtem.potentials.parametrizations import EwaldParametrization
from abtem.potentials.poisson import generate_slices
from abtem.potentials.potentials import PotentialBuilder, Potential
from abtem.potentials.temperature import DummyFrozenPhonons, FrozenPhonons, AbstractFrozenPhonons

try:
    from gpaw import GPAW
except:
    GPAW = None


class GPAWPotential(PotentialBuilder):

    def __init__(self,
                 calculators: Union['GPAW', List['GPAW'], List[str], str],
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 slice_thickness: float = .5,
                 exit_planes: int = None,
                 gridrefinement: int = 4,
                 device: str = None,
                 plane: str = 'xy',
                 box: Tuple[float, float, float] = None,
                 origin: Tuple[float, float, float] = (0., 0., 0.),
                 periodic: bool = True,
                 repetitions: Tuple[int, int, int] = (1, 1, 1),
                 frozen_phonons: AbstractFrozenPhonons = None, ):

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

        if isinstance(calculators, (tuple, list)):
            atoms = safe_read_atoms(calculators[0])

            num_configs = len(calculators)

            if frozen_phonons is not None:
                raise ValueError()

            calculators = [DummyGPAW.from_generic(calculator) for calculator in calculators]

            frozen_phonons = DummyFrozenPhonons(atoms, num_configs=num_configs)

        else:
            atoms = safe_read_atoms(calculators)

            calculators = DummyGPAW.from_generic(calculators)

            if frozen_phonons is None:
                frozen_phonons = DummyFrozenPhonons(atoms, num_configs=None)

        self._calculators = calculators
        self._frozen_phonons = frozen_phonons
        self._gridrefinement = gridrefinement
        self._repetitions = repetitions

        cell = frozen_phonons.atoms.cell * repetitions
        frozen_phonons.atoms.calc = None

        super().__init__(gpts=gpts,
                         sampling=sampling,
                         cell=cell,
                         slice_thickness=slice_thickness,
                         exit_planes=exit_planes,
                         device=device,
                         plane=plane,
                         origin=origin,
                         box=box,
                         periodic=periodic)

    @property
    def frozen_phonons(self):
        return self._frozen_phonons

    @property
    def repetitions(self):
        return self._repetitions

    @property
    def gridrefinement(self):
        return self._gridrefinement

    @property
    def calculators(self):
        return self._calculators

    def _get_all_electron_density(self):

        calculator = self.calculators[0] if isinstance(self.calculators, list) else self.calculators

        # assert len(self.calculators) == 1

        calculator = DummyGPAW.from_generic(calculator)

        atoms = self.frozen_phonons.atoms

        if self.repetitions != (1, 1, 1):
            cell_cv = calculator.gd.cell_cv * self.repetitions
            N_c = tuple(n_c * rep for n_c, rep in zip(calculator.gd.N_c, self.repetitions))
            gd = calculator.gd.new_descriptor(N_c=N_c, cell_cv=cell_cv)
            atoms = atoms * self.repetitions
            nt_sG = np.tile(calculator.nt_sG, self.repetitions)
        else:
            gd = calculator.gd
            nt_sG = calculator.nt_sG

        random_atoms = self.frozen_phonons.randomize(atoms)

        return get_all_electron_density(nt_sG=nt_sG,
                                        gd=gd,
                                        D_asp=calculator.D_asp,
                                        setups=calculator.setups,
                                        atoms=random_atoms,
                                        gridrefinement=self.gridrefinement)

    def generate_slices(self, first_slice: int = 0, last_slice: int = None):
        if last_slice is None:
            last_slice = len(self)

        atoms = self.frozen_phonons.atoms * self.repetitions
        random_atoms = self.frozen_phonons.randomize(atoms)

        ewald_parametrization = EwaldParametrization(width=1)

        ewald_potential = Potential(atoms=random_atoms,
                                    gpts=self.gpts,
                                    sampling=self.sampling,
                                    parametrization=ewald_parametrization,
                                    slice_thickness=self.slice_thickness,
                                    projection='finite',
                                    integral_space='real',
                                    plane=self.plane,
                                    box=self.box,
                                    origin=self.origin,
                                    exit_planes=self.exit_planes,
                                    device=self.device)

        array = self._get_all_electron_density()

        for slic in generate_slices(array, ewald_potential, first_slice=first_slice, last_slice=last_slice):
            yield slic

        # for potential_slice in charge_density_potential.generate_slices(first_slice=first_slice, last_slice=last_slice):
        #    yield potential_slice

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return self._frozen_phonons.ensemble_axes_metadata

    @property
    def num_frozen_phonons(self):
        return len(self.calculators)

    @property
    def ensemble_shape(self):
        return self._frozen_phonons.ensemble_shape

    @staticmethod
    def _gpaw_potential(*args, frozen_phonons_partial, **kwargs):
        args = args[0]
        if hasattr(args, 'item'):
            args = args.item()

        if args['frozen_phonons'] is not None:
            frozen_phonons = frozen_phonons_partial(args['frozen_phonons'])
        else:
            frozen_phonons = None

        calculators = args['calculators']

        return GPAWPotential(calculators, frozen_phonons=frozen_phonons, **kwargs)

    def from_partitioned_args(self):
        kwargs = self.copy_kwargs(exclude=('calculators', 'frozen_phonons'))

        frozen_phonons_partial = self.frozen_phonons.from_partitioned_args()

        return partial(self._gpaw_potential, frozen_phonons_partial=frozen_phonons_partial, **kwargs)

    def partition_args(self, chunks: int = 1, lazy: bool = True):

        chunks = self.validate_chunks(chunks)

        def frozen_phonons(calculators, frozen_phonons):
            arr = np.zeros((1,), dtype=object)
            arr.itemset(0, {'calculators': calculators, 'frozen_phonons': frozen_phonons})
            return arr

        calculators = self.calculators

        if isinstance(self.frozen_phonons, FrozenPhonons):
            array = np.zeros(len(self.frozen_phonons), dtype=object)
            for i, fp in enumerate(self.frozen_phonons.partition_args(chunks, lazy=lazy)[0]):
                if lazy:
                    block = dask.delayed(frozen_phonons)(calculators, fp)

                    array.itemset(i, da.from_delayed(block, shape=(1,), dtype=object))
                else:
                    array.itemset(i, frozen_phonons(calculators, fp))

            if lazy:
                array = da.concatenate(list(array))

            return array,

        else:
            if len(self.ensemble_shape) == 0:
                array = np.zeros((1,), dtype=object)
                calculators = [calculators]
            else:
                array = np.zeros(self.ensemble_shape[0], dtype=object)

            for i, calculator in enumerate(calculators):

                if len(self.ensemble_shape) > 0:
                    calculator = [calculator]

                if lazy:
                    calculator = dask.delayed(calculator)
                    block = da.from_delayed(dask.delayed(frozen_phonons)(calculator, None), shape=(1,), dtype=object)
                else:
                    block = frozen_phonons(calculator, None)

                array.itemset(i, block)

            if lazy:
                return da.concatenate(list(array)),
            else:
                return array,
