from functools import partial
from functools import partial
from typing import Tuple, List, Union

import dask
import dask.array as da
import numpy as np
from ase.cell import Cell

from abtem.core.axes import ThicknessAxis, FrozenPhononsAxis, AxisMetadata
from abtem.core.backend import get_array_module
from abtem.core.chunks import validate_chunks, iterate_chunk_ranges
from abtem.core.grid import Grid, HasGridMixin
from abtem.potentials.temperature import AbstractFrozenPhonons, DummyFrozenPhonons, validate_seeds
from abtem.potentials.potentials import AbstractPotential, validate_exit_planes, PotentialArray, PotentialBuilder


class CrystalPotential(PotentialBuilder):
    """
    The crystal potential may be used to represent a potential consisting of a repeating unit. This may allow
    calculations to be performed with lower computational cost by calculating the potential unit once and repeating it.

    If the potential units is a potential with frozen phonons it is treated as an ensemble from which each repeating
    unit along the z-direction is randomly drawn. If `num_frozen_phonons` an ensemble of crystal potentials are created
    each with a random seed for choosing potential units.

    Parameters
    ----------
    potential_unit : AbstractPotential
        The potential unit to assemble the crystal potential from.
    repetitions : three int
        The repetitions of the potential in x, y and z.
    num_frozen_phonons : int, optional
        Number of frozen phonon configurations assembled from the potential units.
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the slice indices after which an
        exit plane is desired, and hence during a multislice simulation a measurement is created. If `exit_planes` is
        an integer a measurement will be collected every `exit_planes` number of slices.
    seeds: int or sequence of int
        Seed for the random number generator(rng), or one seed for each rng in the frozen phonon ensemble.
    """

    def __init__(self,
                 potential_unit: AbstractPotential,
                 repetitions: Tuple[int, int, int],
                 num_frozen_phonons: int = None,
                 exit_planes: int = None,
                 seeds: Union[int, Tuple[int, ...]] = None):

        if num_frozen_phonons is None and seeds is None:
            self._seeds = None
        else:
            if num_frozen_phonons is None and seeds:
                num_frozen_phonons = len(seeds)
            elif num_frozen_phonons is None and seeds is None:
                num_frozen_phonons = 1

            self._seeds = validate_seeds(seeds, num_frozen_phonons)

        # if (potential_unit.num_frozen_phonon_configs == 1) & (num_frozen_phonon_configs > 1):
        #     warnings.warn('"num_frozen_phonon_configs" is greater than one, but the potential unit does not have'
        #                   'frozen phonons')
        #
        # if (potential_unit.num_frozen_phonon_configs > 1) & (num_frozen_phonon_configs == 1):
        #     warnings.warn('the potential unit has frozen phonons, but "num_frozen_phonon_configs" is set to 1')

        gpts = potential_unit.gpts[0] * repetitions[0], potential_unit.gpts[1] * repetitions[1]
        extent = potential_unit.extent[0] * repetitions[0], potential_unit.extent[1] * repetitions[1]
        sampling = extent[0] / gpts[0], extent[1] / gpts[1]
        box = extent + (potential_unit.thickness * repetitions[2],)
        slice_thickness = potential_unit.slice_thickness * repetitions[2]
        super().__init__(
            gpts=gpts,
            sampling=sampling,
            cell=Cell(np.diag(box)),
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=potential_unit.device,
            plane='xy',
            origin=(0., 0., 0.),
            box=box,
            periodic=True

        )

        self._potential_unit = potential_unit
        self._repetitions = repetitions

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        if self._seeds is None:
            return ()
        else:
            return self.num_frozen_phonons,

    @property
    def num_frozen_phonons(self):
        if self._seeds is None:
            return 1
        else:
            return len(self._seeds)

    @property
    def seeds(self):
        return self._seeds

    @property
    def potential_unit(self) -> AbstractPotential:
        return self._potential_unit

    @HasGridMixin.gpts.setter
    def gpts(self, gpts):
        if not ((gpts[0] % self.repetitions[0] == 0) and (gpts[1] % self.repetitions[0] == 0)):
            raise ValueError('gpts must be divisible by the number of potential repetitions')
        self.grid.gpts = gpts
        self._potential_unit.gpts = (gpts[0] // self._repetitions[0], gpts[1] // self._repetitions[1])

    @HasGridMixin.sampling.setter
    def sampling(self, sampling):
        self.sampling = sampling
        self._potential_unit.sampling = sampling

    @property
    def repetitions(self) -> Tuple[int, int, int]:
        return self._repetitions

    @property
    def num_slices(self) -> int:
        return self._potential_unit.num_slices * self.repetitions[2]

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        if self.seeds is None:
            return []
        else:
            return [FrozenPhononsAxis(values=tuple(range(self.num_frozen_phonons)), _ensemble_mean=True)]

    @staticmethod
    def _wrap_partition_args(*args):
        arr = np.zeros((1,), dtype=object)
        arr.itemset(0, {'potential_args': args[0], 'seeds': args[1], 'num_frozen_phonons': args[2]})
        return arr

    def partition_args(self, chunks: int = 1, lazy: bool = True):
        chunks = validate_chunks(self.ensemble_shape, chunks)

        if len(self.ensemble_shape) == 0:
            array = np.zeros((1,), dtype=object)
            chunks = ((1,),)
        else:
            array = np.zeros(len(chunks[0]), dtype=object)

        for block_indices, chunk_range in iterate_chunk_ranges(chunks):
            if self.seeds is None:
                seeds = None
                num_frozen_phonons = None
            else:
                seeds = self.seeds[chunk_range[0]]
                num_frozen_phonons = len(seeds)

            potential_unit = self.potential_unit.partition_args(-1, lazy=lazy)[0]

            if lazy:
                block = dask.delayed(self._wrap_partition_args)(potential_unit, seeds, num_frozen_phonons)
                block = da.from_delayed(block, shape=(1,), dtype=object)
            else:
                block = self._wrap_partition_args(potential_unit, seeds, num_frozen_phonons)

            array.itemset(block_indices[0], block)

        if lazy:
            array = da.concatenate(array)

        return array,

    @staticmethod
    def _crystal_potential(*args, potential_partial, **kwargs):
        args = args[0]
        if hasattr(args, 'item'):
            args = args.item()

        potential_args = args['potential_args']
        if hasattr(potential_args, 'item'):
            potential_args = potential_args.item()

        potential_unit = potential_partial(potential_args)

        kwargs['seeds'] = args['seeds']
        kwargs['num_frozen_phonons'] = args['num_frozen_phonons']
        potential = CrystalPotential(potential_unit, **kwargs)

        return potential

    def from_partitioned_args(self):
        kwargs = self.copy_kwargs(exclude=('potential_unit', 'seeds', 'num_frozen_phonons'))
        potential_partial = self.potential_unit.from_partitioned_args()
        return partial(self._crystal_potential, potential_partial=potential_partial, **kwargs)

    def generate_slices(self, first_slice: int = 0, last_slice: int = None):

        if hasattr(self.potential_unit, 'array'):
            potentials = self.potential_unit
        else:
            potentials = self.potential_unit.build(lazy=False)

        if len(potentials.shape) == 3:
            potentials = potentials.expand_dims(axis=0)

        # first_layer = first_slice // self._potential_unit.num_slices
        # if last_slice is None:
        #     last_layer = self.repetitions[2]
        # else:
        #     last_layer = last_slice // self._potential_unit.num_slices
        #
        # first_slice = first_slice % self._potential_unit.num_slices
        # last_slice = None

        # if self.random_state:
        #    random_state = self.random_state
        # else:
        #    random_state = np.random.RandomState()
        if self.seeds is None:
            rng = np.random.default_rng(self.seeds)
        else:
            rng = np.random.default_rng(self.seeds[0])

        for i in range(self.repetitions[2]):
            generator = potentials[rng.integers(0, potentials.shape[0])].generate_slices()
            for i in range(len(self.potential_unit)):
                slic = next(generator).tile(self.repetitions[:2])

                yield slic
