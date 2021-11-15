import numpy as np

from abtem.core.utils import generate_chunks
from abtem.potentials.infinite import infinite_potential_projections
from abtem.structures.slicing import SliceIndexedAtoms, SlicedAtoms


class ProjectedPotential:

    def __init__(self, slice_thickness, chunks: int = 1):
        self._slice_thickness = slice_thickness
        self._chunks = chunks
        self._chunk_limits = [x for x in generate_chunks(len(slice_thickness), chunks=chunks)]

    @property
    def chunk_limits(self):
        return self._chunk_limits

    @property
    def chunks(self):
        return self._chunks

    @property
    def slice_thickness(self):
        return self._slice_thickness

    def get_chunk(self, first_slice, last_slice):
        pass

    def get_slice(self, i):
        return self.get_chunk(i, i + 1)


class InfiniteProjectePotential(ProjectedPotential):

    def __init__(self,
                 atoms,
                 sampling,
                 slice_thickness,
                 scattering_factors,
                 chunks=1):
        self._scattering_factors = scattering_factors
        self._gpts = scattering_factors.shape[-2:]
        self._sampling = sampling
        self._slice_thickness = slice_thickness
        self._sliced_atoms = SliceIndexedAtoms(atoms=atoms, num_slices=len(self._slice_thickness))
        super().__init__(slice_thickness=slice_thickness, chunks=chunks)

    def get_chunk(self, first_slice, last_slice):
        positions, numbers, slice_idx = self._sliced_atoms.get_atoms_in_slices(first_slice, last_slice)
        shape = (last_slice - first_slice,) + self._gpts
        return infinite_potential_projections(positions, numbers, slice_idx, shape, self._sampling,
                                              self._scattering_factors)


class FiniteProjectedPotential(ProjectedPotential):

    def __init__(self,
                 atoms,
                 atomic_potentials,
                 gpts,
                 slice_thickness,
                 plane,
                 box,
                 xp,
                 chunks=1):

        self._atoms = atoms
        self._atomic_potentials = atomic_potentials
        self._gpts = gpts
        self._plane = plane
        self._box = box
        self._xp = xp

        super().__init__(slice_thickness=slice_thickness, chunks=chunks)

    @property
    def slice_thickness(self):
        return self._slice_thickness

    def _get_chunk(self, first_slice, last_slice):
        extent = np.diag(self._atoms.cell)[:2]

        sampling = (extent[0] / self._gpts[0], extent[1] / self._gpts[1])

        array = self._xp.zeros((last_slice - first_slice,) + self._gpts, dtype=np.float32)

        cutoffs = {Z: atomic_potential.cutoff for Z, atomic_potential in self._atomic_potentials.items()}
        sliced_atoms = SlicedAtoms(self._atoms, self._slice_thickness, plane=self._plane, box=self._box,
                                   padding=cutoffs)

        for i, slice_idx in enumerate(range(first_slice, last_slice)):
            for Z, atomic_potential in self._atomic_potentials.items():
                atoms = sliced_atoms.get_atoms_in_slices(slice_idx, atomic_number=Z)
                a = sliced_atoms.slice_limits[slice_idx][0] - atoms.positions[:, 2]
                b = sliced_atoms.slice_limits[slice_idx][1] - atoms.positions[:, 2]
                atomic_potential.project_on_grid(array[i], sampling, atoms.positions, a, b)

        return array
