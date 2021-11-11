from abtem.basic.utils import generate_chunks
from abtem.potentials.infinite import infinite_potential_projections
from abtem.structures.slicing import SliceIndexedAtoms


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
