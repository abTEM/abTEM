import dask
import dask.array as da
import numpy as np
from ase import units
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Union

from abtem.core.fft import fft_crop
from abtem.core.utils import generate_chunks
from abtem.potentials.parametrizations.ewald import ewald_sigma
from abtem.potentials.potentials import AbstractPotentialFromAtoms, Potential
from abtem.core.dask import computable
from ase import Atoms

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)


def spatial_frequencies(shape, cell):
    kx, ky, kz = np.meshgrid(*(np.fft.fftfreq(n, d=1 / n) for n in shape), indexing='ij')
    kp = np.array([kx.ravel(), ky.ravel(), kz.ravel()]).T
    kx, ky, kz = np.dot(kp, cell.reciprocal()).T
    kx, ky, kz = kx.reshape(shape), ky.reshape(shape), kz.reshape(shape)
    return kx, ky, kz


def _solve_fourier_space(charge, kx, ky, kz):
    k2 = 2 ** 2 * np.pi ** 2 * (kx ** 2 + ky ** 2 + kz ** 2)
    V = np.zeros(charge.shape, dtype=np.complex)

    nonzero = np.ones_like(V, dtype=bool)
    nonzero[0, 0, 0] = False

    V[nonzero] = charge[nonzero] / k2[nonzero]
    V[0, 0, 0] = 0

    V = np.fft.ifftn(V).real / eps0
    return V


def solve_poisson_equation(charge_density, cell):
    kx, ky, kz = spatial_frequencies(charge_density.shape, cell)
    return _solve_fourier_space(np.fft.fftn(charge_density), kx, ky, kz)


def solve_point_charges(atoms, shape=None, array=None, width=0.):
    if array is None:
        array = np.zeros(shape, dtype=np.complex64)

    def fourier_shift(kx, ky, kz, x, y, z):
        return np.exp(-2 * np.pi * 1j * (kx * x + ky * y + kz * z))

    def fourier_gaussian(kx, ky, kz):
        a = np.sqrt(1 / (2 * width ** 2)) / (2 * np.pi)
        return np.exp(- 1 / (4 * a ** 2) * (kx ** 2 + ky ** 2 + kz ** 2))

    kx, ky, kz = spatial_frequencies(array.shape, atoms.cell)

    pixel_volume = np.prod(np.diag(atoms.cell)) / np.prod(array.shape)
    for atom in atoms:
        scale = atom.number / pixel_volume
        x, y, z = atom.position

        if width > 0.:
            array += scale * fourier_gaussian(kx, ky, kz) * fourier_shift(kx, ky, kz, x, y, z)
        else:
            array += scale * fourier_shift(kx, ky, kz, x, y, z)

    return _solve_fourier_space(array, kx, ky, kz)


class ChargeDensityPotential(AbstractPotentialFromAtoms):

    def __init__(self,
                 atoms: Atoms,
                 charge_density: np.ndarray = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 slice_thickness: float = .5,
                 plane: str = 'xy',
                 box: Tuple[float, float, float] = None,
                 origin: Tuple[float, float, float] = (0., 0., 0.),
                 chunks: int = 1,
                 device: str = None,
                 fft_singularities: bool = False):

        self._charge_density = charge_density
        self._atoms = atoms
        self._fft_singularities = fft_singularities
        self._chunks = chunks

        if fft_singularities and ((box is not None) or (origin != (0., 0., 0.)) or (plane != 'xy')):
            raise NotImplementedError()

        super().__init__(atoms=atoms,
                         gpts=gpts,
                         sampling=sampling,
                         slice_thickness=slice_thickness,
                         plane=plane,
                         box=box,
                         origin=origin,
                         chunks=chunks,
                         device=device)

    @property
    def frozen_phonon_potentials(self):
        return [self]

    def _get_chunk_real_space(self, first_slice, last_slice):
        def create_potential_interpolator(atoms, charge_density):
            # atoms = atoms[atoms.numbers == 0]

            array = solve_point_charges(atoms,
                                        array=-np.fft.fftn(charge_density),
                                        width=ewald_sigma)

            padded_array = np.zeros((array.shape[0] + 1, array.shape[1] + 1, array.shape[2] + 1))
            padded_array[:-1, :-1, :-1] = array
            padded_array[-1] = padded_array[0]
            padded_array[:, -1] = padded_array[:, 0]
            padded_array[:, :, -1] = padded_array[:, :, 0]

            x = np.linspace(0, 1, padded_array.shape[0], endpoint=True)
            y = np.linspace(0, 1, padded_array.shape[1], endpoint=True)
            z = np.linspace(0, 1, padded_array.shape[2], endpoint=True)
            return RegularGridInterpolator((x, y, z), padded_array)

        def interpolate_slice(a, b, h, gpts, old_cell, new_cell, interpolator):
            x = np.linspace(0, 1, gpts[0], endpoint=False)
            y = np.linspace(0, 1, gpts[1], endpoint=False)
            z = np.linspace(0, 1, int((b - a) / h), endpoint=False)

            x, y, z = np.meshgrid(x, y, z, indexing='ij')
            propagation_vector = new_cell[2] / np.linalg.norm(new_cell[2])

            new_cell = new_cell.copy()
            new_cell[2] = propagation_vector * (b - a)
            offset = a * propagation_vector

            points = np.array([x.ravel(), y.ravel(), z.ravel()]).T
            points = np.dot(points, new_cell) + offset

            P_inv = np.linalg.inv(np.array(old_cell))
            scaled_points = np.dot(points, P_inv) % 1.0

            interpolated = interpolator(scaled_points)
            interpolated = interpolated.reshape(x.shape).mean(-1).reshape(gpts).astype(np.float32)
            return interpolated

        def interpolate_chunk(first_slice, last_slice, slice_limits, h, gpts, old_cell, new_cell, interpolator):
            chunk = np.zeros((last_slice - first_slice,) + gpts, dtype=np.float32)
            for i, (a, b) in enumerate(slice_limits[first_slice:last_slice]):
                chunk[i] = interpolate_slice(a, b, h, gpts, old_cell, new_cell, interpolator) * (b - a)
            return chunk

        old_cell = self.atoms.cell
        new_cell = np.diag(self.box)

        interpolator = create_potential_interpolator(self.atoms, self._charge_density)
        h = min(self.sampling)

        ewald_potential = Potential(atoms=self.atoms,
                                    gpts=self.gpts,
                                    parametrization='ewald',
                                    slice_thickness=self.slice_thickness,
                                    projection='finite',
                                    plane=self.plane,
                                    box=self.box,
                                    origin=self.origin)

        # array = []
        # for first_slice, last_slice in generate_chunks(len(self), chunks=self._chunks):
        #     # chunk = dask.delayed(interpolate_chunk)(first_slice,
        #     #                                         last_slice,
        #     #                                         self.slice_limits,
        #     #                                         h,
        #     #                                         self.gpts,
        #     #                                         old_cell,
        #     #                                         new_cell,
        #     #                                         interpolator)
        #     # array.append(da.from_delayed(chunk, shape=(last_slice - first_slice,) + self.gpts,
        #     #                              meta=np.array((), dtype=np.float32)))
        chunk = interpolate_chunk(first_slice,
                                  last_slice,
                                  self.slice_limits,
                                  h,
                                  self.gpts,
                                  old_cell,
                                  new_cell,
                                  interpolator)

        #array.append(chunk)

        #potential = ewald_potential.build()
        potential = ewald_potential.get_chunk(first_slice, last_slice)
        potential._array = potential._array + chunk #np.concatenate(array)
        potential._array -= potential._array.min()
        return potential

    def _get_chunk_fft(self, first_slice, last_slice):
        fourier_density = -np.fft.fftn(self._charge_density)
        C = np.prod(self.gpts + (self.num_slices,)) / np.prod(fourier_density.shape)
        fourier_density = fft_crop(fourier_density, self.gpts + (self.num_slices,)) * C
        potential = solve_point_charges(self.atoms, array=fourier_density)
        potential = np.rollaxis(potential, 2)
        potential = potential * self.atoms.cell[2, 2] / potential.shape[0]
        return potential

    def get_chunk(self, first_slice, last_slice):
        if self._fft_singularities:
            return self._get_chunk_fft(first_slice, last_slice)
        else:
            return self._get_chunk_real_space(first_slice, last_slice)

    def get_potential_configurations(self, lazy: bool = False):
        pass

    def __copy__(self):
        raise NotImplementedError
