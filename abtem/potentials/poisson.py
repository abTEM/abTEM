from typing import Tuple, Union

import numpy as np
from ase import Atoms
from ase import units
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

from abtem.core import dask
from abtem.core.backend import copy_to_device, get_array_module, _validate_device
from abtem.core.fft import fft_crop
from abtem.core.grid import Grid
from abtem.core.utils import generate_chunks
from abtem.potentials.parametrizations import EwaldParametrization
from abtem.potentials.potentials import AbstractPotential, Potential, PotentialArray
from abtem.potentials.temperature import MDFrozenPhonons, LazyAtoms
from abtem.structures.slicing import _validate_slice_thickness
import dask.array as da

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


def superpose_deltas(positions, array, cell, scale=1):
    inverse_cell = np.linalg.inv(np.array(cell))
    scaled_positions = np.dot(positions, inverse_cell)
    scaled_positions *= array.shape
    corners = np.floor(scaled_positions).astype(int)
    shape = array.shape

    xi = np.array([corners[:, 0], (corners[:, 0] + 1) % shape[0]]).T[:, :, None, None]
    xi = np.tile(xi, (1, 1, 2, 2)).reshape((len(positions), -1))
    yi = np.array([corners[:, 1], (corners[:, 1] + 1) % shape[1]]).T[:, None, :, None]
    yi = np.tile(yi, (1, 2, 1, 2)).reshape((len(positions), -1))
    zi = np.array([corners[:, 2], (corners[:, 2] + 1) % shape[2]]).T[:, None, None, :]
    zi = np.tile(zi, (1, 2, 2, 1)).reshape((len(positions), -1))

    x, y, z = (scaled_positions - corners).T
    x = np.array([1 - x, x]).T[:, :, None, None]
    y = np.array([1 - y, y]).T[:, None, :, None]
    z = np.array([1 - z, z]).T[:, None, None, :]

    values = (x * y * z).reshape((len(positions), -1)) * scale
    array[xi, yi, zi] += values
    return array


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

    compensation = np.zeros_like(array)
    for number in np.unique(atoms.numbers):
        superpose_deltas(atoms[atoms.numbers == number].positions, compensation, atoms.cell, scale=number)
        # import matplotlib.pyplot as plt
        # plt.imshow(compensation.sum(-1))
        # plt.show()
    g = fourier_gaussian(kx, ky, kz)
    array = np.fft.fftn(array) + np.fft.fftn(compensation) * g / pixel_volume

    # array = np.fft.fftn(array) * g / pixel_volume

    # for atom in atoms:
    #     scale = atom.number / pixel_volume
    #     x, y, z = atom.position
    #
    #     if width > 0.:
    #         array += scale * g * fourier_shift(kx, ky, kz, x, y, z)
    #     else:
    #         array += scale * fourier_shift(kx, ky, kz, x, y, z)

    return _solve_fourier_space(array, kx, ky, kz)


def interpolate_between_cells(array, new_shape, old_cell, new_cell, offset=(0., 0., 0.), order=3):
    x = np.linspace(0, 1, new_shape[0], endpoint=False)
    y = np.linspace(0, 1, new_shape[1], endpoint=False)
    z = np.linspace(0, 1, new_shape[2], endpoint=False)

    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    coordinates = np.array([x.ravel(), y.ravel(), z.ravel()]).T
    coordinates = np.dot(coordinates, new_cell) + offset

    padded_array = np.pad(array, ((order,) * 2,) * 3, mode='wrap')

    padded_cell = old_cell.copy()
    padded_cell[:, 0] *= (array.shape[0] + order) / array.shape[0]
    padded_cell[:, 1] *= (array.shape[1] + order) / array.shape[1]
    padded_cell[:, 2] *= (array.shape[2] + order) / array.shape[2]

    inverse_old_cell = np.linalg.inv(np.array(old_cell))
    mapped_coordinates = np.dot(coordinates, inverse_old_cell) % 1.0
    mapped_coordinates *= array.shape
    mapped_coordinates += order

    interpolated = map_coordinates(padded_array, mapped_coordinates.T, mode='wrap', order=order)
    interpolated = interpolated.reshape(new_shape)
    return interpolated


class ChargeDensityPotential(AbstractPotential):

    def __init__(self,
                 atoms: Union[Atoms, MDFrozenPhonons],
                 charge_density: np.ndarray = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 slice_thickness: float = .5,
                 box: Tuple[float, float, float] = None,
                 x_vector: str = 'x',
                 y_vector: str = 'y',
                 origin: Tuple[float, float, float] = (0., 0., 0.),
                 chunks: int = 1,
                 device: str = None,
                 fft_singularities: bool = False):

        self._charge_density = charge_density
        self._atoms = atoms
        self._fft_singularities = fft_singularities
        self._chunks = chunks
        self._device = _validate_device(device)

        box = tuple(np.diag(atoms.cell))

        self._grid = Grid(extent=box[:2], gpts=gpts, sampling=sampling, lock_extent=True)

        #if fft_singularities and ((box is not None) or (origin != (0., 0., 0.)) or (plane != 'xy')):
        #   raise NotImplementedError()

        # super().__init__(atoms=atoms,
        #                  gpts=gpts,
        #                  sampling=sampling,
        #                  slice_thickness=slice_thickness,
        #                  plane=plane,
        #                  box=box,
        #                  origin=origin,
        #                  chunks=chunks,
        #                  device=device)

        slice_thickness = _validate_slice_thickness(slice_thickness, thickness=box[2])

        super().__init__(slice_thickness=slice_thickness)

        self._ewald_parametrization = EwaldParametrization(width=1)

        atoms = LazyAtoms(atoms)

        self._ewald_potential = Potential(atoms=atoms,
                                          gpts=gpts,
                                          sampling=sampling,
                                          parametrization=self._ewald_parametrization,
                                          slice_thickness=slice_thickness,
                                          projection='finite',
                                          x_vector=x_vector,
                                          y_vector=y_vector,
                                          box=box,
                                          origin=origin)

        self._electron_potential = None

    @property
    def atoms(self):
        return self._atoms

    @property
    def device(self):
        return self._device

    def build(self) -> 'PotentialArray':
        self.grid.check_is_defined()
        xp = get_array_module(self.device)

        def get_chunk(potential, first_slice, last_slice):
            return potential.get_chunk(first_slice, last_slice).array

        array = []
        if hasattr(self.atoms, 'compute'):
            potential = self.to_delayed()
        else:
            potential = self

        for first_slice, last_slice in generate_chunks(len(self), chunks=self._chunks):
            shape = (last_slice - first_slice,) + self.gpts

            if hasattr(potential, 'compute'):
                new_chunk = dask.delayed(get_chunk)(potential, first_slice, last_slice)
                new_chunk = da.from_delayed(new_chunk, shape=shape, meta=xp.array((), dtype=np.float32))
            else:
                new_chunk = get_chunk(potential, first_slice, last_slice)

            array.append(new_chunk)

        # if lazy:
        array = da.concatenate(array)
        # else:
        #    array = np.concatenate(array)

        return PotentialArray(array, self.slice_thickness, extent=self.extent)

    @property
    def frozen_phonons(self):
        pass

    def get_configurations(self, lazy=False):
        return [self]

    #def get_frozen_phonon_potentials(self, lazy=False):
    #    return [self]

    def _get_compensated_potential(self):
        if self._electron_potential is not None:
            return self._electron_potential

        self._electron_potential = solve_point_charges(self.atoms,
                                                       array=-self._charge_density,
                                                       width=self._ewald_parametrization._width)
        return self._electron_potential

    def _get_chunk_real_space(self, first_slice, last_slice):
        # def create_potential_interpolator(atoms, charge_density):
        #     array = solve_point_charges(atoms,
        #                                 array=-charge_density,
        #                                 width=self._ewald_parametrization._width)
        #
        #     padded_array = np.zeros((array.shape[0] + 1, array.shape[1] + 1, array.shape[2] + 1))
        #     padded_array[:-1, :-1, :-1] = array
        #     padded_array[-1] = padded_array[0]
        #     padded_array[:, -1] = padded_array[:, 0]
        #     padded_array[:, :, -1] = padded_array[:, :, 0]
        #
        #     x = np.linspace(0, 1, padded_array.shape[0], endpoint=True)
        #     y = np.linspace(0, 1, padded_array.shape[1], endpoint=True)
        #     z = np.linspace(0, 1, padded_array.shape[2], endpoint=True)
        #     return RegularGridInterpolator((x, y, z), padded_array)
        #
        # def interpolate_slice(a, b, h, gpts, old_cell, new_cell, interpolator):
        #     x = np.linspace(0, 1, gpts[0], endpoint=False)
        #     y = np.linspace(0, 1, gpts[1], endpoint=False)
        #     z = np.linspace(0, 1, int((b - a) / h), endpoint=False)
        #
        #     x, y, z = np.meshgrid(x, y, z, indexing='ij')
        #     propagation_vector = new_cell[2] / np.linalg.norm(new_cell[2])
        #
        #     new_cell = new_cell.copy()
        #     new_cell[2] = propagation_vector * (b - a)
        #     offset = a * propagation_vector
        #
        #     points = np.array([x.ravel(), y.ravel(), z.ravel()]).T
        #     points = np.dot(points, new_cell) + offset
        #
        #     P_inv = np.linalg.inv(np.array(old_cell))
        #     scaled_points = np.dot(points, P_inv) % 1.0
        #
        #     interpolated = interpolator(scaled_points)
        #     interpolated = interpolated.reshape(x.shape).mean(-1).reshape(gpts).astype(np.float32)
        #     return interpolated
        #
        # def interpolate_chunk(first_slice, last_slice, slice_limits, h, gpts, old_cell, new_cell, interpolator):
        #     chunk = np.zeros((last_slice - first_slice,) + gpts, dtype=np.float32)
        #     for i, (a, b) in enumerate(slice_limits[first_slice:last_slice]):
        #         chunk[i] = interpolate_slice(a, b, h, gpts, old_cell, new_cell, interpolator) * (b - a)
        #     return chunk

        # old_cell = self.atoms.cell
        # new_cell = np.diag(self.box)
        #
        # interpolator = create_potential_interpolator(self.atoms, self._charge_density)
        # h = min(self.sampling)
        # chunk = interpolate_chunk(first_slice,
        #                           last_slice,
        #                           self.slice_limits,
        #                           h,
        #                           self.gpts,
        #                           old_cell,
        #                           new_cell,
        #                           interpolator)

        array = self._get_compensated_potential()

        chunk = np.zeros((last_slice - first_slice,) + self.gpts, dtype=np.float32)

        for i, (a, b) in enumerate(self.slice_limits[first_slice:last_slice]):
            slice_shape = self.gpts + (int((b - a) / min(self.sampling)),)
            new_cell = self.atoms.cell.copy()
            new_cell[2, 2] = b - a
            propagation_vector = new_cell[2] / np.linalg.norm(new_cell[2])
            offset = propagation_vector * a
            dz = (b - a) / slice_shape[-1]
            chunk[i] = interpolate_between_cells(array,
                                                 slice_shape,
                                                 self.atoms.cell,
                                                 new_cell,
                                                 offset
                                                 ).sum(-1) * dz

        potential = self._ewald_potential.get_chunk(first_slice, last_slice)
        potential._array = potential._array + copy_to_device(chunk, potential._array)
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

    def __copy__(self):
        raise NotImplementedError

    def to_delayed(self):
        raise NotImplementedError
