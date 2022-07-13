from functools import partial
from typing import Tuple, Union

import dask
import dask.array as da
import numpy as np
from ase import Atoms
from ase import units
from ase.cell import Cell
from scipy.ndimage import map_coordinates

from abtem.core.backend import copy_to_device
from abtem.core.fft import fft_crop, fft_interpolate, ifftn, fftn
from abtem.potentials.parametrizations import EwaldParametrization
from abtem.potentials.potentials import Potential, PotentialBuilder
from abtem.potentials.temperature import MDFrozenPhonons, DummyFrozenPhonons
from abtem.structures.structures import plane_to_axes

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)


def spatial_frequencies(shape: Tuple[int, int, int], cell: Cell, meshgrid: bool = True) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx, ky, kz = (np.fft.fftfreq(n, d=1 / n) for n in shape)

    if not meshgrid:
        assert np.allclose(np.diag(cell.array), cell.lengths())
        lengths = cell.reciprocal().lengths()
        kx = kx[:, None, None] * lengths[0]
        ky = ky[None, :, None] * lengths[1]
        kz = kz[None, None, :] * lengths[2]
        return kx, ky, kz

    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    kp = np.array([kx.ravel(), ky.ravel(), kz.ravel()]).T
    kx, ky, kz = np.dot(kp, cell.reciprocal().array).T
    kx, ky, kz = kx.reshape(shape), ky.reshape(shape), kz.reshape(shape)
    return kx, ky, kz


def _solve_fourier_space(charge, kx, ky, kz, fourier_space_out=False):
    k2 = 2 ** 2 * np.pi ** 2 * (kx ** 2 + ky ** 2 + kz ** 2)
    k2[0, 0, 0] = 1.
    # V = np.zeros(charge.shape, dtype=np.complex)

    charge /= k2 * eps0

    # nonzero = np.ones_like(V, dtype=bool)
    # nonzero[0, 0, 0] = False

    # V[nonzero] = charge[nonzero] / k2[nonzero] / eps0
    # V[0, 0, 0] = 0

    if fourier_space_out:
        return charge
    else:
        return ifftn(charge, overwrite_x=True).real


def solve_poisson_equation(charge_density, cell):
    kx, ky, kz = spatial_frequencies(charge_density.shape, cell)
    return _solve_fourier_space(np.fft.fftn(charge_density), kx, ky, kz)


def superpose_deltas(positions, array, cell, scale=1):
    inverse_cell = np.linalg.inv(np.array(cell))
    scaled_positions = np.dot(positions, inverse_cell)
    scaled_positions *= array.shape
    corners = np.floor(scaled_positions).astype(int)
    shape = array.shape

    xi = np.array([corners[:, 0] % shape[0], (corners[:, 0] + 1) % shape[0]]).T[:, :, None, None]
    xi = np.tile(xi, (1, 1, 2, 2)).reshape((len(positions), -1))
    yi = np.array([corners[:, 1] % shape[1], (corners[:, 1] + 1) % shape[1]]).T[:, None, :, None]
    yi = np.tile(yi, (1, 2, 1, 2)).reshape((len(positions), -1))
    zi = np.array([corners[:, 2] % shape[2], (corners[:, 2] + 1) % shape[2]]).T[:, None, None, :]
    zi = np.tile(zi, (1, 2, 2, 1)).reshape((len(positions), -1))

    x, y, z = (scaled_positions - corners).T
    x = np.array([1 - x, x]).T[:, :, None, None]
    y = np.array([1 - y, y]).T[:, None, :, None]
    z = np.array([1 - z, z]).T[:, None, None, :]

    values = (x * y * z).reshape((len(positions), -1)) * scale
    array[xi, yi, zi] += values
    return array


def fourier_space_shift_3d(kx, ky, kz, x, y, z):
    return np.exp(-2 * np.pi * 1j * (kx * x + ky * y + kz * z))


def fourier_space_gaussian_3d(kx, ky, kz, width):
    a = np.sqrt(1 / (2 * width ** 2)) / (2 * np.pi)
    return np.exp(- 1 / (4 * a ** 2) * (kx ** 2 + ky ** 2 + kz ** 2))


def real_space_point_charges(atoms, array):
    pixel_volume = np.prod(np.diag(atoms.cell)) / np.prod(array.shape)

    for number in np.unique(atoms.numbers):
        superpose_deltas(atoms[atoms.numbers == number].positions, array, atoms.cell, scale=number / pixel_volume)

    return array


def solve_point_charges(atoms: Atoms,
                        array: np.ndarray,
                        width: float = 0.,
                        real_space_point_charges: bool = False,
                        fourier_space_in: bool = False,
                        fourier_space_out: bool = False):
    pixel_volume = np.prod(np.diag(atoms.cell)) / np.prod(array.shape)

    kx, ky, kz = spatial_frequencies(array.shape, atoms.cell, meshgrid=not atoms.cell.orthorhombic)

    gaussian = fourier_space_gaussian_3d(kx, ky, kz, width) if width > 0. else None

    # real_space_point_charges(atoms, array)

    for atom in atoms:
        scale = atom.number / pixel_volume

        if gaussian is not None:
            array += scale * gaussian * fourier_space_shift_3d(kx, ky, kz, *atom.position)
        else:
            array += scale * fourier_space_shift_3d(kx, ky, kz, *atom.position)

    array = _solve_fourier_space(array, kx, ky, kz, fourier_space_out=fourier_space_out)

    return array


def interpolate_between_cells(array, new_shape, old_cell, new_cell, offset=(0., 0., 0.), order=2):
    x = np.linspace(0, 1, new_shape[0], endpoint=False)
    y = np.linspace(0, 1, new_shape[1], endpoint=False)
    z = np.linspace(0, 1, new_shape[2], endpoint=False)

    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    coordinates = np.array([x.ravel(), y.ravel(), z.ravel()]).T
    coordinates = np.dot(coordinates, new_cell) + offset

    padding = 3
    padded_array = np.pad(array, ((padding,) * 2,) * 3, mode='wrap')

    # padded_cell = old_cell.copy()
    # padded_cell[:, 0] *= (array.shape[0] + order) / array.shape[0]
    # padded_cell[:, 1] *= (array.shape[1] + order) / array.shape[1]
    # padded_cell[:, 2] *= (array.shape[2] + order) / array.shape[2]

    inverse_old_cell = np.linalg.inv(np.array(old_cell))
    mapped_coordinates = np.dot(coordinates, inverse_old_cell) % 1.0
    mapped_coordinates *= array.shape
    mapped_coordinates += padding
    # mapped_coordinates = mapped_coordinates % array.shape

    interpolated = map_coordinates(padded_array, mapped_coordinates.T, mode='wrap', order=order)
    interpolated = interpolated.reshape(new_shape)
    return interpolated


def _interpolate_slice(array, cell, gpts, sampling, a, b):
    slice_shape = gpts + (int((b - a) / (min(sampling))),)

    slice_box = np.diag((gpts[0] * sampling[0], gpts[1] * sampling[1]) + (b - a,))

    slice_array = interpolate_between_cells(array,
                                            slice_shape,
                                            cell,
                                            slice_box,
                                            (0, 0, a))

    return np.trapz(slice_array, axis=-1, dx=(b - a) / (slice_shape[-1] - 1))


def generate_slices(array,
                    ewald_potential,
                    first_slice: int = 0,
                    last_slice: int = None):
    if last_slice is None:
        last_slice = len(ewald_potential)

    if ewald_potential.plane != 'xy':
        axes = plane_to_axes(ewald_potential.plane)
        array = np.moveaxis(array, axes[:2], (0, 1))
        atoms = ewald_potential._transformed_atoms()
    else:
        atoms = ewald_potential.frozen_phonons.atoms
    # assert len(array.shape) == 3

    atoms = ewald_potential.frozen_phonons.randomize(atoms)

    array = -fftn(array, overwrite_x=True)

    array = fft_crop(array, array.shape[:2] + (ewald_potential.num_slices,), normalize=True)

    array = solve_point_charges(atoms,
                                array=array,
                                width=ewald_potential.integrator.parametrization.width,
                                fourier_space_in=True,
                                fourier_space_out=False)

    for i, ((a, b), slic) in enumerate(zip(ewald_potential.slice_limits[first_slice:last_slice],
                                           ewald_potential.generate_slices(first_slice, last_slice))):
        slice_array = _interpolate_slice(array,
                                         atoms.cell,
                                         ewald_potential.gpts,
                                         ewald_potential.sampling,
                                         a,
                                         b)

        slic._array = slic._array + copy_to_device(slice_array[None], slic.array)

        slic._array -= slic._array.min()
        yield slic


class ChargeDensityPotential(PotentialBuilder):

    def __init__(self,
                 atoms: Union[Atoms, MDFrozenPhonons],
                 charge_density: np.ndarray = None,
                 charge_density_cell: Cell = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 slice_thickness: Union[float, Tuple[float]] = .5,
                 plane: str = 'xy',
                 box: Tuple[float, float, float] = None,
                 origin: Tuple[float, float, float] = (0., 0., 0.),
                 periodic: bool = True,
                 exit_planes: int = None,
                 device: str = None, ):

        if hasattr(atoms, 'randomize'):
            self._frozen_phonons = atoms
        else:
            self._frozen_phonons = DummyFrozenPhonons(atoms)

        self._charge_density = charge_density.astype(np.float32)

        self._charge_density_cell = atoms.cell.copy() if charge_density_cell is None else charge_density_cell

        # if fft_singularities and ((box is not None) or (origin != (0., 0., 0.)) or (plane != 'xy')):
        #    raise NotImplementedError()

        super().__init__(gpts=gpts,
                         sampling=sampling,
                         cell=atoms.cell,
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
    def num_frozen_phonons(self):
        return len(self.frozen_phonons)

    @property
    def ensemble_axes_metadata(self):
        return self.frozen_phonons.ensemble_axes_metadata

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        return self.frozen_phonons.ensemble_shape

    @property
    def is_lazy(self):
        return isinstance(self.charge_density, da.core.Array)

    @property
    def charge_density(self):
        return self._charge_density

    @property
    def charge_density_cell(self):
        return self._charge_density_cell

    @staticmethod
    def _wrap_charge_density(charge_density, frozen_phonon):
        return np.array([{'charge_density': charge_density, 'atoms': frozen_phonon}], dtype=object)

    def partition_args(self, chunks: int = 1, lazy: bool = True):

        chunks = self.validate_chunks(chunks)

        charge_densities = self.charge_density

        if len(charge_densities.shape) == 3:
            charge_densities = charge_densities[None]
        elif len(charge_densities.shape) != 4:
            raise RuntimeError()

        if len(self.ensemble_shape) == 0:
            blocks = np.zeros((1,), dtype=object)
        else:
            blocks = np.zeros((len(chunks[0]),), dtype=object)

        if lazy:
            if not isinstance(charge_densities, da.core.Array):
                charge_densities = da.from_array(charge_densities, chunks=(1, -1, -1, -1))

            if charge_densities.shape[0] != self.ensemble_shape:
                charge_densities = da.tile(charge_densities, self.ensemble_shape + (1, 1, 1))

            charge_densities = charge_densities.to_delayed()

        elif hasattr(charge_densities, 'compute'):
            raise RuntimeError

        frozen_phonon_blocks = self._ewald_potential().frozen_phonons.partition_args(lazy=lazy)[0]

        for i, (charge_density, frozen_phonon) in enumerate(zip(charge_densities, frozen_phonon_blocks)):

            if lazy:
                block = dask.delayed(self._wrap_charge_density)(charge_density.item(), frozen_phonon)
                blocks.itemset(i, da.from_delayed(block, shape=(1,), dtype=object))

            else:
                blocks.itemset(i, self._wrap_charge_density(charge_density, frozen_phonon))

        if lazy:
            blocks = da.concatenate(list(blocks))

        return blocks,

    @staticmethod
    def _charge_density_potential(*args, frozen_phonons_partial, **kwargs):
        args = args[0]
        if hasattr(args, 'item'):
            args = args.item()

        args['atoms'] = frozen_phonons_partial(args['atoms'])

        kwargs.update(args)
        potential = ChargeDensityPotential(**kwargs)
        return potential

    def from_partitioned_args(self):
        kwargs = self.copy_kwargs(exclude=('atoms', 'charge_density'), cls=ChargeDensityPotential)
        frozen_phonons_partial = self._ewald_potential().frozen_phonons.from_partitioned_args()

        return partial(self._charge_density_potential, frozen_phonons_partial=frozen_phonons_partial, **kwargs)

    # def _fourier_space(self):
    #     fourier_density = -np.fft.fftn(self._charge_density)
    #     C = np.prod(self.gpts + (self.num_slices,)) / np.prod(fourier_density.shape)
    #     fourier_density = fft_crop(fourier_density, self.gpts + (self.num_slices,)) * C
    #     potential = solve_point_charges(self.atoms, array=fourier_density)
    #     potential = np.rollaxis(potential, 2)
    #     potential = potential * self.atoms.cell[2, 2] / potential.shape[0]

    def _interpolate_slice(self, array, cell, a, b):
        slice_shape = self.gpts + (int((b - a) / min(self.sampling)),)

        slice_box = np.diag(self.box[:2] + (b - a,))

        slice_array = interpolate_between_cells(array,
                                                slice_shape,
                                                cell,
                                                slice_box,
                                                (0, 0, a))

        return np.trapz(slice_array, axis=-1, dx=(b - a) / (slice_shape[-1] - 1))

    def _integrate_slice(self, array, a, b):
        dz = self.box[2] / array.shape[2]
        na = int(np.floor(a / dz))
        nb = int(np.floor(b / dz))
        slice_array = np.trapz(array[..., na:nb], axis=-1, dx=(b - a) / (nb - na - 1))
        return fft_interpolate(slice_array, new_shape=self.gpts, normalization='values')

    def _ewald_potential(self):
        ewald_parametrization = EwaldParametrization(width=1)

        return Potential(atoms=self.frozen_phonons,
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

    def generate_slices(self, first_slice: int = 0, last_slice: int = None):

        if last_slice is None:
            last_slice = len(self)

        if len(self.charge_density.shape) == 4:
            if self.charge_density.shape[0] > 1:
                raise RuntimeError()

            array = self.charge_density[0]
        elif len(self.charge_density.shape) == 3:
            array = self.charge_density
        else:
            raise RuntimeError()

        ewald_potential = self._ewald_potential()

        for slic in generate_slices(array, ewald_potential, first_slice=first_slice, last_slice=last_slice):
            yield slic
