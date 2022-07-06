from functools import partial
from typing import Tuple, Union

import numpy as np
from ase import Atoms
from ase import units
from ase.cell import Cell
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

import dask
from abtem.core.backend import copy_to_device, get_array_module
from abtem.core.fft import fft_crop
from abtem.potentials.parametrizations import EwaldParametrization
from abtem.potentials.potentials import Potential, AbstractPotential, PotentialArray, PotentialBuilder
from abtem.potentials.temperature import MDFrozenPhonons, AbstractFrozenPhonons, LazyAtoms
from abtem.structures.structures import plane_to_axes
from abtem.structures.slicing import validate_slice_thickness
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


# def solve_point_charges(atoms, shape=None, array=None, width=0.):
#     if array is None:
#         array = np.zeros(shape, dtype=np.complex64)
#
#     def fourier_shift(kx, ky, kz, x, y, z):
#         return np.exp(-2 * np.pi * 1j * (kx * x + ky * y + kz * z))
#
#     def fourier_gaussian(kx, ky, kz):
#         a = np.sqrt(1 / (2 * width ** 2)) / (2 * np.pi)
#         return np.exp(- 1 / (4 * a ** 2) * (kx ** 2 + ky ** 2 + kz ** 2))
#
#     kx, ky, kz = spatial_frequencies(array.shape, atoms.cell)
#
#     pixel_volume = np.prod(np.diag(atoms.cell)) / np.prod(array.shape)
#     for atom in atoms:
#         scale = atom.number / pixel_volume
#         x, y, z = atom.position
#
#         if width > 0.:
#             array += scale * fourier_gaussian(kx, ky, kz) * fourier_shift(kx, ky, kz, x, y, z)
#         else:
#             array += scale * fourier_shift(kx, ky, kz, x, y, z)
#
#     return _solve_fourier_space(array, kx, ky, kz)

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


class ChargeDensityPotential(PotentialBuilder):

    def __init__(self,
                 atoms: Union[Atoms, MDFrozenPhonons],
                 charge_density: np.ndarray = None,
                 charge_density_cell=None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 slice_thickness: float = .5,
                 plane: str = 'xy',
                 box: Tuple[float, float, float] = None,
                 origin: Tuple[float, float, float] = (0., 0., 0.),
                 exit_planes: int = None,
                 device: str = None, ):

        self._charge_density = charge_density

        self._charge_density_cell = atoms.cell.copy() if charge_density_cell is None else charge_density_cell

        # if fft_singularities and ((box is not None) or (origin != (0., 0., 0.)) or (plane != 'xy')):
        #    raise NotImplementedError()
        ewald_parametrization = EwaldParametrization(width=1)

        self._ewald_potential = Potential(atoms=atoms,
                                          gpts=gpts,
                                          sampling=sampling,
                                          parametrization=ewald_parametrization,
                                          slice_thickness=slice_thickness,
                                          projection='finite',
                                          integral_space='real',
                                          plane=plane,
                                          box=box,
                                          origin=origin,
                                          exit_planes=exit_planes,
                                          device=device)

        super().__init__()

        self._grid = self._ewald_potential.grid

    @property
    def num_frozen_phonons(self):
        return self.ewald_potential.num_frozen_phonons

    @property
    def ensemble_axes_metadata(self):
        return self.ewald_potential.ensemble_axes_metadata

    @property
    def ensemble_shape(self):
        return self.ewald_potential.ensemble_shape

    @property
    def base_shape(self):
        return self.ewald_potential.base_shape

    @property
    def is_lazy(self):
        return isinstance(self.charge_density, da.core.Array)

    @property
    def box(self):
        return self.ewald_potential.box

    @property
    def plane(self):
        return self.ewald_potential.plane

    @property
    def origin(self):
        return self.ewald_potential.origin

    @property
    def charge_density(self):
        return self._charge_density

    @property
    def charge_density_cell(self):
        return self._charge_density_cell

    @property
    def ewald_potential(self):
        return self._ewald_potential

    @property
    def ewald_parametrization(self):
        return self.ewald_potential.integrator.parametrization

    @property
    def device(self):
        return self.ewald_potential.device

    @property
    def slice_thickness(self) -> Tuple[float, ...]:
        return self.ewald_potential.slice_thickness

    @property
    def exit_planes(self) -> Tuple[int]:
        return self.ewald_potential.exit_planes

    @staticmethod
    def _wrap_charge_density(charge_density, frozen_phonon):
        return np.array([{'charge_density': charge_density, 'atoms': frozen_phonon}], dtype=object)

    def partition_args(self, chunks=None, lazy: bool = True):

        charge_densities = self.charge_density

        if len(self.ensemble_shape) == 0:
            charge_densities = charge_densities[None]
            blocks = np.zeros((1,), dtype=object)
        else:
            blocks = np.zeros((len(chunks[0]),), dtype=object)

        if lazy:
            if not isinstance(charge_densities, da.core.Array):
                charge_densities = da.from_array(charge_densities, chunks=(1, -1, -1, -1))

            charge_densities = charge_densities.to_delayed()

        elif hasattr(charge_densities, 'compute'):
            raise RuntimeError

        frozen_phonon_blocks = self.ewald_potential.frozen_phonons.partition_args(lazy=lazy)[0]

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
        frozen_phonons_partial = self.ewald_potential.frozen_phonons.from_partitioned_args()

        return partial(self._charge_density_potential, frozen_phonons_partial=frozen_phonons_partial, **kwargs)

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

        if hasattr(array, 'compute'):
            array = array.compute(scheduler='single-threaded')

        potential = self.ewald_potential.build(first_slice=first_slice, last_slice=last_slice, lazy=False)

        if self.ewald_potential.plane != 'xy':
            axes = plane_to_axes(self.ewald_potential.plane)
            array = np.moveaxis(array, axes[:2], (0, 1))

        array = solve_point_charges(self.ewald_potential.sliced_atoms.atoms,
                                    array=-array,  # -np.fft.fftn(array),#-array,
                                    width=self.ewald_parametrization.width)

        for i, ((a, b), slic) in enumerate(zip(self.slice_limits[first_slice:last_slice], potential)):
            slice_shape = self.gpts + (int((b - a) / min(self.sampling)),)

            slice_box = np.diag(self.box[:2] + (b - a,))

            slice_array = interpolate_between_cells(array,
                                                    slice_shape,
                                                    self.ewald_potential.sliced_atoms.atoms.cell,
                                                    slice_box,
                                                    (0, 0, a))

            integrated_slice_array = np.trapz(slice_array, axis=-1, dx=(b - a) / (slice_shape[-1] - 1))

            slic._array = slic._array + copy_to_device(integrated_slice_array[None], slic.array)
            yield slic
