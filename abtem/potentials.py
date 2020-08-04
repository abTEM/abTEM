from abc import ABCMeta, abstractmethod
from typing import Union, Sequence

import h5py
import numpy as np
from ase import Atoms
from ase import units
from scipy.optimize import brentq

from abtem.bases import Grid, HasGridMixin, Cache, cached_method, HasAcceleratorMixin, Accelerator, Event, \
    watched_property
from abtem.device import get_device_function, get_array_module, HasDeviceMixin, get_array_module_from_device, \
    copy_to_device
from abtem.measure import calibrations_from_grid
from abtem.parametrizations import kirkland, dvdr_kirkland, load_kirkland_parameters
from abtem.parametrizations import lobato, dvdr_lobato, load_lobato_parameters
from abtem.plot import show_image
from abtem.structures import is_cell_orthogonal
from abtem.tanh_sinh import integrate, tanh_sinh_nodes_and_weights
from abtem.temperature import AbstractFrozenPhonons, DummyFrozenPhonons
from abtem.utils import energy2sigma, ProgressBar

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)

kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)


class AbstractPotential(HasGridMixin, metaclass=ABCMeta):

    def __len__(self):
        return self.num_slices

    @property
    @abstractmethod
    def num_slices(self):
        pass

    def check_slice_idx(self, i):
        if i >= self.num_slices:
            raise RuntimeError('Slice index {} too large for potential with {} slices'.format(i, self.num_slices))

    def get_slice(self, i):
        self.check_slice_idx(i)
        return next(self.generate_slices(i, i + 1))

    @abstractmethod
    def generate_slices(self, start=0, end=None):
        pass

    @abstractmethod
    def get_slice_thickness(self, i):
        pass

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= self.num_slices:
                raise StopIteration
            return self.get_slice(item)
        elif isinstance(item, slice):
            if item.start is None:
                start = 0
            else:
                start = item.start

            if item.stop is None:
                stop = len(self)
            else:
                stop = item.stop

            if item.step is None:
                step = 1
            else:
                step = item.step

            projected = self[start]
            for i in range(start + 1, stop, step):
                projected._array += self[i]._array
                projected._thickness += self.get_slice_thickness(i)
            return projected
        else:
            raise TypeError('Potential indices must be integers, not {}'.format(type(item)))

    def show(self, **kwargs):
        self[:].show(**kwargs)


class AbstractPotentialBuilder(HasDeviceMixin, AbstractPotential):

    def __init__(self, storage='cpu'):
        self._storage = storage

    def build(self, pbar: Union[bool, ProgressBar] = False) -> 'ArrayPotential':
        self.grid.check_is_defined()

        storage_xp = get_array_module_from_device(self._storage)
        array = storage_xp.zeros((self.num_slices,) + (self.gpts[0], self.gpts[1]), dtype=np.float32)
        slice_thicknesses = np.zeros(self.num_slices)

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(self), desc='Potential', disable=not pbar)

        pbar.reset()
        for i, potential_slice in enumerate(self.generate_slices()):
            array[i] = copy_to_device(potential_slice.array, self._storage)
            slice_thicknesses[i] = potential_slice.thickness
            pbar.update(1)

        pbar.refresh()

        return ArrayPotential(array, slice_thicknesses, self.extent)


class AbstractTDSPotentialBuilder(AbstractPotentialBuilder):

    def __init__(self, storage='cpu'):
        super().__init__(storage)

    @property
    @abstractmethod
    def frozen_phonons(self):
        pass

    @abstractmethod
    def generate_frozen_phonon_potentials(self, pbar):
        pass


class PotentialIntegrator:

    def __init__(self, function, r, cache_size=4096, cache_key_decimals=2, tolerance=1e-12):
        self._function = function
        self._r = r
        self.cache = Cache(cache_size)
        self._cache_key_decimals = cache_key_decimals
        self._tolerance = tolerance

    @property
    def r(self):
        return self._r

    @property
    def cutoff(self):
        return self._r[-1]

    def integrate(self, a, b):
        a = round(a, self._cache_key_decimals)
        b = round(b, self._cache_key_decimals)

        split = a * b < 0
        a = max(min(a, b), -self.cutoff)
        b = min(max(a, b), self.cutoff)
        if split:  # split integral
            values1, derivatives1 = self.cached_integrate(0, abs(a))
            values2, derivatives2 = self.cached_integrate(0, abs(b))
            result = (values1 + values2, derivatives1 + derivatives2)
        else:
            result = self.cached_integrate(a, b)
        return result

    @cached_method('cache')
    def cached_integrate(self, a, b):
        zm = (b - a) / 2.
        zp = (a + b) / 2.
        f = lambda z: self._function(np.sqrt(self.r[0] ** 2 + (z * zm + zp) ** 2))
        value, error_estimate, step_size, order = integrate(f, -1, 1, self._tolerance)
        xk, wk = tanh_sinh_nodes_and_weights(step_size, order)
        f = lambda z: self._function(np.sqrt(self.r[:, None] ** 2 + (z * zm + zp) ** 2))
        values = np.sum(f(xk[None]) * wk[None], axis=1) * zm
        derivatives = np.diff(values) / np.diff(self.r)
        return values, derivatives


class ProjectedPotential(HasGridMixin):

    def __init__(self, array, thickness, extent=None, sampling=None):
        self._array = array
        self._thickness = thickness
        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)

    @property
    def thickness(self) -> float:
        return self._thickness

    @property
    def array(self):
        return self._array

    def show(self, **kwargs):
        calibrations = calibrations_from_grid(self.grid.gpts, self.grid.sampling, names=['x', 'y'])
        return show_image(self.array, calibrations, **kwargs)


def pad_atoms(atoms, margin):
    if not is_cell_orthogonal(atoms):
        raise RuntimeError()

    left = atoms[atoms.positions[:, 0] < margin]
    left.positions[:, 0] += atoms.cell[0, 0]
    right = atoms[atoms.positions[:, 0] > atoms.cell[0, 0] - margin]
    right.positions[:, 0] -= atoms.cell[0, 0]

    atoms += left + right

    top = atoms[atoms.positions[:, 1] < margin]
    top.positions[:, 1] += atoms.cell[1, 1]
    bottom = atoms[atoms.positions[:, 1] > atoms.cell[1, 1] - margin]
    bottom.positions[:, 1] -= atoms.cell[1, 1]
    atoms += top + bottom

    return atoms


class Potential(AbstractTDSPotentialBuilder):
    """
    Potential object.

    The potential object is used to calculate the electrostatic potential of a set of atoms represented by an ASE atoms
    object. The potential is calculated with the Independent Atom Model (IAM) using a user-defined parametrization
    of the atomic potentials.

    :param atoms: Atoms object defining the atomic configuration used in the IAM of the electrostatic potential.
    :param gpts: Number of grid points describing each slice of the potential.
    :param sampling: Lateral sampling of the potential [1 / Å].
    :param slice_thickness: Thickness of the potential slices in Å for calculating the number of slices used by
        the multislice algorithm.
    :param parametrization: The potential parametrization describes the radial dependence of the potential for each element.
        Two of the most accurate parametrizations are available by Lobato et. al. and Kirkland.
        See the citation guide for references.
    :param cutoff_tolerance: The error tolerance used for deciding the radial cutoff distance of the potential [eV / e].
    :param device: The device used for calculating the potential.
    :param storage: The device on which to store the created potential.
    """

    def __init__(self,
                 atoms: Union[Atoms, AbstractFrozenPhonons] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 slice_thickness: float = .5,
                 parametrization: str = 'lobato',
                 cutoff_tolerance: float = 1e-3,
                 device='cpu',
                 storage=None):

        self._cutoff_tolerance = cutoff_tolerance
        self._parametrization = parametrization
        self._slice_thickness = slice_thickness

        self._storage = storage

        if parametrization == 'lobato':
            self._parameters = load_lobato_parameters()
            self._function = lobato  # lambda r: lobato(r, parameters[atomic_number])
            self._derivative = dvdr_lobato  # lambda r: dvdr_lobato(r, parameters[atomic_number])

        elif parametrization == 'kirkland':
            self._parameters = load_kirkland_parameters()
            self._function = kirkland  # lambda r: kirkland(r, parameters[atomic_number])
            self._derivative = dvdr_kirkland  # lambda r: dvdr_kirkland(r, parameters[atomic_number])
        else:
            raise RuntimeError('Parametrization {} not recognized'.format(parametrization))

        if isinstance(atoms, AbstractFrozenPhonons):
            self._frozen_phonons = atoms
        else:
            self._frozen_phonons = DummyFrozenPhonons(atoms)

        atoms = next(iter(self._frozen_phonons))

        if np.abs(atoms.cell[2, 2]) < 1e-12:
            raise RuntimeError('Atoms cell has no thickness')

        if not is_cell_orthogonal(atoms):
            raise RuntimeError('Atoms are not orthogonal')

        self._atoms = atoms
        self._grid = Grid(extent=np.diag(atoms.cell)[:2], gpts=gpts, sampling=sampling, lock_extent=True)

        self._cutoffs = {}
        self._integrators = {}

        def grid_changed_callback(*args, **kwargs):
            self._integrators = {}

        self.grid.changed.register(grid_changed_callback)
        self.changed = Event()

        self._device = device

        if storage is None:
            self._storage = device

        super().__init__(storage=storage)

    @property
    def parameters(self):
        return self._parameters

    @property
    def function(self):
        return self._function

    @property
    def atoms(self):
        return self._atoms

    @property
    def frozen_phonons(self):
        return self._frozen_phonons

    @property
    def cutoff_tolerance(self):
        return self._cutoff_tolerance

    @property
    def num_slices(self):
        return int(np.ceil(self._atoms.cell[2, 2] / self._slice_thickness))

    @property
    def slice_thickness(self):
        return self._slice_thickness

    @slice_thickness.setter
    @watched_property('changed')
    def slice_thickness(self, value):
        self._slice_thickness = value

    def get_slice_thickness(self, i):
        return self._atoms.cell[2, 2] / self.num_slices

    def get_cutoff(self, number):
        try:
            return self._cutoffs[number]
        except KeyError:
            f = lambda r: self.function(r, self.parameters[number]) - self.cutoff_tolerance
            self._cutoffs[number] = brentq(f, 1e-7, 1000)
            return self._cutoffs[number]

    def get_soft_function(self, number):
        cutoff = self.get_cutoff(number)
        rolloff = .85 * cutoff

        def soft_function(r):
            result = np.zeros_like(r)
            valid = r < cutoff
            transition = valid * (r > rolloff)
            result[valid] = self._function(r[valid], self.parameters[number])
            result[transition] *= (np.cos(np.pi * (r[transition] - rolloff) / (cutoff - rolloff)) + 1.) / 2
            return result

        return soft_function

    def get_integrator(self, number):
        try:
            return self._integrators[number]
        except KeyError:
            cutoff = self.get_cutoff(number)
            soft_function = self.get_soft_function(number)
            r = np.geomspace(np.min(self.sampling), cutoff, int(np.ceil(cutoff / np.min(self.sampling) * 10)))

            margin = np.int(np.ceil(cutoff / np.min(self.sampling)))
            rows, cols = disc_meshgrid(margin)
            disc_indices = np.hstack((rows[:, None], cols[:, None]))
            self._integrators[number] = (PotentialIntegrator(soft_function, r), disc_indices)
            return self._integrators[number]

    def generate_slices(self, start=0, end=None):
        if end is None:
            end = len(self)

        self.grid.check_is_defined()

        xp = get_array_module(self._device)
        interpolate_radial_functions = get_device_function(xp, 'interpolate_radial_functions')

        atoms = self.atoms.copy()
        indices_by_number = {number: np.where(atoms.numbers == number)[0] for number in np.unique(atoms.numbers)}

        array = xp.zeros(self.gpts, dtype=xp.float32)
        a = np.sum([self.get_slice_thickness(i) for i in range(0, start)])
        for i in range(start, end):
            array[:] = 0.
            b = a + self.get_slice_thickness(i)

            for number, indices in indices_by_number.items():
                slice_atoms = atoms[indices]

                integrator, disc_indices = self.get_integrator(number)
                disc_indices = xp.asarray(disc_indices)

                slice_atoms = slice_atoms[(slice_atoms.positions[:, 2] > a - integrator.cutoff) *
                                          (slice_atoms.positions[:, 2] < b + integrator.cutoff)]

                slice_atoms = pad_atoms(slice_atoms, integrator.cutoff)

                if len(slice_atoms) == 0:
                    continue

                vr = np.zeros((len(slice_atoms), len(integrator.r)), np.float32)
                dvdr = np.zeros((len(slice_atoms), len(integrator.r)), np.float32)
                for j, atom in enumerate(slice_atoms):
                    am, bm = a - atom.z, b - atom.z
                    vr[j], dvdr[j, :-1] = integrator.integrate(am, bm)
                vr = xp.asarray(vr, dtype=xp.float32)
                dvdr = xp.asarray(dvdr, dtype=xp.float32)
                r = xp.asarray(integrator.r, dtype=xp.float32)

                slice_positions = xp.asarray(slice_atoms.positions[:, :2], dtype=xp.float32)
                sampling = xp.asarray(self.sampling, dtype=xp.float32)

                interpolate_radial_functions(array,
                                             disc_indices,
                                             slice_positions,
                                             vr,
                                             r,
                                             dvdr,
                                             sampling)
            a = b

            yield ProjectedPotential(array / kappa, self.get_slice_thickness(i), extent=self.extent)

    def generate_frozen_phonon_potentials(self, pbar: Union[ProgressBar, bool] = True):
        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(self), desc='Potential', disable=not pbar)

        for atoms in self.frozen_phonons:
            self.atoms.positions[:] = atoms.positions
            self.atoms.wrap()
            pbar.reset()
            yield self.build(pbar=pbar)

        pbar.refresh()
        pbar.close()


def disc_meshgrid(r):
    cols = np.zeros((2 * r + 1, 2 * r + 1)).astype(np.int32)
    cols[:] = np.linspace(0, 2 * r, 2 * r + 1) - r
    rows = cols.T
    inside = (rows ** 2 + cols ** 2) <= r ** 2
    return rows[inside], cols[inside]


class ArrayPotential(AbstractPotential, HasGridMixin):
    """
    Array potential object.

    The array potential object represents slices of the electrostatic potential.

    :param array: The array representing the potential slices. The first dimension is the slice index and the last two are the
        spatial dimensions.
    :param slice_thicknesses: The thicknesses of potential slices in Å. If a float, the thickness is the same for all slices.
        If a sequence, the length must equal the length of the potential array.
    :param extent: Lateral extent of the potential [1 / Å].
    :param sampling: Lateral sampling of the potential [1 / Å].
    """

    def __init__(self, array: np.ndarray,
                 slice_thicknesses: Union[float, Sequence],
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None):

        if len(array.shape) != 3:
            raise RuntimeError()

        slice_thicknesses = np.array(slice_thicknesses)

        if slice_thicknesses.shape == ():
            slice_thicknesses = np.tile(slice_thicknesses, array.shape[0])
        elif slice_thicknesses.shape != (array.shape[0],):
            raise ValueError()

        self._array = array
        self._slice_thicknesses = slice_thicknesses
        self._grid = Grid(extent=extent, gpts=self.array.shape[1:], sampling=sampling, lock_gpts=True)

    def as_transmission_functions(self, energy):
        xp = get_array_module(self.array)
        complex_exponential = get_device_function(xp, 'complex_exponential')

        array = complex_exponential(energy2sigma(energy) * self._array)
        return TransmissionFunctions(array, slice_thicknesses=self._slice_thicknesses, extent=self.extent,
                                     energy=energy)

    @property
    def array(self):
        return self._array

    @property
    def num_slices(self):
        return self._array.shape[0]

    def get_slice_thickness(self, i):
        return self._slice_thicknesses[i]

    def generate_slices(self, start=0, end=None):
        if end is None:
            end = len(self)

        for i in range(start, end):
            yield ProjectedPotential(self.array[i], thickness=self.get_slice_thickness(i), extent=self.extent)

    def tile(self, multiples):
        assert len(multiples) == 2
        new_array = np.tile(self.array, (1,) + multiples)
        new_extent = (self.extent[0] * multiples[0], self.extent[1] * multiples[1])
        return self.__class__(array=new_array, slice_thicknesses=self._slice_thicknesses, extent=new_extent)

    def write(self, path):
        with h5py.File(path, 'w') as f:
            f.create_dataset('array', data=self.array)
            f.create_dataset('slice_thicknesses', data=self._slice_thicknesses)
            f.create_dataset('extent', data=self.extent)

    @classmethod
    def read(cls, path):
        with h5py.File(path, 'r') as f:
            datasets = {}
            for key in f.keys():
                datasets[key] = f.get(key)[()]

        return cls(array=datasets['array'], slice_thicknesses=datasets['slice_thicknesses'], extent=datasets['extent'])

    def __copy___(self):
        return self.__class__(array=self.array.copy(),
                              slice_thicknesses=self._slice_thicknesses.copy(),
                              extent=self.extent)


class TransmissionFunctions(ArrayPotential, HasAcceleratorMixin):

    def __init__(self, array: np.ndarray, slice_thicknesses: Union[float, Sequence], extent: np.ndarray = None,
                 sampling: np.ndarray = None, energy: float = None):
        self._accelerator = Accelerator(energy=energy)
        super().__init__(array, slice_thicknesses, extent, sampling)
