"""Module to handle ab initio electrostatic potentials from the DFT code GPAW."""
from typing import Sequence, Tuple, Union

import numpy as np
from ase import units
from scipy.interpolate import interp1d, interpn

from abtem.base_classes import Grid
from abtem.device import get_device_function
from abtem.potentials import AbstractPotentialBuilder, PotentialArray, _disc_meshgrid, pad_atoms, \
    PotentialIntegrator
from abtem.structures import orthogonalize_cell
from abtem.utils import subdivide_into_batches

import warnings

try:
    from gpaw.atom.shapefunc import shape_functions
    from gpaw import GPAW
except ImportError:
    warnings.warn('This functionality of abTEM requires GPAW, see https://wiki.fysik.dtu.dk/gpaw/.')


def interpolate_rectangle(array: np.ndarray,
                          cell: np.ndarray,
                          extent: Sequence[float],
                          gpts: Sequence[int],
                          origin: Sequence[float] = None):
    """
    Interpolation to rectangle function

    A function to interpolate an array to a given rectangle, here used to convert electrostatic potentials
    from non-orthogonal cells to rectangular ones for use in abTEM multislice simulations.

    :param array: Electrostatic potential array to be interpolated.
    :param cell: ASE atoms simulation cell.
    :param extent: Extent of the rectangle [Å].
    :param gpts: Number of interpolation grid points.
    :param origin: Origin of the rectangle. Default is (0,0).
    """

    if origin is None:
        origin = (0., 0.)

    origin = np.array(origin)
    extent = np.array(extent)

    P = np.array(cell)
    P_inv = np.linalg.inv(P)
    origin_t = np.dot(origin, P_inv)
    origin_t = origin_t % 1.0

    lower = np.dot(origin_t, P)
    upper = lower + extent

    padded_array = np.zeros((array.shape[0] + 1, array.shape[1] + 1))
    padded_array[:-1, :-1] = array
    padded_array[-1, :] = padded_array[0, :]
    padded_array[:, -1] = padded_array[:, 0]

    x = np.linspace(0, 1, padded_array.shape[0], endpoint=True)
    y = np.linspace(0, 1, padded_array.shape[1], endpoint=True)

    x_ = np.linspace(lower[0], upper[0], gpts[0], endpoint=False)
    y_ = np.linspace(lower[1], upper[1], gpts[1], endpoint=False)
    x_, y_ = np.meshgrid(x_, y_, indexing='ij')

    p = np.array([x_.ravel(), y_.ravel()]).T
    p = np.dot(p, P_inv) % 1.0

    interpolated = interpn((x, y), padded_array, p, method='splinef2d')
    return interpolated.reshape((gpts[0], gpts[1]))


def get_paw_corrections(atom_index: int, calculator, rcgauss: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
    """
    PAW corrections function

    Function to calculate the projector-augmented wave corrections to the electrostatic potential, needed to
    calculate the all-electron potential from a converged calculation. This is implemented independently in
    abTEM to enable dealing with non-orthogonal cells, and to allow working with slices of large potentials.

    Parameters
    ----------
    atom_index: int
        Index of the atom for which the corrections are calculated.
    calculator: GPAW object
        Converged GPAW calculation.
    rcgauss: float
        Radius of the Gaussian smearing of the nuclear potentials [Å]. Default value is 0.005 Å.

    Returns
    -------
    two 1d arrays
        The evaluation points and values of the core contribution to the electronstatic potential.
    """

    dens = calculator.density
    dens.D_asp.redistribute(dens.atom_partition.as_serial())
    dens.Q_aL.redistribute(dens.atom_partition.as_serial())

    D_sp = dens.D_asp[atom_index]

    setup = dens.setups[atom_index]
    c = setup.xc_correction
    rgd = c.rgd
    ghat_g = shape_functions(rgd, **setup.data.shape_function, lmax=0)[0]
    Z_g = shape_functions(rgd, 'gauss', rcgauss, lmax=0)[0] * setup.Z
    D_q = np.dot(D_sp.sum(0), c.B_pqL[:, :, 0])
    dn_g = np.dot(D_q, (c.n_qg - c.nt_qg)) * np.sqrt(4 * np.pi)
    dn_g += 4 * np.pi * (c.nc_g - c.nct_g)
    dn_g -= Z_g
    dn_g -= dens.Q_aL[atom_index][0] * ghat_g * np.sqrt(4 * np.pi)
    dv_g = rgd.poisson(dn_g) / np.sqrt(4 * np.pi)
    dv_g[1:] /= rgd.r_g[1:]
    dv_g[0] = dv_g[1]
    dv_g[-1] = 0.0

    return rgd.r_g, dv_g


class GPAWPotential(AbstractPotentialBuilder):
    """
    GPAW DFT potential object

    The GPAW potential object is used to calculate electrostatic potential of a converged GPAW calculator object.

    Parameters
    ----------
    calculator: GPAW object
        A converged GPAW calculator.
    origin: two float, optional
        xy-origin of the electrostatic potential relative to the xy-origin of the Atoms object [Å].
    gpts: one or two int
        Number of grid points describing each slice of the potential.
    sampling: one or two float
        Lateral sampling of the potential [1 / Å].
    slice_thickness: float
        Thickness of the potential slices in Å for calculating the number of slices used by the multislice algorithm.
        Default is 0.5 Å.
    core_size: float
        The standard deviation of the Gaussian function representing the atomic core.
    """

    def __init__(self,
                 calculator,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 # origin: Union[float, Sequence[float]] = None,
                 slice_thickness=.5,
                 core_size=.005,
                 storage='cpu'):

        self._calculator = calculator
        self._core_size = core_size

        thickness = calculator.atoms.cell[2, 2]
        nz = calculator.hamiltonian.finegd.N_c[2]
        num_slices = int(np.ceil(nz / np.floor(slice_thickness / (thickness / nz))))

        self._voxel_height = thickness / nz
        self._slice_vertical_voxels = subdivide_into_batches(nz, num_slices)

        # TODO: implement support for non-periodic extent

        self._origin = (0., 0.)
        extent = np.diag(orthogonalize_cell(calculator.atoms.copy()).cell)[:2]

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling, lock_extent=True)

        super().__init__(storage=storage)

    @property
    def calculator(self):
        return self._calculator

    @property
    def core_size(self):
        return self._core_size

    @property
    def origin(self):
        return self._origin

    @property
    def num_slices(self):
        return len(self._slice_vertical_voxels)

    def get_slice_thickness(self, i):
        return self._slice_vertical_voxels[i] * self._voxel_height

    def generate_slices(self, first_slice=0, last_slice=None, max_batch=1):

        interpolate_radial_functions = get_device_function(np, 'interpolate_radial_functions')

        if last_slice is None:
            last_slice = len(self)

        valence = self._calculator.get_electrostatic_potential()
        cell = self._calculator.atoms.cell[:2, :2]

        atoms = self._calculator.atoms.copy()
        atoms.set_tags(range(len(atoms)))
        atoms = orthogonalize_cell(atoms)

        indices_by_number = {number: np.where(atoms.numbers == number)[0] for number in np.unique(atoms.numbers)}

        na = sum(self._slice_vertical_voxels[:first_slice])
        a = na * self._voxel_height
        for i in range(first_slice, last_slice):
            nb = na + self._slice_vertical_voxels[i]
            b = a + self._slice_vertical_voxels[i] * self._voxel_height

            projected_valence = valence[..., na:nb].sum(axis=-1) * self._voxel_height
            projected_valence = interpolate_rectangle(projected_valence, cell, self.extent, self.gpts, self._origin)

            array = np.zeros(self.gpts, dtype=np.float32)
            for number, indices in indices_by_number.items():
                slice_atoms = atoms[indices]

                if len(slice_atoms) == 0:
                    continue

                r = self._calculator.density.setups[indices[0]].xc_correction.rgd.r_g[1:] * units.Bohr
                cutoff = r[-1]

                margin = np.int(np.ceil(cutoff / np.min(self.sampling)))
                rows, cols = _disc_meshgrid(margin)
                disc_indices = np.hstack((rows[:, None], cols[:, None]))

                slice_atoms = slice_atoms[(slice_atoms.positions[:, 2] > a - cutoff) *
                                          (slice_atoms.positions[:, 2] < b + cutoff)]

                slice_atoms = pad_atoms(slice_atoms, cutoff)

                R = np.geomspace(1e-8, cutoff, int(np.ceil(cutoff / np.min(self.sampling))) * 10)

                vr = np.zeros((len(slice_atoms), len(R)), np.float32)
                dvdr = np.zeros((len(slice_atoms), len(R)), np.float32)
                for j, atom in enumerate(slice_atoms):
                    r, v = get_paw_corrections(atom.tag, self._calculator, self._core_size)

                    f = interp1d(r * units.Bohr, v, fill_value=(v[0], 0), bounds_error=False, kind='linear')

                    integrator = PotentialIntegrator(f, R)
                    am, bm = a - atom.z, b - atom.z

                    vr[j], dvdr[j, :-1] = integrator.integrate(am, bm)

                sampling = np.asarray(self.sampling, dtype=np.float32)

                interpolate_radial_functions(array,
                                             disc_indices,
                                             slice_atoms.positions,
                                             vr,
                                             R,
                                             dvdr,
                                             sampling)

            array = -(projected_valence + array / np.sqrt(4 * np.pi) * units.Ha)

            yield i, i + 1, PotentialArray(array, np.array([self.get_slice_thickness(i)]), extent=self.extent)

            a = b
            na = nb

    def __copy__(self):
        slice_thickness = self.calculator.atoms.cell[2, 2] / self.num_slices
        return self.__class__(self.calculator,
                              gpts=self.gpts,
                              sampling=self.sampling,
                              # origin=self.origin,
                              slice_thickness=slice_thickness,
                              core_size=self.core_size,
                              storage=self._storage)
