import numpy as np
from ase import units
from scipy.interpolate import RegularGridInterpolator, interp1d

from abtem.bases import Grid
from abtem.device import get_device_function
from abtem.potentials import AbstractPotentialBuilder, ProjectedPotential, disc_meshgrid, pad_atoms, \
    PotentialIntegrator
from abtem.utils import split_integer
from abtem.structures import orthogonalize_cell
from gpaw.atom.shapefunc import shape_functions

def interpolate_rectangle(array, cell, extent, gpts, origin=None):
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

    interpolated = RegularGridInterpolator((x, y), padded_array)(p)
    return interpolated.reshape((gpts[0], gpts[1]))


def gaussian(radial_grid, alpha):
    return 4 / np.sqrt(np.pi) * alpha ** 1.5 * np.exp(-alpha * radial_grid.r_g ** 2)


def get_paw_corrections(a, calculator, rcgauss=0.005):
    dens = calculator.density
    dens.D_asp.redistribute(dens.atom_partition.as_serial())
    dens.Q_aL.redistribute(dens.atom_partition.as_serial())

    D_sp = dens.D_asp[a]
    #for a, D_sp in dens.D_asp.items():
    setup = dens.setups[a]
    c = setup.xc_correction
    rgd = c.rgd
    ghat_g = shape_functions(rgd,
                             **setup.data.shape_function, lmax=0)[0]
    Z_g = shape_functions(rgd, 'gauss', rcgauss, lmax=0)[0] * setup.Z
    D_q = np.dot(D_sp.sum(0), c.B_pqL[:, :, 0])
    dn_g = np.dot(D_q, (c.n_qg - c.nt_qg)) * np.sqrt(4 * np.pi)
    dn_g += 4 * np.pi * (c.nc_g - c.nct_g)
    dn_g -= Z_g
    dn_g -= dens.Q_aL[a][0] * ghat_g * np.sqrt(4 * np.pi)
    dv_g = rgd.poisson(dn_g) / np.sqrt(4 * np.pi)
    dv_g[1:] /= rgd.r_g[1:]
    dv_g[0] = dv_g[1]
    dv_g[-1] = 0.0

    #dv
    #dv_a1.append([rgd.spline(dv_g, points=POINTS)])

    return rgd.r_g, dv_g


class GPAWPotential(AbstractPotentialBuilder):
    """
    GPAW DFT potential object

    The GPAW potential object is used to calculate electrostatic potential of a converged GPAW calculator object.

    Parameters
    ----------
    calculator : GPAW calculator object
        A converged GPAW calculator.
    origin : two floats, float, optional
        xy-origin of the electrostatic potential relative to the xy-origin of the Atoms object [Å].
    extent : two floats, float, optional
        Lateral extent of potential, if the unit cell of the atoms is too small it will be repeated [Å].
    gpts : two ints, int, optional
        Number of grid points describing each slice of the potential.
    sampling : two floats, float, optional
        Lateral sampling of the potential [1 / Å].
    slice_thickness : float, optional
        Thickness of the potential slices in Å for calculating the number of slices used by the multislice
        algorithm. Default is 0.5 Å.
    num_slices : int, optional
        Number of slices used by the multislice algorithm. If `num_slices` is set, then `slice_thickness` is disabled.
    quadrature_order : int, optional
        Order of the Tanh-Sinh quadrature for numerical integration the potential along the optical axis. Default is 40.
    interpolation_sampling : float, optional
        The average sampling used when calculating the radial dependence of the atomic potentials.
    core_size : float
        The standard deviation of the Gaussian function representing the atomic core.
    """

    def __init__(self, calculator, gpts=None, sampling=None, slice_thickness=.5, core_size=.005, storage='cpu'):
        self._calculator = calculator
        self._core_size = core_size

        thickness = calculator.atoms.cell[2, 2]
        nz = calculator.hamiltonian.finegd.N_c[2]
        num_slices = int(np.ceil(nz / np.floor(slice_thickness / (thickness / nz))))

        self._voxel_height = thickness / nz
        self._slice_vertical_voxels = split_integer(nz, num_slices)

        # TODO: implement support for non-periodic extent

        self._origin = (0., 0.)
        extent = np.diag(orthogonalize_cell(calculator.atoms.copy(), strain_error=True).cell)[:2]

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling, lock_extent=True)

        super().__init__(storage)

    @property
    def num_slices(self):
        return len(self._slice_vertical_voxels)

    def get_slice_thickness(self, i):
        return self._slice_vertical_voxels[i] * self._voxel_height

    def generate_slices(self, start=0, stop=None):
        interpolate_radial_functions = get_device_function(np, 'interpolate_radial_functions')

        if stop is None:
            stop = len(self)

        valence = self._calculator.get_electrostatic_potential()
        cell = self._calculator.atoms.cell[:2, :2]

        atoms = self._calculator.atoms.copy()
        atoms.set_tags(range(len(atoms)))
        atoms = orthogonalize_cell(atoms)

        indices_by_number = {number: np.where(atoms.numbers == number)[0] for number in np.unique(atoms.numbers)}

        na = sum(self._slice_vertical_voxels[:start])
        a = na * self._voxel_height
        for i in range(start, stop):
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
                rows, cols = disc_meshgrid(margin)
                disc_indices = np.hstack((rows[:, None], cols[:, None]))

                slice_atoms = slice_atoms[(slice_atoms.positions[:, 2] > a - cutoff) *
                                          (slice_atoms.positions[:, 2] < b + cutoff)]

                slice_atoms = pad_atoms(slice_atoms, cutoff)

                R = np.geomspace(np.min(self.sampling), cutoff, int(np.ceil(cutoff / np.min(self.sampling) * 10)))

                vr = np.zeros((len(slice_atoms), len(R)), np.float32)
                dvdr = np.zeros((len(slice_atoms), len(R)), np.float32)
                for j, atom in enumerate(slice_atoms):
                    r, v = get_paw_corrections(atom.tag, self._calculator, self._core_size)

                    f = interp1d(r * units.Bohr, v, fill_value=(v[0], 0), bounds_error=False)
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
            yield ProjectedPotential(array, self.get_slice_thickness(i), extent=self.extent)
            a = b
            na = nb
