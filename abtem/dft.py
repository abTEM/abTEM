import numpy as np
from abtem.bases import cached_method, Cache
from abtem.interpolation import interpolation_kernel, interpolate_single
from abtem.parametrizations import project_tanh_sinh
from abtem.potentials import AbstractPotential
# from abtem.structures import fill_rectangle_with_atoms, orthogonalize_array
from abtem.utils import split_integer
from ase import units
from numba import jit


def orthogonalize_array(array, original_cell, origin, extent, new_gpts=None):
    if new_gpts is None:
        new_gpts = array.shape

    origin = np.array(origin)
    extent = np.array(extent)

    P = np.array(original_cell)
    P_inv = np.linalg.inv(P)
    origin_t = np.dot(origin, P_inv)
    origin_t = origin_t % 1.0

    lower = np.dot(origin_t, P)
    upper = lower + extent

    n, m = np.ceil(np.dot(upper, P_inv)).astype(np.int)

    tiled = np.tile(array, (n + 2, m + 2))
    x = np.linspace(-1, n + 1, tiled.shape[0], endpoint=False)
    y = np.linspace(-1, m + 1, tiled.shape[1], endpoint=False)

    x_ = np.linspace(lower[0], upper[0], new_gpts[0], endpoint=False)
    y_ = np.linspace(lower[1], upper[1], new_gpts[1], endpoint=False)
    x_, y_ = np.meshgrid(x_, y_, indexing='ij')

    p = np.array([x_.ravel(), y_.ravel()]).T
    p = np.dot(p, P_inv)

    interpolated = RegularGridInterpolator((x, y), tiled)(p)

    return interpolated.reshape((new_gpts[0], new_gpts[1]))


def gaussian(radial_grid, alpha):
    return 4 / np.sqrt(np.pi) * alpha ** 1.5 * np.exp(-alpha * radial_grid.r_g ** 2)


def get_paw_corrections(a, calculator, rcgauss=0.005):
    density = calculator.density
    density.D_asp.redistribute(density.atom_partition.as_serial())
    density.Q_aL.redistribute(density.atom_partition.as_serial())

    alpha = 1 / (rcgauss / units.Bohr) ** 2
    D_sp = density.D_asp[a]

    setup = density.setups[a]

    radial_grid = setup.xc_correction.rgd
    ghat_g = gaussian(radial_grid, 1 / setup.rcgauss ** 2)

    Z_g = gaussian(radial_grid, alpha) * setup.Z
    D_q = np.dot(D_sp.sum(0), setup.xc_correction.B_pqL[:, :, 0])

    dn_g = np.dot(D_q, (setup.xc_correction.n_qg - setup.xc_correction.nt_qg)) * np.sqrt(4 * np.pi)
    dn_g += 4 * np.pi * (setup.xc_correction.nc_g - setup.xc_correction.nct_g)
    dn_g -= Z_g
    dn_g -= density.Q_aL[a][0] * ghat_g * np.sqrt(4 * np.pi)

    dv_g = radial_grid.poisson(dn_g) / np.sqrt(4 * np.pi)
    dv_g[1:] /= radial_grid.r_g[1:]
    dv_g[0] = dv_g[1]
    dv_g[-1] = 0.0

    return radial_grid.r_g, dv_g


class GPAWPotential(AbstractPotential):
    """
    GPAW DFT potential object

    The GPAW potential object is used to calculate electrostatic potential of a converged GPAW calculator object.

    Parameters
    ----------
    calculator : GPAW calculator object
        A converged GPAW calculator.
    origin : two floats, float, optional
        xy-origin of the electrostatic potential relative to the xy-origin of the Atoms object. Units of Angstrom.
    extent : two floats, float, optional
        Lateral extent of potential, if the unit cell of the atoms is too small it will be repeated. Units of Angstrom.
    gpts : two ints, int, optional
        Number of grid points describing each slice of the potential.
    sampling : two floats, float, optional
        Lateral sampling of the potential. Units of 1 / Angstrom.
    slice_thickness : float, optional
        Thickness of the potential slices in Angstrom for calculating the number of slices used by the multislice
        algorithm. Default is 0.5 Angstrom.
    num_slices : int, optional
        Number of slices used by the multislice algorithm. If `num_slices` is set, then `slice_thickness` is disabled.
    quadrature_order : int, optional
        Order of the Tanh-Sinh quadrature for numerical integration the potential along the optical axis. Default is 40.
    interpolation_sampling : float, optional
        The average sampling used when calculating the radial dependence of the atomic potentials.
    core_size : float
        The standard deviation of the Gaussian function representing the atomic core.
    """

    def __init__(self, calculator, extent=None, gpts=None, sampling=None, slice_thickness=.5, core_size=.005):
        self._calculator = calculator
        self._core_size = core_size

        thickness = calculator.atoms.cell[2, 2]
        Nz = calculator.hamiltonian.finegd.N_c[2]
        self._num_slices = int(np.ceil(Nz / np.floor(slice_thickness / (thickness / Nz))))

        self._nz = split_integer(Nz, self._num_slices)
        self._dz = thickness / Nz

        super().__init__()

    @property
    def num_slices(self):
        return self._num_slices

    @property
    def thickness(self):
        return self._calculator.atoms.cell[2, 2]

    # @cached_method()
    # def _get_paw_correction(self, i):
    #     r, v = get_paw_corrections(i, self._calculator, rcgauss=self._core_size)
    #     r = r * units.Bohr
    #
    #     dvdr = np.diff(v) / np.diff(r)
    #
    #     @jit(nopython=True, nogil=True)
    #     def interpolator(x_interp):
    #         return interpolate_single(r, v, dvdr, x_interp)
    #
    #     return interpolator
    #
    # def get_cutoff(self, atom):
    #     return self._calculator.density.setups[atom].xc_correction.rgd.r_g[-1] * units.Bohr
    #
    # @cached_method()
    # def max_cutoff(self):
    #     density = self._calculator.density
    #     max_cutoff = 0.
    #     for atom in density.D_asp.keys():
    #         max_cutoff = max(self.get_cutoff(atom), max_cutoff)
    #     return max_cutoff
    #
    # @cached_method()
    # def _get_electrostatic_potential(self):
    #     return self._calculator.get_electrostatic_potential()
    #
    # @cached_method(('sampling',))
    # def _get_quadrature(self):
    #     m = self._quadrature_order
    #     h = QUADRATURE_PARAMETER_RATIO / self._quadrature_order
    #     return tanh_sinh_quadrature(m, h)
    #
    # @cached_method()
    # def get_atoms(self):
    #     return fill_rectangle_with_atoms(self._atoms, self._origin, self.extent, margin=self.max_cutoff(),
    #                                      return_atom_labels=True)

    # def slice_thickness(self, i):
    #     return self._nz[i] * self._dz
    #
    # def _get_slice_array(self, i):
    #     start = np.sum(self._nz[:i], dtype=np.int)
    #     stop = np.sum(self._nz[:i + 1], dtype=np.int)
    #     slice_entrance = start * self._dz
    #     slice_exit = stop * self._dz
    #
    #     array = self._get_electrostatic_potential()
    #     array = array[..., start:stop].sum(axis=-1) * self._dz
    #
    #     array = orthogonalize_array(array, self._calculator.atoms.cell[:2, :2], self._origin, self.extent, self.gpts)
    #     array = self._get_projected_paw_corrections(slice_entrance, slice_exit) - array
    #     return array
    #
    # def _get_projected_paw_corrections(self, slice_entrance, slice_exit, num_integration_samples=400):
    #
    #     max_cutoff = self.max_cutoff()
    #
    #     atoms, equivalent = self.get_atoms()
    #     positions = atoms.get_positions()
    #
    #     v = np.zeros(self.gpts)
    #
    #     idx_in_slice = np.where(((slice_entrance - max_cutoff) < positions[:, 2]) &
    #                             ((slice_exit + max_cutoff) > positions[:, 2]))[0]
    #
    #     for idx in idx_in_slice:
    #         paw_correction = self._get_paw_correction(equivalent[idx])
    #         position = positions[idx]
    #         cutoff = self.get_cutoff(equivalent[idx])
    #
    #         z0 = slice_entrance - position[2]
    #         z1 = slice_exit - position[2]
    #
    #         r = np.geomspace(min(self.sampling) / 2, cutoff, int(np.ceil(cutoff / self._interpolation_sampling)))
    #
    #         xk, wk = self._get_quadrature()
    #
    #         vr = project_tanh_sinh(r, np.array([z0]), np.array([z1]), xk, wk, paw_correction)
    #
    #         block_margin = np.int(cutoff / min(self.sampling))
    #         block_size = 2 * block_margin + 1
    #
    #         corner_positions = (np.round(position[:2] / self.sampling)).astype(np.int) - block_margin
    #         block_positions = position[:2] - self.sampling * corner_positions
    #
    #         x = np.linspace(0., block_size * self.sampling[0] - self.sampling[0], block_size)
    #         y = np.linspace(0., block_size * self.sampling[1] - self.sampling[1], block_size)
    #
    #         interpolation_kernel(v, r, vr[0], corner_positions, block_positions, x, y)
    #
    #     return - v / np.sqrt(4 * np.pi) * units.Ha
