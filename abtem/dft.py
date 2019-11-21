import numpy as np
from ase import units
from scipy.interpolate import interp1d

from abtem.bases import cached_method, Cache, cached_method_with_args
from abtem.transform import fill_rectangle_with_atoms, orthogonalize_array
from abtem.potentials import PotentialBase
from abtem.interpolation import interpolation_kernel
from abtem.utils import split_integer
from scipy.interpolate import interp1d


def gaussian(radial_grid, alpha):
    return 4 / np.sqrt(np.pi) * alpha ** 1.5 * np.exp(-alpha * radial_grid.r_g ** 2)


def get_paw_corrections(calculator, rcgauss=0.005, spline_pts=500):
    density = calculator.density
    density.D_asp.redistribute(density.atom_partition.as_serial())
    density.Q_aL.redistribute(density.atom_partition.as_serial())

    alpha = 1 / (rcgauss / units.Bohr) ** 2
    corrections = {}
    for a, D_sp in density.D_asp.items():
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

        corrections[a] = radial_grid.spline(dv_g, points=spline_pts)
    return corrections


def riemann_quadrature(num_samples):
    xk = np.linspace(-1., 1., num_samples + 1).astype(np.float32)
    return xk[:-1], xk[1:] - xk[:-1]


def project_spherical_function(f, r, a, b, num_samples=200):
    c = np.reshape(((b - a) / 2.), (-1, 1))
    d = np.reshape(((b + a) / 2.), (-1, 1))
    xk, wk = riemann_quadrature(num_samples)
    xkab = xk * c + d
    wkab = wk * c
    rxy = np.sqrt((r ** 2.).reshape((-1, 1)) + (xkab ** 2.).reshape((1, -1)))
    return np.sum(f(rxy) * wkab.reshape(1, -1), axis=-1)


class GPAWPotential(PotentialBase, Cache):

    def __init__(self, calc, origin=None, extent=None, gpts=None, sampling=None, num_slices=None, slice_thickness=.5,
                 quadrature_order=40, interpolation_sampling=.001, sigma=.005, assert_equal_thickness=False):
        self._calc = calc
        self._sigma = sigma

        thickness = calc.atoms.cell[2, 2]
        Nz = calc.hamiltonian.finegd.N_c[2]

        if num_slices is None:
            if slice_thickness is None:
                raise RuntimeError()
            num_slices = int(np.ceil(Nz / np.floor(slice_thickness / (thickness / Nz))))

        if (not (calc.hamiltonian.finegd.N_c[2] % num_slices == 0)) & assert_equal_thickness:
            raise RuntimeError('{} {}'.format(calc.hamiltonian.finegd.N_c[2], self.num_slices))

        self._Nz = Nz
        self._nz = split_integer(Nz, num_slices)
        self._dz = thickness / Nz

        self._interpolation_sampling = interpolation_sampling
        self._quadrature_order = quadrature_order

        super().__init__(atoms=calc.atoms.copy(), origin=origin, extent=extent, gpts=gpts,
                         sampling=sampling, num_slices=num_slices)

    @cached_method()
    def _get_paw_corrections(self):
        paw_corrections = get_paw_corrections(self._calc, rcgauss=self._sigma)
        return paw_corrections

    @cached_method()
    def max_cutoff(self):
        return max([spline.get_cutoff() for spline in self._get_paw_corrections().values()]) * units.Bohr

    @cached_method()
    def _get_electrostatic_potential(self):
        return self._calc.get_electrostatic_potential()

    @cached_method()
    def get_atoms(self):
        return fill_rectangle_with_atoms(self._atoms, self._origin, self.extent, margin=self.max_cutoff(),
                                         return_atom_labels=True)

    def slice_thickness(self, i):
        return self._nz[i] * self._dz

    def get_slice(self, i):
        start = np.sum(self._nz[:i], dtype=np.int)
        stop = np.sum(self._nz[:i + 1], dtype=np.int)
        slice_entrance = start * self._dz
        slice_exit = stop * self._dz

        array = self._get_electrostatic_potential()
        array = array[..., start:stop].sum(axis=-1) * self._dz

        array = orthogonalize_array(array, self._calc.atoms.cell[:2, :2], self._origin, self.extent, self.gpts)
        array = self._get_projected_paw_corrections(slice_entrance, slice_exit) - array
        return array

    def _get_projected_paw_corrections(self, slice_entrance, slice_exit, num_integration_samples=400):
        paw_corrections = self._get_paw_corrections()
        max_cutoff = self.max_cutoff()

        atoms, equivalent = self.get_atoms()
        positions = atoms.get_positions()

        v = np.zeros(self.gpts)

        idx_in_slice = np.where(((slice_entrance - max_cutoff) < positions[:, 2]) &
                                ((slice_exit + max_cutoff) > positions[:, 2]))[0]

        for idx in idx_in_slice:
            func = paw_corrections[equivalent[idx]]
            position = positions[idx]
            cutoff = func.get_cutoff()
            z0 = slice_entrance - position[2]
            z1 = slice_exit - position[2]

            r = np.geomspace(min(self.sampling) / units.Bohr, cutoff, int(np.ceil(cutoff / self._interpolation_sampling)))

            vr = project_spherical_function(func.map, r, z0 / units.Bohr, z1 / units.Bohr,
                                            num_samples=num_integration_samples) * units.Bohr

            r = r * units.Bohr
            block_margin = np.int(cutoff / min(self.sampling) * units.Bohr)
            block_size = 2 * block_margin + 1

            corner_positions = (np.round(position[:2] / self.sampling)).astype(np.int) - block_margin
            block_positions = position[:2] - self.sampling * corner_positions

            x = np.linspace(0., block_size * self.sampling[0] - self.sampling[0], block_size)
            y = np.linspace(0., block_size * self.sampling[1] - self.sampling[1], block_size)

            interpolation_kernel(v, r, vr, corner_positions, block_positions, x, y)

        return - v / np.sqrt(4 * np.pi) * units.Ha


def calculate_3d_potential(calc, h=.05, rcgauss=0.02, spline_pts=200, add_corrections=True):
    from gpaw.lfc import LFC
    from gpaw.utilities import h2gpts
    from gpaw.fftw import get_efficient_fft_size
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.mpi import serial_comm
    from gpaw.wavefunctions.pw import PWDescriptor

    v_r = calc.get_electrostatic_potential() / units.Ha

    old_gd = GridDescriptor(calc.hamiltonian.finegd.N_c, calc.hamiltonian.finegd.cell_cv, comm=serial_comm)

    N_c = h2gpts(h / units.Bohr, calc.wfs.gd.cell_cv)
    N_c = np.array([get_efficient_fft_size(N, 2) for N in N_c])
    new_gd = GridDescriptor(N_c, calc.wfs.gd.cell_cv, comm=serial_comm)

    pd1 = PWDescriptor(0.0, old_gd, float)
    pd2 = PWDescriptor(0.0, new_gd, float)

    v_r = pd1.interpolate(v_r, pd2)[0]

    if add_corrections:
        dens = calc.density
        dens.D_asp.redistribute(dens.atom_partition.as_serial())
        dens.Q_aL.redistribute(dens.atom_partition.as_serial())

        alpha = 1 / (rcgauss / units.Bohr) ** 2
        dv_a1 = []
        for a, D_sp in dens.D_asp.items():
            setup = dens.setups[a]
            c = setup.xc_correction
            rgd = c.rgd
            ghat_g = gaussian(rgd, 1 / setup.rcgauss ** 2)
            Z_g = gaussian(rgd, alpha) * setup.Z
            D_q = np.dot(D_sp.sum(0), c.B_pqL[:, :, 0])
            dn_g = np.dot(D_q, (c.n_qg - c.nt_qg)) * np.sqrt(4 * np.pi)
            dn_g += 4 * np.pi * (c.nc_g - c.nct_g)
            dn_g -= Z_g
            dn_g -= dens.Q_aL[a][0] * ghat_g * np.sqrt(4 * np.pi)
            dv_g = rgd.poisson(dn_g) / np.sqrt(4 * np.pi)
            dv_g[1:] /= rgd.r_g[1:]
            dv_g[0] = dv_g[1]
            dv_g[-1] = 0.0
            dv_a1.append([rgd.spline(dv_g, points=spline_pts)])

        dens.D_asp.redistribute(dens.atom_partition)
        dens.Q_aL.redistribute(dens.atom_partition)

        if dv_a1:
            dv = LFC(new_gd, dv_a1)
            dv.set_positions(calc.spos_ac)
            dv.add(v_r)

        dens.gd.comm.broadcast(v_r, 0)

    return - v_r * units.Ha
