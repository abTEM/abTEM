import numpy as np
from ase import units
from scipy.interpolate import interp1d

from .bases import cached_method, HasCache
from .transform import make_orthogonal_atoms, make_orthogonal_array
from .potentials import PotentialBase


def gaussian(radial_grid, alpha):
    return 4 / np.sqrt(np.pi) * alpha ** 1.5 * np.exp(-alpha * radial_grid.r_g ** 2)


def get_paw_corrections(calculator, rcgauss=0.01, spline_pts=200):
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


class GPAWPotential(PotentialBase, HasCache):

    def __init__(self, calc, origin=None, extent=None, gpts=None, sampling=None, slice_thickness=.5, num_slices=None):
        self._atoms = calc.atoms
        self._calc = calc

        if origin is None:
            origin = np.array([0., 0.])
        self._origin = origin

        if extent is None:
            extent = np.diag(self._atoms.cell[:2, :2])

        if num_slices is None:
            if slice_thickness is None:
                raise RuntimeError()

            self._num_slices = int(np.floor(self._atoms.cell[2, 2] / slice_thickness))
        else:
            self._num_slices = num_slices

        assert calc.hamiltonian.finegd.N_c[2] % self._num_slices == 0

        PotentialBase.__init__(self, extent=extent, gpts=gpts, sampling=sampling)
        HasCache.__init__(self)

    @property
    def num_slices(self):
        return self._num_slices

    @property
    def thickness(self):
        return self._atoms.cell[2, 2]

    @cached_method
    def _prepare_paw_corrections(self):
        paw_corrections = get_paw_corrections(self._calc)
        atoms, equivalent = make_orthogonal_atoms(self._atoms, self._origin, self.extent, return_equivalent=True)
        max_cutoff = max([spline.get_cutoff() for spline in paw_corrections.values()]) / units.Bohr
        return paw_corrections, atoms, equivalent, max_cutoff

    @cached_method
    def _prepare_electrostatic_potential(self):
        return self._calc.get_electrostatic_potential()

    def get_slice(self, i):
        array = self._prepare_electrostatic_potential()

        nz = array.shape[-1] // self.num_slices
        dz = self._atoms.cell[2, 2] / array.shape[-1]

        start = i * nz
        stop = (i + 1) * nz
        slice_entrance = start * dz
        slice_exit = stop * dz

        array = array[..., start:stop].sum(axis=-1) * dz

        array = make_orthogonal_array(array, self._atoms.cell[:2, :2], self._origin, self.extent, self.gpts)
        array = self._get_projected_paw_corrections(slice_entrance, slice_exit) - array
        return array

    def _get_projected_paw_corrections(self, slice_entrance, slice_exit):
        paw_corrections, atoms, equivalent, max_cutoff = self._prepare_paw_corrections()

        positions = atoms.get_positions()

        margin = int(np.ceil(max_cutoff / min(self.sampling)))
        padded_gpts = self.gpts + 2 * margin
        v = np.zeros(padded_gpts)

        idx_in_slice = np.where(((slice_entrance - max_cutoff) < positions[:, 2]) &
                                ((slice_exit + max_cutoff) > positions[:, 2]))[0]
        # np.where(np.abs(self.slice_entrance(i) + self.slice_thickness / 2 - positions[:, 2]) <
        #                    (max_cutoff + self.slice_thickness / 2))[0]

        for idx in idx_in_slice:
            func = paw_corrections[equivalent[idx]]
            position = positions[idx]

            rc = func.get_cutoff() / units.Bohr

            block_margin = np.int(rc / min(self.sampling))
            block_size = 2 * block_margin + 1

            a = slice_entrance - position[2]
            b = slice_exit - position[2]

            nodes = np.linspace(0., rc, 1000)

            radial = project_spherical_function(func.map, nodes, a / units.Bohr, b / units.Bohr) * units.Bohr

            block_position = (np.round(position[:2] / self.sampling)).astype(np.int) - block_margin + margin

            position_in_block = position[:2] + self.sampling * margin - self.sampling * block_position

            x = np.linspace(0., block_size * self.sampling[0] - self.sampling[0], block_size)
            y = np.linspace(0., block_size * self.sampling[1] - self.sampling[1], block_size)
            r_interp = np.sqrt(((x - position_in_block[0]) ** 2)[:, None] +
                               ((y - position_in_block[1]) ** 2)[None, :]) / units.Bohr

            f = interp1d(nodes, radial, fill_value=0., bounds_error=False)
            v_interp = f(r_interp)

            v[block_position[0]: block_position[0] + block_size,
            block_position[1]: block_position[1] + block_size] += v_interp

        v[margin:2 * margin] += v[-margin:]
        v[-2 * margin:-margin] += v[:margin]
        v[:, margin:2 * margin] += v[:, -margin:]
        v[:, -2 * margin:-margin] += v[:, :margin]

        return - v[margin:-margin, margin:-margin] / np.sqrt(4 * np.pi) * units.Ha


def project_paw_potential_corrections(gpts, extent, positions, splines, slice_entrance, slice_exit):
    sampling = np.array(extent) / gpts

    def get_indices_in_slice(positions, slice_entrance, slice_exit, rc):
        return np.where(((slice_entrance - rc) < positions[:, 2]) &
                        ((slice_exit + rc) > positions[:, 2]))[0]

    rcmax = max([spline.get_cutoff() for spline in splines.values()]) / units.Bohr

    margin = np.ceil(rcmax / min(sampling)).astype(np.int32)
    padded_gpts = gpts + 2 * margin

    v = np.zeros(padded_gpts)

    for i in get_indices_in_slice(positions, slice_entrance, slice_exit, rcmax):
        func = splines[i]
        position = positions[i]

        rc = func.get_cutoff() / units.Bohr

        block_margin = np.int(rc / min(sampling))
        block_size = 2 * block_margin + 1

        a = slice_entrance - position[2]
        b = slice_exit - position[2]

        nodes = np.linspace(0., rc, 500)

        radial = project_spherical_function(func.map, nodes, a / units.Bohr, b / units.Bohr) * units.Bohr

        corner_index = (np.round(position[:2] / sampling)).astype(np.int) - block_margin + margin

        position_in_block = position[:2] + sampling * margin - sampling * corner_index

        x = np.linspace(0., block_size * sampling[0] - sampling[0], block_size)
        y = np.linspace(0., block_size * sampling[1] - sampling[1], block_size)
        r_interp = np.sqrt(((x - position_in_block[0]) ** 2)[:, None] +
                           ((y - position_in_block[1]) ** 2)[None, :]) / units.Bohr

        r_interp = np.clip(r_interp, 0, rc)

        f = interp1d(nodes, radial)
        v_interp = f(r_interp)

        v[corner_index[0]: corner_index[0] + block_size, corner_index[1]: corner_index[1] + block_size] += v_interp

    v[margin:2 * margin] += v[-margin:]
    v[-2 * margin:-margin] += v[:margin]
    v[:, margin:2 * margin] += v[:, -margin:]
    v[:, -2 * margin:-margin] += v[:, :margin]

    return - v[margin:-margin, margin:-margin] / np.sqrt(4 * np.pi) * units.Ha


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
