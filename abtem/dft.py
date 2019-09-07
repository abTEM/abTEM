import numpy as np
from ase import Atoms
from ase import units
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d


def create_orthogonal_view(origin, extent, atoms, array, new_gpts):
    origin = np.array(origin)
    extent = np.array(extent)

    P = np.array(atoms.cell[:2, :2])
    P_inv = np.linalg.inv(P)
    origin_t = np.dot(origin, P_inv)
    origin_t = origin_t % 1.0

    lower = np.dot(origin_t, P)
    upper = lower + extent

    n, m = np.ceil(np.dot(upper, P_inv)).astype(np.int)

    # print(upper,P_inv)

    atoms = atoms.copy()

    mapping = np.arange(0, len(atoms))
    mapping = np.tile(mapping, (n + 2) * (m + 2))

    displacement = atoms.cell[:2, :2]

    atoms *= (n + 2, m + 2, 1)

    positions = atoms.get_positions()
    positions[:, :2] -= displacement.sum(axis=0)

    inside = ((positions[:, 0] > lower[0]) & (positions[:, 1] > lower[1]) &
              (positions[:, 0] < upper[0]) & (positions[:, 1] < upper[1]))

    atomic_numbers = atoms.get_atomic_numbers()[inside]
    positions = positions[inside]
    mapping = mapping[inside]

    positions[:, :2] -= lower

    tiled = np.tile(array, (n + 2, m + 2))
    x = np.linspace(-1, n + 1, tiled.shape[0], endpoint=False)
    y = np.linspace(-1, m + 1, tiled.shape[1], endpoint=False)

    x_ = np.linspace(lower[0], upper[0], new_gpts[0], endpoint=False)
    y_ = np.linspace(lower[1], upper[1], new_gpts[1], endpoint=False)
    x_, y_ = np.meshgrid(x_, y_, indexing='ij')

    p = np.array([x_.ravel(), y_.ravel()]).T
    p = np.dot(p, P_inv)
    # print(p.min())

    p[:, 0] = p[:, 0] % n
    p[:, 1] = p[:, 1] % m
    # print(p.min(),n,m)
    p[:, 0] = np.clip(p[:, 0], 0, x.max())
    p[:, 1] = np.clip(p[:, 1], 0, y.max())

    interpolated = RegularGridInterpolator((x, y), tiled)(p)
    new_atoms = Atoms(atomic_numbers, positions=positions, cell=[extent[0], extent[1], atoms.cell[2, 2]])
    return new_atoms, interpolated.reshape(new_gpts), mapping


def gaussian(rgd, alpha):
    r_g = rgd.r_g
    g_g = 4 / np.sqrt(np.pi) * alpha ** 1.5 * np.exp(-alpha * r_g ** 2)
    return g_g


def get_paw_corrections(calculator, rcgauss=0.02, spline_pts=200):
    dens = calculator.density
    dens.D_asp.redistribute(dens.atom_partition.as_serial())
    dens.Q_aL.redistribute(dens.atom_partition.as_serial())

    alpha = 1 / (rcgauss / units.Bohr) ** 2
    corrections = []
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
        corrections.append(rgd.spline(dv_g, points=spline_pts))
    return corrections


def riemann_quadrature(num_samples):
    xk = np.linspace(-1., 1., num_samples + 1).astype(np.float32)
    wk = xk[1:] - xk[:-1]
    xk = xk[:-1]
    return xk, wk


def project_spherical_function(func, r, a, b, num_samples=400):
    c = np.reshape(((b - a) / 2.), (-1, 1))
    d = np.reshape(((b + a) / 2.), (-1, 1))

    xk, wk = riemann_quadrature(num_samples)

    xkab = xk * c + d
    wkab = wk * c

    rxy = np.sqrt((r ** 2.).reshape((-1, 1)) + (xkab ** 2.).reshape((1, -1)))
    return np.sum(func(rxy) * wkab.reshape(1, -1), axis=-1)


def get_projected_paw_potential_corrections(gpts, extent, positions, splines, slice_entrance, slice_exit):
    sampling = np.array(extent) / gpts

    def get_indices_in_slice(positions, slice_entrance, slice_exit, rc):
        return np.where(((slice_entrance - rc) < positions[:, 2]) &
                        ((slice_exit + rc) > positions[:, 2]))[0]

    rcmax = max([spline.get_cutoff() for spline in splines]) / units.Bohr

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


# def get_projected_paw_potential(calculator, origin, extent, gpts):
#     funcs = get_paw_corrections(calculator, rcgauss=0.02)


#    return


class Interpolator:
    def __init__(self, gd1, gd2, dtype=float):
        from gpaw.wavefunctions.pw import PWDescriptor
        self.pd1 = PWDescriptor(0.0, gd1, dtype)
        self.pd2 = PWDescriptor(0.0, gd2, dtype)

    def interpolate(self, a_r):
        return self.pd1.interpolate(a_r, self.pd2)[0]


def get_3d_potential_from_GPAW(calc, h=.05, rcgauss=0.02, spline_pts=200, n=2, add_corrections=True):
    from gpaw.lfc import LFC
    from gpaw.utilities import h2gpts
    from gpaw.fftw import get_efficient_fft_size
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.mpi import serial_comm

    v_r = calc.get_electrostatic_potential() / units.Ha

    old_gd = GridDescriptor(calc.hamiltonian.finegd.N_c, calc.hamiltonian.finegd.cell_cv, comm=serial_comm)

    N_c = h2gpts(h / units.Bohr, calc.wfs.gd.cell_cv)
    N_c = np.array([get_efficient_fft_size(N, n) for N in N_c])
    new_gd = GridDescriptor(N_c, calc.wfs.gd.cell_cv, comm=serial_comm)

    interpolator = Interpolator(old_gd, new_gd)

    v_r = interpolator.interpolate(v_r)

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