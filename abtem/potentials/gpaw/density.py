from math import pi, sqrt
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.units import Bohr

try:
    from gpaw.lfc import LFC, BasisFunctions
    from gpaw.transformers import Transformer
    from gpaw.utilities import unpack2

    if TYPE_CHECKING:
        from gpaw.setup import Setups

except ImportError:
    LFC = None
    BasisFunctions = None
    Transformer = None
    unpack2 = None
    Setups = None


def interpolate_pseudo_density(nt_sg, gd, gridrefinement=1):
    if gridrefinement == 1:
        return nt_sg, gd

    assert gridrefinement % 2 == 0

    iterations = int(np.log(gridrefinement) / np.log(2))

    finegd = gd
    n_sg = nt_sg

    for i in range(iterations):
        finegd = gd.refine()
        interpolator = Transformer(gd, finegd, 3)

        n_sg = finegd.empty(nt_sg.shape[0])

        for s in range(nt_sg.shape[0]):
            interpolator.apply(nt_sg[s], n_sg[s])

        nt_sg = n_sg
        gd = finegd

    return n_sg, finegd


def get_all_electron_density(nt_sG,
                             gd,
                             D_asp: dict,
                             setups,
                             atoms: Atoms,
                             gridrefinement: int = 1):
    nspins = nt_sG.shape[0]
    spos_ac = atoms.get_scaled_positions() % 1.0

    n_sg, gd = interpolate_pseudo_density(nt_sG, gd, gridrefinement)

    phi_aj = []
    phit_aj = []
    nc_a = []
    nct_a = []
    for setup in setups:
        phi_j, phit_j, nc, nct = setup.get_partial_waves()[:4]
        phi_aj.append(phi_j)
        phit_aj.append(phit_j)
        nc_a.append([nc])
        nct_a.append([nct])

    # Create localized functions from splines
    phi = BasisFunctions(gd, phi_aj)
    phit = BasisFunctions(gd, phit_aj)
    nc = LFC(gd, nc_a)
    nct = LFC(gd, nct_a)
    phi.set_positions(spos_ac)
    phit.set_positions(spos_ac)
    nc.set_positions(spos_ac)
    nct.set_positions(spos_ac)

    I_sa = np.zeros((nspins, len(spos_ac)))
    a_W = np.empty(len(phi.M_W), np.intc)
    W = 0
    for a in phi.atom_indices:
        nw = len(phi.sphere_a[a].M_w)
        a_W[W:W + nw] = a
        W += nw

    x_W = phi.create_displacement_arrays()[0]

    rho_MM = np.zeros((phi.Mmax, phi.Mmax))
    for s, I_a in enumerate(I_sa):
        M1 = 0
        for a, setup in enumerate(setups):
            ni = setup.ni
            D_sp = D_asp.get(a % len(D_asp))
            if D_sp is None:
                D_sp = np.empty((nspins, ni * (ni + 1) // 2))
            else:
                I_a[a] = setup.Nct / nspins - np.sqrt(4 * pi) * np.dot(D_sp[s], setup.Delta_pL[:, 0])
                I_a[a] -= setup.Nc / nspins

            # rank = D_asp.partition.rank_a[a]
            # D_asp.partition.comm.broadcast(D_sp, rank)
            M2 = M1 + ni
            rho_MM[M1:M2, M1:M2] = unpack2(D_sp[s])
            M1 = M2

        assert np.all(n_sg[s].shape == phi.gd.n_c)
        phi.lfc.ae_valence_density_correction(rho_MM, n_sg[s], a_W, I_a, x_W)
        phit.lfc.ae_valence_density_correction(-rho_MM, n_sg[s], a_W, I_a, x_W)

    a_W = np.empty(len(nc.M_W), np.intc)
    W = 0
    for a in nc.atom_indices:
        nw = len(nc.sphere_a[a].M_w)
        a_W[W:W + nw] = a
        W += nw
    scale = 1.0 / nspins

    for s, I_a in enumerate(I_sa):
        nc.lfc.ae_core_density_correction(scale, n_sg[s], a_W, I_a)
        nct.lfc.ae_core_density_correction(-scale, n_sg[s], a_W, I_a)
        # D_asp.partition.comm.sum(I_a)

        N_c = gd.N_c
        g_ac = np.around(N_c * spos_ac).astype(int) % N_c - gd.beg_c

        for I, g_c in zip(I_a, g_ac):
            if np.all(g_c >= 0) and np.all(g_c < gd.n_c):
                n_sg[s][tuple(g_c)] -= I / gd.dv

    return n_sg.sum(0) / Bohr ** 3
