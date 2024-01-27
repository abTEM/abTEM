import warnings

import numpy as np
import pandas as pd

from abtem.core.backend import get_array_module, cp
from abtem.core.constants import kappa
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.parametrizations import validate_parametrization


def _F_reflection_conditions(hkl):
    all_even = (hkl % 2 == 0).all(axis=1)
    all_odd = (hkl % 2 == 1).all(axis=1)
    return all_even + all_odd


def _I_reflection_conditions(hkl):
    return (hkl.sum(axis=1) % 2 == 0).all(axis=1)


def _A_reflection_conditions(hkl):
    return (hkl[1:].sum(axis=1) % 2 == 0).all(axis=1)


def _B_reflection_conditions(hkl):
    return (hkl[:, [0, 1]].sum(axis=1) % 2 == 0).all(axis=1)


def _C_reflection_conditions(hkl):
    return (hkl[:-1].sum(axis=1) % 2 == 0).all(axis=1)


class StructureFactors:
    def __init__(self, atoms, k_max, parametrization="lobato", thermal_sigma=None):
        self._atoms = atoms
        self._thermal_sigma = thermal_sigma
        self._k_max = k_max
        self._hkl = self._make_hkl_grid()
        self._g_vec = self._hkl @ self._atoms.cell.reciprocal()
        self._array = None
        self._parametrization = validate_parametrization(parametrization)

    def __len__(self):
        return len(self._hkl)

    def _make_hkl_grid(self):
        num_tile = int(np.ceil(self._k_max / min(self.dg)))
        self._gpts = num_tile * 2 + 1
        indices = np.fft.fftshift(np.fft.fftfreq(self._gpts, d=1 / self._gpts)).astype(
            int
        )
        ya, xa, za = np.meshgrid(*(indices,) * 3)
        hkl = np.vstack([xa.ravel(), ya.ravel(), za.ravel()]).T
        return hkl

    @property
    def dg(self):
        return np.linalg.norm(self.atoms.cell.reciprocal(), axis=1)

    @property
    def hkl(self):
        return self._hkl

    @property
    def g_vec(self):
        return self._g_vec

    @property
    def g_vec_length(self):
        return np.linalg.norm(self._g_vec, axis=1)

    @property
    def gpts(self):
        return self._gpts

    @property
    def atoms(self):
        return self._atoms

    @property
    def k_max(self):
        return self._k_max

    def _calculate_scattering_factors(self):
        Z_unique, Z_inverse = np.unique(self.atoms.numbers, return_inverse=True)
        g_unique, g_inverse = np.unique(self.g_vec_length, return_inverse=True)

        f_e_uniq = np.zeros((Z_unique.size, g_unique.size), dtype=np.complex128)

        for idx, Z in enumerate(Z_unique):
            if self._thermal_sigma is not None:
                DWF = np.exp(-0.5 * self._thermal_sigma**2 * g_unique**2 * (2*np.pi)**2)
            else:
                DWF = 1.0

            scattering_factor = self._parametrization.scattering_factor(Z)
            f_e_uniq[idx, :] = scattering_factor(g_unique**2) * DWF

            f_e_uniq[idx, g_unique > self.k_max] = 0.0

        return f_e_uniq[np.ix_(Z_inverse, g_inverse)]

    def get_array(self, cache=True):
        if self._array is None:
            array = self._calculate_structure_factor()
            if cache:
                self._array = array
            return self._array
        else:
            return self._array

    def _calculate_structure_factor(self):
        positions = self.atoms.get_scaled_positions()

        f_e = self._calculate_scattering_factors()

        struct_factors = np.sum(
            f_e * np.exp(2.0j * np.pi * np.squeeze(positions[:, None, :] @ self.hkl.T)),
            axis=0,
        )

        struct_factors /= self.atoms.cell.volume
        struct_factors[self.g_vec_length >= self.k_max] = 0.0

        return struct_factors.reshape((self.gpts,) * 3)

    def get_potential(self):
        v = np.fft.ifftn(np.fft.ifftshift(self._calculate_structure_factor()))
        sampling = np.diag(self.atoms.cell) / self.gpts
        v = v * self.atoms.cell.volume / (kappa * np.prod(sampling))
        v -= v.min()
        return v


def excitation_errors(g, wavelength):
    sg = (2 * g[:, 2] - wavelength * np.sum(g * g, axis=1)) / 2
    return sg


class BlochWaves:
    def __init__(self, structure_factors, energy, sg_max, k_max=None, correct=True):
        self._structure_factors = structure_factors
        self._energy = energy

        if k_max is None:
            k_max = self.structure_factors.k_max / 2
        elif k_max > self.structure_factors.k_max / 2:
            warnings.warn(
                "provided k_max exceed half the k_max of the scattering factors, some couplings are not included"
            )

        sg = excitation_errors(self.structure_factors.g_vec, self.wavelength)

        self._included_hkl = np.where(
            (self.structure_factors.g_vec_length <= k_max)
            & (np.abs(sg) <= sg_max)
            & _F_reflection_conditions(self.structure_factors.hkl)
        )[0]

        self.correct = correct

    @property
    def included_hkl(self):
        return self._included_hkl

    @property
    def structure_factors(self):
        return self._structure_factors

    @property
    def energy(self):
        return self._energy

    @property
    def wavelength(self):
        return energy2wavelength(self.energy)

    def excitation_errors(self):
        g = self.structure_factors.g_vec[self.included_hkl]
        return excitation_errors(g, self.wavelength)

    @property
    def size(self):
        return len(self.included_hkl) ** 2 * 128 * 0.125

    def calculate_U_gmh(self):
        hkl = self.structure_factors.hkl[self.included_hkl]
        g_vec = self.structure_factors.g_vec[self.included_hkl]
        n_beams = len(hkl)

        gmh = np.array(
            (
                (hkl[:, 0][None] - hkl[:, 0][:, None]).ravel(),
                (hkl[:, 1][None] - hkl[:, 1][:, None]).ravel(),
                (hkl[:, 2][None] - hkl[:, 2][:, None]).ravel(),
            )
        ).T

        struct_factors = self._structure_factors.get_array()

        U_gmh = struct_factors[
            gmh[:, 0] - self.structure_factors.hkl[:, 0].min(),
            gmh[:, 1] - self.structure_factors.hkl[:, 1].min(),
            gmh[:, 2] - self.structure_factors.hkl[:, 2].min(),
        ]

        prefactor = energy2sigma(self.energy) / self.wavelength / np.pi / kappa

        U_gmh = prefactor * U_gmh
        U_gmh = U_gmh.reshape((n_beams, n_beams))

        sg = self.excitation_errors()
        diag = 2 / self.wavelength * sg

        if self.correct:
            U_gmh /= np.sqrt(1 + self.wavelength * g_vec[:, 2][:, None]) * np.sqrt(
                1 + self.wavelength * g_vec[:, 2][None]
            )

            diag /= 1 + self.wavelength * g_vec[:, 2]

        np.fill_diagonal(U_gmh, diag)

        return U_gmh

    def _make_plane_wave(self):
        hkl = self.structure_factors.hkl[self.included_hkl]
        psi_0 = cp.zeros((len(hkl),))
        psi_0[int(np.where((hkl == [0, 0, 0]).all(axis=1))[0])] = 1.0
        return psi_0

    def get_exit_wave(self, thicknesses, return_complex=False, device="cpu", tol=0.0):
        xp = get_array_module(device)

        U_gmh = self.calculate_U_gmh()

        U_gmh = xp.array(U_gmh)

        v, C = xp.linalg.eigh(U_gmh)

        gamma = v * self.wavelength / 2.0

        if self.correct:
            C = C / xp.sqrt(
                1
                + self.wavelength
                * xp.array(
                    self.structure_factors.g_vec[self.included_hkl][:, 2][:, None]
                )
            )

        C_inv = xp.conjugate(C.T)

        psi_0 = self._make_plane_wave()

        psi = [
            C @ (xp.exp(2.0j * xp.pi * thickness * gamma) * (C_inv @ psi_0))
            for thickness in thicknesses
        ]

        psi = xp.stack(psi, axis=0)

        if return_complex:
            return psi
        else:
            intensities = xp.abs(psi) ** 2
            intensities = xp.asnumpy(intensities)

            intensities = pd.DataFrame(
                {
                    f"{h} {k} {l}": intensity
                    for ((h, k, l), intensity) in zip(
                        self.structure_factors.hkl[self.included_hkl], intensities.T
                    )
                    if np.any(intensity > tol)
                }
            )
            return intensities
