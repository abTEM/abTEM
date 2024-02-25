import warnings
from numbers import Number

import numpy as np

from abtem import PotentialArray, Waves
from abtem.bloch.matrix_exponential import expm
from abtem.core.axes import ThicknessAxis
from abtem.core.backend import get_array_module
from abtem.core.chunks import equal_sized_chunks
from abtem.core.constants import kappa
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.core.fft import fft_interpolate, ifftn
from abtem.core.grid import Grid
from abtem.core.utils import get_dtype
from abtem.parametrizations import validate_parametrization
from abtem.core.backend import validate_device


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


def slice_potential(
    potential: np.ndarray, depth: float, slice_thickness: float
) -> tuple[np.ndarray, np.ndarray]:
    n_slices = int(np.ceil(depth / slice_thickness))

    n_per_slice = equal_sized_chunks(potential.shape[-1], n_slices)
    dz = depth / potential.shape[-1]

    start = np.cumsum((0,) + n_per_slice)

    potential_sliced = np.stack(
        [
            np.sum(potential[..., start:stop], axis=-1) * dz
            for start, stop in zip(start[:-1], start[1:])
        ],
        axis=-1,
    )

    slice_thickness = np.array(n_per_slice) * dz

    return potential_sliced, slice_thickness


class StructureFactors:
    def __init__(
        self,
        atoms,
        k_max,
        parametrization="lobato",
        thermal_sigma=None,
        cutoff: str = "taper",
        device: str = None,
    ):
        self._atoms = atoms
        self._thermal_sigma = thermal_sigma
        self._k_max = k_max
        self._hkl = self._make_hkl_grid()
        self._g_vec = self._hkl @ self._atoms.cell.reciprocal()
        self._cutoff = cutoff
        self._parametrization = validate_parametrization(parametrization)
        self._device = validate_device(device)

    def __len__(self):
        return len(self._hkl)

    @property
    def gpts(self) -> tuple[int, int, int]:
        k_max = self._k_max

        if isinstance(self.k_max, Number):
            k_max = (self.k_max,) * 3

        assert len(k_max) == 3

        dk = self.atoms.cell.reciprocal().lengths()
        gpts = (
            int(np.ceil(k_max[0] / dk[0])) * 2 + 1,
            int(np.ceil(k_max[1] / dk[1])) * 2 + 1,
            int(np.ceil(k_max[2] / dk[2])) * 2 + 1,
        )
        return gpts

    def _make_hkl_grid(self):
        gpts = self.gpts
        freqs = tuple(np.fft.fftfreq(n, d=1 / n).astype(int) for n in gpts)
        hkl = np.meshgrid(*freqs, indexing="ij")
        hkl = np.stack(hkl, axis=-1)
        hkl = hkl.reshape((-1, 3))
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
    def atoms(self):
        return self._atoms

    @property
    def k_max(self):
        return self._k_max

    def _calculate_scattering_factors(self):
        Z_unique, Z_inverse = np.unique(self.atoms.numbers, return_inverse=True)
        g_unique, g_inverse = np.unique(self.g_vec_length, return_inverse=True)

        f_e_uniq = np.zeros(
            (Z_unique.size, g_unique.size), dtype=get_dtype(complex=True)
        )

        if self._cutoff == "taper":
            T = 0.005
            alpha = 1 - 0.025
            cutoff = 1 / (1 + np.exp((g_unique / self.k_max - alpha) / T))
        elif self._cutoff == "sharp":
            cutoff = g_unique > self.k_max
        elif self._cutoff == "none":
            cutoff = 1.0
        else:
            raise RuntimeError()

        for idx, Z in enumerate(Z_unique):
            if self._thermal_sigma is not None:
                DWF = np.exp(
                    -0.5 * self._thermal_sigma**2 * g_unique**2 * (2 * np.pi) ** 2
                )
            else:
                DWF = 1.0

            scattering_factor = self._parametrization.scattering_factor(Z)

            f_e_uniq[idx, :] = scattering_factor(g_unique**2) * DWF

            f_e_uniq[idx, :] *= cutoff

        return f_e_uniq[np.ix_(Z_inverse, g_inverse)]

    def calculate(self):
        positions = self.atoms.get_scaled_positions()

        f_e = self._calculate_scattering_factors()

        xp = get_array_module(self._device)

        f_e = xp.asarray(f_e, dtype=get_dtype(complex=True))
        positions = xp.asarray(positions, dtype=get_dtype(complex=False))
        hkl = xp.asarray(self.hkl.T, get_dtype(complex=False))

        struct_factors = xp.sum(
            f_e * xp.exp(-2.0j * np.pi * positions[:] @ hkl),
            axis=0,
        )

        struct_factors /= self.atoms.cell.volume

        return struct_factors.reshape(self.gpts)

    # def get_array(self, cache=True):
    #     if self._array is None:
    #         array = self._calculate_structure_factor()
    #         if cache:
    #             self._array = array
    #         return self._array
    #     else:
    #         return self._array

    def get_potential(self):
        array = self.calculate()

        # v = np.fft.ifftn(array).real
        # sampling = np.diag(self.atoms.cell) / self.gpts
        # v = v * self.atoms.cell.volume / (kappa * np.prod(sampling))
        # v -= v.min()

        potential = ifftn(array)
        potential = potential * np.prod(potential.shape) / kappa
        potential -= potential.min()

        return potential.real

    def get_projected_potential(self, slice_thickness, sampling=None, gpts=None):
        potential = self.get_potential()

        if sampling is not None:
            extent = np.diag(self.atoms.cell)[:2]
            grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
            gpts = grid.gpts

        if gpts is not None:
            potential = fft_interpolate(potential, gpts + (potential.shape[-1],))

        potential_sliced, slice_thickness = slice_potential(
            potential, depth=self.atoms.cell[2, 2], slice_thickness=slice_thickness
        )

        potential_sliced = np.rollaxis(potential_sliced, -1)

        sampling = (
            self.atoms.cell[0, 0] / potential_sliced.shape[-2],
            self.atoms.cell[1, 1] / potential_sliced.shape[-1],
        )
        potential_array = PotentialArray(
            potential_sliced,
            slice_thickness=tuple(slice_thickness),
            sampling=sampling,
        )

        return potential_array


def excitation_errors(g, energy):
    wavelength = energy2wavelength(energy)
    sg = (2 * g[:, 2] - wavelength * np.sum(g * g, axis=1)) / 2
    return sg


def get_reflection_condition(hkl: np.ndarray, centering: str):
    if centering.lower() == "f":
        all_even = (hkl % 2 == 0).all(axis=1)
        all_odd = (hkl % 2 == 1).all(axis=1)
        return all_even + all_odd
    elif centering.lower() == "i":
        return hkl.sum(axis=1) % 2 == 0
    elif centering.lower() == "a":
        return (hkl[1:].sum(axis=1) % 2 == 0).all(axis=1)
    elif centering.lower() == "b":
        return (hkl[:, [0, 1]].sum(axis=1) % 2 == 0).all(axis=1)
    elif centering.lower() == "c":
        return (hkl[:-1].sum(axis=1) % 2 == 0).all(axis=1)
    elif centering.lower() == "p":
        return np.ones(len(hkl), dtype=bool)
    else:
        raise ValueError()


def filter_vectors(hkl, cell, energy, sg_max, k_max, centering: str = "P"):
    g = hkl @ cell.reciprocal()

    g_length = np.linalg.norm(g, axis=-1)

    sg = excitation_errors(g, energy)

    mask = np.abs(sg) < sg_max

    mask *= get_reflection_condition(hkl, centering)

    mask *= g_length < k_max
    return mask


def calculate_M_matrix(hkl, cell, energy):
    g = hkl @ cell.reciprocal()
    k0 = 1 / energy2wavelength(energy)
    Mii = 1 / np.sqrt(1 + g[:, 2] / k0)
    return Mii


def calculate_A_matrix(hkl, cell, energy, structure_factor):
    xp = get_array_module(structure_factor)

    g = xp.asarray(hkl @ cell.reciprocal())
    Mii = calculate_M_matrix(hkl, cell, energy)
    hkl = xp.asarray(hkl)

    gmh = hkl[None] - hkl[:, None]

    A = (
        structure_factor[gmh[..., 0], gmh[..., 1], gmh[..., 2]]
        * energy2sigma(energy)
        / kappa
        / energy2wavelength(energy)
        / np.pi
    )

    Mii = xp.asarray(Mii)
    A *= Mii[None] * Mii[:, None]

    sg = xp.asarray(excitation_errors(g, energy))
    diag = 2 * 1 / energy2wavelength(energy) * sg
    diag *= Mii

    xp.fill_diagonal(A, diag)
    return A


def calculate_scattering_matrix(A, hkl, cell, z, energy):
    xp = get_array_module(A)
    S = expm(1.0j * xp.pi * z * A * energy2wavelength(energy))

    Mii = calculate_M_matrix(hkl, cell, energy)
    M = xp.asarray(np.diag(Mii))
    M_inv = xp.asarray(np.diag(1 / Mii))

    S = xp.dot(M, xp.dot(S, M_inv))
    return S


class BlochWaves:
    def __init__(
        self,
        structure_factor: StructureFactors,
        energy: float,
        sg_max: float,
        k_max: float = None,
        centering: str = "P",
        device: str = None,
    ):
        self._structure_factor = structure_factor
        self._energy = energy

        if k_max is None:
            k_max = structure_factor.k_max / 2
        elif k_max > structure_factor.k_max / 2:
            warnings.warn(
                "provided k_max exceed half the k_max of the scattering factors, some couplings are not included"
            )
        self._device = validate_device(device)

        self._hkl_mask = filter_vectors(
            hkl=structure_factor.hkl,
            cell=structure_factor.atoms.cell,
            energy=energy,
            sg_max=sg_max,
            k_max=k_max,
            centering=centering,
        )

    @property
    def hkl_mask(self):
        return self._hkl_mask

    @property
    def structure_factor(self):
        return self._structure_factor

    @property
    def energy(self):
        return self._energy

    @property
    def num_beams(self):
        return int(np.sum(self.hkl_mask))

    @property
    def wavelength(self):
        return energy2wavelength(self.energy)

    def excitation_errors(self):
        g = self.structure_factor.g_vec[self.hkl_mask]
        return excitation_errors(g, self.wavelength)

    @property
    def structure_matrix_nbytes(self):
        bytes_per_element = 128 // 8
        return self.num_beams**2 * bytes_per_element

    def calculate_structure_matrix(self):
        hkl = self.structure_factor.hkl[self.hkl_mask]
        structure_factor = self.structure_factor.calculate()

        A = calculate_A_matrix(
            hkl=hkl,
            cell=self.structure_factor.atoms.cell,
            energy=self.energy,
            structure_factor=structure_factor,
        )
        return A

    def calculate_scattering_matrix(self, z):
        A = self.calculate_structure_matrix()
        hkl = self.structure_factor.hkl[self.hkl_mask]
        cell = self.structure_factor.atoms.cell

        xp = get_array_module(self._device)
        A = xp.asarray(A)

        S = calculate_scattering_matrix(
            A=A, hkl=hkl, cell=cell, z=z, energy=self.energy
        )
        return S

    def calculate_exit_wave(self, thickness, n_steps, shape):
        hkl = self.structure_factor.hkl[self.hkl_mask]
        cell = self.structure_factor.atoms.cell
        dz = thickness / n_steps
        thicknesses = np.linspace(0, thickness, num=n_steps)
        S = self.calculate_scattering_matrix(dz)
        xp = get_array_module(S)

        psi_all = xp.zeros(shape=(n_steps, len(hkl)), dtype=complex)
        psi = (hkl == 0).all(axis=1).astype(complex)

        psi = xp.asarray(psi)

        for i in range(n_steps):
            psi = xp.dot(S, psi)
            psi_all[i] = psi

        wave_bw = xp.zeros(
            (len(psi_all),) + shape + (np.abs(hkl[:, 2]).max() + 1,), dtype=complex
        )

        g = xp.asarray(hkl @ cell.reciprocal())

        for i in range(len(psi_all)):
            phase = xp.exp(-2 * np.pi * 1.0j * g[:, 2] * thicknesses[i])
            wave_bw[i, hkl[:, 0], hkl[:, 1], hkl[:, 2]] = psi_all[i] * phase

        wave_bw = wave_bw.sum(-1)
        wave_bw = xp.fft.ifft2(wave_bw)

        waves = Waves(
            array=wave_bw,
            sampling=(cell[0, 0] / shape[0], cell[1, 1] / shape[1]),
            energy=self.energy,
            ensemble_axes_metadata=[
                ThicknessAxis(label="z", units="Å", values=tuple(thicknesses))
            ],
        )

        return waves

        # wave_bw = xp.zeros(
        #     (len(psi_all),) + shape + (np.abs(hkl[:, 2]).max() + 1,), dtype=complex
        # )
        #
        # g = hkl @ self.structure_factor.atoms.cell.reciprocal()
        # g = xp.asarray(g)
        #
        # depth = 0
        # for i in range(len(psi_all)):
        #     depth += z
        #     print(depth)
        #     phase = np.exp(-2 * np.pi * 1.0j * g[:, 2] * z)
        #     wave_bw[i, hkl[:, 0], hkl[:, 1], hkl[:, 2]] = psi_all[i] * phase
        #
        # wave_bw = wave_bw.sum(-1)
        # wave_bw = xp.fft.ifft2(wave_bw)
        # return wave_bw

    # def _make_plane_wave(self):
    #     hkl = self.structure_factors.hkl[self.included_hkl]
    #     psi_0 = cp.zeros((len(hkl),))
    #     psi_0[int(np.where((hkl == [0, 0, 0]).all(axis=1))[0])] = 1.0
    #     return psi_0
    #
    # def get_exit_wave(self, thicknesses, return_complex=False, device="cpu", tol=0.0):
    #     xp = get_array_module(device)
    #
    #     U_gmh = self.calculate_U_gmh()
    #
    #     U_gmh = xp.array(U_gmh)
    #
    #     v, C = xp.linalg.eigh(U_gmh)
    #
    #     gamma = v * self.wavelength / 2.0
    #
    #     if self.correct:
    #         C = C / xp.sqrt(
    #             1
    #             + self.wavelength
    #             * xp.array(
    #                 self.structure_factors.g_vec[self.included_hkl][:, 2][:, None]
    #             )
    #         )
    #
    #     C_inv = xp.conjugate(C.T)
    #
    #     psi_0 = self._make_plane_wave()
    #
    #     psi = [
    #         C @ (xp.exp(2.0j * xp.pi * thickness * gamma) * (C_inv @ psi_0))
    #         for thickness in thicknesses
    #     ]
    #
    #     psi = xp.stack(psi, axis=0)
    #
    #     if return_complex:
    #         return psi
    #     else:
    #         intensities = xp.abs(psi) ** 2
    #         intensities = xp.asnumpy(intensities)
    #
    #         intensities = pd.DataFrame(
    #             {
    #                 f"{h} {k} {l}": intensity
    #                 for ((h, k, l), intensity) in zip(
    #                     self.structure_factors.hkl[self.included_hkl], intensities.T
    #                 )
    #                 if np.any(intensity > tol)
    #             }
    #         )
    #         return intensities
