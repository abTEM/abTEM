from typing import Sequence
import warnings
from functools import partial
from numbers import Number

import dask.array as da
import numpy as np
from ase import Atoms
from ase.cell import Cell
from scipy.linalg import expm as expm_scipy
from scipy.spatial.transform import Rotation


from abtem.array import ArrayObject
from abtem.bloch.utils import excitation_errors
from abtem.potentials.iam import PotentialArray
from abtem.waves import Waves
from abtem.core import config
from abtem.core.axes import AxisMetadata, NonLinearAxis, ThicknessAxis
from abtem.core.backend import cp, get_array_module, validate_device
from abtem.core.chunks import equal_sized_chunks
from abtem.core.complex import abs2
from abtem.core.constants import kappa
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.core.ensemble import Ensemble, _wrap_with_array, unpack_blockwise_args
from abtem.core.fft import fft_interpolate
from abtem.core.grid import Grid
from abtem.core.utils import CopyMixin, flatten_list_of_lists, get_dtype
from abtem.distributions import validate_distribution
from abtem.parametrizations import validate_parametrization
from abtem.measurements import IndexedDiffractionPatterns

if cp is not None:
    from abtem.bloch.matrix_exponential import expm as expm_cupy


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


class StructureFactor:
    """
    The StructureFactors class calculates the structure factors for a given set of atoms and parametrization.

    Parameters
    ----------
    atoms : Atoms
        Atoms object.
    k_max : float
        Maximum scattering vector length [1/Å].
    parametrization : str
        Parametrization for the scattering factors.
    thermal_sigma : float
        Standard deviation of the atomic displacements for the Debye-Waller factor [Å].
    cutoff : {'taper', 'sharp' or 'none'}
        Cutoff function for the scattering factors. 'taper' is a smooth cutoff, 'sharp' is a hard cutoff and 'none' is no cutoff.
    device : str
        Device to use for calculations. Can be 'cpu' or 'gpu'.
    """

    def __init__(
        self,
        atoms: Atoms,
        k_max: float,
        parametrization: str = "lobato",
        thermal_sigma: float = None,
        cutoff: str = "taper",
        device: str = None,
    ):

        self._atoms = atoms
        self._thermal_sigma = thermal_sigma
        self._k_max = k_max
        self._hkl = self._make_hkl_grid()
        self._g_vec = self._hkl @ self._atoms.cell.reciprocal()

        if not cutoff in ("taper", "sharp", "none"):
            raise ValueError("cutoff must be 'taper', 'sharp' or 'none'")

        self._cutoff = cutoff
        self._parametrization = validate_parametrization(parametrization)
        self._device = validate_device(device)

    def __len__(self):
        return len(self._hkl)

    @property
    def gpts(self) -> tuple[int, int, int]:
        """
        Number of reciprocal space grid points.
        """
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

    def calculate_scattering_factors(self) -> np.ndarray:
        """
        Calculate the scattering factors for each atomic species in the structure.
        """
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

        f_e = self.calculate_scattering_factors()

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

    def get_potential(self):
        array = self.calculate()
        xp = get_array_module(array)
        potential = xp.fft.ifftn(array)
        potential = potential * np.prod(potential.shape) / kappa
        potential -= potential.min()
        return potential.real

    def get_projected_potential(
        self,
        slice_thickness: float | Sequence[float],
        sampling: float | tuple[float, float] = None,
        gpts: int | tuple[int, int] = None,
    ) -> PotentialArray:
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


class StructureFactorArray(ArrayObject):

    def __init__(
        self,
        array: np.ndarray,
        hkl: np.ndarray,
        cell: np.ndarray | Cell,
        k_max: float,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        metadata: dict = None,
    ):
        self._hkl = hkl
        self._cell = cell
        self._k_max = k_max

        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    def to_3d_array(self):
        xp = get_array_module(self.array)
        array = xp.zeros(self.gpts, dtype=self.array.dtype)
        array[self.hkl[:, 0], self.hkl[:, 1], self.hkl[:, 2]] = self.array
        return array


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


def expm(a):
    xp = get_array_module(a)
    if xp == cp:
        return expm_cupy(a)
    else:
        return expm_scipy(a)


def calculate_scattering_matrix(A, hkl, cell, z, energy):
    xp = get_array_module(A)

    S = expm(1.0j * xp.pi * z * A * energy2wavelength(energy))

    Mii = calculate_M_matrix(hkl, cell, energy)
    M = xp.asarray(np.diag(Mii))
    M_inv = xp.asarray(np.diag(1 / Mii))

    S = xp.dot(M, xp.dot(S, M_inv))
    return S


def validate_k_max(k_max, structure_factor):
    if k_max is None:
        k_max = structure_factor.k_max / 2
    elif k_max > structure_factor.k_max / 2:
        warnings.warn(
            "provided k_max exceed half the k_max of the scattering factors, some couplings are not included"
        )
    return k_max


class BlochWaves:
    def __init__(
        self,
        structure_factor: StructureFactor,
        energy: float,
        sg_max: float,
        k_max: float = None,
        orientation_matrix: np.ndarray = None,
        centering: str = "P",
        device: str = None,
    ):
        self._structure_factor = structure_factor
        self._energy = energy
        self._sg_max = sg_max
        self._k_max = validate_k_max(k_max, structure_factor)
        cell = structure_factor.atoms.cell

        if orientation_matrix is not None:
            cell = Cell(np.dot(cell, orientation_matrix.T))

        self._cell = cell
        self._centering = centering
        self._device = validate_device(device)

        self._hkl_mask = filter_reciprocal_space_vectors(
            hkl=structure_factor.hkl,
            cell=self._cell,
            energy=energy,
            sg_max=sg_max,
            k_max=self._k_max,
            centering=centering,
        )

    @property
    def hkl_mask(self):
        return self._hkl_mask

    @property
    def hkl(self):
        return self.structure_factor.hkl[self.hkl_mask]

    @property
    def g_vec(self):
        return self.hkl @ self._cell.reciprocal()

    @property
    def centering(self):
        return self._centering

    @property
    def cell(self):
        return self._cell

    @property
    def k_max(self):
        return self._k_max

    @property
    def sg_max(self):
        return self._sg_max

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
        return excitation_errors(self.g_vec, self.energy)

    @property
    def structure_matrix_nbytes(self):
        bytes_per_element = 128 // 8
        return self.num_beams**2 * bytes_per_element

    def get_kinematical_diffraction_pattern(self, excitation_error_sigma: float = None):
        hkl = self.hkl
        g_vec = self.g_vec

        S = self.structure_factor.calculate().flatten()[self.hkl_mask]
        sg = self.excitation_errors()

        S = abs2(S)

        if excitation_error_sigma is None:
            excitation_error_sigma = self._sg_max / 3.0

        intensity = S * np.exp(-(sg**2) / (2.0 * excitation_error_sigma**2))

        metadata = {"energy": self.energy, "sg_max": self._sg_max, "k_max": self.k_max}

        return IndexedDiffractionPatterns(
            miller_indices=hkl, array=intensity, positions=g_vec, metadata=metadata
        )

    def calculate_structure_matrix(self):
        hkl = self.hkl
        structure_factor = self.structure_factor.calculate()

        A = calculate_A_matrix(
            hkl=hkl,
            cell=self.cell,
            energy=self.energy,
            structure_factor=structure_factor,
        )
        return A

    def calculate_scattering_matrix(self, z):
        A = self.calculate_structure_matrix()
        hkl = self.hkl
        cell = self.cell

        xp = get_array_module(self._device)
        A = xp.asarray(A)

        S = calculate_scattering_matrix(
            A=A, hkl=hkl, cell=cell, z=z, energy=self.energy
        )
        return S

    def calculate_diffraction_patterns(self, thicknesses, return_complex: bool = False):
        hkl = self.structure_factor.hkl[self.hkl_mask]
        cell = self.cell
        A = self.calculate_structure_matrix()

        xp = get_array_module(A)

        Mii = xp.asarray(calculate_M_matrix(hkl, cell, self.energy))

        v, C = xp.linalg.eigh(A)
        gamma = v * self.wavelength / 2.0

        np.fill_diagonal(C, np.diag(C) / Mii)

        C_inv = xp.conjugate(C.T)

        initial = np.all(hkl == 0, axis=1).astype(complex)
        initial = xp.asarray(initial)

        array = xp.zeros(shape=(len(thicknesses), len(hkl)), dtype=complex)

        for i, thickness in enumerate(thicknesses):
            alpha = C_inv @ initial
            array[i] = C @ (xp.exp(2.0j * xp.pi * thickness * gamma) * alpha)

        if not return_complex:
            array = abs2(array)

        ensemble_axes_metadata = [
            ThicknessAxis(label="z", units="Å", values=tuple(thicknesses))
        ]

        reciprocal_lattice_vectors = np.tile(cell, (len(thicknesses), 1, 1))

        return IndexedDiffractionPatterns(
            miller_indices=hkl,
            array=array,
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata={
                "energy": self.energy,
                "sg_max": self.sg_max,
                "k_max": self.k_max,
            },
        )

    def calculate_exit_wave(self, thickness, n_steps, shape):
        cell = self._cell
        thicknesses = np.linspace(0, thickness, num=n_steps)

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

    def rotate(self, *args):
        ensemble = BlochwaveEnsemble(
            *args,
            structure_factor=self.structure_factor,
            energy=self.energy,
            sg_max=self.sg_max,
            k_max=self.k_max,
            centering=self._centering,
            device=self._device,
        )
        return ensemble


class BlochwaveEnsemble(Ensemble, CopyMixin):

    def __init__(
        self,
        *args,
        structure_factor: StructureFactor,
        energy: float,
        sg_max: float,
        k_max: float,
        centering: str = "P",
        device: str = None,
    ):
        self._axes = args[::2]
        self._rotations = tuple(
            validate_distribution(rotation) for rotation in args[1::2]
        )

        self._structure_factor = structure_factor
        self._energy = energy
        self._centering = centering
        self._sg_max = sg_max
        self._k_max = k_max
        self._device = validate_device(device)

    def get_ensemble_hkl_mask(self):
        hkl = self._structure_factor.hkl
        mask = filter_reciprocal_space_vectors(
            hkl=hkl,
            cell=self._structure_factor.atoms.cell,
            energy=self.energy,
            sg_max=self.sg_max,
            k_max=self.k_max,
            centering=self.centering,
            orientation_matrix=self.get_orientation_matrices().reshape(-1, 3, 3),
        )
        return mask

    def get_orientation_matrices(self):
        orientation_matrices = np.eye(3)
        for axes, rotation in zip(self.axes[::-1], self.rotations[::-1]):

            if hasattr(rotation, "values"):
                R = Rotation.from_euler(axes, rotation.values).as_matrix()
                R = R[(slice(None),) + (None,) * (orientation_matrices.ndim - 2)]
            else:
                R = Rotation.from_euler(axes, rotation).as_matrix()

            orientation_matrices = orientation_matrices @ R

        return orientation_matrices

    @property
    def structure_factor(self):
        return self._structure_factor

    @property
    def axes(self):
        return self._axes

    @property
    def rotations(self):
        return self._rotations

    @property
    def energy(self):
        return self._energy

    @property
    def centering(self):
        return self._centering

    @property
    def k_max(self):
        return self._k_max

    @property
    def sg_max(self):
        return self._sg_max

    @property
    def device(self):
        return self._device

    @property
    def ensemble_axes_metadata(self):
        ensemble_axes_metadata = []
        for axes, rotations in zip(self._axes, self.rotations):
            ensemble_axes_metadata += [
                NonLinearAxis(
                    label=f"{axes}-rotation", units="rad", values=rotations.values
                )
            ]
        return ensemble_axes_metadata

    @property
    def _ensemble_args(self):
        return tuple(
            i
            for i, rotation in enumerate(self._rotations)
            if hasattr(rotation, "__len__")
        )

    @property
    def ensemble_shape(self):
        return tuple(len(self._rotations[i]) for i in self._ensemble_args)

    def _partition_args(
        self,
        chunks: int | str | tuple[int | str | tuple[int, ...], ...] = None,
        lazy: bool = True,
    ):
        blocks = ()
        for i, n in zip(self._ensemble_args, chunks):
            blocks += (self._rotations[i].divide(n, lazy=lazy),)
        return blocks

    @property
    def _default_ensemble_chunks(self):
        return ("auto",) * len(self.ensemble_shape)

    @classmethod
    def _partial_transform(cls, *args, axes, order, num_ensemble_dims, **kwargs):

        args = unpack_blockwise_args(args)

        rotations = tuple(
            x for x, _ in sorted(zip(args, order), key=lambda pair: pair[1])
        )
        args = flatten_list_of_lists(zip(axes, rotations))

        new = cls(*args, **kwargs)
        new = _wrap_with_array(new, num_ensemble_dims)
        return new

    def _from_partitioned_args(self):
        non_ensemble_args = tuple(
            i for i in range(len(self._rotations)) if i not in self._ensemble_args
        )
        num_ensemble_dims = len(self._ensemble_args)
        order = non_ensemble_args + self._ensemble_args
        non_ensemble_args = tuple(self._rotations[i] for i in non_ensemble_args)
        kwargs = self._copy_kwargs()

        return partial(
            self._partial_transform,
            *non_ensemble_args,
            axes=self._axes,
            order=order,
            num_ensemble_dims=num_ensemble_dims,
            **kwargs,
        )

    def _calculate_diffraction_intensities(
        self,
        thicknesses: np.ndarray,
        return_complex: bool,
        pbar: bool,
        hkl_mask: np.ndarray = None,
    ):

        if hkl_mask is None:
            hkl_mask = self.get_ensemble_hkl_mask()

        orientation_matrices = self.get_orientation_matrices()

        shape = orientation_matrices.shape[:-2] + (
            len(thicknesses),
            hkl_mask.sum(),
        )

        pbar = TqdmWrapper(
            enabled=pbar,
            total=int(np.prod(orientation_matrices.shape[:-2])),
            leave=False,
        )

        xp = get_array_module(self.device)
        array = xp.zeros(shape, dtype=np.float32)

        # lil_matrix((np.prod(shape[:-1]), shape[-1]))

        for i in np.ndindex(orientation_matrices.shape[:-2]):

            bw = BlochWaves(
                structure_factor=self._structure_factor,
                energy=self.energy,
                sg_max=self.sg_max,
                k_max=self.k_max,
                orientation_matrix=orientation_matrices[i],
                centering=self.centering,
                device=self.device,
            )

            # cols = np.where(bw.hkl_mask)[0]
            # rows = np.ravel_multi_index(
            #     i + (tuple(range(shape[-2])),),
            #     dims=shape[:-1],
            # )

            diffraction_patterns = bw.calculate_diffraction_patterns(
                thicknesses, return_complex=return_complex
            )

            diffraction_patterns = diffraction_patterns.to_cpu()

            array[..., bw.hkl_mask[hkl_mask]] = diffraction_patterns.array

            # if threshold > 0.0:
            #     diffraction_patterns = diffraction_patterns.remove_low_intensity(
            #         threshold=threshold
            #     )

            # array[rows[:, None], cols] = diffraction_patterns.array

            pbar.update_if_exists(1)

        pbar.close_if_exists()

        return array

    def _lazy_calculate_diffraction_patterns(self, thicknesses, return_complex, pbar):
        blocks = self.ensemble_blocks(1)

        def _run_calculate_diffraction_patterns(
            block, hkl_mask, thicknesses, return_complex, pbar
        ):
            block = block.item()

            array = block._calculate_diffraction_intensities(
                thicknesses=thicknesses,
                return_complex=return_complex,
                pbar=pbar,
                hkl_mask=hkl_mask,
            )

            return array

        hkl_mask = self.get_ensemble_hkl_mask()

        shape = self.ensemble_shape + (
            len(thicknesses),
            int(hkl_mask.sum()),
        )

        out_ind = tuple(range(len(shape)))

        out = da.blockwise(
            _run_calculate_diffraction_patterns,
            out_ind,
            blocks,
            tuple(range(len(self.ensemble_shape))),
            da.from_array(hkl_mask),
            (-1,),
            thicknesses=thicknesses,
            return_complex=return_complex,
            new_axes={out_ind[-2]: shape[-2], out_ind[-1]: shape[-1]},
            pbar=pbar,
            concatenate=True,
            dtype=config.get("precision"),
        )
        return out, hkl_mask

    def calculate_diffraction_patterns(
        self,
        thicknesses,
        lazy: bool = True,
        return_complex: bool = False,
        pbar: bool = None,
    ):

        if pbar is None:
            pbar = config.get("local_diagnostics.task_level_progress", False)

        if lazy:
            array, hkl_mask = self._lazy_calculate_diffraction_patterns(
                thicknesses=thicknesses,
                return_complex=return_complex,
                pbar=pbar,
            )
        else:
            array = self._calculate_diffraction_intensities(
                thicknesses=thicknesses,
                return_complex=return_complex,
                pbar=pbar,
            )
            hkl_mask = self.get_ensemble_hkl_mask()

        orientation_matrices = self.get_orientation_matrices()
        hkl = self._structure_factor.hkl[hkl_mask]

        cells = np.matmul(
            self._structure_factor.atoms.cell.reciprocal()[None],
            np.swapaxes(orientation_matrices, -2, -1),
        )

        cells = cells[..., None, :, :]

        # cells = np.tile(cells[..., None, :, :], (len(thicknesses), 1, 1))

        result = IndexedDiffractionPatterns(
            array=array,
            miller_indices=hkl,
            reciprocal_lattice_vectors=cells,
            ensemble_axes_metadata=[
                *self.ensemble_axes_metadata,
                ThicknessAxis(label="z", units="Å", values=tuple(thicknesses)),
            ],
            metadata={
                "energy": self.energy,
                "sg_max": self.sg_max,
                "k_max": self.k_max,
            },
        )

        return result
