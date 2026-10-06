from __future__ import annotations

import itertools
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Sequence,
    SupportsFloat,
    TypeGuard,
    Union,
)

import dask.array as da
import numpy as np
from ase import Atoms
from ase.cell import Cell
from scipy.linalg import expm as expm_scipy  # type: ignore
from scipy.spatial.transform import Rotation  # type: ignore

from abtem.array import ArrayObject
from abtem.atoms import is_cell_orthogonal
from abtem.bloch.utils import (
    auto_detect_centering,
    calculate_g_vec,
    cell_bounds,
    excitation_errors,
    filter_reciprocal_space_vectors,
    get_reflection_condition,
    make_hkl_grid,
    reciprocal_cell,
    reciprocal_space_gpts,
    retrieve_structure_factor_values,
    validate_use_wave_eq,
)
from abtem.core import config
from abtem.core.axes import AxisMetadata, EnergyAxis, NonLinearAxis, ThicknessAxis
from abtem.core.backend import (
    asnumpy,
    cp,
    device_name_from_array_module,
    get_array_module,
    validate_device,
)
from abtem.core.chunks import Chunks, equal_sized_chunks, validate_chunks
from abtem.core.complex import abs2, complex_exponential
from abtem.core.constants import kappa
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.core.ensemble import Ensemble, _wrap_with_array, unpack_blockwise_args
from abtem.core.fft import fft_interpolate, warn_if_slow_gpu_fft
from abtem.core.grid import Grid
from abtem.core.utils import CopyMixin, get_dtype
from abtem.distributions import BaseDistribution, validate_distribution
from abtem.atoms import (
    AtomProperties,
    validate_per_atom_property,
    validate_sigmas,
)
from abtem.measurements import IndexedDiffractionPatterns
from abtem.parametrizations import Parametrization, validate_parametrization
from abtem.potentials.iam import PotentialArray

if cp is not None:
    from abtem.bloch.matrix_exponential import expm as expm_cupy

from abtem.waves import Waves

if TYPE_CHECKING:
    pass


class BlochWavePrecisionWarning(UserWarning):
    """Bloch waves are being computed in single precision.

    Filter it with ``warnings.filterwarnings("ignore",
    category=BlochWavePrecisionWarning)`` once a float32 result has been checked
    against a float64 one.
    """


# Messages already shown in this process. Python's own once-per-location
# registry cannot do this job: it is invalidated whenever the warning filters
# change, which abTEM (catch_warnings in ArrayObject) and dask do on every
# call, so the warning would repeat for every call, energy, orientation and
# dask block.
_issued_precision_warnings: set[str] = set()


def _warn_if_single_precision(device: str) -> None:
    # The eigendecomposition and the propagation phases carry an absolute error
    # of roughly 1e-7 to 3e-5 of the strongest beam in float32, growing with
    # thickness and beam count (Si and Au, 490-850 beams, 1000-20000 Å). Beams
    # above 1e-3 of the strongest agree with float64 to within about 2e-4
    # relative, but a weaker reflection can be off by a large fraction of
    # itself. The matrix exponential adds nothing: it is always taken in double.
    if np.dtype(get_dtype()) != np.float32:
        return

    if device_name_from_array_module(get_array_module(device)) == "mps":
        reason = "the Metal (MPS) device supports single precision only"
        check = "the 'cpu' or 'gpu' device with precision 'float64'"
    else:
        reason = "the 'precision' setting is 'float32'"
        check = "precision 'float64'"

    message = (
        f"Bloch waves are computed in single precision because {reason}. Beams "
        "down to 1e-3 of the strongest typically agree with double precision to "
        "within about 2e-4 relative, but weaker diffraction intensities may be "
        "inaccurate, increasingly so for thick samples and many beams "
        "(scattering matrices are exponentiated in double precision "
        f"regardless). Check the result against {check}, e.g. with "
        "abtem.config.set({'precision': 'float64'})."
    )
    if message in _issued_precision_warnings:
        return

    _issued_precision_warnings.add(message)
    warnings.warn(message, BlochWavePrecisionWarning, stacklevel=3)


def calculate_scattering_factors(
    g_vec: np.ndarray,
    atoms: Atoms,
    parametrization: str | Parametrization,
    g_max: float,
    thermal_sigma: AtomProperties = 0.0,
    occupancy: AtomProperties = 1.0,
    cutoff: str = "taper",
) -> np.ndarray:
    """Calculate the scattering factors for a given set of atoms and parametrization.

    Parameters
    ----------
    g_vec : numpy.ndarray
        Scattering vectors [1/Å]. Either Cartesian vectors with shape (N_g, 3), or
        plain magnitudes with shape (N_g,). Anisotropic Debye-Waller factors require
        shape (N_g, 3); passing magnitudes with anisotropic sigmas raises an error.
    atoms : Atoms
        Atoms object.
    g_max : float
        Maximum scattering vector length [1/Å]. The scattering factors are set to zero
        for g > g_max.
    parametrization : {'lobato', 'kirkland', 'peng'}
        Parametrization for the scattering factors.
    thermal_sigma : dict
        Standard deviation of the atomic displacements for the Debye-Waller factor [Å].
        For anisotropic displacements, provide three values per atom or element (σx, σy, σz).
    cutoff : {'taper', 'hard'}
        Cutoff function for the scattering factors. 'taper' is a smooth cutoff, 'hard'
        is a hard cutoff.
    """

    validated_thermal_sigma, anisotropic = validate_sigmas(
        atoms, thermal_sigma, return_array=True
    )
    validated_occupancy = validate_per_atom_property(
        atoms, occupancy, return_array=True
    )

    assert isinstance(validated_thermal_sigma, np.ndarray)  # Type narrowing for mypy
    assert isinstance(validated_occupancy, np.ndarray)  # Type narrowing for mypy

    parametrization = validate_parametrization(parametrization)

    if g_vec.ndim == 1:
        if anisotropic:
            raise ValueError(
                "Anisotropic Debye-Waller factors require Cartesian g-vectors "
                "with shape (N_g, 3), not plain magnitudes."
            )
        g = g_vec
        g_vec_3d = None
    else:
        g = np.linalg.norm(g_vec, axis=1)
        g_vec_3d = g_vec

    Z_unique = np.unique(atoms.numbers)

    scattering_factors = {Z: parametrization.scattering_factor(Z) for Z in Z_unique}

    f_e = np.zeros((len(atoms), len(g)), dtype=get_dtype(complex=True))

    two_pi_sq = (2 * np.pi) ** 2

    for i in range(len(atoms)):
        Z = atoms.numbers[i]
        s = validated_thermal_sigma[i]
        o = validated_occupancy[i]

        if anisotropic:
            # s has shape (3,); g_vec_3d has shape (N_g, 3)
            # DWF = exp(-0.5 * (2π)² * Σ_α σ_α² gα²)
            if np.any(s != 0.0):
                DWF = np.exp(-0.5 * two_pi_sq * (g_vec_3d**2 @ s**2))
            else:
                DWF = 1.0
        else:
            if s != 0.0:
                DWF = np.exp(-0.5 * s**2 * g**2 * two_pi_sq)
            else:
                DWF = 1.0

        f_e[i] = scattering_factors[Z](g**2) * DWF * o

    if cutoff == "taper":
        T = 0.005
        alpha = 1 - 0.05
        cutoff_array = 1 / (1 + np.exp((g / g_max - alpha) / T))
    elif cutoff == "hard":
        cutoff_array = g <= g_max
    else:
        raise ValueError("cutoff must be 'taper' or 'hard'")

    f_e *= cutoff_array

    return f_e


def calculate_structure_factors(
    hkl: np.ndarray,
    atoms: Atoms,
    parametrization: str | Parametrization,
    g_max: float,
    thermal_sigma: AtomProperties = 0.0,
    occupancy: AtomProperties = 1.0,
    cutoff: str = "taper",
    device: str = "cpu",
) -> np.ndarray:
    """Calculate the structure factors for a given set of atoms and parametrization.

    Parameters
    ----------
    hkl : numpy.ndarray
        The reciprocal space vectors as Miller indices. Given as a (N, 3) array.
    atoms : Atoms
        The Atoms object.
    parametrization : {'lobato', 'kirkland', 'peng'}
        Parametrization for the scattering factors.
    g_max : float
        Maximum scattering vector length [1/Å]. The scattering factors are set to zero
        for g > g_max.
    thermal_sigma : float
        Standard deviation of the atomic displacements for the Debye-Waller factor [Å].
    cutoff : {'taper', 'hard'}
        Cutoff function for the scattering factors. 'taper' is a smooth cutoff, 'hard'
        is a hard cutoff.
    device : {'cpu', 'gpu'}
        Device to use for calculations. Can be 'cpu' or 'gpu'.

    Returns
    -------
    numpy.ndarray
        The structure factors.
    """

    new_cell = atoms.cell.copy().complete()
    positions = np.linalg.solve(new_cell.T, atoms.positions.T).T

    f_e = calculate_scattering_factors(
        g_vec=calculate_g_vec(hkl, atoms.cell),
        atoms=atoms,
        g_max=g_max,
        parametrization=parametrization,
        cutoff=cutoff,
        thermal_sigma=thermal_sigma,
        occupancy=occupancy,
    )

    xp = get_array_module(device)

    f_e = xp.asarray(f_e, dtype=get_dtype(complex=True))
    positions = xp.asarray(positions, dtype=get_dtype(complex=False))
    hkl = xp.asarray(hkl.T, get_dtype(complex=False))

    struct_factors = (
        xp.sum(
            f_e * xp.exp(2.0j * np.pi * positions @ hkl),
            axis=0,
        )
        # A Python float, not ASE's np.float64: under NEP 50 a NumPy scalar
        # widens a single-precision array to double.
        / float(atoms.cell.volume)
    )

    return struct_factors


def structure_factor_1d_to_3d(
    structure_factor: np.ndarray, hkl: np.ndarray, gpts: tuple[int, int, int]
) -> np.ndarray:
    """Convert 1D structure factors to 3D structure factors.

    Parameters
    ----------
    structure_factor : numpy.ndarray
        The structure factors as a 1D array.
    hkl : numpy.ndarray
        The reciprocal space vectors as Miller indices as a (N, 3) array. N must be the
        same as the length of the structure factor.
    gpts : tuple of ints
        The number of grid points in the 3D structure factor.

    Returns
    -------
    numpy.ndarray
        The 3D structure factors.
    """
    xp = get_array_module(structure_factor)
    structure_factor_3d = xp.zeros(gpts, dtype=structure_factor.dtype)
    structure_factor_3d[hkl[:, 0], hkl[:, 1], hkl[:, 2]] = structure_factor
    return structure_factor_3d


def structure_factor_to_potential(
    structure_factor: np.ndarray, hkl: np.ndarray, gpts: tuple[int, int, int]
) -> np.ndarray:
    """Calculate the potential from the structure factors.

    Parameters
    ----------
    structure_factor : numpy.ndarray
        The structure factors as a 1D array.
    hkl : numpy.ndarray
        The reciprocal space vectors as Miller indices as a (N, 3) array. N must be the
        same as the length of the structure factor.
    gpts : tuple of ints
        The number of grid points in the 3D structure factor.

    Returns
    -------
    numpy.ndarray
        The potential.
    """
    xp = get_array_module(structure_factor)
    structure_factor = structure_factor_1d_to_3d(structure_factor, hkl, gpts)
    # Deliberately xp.fft rather than abtem.core.fft.fftn: the FFTW backend
    # behind that wrapper only ever transforms the trailing two axes, so it
    # would silently turn this 3D transform into a 2D one on CPU. The slow-FFT
    # diagnostic is requested explicitly instead -- this grid follows from
    # g_max and the cell, so it is essentially never a fast length.
    #
    # calculate_structure_factors uses the standard crystallographic convention
    # F(g) = sum_j f_j exp(+2pi i g.r_j), i.e. V(r) = sum_g F(g) exp(-2pi i g.r)
    # is the correct Fourier synthesis -- exactly numpy's forward transform
    # (fftn), not the inverse. Unlike ifftn, fftn carries no built-in 1/N
    # normalization, so (unlike the previous ifftn-based version) the result
    # must not be rescaled by the number of grid points.
    warn_if_slow_gpu_fft(structure_factor, "fftn")
    potential = xp.fft.fftn(structure_factor)
    potential = potential / kappa
    potential -= potential.min()
    return potential.real


def equal_slice_thicknesses(
    num_gpts_z: int, slice_thickness: float, depth: float
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    dz = depth / num_gpts_z
    n_slices = int(np.ceil(depth / slice_thickness))
    n_per_slice = equal_sized_chunks(num_items=num_gpts_z, num_chunks=n_slices)
    slice_thicknesses = tuple(n * dz for n in n_per_slice)
    return slice_thicknesses, n_per_slice


def _snap_slice_thicknesses(
    slice_thicknesses: Sequence[float], num_gpts_z: int, depth: float
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Slice thicknesses given as a sequence, snapped to the z grid.

    Each slice boundary (the cumulative thickness) is moved to the nearest z
    grid plane, so the slices tile the cell exactly whenever the thicknesses
    add up to its depth (to within half a z sample). Returns the thicknesses
    actually used and the number of z grid points in each slice.
    """
    sampling_z = depth / num_gpts_z
    thicknesses = np.asarray([float(dz) for dz in slice_thicknesses])
    if abs(thicknesses.sum() - depth) > sampling_z / 2:
        raise ValueError(
            f"the slice thicknesses must add up to the cell depth, {depth:g} Å "
            f"(to within half the z sampling, {sampling_z / 2:.3g} Å); they add up "
            f"to {thicknesses.sum():g} Å"
        )
    boundaries = np.round(np.cumsum(thicknesses) / sampling_z).astype(int)
    boundaries[-1] = num_gpts_z
    chunks = np.diff(np.concatenate(([0], boundaries)))
    if chunks.min() < 1:
        raise ValueError(
            f"every slice must span at least one z grid point ({sampling_z:.3g} Å "
            "here); increase the thinnest slice thicknesses or `g_max`"
        )
    return tuple(float(n * sampling_z) for n in chunks), tuple(int(n) for n in chunks)


def slice_potential(
    potential_3d: np.ndarray,
    slice_chunks: tuple[int, ...],
    slice_thicknesses: tuple[float, ...],
    gpts: Optional[tuple[int, int]] = None,
    rollaxis: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    num_slices = len(slice_chunks)
    assert num_slices == len(slice_thicknesses)
    assert sum(slice_chunks) == potential_3d.shape[-1]

    if gpts is not None and gpts != potential_3d.shape[:2]:
        potential_3d = fft_interpolate(potential_3d, gpts + (potential_3d.shape[-1],))

    z_samplings = tuple(
        thickness / n for n, thickness in zip(slice_chunks, slice_thicknesses)
    )

    start = np.cumsum((0,) + slice_chunks)

    potential_sliced = np.stack(
        [
            np.sum(potential_3d[..., start:stop], axis=-1) * dz
            for start, stop, dz in zip(start[:-1], start[1:], z_samplings)
        ],
        axis=-1,
    )

    if rollaxis:
        potential_sliced = np.rollaxis(potential_sliced, -1)

    return potential_sliced


class BaseStructureFactor(metaclass=ABCMeta):
    def __init__(
        self,
        hkl: np.ndarray,
        g_max: float,
        centering: str,
        *args: Any,
        **kwargs: Any,
    ):
        self._centering = centering
        self._hkl = hkl
        self._g_max = g_max
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.hkl)

    @property
    @abstractmethod
    def device(self) -> str:
        pass

    @property
    def gpts(self) -> tuple[int, int, int]:
        """Number of reciprocal space grid points."""
        return reciprocal_space_gpts(self.cell, self.g_max)

    @property
    def hkl(self) -> np.ndarray:
        """The reciprocal space vectors as Miller indices."""
        return self._hkl

    @property
    @abstractmethod
    def cell(self) -> Cell:
        """The unit cell."""

    @property
    def g_vec(self) -> np.ndarray:
        """The reciprocal space vectors."""
        return self.hkl @ self.cell.reciprocal()

    @property
    def g_vec_length(self) -> np.ndarray:
        """The lengths of the reciprocal space vectors."""
        return np.linalg.norm(self.g_vec, axis=1)

    @property
    def g_max(self) -> float:
        """The maximum scattering vector length."""
        return self._g_max

    @property
    def centering(self) -> str:
        """The lattice centering."""
        return self._centering

    @abstractmethod
    def get_potential_3d(self) -> np.ndarray:
        """Calculate the 3D potential from the structure factors."""

    @abstractmethod
    def get_projected_potential(
        self,
        slice_thickness: Optional[float | Sequence[float]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
    ) -> PotentialArray:
        """Calculate the projected potential from the structure factors."""


class StructureFactor(BaseStructureFactor, CopyMixin):
    """The StructureFactors class calculates the structure factors for a given set of
    atoms and parametrization.

    Parameters
    ----------
    atoms : Atoms
        Atoms object.
    g_max : float
        Maximum scattering vector length [1/Å].
    parametrization : str
        Parametrization for the scattering factors.
    thermal_sigma : float or dict
        Standard deviation of the atomic displacements for the Debye-Waller factor [Å].
    occupancy : float
        The occupancy of the atoms.
    cutoff : {'taper', 'hard'}
        Cutoff function for the scattering factors. 'taper' is a smooth cutoff, 'hard'
        is a hard cutoff.
    device : {'cpu', 'gpu'}
        Device to use for calculations. Can be 'cpu' or 'gpu'.
    centering : {'auto', 'P', 'I', 'A', 'B', 'C', 'F'}
        Lattice centering.
    """

    def __init__(
        self,
        atoms: Atoms,
        g_max: float,
        parametrization: str = "lobato",
        thermal_sigma: float | dict[str, float] | Sequence[float] = 0.0,
        occupancy: float | dict[str, float] | Sequence[float] = 1.0,
        cutoff: str = "taper",
        device: Optional[str] = None,
        centering: str = "auto",
    ):
        self._atoms = atoms

        self._thermal_sigma = validate_sigmas(atoms, thermal_sigma)[0]

        self._occupancy = validate_per_atom_property(atoms, occupancy)

        if centering == "auto":
            centering = auto_detect_centering(atoms)

        self._centering = centering

        hkl = make_hkl_grid(atoms.cell, g_max)
        if self._centering.lower() != "p":
            hkl = hkl[get_reflection_condition(hkl, self._centering)]

        if cutoff not in ("taper", "hard"):
            raise ValueError("cutoff must be 'taper', 'hard'")

        self._cutoff = cutoff
        self._parametrization = validate_parametrization(parametrization)
        self._device = validate_device(device)

        super().__init__(hkl=hkl, g_max=g_max, centering=centering)

    @property
    def device(self) -> str:
        return self._device

    @property
    def atoms(self) -> Atoms:
        return self._atoms

    @property
    def g_max(self) -> float:
        return self._g_max

    @property
    def cell(self) -> Cell:
        return self.atoms.cell

    @property
    def parametrization(self) -> Parametrization:
        return self._parametrization

    @property
    def thermal_sigma(self) -> np.ndarray | dict[str, np.ndarray]:
        return self._thermal_sigma

    @property
    def occupancy(self) -> np.ndarray | dict[str, np.ndarray]:
        return self._occupancy

    def calculate_scattering_factors(self) -> np.ndarray:
        """Calculate the scattering factors for each atomic species in the structure."""
        return calculate_scattering_factors(
            g_vec=self.g_vec,
            atoms=self.atoms,
            parametrization=self._parametrization,
            g_max=self.g_max,
            thermal_sigma=self._thermal_sigma,
            cutoff=self._cutoff,
        )

    def build(self, lazy: bool = True) -> StructureFactorArray:
        """Calculate the structure factors to obtain a StructureFactorArray object.

        Parameters
        ----------
        lazy : bool
            If True, the calculation is done lazily using dask. If False, the
            calculation is done eagerly.

        Returns
        -------
        StructureFactorArray
            The structure factors.
        """
        hkl = self.hkl
        if lazy:
            xp = get_array_module(self._device)
            array = da.from_array(hkl, chunks=-1).map_blocks(
                calculate_structure_factors,
                atoms=self.atoms,
                parametrization=self.parametrization,
                thermal_sigma=self._thermal_sigma,
                occupancy=self._occupancy,
                g_max=self.g_max,
                cutoff=self._cutoff,
                device=self._device,
                drop_axis=1,
                meta=xp.array((), dtype=get_dtype(complex=True)),
            )
        else:
            array = calculate_structure_factors(
                hkl,
                self.atoms,
                parametrization=self._parametrization,
                thermal_sigma=self._thermal_sigma,
                occupancy=self.occupancy,
                g_max=self.g_max,
                cutoff=self._cutoff,
                device=self._device,
            )

        return StructureFactorArray(array, self.hkl, self.atoms.cell, self.g_max)

    def get_potential_3d(self, lazy: bool = True) -> np.ndarray:
        """Calculate the 3D potential from the structure factors.

        Parameters
        ----------
        lazy : bool
            If True, the calculation is done lazily using dask. If False, the
            calculation is done eagerly.

        Returns
        -------
        numpy.ndarray
            The 3D potential.
        """
        return self.build(lazy=lazy).get_potential_3d()

    def get_projected_potential(
        self,
        slice_thickness: Optional[float | Sequence[float]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        lazy: bool = True,
    ) -> PotentialArray:
        """Calculate the projected potential from the structure factors.

        Parameters
        ----------
        slice_thickness : float or sequence of floats
            The thickness of the slices.
        sampling : float or tuple of floats
            The sampling of the projected potential [Å].
        gpts : int or tuple of ints
            The grid points of the projected potential.
        lazy : bool
            If True, the calculation is done lazily using dask. If False, the
            calculation is done eagerly.

        Returns
        -------
        PotentialArray
            The projected potential.
        """
        return self.build(lazy=lazy).get_projected_potential(
            slice_thickness, sampling, gpts
        )


class StructureFactorArray(ArrayObject, BaseStructureFactor):
    """The StructureFactorArray class represents structure factors as an ArrayObject.

    Parameters
    ----------
    array : numpy.ndarray
        The structure factors as a 1D array.
    hkl : numpy.ndarray
        The reciprocal space vectors as Miller indices as a (N, 3) array. N must be the
        same as the length of the structure factor.
    cell : Cell
        The unit cell.
    g_max : float
        Maximum scattering vector length [1/Å].
    ensemble_axes_metadata : list of AxisMetadata
        Metadata for the ensemble axes.
    metadata : dict
        Metadata for the ArrayObject.
    """

    _base_dims = 1

    def __init__(
        self,
        array: np.ndarray,
        hkl: np.ndarray,
        cell: np.ndarray | Cell,
        g_max: float,
        centering: str = "P",
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        metadata: Optional[dict] = None,
    ):
        if not array.shape[-1] == len(hkl):
            raise ValueError(
                "The last dimension of the array must be the same length as the number",
                " of hkl vectors",
            )

        if isinstance(cell, np.ndarray):
            cell = Cell(cell)

        self._cell = cell

        super().__init__(
            hkl=hkl,
            g_max=g_max,
            centering=centering,
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    def cell(self) -> Cell:
        return self._cell

    @classmethod
    def from_array_and_metadata(
        cls: type[StructureFactorArray],
        array: np.ndarray | da.core.Array,
        axes_metadata: list[AxisMetadata],
        metadata: dict,
    ) -> StructureFactorArray:
        raise NotImplementedError

    @property
    def gpts(self) -> tuple[int, int, int]:
        """Number of reciprocal space grid points for 3D structure factors."""
        return reciprocal_space_gpts(self.cell, self.g_max)

    def to_dict(self) -> dict:
        """
        Convert the structure factors to a dictionary. The keys are the Miller indices
        and the values are the structure factors.
        """
        return {(h, k, l): value for (h, k, l), value in zip(self.hkl, self.array)}

    def to_3d_array(self) -> np.ndarray:
        """Convert the 1D structure factors to 3D structure factors.

        Returns
        -------
        numpy.ndarray
            The 3D structure factors.
        """
        if self.is_lazy:
            xp = get_array_module(self.array)
            array = da.map_blocks(
                structure_factor_1d_to_3d,
                self._lazy_array,
                da.from_array(self.hkl, chunks=-1),
                gpts=self.gpts,
                chunks=self.gpts,
                meta=xp.array((), dtype=self.array.dtype),
            )
        else:
            array = structure_factor_1d_to_3d(self._eager_array, self.hkl, self.gpts)
        return array

    def get_potential_3d(self) -> np.ndarray:
        """Calculate the 3D potential from the structure factors.

        Returns
        -------
        numpy.ndarray
            The 3D potential.
        """
        if self.is_lazy:
            xp = get_array_module(self.array)
            array = da.map_blocks(
                structure_factor_to_potential,
                self._lazy_array,
                da.from_array(self.hkl, chunks=-1),
                gpts=self.gpts,
                chunks=self.gpts,
                meta=xp.array((), dtype=get_dtype(complex=False)),
            )
        else:
            array = structure_factor_to_potential(
                self._eager_array, self.hkl, self.gpts
            )
        return array

    def get_projected_potential(
        self,
        slice_thickness: Optional[float | Sequence[float]] = 0.5,
        sampling: Optional[float | tuple[float, float]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        lazy: bool = True,
    ) -> PotentialArray:
        """Calculate the projected potential from the structure factors.

        Parameters
        ----------
        slice_thickness : float or sequence of floats
            The thickness of the slices.
        sampling : float or tuple of floats
            The sampling of the projected potential [Å].
        gpts : int or tuple of ints
            The grid points of the projected potential.
        lazy : bool
            If True, the calculation is done lazily using dask. If False, the
             calculation is done eagerly.

        Returns
        -------
        PotentialArray
            The projected potential.
        """
        if not is_cell_orthogonal(self.cell):
            raise NotImplementedError(
                "Converting structure factor to projected potential is not supported ",
                "for non-orthogonal or rotated cells",
            )

        extent = tuple(np.diag(self.cell)[:2])

        potential_3d = self.get_potential_3d()
        depth = float(np.array(self.cell)[2, 2])
        num_gpts_z = potential_3d.shape[-1]
        sampling_z = depth / num_gpts_z

        # the lateral grid, resolved once from gpts (int or pair) or sampling
        if gpts is None and sampling is None:
            validated_gpts = tuple(potential_3d.shape[:2])
        else:
            if isinstance(gpts, int):
                gpts = (gpts, gpts)
            grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
            validated_gpts = tuple(int(n) for n in grid._valid_gpts)

        if slice_thickness is None:
            slice_thickness = min(1.0, depth)

        if isinstance(slice_thickness, (float, int)):
            validated_slice_thickness, slice_chunks = equal_slice_thicknesses(
                num_gpts_z=num_gpts_z,
                slice_thickness=slice_thickness,
                depth=depth,
            )
        elif isinstance(slice_thickness, Sequence):
            validated_slice_thickness, slice_chunks = _snap_slice_thicknesses(
                slice_thickness, num_gpts_z=num_gpts_z, depth=depth
            )
        else:
            raise ValueError(
                "Invalid `slice_thickness` argument type, must be float or sequence ",
                "of floats",
            )

        if min(validated_slice_thickness) < sampling_z:
            raise RuntimeError(
                "the slice thickness cannot be smaller than the real-space sampling ",
                "increase `g_max` or the slice thickness",
            )

        if self.is_lazy:
            xp = get_array_module(potential_3d)
            potential_sliced = da.map_blocks(
                slice_potential,
                potential_3d,
                slice_chunks=slice_chunks,
                slice_thicknesses=validated_slice_thickness,
                gpts=validated_gpts,
                chunks=(len(slice_chunks),) + validated_gpts,
                meta=xp.array((), dtype=potential_3d.dtype),
            )
        else:
            potential_sliced = slice_potential(
                potential_3d,
                slice_chunks=slice_chunks,
                slice_thicknesses=validated_slice_thickness,
                gpts=validated_gpts,
            )

        sampling = (
            extent[0] / potential_sliced.shape[-2],
            extent[1] / potential_sliced.shape[-1],
        )

        potential_array = PotentialArray(
            potential_sliced,
            slice_thickness=tuple(validated_slice_thickness),
            sampling=sampling,
        )

        return potential_array


def calculate_M_matrix(
    hkl: np.ndarray, cell: np.ndarray | Cell, energy: float
) -> np.ndarray:
    """Calculate the M matrix for a given set of reciprocal space vectors.

    Parameters
    ----------
    hkl : numpy.ndarray
        The reciprocal space vectors as Miller indices. Given as a (N, 3) array.
    cell : Cell
        The unit cell.
    energy : float
        The energy of the electrons [eV].

    Returns
    -------
    numpy.ndarray
        The M matrix.
    """
    g = hkl @ reciprocal_cell(cell)
    k0 = 1 / energy2wavelength(energy)
    Mii = 1 / np.sqrt(1 + g[:, 2] / k0)
    return Mii


def _metric(
    hkl: np.ndarray,
    cell: np.ndarray | Cell,
    energy: float,
    use_wave_eq: bool | Literal["exact"],
) -> np.ndarray:
    """Diagonal of M, the symmetrizing metric of the Bloch-wave eigenproblem.

    The standard Bloch-wave equation (use_wave_eq=False; the Helmholtz equation
    with only gamma**2 dropped) is the generalized eigenproblem

        A C = 2 K gamma B C,   B = diag(1 + g_z / K),

    solved as the Hermitian problem (M A M) C' = 2 K gamma C' with
    M = B**(-1/2) (calculate_M_matrix) and C = M C'. The wave-equation forms
    (use_wave_eq=True or "exact") -- the paraxial and non-paraxial equations
    multislice solves -- have gamma with the constant coefficient 2 K instead:
    an ordinary Hermitian eigenproblem, M = 1.
    """
    if use_wave_eq:
        return np.ones(len(hkl))
    return calculate_M_matrix(hkl, cell, energy)


def calculate_structure_matrix(
    structure_factor: np.ndarray,
    hkl: np.ndarray,
    hkl_selected: np.ndarray,
    cell: Cell | np.ndarray,
    energy: float,
    gpts: tuple[int, int, int],
    use_wave_eq: bool | Literal["exact"] = "exact",
) -> np.ndarray:
    """Calculate the structure matrix for a given set of reciprocal space vectors.

    Parameters
    ----------
    structure_factor : numpy.ndarray
        The structure factors as a 1D array.
    hkl : numpy.ndarray
        The reciprocal space vectors as Miller indices corresponding to the structure
        factors. Given as a (N, 3) array.
    hkl_selected : numpy.ndarray
        The reciprocal space vectors as Miller indices for which the structure matrix is
        calculated. Given as a (N, 3) array.
    cell : Cell
        The unit cell.
    energy : float
        The energy of the electrons [eV].
    gpts : tuple of ints
        The number of grid points in the 3D structure factor.
    use_wave_eq : bool or 'exact', optional
        The form of the Bloch-wave equation. If 'exact' (default), the non-paraxial
        wave equation solved by multislice with the default exact propagator,
        ``FourierMultislice(order="exact")``; the most accurate form (converging to
        exact multislice needs an `sg_max` large enough to include the beams whose
        paraxial and exact excitation errors differ). If True, the paraxial wave
        equation, matching ``FourierMultislice(order=1)``. If False, the standard
        (textbook) Bloch-wave equation: the Helmholtz equation with the second
        z-derivative of the Bloch-wave amplitudes dropped, with excitation errors
        measured from the Ewald sphere. See
        :func:`abtem.bloch.utils.excitation_errors`.

    Returns
    -------
    numpy.ndarray
        The structure matrix.
    """
    xp = get_array_module(structure_factor)

    g = calculate_g_vec(hkl_selected, cell)
    Mii = _metric(hkl_selected, cell, energy, use_wave_eq)

    hkl_selected = np.asarray(hkl_selected)

    gmh = hkl_selected[None] - hkl_selected[:, None]
    gmh = gmh.reshape(-1, 3)

    A = retrieve_structure_factor_values(structure_factor, hkl, gmh, gpts)
    A = xp.asarray(A.reshape((len(hkl_selected),) * 2), dtype=get_dtype(complex=True))

    # structure_factor_dict = {
    #     (h, k, l): value for (h, k, l), value in zip(hkl, structure_factor)
    # }
    # A = np.array([structure_factor_dict[(h, k, l)] for h, k, l in gmh])
    # A = A.reshape((len(hkl_selected),) * 2)

    prefactor = energy2sigma(energy) / (kappa * energy2wavelength(energy) * np.pi)

    # The geometry is computed in double precision on the host and only then
    # cast to the working precision: a float64 Mii or diagonal would otherwise
    # widen a single-precision structure matrix to complex128.
    sg = excitation_errors(g, energy, use_wave_eq=use_wave_eq)
    # M (2 K diag(s_g) + U) M: the diagonal carries M**2, like the off-diagonal
    # M_i M_j
    diag = xp.asarray(
        2 * 1 / energy2wavelength(energy) * sg * Mii**2, dtype=get_dtype()
    )
    Mii = xp.asarray(Mii, dtype=get_dtype())

    A = A * prefactor * Mii[None] * Mii[:, None]

    xp.fill_diagonal(A, diag)
    return A


def plane_wave_coefficients(hkl: np.ndarray, xp) -> np.ndarray:
    array = np.all(hkl == [0, 0, 0], axis=1)
    array = xp.asarray(array, dtype=get_dtype(complex=True))
    return array


def calculate_dynamical_scattering(
    structure_matrix: np.ndarray,
    hkl: np.ndarray,
    cell: np.ndarray | Cell,
    energy: float,
    thicknesses: float | Iterable[float],
    use_wave_eq: bool | Literal["exact"] = "exact",
) -> np.ndarray:
    """Calculate the dynamical scattering given a structure matrix.

    Parameters
    ----------
    structure_matrix : numpy.ndarray
        The structure matrix as a (N, N) array.
    hkl : numpy.ndarray
        The reciprocal space vectors as Miller indices. Given as a (N, 3) array.
    cell : Cell
        The unit cell.
    energy : float
        The energy of the electrons [eV].
    thicknesses : sequence of floats
        The thicknesses of the sample [Å].
    use_wave_eq : bool or 'exact', optional
        The form of the Bloch-wave equation the structure matrix was built for
        (see :func:`calculate_structure_matrix`; default 'exact'); decides the
        metric used to map its eigenvectors back to beam amplitudes.

    Returns
    -------
    numpy.ndarray
        The dynamical scattering as a complex array with shape
        (len(thicknesses), len(hkl)).
    """

    xp = get_array_module(structure_matrix)

    thicknesses = np.asarray(thicknesses)

    Mii = xp.asarray(_metric(hkl, cell, energy, use_wave_eq), dtype=get_dtype())

    # eigenvectors C' of the symmetrized structure matrix M A M; orthonormal
    v, C = xp.linalg.eigh(structure_matrix)

    gamma = v * energy2wavelength(energy) / 2.0

    # The incident plane wave, C alpha = e_0 at z = 0, with C = M C' the
    # physical Bloch-wave coefficients: alpha = C'^H M^-1 e_0 = C'^H e_0, as
    # M_00 = 1 (g_z = 0 for the direct beam).
    initial = plane_wave_coefficients(hkl, xp)
    alpha = xp.conjugate(C.T) @ initial

    C = Mii[:, None] * C

    z = np.atleast_1d(thicknesses)
    # thicknesses in gamma's precision: float64 would widen a single-precision
    # result to complex128 (NEP 50)
    z = xp.asarray(z, dtype=gamma.dtype)
    phases = xp.exp(2.0j * xp.pi * z[None] * gamma[:, None])
    array = (C @ (phases * alpha[:, None])).T

    if not thicknesses.shape:
        return array[0]
    return array


def expm(A: np.ndarray) -> np.ndarray:
    """Calculate the matrix exponential of a given array.

    This is a device agnostic version of the scipy.linalg.expm function.

    Parameters
    ----------
    A : numpy.ndarray
        Input with last two dimensions are square.

    Returns
    -------
    numpy.ndarray
        The resulting matrix exponential with the same shape of A.
    """
    xp = get_array_module(A)

    if xp == cp:
        return expm_cupy(A)
    elif xp is np:
        return expm_scipy(A)
    else:
        # Metal: exponentiate on the host, in double precision, and hand the
        # result back in the device's complex64. Scaling and squaring breaks
        # down at single precision for the norms of order 10^3 that realistic
        # beam counts and thicknesses give (721 Si beams at 1000 Å: S off by
        # 2.5e-3 exponentiated in complex64, by 3.5e-4 -- the share of the
        # single-precision structure matrix -- in complex128), and torch's own
        # matrix_exp, which runs on the device, is single precision too.
        A = asnumpy(A)
        return xp.asarray(expm_scipy(A.astype(np.complex128)).astype(A.dtype))


def calculate_scattering_matrix(
    A: np.ndarray,
    hkl: np.ndarray,
    cell: np.ndarray | Cell,
    z: float,
    energy: float,
    method: str = "expm",
    use_wave_eq: bool | Literal["exact"] = "exact",
) -> np.ndarray:
    """Calculate the scattering matrix for a given set of reciprocal space vectors.

    Parameters
    ----------
    A : numpy.ndarray
        The structure matrix. The last two dimensions must be square.
    hkl : numpy.ndarray
        The reciprocal space vectors as Miller indices. Given as a (N, 3) array.
    cell : Cell
        The unit cell.
    z : float
        The thickness of the sample [Å].
    energy : float
        The energy of the electrons [eV].
    method : {'expm', 'decomposition'}
        The method to use for calculating the scattering matrix.
            ``expm`` :
                Use a matrix exponential.
            ``decomposition`` :
                Use a Hermitian matrix eigendecomposition.
    use_wave_eq : bool or 'exact', optional
        The form of the Bloch-wave equation the structure matrix was built for
        (see :func:`calculate_structure_matrix`; default 'exact'); decides the
        metric used to map the result back to beam amplitudes.

    Returns
    -------
    numpy.ndarray
        The scattering matrix.
    """
    xp = get_array_module(A)

    if method == "expm":
        # Bloch waves are accurate enough in single precision, but the matrix
        # exponential is the exception: scaling and squaring breaks down at
        # single precision for the norms of order 10^3 that realistic beam
        # counts and thicknesses give (721 Si beams at 1000 Å come out NaN in
        # complex64). The exponent is therefore formed and exponentiated in
        # double, and only the result follows the 'precision' setting. Metal
        # has no double precision; expm takes it to the host instead.
        if xp is np or xp == cp:
            A = A.astype(xp.complex128)
        S = expm(1.0j * xp.pi * float(z) * A * energy2wavelength(energy))
        S = S.astype(get_dtype(complex=True), copy=False)
    else:
        raise NotImplementedError("Only 'expm' method is implemented")

    # S = C exp(2 pi i gamma z) C^-1 with C = M C': M expm(...) M^-1
    Mii = _metric(hkl, cell, energy, use_wave_eq)
    M = xp.asarray(np.diag(Mii), dtype=get_dtype())
    M_inv = xp.asarray(np.diag(1 / Mii), dtype=get_dtype())

    S = xp.dot(M, xp.dot(S, M_inv))
    return S


def validate_g_max(
    g_max: Optional[float] = None,
    structure_factor: Optional[BaseStructureFactor] = None,
) -> float:
    """Check if the provided g_max is valid. If g_max is None, it is set to half the
    g_max of the structure factor.

    Parameters
    ----------
    g_max : float
        The maximum scattering vector length [1/Å].
    structure_factor : BaseStructureFactor
        The structure factor.

    Returns
    -------
    float
        The validated g_max.
    """
    if g_max is None:
        if structure_factor is None:
            raise ValueError(
                "g_max must be provided if structure_factor is not provided"
            )

        g_max = structure_factor.g_max / 2

    if structure_factor is not None and g_max > structure_factor.g_max / 2:
        warnings.warn(
            "provided g_max exceed half the g_max of the scattering factors, "
            "some couplings are not included"
        )

    return g_max


def exctinction_distances(
    structure_factor: np.ndarray, cell: Cell, energy: float
) -> np.ndarray:
    xp = get_array_module(structure_factor)
    V = cell.volume
    return np.pi * V / (xp.abs(structure_factor) * energy2wavelength(energy) + 1e-12)


def plane_wave_basis(
    g: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """
    Calculate a plane wave basis for a given set of reciprocal space vectors
    at a set of real space positions.

    Parameters
    ----------
    g : numpy.ndarray
        The reciprocal space vectors as an Nx3 array [1 / Å].
    x : numpy.ndarray
        The x positions as a 1D array [Å].
    y : numpy.ndarray
        The y positions as a 1D array [Å].
    z : numpy.ndarray
        The z positions as a 1D array [Å].

    Returns
    -------
    numpy.ndarray
        The plane wave basis at the given positions.
    """
    g = get_array_module(x).asarray(g)  # on the device of the positions
    plane_waves_x = complex_exponential(
        2 * np.pi * g[None, :, 0, None, None] * x[None, None, :, None]
    )
    plane_waves_y = complex_exponential(
        2 * np.pi * g[None, :, 1, None, None] * y[None, None, None, :]
    )
    plane_waves_z = complex_exponential(
        2 * np.pi * g[None, :, 2, None, None] * z[..., None, None, None]
    )
    plane_waves = plane_waves_x * plane_waves_y * plane_waves_z
    return plane_waves


def reduce_plane_wave_expansion(values, plane_waves):
    wave = values[..., None, None] * plane_waves
    wave = wave.sum(-3)
    return wave


def calculate_wave_functions(amplitudes, g_vec, extent, gpts, thicknesses):
    xp = get_array_module(amplitudes)
    g_vec = xp.asarray(g_vec, dtype=get_dtype())
    x = xp.asarray(
        np.linspace(0, extent[0], gpts[0], endpoint=False), dtype=get_dtype()
    )
    y = xp.asarray(
        np.linspace(0, extent[1], gpts[1], endpoint=False), dtype=get_dtype()
    )
    z = xp.asarray(thicknesses, dtype=get_dtype())

    basis = plane_wave_basis(g_vec, x, y, z)
    wave_functions = reduce_plane_wave_expansion(amplitudes, basis)
    return wave_functions


AllowedRotations = Union[BaseDistribution, np.ndarray, SupportsFloat]


def allowed_chars(s: str, allowed_chars: str) -> bool:
    """
    Check if the string `s` only contains characters from `allowed_chars`.

    Parameters
    ----------
    s : str
        The string to check.
    allowed_chars : str
        A string containing all allowed characters.

    Returns
    --------
    bool
        True if `s` only contains characters from `allowed_chars`, False otherwise.
    """
    return all(char in allowed_chars for char in s)


def is_valid_rotation_axes(
    args: tuple[str | AllowedRotations, ...],
) -> TypeGuard[tuple[str, ...]]:
    return all(isinstance(arg, str) and allowed_chars(arg, "xyz") for arg in args)


def is_valid_rotations(
    args: tuple[str | AllowedRotations, ...],
) -> TypeGuard[tuple[AllowedRotations, ...]]:
    return all(isinstance(arg, (BaseDistribution, np.ndarray, Number)) for arg in args)


def validate_rotations(
    args: tuple[str | AllowedRotations, ...],
) -> tuple[tuple[str, ...], tuple[AllowedRotations, ...]]:
    axes = args[::2]
    rotations = args[1::2]

    assert is_valid_rotation_axes(axes)
    assert is_valid_rotations(rotations)

    return axes, rotations


def is_rotations_ensemble(axes: str, rotations: AllowedRotations) -> bool:
    if isinstance(rotations, BaseDistribution):
        ensemble = True
    elif isinstance(rotations, Iterable):
        rotations = np.array(rotations)
        if rotations.ndim == 1 and len(axes) > 1:
            assert len(axes) == len(rotations)
            ensemble = False
        elif rotations.ndim == 1:
            ensemble = True
        elif rotations.ndim == 2:
            assert len(axes) == rotations.shape[1]
            ensemble = True
        else:
            raise ValueError(
                "The rotation must be given as a sequence of angles or a "
                "sequence of sequences of angles"
            )
    else:
        ensemble = False
    return ensemble


class BlochWaves:
    """The BlochWaves class represents a set of Bloch waves. It may be used to calculate
    the dynamical diffraction patterns.

    Parameters
    ----------
    structure_factor : StructureFactor
        The structure factor.
    energy : float or list of float
        Electron energy [eV]. A single float runs a standard single-energy
        calculation. A list or array of floats runs the calculation at each
        energy, using a union of the allowed reciprocal-space vectors across
        all energies; beams that are inactive at a given energy are set to
        zero. The output gains a leading :class:`.EnergyAxis` dimension.
    sg_max : float
        The maximum excitation error [1/Å].
    g_max : float
        The maximum scattering vector length [1/Å].
    orientation_matrix : numpy.ndarray
        An optional orientation matrix given as a (3, 3) array. If provided, the unit
        cell is rotated.
        Instead of providing an orientation matrix, the `.rotate` method can be used.
    centering : {'auto', 'P', 'I', 'A', 'B', 'C', 'F'}
        Lattice centering.
    device : {'cpu', 'gpu'}
        Device to use for calculations. Can be 'cpu' or 'gpu'.
    use_wave_eq : bool or 'exact', optional
        The form of the Bloch-wave equation. If 'exact' (default), the non-paraxial
        wave equation solved by multislice with the default exact propagator,
        ``FourierMultislice(order="exact")``; the most accurate form (converging to
        exact multislice needs an `sg_max` large enough to include the beams whose
        paraxial and exact excitation errors differ). If True, the paraxial wave
        equation, matching ``FourierMultislice(order=1)``. If False, the standard
        (textbook) Bloch-wave equation: the Helmholtz equation with the second
        z-derivative of the Bloch-wave amplitudes dropped, with excitation errors
        measured from the Ewald sphere. See
        :func:`abtem.bloch.utils.excitation_errors`.
    """

    def __init__(
        self,
        structure_factor: BaseStructureFactor | Atoms,
        energy: float | list | np.ndarray,
        sg_max: float,
        g_max: Optional[float] = None,
        orientation_matrix: Optional[np.ndarray] = None,
        centering: str = "auto",
        device: Optional[str] = None,
        use_wave_eq: bool | Literal["exact"] = "exact",
    ):
        if isinstance(structure_factor, Atoms):
            if g_max is None:
                raise ValueError("g_max must be provided if structure_factor is Atoms")

            structure_factor = StructureFactor(structure_factor, g_max=g_max * 2)

        cell = structure_factor.cell

        if orientation_matrix is not None:
            cell = Cell(np.dot(cell, orientation_matrix.T))

        g_max = validate_g_max(g_max, structure_factor)

        if centering.lower() == "auto":
            centering = structure_factor.centering

        self._structure_factor = structure_factor
        self._sg_max = sg_max
        self._g_max = g_max
        self._cell = cell
        self._centering = centering
        self._use_wave_eq = validate_use_wave_eq(use_wave_eq)
        self._device = validate_device(device)

        energies = np.atleast_1d(np.asarray(energy, dtype=float)).ravel()
        self._energy = float(energies[0])  # always scalar; .energy property is backward-compat
        self._energies = energies          # full array for multi-energy paths
        if len(energies) == 1:
            # Scalar path — unchanged behaviour
            self._hkl_mask = filter_reciprocal_space_vectors(
                hkl=structure_factor.hkl,
                cell=cell,
                energy=float(energies[0]),
                sg_max=sg_max,
                g_max=self._g_max,
                centering=centering,
            )
            self._energy_hkl_masks: np.ndarray | None = None
        else:
            # Compute per-energy masks, then take their union so all energies
            # share the same reciprocal-space basis (higher energy → more beams,
            # so the union equals the mask at the highest energy, but OR-ing is
            # more rigorous and mirrors BlochwaveEnsemble.get_ensemble_hkl_mask).
            per_energy = [
                filter_reciprocal_space_vectors(
                    hkl=structure_factor.hkl,
                    cell=cell,
                    energy=float(e),
                    sg_max=sg_max,
                    g_max=self._g_max,
                    centering=centering,
                )
                for e in energies
            ]
            union_mask = per_energy[0].copy()
            for m in per_energy[1:]:
                union_mask |= m
            self._hkl_mask = union_mask
            # Boolean submask within the union for each energy:
            # _energy_hkl_masks[i] is True at positions (in union_hkl) where
            # energy[i]'s beams are active. Zeros fill inactive positions.
            self._energy_hkl_masks = np.stack(
                [m[union_mask] for m in per_energy], axis=0
            )

    def _require_single_energy(self, method: str) -> None:
        # The structure and scattering matrices of different energies span
        # different beam sets (different sizes), so they have no common array
        # to stack; with several energies these methods used to answer silently
        # for the first one.
        if len(self._energies) > 1:
            energies = ", ".join(f"{e:g}" for e in self._energies)
            raise ValueError(
                f"BlochWaves.{method} is defined for a single energy, but this "
                f"BlochWaves has {len(self._energies)} ({energies} eV); select "
                "one with select_energy(energy)"
            )

    def select_energy(self, energy: float) -> "BlochWaves":
        """The Bloch waves at one of this object's energies.

        Parameters
        ----------
        energy : float
            One of the energies of this BlochWaves [eV].

        Returns
        -------
        BlochWaves
            Single-energy Bloch waves, with the beams selected for that energy:
            the same as constructing BlochWaves with that energy alone.
        """
        matches = np.flatnonzero(np.isclose(self._energies, float(energy)))
        if len(matches) == 0:
            energies = ", ".join(f"{e:g}" for e in self._energies)
            raise ValueError(f"energy {energy:g} eV is not one of {energies} eV")
        if len(self._energies) == 1:
            return self
        idx = int(matches[0])
        return self._with_energy(idx, float(self._energies[idx]))

    def _with_energy(self, idx: int, e: float) -> "BlochWaves":
        """Return a single-energy clone using only the beams valid at energy *e*.

        The clone's ``_hkl_mask`` is narrowed to the per-energy subset of the
        union mask so that the structure-matrix and dynamical-scattering
        calculation work only on the active beams.  The caller embeds the
        result back into the union-sized output array using
        ``self._energy_hkl_masks[idx]``.
        """
        clone = object.__new__(BlochWaves)
        clone.__dict__.update(self.__dict__)  # shallow copy all attrs
        clone._energy = float(e)
        clone._energies = np.array([float(e)])
        # Translate _energy_hkl_masks[idx] (boolean over union_hkl) back to a
        # boolean mask over the full structure_factor.hkl index space.
        e_mask = np.zeros(len(self._hkl_mask), dtype=bool)
        e_mask[self._hkl_mask] = self._energy_hkl_masks[idx]
        clone._hkl_mask = e_mask
        clone._energy_hkl_masks = None  # scalar clone — no further splitting
        return clone

    @property
    def device(self) -> str:
        return self.structure_factor.device

    def __len__(self) -> int:
        return int(np.sum(self.hkl_mask))

    @property
    def hkl_mask(self) -> np.ndarray:
        return self._hkl_mask

    @property
    def hkl(self) -> np.ndarray:
        return self.structure_factor.hkl[self.hkl_mask]

    @property
    def g_vec(self) -> np.ndarray:
        return self.hkl @ self._cell.reciprocal()

    @property
    def g_vec_length(self) -> np.ndarray:
        return np.linalg.norm(self.g_vec, axis=1)

    @property
    def use_wave_eq(self) -> bool | Literal["exact"]:
        return self._use_wave_eq

    @property
    def cell(self) -> Cell:
        return self._cell

    @property
    def g_max(self) -> float:
        return self._g_max

    @property
    def sg_max(self) -> float:
        return self._sg_max

    @property
    def structure_factor(self) -> BaseStructureFactor:
        return self._structure_factor

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def num_bloch_waves(self) -> int:
        """The number of Bloch waves used."""
        return int(np.sum(self.hkl_mask))

    @property
    def wavelength(self) -> float:
        """The wavelength of the electrons [Å]."""
        return energy2wavelength(self.energy)

    def excitation_errors(self) -> np.ndarray:
        """Excitation errors for the Bloch waves [1/Å], in the form set by
        `use_wave_eq`.

        With several energies, an array of shape (energies, beams) over the
        union of the energies' beam sets (``hkl``); otherwise shape (beams,).
        """
        use_wave_eq = self.use_wave_eq
        if len(self._energies) > 1:
            # one row per energy, over the union of the energies' beams
            return np.stack(
                [
                    excitation_errors(self.g_vec, e, use_wave_eq=use_wave_eq)
                    for e in self._energies
                ]
            )
        return excitation_errors(self.g_vec, self.energy, use_wave_eq=use_wave_eq)

    @property
    def structure_matrix_nbytes(self) -> int:
        """The number of bytes used by the structure matrix."""
        bytes_per_element = np.dtype(get_dtype(complex=True)).itemsize
        return self.num_bloch_waves**2 * bytes_per_element

    def _get_structure_factor_array(self, lazy: bool = False) -> StructureFactorArray:
        if isinstance(self.structure_factor, StructureFactor):
            return self.structure_factor.build(lazy=lazy)
        elif isinstance(self.structure_factor, StructureFactorArray):
            return self.structure_factor
        else:
            raise ValueError(
                "structure_factor must be a StructureFactor or StructureFactorArray"
            )

    def get_kinematical_diffraction_pattern(
        self, excitation_error_sigma: Optional[float] = None
    ) -> IndexedDiffractionPatterns:
        """Calculate the kinematical diffraction pattern.

        Parameters
        ----------
        excitation_error_sigma : float
            The standard deviation of the excitation errors used for weigting the
            structure factor intensities [1/Å].

        Returns
        -------
        IndexedDiffractionPatterns
            The kinematical diffraction pattern.
        """
        if len(self._energies) > 1:
            return self._multi_energy_kinematical_diffraction_pattern(
                excitation_error_sigma
            )
        hkl = self.hkl

        structure_factor = self._get_structure_factor_array()

        S_array = structure_factor.array[self.hkl_mask]
        sg = self.excitation_errors()

        S_array = abs2(S_array)

        if excitation_error_sigma is None:
            excitation_error_sigma = self._sg_max / 3.0

        xp = get_array_module(S_array)
        sg = xp.asarray(sg)
        intensity = S_array * xp.exp(-(sg**2) / (2.0 * excitation_error_sigma**2))

        metadata = {"energy": self.energy, "sg_max": self._sg_max, "g_max": self.g_max}

        reciprocal_lattice_vectors = reciprocal_cell(self.cell)

        return IndexedDiffractionPatterns(
            miller_indices=hkl,
            array=intensity,
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
            metadata=metadata,
        )

    def _multi_energy_kinematical_diffraction_pattern(
        self, excitation_error_sigma: Optional[float]
    ) -> IndexedDiffractionPatterns:
        # each energy on its own beams, embedded into the union beam set (zero
        # where an energy does not include a beam), as for the dynamical
        # calculate_diffraction_patterns
        n_union = int(self._hkl_mask.sum())
        members = []
        for i, e in enumerate(self._energies):
            pattern = self._with_energy(i, float(e)).get_kinematical_diffraction_pattern(
                excitation_error_sigma
            )
            xp = get_array_module(pattern.array)
            padded = xp.zeros((n_union,), dtype=pattern.array.dtype)
            padded[self._energy_hkl_masks[i]] = pattern.array
            members.append(padded)
        xp = get_array_module(members[0])
        return IndexedDiffractionPatterns(
            miller_indices=self.hkl,
            array=xp.stack(members),
            reciprocal_lattice_vectors=reciprocal_cell(self.cell),
            ensemble_axes_metadata=[
                EnergyAxis(values=tuple(float(e) for e in self._energies))
            ],
            metadata={
                "energy": list(self._energies),
                "sg_max": self._sg_max,
                "g_max": self.g_max,
            },
        )

    def calculate_structure_matrix(self, lazy: bool = True) -> np.ndarray:
        """Calculate the structure matrix.

        Parameters
        ----------
        lazy : bool
            If True, the calculation is done lazily using dask. If False, the
            calculation is done eagerly.
        """
        self._require_single_energy("calculate_structure_matrix")
        hkl = self.hkl

        structure_factor = self._get_structure_factor_array(lazy=lazy)

        if lazy:
            xp = get_array_module(self._device)
            A = da.map_blocks(
                calculate_structure_matrix,
                structure_factor._lazy_array,
                hkl=structure_factor.hkl,
                hkl_selected=hkl,
                cell=self.cell,
                energy=self.energy,
                use_wave_eq=self.use_wave_eq,
                gpts=structure_factor.gpts,
                new_axis=1,
                chunks=(len(hkl), len(hkl)),
                meta=xp.array((), dtype=get_dtype(complex=True)),
            )
        else:
            A = calculate_structure_matrix(
                structure_factor=structure_factor._eager_array,
                hkl=structure_factor.hkl,
                hkl_selected=hkl,
                cell=self.cell,
                energy=self.energy,
                use_wave_eq=self.use_wave_eq,
                gpts=structure_factor.gpts,
            )
        return A

    def calculate_scattering_matrix(self, z: float) -> np.ndarray:
        """Calculate the scattering matrix for a given thickness.

        Parameters
        ----------
        z : float
            The thickness of the sample [Å].

        Returns
        -------
        numpy.ndarray
            The scattering matrix.
        """
        self._require_single_energy("calculate_scattering_matrix")
        _warn_if_single_precision(self._device)
        # Eager: the result feeds xp.asarray, and CuPy refuses to convert a
        # dask array implicitly (Metal and NumPy happen to accept one).
        A = self.calculate_structure_matrix(lazy=False)
        hkl = self.hkl
        cell = self.cell

        xp = get_array_module(self._device)
        A = xp.asarray(A)

        S = calculate_scattering_matrix(
            A=A,
            hkl=hkl,
            cell=cell,
            z=z,
            energy=self.energy,
            use_wave_eq=self.use_wave_eq,
        )
        return S

    def _calculate_array(
        self, thicknesses: np.ndarray, lazy: bool = True
    ) -> np.ndarray | da.core.Array:
        assert isinstance(thicknesses, np.ndarray)
        _warn_if_single_precision(self._device)
        hkl = self.hkl

        A = self.calculate_structure_matrix(lazy=lazy)

        if lazy:
            xp = get_array_module(self._device)

            chunks: tuple[int, ...]
            if not thicknesses.shape:
                chunks = (len(hkl),)
            else:
                chunks = (len(thicknesses), len(hkl))

            array = da.map_blocks(
                calculate_dynamical_scattering,
                A,
                hkl=hkl,
                cell=self.cell,
                energy=self.energy,
                thicknesses=thicknesses,
                use_wave_eq=self.use_wave_eq,
                drop_axis=1,
                chunks=chunks,
                meta=xp.array((), dtype=get_dtype(complex=True)),
            )
        else:
            array = calculate_dynamical_scattering(
                structure_matrix=A,
                hkl=hkl,
                cell=self.cell,
                energy=self.energy,
                thicknesses=thicknesses,
                use_wave_eq=self.use_wave_eq,
            )

        return array

    def calculate_diffraction_patterns(
        self,
        thicknesses: float | Sequence[float] | np.ndarray,
        return_complex: bool = False,
        lazy: bool = True,
    ) -> IndexedDiffractionPatterns:
        """Calculate the dynamical diffraction patterns for a given set of thicknesses.

        Parameters
        ----------
        thicknesses : float or sequence of floats
            The thicknesses of the sample [Å].
        return_complex : bool
            If True, the complex diffraction patterns are returned. If False,
            the intensity is returned. Default is False.
        lazy : bool
            If True, the calculation is done lazily using dask. If False,
            the calculation is done eagerly.

        Returns
        -------
        IndexedDiffractionPatterns
            The dynamical diffraction patterns.
        """
        # --- Multi-energy ensemble path ---
        if len(self._energies) > 1:
            energies = self._energies
            n_union = int(self._hkl_mask.sum())

            def _embed_beams(arr, active_mask, n_total):
                """Embed (..., n_active) array into (..., n_total) with zeros."""
                # arr is a GPU (cupy) array whenever this ensemble runs on
                # device="gpu" -- both here (the lazy=False, eager path) and
                # per-block inside the map_blocks call below (the lazy path,
                # where the block itself is cupy-backed). A bare np.zeros(...)
                # always allocates on host, and cupy refuses the implicit
                # device->host copy that assigning it into a numpy array's
                # boolean-masked slice would require, raising a TypeError
                # instead of doing the transfer silently. Allocate `out` on
                # whichever device `arr` is actually on.
                xp = get_array_module(arr)
                out = xp.zeros(arr.shape[:-1] + (n_total,), dtype=arr.dtype)
                out[..., active_mask] = arr
                return out

            padded_arrays = []
            first_result = None
            for i, e in enumerate(energies):
                clone = self._with_energy(i, float(e))
                res = clone.calculate_diffraction_patterns(
                    thicknesses, return_complex=return_complex, lazy=lazy
                )
                if first_result is None:
                    first_result = res
                active = self._energy_hkl_masks[i]
                if lazy:
                    new_chunks = res.array.chunks[:-1] + ((n_union,),)
                    padded = res.array.map_blocks(
                        _embed_beams,
                        active_mask=active,
                        n_total=n_union,
                        dtype=res.array.dtype,
                        chunks=new_chunks,
                    )
                else:
                    # lazy=False -- res.array is a plain numpy/cupy array with
                    # no .chunks to preserve, so pad it directly instead of
                    # going through the dask-only map_blocks path above.
                    padded = _embed_beams(res.array, active, n_union)
                padded_arrays.append(padded)

            if lazy:
                stacked = da.stack(padded_arrays, axis=0)
            else:
                # Same device concern as _embed_beams above: np.stack on a
                # list of cupy arrays happens to work today via cupy's
                # __array_function__ dispatch, but that's an implementation
                # detail of cupy's NEP-18 support, not something this file
                # should depend on implicitly elsewhere. Stack on whichever
                # device the padded arrays are actually on.
                xp = get_array_module(padded_arrays[0])
                stacked = xp.stack(padded_arrays, axis=0)
            energy_ax = EnergyAxis(values=tuple(float(e) for e in energies))
            rlv = first_result.reciprocal_lattice_vectors
            if rlv.ndim == 3:
                rlv = rlv[0]
            return IndexedDiffractionPatterns(
                miller_indices=self.hkl,  # union hkl
                array=stacked,
                reciprocal_lattice_vectors=rlv,
                ensemble_axes_metadata=[energy_ax]
                + first_result.ensemble_axes_metadata,
                metadata={
                    "energy": list(energies),
                    "sg_max": self.sg_max,
                    "g_max": self.g_max,
                    "label": "Intensity",
                    "units": "arb. unit",
                },
            )

        # --- Single-energy path (unchanged) ---
        ensemble_axes_metadata: list[AxisMetadata]
        if isinstance(thicknesses, (int, float)):
            ensemble_axes_metadata = []
        else:
            ensemble_axes_metadata = [
                ThicknessAxis(label="z", units="Å", values=tuple(thicknesses))
            ]

        thicknesses = np.array(thicknesses, dtype=get_dtype())
        array = self._calculate_array(thicknesses, lazy=lazy)

        reciprocal_lattice_vectors = reciprocal_cell(self.cell)

        if len(ensemble_axes_metadata) > 0:
            reciprocal_lattice_vectors = reciprocal_lattice_vectors[None]

        if not return_complex:
            array = abs2(array)

        return IndexedDiffractionPatterns(
            miller_indices=self.hkl,
            array=array,
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata={
                "energy": self.energy,
                "sg_max": self.sg_max,
                "g_max": self.g_max,
                "label": "Intensity",
                "units": "arb. unit",
            },
        )

    @staticmethod
    def _calculate_exit_waves(amplitudes, g_vec, x, y, z):
        xp = get_array_module(amplitudes)
        g_vec = xp.asarray(g_vec, dtype=get_dtype())
        x = xp.asarray(x, dtype=get_dtype())
        y = xp.asarray(y, dtype=get_dtype())
        z = xp.asarray(z, dtype=get_dtype())

        basis = plane_wave_basis(g_vec, x, y, z)

        if not z.ndim:
            basis = basis[0]

        wave_functions = reduce_plane_wave_expansion(amplitudes, basis)
        return wave_functions

    def calculate_exit_waves(
        self,
        thicknesses: float | Iterable[float],
        gpts: Optional[tuple[int, int]] = None,
        extent: Optional[tuple[float, float]] = None,
        normalization: str = "values",
        g_max: Optional[float] = None,
        lazy: bool = True,
    ) -> Waves:
        """Calculate the exit waves for a given set of thicknesses.

        Parameters
        ----------
        thicknesses : float or sequence of floats
            The thicknesses of the sample [Å].
        gpts : tuple of ints
            The grid points of the exit waves.
        extent : tuple of floats
            The extent of the exit waves [Å].
        normalization : {'values', 'amplitude'}
            The normalization of the exit waves. If 'values', the exit waves are
        lazy : bool
            If True, the calculation is done lazily using dask. If False, the
            calculation is done eagerly.

        Returns
        -------
        Waves
            The exit waves.
        """

        # --- Multi-energy ensemble path ---
        if len(self._energies) > 1:
            energies = self._energies
            results = [
                self._with_energy(i, float(e)).calculate_exit_waves(
                    thicknesses,
                    gpts=gpts,
                    extent=extent,
                    normalization=normalization,
                    g_max=g_max,
                    lazy=lazy,
                )
                for i, e in enumerate(energies)
            ]
            arrays = [r.array for r in results]
            if lazy:
                stacked = da.stack(arrays, axis=0)
            else:
                # da.stack would wrap the eager per-energy arrays in dask
                stacked = get_array_module(arrays[0]).stack(arrays, axis=0)
            energy_ax = EnergyAxis(values=tuple(float(e) for e in energies))
            return Waves(
                array=stacked,
                extent=results[0].extent,
                energy=None,
                ensemble_axes_metadata=[energy_ax]
                + results[0].ensemble_axes_metadata,
                metadata=results[0].metadata,
            )

        # --- Single-energy path (unchanged) ---
        if extent is None:
            extent = tuple(cell_bounds(self.cell)[:2])

        if gpts is None:
            sampling = (1 / self.g_max / 2, 1 / self.g_max / 2)
            gpts = (
                int(np.ceil(extent[0] / sampling[0])),
                int(np.ceil(extent[1] / sampling[1])),
            )

        xp = get_array_module(self.device)

        thicknesses = np.array(thicknesses)
        g_vec = self.g_vec
        hkl = self.hkl
        values = self._calculate_array(thicknesses, lazy=lazy)

        if g_max is not None:
            mask = self.g_vec_length < g_max
            g_vec = g_vec[mask]
            hkl = hkl[mask]
            values = values[..., mask]

        shape = values.shape + gpts
        chunks = values.shape + ("auto", "auto")
        chunks = validate_chunks(shape, chunks, dtype=values.dtype)

        if lazy:
            x = da.linspace(0, extent[0], gpts[0], endpoint=False, chunks=chunks[-2])
            y = da.linspace(0, extent[1], gpts[1], endpoint=False, chunks=chunks[-1])

            args: tuple[Any, ...]
            out_ind: tuple[int, ...]
            values_ind: tuple[int, ...]
            if not thicknesses.shape:
                args = ()
                kwargs = {"z": np.array(thicknesses)}
                out_ind = (3, 4)
                values_ind = (1,)
            else:
                args = (da.from_array(thicknesses, chunks=-1), (0,))
                kwargs = {}
                out_ind = (0, 3, 4)
                values_ind = (0, 1)

            array = da.blockwise(
                self._calculate_exit_waves,
                out_ind,
                values,
                values_ind,
                g_vec,
                (1, 2),
                x,
                (3,),
                y,
                (4,),
                *args,
                **kwargs,
                concatenate=True,
                meta=xp.array((), dtype=values.dtype),
            )
        else:
            array = calculate_wave_functions(values, g_vec, extent, gpts, thicknesses)

            if thicknesses.ndim == 0:
                # a scalar thickness has no ensemble (thickness) axis; drop the spurious
                # leading size-1 axis so the array matches the empty axis metadata (the
                # lazy path already returns a 2D array in this case).
                array = array[0]

        ensemble_axes_metadata: list[AxisMetadata] = []

        if isinstance(thicknesses, np.ndarray) and thicknesses.ndim > 0:
            ensemble_axes_metadata = [
                ThicknessAxis(label="z", units="Å", values=tuple(thicknesses))
            ]

        waves = Waves(
            array=array,
            extent=extent,
            energy=self.energy,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata={"normalization": normalization},
        )

        return waves

    #     xp = get_array_module(array)

    #     thicknesses1 = xp.asarray(thicknesses)
    #     array2 = xp.zeros(array.shape[:-1] + gpts, dtype=array.dtype)
    #     for i, nmi in enumerate(nm):
    #         phase = xp.exp(-2 * np.pi * 1.0j * g_vec[i, 2] * thicknesses1)
    #         array2[..., nmi[0], nmi[1]] += array[..., i] * phase

    #     array = ifft2(xp.fft.ifftshift(array2, axes=(-2, -1)))

    #     if normalization == "values":
    #         array *= np.prod(gpts)

    #     waves = Waves(
    #         array=array,
    #         extent=extent,
    #         energy=self.energy,
    #         ensemble_axes_metadata=[
    #             ThicknessAxis(label="z", units="Å", values=tuple(thicknesses))
    #         ],
    #         metadata={"normalization": "values"},
    #     )

    #     return waves

    def rotate(
        self,
        *args: str | BaseDistribution | np.ndarray | SupportsFloat,
        degrees: bool = False,
    ) -> BlochWaves | BlochwaveEnsemble:
        """Rotate the unit cell by a given set of Euler angles.

        Parameters
        ----------
        args : sequence of (str, float)
            The rotation axes and angles. The axes must be given as a string of 'x',
            'y' or 'z', representing a sequence of rotation axes.
        degrees : bool
            If True, the angles are given in degrees. Default is False.

        Returns
        -------
        BlochWaves
            The rotated Bloch waves.
        BlochWavesEnsemble
            The rotated Bloch waves ensemble.
        """

        all_axes, all_rotations = validate_rotations(args)

        bloch_waves: BlochWaves | BlochwaveEnsemble

        if any(
            is_rotations_ensemble(axes, rotations)
            for axes, rotations in zip(all_axes, all_rotations)
        ):
            bloch_waves = BlochwaveEnsemble(
                *args,
                structure_factor=self.structure_factor,
                energy=self.energy,
                sg_max=self.sg_max,
                g_max=self.g_max,
                centering=self._centering,
                use_wave_eq=self.use_wave_eq,
                device=self._device,
                use_degrees=degrees,
            )
        else:
            orientation_matrix = np.eye(3)

            for axes, rotation in zip(all_axes, all_rotations):
                R = Rotation.from_euler(axes, rotation, degrees=degrees).as_matrix()
                orientation_matrix = R @ orientation_matrix

            bloch_waves = BlochWaves(
                structure_factor=self.structure_factor,
                energy=self.energy,
                sg_max=self.sg_max,
                g_max=self.g_max,
                centering=self._centering,
                orientation_matrix=orientation_matrix,
                use_wave_eq=self.use_wave_eq,
                device=self._device,
            )

        return bloch_waves


def is_base_distribution_tuple(
    rotations: tuple[BaseDistribution | np.ndarray | float, ...],
) -> TypeGuard[tuple[BaseDistribution, ...]]:
    return all(isinstance(rotation, BaseDistribution) for rotation in rotations)


class BlochwaveEnsemble(Ensemble, CopyMixin):
    def __init__(
        self,
        *args: str | BaseDistribution | np.ndarray | SupportsFloat,
        structure_factor: BaseStructureFactor,
        energy: float,
        sg_max: float,
        g_max: float,
        centering: str = "P",
        device: Optional[str] = None,
        use_wave_eq: bool | Literal["exact"] = "exact",
        use_degrees: bool = False,
    ):
        axes = args[::2]
        if not is_valid_rotation_axes(axes):
            raise ValueError("The axes must be given as a tuple of strings")

        self._axes: tuple[str, ...] = axes

        rotations = args[1::2]

        assert is_valid_rotations(rotations)

        validated_rotations = tuple(
            validate_distribution(rotation) for rotation in rotations
        )

        if not all(
            isinstance(rotation, (BaseDistribution, Number))
            for rotation in validated_rotations
        ):
            raise ValueError(
                "The rotations must be given as a tuple of BaseDistribution, sequence "
                "of angles or single angles"
            )

        self._rotations = validated_rotations

        self._use_degrees = use_degrees
        self._structure_factor = structure_factor
        self._energy = energy
        self._centering = centering
        self._sg_max = sg_max
        self._g_max = g_max
        self._use_wave_eq = validate_use_wave_eq(use_wave_eq)
        self._device = validate_device(device)

    def get_ensemble_hkl_mask(self) -> np.ndarray:
        """Get the mask selecting all the reciprocal space vectors included in the
        ensemble.

        Returns
        -------
        numpy.ndarray
            The mask selecting the reciprocal space vectors.
        """
        hkl = self._structure_factor.hkl
        mask = filter_reciprocal_space_vectors(
            hkl=hkl,
            cell=self._structure_factor.cell,
            energy=self.energy,
            sg_max=self.sg_max,
            g_max=self.g_max,
            centering=self.centering,
            orientation_matrices=self.get_orientation_matrices().reshape(-1, 3, 3),
        )
        return mask

    def get_orientation_matrices(self) -> np.ndarray:
        """Get the orientation matrices for the ensemble.

        Returns
        -------
        numpy.ndarray
            The orientation matrices. The shape is the ensemble shape + (3, 3).
        """
        orientation_matrices = np.eye(3)
        for axes, rotation in zip(self.axes[::-1], self.rotations[::-1]):
            if hasattr(rotation, "values"):
                # SciPy >= 1.18 requires an explicit (N, len(axes)) shape, even for
                # a single-axis sequence.
                values = np.asarray(rotation.values, dtype=float).reshape(-1, len(axes))
                R = Rotation.from_euler(
                    axes, values, degrees=self._use_degrees
                ).as_matrix()
                R = R[(slice(None),) + (None,) * (orientation_matrices.ndim - 2)]
            else:
                R = Rotation.from_euler(
                    axes, rotation, degrees=self._use_degrees
                ).as_matrix()

            orientation_matrices = orientation_matrices @ R

        return orientation_matrices

    @property
    def structure_factor(self) -> BaseStructureFactor:
        return self._structure_factor

    @property
    def axes(self) -> Sequence[str]:
        return self._axes

    @property
    def rotations(self) -> tuple[BaseDistribution | Number, ...]:
        return self._rotations

    @property
    def use_degrees(self) -> bool:
        return self._use_degrees

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def centering(self) -> str:
        return self._centering

    @property
    def g_max(self) -> float:
        return self._g_max

    @property
    def use_wave_eq(self) -> bool | Literal["exact"]:
        return self._use_wave_eq

    @property
    def sg_max(self) -> float:
        return self._sg_max

    @property
    def device(self) -> str:
        return self._device

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        if self.use_degrees:
            units = "deg"
        else:
            units = "rad"

        ensemble_axes_metadata: list[AxisMetadata] = []
        for axes, rotations in zip(self._axes, self.rotations):
            if isinstance(rotations, BaseDistribution):
                if len(axes) == 1:
                    ensemble_axes_metadata.append(
                        NonLinearAxis(
                            label=f"{axes}_rotation",
                            units=units,
                            values=tuple(rotations.values),
                            tex_label=f"${axes}_{{rotation}}$",
                        )
                    )
                else:
                    ensemble_axes_metadata.append(
                        NonLinearAxis(
                            label=f"{axes}_rotation",
                            values=tuple(tuple(value) for value in rotations.values),
                            units=units,
                            tex_label=f"${axes}_{{rotation}}$",
                        )
                    )

        return ensemble_axes_metadata

    @property
    def _ensemble_args(self) -> tuple[int, ...]:
        args = tuple(
            i
            for i, rotation in enumerate(self._rotations)
            if hasattr(rotation, "__len__")
        )
        return args

    @property
    def _ensemble_rotations(self) -> tuple[BaseDistribution, ...]:
        rotations = tuple(self._rotations[i] for i in self._ensemble_args)
        if is_base_distribution_tuple(rotations):
            return rotations
        else:
            raise RuntimeError("All ensemble rotations must be BaseDistribution")

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return tuple(len(rotation) for rotation in self._ensemble_rotations)

    def _partition_args(
        self,
        chunks: Optional[Chunks] = None,
        lazy: bool = True,
    ) -> tuple:
        assert chunks is not None
        chunks = validate_chunks(self.ensemble_shape, chunks)
        blocks = tuple(
            rotation.divide(n, lazy=lazy)
            for rotation, n in zip(self._ensemble_rotations, chunks)
        )
        return blocks

    @property
    def _default_ensemble_chunks(self) -> tuple[str, ...]:
        return ("auto",) * len(self.ensemble_shape)

    @classmethod
    def _partial_transform(
        cls,
        *args: Any,
        axes: tuple[str, ...],
        order: tuple[int, ...],
        num_ensemble_dims: int,
        **kwargs: Any,
    ) -> np.ndarray:
        args = unpack_blockwise_args(args)

        rotations = tuple(
            x for x, _ in sorted(zip(args, order), key=lambda pair: pair[1])
        )

        args = tuple(tuple(item) for item in zip(axes, rotations))
        args = tuple(itertools.chain(*args))

        new = _wrap_with_array(cls(*args, **kwargs), num_ensemble_dims)
        return new

    def _from_partitioned_args(self) -> Callable:
        non_ensemble_args_ind = tuple(
            i for i in range(len(self.rotations)) if i not in self._ensemble_args
        )
        non_ensemble_args = tuple(self.rotations[i] for i in non_ensemble_args_ind)

        num_ensemble_dims = len(self._ensemble_args)
        order = non_ensemble_args_ind + self._ensemble_args

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
        hkl_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if hkl_mask is None:
            hkl_mask = self.get_ensemble_hkl_mask()

        orientation_matrices = self.get_orientation_matrices()

        shape = orientation_matrices.shape[:-2] + (
            len(thicknesses),
            hkl_mask.sum(),
        )

        pbar_obj = TqdmWrapper(
            enabled=pbar,
            total=int(np.prod(orientation_matrices.shape[:-2])),
            leave=False,
        )

        xp = get_array_module(self.device)
        array = xp.zeros(shape, dtype=get_dtype(complex=return_complex))

        # lil_matrix((np.prod(shape[:-1]), shape[-1]))

        for i in np.ndindex(orientation_matrices.shape[:-2]):
            bw = BlochWaves(
                structure_factor=self._structure_factor,
                energy=self.energy,
                sg_max=self.sg_max,
                g_max=self.g_max,
                orientation_matrix=orientation_matrices[i],
                centering=self.centering,
                device=self.device,
                use_wave_eq=self._use_wave_eq,
            )

            # cols = np.where(bw.hkl_mask)[0]
            # rows = np.ravel_multi_index(
            #     i + (tuple(range(shape[-2])),),
            #     dims=shape[:-1],
            # )

            diffraction_patterns = bw.calculate_diffraction_patterns(
                thicknesses,
                return_complex=return_complex,
                lazy=False,
            )

            array[i][..., bw.hkl_mask[hkl_mask]] = diffraction_patterns.array

            pbar_obj.update_if_exists(1)

        pbar_obj.close_if_exists()

        return array

    @staticmethod
    def _run_calculate_diffraction_patterns(
        block: np.ndarray,
        hkl_mask: np.ndarray,
        thicknesses: np.ndarray,
        return_complex: bool,
        pbar: bool,
    ) -> np.ndarray:
        unpacked_block: BlochwaveEnsemble = block.item()

        array = unpacked_block._calculate_diffraction_intensities(
            thicknesses=thicknesses,
            return_complex=return_complex,
            pbar=pbar,
            hkl_mask=hkl_mask,
        )

        return array

    def _lazy_calculate_diffraction_patterns(
        self,
        thicknesses: np.ndarray,
        return_complex: bool,
        pbar: bool,
    ) -> tuple[da.core.Array, np.ndarray]:
        blocks = self.ensemble_blocks(1)

        hkl_mask = self.get_ensemble_hkl_mask()

        shape = self.ensemble_shape + (
            len(thicknesses),
            int(hkl_mask.sum()),
        )

        out_ind = tuple(range(len(shape)))

        xp = get_array_module(self.device)

        out = da.blockwise(
            self._run_calculate_diffraction_patterns,
            out_ind,
            blocks,
            tuple(range(len(self.ensemble_shape))),
            da.from_array(hkl_mask),
            (-1,),
            new_axes={out_ind[-2]: shape[-2], out_ind[-1]: shape[-1]},
            thicknesses=thicknesses,
            return_complex=return_complex,
            pbar=pbar,
            concatenate=True,
            meta=xp.zeros(shape, dtype=get_dtype(complex=return_complex)),
        )
        return out, hkl_mask

    def calculate_diffraction_patterns(
        self,
        thicknesses: float | Sequence[float] | np.ndarray,
        return_complex: bool = False,
        lazy: bool = True,
        pbar: Optional[bool] = None,
    ) -> IndexedDiffractionPatterns:
        """Calculate the dynamical diffraction patterns of the ensemble for a given set
        of thicknesses.

        Parameters
        ----------
        thicknesses : float or sequence of floats
            The thicknesses of the sample [Å].
        return_complex : bool
            If True, the complex diffraction patterns are returned. If False, the
            intensity is returned. Default is False.
        lazy : bool
            If True, the calculation is done lazily using dask. If False, the
            calculation is done eagerly.
        pbar : bool
            If True, a progress bar is shown. Default is None, which means the value is
            taken from the configuration.

        Returns
        -------
        IndexedDiffractionPatterns
            The diffraction patterns.
        """

        if pbar is None:
            pbar = config.get("diagnostics.task_progress", False)

        if isinstance(thicknesses, (float, int)):
            ensemble_axes_metadata = []
        else:
            ensemble_axes_metadata = [
                ThicknessAxis(label="z", units="Å", values=tuple(thicknesses))
            ]

        _warn_if_single_precision(self.device)
        thicknesses = np.array(thicknesses, dtype=get_dtype())

        if thicknesses.ndim == 0:
            thicknesses = thicknesses[None]
            squeeze_thickness_dim = True
        else:
            squeeze_thickness_dim = False

        array: np.ndarray | da.core.Array
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
        hkl = self.structure_factor.hkl[hkl_mask]

        reciprocal_lattice_vectors = np.matmul(
            reciprocal_cell(self.structure_factor.cell)[None],
            np.swapaxes(orientation_matrices, -2, -1),
        )

        if squeeze_thickness_dim:
            array = array[..., 0, :]
            ensemble_axes_metadata = ensemble_axes_metadata[:-1]
        else:
            reciprocal_lattice_vectors = reciprocal_lattice_vectors[..., None, :, :]

        result = IndexedDiffractionPatterns(
            array=array,
            miller_indices=hkl,
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
            ensemble_axes_metadata=[
                *self.ensemble_axes_metadata,
                *ensemble_axes_metadata,
            ],
            metadata={
                "label": "intensity",
                "units": "arb. unit",
                "energy": self.energy,
                "sg_max": self.sg_max,
                "g_max": self.g_max,
            },
        )

        return result

    def _calculate_exit_waves_eager(
        self,
        thicknesses: np.ndarray,
        gpts: tuple[int, int],
        extent: tuple[float, float],
        normalization: str,
        g_max: Optional[float],
        pbar: bool,
    ) -> np.ndarray:
        orientation_matrices = self.get_orientation_matrices()

        shape = orientation_matrices.shape[:-2] + (len(thicknesses),) + gpts

        pbar_obj = TqdmWrapper(
            enabled=pbar,
            total=int(np.prod(orientation_matrices.shape[:-2])),
            leave=False,
        )

        xp = get_array_module(self.device)
        array = xp.zeros(shape, dtype=get_dtype(complex=True))

        for i in np.ndindex(orientation_matrices.shape[:-2]):
            bw = BlochWaves(
                structure_factor=self._structure_factor,
                energy=self.energy,
                sg_max=self.sg_max,
                g_max=self.g_max,
                orientation_matrix=orientation_matrices[i],
                centering=self.centering,
                device=self.device,
                use_wave_eq=self._use_wave_eq,
            )

            waves = bw.calculate_exit_waves(
                thicknesses=thicknesses,
                gpts=gpts,
                extent=extent,
                normalization=normalization,
                g_max=g_max,
                lazy=False,
            )

            array[i] = waves.array
            pbar_obj.update_if_exists(1)

        pbar_obj.close_if_exists()
        return array

    @staticmethod
    def _run_calculate_exit_waves(
        block: np.ndarray,
        thicknesses: np.ndarray,
        gpts: tuple[int, int],
        extent: tuple[float, float],
        normalization: str,
        g_max: Optional[float],
        pbar: bool,
    ) -> np.ndarray:
        unpacked_block: BlochwaveEnsemble = block.item()
        return unpacked_block._calculate_exit_waves_eager(
            thicknesses=thicknesses,
            gpts=gpts,
            extent=extent,
            normalization=normalization,
            g_max=g_max,
            pbar=pbar,
        )

    def _lazy_calculate_exit_waves(
        self,
        thicknesses: np.ndarray,
        gpts: tuple[int, int],
        extent: tuple[float, float],
        normalization: str,
        g_max: Optional[float],
        pbar: bool,
    ) -> da.core.Array:
        blocks = self.ensemble_blocks(1)

        shape = self.ensemble_shape + (len(thicknesses),) + gpts
        out_ind = tuple(range(len(shape)))

        xp = get_array_module(self.device)

        out = da.blockwise(
            self._run_calculate_exit_waves,
            out_ind,
            blocks,
            tuple(range(len(self.ensemble_shape))),
            new_axes={
                out_ind[-3]: shape[-3],
                out_ind[-2]: shape[-2],
                out_ind[-1]: shape[-1],
            },
            thicknesses=thicknesses,
            gpts=gpts,
            extent=extent,
            normalization=normalization,
            g_max=g_max,
            pbar=pbar,
            concatenate=True,
            meta=xp.zeros(shape, dtype=get_dtype(complex=True)),
        )
        return out

    def calculate_exit_waves(
        self,
        thicknesses: float | Sequence[float] | np.ndarray,
        gpts: Optional[tuple[int, int]] = None,
        extent: Optional[tuple[float, float]] = None,
        normalization: str = "values",
        g_max: Optional[float] = None,
        lazy: bool = True,
        pbar: Optional[bool] = None,
    ) -> Waves:
        """Calculate the exit waves for the ensemble for a given set of
        thicknesses.

        Parameters
        ----------
        thicknesses : float or sequence of floats
            The thicknesses of the sample [Å].
        gpts : tuple of ints, optional
            The grid points of the exit waves.
        extent : tuple of floats, optional
            The extent of the exit waves [Å].
        normalization : {'values', 'amplitude'}
            The normalization of the exit waves.
        g_max : float, optional
            Maximum scattering vector length for the plane wave
            expansion [1/Å].
        lazy : bool
            If True, the calculation is done lazily using dask. If False,
            the calculation is done eagerly.
        pbar : bool, optional
            If True, a progress bar is shown. Default is None, which means
            the value is taken from the configuration.

        Returns
        -------
        Waves
            The exit waves.
        """

        if pbar is None:
            pbar = config.get("diagnostics.task_progress", False)

        if extent is None:
            base_cell = np.array(self._structure_factor.cell)
            orientation_matrices = self.get_orientation_matrices()
            max_extent = np.zeros(2)
            for i in np.ndindex(orientation_matrices.shape[:-2]):
                rotated_cell = Cell(np.dot(base_cell, orientation_matrices[i].T))
                bounds = cell_bounds(rotated_cell)[:2]
                max_extent = np.maximum(max_extent, bounds)
            extent = (float(max_extent[0]), float(max_extent[1]))

        if gpts is None:
            effective_g_max = g_max if g_max is not None else self.g_max
            sampling = (1 / effective_g_max / 2, 1 / effective_g_max / 2)
            gpts = (
                int(np.ceil(extent[0] / sampling[0])),
                int(np.ceil(extent[1] / sampling[1])),
            )

        if isinstance(thicknesses, (float, int)):
            ensemble_axes_metadata: list[AxisMetadata] = []
        else:
            ensemble_axes_metadata = [
                ThicknessAxis(
                    label="z", units="Å", values=tuple(thicknesses)
                )
            ]

        _warn_if_single_precision(self.device)
        thicknesses = np.array(thicknesses, dtype=get_dtype())

        if thicknesses.ndim == 0:
            thicknesses = thicknesses[None]
            squeeze_thickness_dim = True
        else:
            squeeze_thickness_dim = False

        array: np.ndarray | da.core.Array
        if lazy:
            array = self._lazy_calculate_exit_waves(
                thicknesses=thicknesses,
                gpts=gpts,
                extent=extent,
                normalization=normalization,
                g_max=g_max,
                pbar=pbar,
            )
        else:
            array = self._calculate_exit_waves_eager(
                thicknesses=thicknesses,
                gpts=gpts,
                extent=extent,
                normalization=normalization,
                g_max=g_max,
                pbar=pbar,
            )

        if squeeze_thickness_dim:
            array = array[..., 0, :, :]

        waves = Waves(
            array=array,
            extent=extent,
            energy=self.energy,
            ensemble_axes_metadata=[
                *self.ensemble_axes_metadata,
                *ensemble_axes_metadata,
            ],
            metadata={"normalization": normalization},
        )

        return waves
