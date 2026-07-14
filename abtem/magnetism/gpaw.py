from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from ase import Atoms
from scipy.spatial.transform import Rotation as R  # type: ignore

from abtem.atoms import plane_to_axes
from abtem.bloch.dynamical import equal_slice_thicknesses
from abtem.core.backend import get_array_module
from abtem.core.fft import fft_interpolate
from abtem.inelastic.phonons import BaseFrozenPhonons
from abtem.magnetism.iam import (
    BaseMagneticField,
    BaseVectorPotential,
    MagneticFieldArray,
    VectorPotentialArray,
)
from abtem.magnetism.utils import bohr_magneton, vacuum_permeability
from abtem.potentials.charge_density import curl_fourier, integrate_gradient_fourier
from abtem.potentials.gpaw import GPAWPotential
from abtem.potentials.iam import PotentialArray, _FieldBuilder


def calculate_constant_magnetic_field(magnetization: np.ndarray) -> np.ndarray:
    """
    Uniform part of the magnetic field, B_avg = mu_0 * mu_B * <m> [T].

    A periodic vector potential can only represent a magnetic field with
    zero cell average [PRB 94, 174414 (2016), Appendix], so the field of a
    magnetic solid decomposes as B = B_p + B_avg, where B_p (zero average)
    comes from the periodic vector potential and B_avg = mu_0 * M is the
    volume average of the field, with M the magnetization of the material.
    B_avg vanishes for antiferromagnets and other compensated textures.

    Parameters
    ----------
    magnetization : np.ndarray
        Magnetization density of shape (3,) + (nx, ny, nz) in units of
        Bohr magnetons per cubic Ångström.

    Returns
    -------
    average_field : np.ndarray
        The uniform magnetic field as a vector of shape (3,) [T].
    """
    magnetization = np.asarray(magnetization)
    average_magnetization = magnetization.reshape(3, -1).mean(axis=1)
    return vacuum_permeability * bohr_magneton * average_magnetization


def calculate_non_periodic_magnetic_vector_potential(
    average_field: np.ndarray,
    cell,
    gpts: tuple[int, int, int],
    origin: str | tuple[float, float, float] = "center",
) -> np.ndarray:
    """
    Non-periodic vector potential of a uniform magnetic field on a 3D grid.

    In the Coulomb gauge, A_np(r) = 1/2 * B_avg x (r - r0) represents the
    uniform field component B_avg = curl(A_np) that a periodic vector
    potential cannot carry [PRB 94, 174414 (2016), Appendix]. A_np grows
    linearly with distance from `origin` and is *not* periodic: simulations
    including it are only physical while the wave function has negligible
    amplitude at the supercell boundary, so use a laterally large supercell
    (in the reference calculations of PRB 94, 174414 this was 48x48 to
    60x60 unit cells).

    Parameters
    ----------
    average_field : np.ndarray
        The uniform magnetic field as a vector of shape (3,) [T], e.g. from
        `calculate_constant_magnetic_field`.
    cell : ase.cell.Cell or np.ndarray
        Cell defining the region of space the grid spans.
    gpts : tuple of three int
        Number of grid points along each cell vector.
    origin : "center" or tuple of three float
        Gauge origin r0 [Å]. Defaults to the cell center, which minimizes
        |A_np| over the cell.

    Returns
    -------
    vector_potential : np.ndarray
        The non-periodic vector potential of shape (3,) + gpts [ÅT].
    """
    average_field = np.asarray(average_field, dtype=float)
    cell = np.asarray(cell, dtype=float)

    fractional = np.stack(
        np.meshgrid(*(np.arange(n) / n for n in gpts), indexing="ij"), axis=-1
    )
    r = fractional @ cell

    if isinstance(origin, str) and origin == "center":
        r0 = cell.sum(axis=0) / 2
    else:
        r0 = np.asarray(origin, dtype=float)

    A = 0.5 * np.cross(average_field, r - r0)
    return np.moveaxis(A, -1, 0)


def _apply_rotation_matrix(
    vector_field: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    shape = vector_field.shape[1:]
    vector_field_reshaped = vector_field.reshape(3, -1)

    rotated_field_reshaped = rotation_matrix @ vector_field_reshaped

    return rotated_field_reshaped.reshape((3,) + shape)


def rotate_vector_field(
    vector_field: np.ndarray, euler_angles: tuple[float, float, float]
) -> np.ndarray:
    """
    Rotate a 3D vector field defined on a grid using Euler angles.

    Parameters
    ----------
    vector_field : np.ndarray
        3xNxMxK array representing the 3D vector field.
    euler_angles : tuple
        Euler angles (xyz) for the rotation.

    Returns
    -------
    rotated_field : np.ndarray
        Rotated 3D vector field.
    """
    rotation_matrix = R.from_euler("xyz", euler_angles).as_matrix()
    return _apply_rotation_matrix(vector_field, rotation_matrix)


def calculate_vector_potential_from_magnetization(
    magnetization: np.ndarray, cell
) -> np.ndarray:
    """
    Periodic part of the magnetic vector potential from a magnetization
    density.

    The spin current density j = curl(mu_B * m) sources the periodic vector
    potential through the Poisson equation laplace(A_p) = -mu_0 * j, solved
    here in Fourier space (Coulomb gauge) [PRB 94, 174414 (2016), Sec. II D].

    A periodic vector potential can only represent a magnetic field with
    zero cell average, so for a ferromagnet the uniform remainder must be
    added separately; see `calculate_constant_magnetic_field` and
    `calculate_non_periodic_magnetic_vector_potential`.

    Parameters
    ----------
    magnetization : np.ndarray
        Magnetization density of shape (3,) + (nx, ny, nz) in units of
        Bohr magnetons per cubic Ångström.
    cell : ase.cell.Cell
        ASE `Cell` object defining the region of space where the
        magnetization is defined.

    Returns
    -------
    vector_potential : np.ndarray
        The periodic vector potential of shape (3,) + (nx, ny, nz) [ÅT].
    """
    j = bohr_magneton * curl_fourier(magnetization, cell)
    A = -vacuum_permeability * integrate_gradient_fourier(j, cell)
    return A


def calculate_magnetic_vector_potential(spin_density, cell):
    """
    Periodic vector potential [ÅT] of a collinear spin density (m along z).

    See `calculate_vector_potential_from_magnetization` for the general
    (non-collinear) form.
    """
    m = np.stack(
        [np.zeros_like(spin_density), np.zeros_like(spin_density), spin_density]
    )
    return calculate_vector_potential_from_magnetization(m, cell)


def get_magnetization_from_gpaw(calc, gridrefinement: int = 2):
    """
    Extract the all-electron magnetization density from a converged GPAW
    calculation.

    Supports collinear spin-polarized calculations (two spin channels; the
    magnetization is placed along z by convention, since collinear spin has
    no real-space direction) and non-collinear calculations (four density
    components n, mx, my, mz; requires GPAW's new-style calculator with
    plane-wave mode, ``symmetry='off'`` and ``magmoms`` given as an (N, 3)
    array).

    Parameters
    ----------
    calc : gpaw.new.ase_interface.ASECalculator or gpaw.calculator.GPAW
        Converged GPAW calculator. Non-collinear extraction requires the
        new-style calculator (GPAW >= 25).
    gridrefinement : int
        Grid refinement factor for the all-electron density.

    Returns
    -------
    magnetization : np.ndarray
        Magnetization density of shape (3,) + (nx, ny, nz) in units of
        Bohr magnetons per cubic Ångström.
    collinear : bool
        True when extracted from a collinear calculation, where the spin
        axis is arbitrary (represented along z); False for genuinely
        directional non-collinear magnetization.
    """
    dft = getattr(calc, "dft", None)

    if dft is not None:
        n = dft.densities().all_electron_densities(grid_refinement=gridrefinement)
        density = n.gather(broadcast=True).data

        if density.shape[0] == 1:
            raise ValueError(
                "The GPAW calculation is spin-paired and has no magnetization; "
                "run a spin-polarized (or non-collinear) calculation."
            )
        elif density.shape[0] == 2:
            rho = density[0] - density[1]
            zeros = np.zeros_like(rho)
            return np.stack([zeros, zeros, rho]), True
        else:
            assert density.shape[0] == 4  # (n, mx, my, mz)
            return np.ascontiguousarray(density[1:]), False

    # Legacy calculator API: only collinear calculations are supported.
    n_up = calc.get_all_electron_density(spin=0, gridrefinement=gridrefinement)
    n_down = calc.get_all_electron_density(spin=1, gridrefinement=gridrefinement)
    rho = n_up - n_down
    zeros = np.zeros_like(rho)
    return np.stack([zeros, zeros, rho]), True


def get_vector_potential_from_gpaw(calc, gridrefinement=2):
    """Periodic vector potential [ÅT] from a converged GPAW calculation."""
    magnetization, _ = get_magnetization_from_gpaw(
        calc, gridrefinement=gridrefinement
    )
    return calculate_vector_potential_from_magnetization(
        magnetization, calc.atoms.cell
    )


def get_magnetic_field_from_gpaw(calc, gridrefinement=2):
    """Periodic magnetic field [T] from a converged GPAW calculation."""
    A = get_vector_potential_from_gpaw(calc, gridrefinement=gridrefinement)
    B = curl_fourier(A, calc.atoms.cell)
    return B


#: Sentinel default for `rotate_field`: automatically rotate the
#: largest-magnitude in-plane component into z (see
#: `_auto_rotation_matrix_for_vector_field`). Pass an explicit Euler-angle
#: tuple to pick a specific orientation, or `None` to disable rotation and
#: see the raw (Az == 0) output.
_AUTO_ROTATE_FIELD = "auto"

#: Swaps x into z: (Ax, Ay, 0) -> (0, Ay, -Ax). Euler angles (0, pi/2, 0).
_ROTATION_X_INTO_Z = R.from_euler("xyz", (0.0, np.pi / 2, 0.0)).as_matrix()

#: Swaps y into z: (Ax, Ay, 0) -> (Ax, 0, Ay). Euler angles (pi/2, 0, 0).
_ROTATION_Y_INTO_Z = R.from_euler("xyz", (np.pi / 2, 0.0, 0.0)).as_matrix()


def _auto_rotation_matrix_for_vector_field(vector_field: np.ndarray) -> np.ndarray:
    """
    Rotation matrix that swaps whichever of the in-plane (x, y) components
    of `vector_field` has the larger aggregate magnitude into z.

    `calculate_magnetic_vector_potential` always builds the magnetization as
    m = (0, 0, rho): collinear spin has no real-space direction, so GPAW's
    internal spin axis is arbitrary. Because curl(m) and the subsequent
    Poisson solve are applied component-wise, this makes the z-component of
    `vector_field` (and hence the only component `adjust_coulomb_potential`
    uses) identically zero for every collinear calculation, not just some.

    Only the two 90-degree swaps (x into z, or y into z) are considered,
    not an arbitrary in-plane rotation angle: x, y and z are the only
    directions with a physical meaning here (the orthogonal axes of the
    simulation cell), so a continuous "optimal" blend of Ax and Ay has no
    real-space interpretation as a magnetization direction -- it would just
    fit whatever numerical asymmetry happens to be in the grid.
    """
    Ax = vector_field[0].astype(np.float64, copy=False)
    Ay = vector_field[1].astype(np.float64, copy=False)

    Sxx = float(np.sum(Ax * Ax))
    Syy = float(np.sum(Ay * Ay))

    return _ROTATION_X_INTO_Z if Sxx >= Syy else _ROTATION_Y_INTO_Z


def _check_unsupported_ensemble_params(frozen_phonons, repetitions):
    if frozen_phonons is not None:
        raise NotImplementedError(
            "frozen_phonons is not supported for magnetic fields/vector "
            "potentials; build the field from a single calculator and combine "
            "it with an electrostatic FrozenPhonons ensemble instead."
        )
    if tuple(repetitions) != (1, 1, 1):
        raise NotImplementedError(
            "repetitions is not supported for magnetic fields/vector "
            "potentials; build the field for a single unit cell and call "
            ".tile() on the resulting array instead."
        )


@runtime_checkable
class GPAW(Protocol):
    @property
    def atoms(self) -> Atoms:
        ...

    def get_number_of_grid_points(self) -> np.ndarray:
        ...


class _MagnetizationMagnetics(_FieldBuilder):
    """
    Base builder for sliced magnetic fields/vector potentials computed from
    a magnetization density. Subclasses provide `_get_magnetization`,
    returning the (3,) + (nx, ny, nz) magnetization [μB/Å³] and a flag
    marking it as collinear (arbitrary spin axis, represented along z).
    """

    def __init__(
        self,
        array_object,
        cell,
        n_grid_points_z: int,
        quantity: str = "magnetic_field",
        projection: str = "fft",
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        device: Optional[str] = None,
    ):
        self._cell3d = cell

        self._rotate_field = rotate_field
        #: The rotation applied to the field's vector components on the last
        #: `generate_slices` pass (None until then, or when no rotation was
        #: applied). Consumers needing consistency with a separately computed
        #: constant field (see `gpaw_magnetic_fields`) apply the same matrix.
        self._resolved_rotation_matrix: Optional[np.ndarray] = None

        if projection == "real_space" and isinstance(slice_thickness, (float, int)):
            slice_thickness = float(slice_thickness)

            axes = plane_to_axes(plane)
            depth = np.diag(cell)[axes[2]]

            slice_thickness, n_per_slice = equal_slice_thicknesses(
                n_grid_points_z, slice_thickness, depth=depth
            )
        elif projection == "real_space":
            raise NotImplementedError(
                "Non-uniform slice thicknesses not supported for real-space projection."
            )

        self._projection = projection

        self._quantity = quantity

        super().__init__(
            array_object=array_object,
            gpts=gpts,
            cell=cell,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            origin=origin,
            box=box,
            periodic=periodic,
        )

    def _get_magnetization(self) -> tuple[np.ndarray, bool]:
        raise NotImplementedError

    @property
    def num_configurations(self):
        return 1

    @property
    def base_axes_metadata(self):
        pass

    @property
    def plane(self):
        assert isinstance(self._plane, str)
        return self._plane

    def generate_slices(self, first_slice: int = 0, last_slice: Optional[int] = None):
        if last_slice is None:
            last_slice = self.num_slices

        magnetization, collinear = self._get_magnetization()

        vector_potential = calculate_vector_potential_from_magnetization(
            magnetization, self._cell3d
        )

        if self._quantity == "vector_potential":
            array = vector_potential
        elif self._quantity == "magnetic_field":
            array = curl_fourier(vector_potential, self._cell3d)
        else:
            raise ValueError(f"Unknown quantity: {self._quantity}")

        if self.plane != "xy":
            axes = plane_to_axes(self.plane)
            moved_axes = (axes[0] + 1, axes[1] + 1)
            array = np.moveaxis(array, moved_axes, (1, 2))[axes, ...]
            vector_potential = np.moveaxis(vector_potential, moved_axes, (1, 2))[
                axes, ...
            ]

        rotate_field = self._rotate_field
        self._resolved_rotation_matrix = None
        if isinstance(rotate_field, str) and rotate_field == _AUTO_ROTATE_FIELD:
            # The auto swap exists because collinear spin has no real-space
            # direction; a non-collinear magnetization is genuinely
            # directional, so it is never auto-rotated.
            if collinear:
                rotation_matrix = _auto_rotation_matrix_for_vector_field(
                    vector_potential
                )
                array = _apply_rotation_matrix(array, rotation_matrix)
                self._resolved_rotation_matrix = rotation_matrix
        elif rotate_field:
            array = rotate_vector_field(array, rotate_field)
            self._resolved_rotation_matrix = R.from_euler(
                "xyz", rotate_field
            ).as_matrix()

        slice_thicknesses = np.array(self.slice_thickness)
        slice_shape = (3,) + self._valid_gpts
        # calculate_vector_potential_from_magnetization/curl_fourier above
        # are host-only (hardcoded np.fft), so `array` is always NumPy here
        # regardless of self.device; move each yielded slice to the
        # requested device at this boundary instead (mirrors the identical
        # fix for QuasiDipoleProjections.integrate_on_grid in
        # abtem/magnetism/iam.py).
        xp = get_array_module(self.device)

        if self._projection == "real_space":
            depth = self._cell3d[2, 2]
            pixels_per_slice = (slice_thicknesses / depth * array.shape[-1]).astype(int)

            dz = slice_thicknesses.sum() / array.shape[-1]

            start = 0
            for i, slice_idx in enumerate(range(first_slice, last_slice)):
                slice_array = (
                    array[..., start : start + pixels_per_slice[i]].sum(-1) * dz
                )
                start += pixels_per_slice[i]

                if self._valid_gpts != slice_array.shape[1:]:
                    slice_array = fft_interpolate(slice_array, slice_shape)

                yield self._array_object(
                    xp.asarray(slice_array[None]),
                    extent=self.extent,
                    slice_thickness=slice_thicknesses[i],
                )

        else:
            shape = array.shape[:-1] + (self.num_slices,)
            array = fft_interpolate(array, shape)

            for i, slice_idx in enumerate(range(first_slice, last_slice)):
                # Multiply the pointwise sample at the slice plane by the
                # slice thickness so both projection modes yield
                # slice-integrated (projected) fields, matching the
                # convention of PotentialArray and the consumers
                # (adjust_coulomb_potential, the Pauli multislice solver).
                slice_array = array[..., slice_idx] * slice_thicknesses[i]

                if self._valid_gpts != slice_array.shape[1:]:
                    slice_array = fft_interpolate(slice_array, slice_shape)

                yield self._array_object(
                    xp.asarray(slice_array[None]),
                    extent=self.extent,
                    slice_thickness=slice_thicknesses[i],
                )

    def build(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        max_batch: int | str = 1,
        lazy: Optional[bool] = None,
    ):
        if lazy:
            raise ValueError("Lazy not supported for magnetics.")
        return super().build(
            first_slice=first_slice,
            last_slice=last_slice,
            max_batch=max_batch,
            lazy=False,
        )


class _GPAWMagnetics(_MagnetizationMagnetics):
    def __init__(
        self,
        calculators: GPAW | list[GPAW] | list[str] | str,
        array_object,
        quantity: str = "magnetic_field",
        projection: str = "fft",
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        frozen_phonons: Optional[BaseFrozenPhonons] = None,
        repetitions: tuple[int, int, int] = (1, 1, 1),
        gridrefinement: int = 4,
        device: Optional[str] = None,
    ):
        _check_unsupported_ensemble_params(frozen_phonons, repetitions)

        self.gridrefinement = gridrefinement

        assert isinstance(calculators, GPAW)
        self._calculators = calculators

        super().__init__(
            array_object=array_object,
            cell=calculators.atoms.cell,
            n_grid_points_z=calculators.get_number_of_grid_points()[2]
            * gridrefinement,
            quantity=quantity,
            projection=projection,
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            plane=plane,
            rotate_field=rotate_field,
            origin=origin,
            box=box,
            periodic=periodic,
            device=device,
        )

    def _get_magnetization(self) -> tuple[np.ndarray, bool]:
        return get_magnetization_from_gpaw(
            self._calculators, gridrefinement=self.gridrefinement
        )


class _ArrayMagnetics(_MagnetizationMagnetics):
    def __init__(
        self,
        magnetization: np.ndarray,
        cell,
        array_object,
        quantity: str = "magnetic_field",
        projection: str = "fft",
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        device: Optional[str] = None,
    ):
        from ase.cell import Cell

        magnetization = np.asarray(magnetization)

        if magnetization.ndim == 3:
            # A scalar spin density: collinear, along z by convention.
            zeros = np.zeros_like(magnetization)
            magnetization = np.stack([zeros, zeros, magnetization])
            self._collinear = True
        elif magnetization.ndim == 4 and magnetization.shape[0] == 3:
            self._collinear = False
        else:
            raise ValueError(
                "magnetization must have shape (nx, ny, nz) (a collinear "
                "spin density) or (3, nx, ny, nz) (a vector magnetization "
                f"density), got {magnetization.shape}"
            )

        self._magnetization = magnetization

        cell = Cell(np.diag(cell) if np.ndim(cell) == 1 else cell)

        super().__init__(
            array_object=array_object,
            cell=cell,
            n_grid_points_z=magnetization.shape[-1],
            quantity=quantity,
            projection=projection,
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            plane=plane,
            rotate_field=rotate_field,
            origin=origin,
            box=box,
            periodic=periodic,
            device=device,
        )

    def _get_magnetization(self) -> tuple[np.ndarray, bool]:
        return self._magnetization, self._collinear


class MagnetizationMagneticField(_ArrayMagnetics, BaseMagneticField):
    """
    Sliced magnetic field built from a magnetization density on a 3D grid.

    The magnetization may come from any source (e.g. a micromagnetic model
    or a DFT code other than GPAW), making this the DFT-agnostic
    counterpart of `GPAWMagneticField`. Only the periodic part of the
    field is built; see `calculate_constant_magnetic_field` for the
    uniform remainder of a ferromagnet.

    Parameters
    ----------
    magnetization : np.ndarray
        Magnetization density in units of Bohr magnetons per cubic
        Ångström. Either shape (3,) + (nx, ny, nz) for a vector
        (non-collinear) magnetization, or (nx, ny, nz) for a collinear
        spin density (placed along z by convention).
    cell : ase.cell.Cell or np.ndarray
        Cell defining the region of space where the magnetization is
        defined.
    rotate_field : tuple of three float, "auto", or None
        Euler-angle rotation applied to the built field. Defaults to None:
        an explicitly given magnetization is genuinely directional. "auto"
        only applies to collinear (scalar) input.

    Other parameters follow `GPAWMagneticField`.
    """

    def __init__(
        self,
        magnetization: np.ndarray,
        cell,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        projection: str = "fft",
        device: Optional[str] = None,
    ):
        super().__init__(
            magnetization=magnetization,
            cell=cell,
            array_object=MagneticFieldArray,
            quantity="magnetic_field",
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            plane=plane,
            rotate_field=rotate_field,
            origin=origin,
            box=box,
            periodic=periodic,
            projection=projection,
            device=device,
        )


class MagnetizationVectorPotential(_ArrayMagnetics, BaseVectorPotential):
    """
    Sliced magnetic vector potential built from a magnetization density on
    a 3D grid; the DFT-agnostic counterpart of `GPAWVectorPotential`.

    See `MagnetizationMagneticField` for the parameters.
    """

    def __init__(
        self,
        magnetization: np.ndarray,
        cell,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        projection: str = "fft",
        device: Optional[str] = None,
    ):
        super().__init__(
            magnetization=magnetization,
            cell=cell,
            array_object=VectorPotentialArray,
            quantity="vector_potential",
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            plane=plane,
            rotate_field=rotate_field,
            origin=origin,
            box=box,
            periodic=periodic,
            projection=projection,
            device=device,
        )


class SpinDensityMagneticField(MagnetizationMagneticField):
    """
    Sliced magnetic field built from a collinear spin density on a 3D grid.

    Thin convenience wrapper around `MagnetizationMagneticField` for the
    collinear case: the magnetization is `spin_density` along z. Like the
    GPAW builders, `rotate_field` defaults to "auto" here, since a
    collinear spin axis has no real-space meaning.
    """

    def __init__(
        self,
        spin_density: np.ndarray,
        cell,
        rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
        **kwargs,
    ):
        spin_density = np.asarray(spin_density)
        if spin_density.ndim != 3:
            raise ValueError(
                f"spin_density must have shape (nx, ny, nz), got "
                f"{spin_density.shape}"
            )
        super().__init__(
            magnetization=spin_density,
            cell=cell,
            rotate_field=rotate_field,
            **kwargs,
        )


class GPAWMagneticField(_GPAWMagnetics, BaseMagneticField):
    def __init__(
        self,
        calculators: GPAW | list[GPAW] | list[str] | str,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        frozen_phonons: Optional[BaseFrozenPhonons] = None,
        repetitions: tuple[int, int, int] = (1, 1, 1),
        gridrefinement: int = 1,
        projection: str = "fft",
        device: Optional[str] = None,
    ):
        _check_unsupported_ensemble_params(frozen_phonons, repetitions)

        super().__init__(
            calculators=calculators,
            array_object=MagneticFieldArray,
            quantity="magnetic_field",
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            rotate_field=rotate_field,
            origin=origin,
            box=box,
            gridrefinement=gridrefinement,
            projection=projection,
            periodic=periodic,
        )


class GPAWVectorPotential(_GPAWMagnetics, BaseVectorPotential):
    def __init__(
        self,
        calculators: GPAW | list[GPAW] | list[str] | str,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        frozen_phonons: Optional[BaseFrozenPhonons] = None,
        repetitions: tuple[int, int, int] = (1, 1, 1),
        gridrefinement: int = 1,
        projection: str = "fft",
        device: Optional[str] = None,
    ):
        _check_unsupported_ensemble_params(frozen_phonons, repetitions)

        super().__init__(
            calculators=calculators,
            array_object=VectorPotentialArray,
            quantity="vector_potential",
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            rotate_field=rotate_field,
            origin=origin,
            box=box,
            gridrefinement=gridrefinement,
            projection=projection,
            periodic=periodic,
        )


@dataclass
class GPAWMagneticFields:
    """
    Bundles the electrostatic potential, magnetic vector potential and
    (optionally) magnetic field built from the same GPAW calculator(s) by
    `gpaw_magnetic_fields`.

    `potential` may carry a frozen-phonon ensemble axis (or, after tiling,
    come from a `CrystalPotential` build); `vector_potential` and
    `magnetic_field` are always for a single, rigid configuration -- see
    `_check_unsupported_ensemble_params`. Use `.tile()` to bring the
    magnetic components up to a repeated crystal's size, and
    `.combined_potential()` to fold the vector potential into an
    electrostatic potential via `adjust_coulomb_potential`.

    `average_field` (B_avg, the uniform part of the field that a periodic
    vector potential cannot represent -- see `calculate_constant_magnetic_field`)
    is *not* recoverable from `magnetic_field`, since the latter is built
    from the periodic vector potential and integrates to zero average by
    construction. It is computed once from the calculator's magnetization
    while `gpaw_magnetic_fields` still holds a reference to it, permuted by
    the same `plane`-dependent axis order as the field arrays. Pass it as
    `pauli_multislice`'s `average_field` argument; it is left untouched by
    `.tile()` since it is a single global constant, not a per-slice field.
    """

    potential: PotentialArray
    vector_potential: VectorPotentialArray
    magnetic_field: Optional[MagneticFieldArray] = None
    average_field: Optional[np.ndarray] = None

    def tile(
        self, repetitions: tuple[int, int] | tuple[int, int, int]
    ) -> "GPAWMagneticFields":
        """
        Tile `vector_potential` (and `magnetic_field`, if present) to match
        a separately tiled/repeated electrostatic potential, e.g. built via
        `abtem.CrystalPotential`.

        `potential` is left untouched here -- tile or rebuild it separately
        (e.g. `CrystalPotential(electrostatic_ensemble, repetitions=...)`)
        before calling `combined_potential`.
        """
        return replace(
            self,
            vector_potential=self.vector_potential.tile(repetitions),
            magnetic_field=(
                self.magnetic_field.tile(repetitions)
                if self.magnetic_field is not None
                else None
            ),
        )

    def combined_potential(
        self, energy: float, potential: Optional[PotentialArray] = None
    ) -> PotentialArray:
        """
        Combine an electrostatic potential with `vector_potential` via
        `VectorPotentialArray.adjust_coulomb_potential`.

        Parameters
        ----------
        energy : float
            Electron energy [eV].
        potential : PotentialArray, optional
            The electrostatic potential to combine with `vector_potential`.
            Defaults to `self.potential`; pass a separately
            tiled/ensembled potential (e.g. a `CrystalPotential` build)
            after calling `.tile()` for the frozen-phonon workflow.

        Returns
        -------
        PotentialArray
        """
        if potential is None:
            potential = self.potential
        return self.vector_potential.adjust_coulomb_potential(potential, energy=energy)

    def show(
        self,
        tile: tuple[int, int] = (1, 1),
        figsize: tuple[int, int] = (8, 8),
    ):
        """
        Show side-by-side projections of the electrostatic potential and
        the x and z components of the vector potential -- and, if built,
        the magnetic field. `Az`/`Bz` are the components that matter for
        `combined_potential`; `Ax`/`Bx` are shown alongside for a sanity
        check that the in-plane and z-swapped components look sensible
        relative to each other.

        Parameters
        ----------
        tile : two int, optional
            Tile the projected images before plotting, e.g. to preview the
            periodicity of a repeated unit cell.
        figsize : two int, optional
            Figure size passed to `matplotlib.pyplot.figure`.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid

        panels = [
            (self.potential.project(), "$V$", "$Å^{-3}$"),
            (self.vector_potential.project()[0], "$A_x$", "ÅT"),
            (self.vector_potential.project()[2], "$A_z$", "ÅT"),
        ]
        if self.magnetic_field is not None:
            panels += [
                (self.magnetic_field.project()[0], "$B_x$", "T"),
                (self.magnetic_field.project()[2], "$B_z$", "T"),
            ]

        # Create the figure with pyplot's interactive auto-display off, then
        # return it: otherwise Jupyter's inline backend renders it once from
        # the auto-display hook and a second time from the returned value,
        # showing the same plot twice.
        with plt.ioff():
            fig = plt.figure(figsize=figsize)
            grid = ImageGrid(
                fig,
                111,
                nrows_ncols=(1, len(panels)),
                cbar_mode="edge",
                cbar_location="bottom",
                cbar_pad=0.1,
                cbar_size=0.1,
                axes_pad=0.1,
            )

            for ax, (image, name, unit) in zip(grid, panels):
                array = np.asarray(image.tile(tile).array)
                if np.abs(array).max() < 1e-5:
                    vmin, vmax = -1e-5, 1e-5
                else:
                    vmin, vmax = None, None
                im = ax.imshow(array.T, vmin=vmin, vmax=vmax, origin="lower")
                ax.set_title(name)
                ax.cax.colorbar(im, label=unit)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

        return fig


def gpaw_magnetic_fields(
    calculators: GPAW | list[GPAW] | list[str] | str,
    gpts: Optional[int | tuple[int, int]] = None,
    sampling: Optional[float | tuple[float, float]] = None,
    slice_thickness: float | tuple[float, ...] = 1.0,
    exit_planes: Optional[int | tuple[int, ...]] = None,
    plane: str = "xy",
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    box: Optional[tuple[float, float, float]] = None,
    periodic: bool = True,
    frozen_phonons: Optional[BaseFrozenPhonons] = None,
    rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
    include_magnetic_field: bool = False,
    magnetic_calculator: Optional[GPAW] = None,
    device: Optional[str] = None,
    lazy: Optional[bool] = None,
    potential_kwargs: Optional[dict] = None,
    field_kwargs: Optional[dict] = None,
) -> GPAWMagneticFields:
    """
    Build the electrostatic potential, magnetic vector potential and
    (optionally) magnetic field from the same GPAW calculator(s) in one
    call.

    `frozen_phonons` (an ensemble of atomic-displacement configurations) is
    only supported for the electrostatic `potential` -- the magnetic
    components come from a single, rigid ab initio calculation and cannot
    vary per configuration. Tile the returned object with `.tile()` to
    match a separately built, possibly-ensembled electrostatic potential
    (e.g. from `abtem.CrystalPotential`), then combine with
    `.combined_potential()`.

    Parameters
    ----------
    calculators : (list of) gpaw.calculator.GPAW or (list of) str
        One or more converged GPAW calculators (or paths to `.gpw` files).
        Forwarded to `GPAWPotential`. If a list (a frozen-phonon ensemble),
        `magnetic_calculator` must be given explicitly, since
        `GPAWVectorPotential`/`GPAWMagneticField` only support a single
        calculator.
    gpts, sampling, slice_thickness, exit_planes, plane, origin, box,
    periodic, device : see `GPAWPotential`, `GPAWVectorPotential`
        Forwarded to all built components.
    frozen_phonons : BaseFrozenPhonons, optional
        Forwarded to `GPAWPotential` only.
    rotate_field : tuple of three float, "auto", or None
        Forwarded to `GPAWVectorPotential`/`GPAWMagneticField`. Defaults to
        `"auto"`: automatically swap the larger-magnitude in-plane
        component into z (see `_auto_rotation_matrix_for_vector_field`).
    include_magnetic_field : bool
        If True, also build the magnetic field `B` (not used by
        `combined_potential`, only for inspection/visualization). Roughly
        doubles the GPAW-side cost of the magnetic part, so it is off by
        default.
    magnetic_calculator : gpaw.calculator.GPAW, optional
        The single calculator representing the (rigid) magnetic
        contribution. Defaults to `calculators` when that is a single
        calculator; required when `calculators` is a list.
    lazy : bool, optional
        Passed to the electrostatic potential's `.build()`. The magnetic
        components are always built eagerly, since
        `GPAWVectorPotential`/`GPAWMagneticField` do not support lazy
        building.
    potential_kwargs, field_kwargs : dict, optional
        Extra keyword arguments forwarded only to `GPAWPotential`, or only
        to `GPAWVectorPotential`/`GPAWMagneticField`, respectively (e.g.
        their differing `gridrefinement` defaults).

    Returns
    -------
    GPAWMagneticFields
        Its `average_field` (B_avg, the uniform field a periodic vector
        potential cannot represent) is computed from `magnetic_calculator`'s
        magnetization and permuted by the same `plane`-dependent axis order
        as `vector_potential`/`magnetic_field`, so it stays consistent with
        them under a non-default `plane`. Pass it to `pauli_multislice`'s
        `average_field` argument.
    """
    if isinstance(calculators, (list, tuple)):
        if magnetic_calculator is None:
            raise ValueError(
                "calculators is a list (a frozen-phonon ensemble); "
                "GPAWVectorPotential/GPAWMagneticField only support a "
                "single calculator. Pass magnetic_calculator explicitly to "
                "pick which one represents the (rigid) magnetic "
                "contribution."
            )
    elif magnetic_calculator is None:
        magnetic_calculator = calculators

    potential_kwargs = dict(potential_kwargs or {})
    field_kwargs = dict(field_kwargs or {})

    shared = dict(
        gpts=gpts,
        sampling=sampling,
        slice_thickness=slice_thickness,
        exit_planes=exit_planes,
        plane=plane,
        origin=origin,
        box=box,
        periodic=periodic,
        device=device,
    )

    potential = GPAWPotential(
        calculators,
        frozen_phonons=frozen_phonons,
        **shared,
        **potential_kwargs,
    ).build(lazy=lazy)
    if not lazy:
        potential = potential.compute()

    vector_potential_builder = GPAWVectorPotential(
        magnetic_calculator,
        rotate_field=rotate_field,
        **shared,
        **field_kwargs,
    )
    vector_potential = vector_potential_builder.build().compute()

    magnetic_field = None
    if include_magnetic_field:
        magnetic_field = (
            GPAWMagneticField(
                magnetic_calculator,
                rotate_field=rotate_field,
                **shared,
                **field_kwargs,
            )
            .build()
            .compute()
        )

    magnetization, _ = get_magnetization_from_gpaw(
        magnetic_calculator, gridrefinement=field_kwargs.get("gridrefinement", 1)
    )
    average_field = calculate_constant_magnetic_field(magnetization)
    # Keep the constant field consistent with the built arrays: the same
    # plane-dependent component permutation, then the same rotation (if any)
    # that `rotate_field` applied to the vector components.
    average_field = average_field[list(plane_to_axes(plane))]
    if vector_potential_builder._resolved_rotation_matrix is not None:
        average_field = (
            vector_potential_builder._resolved_rotation_matrix @ average_field
        )

    return GPAWMagneticFields(
        potential=potential,
        vector_potential=vector_potential,
        magnetic_field=magnetic_field,
        average_field=average_field,
    )
