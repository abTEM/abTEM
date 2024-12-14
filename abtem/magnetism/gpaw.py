from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np
from ase import Atoms
from scipy.spatial.transform import Rotation as R  # type: ignore

from abtem.atoms import plane_to_axes
from abtem.bloch.dynamical import equal_slice_thicknesses
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
from abtem.potentials.base import FieldBuilder


def _calculate_non_periodic_magnetic_vector_potential():
    # A_np = mu_0 * M x r
    pass


def calculate_constant_magnetic_field():
    # B_avg = mu_0 * M
    pass


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
    rotation = R.from_euler("xyz", euler_angles)
    rotation_matrix = rotation.as_matrix()

    shape = vector_field.shape[1:]
    vector_field_reshaped = vector_field.reshape(3, -1)

    rotated_field_reshaped = rotation_matrix @ vector_field_reshaped

    rotated_field = rotated_field_reshaped.reshape((3,) + shape)

    return rotated_field


def calculate_magnetic_vector_potential(spin_density, cell):
    m = np.stack(
        [np.zeros_like(spin_density), np.zeros_like(spin_density), spin_density]
    )

    j = bohr_magneton * curl_fourier(m, cell)
    A = -vacuum_permeability * integrate_gradient_fourier(j, cell)
    return A


def get_vector_potential_from_gpaw(calc, gridrefinement=2, assume_colinear=True):
    if not assume_colinear:
        raise NotImplementedError("Non-collinear calculations not supported.")
    n = calc.get_all_electron_density(spin=True, gridrefinement=gridrefinement)
    rho = n[0][0] - n[0][1]
    A = calculate_magnetic_vector_potential(rho, calc.atoms.cell)
    return A


def get_magnetic_field_from_gpaw(calc, gridrefinement=2, assume_colinear=True):
    if not assume_colinear:
        raise NotImplementedError("Non-collinear calculations not supported.")
    A = get_vector_potential_from_gpaw(calc, gridrefinement=gridrefinement)
    B = curl_fourier(A, calc.atoms.cell)
    return B


@runtime_checkable
class GPAW(Protocol):
    @property
    def atoms(self) -> Atoms: ...

    def get_number_of_grid_points(self) -> np.ndarray: ...


class _GPAWMagnetics(FieldBuilder):
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
        rotate_field: Optional[tuple[float, float, float]] = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        frozen_phonons: Optional[BaseFrozenPhonons] = None,
        repetitions: tuple[int, int, int] = (1, 1, 1),
        gridrefinement: int = 4,
        device: Optional[str] = None,
        assume_colinear: bool = True,
    ):
        if not assume_colinear:
            raise NotImplementedError("Non-collinear calculations not supported.")

        self.gridrefinement = gridrefinement

        assert isinstance(calculators, GPAW)
        self._calculators = calculators

        atoms = calculators.atoms

        cell = atoms.cell

        self._rotate_field = rotate_field

        if projection == "real_space" and isinstance(slice_thickness, (float, int)):
            n_z = calculators.get_number_of_grid_points()[2] * gridrefinement

            slice_thickness = float(slice_thickness)

            axes = plane_to_axes(plane)
            depth = np.diag(cell)[axes[2]]

            slice_thickness, n_per_slice = equal_slice_thicknesses(
                n_z, slice_thickness, depth=depth
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

        if self._quantity == "vector_potential":
            array = get_vector_potential_from_gpaw(
                self._calculators, gridrefinement=self.gridrefinement
            )
        elif self._quantity == "magnetic_field":
            array = get_magnetic_field_from_gpaw(
                self._calculators, gridrefinement=self.gridrefinement
            )
        else:
            raise ValueError(f"Unknown quantity: {self._quantity}")

        if self.plane != "xy":
            axes = plane_to_axes(self.plane)
            array = np.moveaxis(array, (axes[0] + 1, axes[1] + 1), (1, 2))
            array = array[axes, ...]

        if self._rotate_field:
            array = rotate_vector_field(array, self._rotate_field)

        slice_thicknesses = np.array(self.slice_thickness)
        slice_shape = (3,) + self._valid_gpts

        if self._projection == "real_space":
            depth = self._calculators.atoms.cell[2, 2]
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
                    slice_array[None],
                    extent=self.extent,
                    slice_thickness=slice_thicknesses[i],
                )

        else:
            shape = array.shape[:-1] + (self.num_slices,)
            array = fft_interpolate(array, shape)

            for i, slice_idx in enumerate(range(first_slice, last_slice)):
                slice_array = array[..., slice_idx]

                if self._valid_gpts != slice_array.shape[1:]:
                    slice_array = fft_interpolate(slice_array, slice_shape)

                yield self._array_object(
                    slice_array[None],
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


class GPAWMagneticField(_GPAWMagnetics, BaseMagneticField):
    def __init__(
        self,
        calculators: GPAW | list[GPAW] | list[str] | str,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        frozen_phonons: Optional[BaseFrozenPhonons] = None,
        repetitions: tuple[int, int, int] = (1, 1, 1),
        gridrefinement: int = 1,
        projection: str = "fft",
        device: Optional[str] = None,
    ):
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
        rotate_field: Optional[tuple[float, float, float]] = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        frozen_phonons: Optional[BaseFrozenPhonons] = None,
        repetitions: tuple[int, int, int] = (1, 1, 1),
        gridrefinement: int = 1,
        projection: str = "fft",
        device: Optional[str] = None,
    ):
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


class SpinDensityMagneticField:
    def __init__(self, spin_density, cell):
        raise NotImplementedError
