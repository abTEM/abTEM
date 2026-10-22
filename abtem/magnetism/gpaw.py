from __future__ import annotations

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
from abtem.potentials.gpaw import _GPAW_LOCK
from abtem.potentials.iam import _FieldBuilder
from abtem.slicing import _integrate_planes


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
    with _GPAW_LOCK:
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


def _fourier_slice_integrals(
    array: np.ndarray, slice_limits: list[tuple[float, float]], depth: float
) -> np.ndarray:
    """
    The integrals of `array` along its last axis between each pair of
    `slice_limits` [Å], stacked along a new first axis.

    The last axis holds evenly spaced samples of a field that is periodic over
    `depth`, the first at height 0. The field between the samples is their
    band-limited (trigonometric) interpolant, whose integral over a slice is exact
    in Fourier space for any slice limits. The Nyquist term of an even number of
    samples is taken as a cosine, as for a real field.
    """
    n = array.shape[-1]
    q = (2 * np.pi * np.arange(n // 2 + 1) / depth)[:, None]
    a, b = (np.array(limits, dtype=float) for limits in zip(*slice_limits))

    # The field is real, so the half spectrum of `rfft` holds it: the term of each
    # positive frequency stands for itself and its conjugate at the negative one,
    # twice its real part. The zero frequency and the Nyquist term of an even
    # number of samples stand for themselves.
    weights = np.empty((len(q), len(a)), dtype=complex)
    weights[0] = b - a
    weights[1:] = (np.exp(1j * q[1:] * b) - np.exp(1j * q[1:] * a)) / (1j * q[1:])
    weights[1:] *= 2
    if n % 2 == 0:
        q_nyquist = q[-1]
        weights[-1] = (np.sin(q_nyquist * b) - np.sin(q_nyquist * a)) / q_nyquist
    weights /= n

    # One component at a time, so that only one of their spectra is held at once.
    integrals = np.empty(array.shape[:-1] + (len(a),))
    for component in np.ndindex(array.shape[:-3]):
        integrals[component] = (np.fft.rfft(array[component], axis=-1) @ weights).real
    return np.moveaxis(integrals, -1, 0)


@runtime_checkable
class GPAW(Protocol):
    @property
    def atoms(self) -> Atoms:
        ...

    def get_number_of_grid_points(self) -> np.ndarray:
        ...


class _GPAWMagnetics(_FieldBuilder):
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

        # The number of z planes of the density summed into each real-space slice.
        self._planes_per_slice: Optional[tuple[int, ...]] = None

        if projection == "real_space" and isinstance(slice_thickness, (float, int)):
            # The slices are stacked along the third axis of `plane`, which is the
            # last axis of the field once `generate_slices` has moved its axes.
            axes = plane_to_axes(plane)
            n_z = calculators.get_number_of_grid_points()[axes[2]] * gridrefinement

            slice_thickness = float(slice_thickness)

            depth = np.diag(cell)[axes[2]]

            slice_thickness, n_per_slice = equal_slice_thicknesses(
                n_z, slice_thickness, depth=depth
            )
            self._planes_per_slice = tuple(int(n) for n in n_per_slice)
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
        # The density and its curl are computed on the host; each slice is moved to
        # `device`.
        xp = get_array_module(self.device)

        if self._projection == "real_space":
            bounds = np.cumsum((0,) + self._planes_per_slice)

            dz = slice_thicknesses.sum() / array.shape[-1]

            # Plane k, the sample at z = k dz, stands for the field from
            # k dz - dz / 2 to k dz + dz / 2; a slice boundary lies on a plane,
            # which is split between the slices on either side (see
            # `_integrate_planes`). Summing whole planes from each lower limit
            # (a left Riemann sum) put each slice integral dz / 2 below its slice.
            for slice_idx in range(first_slice, last_slice):
                start, stop = bounds[slice_idx], bounds[slice_idx + 1]
                slice_array = _integrate_planes(array, start, stop) * dz

                if self._valid_gpts != slice_array.shape[1:]:
                    slice_array = fft_interpolate(slice_array, slice_shape)

                yield self._array_object(
                    xp.asarray(slice_array[None]),
                    extent=self.extent,
                    slice_thickness=slice_thicknesses[slice_idx],
                )

        else:
            # A slice is the field integrated through it, in field units times Å,
            # as in the real-space projection and the slices of MagneticField and
            # VectorPotential, not a point sample of the field at its entrance.
            limits = np.cumsum((0.0,) + tuple(slice_thicknesses))
            integrals = _fourier_slice_integrals(
                array,
                list(zip(limits[first_slice:last_slice], limits[first_slice + 1 :])),
                float(limits[-1]),
            )

            for slice_idx in range(first_slice, last_slice):
                slice_array = integrals[slice_idx - first_slice]

                if self._valid_gpts != slice_array.shape[1:]:
                    slice_array = fft_interpolate(slice_array, slice_shape)

                yield self._array_object(
                    xp.asarray(slice_array[None]),
                    extent=self.extent,
                    slice_thickness=slice_thicknesses[slice_idx],
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
