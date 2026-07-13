from __future__ import annotations

from dataclasses import dataclass, replace
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
from abtem.potentials.gpaw import GPAWPotential
from abtem.potentials.iam import PotentialArray, _FieldBuilder


def _calculate_non_periodic_magnetic_vector_potential():
    # A_np = mu_0 * M x r
    pass


def calculate_constant_magnetic_field():
    # B_avg = mu_0 * M
    pass


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
        rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
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

        _check_unsupported_ensemble_params(frozen_phonons, repetitions)

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

        vector_potential = get_vector_potential_from_gpaw(
            self._calculators, gridrefinement=self.gridrefinement
        )

        if self._quantity == "vector_potential":
            array = vector_potential
        elif self._quantity == "magnetic_field":
            array = curl_fourier(vector_potential, self._calculators.atoms.cell)
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
        if isinstance(rotate_field, str) and rotate_field == _AUTO_ROTATE_FIELD:
            rotation_matrix = _auto_rotation_matrix_for_vector_field(vector_potential)
            array = _apply_rotation_matrix(array, rotation_matrix)
        elif rotate_field:
            array = rotate_vector_field(array, rotate_field)

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
    """

    potential: PotentialArray
    vector_potential: VectorPotentialArray
    magnetic_field: Optional[MagneticFieldArray] = None

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

    vector_potential = (
        GPAWVectorPotential(
            magnetic_calculator,
            rotate_field=rotate_field,
            **shared,
            **field_kwargs,
        )
        .build()
        .compute()
    )

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

    return GPAWMagneticFields(
        potential=potential,
        vector_potential=vector_potential,
        magnetic_field=magnetic_field,
    )


class SpinDensityMagneticField:
    def __init__(self, spin_density, cell):
        raise NotImplementedError
