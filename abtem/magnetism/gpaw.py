from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from ase import Atoms
from scipy.spatial.transform import Rotation as R  # type: ignore

from abtem.atoms import (
    _box_repetitions,
    _box_strain_warning_silenced,
    _cell_in_plane_frame,
    plane_to_axes,
)
from abtem.bloch.dynamical import equal_slice_thicknesses
from abtem.core.backend import asnumpy, get_array_module
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
from abtem.potentials.gpaw import _GPAW_LOCK, GPAWPotential
from abtem.potentials.iam import PotentialArray, _default_box, _FieldBuilder
from abtem.slicing import is_number


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
    vector_field : numpy.ndarray
        3xNxMxK array representing the 3D vector field.
    euler_angles : tuple
        Euler angles (xyz) for the rotation.

    Returns
    -------
    rotated_field : numpy.ndarray
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


def _real_space_slicing(
    slice_thickness: float | tuple[float, ...], num_planes: int, depth: float
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """
    The slice thicknesses of a real-space projection and the number of z planes
    of the density summed into each slice, which add up to `num_planes`.

    A single thickness is divided into slices of whole planes as evenly as the
    planes allow. A sequence of thicknesses must put every slice boundary on a
    plane, as the thicknesses returned here do, so that a builder made with the
    `slice_thickness` of another slices the same way.
    """
    if is_number(slice_thickness):
        thicknesses, planes = equal_slice_thicknesses(
            num_planes, float(slice_thickness), depth=depth
        )
        return tuple(thicknesses), tuple(int(n) for n in planes)

    dz = depth / num_planes
    thicknesses = np.array([float(t) for t in slice_thickness])
    planes = np.rint(thicknesses / dz).astype(int)
    if (
        planes.min() < 1
        or planes.sum() != num_planes
        or not np.allclose(planes * dz, thicknesses, rtol=1e-6, atol=0.0)
    ):
        raise NotImplementedError(
            "The slice thicknesses of the real-space projection must be whole "
            f"numbers of the z grid spacing, {dz:.6g} Å, that add up to the depth, "
            f"{depth:.6g} Å; the nearest are {tuple(float(n * dz) for n in planes)}."
        )
    return tuple(float(n * dz) for n in planes), tuple(int(n) for n in planes)


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
    spectrum = np.fft.fft(array, axis=-1)
    k = np.fft.fftfreq(n, d=1.0 / n)
    q = 2 * np.pi * k / depth
    a, b = (np.array(limits, dtype=float) for limits in zip(*slice_limits))

    weights = np.empty((n, len(a)), dtype=complex)
    weights[0] = b - a
    q = q[:, None]
    weights[1:] = (np.exp(1j * q[1:] * b) - np.exp(1j * q[1:] * a)) / (1j * q[1:])
    if n % 2 == 0:
        q_nyquist = q[n // 2]
        weights[n // 2] = (np.sin(q_nyquist * b) - np.sin(q_nyquist * a)) / q_nyquist

    integrals = (spectrum @ weights).real / n
    return np.moveaxis(integrals, -1, 0)


def _is_skewed(cell) -> bool:
    """Whether `cell` is not orthogonal by the tolerance of `orthogonalize_cell`:
    off-diagonal components up to 1e-6 of the longest lattice vector are
    round-off."""
    cell = np.array(cell, dtype=float)
    off_diagonal = cell[~np.eye(3, dtype=bool)]
    return bool(np.abs(off_diagonal).max() > 1e-6 * np.linalg.norm(cell, axis=1).max())


def _supercell_of_box(cell, box) -> tuple[np.ndarray, np.ndarray]:
    """
    The lattice vectors of the supercell of `cell` that fills `box` (an integer
    matrix, rows along x, y and z, in units of the lattice vectors of `cell`), and
    the rotation that `orthogonalize_cell` gives the vectors when it fits that
    supercell into the box (acting on row vectors).

    `orthogonalize_cell` keeps the fractional coordinates of the supercell: a point
    at r goes to r @ A with A the map from the supercell onto the box, which is a
    rotation times a strain. A vector field placed the same way turns its vectors
    by the rotation; a strain has no single rule for the vectors and leaves them.
    """
    cell = _cell_in_plane_frame(cell, "xy")
    vectors = _box_repetitions(cell, box)
    transform = np.linalg.solve(vectors @ cell, np.diag(box))
    u, _, vt = np.linalg.svd(transform)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        raise NotImplementedError(
            "The cell of the calculator is left-handed, so its box mirrors the "
            "field, which a magnetic field does not follow as a vector field does."
        )
    return vectors.astype(int), rotation


def _box_grid(vectors: np.ndarray, shape: tuple[int, int, int]):
    """
    The grid of the box filled by the supercell with lattice vectors `vectors` that
    holds every Fourier component of a field sampled on a grid of `shape` over the
    cell, and where each component goes on it.

    The component of the cell's frequency m (in units of its reciprocal lattice
    vectors) is the component of frequency `vectors @ m` of the box. A box axis
    made of one lattice vector that no other box axis uses takes the frequencies
    of that axis alone, so it keeps their number of samples per period. The other
    axes mix frequencies of several axes of the cell; the Nyquist component of an
    even number of samples along these is split into halves at +N/2 and -N/2,
    which is the same real field, and the box axis holds every frequency the mix
    gives, so that no two components share a frequency of the box.

    Returns the shape of the box grid and, for each axis of the cell, the indices
    of the FFT output taken, their frequencies and their weights.
    """
    vectors = np.asarray(vectors, dtype=int)

    alone = [False] * 3
    box_shape = [0] * 3
    for k in range(3):
        (used,) = np.nonzero(vectors[k])
        if len(used) == 1 and np.count_nonzero(vectors[:, used[0]]) == 1:
            alone[used[0]] = True
            box_shape[k] = abs(int(vectors[k, used[0]])) * shape[used[0]]

    indices, frequencies, weights = [], [], []
    for j, n in enumerate(shape):
        index = np.arange(n)
        frequency = np.fft.fftfreq(n, d=1.0 / n).astype(int)
        weight = np.ones(n)
        if not alone[j] and n % 2 == 0:
            index = np.append(index, n // 2)
            frequency = np.append(frequency, n // 2)
            weight = np.append(weight, 0.5)
            weight[n // 2] = 0.5
        indices.append(index)
        frequencies.append(frequency)
        weights.append(weight)

    for k in range(3):
        if box_shape[k] == 0:
            box_shape[k] = 1 + sum(
                abs(int(vectors[k, j])) * int(np.ptp(frequencies[j])) for j in range(3)
            )

    return tuple(box_shape), indices, frequencies, weights


def _map_to_box_grid(array: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    The field `array` (components first, then the three axes of a grid over the
    cell along its lattice vectors) on the grid of the box filled by the supercell
    with lattice vectors `vectors`, as `_box_grid` lays it out.

    The field is placed in the fractional coordinates of the supercell, as
    `orthogonalize_cell` places atoms. Each Fourier component moves to its
    frequency on the box without interpolation, so the band-limited field is the
    same at every point.
    """
    grid_shape = array.shape[-3:]
    box_shape, indices, frequencies, weights = _box_grid(vectors, grid_shape)

    spectrum = np.fft.fftn(array, axes=(-3, -2, -1))
    spectrum = spectrum[(...,) + np.ix_(*indices)]
    spectrum = spectrum * (
        weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None]
    )

    target = tuple(
        (
            vectors[k, 0] * frequencies[0][:, None, None]
            + vectors[k, 1] * frequencies[1][None, :, None]
            + vectors[k, 2] * frequencies[2][None, None]
        )
        % box_shape[k]
        for k in range(3)
    )

    box_spectrum = np.zeros(array.shape[:-3] + box_shape, dtype=complex)
    box_spectrum[(...,) + target] = spectrum
    box_spectrum *= np.prod(box_shape) / np.prod(grid_shape)
    return np.fft.ifftn(box_spectrum, axes=(-3, -2, -1)).real


@runtime_checkable
class GPAW(Protocol):
    @property
    def atoms(self) -> Atoms: ...

    def get_number_of_grid_points(self) -> np.ndarray: ...


class _GPAWMagnetics(_FieldBuilder):
    _supports_box_and_origin = False
    # The slices computed from the calculator on first use, kept for the later
    # calls of `generate_slices`; incidental state, never identity.
    _eq_exclude = ("_slices",)

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

        if projection not in ("fft", "real_space"):
            raise ValueError(
                f"projection must be 'fft' or 'real_space', got {projection!r}"
            )

        grid_shape = tuple(
            int(n) * gridrefinement for n in calculators.get_number_of_grid_points()
        )

        # The grid of the calculator runs along the lattice vectors. For a skewed
        # cell `generate_slices` moves the field onto a grid of the default box,
        # whose lattice vectors in units of the cell's are `_box_vectors`.
        self._box_vectors: Optional[np.ndarray] = None
        self._box_rotation: Optional[np.ndarray] = None
        if _is_skewed(cell):
            # Raises the RuntimeError of a cell that cannot be rotated to `plane`.
            default_box = _default_box(cell, plane)
            if plane != "xy":
                raise NotImplementedError(
                    f"plane={plane!r} is not supported for the magnetic field of a "
                    "non-orthogonal cell, whose vectors would have to be rotated "
                    "with it; use plane='xy', or a calculation of an orthogonal cell."
                )
            self._box_vectors, self._box_rotation = _supercell_of_box(cell, default_box)
            grid_shape = _box_grid(self._box_vectors, grid_shape)[0]
            depth = float(default_box[2])
        else:
            depth = float(np.diag(cell)[plane_to_axes(plane)[2]])

        # The number of z planes of the density summed into each real-space slice.
        self._planes_per_slice: Optional[tuple[int, ...]] = None

        if projection == "real_space":
            # The slices are stacked along the third axis of `plane`, which is the
            # last axis of the field once `generate_slices` has moved its axes.
            slice_thickness, self._planes_per_slice = _real_space_slicing(
                slice_thickness,
                num_planes=grid_shape[plane_to_axes(plane)[2]],
                depth=depth,
            )

        self._projection = projection

        # The slices of the whole field, (slice, component, x, y) on the host,
        # computed from the calculator on the first call of `generate_slices`.
        self._slices: Optional[np.ndarray] = None

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

    def _field(self) -> np.ndarray:
        """The field on the host, its components and axes in the frame of `plane`
        and on a grid over the box, the slicing axis last."""
        vector_potential = get_vector_potential_from_gpaw(
            self._calculators, gridrefinement=self.gridrefinement
        )

        if self._quantity == "vector_potential":
            array = vector_potential
        elif self._quantity == "magnetic_field":
            array = curl_fourier(vector_potential, self._calculators.atoms.cell)
        else:
            raise ValueError(f"Unknown quantity: {self._quantity}")

        if self._box_rotation is not None and not np.allclose(
            self._box_rotation, np.eye(3), rtol=0.0, atol=1e-12
        ):
            # `_apply_rotation_matrix` acts on column vectors.
            array = _apply_rotation_matrix(array, self._box_rotation.T)
            vector_potential = _apply_rotation_matrix(
                vector_potential, self._box_rotation.T
            )

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

        if self._box_vectors is not None:
            array = _map_to_box_grid(array, self._box_vectors)

        return array

    def _project(self) -> np.ndarray:
        """
        The field integrated through each slice, (slice, component, x, y) on the
        host, in field units times Å.

        The real-space projection sums the z planes of each slice times their
        spacing. The fft projection integrates the band-limited field between the
        limits of each slice in Fourier space, for any slice thicknesses.
        """
        array = self._field()

        if self._projection == "real_space":
            planes_per_slice = self._planes_per_slice
            assert planes_per_slice is not None
            if sum(planes_per_slice) != array.shape[-1]:
                raise RuntimeError(
                    f"The slices span {sum(planes_per_slice)} z planes, but the "
                    f"calculator's density has {array.shape[-1]}."
                )
            bounds = np.cumsum((0,) + planes_per_slice)
            dz = sum(self.slice_thickness) / array.shape[-1]
            return np.stack(
                [
                    array[..., start:stop].sum(-1) * dz
                    for start, stop in zip(bounds[:-1], bounds[1:])
                ]
            )

        return _fourier_slice_integrals(array, self.slice_limits, self.box[2])

    def generate_slices(self, first_slice: int = 0, last_slice: Optional[int] = None):
        if last_slice is None:
            last_slice = self.num_slices

        # The field is computed from the calculator once and its slices kept, so
        # that building or chunking the slices range by range does not repeat it.
        if self._slices is None:
            self._slices = self._project()

        slice_thicknesses = np.array(self.slice_thickness)
        slice_shape = (3,) + self._valid_gpts
        # The slices are computed on the host; each is moved to `device`.
        xp = get_array_module(self.device)

        for slice_idx in range(first_slice, last_slice):
            slice_array = self._slices[slice_idx]

            if self._valid_gpts != slice_array.shape[1:]:
                slice_array = fft_interpolate(slice_array, slice_shape)
            else:
                # Not a view of the kept slices, which a caller could write into.
                slice_array = slice_array.copy()

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
                array = asnumpy(image.tile(tile).compute().array)
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
    gpts : one or two int, optional
        Forwarded to all built components. See `GPAWPotential`.
    sampling : one or two float, optional
        Forwarded to all built components. See `GPAWPotential`.
    slice_thickness : float or sequence of float, optional
        Forwarded to all built components. See `GPAWPotential`.
    exit_planes : int or tuple of int, optional
        Forwarded to all built components. See `GPAWPotential`.
    plane : str or two tuples of three float, optional
        Forwarded to all built components. See `GPAWPotential`.
    origin : three float, optional
        Forwarded to all built components. See `GPAWPotential`.
    box : three float, optional
        Forwarded to all built components. See `GPAWPotential`.
    periodic : bool
        Forwarded to all built components. See `GPAWPotential`.
    device : str, optional
        Forwarded to all built components. See `GPAWPotential`.
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

    # A default box that strains the atoms was reported by the potential above.
    with _box_strain_warning_silenced():
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
