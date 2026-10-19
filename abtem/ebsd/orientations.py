"""Zone-axis patches and the rotated slabs they are calculated in.

A multislice calculation is accurate for directions close to its slab's axis,
so a reference pattern tiles the hemisphere into patches, each calculated in a
slab cut with the patch's zone axis along ``z``. :func:`fibonacci_hemisphere`
chooses the patch centres and :func:`rotated_slab` cuts the slabs.

.. warning::
    A box cut from a rotated crystal is in general not periodic across its
    faces, but the multislice treats it as periodic. The resulting seams are
    the main approximation of a tiled pattern; their effect grows with a
    direction's angle from the slab axis and with the slab's thickness over its
    width.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms

from abtem.inelastic.phonons import BaseFrozenPhonons, FrozenPhonons

__all__ = [
    "is_centrosymmetric",
    "fibonacci_hemisphere",
    "zone_axis_rotation",
    "estimate_repetitions",
    "central_origin",
    "bulk_block",
    "rotated_slab",
]


def fibonacci_hemisphere(n: int) -> np.ndarray:
    """Evenly distributed directions on the northern hemisphere.

    A Fibonacci lattice, near-uniform in solid angle.

    Parameters
    ----------
    n : int
        Number of directions to generate.

    Returns
    -------
    directions : np.ndarray
        Unit vectors of shape ``(n, 3)``, ordered from the pole down towards
        the equator.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    i = np.arange(n)
    golden_ratio_inverse = (np.sqrt(5.0) - 1.0) / 2.0

    # half the step of a full sphere keeps z in (0, 1)
    z = 1.0 - (2.0 * i + 1.0) / (2.0 * n)
    r = np.sqrt(1.0 - z**2)
    theta = 2.0 * np.pi * i * golden_ratio_inverse

    return np.column_stack((r * np.cos(theta), r * np.sin(theta), z))


def zone_axis_rotation(zone_axis: np.ndarray) -> np.ndarray:
    """Rotation matrix taking a zone axis onto ``+z``.

    The matrix ``R`` maps the crystal frame to the slab frame: a direction
    transforms as ``R @ d``. It is the rotation of ``atoms.rotate(zone_axis,
    "z")`` in ASE.

    Parameters
    ----------
    zone_axis : np.ndarray
        Beam direction in the crystal frame, of shape ``(3,)``. Need not be
        normalized.

    Returns
    -------
    rotation : np.ndarray
        Rotation matrix of shape ``(3, 3)``.
    """
    zone_axis = np.asarray(zone_axis, dtype=float).ravel()

    if zone_axis.shape != (3,):
        raise ValueError(f"zone_axis must have shape (3,), got {zone_axis.shape}")

    norm = np.linalg.norm(zone_axis)
    if norm == 0.0:
        raise ValueError("zone_axis must be nonzero")

    v = zone_axis / norm
    target = np.array([0.0, 0.0, 1.0])

    cos_angle = float(np.dot(v, target))
    axis = np.cross(v, target)
    sin_angle = float(np.linalg.norm(axis))

    if sin_angle < 1e-7:
        # parallel or antiparallel to z: any perpendicular axis, as ASE picks it
        axis = np.cross((0.0, 0.0, 1.0), target)
        if np.linalg.norm(axis) < 1e-7:
            axis = np.cross((1.0, 0.0, 0.0), target)
        axis = axis / np.linalg.norm(axis)
    else:
        axis = axis / sin_angle

    # Rodrigues' rotation formula
    cross_matrix = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )

    return (
        cos_angle * np.eye(3)
        + sin_angle * cross_matrix
        + (1.0 - cos_angle) * np.outer(axis, axis)
    )


def estimate_repetitions(
    atoms: Atoms, cell: tuple[float, float, float]
) -> tuple[int, int, int]:
    """Repetitions of `atoms` needed to fill `cell` at any orientation.

    The block has to contain the slab's bounding sphere, whose diameter is the
    slab's diagonal, along each of its face separations (shorter than its edges
    for a non-orthogonal cell). One cell more is added along each axis, since
    the slab is cut about a point up to half a cell from the block's centre.

    Parameters
    ----------
    atoms : ase.Atoms
        The unit cell to repeat.
    cell : tuple of three float
        Dimensions of the slab to be cut [Å].

    Returns
    -------
    repetitions : tuple of three int
    """
    diagonal = float(np.linalg.norm(cell))
    lattice = np.asarray(atoms.cell.array, dtype=float)

    volume = abs(float(np.linalg.det(lattice)))
    if volume == 0.0:
        raise ValueError("atoms must have a cell with nonzero volume")

    # separation of the faces spanned by the other two lattice vectors
    separations = [
        volume / np.linalg.norm(np.cross(lattice[(i + 1) % 3], lattice[(i + 2) % 3]))
        for i in range(3)
    ]

    return tuple(int(np.ceil(diagonal / s)) + 1 for s in separations)  # type: ignore[return-value]


def central_origin(
    atoms: Atoms, repetitions: tuple[int, int, int], origin: np.ndarray
) -> np.ndarray:
    """Move a point of `atoms` to the equivalent point in the middle of its block.

    The block ``atoms * repetitions`` grows away from the corner of `atoms`,
    so a point given in `atoms` is shifted by whole lattice vectors to the
    central copy. With ``repetitions == (1, 1, 1)`` a point inside the cell is
    left where it is.

    Parameters
    ----------
    atoms : ase.Atoms
        The unrepeated cell.
    repetitions : tuple of three int
        Repetitions the block was built with.
    origin : np.ndarray
        Cartesian point of `atoms`, of shape ``(3,)``.

    Returns
    -------
    origin : np.ndarray
        The equivalent Cartesian point of the block.
    """
    lattice = np.asarray(atoms.cell.array, dtype=float)
    origin = np.asarray(origin, dtype=float).ravel()

    fractional = np.linalg.solve(lattice.T, origin)
    shift = np.round(np.asarray(repetitions, dtype=float) / 2.0 - fractional)

    # np.round sends 0.5 to 0, so a point inside a (1, 1, 1) cell does not move
    return origin + shift @ lattice


def bulk_block(
    atoms: Atoms,
    cell: tuple[float, float, float],
    repetitions: tuple[int, int, int] | None = None,
) -> Atoms:
    """The repeated block that :func:`rotated_slab` cuts a slab out of.

    The block does not depend on the zone axis, so it can be built once and
    passed to :func:`rotated_slab` with ``repetitions=(1, 1, 1)``.

    Parameters
    ----------
    atoms : ase.Atoms
        The unit cell of the crystal.
    cell : tuple of three float
        Dimensions of the slab to be cut [Å].
    repetitions : tuple of three int, optional
        Repetitions of `atoms`. If not given, :func:`estimate_repetitions`
        chooses the smallest block that can contain the slab at any
        orientation.

    Returns
    -------
    block : ase.Atoms
    """
    if repetitions is None:
        repetitions = estimate_repetitions(atoms, cell)

    return atoms * repetitions


def rotated_slab(
    atoms: Atoms,
    zone_axis: np.ndarray,
    cell: tuple[float, float, float],
    repetitions: tuple[int, int, int] | None = None,
    origin: np.ndarray | None = None,
) -> tuple[Atoms, np.ndarray]:
    """Cut an orthogonal slab with the given zone axis along ``z``.

    The unit cell is repeated into a block large enough to contain the slab at
    any orientation, rotated so that `zone_axis` points along ``+z``, and a box
    of the requested size is cut about `origin`. The slab is not periodic in
    ``x`` and ``y`` (see the module docstring).

    Parameters
    ----------
    atoms : ase.Atoms
        The unit cell of the crystal.
    zone_axis : np.ndarray
        Beam direction in the crystal frame, of shape ``(3,)``.
    cell : tuple of three float
        Dimensions of the slab to cut [Å]. The third entry is the thickness
        along the beam.
    repetitions : tuple of three int, optional
        Repetitions of `atoms` in the block the slab is cut from. Defaults to
        :func:`estimate_repetitions`. Pass ``(1, 1, 1)`` when `atoms` is
        already a block, from :func:`bulk_block` or an atomic model holding a
        defect. Without an `origin` the repetitions also decide which point of
        the crystal is at the centre of the slab.
    origin : np.ndarray, optional
        Cartesian point of `atoms` to place at the centre of the slab and to
        rotate about, for example a defect. Defaults to the centroid of the
        block. When the block is built here, the origin is moved to its central
        copy with :func:`central_origin`; with ``repetitions=(1, 1, 1)`` it is
        used as given.

    Returns
    -------
    slab : ase.Atoms
        The cut slab, with its cell set to `cell`.
    rotation : np.ndarray
        The rotation of shape ``(3, 3)`` taking the crystal frame to the slab
        frame, as returned by :func:`zone_axis_rotation`.
    """
    cell_array = np.asarray(cell, dtype=float)

    if cell_array.shape != (3,):
        raise ValueError(f"cell must have shape (3,), got {cell_array.shape}")

    zone_axis = np.asarray(zone_axis, dtype=float).ravel()
    rotation = zone_axis_rotation(zone_axis)

    if origin is not None:
        origin = np.asarray(origin, dtype=float).ravel()
        if origin.shape != (3,):
            raise ValueError(f"origin must have shape (3,), got {origin.shape}")

    if repetitions == (1, 1, 1):
        # the rotation below is in place, and the caller may reuse the block
        block = atoms.copy()
    else:
        if repetitions is None:
            repetitions = estimate_repetitions(atoms, cell)
        block = bulk_block(atoms, cell, repetitions)
        if origin is not None:
            origin = central_origin(atoms, repetitions, origin)

    anchor = np.mean(block.positions, axis=0) if origin is None else origin

    # the same rotation as `rotation`, about the anchor
    block.rotate(
        zone_axis / np.linalg.norm(zone_axis), "z", center=anchor, rotate_cell=True
    )

    positions = block.positions - anchor
    half = cell_array / 2.0

    inside = np.logical_and.reduce(
        (
            positions[:, 0] > -half[0],
            positions[:, 0] < half[0],
            positions[:, 1] > -half[1],
            positions[:, 1] < half[1],
            positions[:, 2] > -half[2],
            positions[:, 2] < half[2],
        )
    )

    block.positions = positions
    slab = block[inside]
    slab.translate(half)
    slab.cell = cell_array

    return slab, rotation


def _crystal_and_displacements(
    atoms: Atoms | BaseFrozenPhonons,
) -> tuple[Atoms, Optional[FrozenPhonons]]:
    """The crystal the slabs are cut from, and how to displace them thermally.

    Each atom is tagged with its index in the crystal, so that displacements
    given per atom follow it into every slab.
    """
    if isinstance(atoms, FrozenPhonons):
        frozen_phonons, atoms = atoms, atoms.atoms
    elif isinstance(atoms, BaseFrozenPhonons):
        raise TypeError(
            f"slabs are cut from one crystal, and a {type(atoms).__name__} holds "
            f"several; give Atoms, or FrozenPhonons for thermal displacements"
        )
    else:
        frozen_phonons = None

    atoms = atoms.copy()
    atoms.set_array("cell_index", np.arange(len(atoms)))
    return atoms, frozen_phonons


def _displaced(
    slab: Atoms, frozen_phonons: Optional[FrozenPhonons]
) -> Atoms | FrozenPhonons:
    """The slab, displaced as `frozen_phonons` displaces the crystal it came from.

    The same configurations and seeds, and each atom the displacements of its
    site in the crystal.
    """
    if frozen_phonons is None:
        return slab

    sigmas = frozen_phonons.sigmas
    if not isinstance(sigmas, dict):
        # given per atom of the crystal
        sigmas = np.asarray(sigmas)[slab.arrays["cell_index"]]

    return FrozenPhonons(
        slab,
        num_configs=frozen_phonons.num_configs,
        sigmas=sigmas,
        directions=frozen_phonons.directions,
        seed=frozen_phonons.seed,
    )


def is_centrosymmetric(atoms: Atoms, symprec: float = 1e-3) -> bool:
    """Whether a crystal has a centre of inversion.

    For a centrosymmetric crystal the pattern obeys ``I(-k) = I(k)``, so either
    hemisphere follows from the other.

    Parameters
    ----------
    atoms : ase.Atoms
        The crystal's unit cell.
    symprec : float, optional
        Tolerance on positions [Å] for the symmetry search (default 1e-3).

    Returns
    -------
    centrosymmetric : bool
    """
    import warnings

    import spglib  # type: ignore[import-untyped]

    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    with warnings.catch_warnings():
        # spglib warns about its own internal error handling
        warnings.simplefilter("ignore", DeprecationWarning)
        symmetry = spglib.get_symmetry(cell, symprec=symprec)
    if symmetry is None:
        raise ValueError("spglib could not determine the symmetry of the crystal")
    inversion = -np.eye(3, dtype=int)
    return any(
        np.array_equal(rotation, inversion) for rotation in symmetry["rotations"]
    )
