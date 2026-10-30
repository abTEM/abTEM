"""Module for the exact two-way solver.

Multislice marches the wave in one direction and cannot represent reflection. At low
energies (a few hundred eV) the scattering by the atom cores is strong, the evanescent
near field couples the forward and backward waves, and several percent of the beam is
reflected. The two-way solver solves the relativistically corrected Helmholtz equation
for a plane wave incident on a periodic slab,

    [laplacian + k^2 + 2 k sigma V(r)] psi = 0,   k = 2 pi / wavelength,

in a basis of the beams g with |g| < cutoff, psi = sum_g phi_g(z) exp(2 pi i g.r). The
coupled equations for phi_g(z) are integrated with the Numerov method (fourth order) on
the potential's slices, with outgoing boundary conditions in the vacuum on both sides,
by a block-tridiagonal solve. Backscattering and evanescent beams are exact; the only
truncations are the beam set and the z step. For a plane wave of any direction the
beams are the plane waves k_t + g of the reciprocal lattice, reduced to the sublattice
that the structure's primitive translations leave coupled; for a normally incident wave
the basis is further reduced to one combination per orbit of the structure's point
symmetries (see :func:`structure_symmetry`).
"""

from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from abtem.core import config
from abtem.core.axes import EnergyAxis
from abtem.core.backend import asnumpy, cp, get_array_module
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.potentials.iam import PotentialArray

if TYPE_CHECKING:
    from ase import Atoms

    from abtem.potentials.iam import BasePotential
    from abtem.waves import PlaneWave, Waves


def _linear_solver(xp):
    """``solve(a, b)`` of the array module. On the CPU it is NumPy's, which releases the
    GIL (SciPy's ``lu_factor`` does not), so that members of an ensemble solved in
    threads run in parallel."""
    if xp is np:
        return np.linalg.solve
    from cupyx.scipy import linalg as cupy_linalg  # type: ignore

    return lambda a, b: cupy_linalg.lu_solve(cupy_linalg.lu_factor(a), b)


def _memory_budget(xp, limit: float) -> float:
    """Bytes the solve may keep: ``limit``, and on a GPU at most 40 % of the free
    memory."""
    if xp is np:
        return limit
    free, _ = cp.cuda.Device().mem_info
    return min(limit, 0.4 * free)


def _available_memory() -> float:
    """Bytes of memory available on the host, a conservative guess without psutil."""
    try:
        import psutil

        return float(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        return 0.5 * os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 8e9


def _member_bytes(nz: int, size: int, itemsize: int) -> float:
    """The bytes that the solve of one member keeps: the cached coupling matrices and
    the elimination blocks of the back substitution (all of them, or the checkpoints
    and one segment), by the limits of :func:`_solve_blocks`."""
    full = nz * size**2 * itemsize
    cache = full if full < _COUPLING_CACHE_BYTES else 0.0
    stored = full
    if full >= _CHECKPOINT_BYTES:
        stored = 2 * np.ceil(np.sqrt(nz)) * size**2 * itemsize
    return cache + stored + 16.0 * size**2 * itemsize


def _blas_threads() -> int:
    """The number of threads that the BLAS libraries use: the cores unless the
    environment (``OMP_NUM_THREADS`` and the like) or a limit says otherwise."""
    counts = [info["num_threads"] for info in threadpool_info()]
    return max(counts) if counts else (os.cpu_count() or 1)


def _default_workers() -> int:
    """Members solved at the same time by default: a quarter of the BLAS threads, each
    member with its share (the blocks are small, and more members than that gave no
    speedup on a 10-core machine with 20 threads). A single thread solves one at a
    time."""
    return max(1, _blas_threads() // 4)


class _MemoryGate:
    """Lets members of an ensemble that run at the same time reserve the memory they
    keep, so that together they stay within the total: a member waits until the others
    leave room, and the only one running takes what it needs."""

    def __init__(self, total: float):
        self.total, self.used = total, 0.0
        self.condition = threading.Condition()

    @contextmanager
    def reserve(self, nbytes: float):
        with self.condition:
            self.condition.wait_for(
                lambda: self.used == 0.0 or self.used + nbytes <= self.total
            )
            self.used += nbytes
        try:
            yield
        finally:
            with self.condition:
                self.used -= nbytes
                self.condition.notify_all()


_MAX_CENTRE_CANDIDATES = 24  # atoms of the smallest class tried for rotation centres
_CHECKPOINT_BYTES = 4e9  # store every elimination block below this size
_COUPLING_CACHE_BYTES = 2e9  # precompute every slice's coupling matrix below this size


def _candidate_operations():
    """Rotations and mirrors of the point groups of 2, 3, 4 and 6-fold lattices."""
    operations = []
    for degrees in sorted({*range(0, 360, 30), *range(0, 360, 45)}):
        t = np.deg2rad(degrees)
        matrix = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        operations.append((f"rot{degrees:g}", matrix))
    for degrees in sorted({*np.arange(0, 180, 15.0), *np.arange(0, 180, 22.5)}):
        t = np.deg2rad(2 * degrees)
        matrix = np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]])
        operations.append((f"mir{degrees:g}", matrix))
    return operations


def _atom_classes(atoms: Atoms, tolerance: float):
    """Index arrays of the atoms of one species at one height."""
    z = atoms.positions[:, 2]
    order = np.argsort(z, kind="stable")
    new_height = np.concatenate([[True], np.diff(z[order]) > tolerance])
    height_class = np.empty(len(z), dtype=int)
    height_class[order] = np.cumsum(new_height) - 1
    keys = np.stack([atoms.numbers, height_class], axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    return [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]


def _maps_onto_itself(new_xy, xy, classes, cell, inverse_cell, tolerance):
    """Whether moving the atoms from xy to new_xy gives the same set (mod the cell)."""
    for members in classes:
        difference = new_xy[members][:, None, :] - xy[members][None, :, :]
        fractional = difference @ inverse_cell
        fractional -= np.round(fractional)
        distance = np.linalg.norm(fractional @ cell, axis=-1)
        if not (distance.min(axis=1) < tolerance).all():
            return False
    return True


def structure_symmetry(atoms: Atoms, tolerance: float = 1e-4):
    """The in-plane symmetry of a slab that a normally incident wave keeps.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure, with an in-plane cell that is orthogonal.
    tolerance : float, optional
        Largest displacement [Å] by which an operation may miss an atom.

    Returns
    -------
    origin : tuple of float
        A point on the axis of the most symmetric point group found [Å].
    operations : list of (str, numpy.ndarray)
        The point operations about the origin (always including the identity).
    translations : list of tuple of float
        Nonzero in-plane translations, as fractions of the cell, that map the structure
        onto itself.
    """
    cell = np.asarray(atoms.cell[:2, :2], dtype=float)
    inverse_cell = np.linalg.inv(cell)
    xy = atoms.positions[:, :2].astype(float)
    classes = _atom_classes(atoms, tolerance)

    # translations: from the first atom of the smallest class to every atom of that
    # class
    smallest = min(classes, key=len)
    translations = []
    for j in smallest[1:]:
        shift = xy[j] - xy[smallest[0]]
        fraction = shift @ inverse_cell
        wrapped = fraction - np.round(fraction)
        if (
            np.linalg.norm(wrapped @ cell) < tolerance
        ):  # a lattice vector: not a new symmetry
            continue
        fraction = fraction - np.floor(fraction)
        if _maps_onto_itself(xy + shift, xy, classes, cell, inverse_cell, tolerance):
            if not any(np.allclose(fraction, t, atol=1e-6) for t in translations):
                translations.append(tuple(fraction))

    # point operations about the best of the candidate axes: the cell origin, one atom
    # of every class, and the centre of every rotation that maps the first atom of the
    # smallest class onto an atom of its class (hexagon and bond centres, for example)
    best_origin, best_operations = (0.0, 0.0), [("rot0", np.eye(2))]
    candidates = [np.zeros(2)] + [xy[members[0]] for members in classes]
    for name, matrix in _candidate_operations():
        if name.startswith("rot") and name != "rot0":
            for j in smallest[:_MAX_CENTRE_CANDIDATES]:
                moved = xy[j] - matrix @ xy[smallest[0]]
                candidates.append(np.linalg.solve(np.eye(2) - matrix, moved))
    unique = []
    for candidate in candidates:
        wrapped = candidate @ inverse_cell
        wrapped = (wrapped - np.floor(wrapped + 1e-9)) @ cell
        if not any(np.linalg.norm(wrapped - u) < 1e-6 for u in unique):
            unique.append(wrapped)
    for origin in unique:
        operations = []
        for name, matrix in _candidate_operations():
            new_xy = (xy - origin) @ matrix.T + origin
            if _maps_onto_itself(new_xy, xy, classes, cell, inverse_cell, tolerance):
                operations.append((name, matrix))
        if len(operations) > len(best_operations):
            best_origin, best_operations = tuple(origin), operations

    return best_origin, best_operations, translations


def structure_glides(
    atoms: Atoms,
    origin,
    operations,
    translations,
    tolerance: float = 1e-4,
):
    """The point operations of the in-plane symmetry that need a translation.

    A glide plane (a mirror with a translation along it, which is not a lattice vector
    or a translation of the structure) or any other operation that maps the structure
    onto itself only together with a translation, about the origin of
    :func:`structure_symmetry`, which finds the operations that need none.

    Returns
    -------
    glides : list of (str, numpy.ndarray, numpy.ndarray)
        The name, the matrix and the translation [Å] of every operation
        ``r -> matrix (r - origin) + origin + translation`` that maps the structure onto
        itself and is not in ``operations``; one for each matrix.
    """
    cell = np.asarray(atoms.cell[:2, :2], dtype=float)
    inverse_cell = np.linalg.inv(cell)
    xy = atoms.positions[:, :2].astype(float)
    classes = _atom_classes(atoms, tolerance)
    smallest = min(classes, key=len)
    origin = np.asarray(origin, dtype=float)
    known = [matrix for _, matrix in operations]
    glides = []
    for name, matrix in _candidate_operations():
        if any(np.allclose(matrix, other, atol=1e-9) for other in known):
            continue
        moved = matrix @ (xy[smallest[0]] - origin) + origin
        for j in smallest:
            translation = xy[j] - moved
            new_xy = (xy - origin) @ matrix.T + origin + translation
            if _maps_onto_itself(new_xy, xy, classes, cell, inverse_cell, tolerance):
                glides.append((name, matrix, translation))
                break
    return glides


class _BeamBasis:
    """The beams of the incident wave and their reduction by symmetry.

    ``shift`` is the in-plane wave vector of the incident wave [1/Å]: the beams are the
    plane waves g + shift of the reciprocal lattice with |g + shift| < cutoff.
    """

    def __init__(
        self,
        gpts,
        sampling,
        cutoff,
        origin,
        operations,
        translations,
        shift=(0.0, 0.0),
    ):
        nx, ny = gpts
        lx, ly = nx * sampling[0], ny * sampling[1]
        ix = np.fft.fftfreq(nx, 1 / nx).round().astype(int)[:, None] * np.ones(
            (1, ny), int
        )
        iy = np.fft.fftfreq(ny, 1 / ny).round().astype(int)[None, :] * np.ones(
            (nx, 1), int
        )
        gx, gy = ix / lx, iy / ly

        lattice = np.ones((nx, ny), bool)
        for tx, ty in translations:
            lattice &= np.abs(np.exp(2j * np.pi * (ix * tx + iy * ty)) - 1) < 1e-9
        keep = lattice & (np.hypot(gx + shift[0], gy + shift[1]) < cutoff)
        self.gpts, self.origin = gpts, origin
        self.indices = np.argwhere(keep)  # (n_beams, 2): the grid indices of the beams
        self.gx, self.gy = gx[keep], gy[keep]
        self.qx, self.qy = self.gx + shift[0], self.gy + shift[1]
        index = {
            (int(a), int(b)): i for i, (a, b) in enumerate(zip(ix[keep], iy[keep]))
        }

        # operations that map every coupling g - h (|g - h| < 2 cutoff) onto the grid
        coupling_range = (lattice & (np.hypot(gx, gy) < 2 * cutoff)).ravel()
        grid_g = np.stack([gx.ravel(), gy.ravel()], axis=1)
        usable = []
        for name, matrix, translation in operations:
            rotated = grid_g @ matrix.T
            ri = np.rint(rotated[:, 0] * lx).astype(int)
            rj = np.rint(rotated[:, 1] * ly).astype(int)
            on_grid = (np.abs(rotated[:, 0] * lx - ri) < 1e-6) & (
                np.abs(rotated[:, 1] * ly - rj) < 1e-6
            )
            inside = (np.abs(ri) <= nx // 2 - 1) & (np.abs(rj) <= ny // 2 - 1)
            if (on_grid & inside)[coupling_range].all():
                usable.append((matrix, np.asarray(translation, dtype=float)))

        # The wave of the symmetric incident wave obeys psi(R r + t) = psi(r) for every
        # operation (R, t), that is phi_h = exp(2 pi i (R h) . t) phi_{R h}: the beams
        # of an orbit carry fixed phases, from the representative (phase 1) over the
        # operations. An orbit that gives two phases to one beam (a systematic absence,
        # for a glide plane) has no symmetric wave in it and is left out.
        orbit = -np.ones(len(self.indices), dtype=int)
        phases = np.zeros(len(self.indices), dtype=complex)
        members = []
        for i in range(len(self.indices)):
            if orbit[i] != -1:
                continue
            found = {i: 1.0 + 0.0j}
            queue, consistent = [i], True
            while queue:
                source = queue.pop()
                g = np.array([self.gx[source], self.gy[source]])
                for matrix, translation in usable:
                    rotated = matrix @ g
                    target = index.get(
                        (int(np.rint(rotated[0] * lx)), int(np.rint(rotated[1] * ly)))
                    )
                    if target is None:
                        continue
                    phase = found[source] * np.exp(-2j * np.pi * rotated @ translation)
                    if target in found:
                        consistent &= abs(found[target] - phase) < 1e-6
                    else:
                        found[target] = phase
                        queue.append(target)
            if consistent:
                for j, phase in found.items():
                    orbit[j] = len(members)
                    phases[j] = phase
                members.append(sorted(found))
            else:
                for j in found:
                    orbit[j] = -2

        self.orbit = orbit
        self.phases = phases
        self.members = members
        self.number_of_orbits = len(members)
        self.representatives = np.array([m[0] for m in members], dtype=int)
        self.sizes = np.array([len(m) for m in members], dtype=float)
        self.reduced = len(usable) > 1 and self.number_of_orbits < len(self.indices)

    def coupling_function(self, fourier_coefficients, wave_number, sigma, xp=np):
        """The coupling matrix U_{g - h}(z_n) in the basis of orbits, for slice n,
        an array of the module ``xp``."""
        nx, ny = self.gpts
        u_k = 2 * wave_number * sigma * fourier_coefficients
        di = (self.indices[:, None, 0] - self.indices[None, :, 0]) % nx
        dj = (self.indices[:, None, 1] - self.indices[None, :, 1]) % ny

        if not self.reduced:
            di, dj = xp.asarray(di), xp.asarray(dj)
            return lambda n: u_k[n][di, dj]

        kept = np.flatnonzero(self.orbit >= 0)
        order = kept[np.argsort(self.orbit[kept], kind="stable")]
        starts = np.searchsorted(self.orbit[order], np.arange(self.number_of_orbits))
        column_di = xp.asarray(di[:, self.representatives])
        column_dj = xp.asarray(dj[:, self.representatives])
        real_dtype = np.empty(0, dtype=u_k.dtype).real.dtype
        weights = xp.asarray(
            np.sqrt(self.sizes[None, :] / self.sizes[:, None]).astype(real_dtype)
        )
        phases = np.conj(self.phases[order]).astype(u_k.dtype)

        if xp is np:
            order_indices = order
            conjugate_phases = phases[:, None]

            def orbit_sum(columns):
                return np.add.reduceat(
                    columns[order_indices] * conjugate_phases, starts, axis=0
                )

        else:
            # no reduceat on the GPU: the sum over the members of every orbit is a
            # product with the matrix of the (conjugate) phases of the orbits
            indicator = np.zeros((self.number_of_orbits, len(self.indices)), u_k.dtype)
            indicator[self.orbit[order], order] = phases
            indicator = xp.asarray(indicator)

            def orbit_sum(columns):
                return indicator @ columns

        def coupling(n):
            columns = u_k[n][
                column_di, column_dj
            ]  # U_{g - h0} for every g, h0 = member of O'
            return orbit_sum(columns) * weights

        return coupling


def _numerov_wave_numbers(kz2, h):
    """The longitudinal wave numbers of the Numerov recurrence in vacuum.

    A plane wave of the beam solves the discrete equations only with the wave number
    kappa_n given by cos(kappa_n h) = (1 - 5 c kz2) / (1 + c kz2), c = h^2 / 12, not
    with the exact sqrt(kz2). Using it in the boundary conditions removes the spurious
    reflection of a plane wave at the ends of the slab. Positive imaginary part when
    evanescent.
    """
    c = h**2 / 12
    cosine = (1 - 5 * c * kz2) / (1 + c * kz2)
    propagating = np.abs(cosine) <= 1.0
    with np.errstate(invalid="ignore"):
        phase = np.where(
            propagating,
            np.arccos(np.clip(cosine, -1.0, 1.0)),
            1j * np.arccosh(np.where(cosine > 1.0, cosine, 1.0)),
        )
    return phase / h


def _solve_blocks(
    coupling, kz2, kappa, h, nz, incident, xp=np, dtype=np.complex128, source=None
):
    """Solve the Numerov equations for waves incident from below (z < 0) on the slab,
    and for sources inside it.

    ``coupling(n)`` is the coupling matrix of slice n (an array of ``xp``), ``kz2`` the
    squared longitudinal wave numbers of the beams in vacuum, ``kappa`` the wave numbers
    of the Numerov recurrence in vacuum (:func:`_numerov_wave_numbers`), ``incident``
    the amplitudes of the incident beams at z = 0 on the host: (n_beams,), or
    (n_beams, m) for m right-hand sides that share the factorisation of the blocks.
    ``source`` is the source S_n of the equation phi'' + (kz2 + U) phi = -S at the slice
    centres: an array (nz, n_beams, m), or a function n -> (n_beams, m) that is called
    once for every n in order (so that the sources need not be stored); a one-column
    ``incident`` is shared by all of its m columns. Returns the amplitudes phi (nz,
    n_beams), or (nz, n_beams, m) with a source or several columns, of the complex type
    ``dtype`` and the array module ``xp``, at the slice centres z_n = (n + 1/2) h, in
    the block-Thomas solve with checkpointing.
    """
    solve = _linear_solver(xp)
    nb = len(kz2)
    incident = np.asarray(incident)
    squeeze = incident.ndim == 1 and source is None
    if incident.ndim == 1:
        incident = incident[:, None]
    get_source = (lambda n: source[n]) if source is not None and not callable(
        source
    ) else source
    columns = incident.shape[1]
    first_source = None
    if get_source is not None:
        first_source = xp.asarray(get_source(0)).astype(dtype)
        if first_source.ndim != 2 or first_source.shape[0] != nb:
            raise ValueError("the source must have the shape (nz, n_beams, m)")
        if columns == 1:
            columns = first_source.shape[1]
        elif first_source.shape[1] != columns:
            raise ValueError("the source and the incident amplitudes differ in m")
    incident = np.broadcast_to(incident, (nb, columns))

    identity = xp.eye(nb, dtype=dtype)
    c = h**2 / 12
    outgoing = xp.asarray((np.exp(1j * kappa * h)).astype(dtype))
    base = xp.diag(xp.asarray(kz2.astype(dtype)))

    def operator(n):
        return base if n < 0 or n >= nz else base + coupling(n)

    vacuum_boundary = identity + c * operator(-1)
    z0 = h / 2
    incident_at_first = xp.asarray(
        (incident * np.exp(1j * kappa * z0)[:, None]).astype(dtype)
    )
    entrance = xp.asarray(
        (np.exp(-1j * kappa * h) - np.exp(1j * kappa * h)).astype(dtype)
    )
    cached = {0: first_source}

    def source_at(n):
        """The source at slice n, zero outside the slab."""
        if n < 0 or n >= nz:
            return 0.0
        if n not in cached:
            cached[n] = xp.asarray(get_source(n)).astype(dtype)
        return cached[n]

    def rhs(n):
        """The right-hand side of slice n: the sources of its neighbours (the
        Numerov average) and, at the entrance, the incident beams."""
        right = xp.zeros((nb, columns), dtype)
        if get_source is not None:
            neighbours = source_at(n - 1) + 10 * source_at(n) + source_at(n + 1)
            right = right - c * neighbours
            cached.pop(n - 1, None)
        if n == 0:
            right = right - vacuum_boundary @ (incident_at_first * entrance[:, None])
        return right

    def blocks(n):
        diagonal = -2 * (identity - 5 * c * operator(n))
        if n == 0 or n == nz - 1:
            diagonal = diagonal + vacuum_boundary * outgoing[None, :]
        upper = identity + c * operator(n + 1) if n < nz - 1 else None
        lower = identity + c * operator(n - 1) if n > 0 else None
        return diagonal, upper, lower

    def eliminate(n, previous, right_hand_side=None):
        """The elimination block of slice n and, with a right-hand side (a function of
        the lower block), its solution after the elimination of the slices before it,
        from one factorisation."""
        diagonal, upper, lower = blocks(n)
        if n > 0:
            diagonal = diagonal - lower @ previous
        right = None if right_hand_side is None else right_hand_side(lower)
        if upper is None:
            return solve(diagonal, right), lower, None
        if right is None:
            return None, lower, solve(diagonal, upper)
        both = solve(diagonal, xp.concatenate([upper, right], axis=1))
        return both[:, nb:], lower, both[:, :nb]

    itemsize = np.dtype(dtype).itemsize
    full_bytes = nz * nb * nb * itemsize
    stride = (
        nz
        if full_bytes < _memory_budget(xp, _CHECKPOINT_BYTES)
        else int(np.ceil(np.sqrt(nz)))
    )

    # the forward sweep, whose result is overwritten by the back substitution
    phi = xp.empty((nz, nb, columns), dtype)
    stored, checkpoints = {}, {}
    previous = None
    for n in range(nz):
        solved, _, eliminated = eliminate(
            n,
            previous,
            lambda lower, n=n: rhs(n) if n == 0 else rhs(n) - lower @ phi[n - 1],
        )
        phi[n] = solved
        if stride == nz:
            stored[n] = eliminated
        elif n % stride == 0:
            checkpoints[n] = eliminated
        previous = eliminated

    if stride == nz:
        for n in range(nz - 2, -1, -1):
            phi[n] -= stored[n] @ phi[n + 1]
    else:
        for start in sorted(checkpoints, reverse=True):
            end = min(start + stride, nz - 1)
            segment = {start: checkpoints[start]}
            for n in range(start + 1, end):
                segment[n] = eliminate(n, segment[n - 1])[2]
            for n in range(end - 1, start - 1, -1):
                phi[n] -= segment[n] @ phi[n + 1]
    return phi[..., 0] if squeeze else phi


def _tilt_to_offset(tilt, energy):
    """The in-plane wave vector k_t [1/Å] of a wave of this tilt [mrad] and energy [eV]
    that travels along (tan x, tan y, 1)."""
    tangent = np.tan(np.asarray(tilt, dtype=float) / 1e3)
    return tangent / (energy2wavelength(energy) * np.sqrt(1.0 + np.sum(tangent**2)))


def _fourier_coefficients(projected, h, sampling, origin, xp, dtype):
    """The Fourier coefficients (nz, nx, ny) of the potential per unit length, the
    projected potential over the slice thickness h, with the origin of the real space
    moved to ``origin`` [Å] (the symmetry axis)."""
    nz, nx, ny = projected.shape
    fx = np.fft.fftfreq(nx, sampling[0])[:, None]
    fy = np.fft.fftfreq(ny, sampling[1])[None, :]
    shift = np.exp(2j * np.pi * (fx * origin[0] + fy * origin[1]))
    fourier = xp.fft.fft2(projected / h, axes=(1, 2)) / (nx * ny)
    return (fourier * xp.asarray(shift.astype(dtype))[None]).astype(dtype)


def _check_offset(h, sampling, cutoff, offset, energy):
    """Raise for an energy [eV] and an in-plane offset [1/Å] that the slices or the
    sampling of the potential cannot resolve; it is cheap, so the solver runs it
    before it builds the potential."""
    _check_step(h, 2 * np.pi / energy2wavelength(energy), cutoff)
    if 1 / (2 * max(sampling)) < 2 * cutoff + np.hypot(*offset):
        raise ValueError(
            f"The potential is sampled too coarsely for cutoff={cutoff}: "
            f"use a sampling of at most "
            f"{1 / (4 * cutoff + 2 * np.hypot(*offset)):.4f} Å or a "
            "lower cutoff."
        )


def _ensemble_members(waves):
    """The total tilt [mrad] (..., 2) of every member of the ensemble of waves, the base
    tilt plus the value of each tilt axis, and its energy [eV] (...): that of the energy
    axis (or the energy of the waves), broadcast over the ensemble axes in their
    order."""
    axes = waves.ensemble_axes_metadata
    ensemble_shape = waves.shape[:-2]
    tilts = np.broadcast_to(
        np.asarray(waves.base_tilt, dtype=float), ensemble_shape + (2,)
    ).copy()
    energies = np.full(ensemble_shape, np.nan)
    if waves.energy is not None:
        energies[...] = waves.energy
    for i, axis in enumerate(axes):
        if isinstance(axis, EnergyAxis):
            shape = [1] * len(axes)
            shape[i] = len(axis.values)
            energies = np.broadcast_to(
                np.asarray(axis.values, dtype=float).reshape(shape), ensemble_shape
            ).copy()
        else:
            values = np.asarray(axis.tilt, dtype=float).reshape(-1, 2)
            shape = [1] * len(axes) + [2]
            shape[i] = len(values)
            tilts += values.reshape(shape)
    return tilts, energies


def beam_flux(waves: Waves) -> np.ndarray:
    """The flux of every beam of the waves of :meth:`TwoWaySolver.scatter`, per unit
    flux of the incident beam: ``|a_g|^2 kz_g / kz_incident``, with the amplitude
    ``a_g`` of the beam and its longitudinal wave number ``kz_g``, zero for evanescent
    beams.

    The squared amplitude of a beam is not its flux: the beams of a wave that leaves the
    slab at a large angle carry less than their intensity (kz / k of it), and the
    evanescent ones none. The flux is what the solver conserves: the sums of the
    transmitted and the reflected flux add up to 1 without absorption. Use it, not
    ``|a_g|^2``, for the intensity of the beams that are reflected and transmitted.

    Parameters
    ----------
    waves : Waves
        The transmitted or the reflected waves of :meth:`TwoWaySolver.scatter`, with
        their ensembles of tilts and energies.

    Returns
    -------
    ndarray
        The flux of every beam, (..., nx, ny), indexed like the FFT of the waves (the
        beam g of the stored wave is the plane wave g + k_t of a tilted incident wave).
    """
    array = np.asarray(asnumpy(waves.array))
    nx, ny = array.shape[-2:]
    sampling = tuple(waves._valid_sampling)
    tilts, energies = _ensemble_members(waves)
    amplitudes = np.fft.fft2(array) / (nx * ny)
    flux = np.empty(array.shape, dtype=np.float64)
    for index in np.ndindex(*array.shape[:-2]):
        wavelength = energy2wavelength(energies[index])
        k_t = _tilt_to_offset(tilts[index], energies[index])
        tangent = np.tan(np.asarray(tilts[index], dtype=float) / 1e3)
        incident_kz = 2 * np.pi / (wavelength * np.sqrt(1.0 + np.sum(tangent**2)))
        qx = np.fft.fftfreq(nx, sampling[0])[:, None] + k_t[0]
        qy = np.fft.fftfreq(ny, sampling[1])[None, :] + k_t[1]
        kz2 = (2 * np.pi / wavelength) ** 2 - 4 * np.pi**2 * (qx**2 + qy**2)
        weights = np.sqrt(np.maximum(kz2, 0.0)) / incident_kz
        flux[index] = np.abs(amplitudes[index]) ** 2 * weights
    return flux


def _check_member(h, sampling, cutoff, tilt, energy):
    """:func:`_check_offset` for a member of an ensemble (tilt [mrad], energy [eV])."""
    _check_offset(h, sampling, cutoff, _tilt_to_offset(tilt, energy), energy)


def _check_step(h, wave_number, cutoff):
    """The Numerov step must resolve the oscillation of the fastest beam and the
    decay of the most evanescent one: the phase error per step grows like
    (kappa h)^6."""
    evanescent = (
        2 * np.pi * np.sqrt(max(cutoff**2 - (wave_number / (2 * np.pi)) ** 2, 0.0))
    )
    largest = max(wave_number, evanescent)
    if largest * h > 0.6:
        raise ValueError(
            f"The slices of {h:.4f} Å are too thick for the Numerov step: the largest "
            f"longitudinal wave number {largest:.1f} rad/Å times the thickness must "
            f"stay below 0.6 (below 0.3 for converged results); use slices of at most "
            f"{0.3 / largest:.4f} Å."
        )
    if largest * h > 0.3:
        warnings.warn(
            f"The slices of {h:.4f} Å are thick for the Numerov step (wave number "
            f"times thickness {largest * h:.2f}); check the convergence by halving the "
            "thickness.",
            stacklevel=3,
        )


@dataclass(frozen=True)
class Beams:
    """The beams of the solver for one energy and in-plane offset: the plane waves
    ``exp(2 pi i (q + g) . r)`` of the reciprocal lattice of the grid with
    ``|q + g| < cutoff``, evanescent ones included.

    Attributes
    ----------
    indices : ndarray of int, shape (n, 2)
        Indices of every beam in the (FFT-ordered) grid of the potential.
    qx, qy : ndarray, shape (n,)
        In-plane wave vector q + g of every beam [1/Å].
    kz : ndarray of complex, shape (n,)
        Longitudinal wave number [rad/Å] of every beam in vacuum, ``sqrt(k^2 - 4 pi^2
        |q + g|^2)`` with a positive imaginary part for the evanescent beams.
    wave_number : float
        The wave number k = 2 pi / wavelength of the electron [rad/Å].
    energy : float
        The energy of the electron [eV].
    gpts : tuple of int
        The grid of the potential.
    """

    indices: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    kz: np.ndarray
    wave_number: float
    energy: float
    gpts: tuple[int, int]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def propagating(self) -> np.ndarray:
        """Whether each beam propagates in vacuum (it is not evanescent)."""
        return self.kz.imag == 0.0

    def flux_weights(self, reference_energy: float | None = None) -> np.ndarray:
        """The flux of every beam per unit squared amplitude, ``kz / k``: zero for the
        evanescent beams. ``k`` is the wave number of the electron of
        ``reference_energy`` [eV], by default that of the beams: pass the energy of the
        incident electron when the beams belong to an electron that has lost energy,
        to measure their flux per unit incident flux."""
        reference = self.wave_number
        if reference_energy is not None:
            reference = 2 * np.pi / energy2wavelength(reference_energy)
        return np.where(self.propagating, self.kz.real, 0.0) / reference

    def index(self, i: int, j: int) -> int:
        """Position in the beams of the one with the grid index (i, j); the beam of the
        wave itself (g = 0) is ``index(0, 0)``."""
        match = np.flatnonzero(
            (self.indices[:, 0] == i % self.gpts[0])
            & (self.indices[:, 1] == j % self.gpts[1])
        )
        if len(match) == 0:
            raise ValueError(f"the beam ({i}, {j}) is outside the cutoff")
        return int(match[0])

    def to_grid(self, amplitudes: np.ndarray) -> np.ndarray:
        """Beam amplitudes (..., n) as an array (..., nx, ny) of the grid of the
        potential, zero outside the beams."""
        amplitudes = np.asarray(amplitudes)
        grid = np.zeros(amplitudes.shape[:-1] + tuple(self.gpts), amplitudes.dtype)
        grid[..., self.indices[:, 0], self.indices[:, 1]] = amplitudes
        return grid

    def from_grid(self, array: np.ndarray) -> np.ndarray:
        """The amplitudes (..., n) of the beams in an array (..., nx, ny)."""
        return np.asarray(array)[..., self.indices[:, 0], self.indices[:, 1]]


@dataclass(frozen=True)
class BeamSolution:
    """The solution of :meth:`TwoWaySolver.solve`, per beam and per column ``m`` of the
    incident amplitudes or the sources.

    Attributes
    ----------
    beams : Beams
        The beams of the amplitudes.
    z : ndarray, shape (nz,)
        The slice centres [Å] from the entrance plane of the slab, z = 0.
    phi : ndarray of complex, shape (nz, n, m), optional
        The amplitude of every beam at every slice centre (the interior wave), when it
        was asked for; ``phi[n, g, i]`` is the coefficient of the beam g, the plane
        wave ``exp(2 pi i (q + g) . r)``, in column i at ``z[n]``.
    reflected : ndarray of complex, shape (n, m)
        The wave leaving the slab on the entrance side, extrapolated to z = 0 in vacuum.
    transmitted : ndarray of complex, shape (n, m)
        The wave leaving the slab on the far side, extrapolated to its exit plane
        z = nz h. In contrast to :meth:`TwoWaySolver.scatter`, neither carries a
        vacuum phase that has been removed: the beams have their own phases from z = 0.
    """

    beams: Beams
    z: np.ndarray
    phi: np.ndarray | None
    reflected: np.ndarray
    transmitted: np.ndarray

    def reflected_flux(self, reference_energy: float | None = None) -> np.ndarray:
        """The flux (n, m) of the reflected beams, ``|amplitude|^2 kz / k``, zero for
        the evanescent beams; see :meth:`Beams.flux_weights` for the reference. The
        squared amplitudes are not the intensity that leaves the slab: at large angles
        they overestimate it."""
        weights = self.beams.flux_weights(reference_energy)
        return weights[:, None] * np.abs(self.reflected) ** 2

    def transmitted_flux(self, reference_energy: float | None = None) -> np.ndarray:
        """The flux (n, m) of the transmitted beams, as :meth:`reflected_flux`."""
        weights = self.beams.flux_weights(reference_energy)
        return weights[:, None] * np.abs(self.transmitted) ** 2


@dataclass(frozen=True)
class MeanFreePathAbsorber:
    """An absorbing layer given by an inelastic mean free path: a constant imaginary
    potential ``V_i = 1 / (2 sigma mean_free_path)`` [V] in a range of z, so that the
    intensity of the elastic wave decays as ``exp(-z / mean_free_path)`` there. Pass it
    to :class:`TwoWaySolver` as ``absorber``.

    The imaginary potential depends on the energy through the interaction constant
    sigma, and a mean free path that depends on the energy is a function: in an ensemble
    of energies each member has its own. In the wave equation the absorber adds ``i k /
    mean_free_path`` to the squared longitudinal wave number of every beam in the range,
    to which the interaction constant does not contribute.

    Attributes
    ----------
    mean_free_path : float or callable
        The inelastic mean free path [Å], or a function of the energy [eV] that returns
        it.
    z_range : tuple of float, optional
        The range (z_min, z_max) [Å], measured from the entrance plane of the slab, of
        the slices that absorb: those with their centre in it. By default the whole
        slab.
    """

    mean_free_path: float | Callable[[float], float]
    z_range: tuple[float, float] | None = None

    def __post_init__(self):
        if not callable(self.mean_free_path) and not self.mean_free_path > 0.0:
            raise ValueError("mean_free_path must be positive")
        if self.z_range is not None and not self.z_range[0] < self.z_range[1]:
            raise ValueError("z_range must be (z_min, z_max) with z_min < z_max")

    def _mean_free_path(self, energy: float) -> float:
        value = (
            self.mean_free_path(energy)
            if callable(self.mean_free_path)
            else self.mean_free_path
        )
        if not value > 0.0:
            raise ValueError(f"mean free path {value} Å at {energy} eV is not positive")
        return float(value)

    def slices(self, nz: int, thickness: float) -> np.ndarray:
        """Whether each of the ``nz`` slices of this thickness [Å] absorbs, (nz,)."""
        if self.z_range is None:
            return np.ones(nz, dtype=bool)
        centres = (np.arange(nz) + 0.5) * thickness
        return (centres >= self.z_range[0]) & (centres <= self.z_range[1])

    def imaginary_potential(self, energy: float) -> float:
        """The imaginary potential ``V_i = 1 / (2 sigma mean_free_path)`` [V] at this
        energy [eV]."""
        return 1.0 / (2.0 * energy2sigma(energy) * self._mean_free_path(energy))

    def absorptive_potential(self, potential: BasePotential, energy: float):
        """The absorber of one energy as the ``absorptive_potential`` of
        :meth:`TwoWaySolver.scatter`: a :class:`.PotentialArray` of ``V_i`` times the
        slice thickness in the slices that absorb, on the grid of the potential."""
        thicknesses = np.asarray(potential.slice_thickness, dtype=float)
        gpts = tuple(int(n) for n in potential.gpts)
        mask = self.slices(len(thicknesses), float(thicknesses.mean()))
        array = (
            np.zeros((len(thicknesses),) + gpts)
            + (self.imaginary_potential(energy) * thicknesses * mask)[:, None, None]
        )
        return PotentialArray(
            array, slice_thickness=thicknesses, extent=tuple(potential.extent)
        )


def _with_absorber(coupling, absorber, energy, h, nz, size, xp, dtype):
    """The coupling of the slices with the absorber's ``i k / mean_free_path`` on the
    diagonal in the slices that absorb."""
    if absorber is None:
        return coupling
    wave_number = 2 * np.pi / energy2wavelength(energy)
    damping = np.where(
        absorber.slices(nz, h), 1j * wave_number / absorber._mean_free_path(energy), 0.0
    ).astype(dtype)
    damping = xp.asarray(damping)
    identity = xp.eye(size, dtype=dtype)
    return lambda n: coupling(n) + damping[n] * identity


@dataclass(frozen=True)
class TwoWaySolver:
    """
    Exact two-way solution of the elastic scattering of a plane wave by a periodic slab.

    The solution includes backscattering and the evanescent beams up to the cutoff,
    which at low energies carry most of the reflection (graphene at 200 eV reflects 7 %,
    almost all of it through evanescent beams) and change the transmitted beams by tens
    of percent. The cost grows with the cube of the number of beams: use it for the
    small cells of crystals and 2D materials. A plane wave of any direction is
    supported, with ensembles of tilts and energies (the members are solved one after
    the other), on the CPU and on CUDA devices, in single or double precision, with or
    without absorption.

    The potential must be given on slices of equal thickness and consist of finite
    projections, the slices thin enough to resolve the cores (about 0.01 to 0.02 Å;
    check by halving) and sampled finely enough for the cutoff: 1 / (2 sampling) at
    least 2 cutoff. The slab must start and end in vacuum. The phases of the beams carry
    the dispersion error of the Numerov step inside the material (the part in vacuum is
    corrected), which falls with the fourth power of the step; intensities converge
    faster.

    Parameters
    ----------
    cutoff : float, optional
        Beams with in-plane frequency below this value [1/Å] are included (default 4.5).
        Beams above 1 / wavelength are evanescent. Pass ``0.95 / wavelength`` for the
        propagating beams only.
    symmetry : bool, optional
        Reduce the basis by the translation, point and glide symmetry of the structure,
        found from the atoms of the potential (default True). A tilted wave keeps the
        translations and the point operations that leave its in-plane wave vector
        unchanged. The results agree with the full
        basis to the asymmetry of the sampled potential; a distorted structure is not
        symmetrised. A built :class:`.PotentialArray` carries no atoms and is solved in
        the full basis.
    symmetry_tolerance : float, optional
        Largest displacement [Å] by which a symmetry operation may miss an atom (default
        1e-4).
    workers : int, optional
        The number of members of an ensemble of tilts and energies that are solved at
        the same time on the CPU, each with its share of the BLAS threads (default: a
        quarter of the BLAS threads, at most the number of members; one thread solves
        them one after another). The memory that the solves keep (the coupling and the
        elimination blocks, up to 6 GB each) is shared: a member waits until the others
        leave room for it. A CUDA device solves the members one after another. Use 1
        for no parallelism.
    absorber : MeanFreePathAbsorber, optional
        An absorbing layer from an inelastic mean free path, which adds to the
        absorption of the other two options (default none). Its imaginary potential
        follows the energy of each member of an ensemble of energies.
    absorption : float, optional
        The imaginary part of the potential as a fraction of the real part (default 0,
        no absorption): the solver uses ``V (1 + i absorption)``, so that the
        transmitted and the reflected flux add up to less than 1 by what is absorbed.
        For a different shape of the absorption give ``absorptive_potential`` to
        :meth:`scatter`, which adds to it.
    precision : {"float32", "float64"}, optional
        Precision of the solve (complex64 or complex128). By default it is the
        ``precision`` of the configuration, as for the rest of abTEM. The evanescent
        beams make the blocks ill-conditioned, but in single precision the strong beams
        are still accurate to about 1e-4 (up to 1e-3 for thick specimens of strong
        scatterers at 30 keV, which take thousands of steps) and the reflected wave to
        1e-5; use double precision for a reference.
    """

    cutoff: float = 4.5
    symmetry: bool = True
    symmetry_tolerance: float = 1e-4
    precision: str | None = None
    absorption: float = 0.0
    absorber: MeanFreePathAbsorber | None = None
    workers: int | None = None

    def __post_init__(self):
        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if self.absorption < 0.0:
            raise ValueError("absorption must not be negative")
        if self.workers is not None and self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.absorber is not None and not isinstance(
            self.absorber, MeanFreePathAbsorber
        ):
            raise TypeError("absorber must be a MeanFreePathAbsorber")
        if self.precision not in (None, "float32", "float64"):
            raise ValueError(
                "precision must be 'float32', 'float64' or None, "
                f"got {self.precision!r}"
            )

    def _prepare(self, potential, absorptive_potential, thicknesses):
        """The array module, the complex type and the projected potential (nz, nx, ny)
        of the solve, complex with the absorption (``absorption`` and the
        ``absorptive_potential``) as its imaginary part."""
        built = (
            potential
            if isinstance(potential, PotentialArray)
            else potential.build(lazy=False)
        )
        device = built.device
        if device == "mps":
            raise NotImplementedError(
                "TwoWaySolver does not run on Metal (no complex linear algebra); use "
                "the CPU or a CUDA device."
            )
        xp = get_array_module(device)
        precision = self.precision or config.get("precision")
        real_dtype = np.float32 if precision == "float32" else np.float64
        complex_dtype = np.complex64 if precision == "float32" else np.complex128
        projected = xp.asarray(built.array, dtype=real_dtype)
        imaginary = self.absorption * projected
        if absorptive_potential is not None:
            absorbing = (
                absorptive_potential
                if isinstance(absorptive_potential, PotentialArray)
                else absorptive_potential.build(lazy=False)
            )
            if absorbing.array.shape != projected.shape or not np.allclose(
                absorbing.slice_thickness, thicknesses
            ):
                raise ValueError(
                    "the absorptive potential must have the grid and the slices of the "
                    "potential"
                )
            absorbing_array = xp.asarray(absorbing.array, dtype=real_dtype)
            if absorbing_array.min() < -1e-6 * max(float(absorbing_array.max()), 1e-30):
                raise ValueError("the absorptive potential must not be negative")
            imaginary = imaginary + absorbing_array
        if self.absorption or absorptive_potential is not None:
            projected = projected + 1j * imaginary
        return xp, complex_dtype, projected

    def beams(
        self,
        potential: BasePotential,
        energy: float,
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> Beams:
        """The beams of :meth:`solve` for a potential, an energy [eV] and an in-plane
        offset [1/Å] (the wave vector of the beam g = 0), in the order of the
        amplitudes of its ``incident`` and ``source``. The potential is not built."""
        return self._beams(potential, energy, offset)[0]

    def _beams(self, potential, energy, offset):
        """The :class:`Beams` and the basis of the full set of them."""
        gpts = tuple(int(n) for n in potential.gpts)
        sampling = tuple(float(x) for x in potential.sampling)
        offset = tuple(float(x) for x in offset)
        basis = _BeamBasis(
            gpts,
            sampling,
            self.cutoff,
            (0.0, 0.0),
            [("rot0", np.eye(2), np.zeros(2))],
            [],
            shift=offset,
        )
        wave_number = 2 * np.pi / energy2wavelength(energy)
        kz2 = wave_number**2 - 4 * np.pi**2 * (basis.qx**2 + basis.qy**2)
        beams = Beams(
            indices=basis.indices,
            qx=basis.qx,
            qy=basis.qy,
            kz=np.sqrt(kz2.astype(np.complex128)),
            wave_number=float(wave_number),
            energy=float(energy),
            gpts=gpts,
        )
        return beams, basis, kz2

    def solve(
        self,
        potential: BasePotential,
        energy: float,
        offset: tuple[float, float] = (0.0, 0.0),
        incident: np.ndarray | None = None,
        source=None,
        absorptive_potential: BasePotential | None = None,
        interior: bool = False,
    ) -> BeamSolution:
        """
        Solve the scattering by the slab for any beams incident on it and any sources
        in it, at one energy and in-plane offset, in the full beam basis.

        This is the lower-level counterpart of :meth:`scatter`. It solves
        ``phi'' + (k_z^2 + U) phi = -S`` for the amplitudes of the beams, ``U = 2 k
        sigma V``, with outgoing waves in the vacuum on both sides, for ``m`` right-hand
        sides at once with one factorisation of the blocks. A source has no incident
        direction and breaks the symmetries of the structure, so the basis is never
        reduced by symmetry (``symmetry`` is ignored), and the offset is any in-plane
        wave vector, not that of a tilt.

        Parameters
        ----------
        potential : BasePotential
            The slab, as for :meth:`scatter`, with slices of equal thickness.
        energy : float
            Energy of the electron [eV]. For an ensemble of energies call this for each
            energy.
        offset : tuple of float, optional
            In-plane wave vector q [1/Å] of the beam g = 0: the beam g is the plane wave
            ``exp(2 pi i (q + g) . r)`` (default 0).
        incident : ndarray, optional
            Amplitudes at z = 0 of the beams incident from z < 0, evanescent ones
            included, (n,) or (n, m), in the order of :meth:`beams`. A beam with an
            imaginary ``kz`` decays from the plane z = 0 into the slab. By default, the
            beam g = 0 with unit amplitude when there is no ``source``, else none.
        source : ndarray or callable, optional
            The source S inside the slab, as the amplitudes of the beams at the slice
            centres: an array (nz, n, m), or a function of the slice index that returns
            (n, m) and is called once per slice, in order, which avoids storing all of
            them. A one-column ``incident`` is shared by all m columns of the sources.
        absorptive_potential : BasePotential, optional
            Imaginary part of the potential, as for :meth:`scatter`.
        interior : bool, optional
            Keep the amplitudes ``phi`` of every beam at every slice centre (the
            interior wave) in the result (default False).

        Returns
        -------
        BeamSolution
            The reflected and transmitted amplitudes and, with ``interior``, ``phi``.
        """
        thicknesses = np.asarray(potential.slice_thickness, dtype=float)
        h = float(thicknesses.mean())
        if not np.allclose(thicknesses, h, rtol=1e-6):
            raise ValueError("TwoWaySolver needs slices of equal thickness.")
        if potential.ensemble_shape:
            raise NotImplementedError(
                "TwoWaySolver does not support ensemble potentials."
            )
        offset = tuple(float(x) for x in offset)
        sampling = tuple(float(x) for x in potential.sampling)
        _check_offset(h, sampling, self.cutoff, offset, energy)
        beams, basis, kz2 = self._beams(potential, energy, offset)
        nb = len(beams)

        if incident is None:
            incident = np.zeros(nb, np.complex128)
            if source is None:
                incident[beams.index(0, 0)] = 1.0
        incident = np.asarray(incident, dtype=np.complex128)
        if incident.shape[0] != nb or incident.ndim not in (1, 2):
            raise ValueError(
                f"incident must have the shape ({nb},) or ({nb}, m), got "
                f"{incident.shape}"
            )

        xp, complex_dtype, projected = self._prepare(
            potential, absorptive_potential, thicknesses
        )
        nz, nx, ny = projected.shape
        sigma = energy2sigma(energy)
        fourier = _fourier_coefficients(
            projected, h, sampling, (0.0, 0.0), xp, complex_dtype
        )
        coupling = basis.coupling_function(fourier, beams.wave_number, sigma, xp)
        itemsize = np.dtype(complex_dtype).itemsize
        if nz * nb**2 * itemsize < _memory_budget(xp, _COUPLING_CACHE_BYTES):
            cache = xp.stack([coupling(n) for n in range(nz)])
            coupling = lambda n: cache[n]  # noqa: E731
        kappa = _numerov_wave_numbers(kz2, h)
        coupling = _with_absorber(
            coupling, self.absorber, energy, h, nz, nb, xp, complex_dtype
        )
        if source is not None and not callable(source):
            if not hasattr(source, "shape"):
                source = np.asarray(source)
            if source.ndim != 3 or source.shape[:2] != (nz, nb):
                raise ValueError(
                    f"source must have the shape ({nz}, {nb}, m), got {source.shape}"
                )

        phi = _solve_blocks(
            coupling, kz2, kappa, h, nz, incident, xp, complex_dtype, source=source
        )
        phi = asnumpy(phi).astype(np.complex128, copy=False)
        if phi.ndim == 2:
            phi = phi[..., None]
        half = np.exp(1j * kappa * h / 2)[:, None]
        columns = phi.shape[2]
        shared = np.broadcast_to(
            incident if incident.ndim == 2 else incident[:, None], (nb, columns)
        )
        return BeamSolution(
            beams=beams,
            z=(np.arange(nz) + 0.5) * h,
            phi=phi if interior else None,
            reflected=(phi[0] - shared * half) * half,
            transmitted=phi[-1] * half,
        )

    def scatter(
        self,
        plane_wave: PlaneWave,
        potential: BasePotential,
        absorptive_potential: BasePotential | None = None,
    ) -> tuple[Waves, Waves]:
        """
        Scatter a plane wave from the slab.

        Parameters
        ----------
        plane_wave : PlaneWave
            The incident wave. Its energy and its tilt, or its ensembles of energies and
            of tilts, are used, and its grid is matched to the potential. A tilted wave
            travels along (tan x, tan y, 1) and, as in multislice, is stored without the
            carrier of its direction: the beam g of the result is the plane wave g +
            k_t, the incident one is g = 0, and the result carries the tilt in its
            metadata.
        potential : BasePotential
            The slab. A :class:`.Potential` or a built :class:`.PotentialArray`, without
            an ensemble axis.
        absorptive_potential : BasePotential, optional
            The imaginary part of the potential, of the same atoms, grid and slices as
            ``potential`` and nowhere negative: the potential of the solver is
            ``V (1 + i absorption) + i V_absorptive``. See also ``absorption``.

        Returns
        -------
        transmitted, reflected : Waves
            The wave leaving the slab on the far side (after the last slice) and the
            wave leaving on the near side (at the entrance plane), both with the beams
            of the solution and the phase convention of multislice: the vacuum phase
            over the slab's thickness is removed, so an empty slab gives a transmitted
            wave of 1 and no reflected wave. The squared amplitude of a beam is its
            intensity, not its flux (see :func:`beam_flux`).
        """
        from abtem.waves import Waves

        waves = plane_wave.match_grid(potential).build(lazy=False)
        if any(
            not (hasattr(axis, "tilt") or isinstance(axis, EnergyAxis))
            for axis in waves.ensemble_axes_metadata
        ):
            raise NotImplementedError(
                "TwoWaySolver supports ensembles of tilts and energies only, not of "
                "other parameters."
            )
        if potential.ensemble_shape:
            raise NotImplementedError(
                "TwoWaySolver does not support ensemble potentials."
            )

        ensemble_shape = waves.shape[:-2]
        tilts, energies = _ensemble_members(waves)

        thicknesses = np.asarray(potential.slice_thickness, dtype=float)
        h = float(thicknesses.mean())
        if not np.allclose(thicknesses, h, rtol=1e-6):
            raise ValueError("TwoWaySolver needs slices of equal thickness.")
        sampling = tuple(waves._valid_sampling)
        for tilt, energy in zip(tilts.reshape(-1, 2), energies.ravel()):
            _check_member(h, sampling, self.cutoff, tilt, energy)

        xp, complex_dtype, projected = self._prepare(
            potential, absorptive_potential, thicknesses
        )
        nz, nx, ny = projected.shape

        origin, all_operations, translations = (
            (0.0, 0.0),
            [("rot0", np.eye(2), np.zeros(2))],
            [],
        )
        if self.symmetry:
            try:
                atoms = potential.frozen_phonons.atoms  # type: ignore[attr-defined]
            except AttributeError:
                atoms = None
            other = getattr(
                getattr(absorptive_potential, "frozen_phonons", None), "atoms", None
            )
            if (
                atoms is not None
                and other is not None
                and (
                    len(atoms) != len(other)
                    or not np.array_equal(atoms.numbers, other.numbers)
                    or not np.allclose(atoms.positions, other.positions, atol=1e-6)
                )
            ):
                raise ValueError(
                    "the absorptive potential must be of the same atoms as the "
                    "potential"
                )
            if atoms is not None and len(atoms) > 0:
                origin, point_operations, translations = structure_symmetry(
                    atoms, self.symmetry_tolerance
                )
                all_operations = [
                    (name, matrix, np.zeros(2)) for name, matrix in point_operations
                ] + structure_glides(
                    atoms,
                    origin,
                    point_operations,
                    translations,
                    self.symmetry_tolerance,
                )

        fourier = _fourier_coefficients(
            projected, h, sampling, origin, xp, complex_dtype
        )

        def solve_member(tilt, energy, reserve):
            """The transmitted and the reflected wave [arrays of the grid, on the host]
            of the wave of this tilt [mrad] and energy [eV]; ``reserve(nbytes)`` holds
            the memory that the solve keeps until the member is done."""
            wavelength = energy2wavelength(energy)
            wave_number = 2 * np.pi / wavelength
            sigma = energy2sigma(energy)

            # The incident wave travels along (tan x, tan y, 1), and is stored without
            # its carrier exp(2 pi i (k_t . r + k_zt z)), as multislice does: the stored
            # beam g is the plane wave g + k_t, and the incident one is g = 0.
            tangent = np.tan(np.asarray(tilt, dtype=float) / 1e3)
            norm = np.sqrt(1.0 + np.sum(tangent**2))
            in_plane = tuple(_tilt_to_offset(tilt, energy))  # k_t [1/Å]
            incident_kz = 2 * np.pi / (wavelength * norm)

            # a tilted wave is invariant under the translations and under the point
            # operations that leave its in-plane wave vector unchanged, the mirror along
            # its direction, for example
            operations = [
                (name, matrix, translation)
                for name, matrix, translation in all_operations
                if np.allclose(matrix @ np.array(in_plane), in_plane, atol=1e-6)
            ]
            basis = _BeamBasis(
                (nx, ny),
                sampling,
                self.cutoff,
                origin,
                operations,
                translations,
                shift=in_plane,
            )
            itemsize = np.dtype(complex_dtype).itemsize
            reserve(
                _member_bytes(
                    nz,
                    len(basis.representatives) if basis.reduced else len(basis.indices),
                    itemsize,
                )
            )
            coupling = basis.coupling_function(fourier, wave_number, sigma, xp)
            if nz * basis.number_of_orbits**2 * itemsize < _memory_budget(
                xp, _COUPLING_CACHE_BYTES
            ):
                cache = xp.stack([coupling(n) for n in range(nz)])
                coupling = lambda n: cache[n]  # noqa: E731

            qx, qy = basis.qx, basis.qy
            if basis.reduced:
                qx, qy = qx[basis.representatives], qy[basis.representatives]
            kz2 = wave_number**2 - 4 * np.pi**2 * (qx**2 + qy**2)
            exact = np.sqrt(kz2.astype(np.complex128))
            kappa = _numerov_wave_numbers(kz2, h)
            coupling = _with_absorber(
                coupling, self.absorber, energy, h, nz, len(kz2), xp, complex_dtype
            )
            incident = np.zeros(len(kz2), np.complex128)
            incident[int(np.argmin(np.abs(exact - incident_kz)))] = 1.0

            phi = asnumpy(
                _solve_blocks(coupling, kz2, kappa, h, nz, incident, xp, complex_dtype)
            ).astype(np.complex128)
            half = np.exp(1j * kappa * h / 2)
            reflected = (phi[0] - incident * half) * half  # at the entrance plane z = 0
            # The vacuum phase is removed with the exact wave number of the incident
            # beam, and each beam's Numerov dispersion over the slab's thickness with it
            # (exact in the vacuum parts of the slab, very nearly so in the rest).
            thickness = nz * h
            dispersion = np.where(
                np.abs(exact.imag) < 1e-12,
                np.exp(-1j * (kappa - exact) * thickness),
                1.0,
            )
            vacuum_phase = np.exp(-1j * incident_kz * thickness)
            transmitted = phi[-1] * half * dispersion * vacuum_phase

            def to_array(amplitudes):
                """Real-space wave from per-orbit (or per-beam) amplitudes."""
                if basis.reduced:
                    beam_amplitudes = np.zeros(len(basis.indices), np.complex128)
                    for o, members in enumerate(basis.members):
                        beam_amplitudes[members] = (
                            amplitudes[o]
                            * basis.phases[members]
                            / np.sqrt(len(members))
                        )
                else:
                    beam_amplitudes = amplitudes
                beam_amplitudes = beam_amplitudes * np.exp(
                    -2j * np.pi * (basis.gx * origin[0] + basis.gy * origin[1])
                )
                spectrum = np.zeros((nx, ny), np.complex128)
                spectrum[basis.indices[:, 0], basis.indices[:, 1]] = beam_amplitudes
                return np.fft.ifft2(spectrum) * (nx * ny)

            return to_array(transmitted), to_array(reflected)

        gate = _MemoryGate(0.7 * _available_memory())
        members = list(zip(tilts.reshape(-1, 2), energies.ravel()))
        workers = 1 if xp is not np else self.workers or _default_workers()
        workers = max(1, min(workers, len(members)))
        # the cores are shared between the BLAS threads of the members that run at the
        # same time; the limit is set in each thread, since the OpenMP thread count of
        # a BLAS is a property of the thread that calls it
        threads = max(1, _blas_threads() // workers) if workers > 1 else None

        def solve(member):
            with ExitStack() as stack:
                if threads is not None:
                    stack.enter_context(threadpool_limits(limits=threads))
                return solve_member(
                    *member, lambda nbytes: stack.enter_context(gate.reserve(nbytes))
                )

        if workers == 1:
            results = [solve(member) for member in members]
        else:
            # the members are independent: one thread each (NumPy's solve and matmul
            # release the GIL)
            with ThreadPoolExecutor(workers) as pool:
                results = list(pool.map(solve, members))
        module = get_array_module(waves.device)

        def to_waves(index):
            array = np.stack([result[index] for result in results]).reshape(
                ensemble_shape + (nx, ny)
            )
            array = module.asarray(array.astype(waves.array.dtype))
            return Waves(array, **waves._copy_kwargs(exclude=("array",)))

        return to_waves(0), to_waves(1)
