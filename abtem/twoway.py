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

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as sla

from abtem.core import config
from abtem.core.backend import asnumpy, cp, get_array_module
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.potentials.iam import PotentialArray
from abtem.tilt import _get_tilt_axes

if TYPE_CHECKING:
    from ase import Atoms

    from abtem.potentials.iam import BasePotential
    from abtem.waves import PlaneWave, Waves


def _linear_algebra(xp):
    """The LU factorisation and solve of the array module."""
    if xp is np:
        return sla.lu_factor, sla.lu_solve
    from cupyx.scipy import linalg as cupy_linalg  # type: ignore

    return cupy_linalg.lu_factor, cupy_linalg.lu_solve


def _memory_budget(xp, limit: float) -> float:
    """Bytes the solve may keep: ``limit``, and on a GPU at most 40 % of the free
    memory."""
    if xp is np:
        return limit
    free, _ = cp.cuda.Device().mem_info
    return min(limit, 0.4 * free)


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
        for name, matrix in operations:
            rotated = grid_g @ matrix.T
            ri = np.rint(rotated[:, 0] * lx).astype(int)
            rj = np.rint(rotated[:, 1] * ly).astype(int)
            on_grid = (np.abs(rotated[:, 0] * lx - ri) < 1e-6) & (
                np.abs(rotated[:, 1] * ly - rj) < 1e-6
            )
            inside = (np.abs(ri) <= nx // 2 - 1) & (np.abs(rj) <= ny // 2 - 1)
            if (on_grid & inside)[coupling_range].all():
                usable.append(matrix)

        orbit = -np.ones(len(self.indices), dtype=int)
        members = []
        for i in range(len(self.indices)):
            if orbit[i] >= 0:
                continue
            orbit_members = {i}
            for matrix in usable:
                rotated = matrix @ np.array([self.gx[i], self.gy[i]])
                j = index.get(
                    (int(np.rint(rotated[0] * lx)), int(np.rint(rotated[1] * ly)))
                )
                if j is not None:
                    orbit_members.add(j)
            for j in orbit_members:
                orbit[j] = len(members)
            members.append(sorted(orbit_members))

        self.orbit = orbit
        self.members = members
        self.number_of_orbits = len(members)
        self.representatives = np.array([m[0] for m in members])
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

        order = np.argsort(self.orbit, kind="stable")
        starts = np.searchsorted(self.orbit[order], np.arange(self.number_of_orbits))
        column_di = xp.asarray(di[:, self.representatives])
        column_dj = xp.asarray(dj[:, self.representatives])
        real_dtype = np.empty(0, dtype=u_k.dtype).real.dtype
        weights = xp.asarray(
            np.sqrt(self.sizes[None, :] / self.sizes[:, None]).astype(real_dtype)
        )

        if xp is np:
            order_indices = order

            def orbit_sum(columns):
                return np.add.reduceat(columns[order_indices], starts, axis=0)

        else:
            # no reduceat on the GPU: the sum over the members of every orbit is a
            # product with the indicator matrix of the orbits
            indicator = np.zeros((self.number_of_orbits, len(self.indices)), real_dtype)
            indicator[self.orbit, np.arange(len(self.indices))] = 1.0
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


def _solve_blocks(coupling, kz2, kappa, h, nz, incident, xp=np, dtype=np.complex128):
    """Solve the Numerov equations for waves incident from below (z < 0) on the slab.

    ``coupling(n)`` is the coupling matrix of slice n (an array of ``xp``), ``kz2`` the
    squared longitudinal wave numbers of the beams in vacuum, ``kappa`` the wave numbers
    of the Numerov recurrence in vacuum (:func:`_numerov_wave_numbers`), ``incident``
    the amplitudes (n_beams,) of the incident beams at z = 0, all of them on the host.
    Returns the amplitudes phi (nz, n_beams), of the complex type ``dtype`` and the
    array module ``xp``, at the slice centres z_n = (n + 1/2) h, in the block-Thomas
    solve with checkpointing.
    """
    lu_factor, lu_solve = _linear_algebra(xp)
    nb = len(kz2)
    identity = xp.eye(nb, dtype=dtype)
    c = h**2 / 12
    outgoing = xp.asarray((np.exp(1j * kappa * h)).astype(dtype))
    base = xp.diag(xp.asarray(kz2.astype(dtype)))

    def operator(n):
        return base if n < 0 or n >= nz else base + coupling(n)

    vacuum_boundary = identity + c * operator(-1)
    rhs = xp.zeros((nz, nb), dtype)
    z0 = h / 2
    incident_at_first = xp.asarray((incident * np.exp(1j * kappa * z0)).astype(dtype))
    entrance = xp.asarray(
        (np.exp(-1j * kappa * h) - np.exp(1j * kappa * h)).astype(dtype)
    )
    rhs[0] = -vacuum_boundary @ (incident_at_first * entrance)

    def blocks(n):
        diagonal = -2 * (identity - 5 * c * operator(n))
        if n == 0 or n == nz - 1:
            diagonal = diagonal + vacuum_boundary * outgoing[None, :]
        upper = identity + c * operator(n + 1) if n < nz - 1 else None
        lower = identity + c * operator(n - 1) if n > 0 else None
        return diagonal, upper, lower

    def eliminate(n, previous):
        diagonal, upper, lower = blocks(n)
        if n > 0:
            diagonal = diagonal - lower @ previous
        factor = lu_factor(diagonal)
        return (
            factor,
            lower,
            (lu_solve(factor, upper) if upper is not None else None),
        )

    itemsize = np.dtype(dtype).itemsize
    full_bytes = nz * nb * nb * itemsize
    stride = (
        nz
        if full_bytes < _memory_budget(xp, _CHECKPOINT_BYTES)
        else int(np.ceil(np.sqrt(nz)))
    )

    forward = xp.empty((nz, nb), dtype)
    stored, checkpoints = {}, {}
    previous = None
    for n in range(nz):
        factor, lower, eliminated = eliminate(n, previous)
        right = rhs[n] - lower @ forward[n - 1] if n > 0 else rhs[n]
        forward[n] = lu_solve(factor, right)
        if stride == nz:
            stored[n] = eliminated
        elif n % stride == 0:
            checkpoints[n] = eliminated
        previous = eliminated

    phi = xp.empty((nz, nb), dtype)
    phi[-1] = forward[-1]
    if stride == nz:
        for n in range(nz - 2, -1, -1):
            phi[n] = forward[n] - stored[n] @ phi[n + 1]
    else:
        for start in sorted(checkpoints, reverse=True):
            end = min(start + stride, nz - 1)
            segment = {start: checkpoints[start]}
            for n in range(start + 1, end):
                segment[n] = eliminate(n, segment[n - 1])[2]
            for n in range(end - 1, start - 1, -1):
                phi[n] = forward[n] - segment[n] @ phi[n + 1]
    return phi


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
class TwoWaySolver:
    """
    Exact two-way solution of the elastic scattering of a plane wave by a periodic slab.

    The solution includes backscattering and the evanescent beams up to the cutoff,
    which at low energies carry most of the reflection (graphene at 200 eV reflects 7 %,
    almost all of it through evanescent beams) and change the transmitted beams by tens
    of percent. The cost grows with the cube of the number of beams: use it for the
    small cells of crystals and 2D materials. A plane wave of any direction is
    supported (one direction at a time), on the CPU and on CUDA devices, in single or
    double precision, without absorption.

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
        Reduce the basis by the translation and point symmetry of the structure, found
        from the atoms of the potential (default True). A tilted wave is only invariant
        under the translations, which are used alone. The results agree with the full
        basis to the asymmetry of the sampled potential; a distorted structure is not
        symmetrised. A built :class:`.PotentialArray` carries no atoms and is solved in
        the full basis.
    symmetry_tolerance : float, optional
        Largest displacement [Å] by which a symmetry operation may miss an atom (default
        1e-4).
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

    def __post_init__(self):
        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if self.precision not in (None, "float32", "float64"):
            raise ValueError(
                "precision must be 'float32', 'float64' or None, "
                f"got {self.precision!r}"
            )

    def scatter(
        self, plane_wave: PlaneWave, potential: BasePotential
    ) -> tuple[Waves, Waves]:
        """
        Scatter a plane wave from the slab.

        Parameters
        ----------
        plane_wave : PlaneWave
            The incident wave. Its energy and its tilt are used (a single tilt, not an
            ensemble), and its grid is matched to the potential. A tilted wave travels
            along (tan x, tan y, 1) and, as in multislice, is stored without the carrier
            of its direction: the beam g of the result is the plane wave g + k_t, the
            incident one is g = 0, and the result carries the tilt in its metadata.
        potential : BasePotential
            The slab. A :class:`.Potential` or a built :class:`.PotentialArray`, without
            an ensemble axis.

        Returns
        -------
        transmitted, reflected : Waves
            The wave leaving the slab on the far side (after the last slice) and the
            wave leaving on the near side (at the entrance plane), both with the beams
            of the solution and the phase convention of multislice: the vacuum phase
            over the slab's thickness is removed, so an empty slab gives a transmitted
            wave of 1 and no reflected wave.
        """
        from abtem.waves import Waves

        waves = plane_wave.match_grid(potential).build(lazy=False)
        if _get_tilt_axes(waves):
            raise NotImplementedError(
                "TwoWaySolver does not support tilt ensembles: scatter one tilt at a "
                "time."
            )
        if potential.ensemble_shape:
            raise NotImplementedError(
                "TwoWaySolver does not support ensemble potentials."
            )

        energy = waves._valid_energy
        wavelength = energy2wavelength(energy)
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
        thicknesses = np.asarray(built.slice_thickness, dtype=float)
        h = float(thicknesses.mean())
        if not np.allclose(thicknesses, h, rtol=1e-6):
            raise ValueError("TwoWaySolver needs slices of equal thickness.")
        projected = xp.asarray(built.array, dtype=real_dtype)
        nz, nx, ny = projected.shape
        sampling = tuple(waves._valid_sampling)

        # The incident wave travels along (tan x, tan y, 1), and is stored without its
        # carrier exp(2 pi i (k_t . r + k_zt z)), as multislice does: the stored beam g
        # is the plane wave g + k_t, and the incident one is g = 0.
        tangent = np.tan(np.asarray(waves.base_tilt, dtype=float) / 1e3)
        norm = np.sqrt(1.0 + np.sum(tangent**2))
        in_plane = tuple(tangent / (wavelength * norm))  # k_t [1/Å]
        tilted = any(in_plane)
        incident_kz = 2 * np.pi / (wavelength * norm)

        if 1 / (2 * max(sampling)) < 2 * self.cutoff + np.hypot(*in_plane):
            raise ValueError(
                f"The potential is sampled too coarsely for cutoff={self.cutoff}: use "
                f"a sampling of at most "
                f"{1 / (4 * self.cutoff + 2 * np.hypot(*in_plane)):.4f} Å or a lower "
                "cutoff."
            )

        origin, operations, translations = (0.0, 0.0), [("rot0", np.eye(2))], []
        if self.symmetry:
            try:
                atoms = potential.frozen_phonons.atoms  # type: ignore[attr-defined]
            except AttributeError:
                atoms = None
            if atoms is not None and len(atoms) > 0:
                origin, operations, translations = structure_symmetry(
                    atoms, self.symmetry_tolerance
                )
                if tilted:
                    # a tilted wave is invariant under the translations only
                    origin, operations = (0.0, 0.0), [("rot0", np.eye(2))]

        wave_number = 2 * np.pi / wavelength
        sigma = energy2sigma(energy)
        _check_step(h, wave_number, self.cutoff)
        basis = _BeamBasis(
            (nx, ny),
            sampling,
            self.cutoff,
            origin,
            operations,
            translations,
            shift=in_plane,
        )

        # the potential's Fourier coefficients, with the origin moved onto the symmetry
        # axis
        fx = np.fft.fftfreq(nx, sampling[0])[:, None]
        fy = np.fft.fftfreq(ny, sampling[1])[None, :]
        shift = np.exp(2j * np.pi * (fx * origin[0] + fy * origin[1]))
        fourier = xp.fft.fft2(projected / h, axes=(1, 2)) / (nx * ny)
        fourier = (fourier * xp.asarray(shift.astype(complex_dtype))[None]).astype(
            complex_dtype
        )
        coupling = basis.coupling_function(fourier, wave_number, sigma, xp)
        itemsize = np.dtype(complex_dtype).itemsize
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
        incident = np.zeros(len(kz2), np.complex128)
        incident[int(np.argmin(np.abs(exact - incident_kz)))] = 1.0

        phi = asnumpy(
            _solve_blocks(coupling, kz2, kappa, h, nz, incident, xp, complex_dtype)
        ).astype(np.complex128)
        half = np.exp(1j * kappa * h / 2)
        reflected = (phi[0] - incident * half) * half  # at the entrance plane z = 0
        # The vacuum phase is removed with the exact wave number of the incident beam,
        # and each beam's Numerov dispersion over the slab's thickness with it (exact in
        # the vacuum parts of the slab, very nearly so in the rest).
        thickness = nz * h
        dispersion = np.where(
            np.abs(exact.imag) < 1e-12, np.exp(-1j * (kappa - exact) * thickness), 1.0
        )
        vacuum_phase = np.exp(-1j * incident_kz * thickness)
        transmitted = phi[-1] * half * dispersion * vacuum_phase

        def to_waves(amplitudes):
            """Real-space wave from per-orbit (or per-beam) amplitudes."""
            if basis.reduced:
                beam_amplitudes = np.zeros(len(basis.indices), np.complex128)
                for o, members in enumerate(basis.members):
                    beam_amplitudes[members] = amplitudes[o] / np.sqrt(len(members))
            else:
                beam_amplitudes = amplitudes
            beam_amplitudes = beam_amplitudes * np.exp(
                -2j * np.pi * (basis.gx * origin[0] + basis.gy * origin[1])
            )
            spectrum = np.zeros((nx, ny), np.complex128)
            spectrum[basis.indices[:, 0], basis.indices[:, 1]] = beam_amplitudes
            array = np.fft.ifft2(spectrum) * (nx * ny)
            array = get_array_module(waves.device).asarray(
                array.astype(waves.array.dtype)
            )
            return Waves(array, **waves._copy_kwargs(exclude=("array",)))

        return to_waves(transmitted), to_waves(reflected)
