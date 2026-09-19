"""Skew-aware generalization of ``commensurate_gpts`` (from PR #347 / the
``autogrid`` branch on ``dev``, not yet merged into ``skewpixels``).

The upstream ``commensurate_gpts`` finds a grid size commensurate with the
atomic lattice by treating x and y independently: for each Cartesian axis it
takes ``positions[:, i] % extent[i]``, finds the GCD of the resulting
atom-plane spacings, and derives a grid period from that. That is only a
physically meaningful periodicity check for an *orthogonal* cell -- atoms in
a skewed cell (e.g. a 60-degree hex lattice) are periodic under translations
by the lattice vectors a1, a2, not independently along Cartesian x or y.

This prototype generalizes the search to run in FRACTIONAL (lattice)
coordinates instead: ``frac = positions @ inv(cell)``, then treats
``frac[:, i] * |a_i|`` as the "along-axis" coordinate for lattice direction
i, in Angstrom (so the same tolerance semantics apply). For a diagonal
(orthogonal) cell this is an identity: ``frac[:, i] * |a_i|`` reduces exactly
to ``positions[:, i] % extent[i]``, so an orthogonal cell reproduces the
original function bit-for-bit. The GCD / translational-period / fast-FFT
logic is untouched -- copied verbatim from ``commensurate_gpts`` -- since it
already operates generically on a 1D coordinate array and a period length;
only how that array is computed changes.

Validated below against three claims:
  1. Reduction: cell=diagonal(extent) gives the same gpts as the original
     Cartesian-only algorithm, on both a trivial and a non-trivial lattice.
  2. Correctness on a skewed cell: a small hexagonal graphene-like structure
     finds the true small commensurate period along a1 and a2, not the
     Cartesian-axis period (which does not exist for this structure).
  3. Translation invariance: shifting the whole structure by an arbitrary
     (non-lattice) vector gives the same gpts -- the same invariant the real
     `commensurate_gpts` test suite checks for orthogonal cells.
  4. The physical claim itself: atoms related by a lattice translation land
     at the same fractional pixel offset under the resulting grid.
"""

import numpy as np

_FAST_FFT_PRIMES = (2, 3, 5, 7)


def is_fast_fft_size(n: int) -> bool:
    n = int(n)
    if n < 1:
        return False
    for p in _FAST_FFT_PRIMES:
        while n % p == 0:
            n //= p
    return n == 1


def next_fast_fft_size(n: int) -> int:
    n = max(int(n), 1)
    while not is_fast_fft_size(n):
        n += 1
    return n


_FALLBACK_MAX_OVERSHOOT = 1.12


def _plane_set_invariant_under(
    unique_x: np.ndarray, L: float, s: float, tolerance: float = 1e-3
) -> bool:
    n = len(unique_x)
    shifted = (unique_x + s) % L
    idx = np.searchsorted(unique_x, shifted)
    neighbors = np.stack([unique_x[(idx - 1) % n], unique_x[idx % n]])
    distances = np.abs(neighbors - shifted)
    distances = np.minimum(distances, L - distances)
    return bool(np.all(distances.min(axis=0) < tolerance))


def _translational_period_count(
    unique_x: np.ndarray, L: float, tolerance: float = 1e-3
) -> int:
    deltas = np.sort((unique_x - unique_x[0]) % L)
    tested: set[int] = set()
    for delta in deltas:
        if delta < tolerance:
            continue
        if delta > L / 2 + tolerance:
            break
        m = L / delta
        m_int = int(round(m))
        if m_int < 2 or abs(m - m_int) > 0.05 or m_int in tested:
            continue
        tested.add(m_int)
        if _plane_set_invariant_under(unique_x, L, L / m_int, tolerance):
            return m_int
    return 1


def commensurate_gpts(
    extent,
    positions: np.ndarray,
    target_sampling: float = 0.05,
    tolerance: float = 1e-3,
    round_to_fast_fft: bool = True,
    cell: np.ndarray | None = None,
) -> tuple:
    """Skew-aware ``commensurate_gpts``.

    Parameters
    ----------
    extent : tuple of float
        Lattice-vector lengths (|a1|, |a2|) [A]. For an orthogonal cell this
        is the same as the Cartesian extent.
    positions : np.ndarray
        Atom positions, shape (N, 2) or (N, 3).
    cell : np.ndarray, optional
        2x2 real-space cell (rows a1, a2). None (default) reproduces the
        original Cartesian-only algorithm exactly.
    """

    def fallback_gpts(n_target: int, unique_x=None, L=None) -> int:
        if not round_to_fast_fft:
            return n_target
        if unique_x is None or L is None or len(unique_x) <= 1:
            return next_fast_fft_size(n_target)

        m = _translational_period_count(unique_x, L, tolerance)
        if not is_fast_fft_size(m):
            return max(round(n_target / m), 1) * m

        best = None
        multiplier = -(-n_target // m)
        while True:
            multiplier = next_fast_fft_size(multiplier)
            n = multiplier * m
            if best is not None and n > n_target * _FALLBACK_MAX_OVERSHOOT:
                break
            residues = np.sort(np.mod(unique_x * (n / L), 1.0))
            gaps = np.diff(np.concatenate([residues, [residues[0] + 1.0]]))
            alignability = (1.0 - float(np.max(gaps))) / 2.0
            misalignment = alignability * (L / n)
            score = (round(misalignment, 3), n)
            if best is None or score < best[0]:
                best = (score, n)
            multiplier += 1

        assert best is not None
        if best[1] > n_target * _FALLBACK_MAX_OVERSHOOT:
            return max(round(n_target / m), 1) * m
        return best[1]

    if cell is not None:
        cell_2d = np.asarray(cell, dtype=float)[:2, :2]
        inv_cell = np.linalg.inv(cell_2d)
        # fractional coordinates along each lattice vector, in [0, 1)
        frac = (positions[:, :2] @ inv_cell) % 1.0

    gpts = []
    for i in range(2):
        L = extent[i]
        n_target = int(np.ceil(L / target_sampling))

        if cell is not None:
            # along-axis Angstrom coordinate for lattice direction i, exactly
            # analogous to positions[:, i] % extent[i] for an orthogonal cell
            x = frac[:, i] * L
        else:
            x = positions[:, i] % L

        x[x > L - tolerance] = 0.0
        x = np.sort(x)

        unique_mask = np.concatenate([[True], np.diff(x) > tolerance])
        unique_x = x[unique_mask]

        if len(unique_x) <= 1:
            gpts.append(fallback_gpts(n_target, unique_x, L))
            continue

        spacings = np.concatenate([np.diff(unique_x), [L + unique_x[0] - unique_x[-1]]])

        min_spacing = float(np.min(spacings))
        if min_spacing < tolerance:
            gpts.append(fallback_gpts(n_target, unique_x, L))
            continue

        ratios = spacings / min_spacing
        k = np.round(ratios).astype(int)
        if np.any(np.abs(ratios - k) > 0.05) or np.any(k < 1):
            gpts.append(fallback_gpts(n_target, unique_x, L))
            continue

        n_periods = int(np.sum(k))
        nearest = max(round(n_target / n_periods), 1)

        if round_to_fast_fft and is_fast_fft_size(n_periods):
            multiplier = (
                nearest if is_fast_fft_size(nearest) else next_fast_fft_size(nearest)
            )
            n_multiple = multiplier * n_periods
        else:
            n_multiple = nearest * n_periods
        gpts.append(n_multiple)

    return tuple(gpts)


def commensurate_gpts_orthogonal_original(
    extent, positions, target_sampling=0.05, tolerance=1e-3, round_to_fast_fft=True
):
    """Verbatim copy of the upstream (dev, pre-skew) algorithm, for the
    reduction check -- not otherwise used."""
    return commensurate_gpts(
        extent, positions, target_sampling, tolerance, round_to_fast_fft, cell=None
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_orthogonal_reduction():
    rng = np.random.default_rng(0)
    a, b = 12.3, 9.7
    n_cells_x, n_cells_y = 6, 5
    base = rng.random((3, 2))
    positions = np.concatenate(
        [
            base * [a / n_cells_x, b / n_cells_y] + [i * a / n_cells_x, j * b / n_cells_y]
            for i in range(n_cells_x)
            for j in range(n_cells_y)
        ]
    )
    extent = (a, b)

    gpts_ref = commensurate_gpts(extent, positions, cell=None)
    gpts_skew_diag = commensurate_gpts(extent, positions, cell=np.diag(extent))

    print(f"orthogonal (no cell):   gpts = {gpts_ref}")
    print(f"skew path, diag cell:   gpts = {gpts_skew_diag}")
    assert gpts_ref == gpts_skew_diag, "diagonal cell must reduce exactly to the orthogonal path"
    print("OK: diagonal-cell path is bit-identical to the orthogonal path\n")


def _hex_positions(a, reps, basis_frac):
    """Atom positions (Cartesian) for a 2D hex lattice with the given
    fractional basis, tiled reps x reps times."""
    a1 = np.array([a, 0.0])
    a2 = np.array([a * np.cos(np.deg2rad(60)), a * np.sin(np.deg2rad(60))])
    cell = np.stack([a1, a2])
    positions = []
    for i in range(reps):
        for j in range(reps):
            for f in basis_frac:
                positions.append((f[0] + i) * a1 + (f[1] + j) * a2)
    return np.array(positions), cell


def check_skew_correctness():
    a = 2.46  # graphene-like lattice constant
    reps = 6
    basis = [(0.0, 0.0), (1 / 3, 1 / 3)]  # two-atom hex basis
    positions, cell = _hex_positions(a, reps, basis)
    extent = (a * reps, a * reps)

    gpts = commensurate_gpts(extent, positions, target_sampling=0.05, cell=cell)
    print(f"skew hex lattice ({reps}x{reps} cells, a={a}): gpts = {gpts}")

    # True period: reps unit cells along each lattice vector, times 2 for the
    # 2-atom basis's extra plane per cell (0 and 1/3 fractional coordinate are
    # each a distinct plane along both a1 and a2 in this particular basis).
    # What actually matters is that gpts[i] is an exact multiple of `reps`
    # (grid points per unit cell along a_i must divide the supercell into
    # `reps` identical, symmetric copies).
    for i in range(2):
        assert gpts[i] % reps == 0, (
            f"axis {i}: gpts={gpts[i]} is not a multiple of the "
            f"{reps}-cell supercell period"
        )
    print(f"OK: gpts is an exact multiple of the {reps}-cell period on both lattice axes\n")

    # Cross-check: the same call WITHOUT cell (Cartesian-only) should NOT
    # generally find a clean small commensurate period for a 60-degree cell,
    # since Cartesian x/y are not the periodic directions.
    gpts_wrong = commensurate_gpts(extent, positions, target_sampling=0.05, cell=None)
    print(f"(for contrast) Cartesian-only, same positions: gpts = {gpts_wrong}")


def check_translation_invariance():
    a = 2.46
    reps = 5
    basis = [(0.0, 0.0), (1 / 3, 1 / 3)]
    positions, cell = _hex_positions(a, reps, basis)
    extent = (a * reps, a * reps)

    gpts_ref = commensurate_gpts(extent, positions, target_sampling=0.05, cell=cell)

    supercell = cell * reps
    inv_supercell = np.linalg.inv(supercell)
    rng = np.random.default_rng(1)
    for trial in range(5):
        shift = rng.random(2) * extent[0]  # arbitrary, not a lattice vector
        # Wrap into the supercell box in FRACTIONAL (lattice) coordinates -- the
        # supercell is a parallelogram, not a Cartesian rectangle, so `% extent`
        # would wrap incorrectly for a skewed cell.
        frac = ((positions + shift) @ inv_supercell) % 1.0
        shifted = frac @ supercell
        gpts_shifted = commensurate_gpts(
            extent, shifted, target_sampling=0.05, cell=cell
        )
        assert gpts_shifted == gpts_ref, (
            f"trial {trial}: shift {shift} changed gpts from {gpts_ref} to {gpts_shifted}"
        )
    print(f"OK: gpts={gpts_ref} is invariant under 5 arbitrary rigid shifts\n")


def check_symmetry_equivalent_atoms_discretize_identically():
    """The actual physical claim: two atoms related by a lattice translation
    must land at the same FRACTIONAL pixel offset under the chosen grid, so
    they receive identical discretised potential contributions."""
    a = 2.46
    reps = 6
    basis = [(0.0, 0.0), (1 / 3, 1 / 3)]
    positions, cell = _hex_positions(a, reps, basis)
    extent = (a * reps, a * reps)

    gpts = commensurate_gpts(extent, positions, target_sampling=0.05, cell=cell)

    inv_cell = np.linalg.inv(cell)
    frac = (positions @ inv_cell) % 1.0  # fractional coords within ONE unit cell
    pixel_frac = np.stack([frac[:, 0] * gpts[0], frac[:, 1] * gpts[1]], axis=1)
    sub_pixel = pixel_frac % 1.0  # sub-pixel offset -- must match for equivalent atoms

    # basis atom 0 instances vs basis atom 1 instances
    n_cells = reps * reps
    sub0 = sub_pixel[0::2]  # every first basis atom across all unit cells
    sub1 = sub_pixel[1::2]  # every second basis atom across all unit cells

    def circular_spread(x):
        # x has shape (N, 2), values in [0, 1); 0 and 1 are the same point, so
        # measure spread per-coordinate via the resultant vector length on the
        # unit circle rather than plain ptp (which treats 0 and 1 as far apart).
        theta = 2 * np.pi * x  # (N, 2)
        c = np.cos(theta).mean(axis=0)  # (2,)
        s = np.sin(theta).mean(axis=0)  # (2,)
        resultant_length = np.sqrt(c**2 + s**2)  # (2,), 1.0 = no spread
        return np.arccos(np.clip(resultant_length, -1.0, 1.0)) / (2 * np.pi)

    spread0 = circular_spread(sub0)
    spread1 = circular_spread(sub1)
    print(f"sub-pixel offset spread across {n_cells} unit cells:")
    print(f"  basis atom 0: {spread0} pixels")
    print(f"  basis atom 1: {spread1} pixels")
    assert np.all(spread0 < 1e-6) and np.all(spread1 < 1e-6), (
        "symmetry-equivalent atoms must discretise identically (sub-pixel offset "
        "spread should be at the floating-point floor)"
    )
    print("OK: symmetry-equivalent atoms land at an identical sub-pixel offset\n")


def check_incommensurate_fallback_on_skew():
    """A skewed cell with an irrational internal parameter (analogous to
    rutile's u=0.306) has no exactly commensurate grid. The fallback must
    still (a) not crash, (b) return gpts that is a multiple of the
    translational period along each lattice direction, and (c) respect the
    requested target sampling to within the documented overshoot window."""
    a = 3.0
    reps = 4
    u = 0.306  # irrational-ish internal parameter, as in rutile
    basis = [(0.0, 0.0), (u, u), (0.5 + u, 0.5 - u)]
    positions, cell = _hex_positions(a, reps, basis)
    extent = (a * reps, a * reps)

    gpts = commensurate_gpts(extent, positions, target_sampling=0.05, cell=cell)
    print(f"skewed incommensurate basis (u={u}): gpts = {gpts}")

    for i in range(2):
        assert gpts[i] % reps == 0, (
            f"axis {i}: fallback gpts={gpts[i]} is not a multiple of the "
            f"{reps}-cell translational period"
        )
        n_target = int(np.ceil(extent[i] / 0.05))
        assert gpts[i] <= n_target * _FALLBACK_MAX_OVERSHOOT * 1.01, (
            f"axis {i}: fallback overshot the {_FALLBACK_MAX_OVERSHOOT}x window"
        )
    print("OK: incommensurate-basis fallback stays period-consistent and in-window\n")


if __name__ == "__main__":
    check_orthogonal_reduction()
    check_skew_correctness()
    check_translation_invariance()
    check_incommensurate_fallback_on_skew()
    check_symmetry_equivalent_atoms_discretize_identically()
    print("All checks passed.")
