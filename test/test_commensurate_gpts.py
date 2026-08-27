"""Tests for commensurate automatic grids and their interplay with fast FFT sizes."""

import numpy as np
import pytest
from ase.build import bulk

from abtem.core.fft import is_fast_fft_size, next_fast_fft_size
from abtem.potentials.iam import Potential
from abtem.slicing import (
    _plane_set_invariant_under,
    _translational_period_count,
    commensurate_gpts,
)


def _plane_positions(x_planes, y_planes):
    positions = np.zeros((len(x_planes) * len(y_planes), 3))
    positions[:, 0] = np.repeat(x_planes, len(y_planes))
    positions[:, 1] = np.tile(y_planes, len(x_planes))
    return positions


def test_commensurate_and_fast():
    # 4 periods in x, 5 in y — both fast-compatible.
    extent = (10.0, 10.0)
    positions = _plane_positions(np.arange(4) * 2.5, np.arange(5) * 2.0)

    gpts = commensurate_gpts(extent, positions, target_sampling=0.05)

    for n, n_periods in zip(gpts, (4, 5)):
        assert n % n_periods == 0
        assert is_fast_fft_size(n)
        # Never coarser than the target sampling.
        assert extent[0] / n <= 0.05 + 1e-12


def test_commensurability_beats_fast_fft():
    # 11 periods in x: no multiple of 11 is 7-smooth, so the grid stays
    # commensurate and cannot be a fast FFT size.
    extent = (10.0, 10.0)
    positions = _plane_positions(np.arange(11) * (10.0 / 11.0), np.arange(4) * 2.5)

    gpts = commensurate_gpts(extent, positions, target_sampling=0.05)

    assert gpts[0] % 11 == 0
    assert not is_fast_fft_size(gpts[0])
    assert gpts[1] % 4 == 0
    assert is_fast_fft_size(gpts[1])


def test_round_to_fast_fft_disabled_matches_plain_commensurate():
    extent = (10.0, 10.0)
    positions = _plane_positions(np.arange(4) * 2.5, np.arange(5) * 2.0)

    gpts = commensurate_gpts(
        extent, positions, target_sampling=0.05, round_to_fast_fft=False
    )

    for n, n_periods in zip(gpts, (4, 5)):
        n_target = int(np.ceil(10.0 / 0.05))
        assert n == max(round(n_target / n_periods), 1) * n_periods


def test_fallback_rounds_to_fast_fft():
    # A single atom gives no plane spacings, so the target size is used and
    # rounded up to a fast FFT length. 131.1 / 0.05 -> 2622 -> 2625 = 3*5^3*7.
    extent = (131.1, 131.1)
    positions = np.zeros((1, 3))

    gpts = commensurate_gpts(extent, positions, target_sampling=0.05)
    assert gpts == (2625, 2625)

    gpts = commensurate_gpts(
        extent, positions, target_sampling=0.05, round_to_fast_fft=False
    )
    assert gpts == (2622, 2622)


def test_smallest_fast_commensurate_multiple():
    # 6 periods, target 100.35 Å at 0.05 Å -> n_target = 2007, ceil(2007/6) = 335,
    # next fast multiplier is 336 = 2^4*3*7 -> gpts = 2016.
    extent = (100.35, 100.35)
    positions = _plane_positions(np.arange(6) * (100.35 / 6.0), np.arange(6) * (100.35 / 6.0))

    gpts = commensurate_gpts(extent, positions, target_sampling=0.05)

    assert gpts == (2016, 2016)
    assert next_fast_fft_size(335) == 336


@pytest.mark.parametrize("name,crystalstructure,a", [("Si", "diamond", 5.431), ("Au", "fcc", 4.078)])
def test_potential_auto_sampling_is_commensurate_and_fast(name, crystalstructure, a):
    from abtem.core import config

    atoms = bulk(name, crystalstructure, a=a, cubic=True) * (2, 2, 1)

    with config.set({"grid.round-to-fast-fft": "auto"}):
        potential = Potential(atoms, sampling="auto", projection="infinite")

    # Derived independently: the plain commensurate grid is 216 = 2**3 * 27 for
    # Si (already fast, so kept) and 164 = 2**2 * 41 for Au (not fast, so the
    # multiplier moves up: 41 -> 42, giving 168).
    assert potential.gpts == {"Si": (216, 216), "Au": (168, 168)}[name]

    for axis, (n, extent) in enumerate(zip(potential.gpts, potential.extent)):
        assert is_fast_fft_size(n)
        # Close to the target sampling. Not necessarily finer than it: the
        # commensurate grid nearest the target wins whenever it is already
        # fast, exactly as with rounding switched off.
        assert abs(extent / n - 0.05) / 0.05 < 0.12
        # Commensurate: every atom plane falls on a grid point.
        fractional = atoms.positions[:, axis] / (extent / n)
        offsets = np.abs(fractional - np.round(fractional))
        assert np.all(offsets < 1e-3)


def test_incommensurate_fallback_preserves_translational_period():
    # Rutile-like planes: intra-cell spacings with an irrational ratio (no
    # exactly commensurate grid exists), repeated 3 times. The chosen grid must
    # remain a multiple of the repeat count so symmetry-equivalent atoms in
    # different unit cells see identical discretised potentials, and fast.
    cell = 4.594
    base = np.array([0.0, 0.8912, 1.4058, 2.297, 3.1882, 3.7028])
    planes = np.concatenate([base + i * cell for i in range(3)])
    extent = (3 * cell, 3 * cell)
    positions = _plane_positions(planes, planes)

    gpts = commensurate_gpts(extent, positions, target_sampling=0.05)

    assert gpts[0] % 3 == 0
    assert is_fast_fft_size(gpts[0])
    # 294 = 2*3*7^2 aligns the rutile planes to ~0.01 px (u = 0.306 ~ 30/98).
    assert gpts[0] == 294


def test_incommensurate_fallback_with_non_fast_period():
    # 11 repeats of an incommensurate cell: no multiple of 11 is 7-smooth, so
    # translational commensurability wins over the fast-FFT condition.
    cell = 1.3
    base = np.array([0.0, 0.4132, 0.9271])
    planes = np.concatenate([base + i * cell for i in range(11)])
    extent = (11 * cell, 11 * cell)
    positions = _plane_positions(planes, planes)

    gpts = commensurate_gpts(extent, positions, target_sampling=0.05)

    assert gpts[0] % 11 == 0


def test_incommensurate_fallback_scales_to_large_supercells():
    # The translational-period search prunes candidate shifts to the plane
    # differences with near-integer L/s, so a large supercell of an
    # incommensurate structure stays cheap (review flagged the previous
    # search-every-m version as O(n^2 log n)).
    import time

    cell = 9.184
    base = np.sort(np.random.RandomState(0).uniform(0, cell, 12))
    planes = np.concatenate([base + i * cell for i in range(200)])
    extent = 200 * cell
    positions = _plane_positions(planes, planes)

    start = time.perf_counter()
    gpts = commensurate_gpts((extent, extent), positions, target_sampling=0.05)
    elapsed = time.perf_counter() - start

    # Correctness: the 200-cell translational period must be preserved.
    assert gpts[0] % 200 == 0
    # Performance: ~2 ms measured; 1 s is a generous CI-safe bound that the
    # old quadratic search would still miss by an order of magnitude here.
    assert elapsed < 1.0


def test_plane_set_invariance_sees_across_the_periodic_wrap():
    # A shifted plane landing an ulp below the cell edge is nearest to the
    # plane at 0 through the wrap. Clipping the searchsorted index collapses
    # both neighbours onto the last plane and misses it, rejecting a genuine
    # period; float cell transforms produce exactly this (orthogonalize_cell
    # returns rutile's L / 2 plane one ulp short).
    L = 10.0
    planes = np.array([0.0, 1.3, 5.0 - 1e-12, 6.3])

    # Shifting by L / 2 maps this set onto itself to well within tolerance.
    assert _plane_set_invariant_under(planes, L, 5.0 - 1e-12)
    assert _translational_period_count(planes, L) == 2


def test_auto_gpts_is_translation_invariant_for_incommensurate_structures():
    # Rutile has no exactly commensurate grid, so it takes the fallback path.
    # Its grid must not depend on where the structure sits in the cell -- the
    # property commensurate_gpts exists to guarantee, and the one the fallback
    # scoring has to preserve. (The regression guard for the periodic-wrap bug
    # itself is test_plane_set_invariance_sees_across_the_periodic_wrap; this
    # is the end-to-end property that bug was one way of breaking.)
    from ase.spacegroup import crystal

    rutile = crystal(
        ["Ti", "O"],
        basis=[(0, 0, 0), (0.30478, 0.30478, 0)],
        spacegroup=136,
        cellpar=[4.5937, 4.5937, 2.9587, 90, 90, 90],
    )

    for plane in ("xy", "xz", "yz"):
        reference = Potential(rutile, sampling="auto", plane=plane).gpts
        for shift in (1e-6, 0.001, 0.25, 1.3):
            shifted = rutile.copy()
            shifted.positions[:, 0] += shift
            shifted.wrap()
            assert Potential(shifted, sampling="auto", plane=plane).gpts == reference


def test_period_search_is_not_abandoned_on_dense_plane_sets():
    # A tiled disordered cell (e.g. an amorphous support film repeated to cover
    # the field of view) offers hundreds of near-integer-ratio plane
    # differences that fail the invariance test before the real period is
    # reached. Abandoning the search after a fixed number of candidates
    # silently dropped the translational period the fallback exists to keep.
    rng = np.random.RandomState(3)
    cell = 25.0
    base = rng.uniform(0, cell, 900)
    planes = np.sort(np.concatenate([base, base + cell]))

    # The period itself is the assertion: the grid happens to come out even
    # either way here, so asserting only on gpts would not discriminate.
    assert _translational_period_count(planes, 2 * cell) == 2


def test_auto_sampling_respects_the_fast_fft_config():
    # The config option governs every automatically derived grid, so a user who
    # turns it off gets no rounding anywhere -- previously sampling='auto'
    # rounded unconditionally and could not be switched off.
    from abtem.core import config

    # Gold: the nearest commensurate grid is 164 = 2**2 * 41, not a fast size,
    # so the two modes genuinely differ here.
    atoms = bulk("Au", "fcc", a=4.078, cubic=True) * (2, 2, 1)

    with config.set({"grid.round-to-fast-fft": "auto"}):
        rounded = Potential(atoms, sampling="auto").gpts
    with config.set({"grid.round-to-fast-fft": False}):
        plain = Potential(atoms, sampling="auto").gpts

    assert all(is_fast_fft_size(n) for n in rounded)
    assert not any(is_fast_fft_size(n) for n in plain)
    assert plain != rounded
    # 'false' must reproduce the un-rounded commensurate grid exactly.
    extent = (float(atoms.cell[0, 0]), float(atoms.cell[1, 1]))
    assert plain == commensurate_gpts(
        extent, atoms.positions, target_sampling=0.05, round_to_fast_fft=False
    )


def test_ensemble_auto_sampling_respects_the_fast_fft_config():
    from abtem.core import config

    atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (2, 2, 1)
    trajectory = [atoms, atoms.copy()]

    with config.set({"grid.round-to-fast-fft": "auto"}):
        rounded = Potential(trajectory, sampling="auto").gpts
    with config.set({"grid.round-to-fast-fft": False}):
        plain = Potential(trajectory, sampling="auto").gpts

    assert all(is_fast_fft_size(n) for n in rounded)
    assert plain == tuple(
        int(np.ceil(float(atoms.cell[i, i]) / 0.05)) for i in range(2)
    )


def test_invalid_fast_fft_config_raises():
    # 'auto' is truthy, so a bare truthiness test on the config value would
    # silently treat any string as "on"; every read goes through the mode
    # helper instead.
    from abtem.core import config
    from abtem.core.grid import _fast_fft_rounding_mode

    with config.set({"grid.round-to-fast-fft": "sometimes"}):
        with pytest.raises(ValueError, match="must be True, False or 'auto'"):
            _fast_fft_rounding_mode()


def test_already_fast_commensurate_grids_are_not_enlarged():
    # Both the nearest commensurate grid and any larger fast multiple are
    # exactly commensurate, so enlarging one that is already fast buys no
    # accuracy -- it only costs memory and FFT time (up to +23 % pixels).
    from abtem.core import config

    for atoms in (
        bulk("Si", "diamond", a=5.431, cubic=True) * (2, 2, 1),
        bulk("Cu", "fcc", a=3.61, cubic=True) * (6, 6, 1),
    ):
        with config.set({"grid.round-to-fast-fft": False}):
            plain = Potential(atoms, sampling="auto").gpts
        assert all(is_fast_fft_size(n) for n in plain)  # guard the premise

        with config.set({"grid.round-to-fast-fft": "auto"}):
            assert Potential(atoms, sampling="auto").gpts == plain


def test_incommensurate_fallback_respects_its_overshoot_window():
    # Rutile has no commensurate grid, so the fallback trades pixels for
    # alignment -- but only inside the documented window. Scoring the
    # candidate that terminates the search let a size 17 % above the target
    # (38 % more pixels) win from outside it.
    from ase.spacegroup import crystal

    rutile = crystal(
        ["Ti", "O"],
        basis=[(0, 0, 0), (0.30478, 0.30478, 0)],
        spacegroup=136,
        cellpar=[4.5937, 4.5937, 2.9587, 90, 90, 90],
    ) * (6, 6, 1)

    potential = Potential(rutile, sampling="auto")

    for n, extent in zip(potential.gpts, potential.extent):
        n_target = int(np.ceil(extent / 0.05))
        # Written out rather than imported from the module under test, which
        # would move both sides of the assertion together.
        assert n <= n_target * 1.12
