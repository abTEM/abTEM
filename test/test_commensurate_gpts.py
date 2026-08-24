"""Tests for commensurate automatic grids and their interplay with fast FFT sizes."""

import numpy as np
import pytest
from ase.build import bulk

from abtem.core.fft import is_fast_fft_size, next_fast_fft_size
from abtem.potentials.iam import Potential
from abtem.slicing import commensurate_gpts


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
    atoms = bulk(name, crystalstructure, a=a, cubic=True) * (2, 2, 1)

    potential = Potential(atoms, sampling="auto", projection="infinite")

    for axis, (n, extent) in enumerate(zip(potential.gpts, potential.extent)):
        assert is_fast_fft_size(n)
        assert extent / n <= 0.05 + 1e-12
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
