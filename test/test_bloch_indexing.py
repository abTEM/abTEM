"""Tests for abtem/bloch/indexing.py"""

import numpy as np
import pytest
from ase.build import bulk
from ase.cell import Cell
from hypothesis import given
from hypothesis import strategies as st

from abtem.bloch.indexing import (
    _find_projected_pixel_index,
    _pixel_edges,
    antialiased_disk,
    create_ellipse,
    estimate_necessary_excitation_error,
    index_diffraction_spots,
    integrate_ellipse_around_pixels,
    miller_to_miller_bravais,
    overlapping_spots_mask,
    validate_cell,
)


# ---------------------------------------------------------------------------
# _pixel_edges
# ---------------------------------------------------------------------------

@given(
    shape=st.tuples(st.integers(4, 32), st.integers(4, 32)),
    sampling=st.tuples(st.floats(0.05, 0.5), st.floats(0.05, 0.5)),
)
def test_pixel_edges(shape, sampling):
    x, y = _pixel_edges(shape, sampling)
    assert x.shape == (shape[0],) and y.shape == (shape[1],)
    assert np.allclose(np.diff(x), sampling[0])
    assert np.allclose(np.diff(y), sampling[1])


# ---------------------------------------------------------------------------
# _find_projected_pixel_index
# ---------------------------------------------------------------------------

def test_find_projected_pixel_index():
    g = np.array([[0.0, 0.0], [0.05, 0.05]])
    nm = _find_projected_pixel_index(g, (8, 8), (0.1, 0.1))
    assert nm.shape == (2, 2)
    # origin should map somewhere in-bounds
    assert 0 <= nm[0, 0] < 8 and 0 <= nm[0, 1] < 8


# ---------------------------------------------------------------------------
# estimate_necessary_excitation_error
# ---------------------------------------------------------------------------

def test_excitation_error_positive_and_monotone():
    sg1 = estimate_necessary_excitation_error(energy=100e3, k_max=1.0)
    sg2 = estimate_necessary_excitation_error(energy=100e3, k_max=4.0)
    assert sg1 > 0 and sg2 > sg1


# ---------------------------------------------------------------------------
# validate_cell
# ---------------------------------------------------------------------------

class TestValidateCell:
    def test_from_atoms(self):
        assert isinstance(validate_cell(bulk("Al", cubic=True)), Cell)

    def test_from_float(self):
        cell = validate_cell(4.05)
        assert isinstance(cell, Cell) and np.allclose(np.diag(cell), [4.05, 4.05, 4.05])

    def test_from_1d_array(self):
        cell = validate_cell(np.array([4.0, 4.0, 4.0]))
        assert isinstance(cell, Cell) and np.allclose(np.diag(cell), [4.0, 4.0, 4.0])

    def test_from_3x3_array(self):
        assert isinstance(validate_cell(np.diag([4.0, 4.0, 4.0])), Cell)

    def test_from_cell_object(self):
        original = bulk("Al", cubic=True).cell
        assert isinstance(validate_cell(original), Cell)

    def test_invalid_raises(self):
        with pytest.raises((ValueError, TypeError)):
            validate_cell("not_a_cell")


# ---------------------------------------------------------------------------
# create_ellipse
# ---------------------------------------------------------------------------

@given(a=st.integers(1, 10), b=st.integers(1, 10))
def test_create_ellipse_shape_and_mask(a, b):
    e = create_ellipse(a, b)
    assert e.shape == (2 * a + 1, 2 * b + 1)
    assert e[a, b]      # center always True
    assert not e[0, 0]  # corner: (1/a)^2 + (1/b)^2 = 2 > 1 for any a,b >= 1


def test_create_ellipse_zero_axes():
    e = create_ellipse(0, 0)
    assert e.shape == (1, 1) and e[0, 0]


# ---------------------------------------------------------------------------
# antialiased_disk
# ---------------------------------------------------------------------------

@given(
    radius=st.floats(0.5, 5.0),
    sampling=st.tuples(st.floats(0.1, 0.5), st.floats(0.1, 0.5)),
)
def test_antialiased_disk_properties(radius, sampling):
    d = antialiased_disk(radius, sampling)
    assert d.ndim == 2
    assert np.all(d >= 0.0) and np.all(d <= 1.0)
    cy, cx = d.shape[0] // 2, d.shape[1] // 2
    assert d[cy, cx] == 1.0


def test_antialiased_disk_monotone_in_radius():
    d1 = antialiased_disk(1.0, (0.2, 0.2))
    d2 = antialiased_disk(3.0, (0.2, 0.2))
    assert d2.sum() > d1.sum()


# ---------------------------------------------------------------------------
# overlapping_spots_mask
# ---------------------------------------------------------------------------

class TestOverlappingSpotsMask:
    def test_unique_spots_all_true(self):
        nm = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        mask = overlapping_spots_mask(nm, np.ones(4))
        assert mask.shape == (4,) and mask.all()

    def test_duplicate_spots_one_survives(self):
        nm = np.array([[2, 2], [2, 2]])
        mask = overlapping_spots_mask(nm, np.array([1.0, 2.0]))
        assert mask.sum() == 1


# ---------------------------------------------------------------------------
# integrate_ellipse_around_pixels
# ---------------------------------------------------------------------------

def test_integrate_ellipse_around_pixels():
    arr = np.ones((16, 16))
    nm = np.array([[8, 8], [4, 4]])
    result = integrate_ellipse_around_pixels(arr, nm, 1.0, (1.0, 1.0))
    assert result.shape == (2,) and result[0] > 0
    # with priority weights
    priority = np.array([2.0, 1.0])
    result_p = integrate_ellipse_around_pixels(arr, nm, 1.0, (1.0, 1.0), priority)
    assert result_p.shape == (2,)


# ---------------------------------------------------------------------------
# index_diffraction_spots
# ---------------------------------------------------------------------------

def test_index_diffraction_spots():
    atoms = bulk("Al", cubic=True)
    hkl = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]])
    arr = np.abs(np.random.default_rng(42).standard_normal((64, 64)))
    result = index_diffraction_spots(arr, hkl, (1 / 64, 1 / 64), atoms.cell, energy=100e3)
    assert result.shape == (4,) and np.all(result >= 0)
    # With explicit radius
    result_r = index_diffraction_spots(
        arr, hkl, (1 / 64, 1 / 64), atoms.cell, energy=100e3, radius=2.0
    )
    assert result_r.shape == (4,)


# ---------------------------------------------------------------------------
# miller_to_miller_bravais
# ---------------------------------------------------------------------------

@given(
    h=st.integers(-5, 5),
    k=st.integers(-5, 5),
    l=st.integers(-5, 5),
)
def test_miller_bravais_constraint(h, k, l):
    H, K, I, L = miller_to_miller_bravais((h, k, l))
    assert H + K + I == 0 and L == l


def test_miller_bravais_known_values():
    H, K, I, L = miller_to_miller_bravais((1, 1, 0))
    assert H == 1 and K == 1 and I == -2 and L == 0
