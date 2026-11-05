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
    """Pixels follow the fftshifted layout of a diffraction pattern: pixel p
    of an n-pixel axis holds spatial frequency f = p - n // 2 (in units of
    the sampling) and covers [(f - 1/2) s, (f + 1/2) s). So a vector
    g = (f + d) s with |d| < 1/2 lands on p = f + n // 2. Non-square shape
    and unequal samplings catch an x/y swap; offsets of +-0.3 pixel catch
    a missing half-pixel shift of the edges."""
    shape, sampling = (8, 10), (0.1, 0.05)
    f_and_d = [
        ((0, 0.0), (0, 0.0)),    # origin -> (4, 5)
        ((1, -0.3), (-4, 0.3)),  # -> (5, 1)
        ((-3, 0.3), (4, -0.3)),  # -> (1, 9)
        ((3, -0.3), (-5, 0.3)),  # -> (7, 0)
    ]
    g = np.array(
        [
            [(fx + dx) * sampling[0], (fy + dy) * sampling[1]]
            for (fx, dx), (fy, dy) in f_and_d
        ]
    )
    expected = np.array(
        [[fx + shape[0] // 2, fy + shape[1] // 2] for (fx, _), (fy, _) in f_and_d]
    )
    np.testing.assert_array_equal(
        _find_projected_pixel_index(g, shape, sampling), expected
    )


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

    def test_duplicate_spots_smallest_excitation_error_survives(self):
        """Of spots sharing a pixel, the one closest to the Bragg condition
        (smallest |sg|, not smallest signed sg, not the first listed) is
        kept; a spot alone on its pixel always survives."""
        nm = np.array([[2, 2], [5, 5], [2, 2], [2, 2]])
        sg = np.array([0.3, 9.0, 0.1, -0.2])
        mask = overlapping_spots_mask(nm, sg)
        np.testing.assert_array_equal(mask, [False, True, True, False])


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

@pytest.mark.parametrize("radius", [None, "1.5 pixels"])
def test_index_diffraction_spots(radius):
    """A pattern with delta spots at known lattice positions.

    For cubic Al (a = 4.05 A) g_hkl = (h, k, l) / a [1/A, no 2 pi], so with
    reciprocal sampling 1/(4a) the spot (h, k, .) sits 4h, 4k pixels from
    the centre pixel (32, 32) of a 64 x 64 fftshifted pattern. Each spot
    gets a distinct intensity, so a swapped or mis-scaled index reads the
    wrong one. (0, 0, 1) projects onto the same pixel as (0, 0, 0) but has
    the larger |sg| (sg(000) = 0), so it must be masked to zero. With a
    radius of 1.5 pixels the integration disks (weight 1 at the centre)
    do not reach the neighbouring spots, 4 pixels away.
    """
    atoms = bulk("Al", cubic=True)
    a = atoms.cell[0, 0]
    sampling = (1 / (4 * a), 1 / (4 * a))
    spots = {
        (0, 0, 0): ((32, 32), 10.0),
        (1, 0, 0): ((36, 32), 1.0),
        (0, 1, 0): ((32, 36), 2.0),
        (-1, 0, 0): ((28, 32), 3.0),
        (1, 1, 0): ((36, 36), 5.0),
        (2, -1, 0): ((40, 28), 7.0),
    }
    arr = np.zeros((64, 64))
    for (n, m), intensity in spots.values():
        arr[n, m] = intensity

    hkl = np.array(list(spots) + [(0, 0, 1)])
    expected = np.array([intensity for _, intensity in spots.values()] + [0.0])

    kwargs = {} if radius is None else {"radius": 1.5 * sampling[0]}
    result = index_diffraction_spots(
        arr, hkl, sampling, atoms.cell, energy=100e3, **kwargs
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# miller_to_miller_bravais
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hkl, expected",
    [
        # Planes: (h k l) -> (h k i l), i = -(h + k), derived by hand.
        ((1, 0, 0), (1, 0, -1, 0)),   # a prism plane, (10-10)
        ((1, 0, 1), (1, 0, -1, 1)),   # a pyramidal plane, (10-11)
        ((2, 1, 3), (2, 1, -3, 3)),
        ((1, 1, 0), (1, 1, -2, 0)),   # (11-20)
        ((0, 0, 1), (0, 0, 0, 1)),    # the basal plane, (0001)
    ],
)
def test_miller_bravais_known_values(hkl, expected):
    assert tuple(miller_to_miller_bravais(hkl)) == expected


@given(
    h=st.integers(-5, 5),
    k=st.integers(-5, 5),
    l=st.integers(-5, 5),
)
def test_miller_bravais_indices_are_the_plane_intercepts(h, k, l):
    """Geometric oracle: a plane's Miller(-Bravais) indices are g . a_j for
    its reciprocal-lattice vector g and each axis a_j. Using a real
    hexagonal cell with the three basal axes a1, a2, a3 = -(a1 + a2) and c,
    the four indices must be (g.a1, g.a2, g.a3, g.c)."""
    cell = Cell.new([3.2, 3.2, 5.2, 90, 90, 120])
    a1, a2, c = np.array(cell)
    a3 = -(a1 + a2)
    g = np.array([h, k, l]) @ np.linalg.inv(np.array(cell)).T
    intercepts = [g @ axis for axis in (a1, a2, a3, c)]
    np.testing.assert_allclose(
        miller_to_miller_bravais((h, k, l)), intercepts, atol=1e-9
    )
