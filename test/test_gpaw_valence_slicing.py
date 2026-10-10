"""The planes of GPAW's valence potential must be integrated into the slices once.

Plane k is the sample at z = k dz and stands for the heights within dz / 2 of it,
so a slice takes each plane times the length of its heights inside the slice, and
the weights of every plane add up to dz over the slices of the cell.
"""

import sys
from fractions import Fraction

import numpy as np
import pytest
from ase import Atoms

import abtem
from abtem.potentials.gpaw import integrate_slice
from abtem.potentials.iam import Potential

try:
    from gpaw import GPAW, PW

    from abtem.potentials.gpaw import GPAWPotential
except ImportError:
    pass

CELL = (3.2, 2.8, 3.6)
NZ = 36  # planes of the valence potential along z, spacing 0.1 A
SLICE_THICKNESSES = [
    0.1,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.2,
    1.7,
    3.6,
]


def _slice_limits(slice_thickness):
    """The slice limits and the depth GPAWPotential passes to integrate_slice."""
    atoms = Atoms("C", positions=[(1.0, 1.0, 1.0)], cell=CELL, pbc=True)
    potential = Potential(atoms, sampling=0.5, slice_thickness=slice_thickness)
    return potential.get_sliced_atoms().slice_limits, potential.thickness


@pytest.mark.parametrize("slice_thickness", SLICE_THICKNESSES + [0.05, 0.15])
def test_the_weights_of_every_plane_add_up_to_its_spacing(slice_thickness):
    limits, depth = _slice_limits(slice_thickness)
    gpts = (4, 3)

    # A potential that is 1 on plane k and 0 elsewhere: the slices must add up to
    # the plane's weight dz once.
    dz = depth / NZ
    for plane in range(NZ):
        array = np.zeros(gpts + (NZ,))
        array[..., plane] = 1.0
        total = sum(
            integrate_slice(array, gpts, a, b, depth).sum() / np.prod(gpts)
            for a, b in limits
        )
        assert total == pytest.approx(dz, rel=1e-12)


def _plane_weights(start, stop):
    """
    The weights, in plane spacings, of the NZ planes in a slice from `start` to
    `stop` plane spacings, as exact fractions. Plane j stands for the heights from
    j - 1/2 to j + 1/2, and plane 0 also for those from NZ - 1/2 to NZ.
    """
    weights = []
    for j in range(NZ):
        cells = [(j - Fraction(1, 2), j + Fraction(1, 2))]
        if j == 0:
            cells.append((NZ - Fraction(1, 2), NZ + Fraction(1, 2)))
        weights.append(
            sum(max(Fraction(0), min(stop, hi) - max(start, lo)) for lo, hi in cells)
        )
    return weights


@pytest.mark.parametrize("slice_thickness", SLICE_THICKNESSES)
def test_each_plane_is_weighted_by_its_heights_in_the_slice(slice_thickness):
    # n equal slices of nz planes: slice k runs from k * nz / n to (k + 1) * nz / n
    # plane spacings. The oracle uses no float slice limits. A plane has the value
    # j + 1, so the integral of a slice identifies which planes it holds and with
    # what weights, not only how many.
    limits, depth = _slice_limits(slice_thickness)
    gpts = (2, 2)
    n = len(limits)
    dz = depth / NZ
    array = np.ones(gpts + (NZ,)) * np.arange(1, NZ + 1)

    for k, (a, b) in enumerate(limits):
        weights = _plane_weights(Fraction(k * NZ, n), Fraction((k + 1) * NZ, n))
        expected = dz * float(sum(w * (j + 1) for j, w in enumerate(weights)))
        result = integrate_slice(array, gpts, a, b, depth)
        assert result == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("slice_thickness", SLICE_THICKNESSES + [0.05, 0.15])
def test_slice_integrals_of_a_smooth_potential_are_centred_on_the_slices(
    slice_thickness,
):
    # The planes of cos(q z + 0.3), one period over the depth, against its exact
    # integral through each slice. The error of the planes is of second order in
    # dz: the trapezoidal rule over the whole planes, (q dz)^2 / 12 of a slice, and
    # at most dz^2 / 8 times the largest slope at each limit inside a plane's
    # heights. Taking the planes from floor(a / dz) whole put each slice integral
    # dz / 2 below its slice, up to 0.14 off here (0.8 A), and gave slices thinner
    # than a plane either none or a whole plane.
    limits, depth = _slice_limits(slice_thickness)
    gpts = (2, 2)
    dz = depth / NZ
    q = 2 * np.pi / depth
    z = np.arange(NZ) * dz
    array = np.ones(gpts + (1,)) * np.cos(q * z + 0.3)

    def antiderivative(z):
        return np.sin(q * z + 0.3) / q

    result = np.array([integrate_slice(array, gpts, a, b, depth) for a, b in limits])
    exact = np.array([antiderivative(b) - antiderivative(a) for a, b in limits])
    exact = exact[:, None, None] * np.ones(gpts)

    scale = np.abs(exact).max()
    atol = (q * dz) ** 2 / 12 * scale + 2 * dz**2 / 8 * q
    np.testing.assert_allclose(result, exact, rtol=0, atol=atol)

    # With every limit on a plane the trapezoidal rule integrates the one harmonic
    # to (q dz / 2) cot(q dz / 2) of its integral, without a shift.
    planes = np.array(limits) / dz
    if np.allclose(planes, np.rint(planes), rtol=0, atol=1e-9):
        quadrature = (q * dz / 2) / np.tan(q * dz / 2)
        np.testing.assert_allclose(result, quadrature * exact, rtol=0, atol=1e-12)


@pytest.mark.parametrize("slice_thickness", SLICE_THICKNESSES)
def test_slices_of_the_valence_potential_add_up_to_the_whole_cell(slice_thickness):
    limits, depth = _slice_limits(slice_thickness)
    gpts = (8, 6)
    array = np.random.default_rng(0).normal(size=gpts + (NZ,))

    slices = [integrate_slice(array, gpts, a, b, depth) for a, b in limits]

    assert all(np.isfinite(s).all() for s in slices)
    whole = array.sum(-1) * depth / NZ
    np.testing.assert_allclose(
        np.sum(slices, 0), whole, rtol=0, atol=1e-12 * np.abs(whole).max()
    )


def test_a_slice_thinner_than_a_plane_gets_its_part_of_the_plane():
    # 0.05 A slices on 0.1 A planes: slice m lies within the heights of plane
    # (m + 1) // 2 (plane 0 for the last), and gets 0.05 A of it.
    limits, depth = _slice_limits(0.05)
    gpts = (8, 6)
    array = np.random.default_rng(1).normal(size=gpts + (NZ,))

    slices = [integrate_slice(array, gpts, a, b, depth) for a, b in limits]

    for m, s in enumerate(slices):
        np.testing.assert_allclose(
            s, 0.05 * array[..., ((m + 1) // 2) % NZ], rtol=0, atol=1e-12
        )
    whole = array.sum(-1) * depth / NZ
    np.testing.assert_allclose(
        np.sum(slices, 0), whole, rtol=0, atol=1e-12 * np.abs(whole).max()
    )


@pytest.mark.parametrize("limit", [0.25, 0.42, 1.0, 2.93])
def test_a_limit_splits_the_plane_whose_heights_it_lies_in(limit):
    # Plane j (from (j - 1/2) dz to (j + 1/2) dz) contains the limit when
    # j = floor(limit / dz + 1/2); the slices on either side share it by the
    # length of its heights on their side. A limit on a plane (1.0) halves it.
    gpts = (4, 3)
    dz = CELL[2] / NZ
    plane = int(np.floor(limit / dz + 0.5))
    array = np.zeros(gpts + (NZ,))
    array[..., plane] = 1.0
    before = integrate_slice(array, gpts, 0.0, limit, CELL[2])
    in_slice = integrate_slice(array, gpts, limit, limit + 0.5, CELL[2])
    np.testing.assert_allclose(before, limit - (plane - 0.5) * dz, rtol=0, atol=1e-12)
    np.testing.assert_allclose(in_slice, (plane + 0.5) * dz - limit, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "reps", [(1, 1, 1), (2, 3, 1), (3, 1, 2), (1, 2, 3), (2, 1, 5)]
)
@pytest.mark.parametrize("slice_thickness", [0.1, 0.3, 0.4, 0.7, 1.1])
@pytest.mark.parametrize("multiple", [True, False])
@pytest.mark.parametrize("strided", [False, True])
def test_integrate_slice_of_one_period_equals_that_of_the_repeated_grid(
    reps, slice_thickness, multiple, strided
):
    # One period of 6 x 5 x 7 points, so the three axes differ, with a plane
    # spacing of 0.1 A along the last axis.
    array = np.random.default_rng(0).normal(size=(6, 5, 7))
    if strided:
        # As `_generate_slices` passes it for plane="yz": the planes lie along
        # the first axis of the calculator's grid.
        array = np.moveaxis(array, (1, 2), (0, 1))
    shape = array.shape
    gpts = (shape[0] * reps[0], shape[1] * reps[1])
    if not multiple:
        gpts = (gpts[0] + 3, gpts[1] - 1)
    dz = 0.1
    length = shape[2] * reps[2] * dz
    atoms = Atoms("C", positions=[(0.1, 0.1, 0.1)], cell=(1.0, 1.0, length), pbc=True)
    potential = Potential(atoms, sampling=0.5, slice_thickness=slice_thickness)
    limits = list(potential.get_sliced_atoms().slice_limits)
    # Slices from 1.5 planes below to 1.5 planes above each face between two
    # periods take planes from both; the slice limits often land on the faces.
    faces = [k * shape[2] * dz for k in range(1, reps[2])]
    limits += [(z - 1.5 * dz, z + 1.5 * dz) for z in faces]
    # A slice from inside the first period to inside the last holds the end of a
    # period, whole periods and the start of a period.
    limits.append((2.5 * dz, length - 1.5 * dz))
    repeated = np.tile(array, reps)

    for a, b in limits:
        expected = integrate_slice(repeated, gpts, a, b, potential.thickness)
        result = integrate_slice(array, gpts, a, b, potential.thickness, reps)
        np.testing.assert_allclose(
            result, expected, rtol=0, atol=1e-13 * np.abs(repeated).max()
        )


@pytest.fixture(scope="module")
def calculator():
    atoms = Atoms(
        "CO",
        positions=[(0.6, 0.8, 1.0), (1.9, 1.5, 2.4)],
        cell=CELL,
        pbc=True,
    )
    atoms.calc = GPAW(mode=PW(250), h=0.2, txt=None, symmetry="off")
    atoms.get_potential_energy()
    return atoms.calc


@pytest.fixture
def double_precision_numpy_fft():
    with abtem.config.set({"precision": "float64", "fft": "numpy"}):
        yield


needs_gpaw = pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")


@needs_gpaw
@pytest.mark.parametrize("slice_thickness", SLICE_THICKNESSES)
def test_gpaw_potential_slices_add_up_to_the_projection(
    calculator, double_precision_numpy_fft, slice_thickness
):
    gpts = (32, 28)
    whole = GPAWPotential(calculator, gpts=gpts, slice_thickness=3.6).build(lazy=False)
    sliced = GPAWPotential(
        calculator, gpts=gpts, slice_thickness=slice_thickness
    ).build(lazy=False)

    assert np.isfinite(sliced.array).all()
    # The slices add up to the one-slice projection to the accuracy of the
    # per-atom core corrections (6.6e-7 of the maximum).
    error = np.abs(sliced.array.sum(0) - whole.array[0]).max()
    assert error < 1e-5 * np.abs(whole.array).max()


@needs_gpaw
@pytest.mark.parametrize(
    "plane, gpts, length", [("xz", (32, 36), 2.8), ("yz", (28, 36), 3.2)]
)
@pytest.mark.parametrize("slice_thickness", [0.1, 0.35, 0.4, 0.7, 0.8])
def test_gpaw_potential_slices_add_up_to_the_projection_in_other_planes(
    calculator, double_precision_numpy_fft, plane, gpts, length, slice_thickness
):
    whole = GPAWPotential(
        calculator, gpts=gpts, slice_thickness=length, plane=plane
    ).build(lazy=False)
    sliced = GPAWPotential(
        calculator, gpts=gpts, slice_thickness=slice_thickness, plane=plane
    ).build(lazy=False)

    assert np.isfinite(sliced.array).all()
    error = np.abs(sliced.array.sum(0) - whole.array[0]).max()
    assert error < 1e-5 * np.abs(whole.array).max()


@needs_gpaw
@pytest.mark.parametrize("reps", [(1, 1, 2), (2, 3, 2)])
@pytest.mark.parametrize("slice_thickness", [0.1, 0.3, 0.4, 0.6])
def test_repeated_gpaw_potential_in_thin_slices_is_the_one_cell_potential_tiled(
    calculator, double_precision_numpy_fft, reps, slice_thickness
):
    gpts = (32, 28)
    unit = GPAWPotential(calculator, gpts=gpts, slice_thickness=slice_thickness).build(
        lazy=False
    )
    repeated = GPAWPotential(
        calculator,
        gpts=(gpts[0] * reps[0], gpts[1] * reps[1]),
        slice_thickness=slice_thickness,
        repetitions=reps,
    ).build(lazy=False)

    tiled = np.tile(unit.array, (reps[2], reps[0], reps[1]))
    assert repeated.array.shape == tiled.shape
    np.testing.assert_allclose(
        repeated.array, tiled, rtol=0, atol=1e-5 * np.abs(tiled).max()
    )
