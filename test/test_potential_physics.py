"""Independent-oracle physics tests for the potentials.

Every expected value here comes from something other than the code path under
test: a parametrization's own documented analytic formula, a sum rule, a
translation, or an independent 1D quadrature (``scipy.integrate``). None is
pasted output, and none re-implements the integrator being checked.

Tolerances are derived from the numerics named next to them, not fitted to
the observed error. Where an oracle exposed a genuine discrepancy the test is
marked ``xfail(strict=True)`` with the evidence rather than loosened.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.data import chemical_symbols
from scipy import integrate
from utils import devices

from abtem.core import config
from abtem.core.backend import asnumpy
from abtem.integrals import (
    GaussianProjectionIntegrals,
    QuadratureProjectionIntegrals,
)
from abtem.parametrizations import (
    KirklandParametrization,
    LobatoParametrization,
    PengParametrization,
)
from abtem.potentials.iam import Potential

PARAMETRIZATIONS = {
    "lobato": LobatoParametrization,
    "kirkland": KirklandParametrization,
}
ELEMENTS = (6, 14, 29, 79)
PRECISIONS = ("float32", "float64")


def _f0(parametrization, Z) -> float:
    """``projected_scattering_factor(k=0)``.

    In abTEM's units the projected scattering factor is the 2D Fourier
    transform of the projected potential, F(k) = int V_p(r) exp(-2 pi i k.r) d^2r.
    Check for Lobato: V_p = 2 sum_i [2 a_i/b_i K0(b_i r) + a_i r K1(b_i r)],
    and int_0^inf K0(br) r dr = 1/b^2, int_0^inf r^2 K1(br) dr = 2/b^3, so
    int V_p dA = 16 pi sum a_i/b_i^3 = 8 pi sum (a/b^3 + a b/b^4) = F(0).
    Hence F(0) = int V_p dA = int V d^3r [eV/e A^3], with no further factor.

    Evaluated at float64 regardless of the configured precision: this is the
    oracle, and ``get_function`` casts the parameters to the configured dtype.
    """
    with config.set({"precision": "float64"}):
        f = parametrization.projected_scattering_factor(chemical_symbols[Z])
        return float(f(np.array([0.0]))[0])


def _parameter_rounding(parametrization, Z, precision) -> float:
    """Relative change of F(0) when the parameters are stored at `precision`.

    ``get_function`` casts the parameters to the configured dtype, and the
    Lobato sum has nearly cancelling terms -- for carbon, float32 storage alone
    moves F(0) by 1.8e-5, ~300 eps. That is a property of the parameters, not
    of the integrator, so it is added to the round-off budget. Computed from
    the analytic formula at the two precisions, not from the code under test.
    """
    with config.set({"precision": precision}):
        f = parametrization.projected_scattering_factor(chemical_symbols[Z])
        low = float(f(np.array([0.0]))[0])
    return abs(low / _f0(parametrization, Z) - 1)


def _eps(precision) -> float:
    return float(np.finfo(np.dtype(precision)).eps)


def _slab_integral(V, a: float, b: float, r_max: float = 60.0) -> float:
    """int_{a < z < b} V(|r|) d^3r for a radial function centred at the origin.

    Archimedes' hat-box theorem: the area of the sphere of radius r lying
    between the planes z = a and z = b is 2 pi r h(r), with
    h(r) = |[a, b] n [-r, r]|. So the slab integral is the 1D integral
    2 pi int_0^inf r V(r) h(r) dr, done here with scipy.integrate.quad --
    independent of abTEM's radial tables and Gauss-Legendre quadrature.
    r V(r) stays finite at r = 0 for a screened Coulomb potential, and h has
    kinks only at |a| and |b|, which are passed as breakpoints.
    """

    def integrand(r):
        h = max(0.0, min(b, r) - max(a, -r))
        return 2 * np.pi * r * float(V(np.array([r]))[0]) * h

    total, lower = 0.0, 0.0
    for upper in sorted({abs(a), abs(b)}) + [r_max]:
        if upper > lower:
            total += integrate.quad(
                integrand, lower, upper, limit=200, epsabs=1e-12, epsrel=1e-10
            )[0]
            lower = upper
    return total


def _periodic_slab_integrals(V, limits, z0, Lz, images=(-1, 0, 1)):
    """``_slab_integral`` per slice for an atom at z0 plus its images along z."""
    return np.array(
        [
            sum(_slab_integral(V, a - z0 - n * Lz, b - z0 - n * Lz) for n in images)
            for a, b in limits
        ]
    )


# --------------------------------------------------------------------------
# Task 1: infinite projection against the analytic projected potential
# --------------------------------------------------------------------------

# 12 A cell: the nearest periodic image of the atom is > 11 A from every point
# of the r < 1 A band tested, and V_p(11 A) / V_p(1 A) < 1e-8 for all four
# elements and both parametrizations, so the single-atom analytic V_p is the
# oracle to far better than the tolerance.
_CELL = 12.0
# Deliberately off-grid, so the bilinear delta superposition is exercised.
_OFFSET = (0.0071, 0.0133)


def _infinite_projection(parametrization, Z, gpts, device, periodic=True, pos=None):
    if pos is None:
        pos = (_CELL / 2 + _OFFSET[0], _CELL / 2 + _OFFSET[1], _CELL / 2)
    atoms = Atoms([Z], positions=[pos], cell=[_CELL] * 3)
    potential = Potential(
        atoms,
        gpts=gpts,
        slice_thickness=_CELL,
        projection="infinite",
        parametrization=parametrization,
        periodic=periodic,
        device=device,
    )
    projected = asnumpy(potential.build(lazy=False).project().array)
    return projected, pos, potential.sampling


def _band_relative_error(projected, pos, sampling, parametrization, Z, band):
    x = np.arange(projected.shape[0]) * sampling[0]
    y = np.arange(projected.shape[1]) * sampling[1]
    R = np.hypot(x[:, None] - pos[0], y[None, :] - pos[1])
    mask = (R > band[0]) & (R < band[1])
    analytic = parametrization.projected_potential(chemical_symbols[Z])(R[mask])
    return projected[mask] / analytic - 1


@devices
@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("Z", ELEMENTS)
@pytest.mark.parametrize("name", list(PARAMETRIZATIONS))
def test_infinite_projection_matches_analytic_projected_potential(
    name, Z, precision, device
):
    """V_p(r) from the FFT integrator vs the parametrization's closed form.

    Oracle: ``projected_potential`` (K0/K1 Bessel sums for Lobato, K0 plus
    Gaussians for Kirkland) -- the analytic z-integral of the radial
    potential -- at the exact distance of each pixel from the atom.

    Tolerance, 1 %: the grid holds the band-limited (k < Nyquist) version of a
    function with a log-singular core, so pixel values carry a Gibbs-like
    error falling off as dx^2 (checked by
    test_infinite_projection_error_is_second_order_in_sampling). The band
    excludes r < 0.1 A (5 pixels of that core) and r > 1 A, where V_p is small
    enough that the O(dx^2) offset the band limit spreads over the cell turns
    into a large *relative* error. Float32 round-off (~1e-6) is negligible.
    A 2 % error in the scattering factor cannot hide inside a 1 % bound.
    """
    parametrization = PARAMETRIZATIONS[name]()
    with config.set({"precision": precision}):
        projected, pos, sampling = _infinite_projection(parametrization, Z, 600, device)
    rel = _band_relative_error(projected, pos, sampling, parametrization, Z, (0.1, 1.0))
    assert np.abs(rel).max() < 1e-2, np.abs(rel).max()


@pytest.mark.parametrize("Z", (6, 79))
def test_infinite_projection_error_is_second_order_in_sampling(Z):
    """Halving dx must cut the band error by ~4 (second order), not ~2.

    This is what licenses the tolerance above: the residual is discretisation
    error of the band-limited core, converging at the order that model
    predicts. A wrong prefactor or a misplaced atom does not converge at all.
    2.5 separates second order (4) from first order (2), with room for the
    pre-asymptotic regime at dx = 0.04 A.
    """
    parametrization = LobatoParametrization()
    errors = []
    with config.set({"precision": "float64"}):
        for gpts in (300, 600):
            projected, pos, sampling = _infinite_projection(
                parametrization, Z, gpts, "cpu"
            )
            rel = _band_relative_error(
                projected, pos, sampling, parametrization, Z, (0.1, 1.0)
            )
            errors.append(np.abs(rel).max())
    assert errors[0] / errors[1] > 2.5, errors


@devices
@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("periodic", [True, False], ids=["periodic", "nonperiodic"])
@pytest.mark.parametrize("Z", ELEMENTS)
@pytest.mark.parametrize("name", list(PARAMETRIZATIONS))
def test_infinite_projection_sum_rule(name, Z, periodic, precision, device):
    """int V_p dA = F(0), the k = 0 sum rule (units: see _f0).

    Exact for the FFT method: the k = 0 coefficient of the grid is
    sum(V) dx dy, and the superposed delta has unit weight however it is split
    between pixels. The only error is round-off, bounded by
    10 eps log2(N) for the O(log N) pairwise-summation and FFT round-off of an
    N-pixel grid, plus the float32 storage error of the parameters themselves
    (_parameter_rounding). The second position sits on the cell corner, testing the
    wrap of a delta split across the periodic boundary.
    """
    parametrization = PARAMETRIZATIONS[name]()
    f0 = _f0(parametrization, Z)
    tolerance = 10 * _eps(precision) * np.log2(600**2) + _parameter_rounding(
        parametrization, Z, precision
    )
    for pos in [None, (0.003, _CELL - 0.004, 1.0)]:
        with config.set({"precision": precision}):
            projected, _, sampling = _infinite_projection(
                parametrization, Z, 600, device, periodic=periodic, pos=pos
            )
        total = projected.astype(np.float64).sum() * np.prod(sampling)
        assert abs(total / f0 - 1) < tolerance, (pos, total / f0 - 1)


@devices
def test_infinite_projection_is_translation_invariant_across_the_boundary(device):
    """Moving the atom by whole pixels, through both cell edges, must roll the
    projected potential by exactly that many pixels.

    Oracle: translation invariance of a periodic cell. Integer-pixel shifts
    leave the sub-pixel split of the delta unchanged, so only round-off is
    allowed.
    """
    if device == "mps":
        pytest.skip("Metal is single precision; this test runs in float64")
    parametrization = LobatoParametrization()
    gpts = 240
    dx = _CELL / gpts
    base = (5.0 + 0.3 * dx, 7.0 + 0.6 * dx, 1.0)
    shift = (150, -170)
    moved = (
        (base[0] + shift[0] * dx) % _CELL,
        (base[1] + shift[1] * dx) % _CELL,
        1.0,
    )
    with config.set({"precision": "float64"}):
        a, _, _ = _infinite_projection(parametrization, 29, gpts, device, pos=base)
        b, _, _ = _infinite_projection(parametrization, 29, gpts, device, pos=moved)
    np.testing.assert_allclose(
        np.roll(a, shift, axis=(0, 1)), b, rtol=0, atol=1e-9 * np.abs(a).max()
    )


# --------------------------------------------------------------------------
# Task 2: finite projection z-resolution against 3D slab integrals
# --------------------------------------------------------------------------

_FINITE_CELL = (12.0, 12.0, 8.0)
_FINITE_Z0 = 3.1  # inside slice [3.0, 3.5) of 0.5 A slices
_FINITE_DZ = 0.5

# Cutoff tolerance for the oracle comparisons. At the 1e-4 eV default the
# potential is truncated at 4-5.8 A, which removes 0.1-0.5 % of int V d^3r and
# up to tens of per cent of the (small) slab integrals 3-5 A from the atom --
# a documented approximation, tested separately below. At 1e-7 eV the
# truncated tail is < 1e-5 of every slab integral here, so the untruncated
# analytic integral is the oracle.
_TIGHT_CUTOFF = 1e-7


def _finite_potential(
    parametrization, Z, sampling, device, cutoff_tolerance, pos_xy=None
):
    L = _FINITE_CELL
    if pos_xy is None:
        pos_xy = (L[0] / 2 + 0.013, L[1] / 2 - 0.007)
    atoms = Atoms([Z], positions=[(*pos_xy, _FINITE_Z0)], cell=L)
    integrator = QuadratureProjectionIntegrals(
        parametrization, cutoff_tolerance=cutoff_tolerance
    )
    return Potential(
        atoms,
        sampling=sampling,
        slice_thickness=_FINITE_DZ,
        integrator=integrator,
        device=device,
    )


def _slice_integrals(potential):
    array = asnumpy(potential.build(lazy=False).array).astype(np.float64)
    return array.sum((1, 2)) * np.prod(potential.sampling)


def _slab_oracle(parametrization, Z, potential):
    # Images at z0 +- Lz are within the 1e-7 cutoff (6.5-10.4 A) of the cell's
    # slices, and abTEM pads the atoms along z by that cutoff, so they belong
    # in the oracle; the +-2 Lz images are > 12.9 A away from every slice.
    with config.set({"precision": "float64"}):
        V = parametrization.potential(chemical_symbols[Z])
        return _periodic_slab_integrals(
            V, potential.slice_limits, _FINITE_Z0, _FINITE_CELL[2]
        )


@devices
@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("Z", ELEMENTS)
@pytest.mark.parametrize("name", list(PARAMETRIZATIONS))
def test_finite_projection_slices_match_slab_integrals(name, Z, precision, device):
    """Per-slice int V dA of the finite projection vs the independent
    int_{slab} V d^3r (hat-box 1D quadrature, see _slab_integral).

    Tolerances from the numerics:

    * Slices not containing the atom: smooth in-plane integrand, so the pixel
      sum is spectrally accurate; what remains is linear interpolation of the
      integral table (0.02 A steps in z, a geometric radial grid with ~1.7 %
      spacing, so h^2/8 ~ 4e-5 relative) and the taper beyond 0.85 of the
      cutoff (< 1e-7 eV there). 2e-3 is ~10x that budget.
    * The atom's own slice holds the log-singular core, clamped to the table's
      innermost radius sampling/2; that pixel error scales as dx^2 (see
      test_finite_projection_atom_slice_converges). 2 % at dx = 0.05 A.

    The atom's slice must also be the peak, and the slices must sum to the
    k = 0 sum rule F(0) = int V d^3r -- which the infinite projection meets
    exactly (test_infinite_projection_sum_rule) -- within the atom-slice
    budget.
    """
    parametrization = PARAMETRIZATIONS[name]()
    with config.set({"precision": precision}):
        potential = _finite_potential(parametrization, Z, 0.05, device, _TIGHT_CUTOFF)
        numeric = _slice_integrals(potential)
    oracle = _slab_oracle(parametrization, Z, potential)

    atom_slice = int(_FINITE_Z0 // _FINITE_DZ)
    assert potential.slice_limits[atom_slice][0] <= _FINITE_Z0
    assert _FINITE_Z0 < potential.slice_limits[atom_slice][1]

    rel = numeric / oracle - 1
    others = np.delete(np.abs(rel), atom_slice)
    assert others.max() < 2e-3, rel
    assert abs(rel[atom_slice]) < 2e-2, rel
    assert np.argmax(numeric) == atom_slice
    assert abs(numeric.sum() / _f0(parametrization, Z) - 1) < 2e-2


@pytest.mark.parametrize("Z", (14, 79))
def test_finite_projection_atom_slice_converges(Z):
    """The atom-slice error must shrink as dx^2 when the sampling is halved,
    identifying it as the discretisation error of the singular core (the
    premise of the 2 % bound above). 3 separates second order (4) from
    first (2)."""
    parametrization = LobatoParametrization()
    atom_slice = int(_FINITE_Z0 // _FINITE_DZ)
    errors = []
    with config.set({"precision": "float64"}):
        for sampling in (0.1, 0.05):
            potential = _finite_potential(
                parametrization, Z, sampling, "cpu", _TIGHT_CUTOFF
            )
            numeric = _slice_integrals(potential)
            oracle = _slab_oracle(parametrization, Z, potential)
            errors.append(abs(numeric[atom_slice] / oracle[atom_slice] - 1))
    assert errors[0] / errors[1] > 3, errors


@pytest.mark.parametrize("Z", ELEMENTS)
@pytest.mark.parametrize("name", list(PARAMETRIZATIONS))
def test_finite_projection_total_at_default_cutoff(name, Z):
    """At the default 1e-4 eV cutoff the slices must still sum to F(0) within
    what the truncation can remove.

    Bound: the truncated tail, int_{r > 0.85 r_c} V d^3r (0.85 r_c is where
    the taper starts), computed from the parametrization with the hat-box
    quadrature, plus the 2 % atom-slice budget share of the total (the atom
    slice is < 1/2 of the total, so 1 %). The deficit must also be a deficit:
    truncation and taper only remove potential.
    """
    parametrization = PARAMETRIZATIONS[name]()
    with config.set({"precision": "float64"}):
        integrator = QuadratureProjectionIntegrals(parametrization)
        cutoff = integrator.cutoff(chemical_symbols[Z])
        potential = _finite_potential(parametrization, Z, 0.05, "cpu", 1e-4)
        numeric = _slice_integrals(potential)
        V = parametrization.potential(chemical_symbols[Z])
        tail = (
            4
            * np.pi
            * integrate.quad(
                lambda r: r**2 * float(V(np.array([r]))[0]), 0.85 * cutoff, 60.0
            )[0]
        )
    f0 = _f0(parametrization, Z)
    deficit = 1 - numeric.sum() / f0
    assert -1e-3 < deficit < tail / f0 + 1e-2, (deficit, tail / f0)


def test_finite_and_infinite_projections_agree_on_the_total():
    """Summed over slices, the finite projection is an infinite projection:
    compare the two integrators on the same atom, independently of either's
    parametrization formula. Budget as in the slab test (2 %)."""
    parametrization = LobatoParametrization()
    with config.set({"precision": "float64"}):
        finite = _finite_potential(parametrization, 29, 0.05, "cpu", _TIGHT_CUTOFF)
        finite_total = _slice_integrals(finite).sum()
        infinite = Potential(
            Atoms([29], positions=[(6.013, 5.993, _FINITE_Z0)], cell=_FINITE_CELL),
            sampling=0.05,
            slice_thickness=_FINITE_DZ,
            projection="infinite",
            parametrization=parametrization,
        )
        infinite_total = _slice_integrals(infinite).sum()
    assert abs(finite_total / infinite_total - 1) < 2e-2


# --------------------------------------------------------------------------
# Task 3: GaussianProjectionIntegrals
# --------------------------------------------------------------------------


def _gaussian_potential(Z, slice_thickness, device, gpts=None, sampling=None):
    atoms = Atoms(
        [Z],
        positions=[
            (_FINITE_CELL[0] / 2 + 0.013, _FINITE_CELL[1] / 2 - 0.007, _FINITE_Z0)
        ],
        cell=_FINITE_CELL,
    )
    return Potential(
        atoms,
        gpts=gpts,
        sampling=sampling,
        slice_thickness=slice_thickness,
        integrator=GaussianProjectionIntegrals(cutoff_tolerance=_TIGHT_CUTOFF),
        device=device,
    )


@devices
@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("Z", ELEMENTS)
def test_gaussian_projection_single_slice_matches_analytic(Z, precision, device):
    """With one slice holding the whole atom, the Gaussian integrator (Peng
    Gaussians + infinite-projected Lobato-minus-Peng correction) is by
    construction an infinite projection of Lobato, so it must reproduce
    Lobato's closed-form V_p in the band, and F_Lobato(0) in total.

    Tolerances: the band error is the same band-limit error as the FFT
    integrator (1 %, see the Task 1 test). The total is exact up to the
    documented Gaussian-form residual, _GAUSSIAN_FORM_TOLERANCE = 1e-4 of the
    peak of F (in practice 1.8e-7 for Peng), plus round-off and parameter
    storage as in test_infinite_projection_sum_rule.
    """
    lobato = LobatoParametrization()
    with config.set({"precision": precision}):
        potential = _gaussian_potential(Z, _FINITE_CELL[2], device, sampling=0.02)
        projected = asnumpy(potential.build(lazy=False).project().array)
    pos = (_FINITE_CELL[0] / 2 + 0.013, _FINITE_CELL[1] / 2 - 0.007)
    rel = _band_relative_error(
        projected, pos, potential.sampling, lobato, Z, (0.1, 1.0)
    )
    assert np.abs(rel).max() < 1e-2, np.abs(rel).max()

    total = projected.astype(np.float64).sum() * np.prod(potential.sampling)
    f0 = _f0(lobato, Z)
    tolerance = (
        1e-4
        + 10 * _eps(precision) * np.log2(projected.size)
        + _parameter_rounding(lobato, Z, precision)
        + _parameter_rounding(PengParametrization(), Z, precision)
    )
    assert abs(total / f0 - 1) < tolerance, total / f0 - 1


@devices
@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("slice_thickness", (1.0, 0.5))
@pytest.mark.parametrize("Z", ELEMENTS)
def test_gaussian_projection_slices_follow_the_documented_model(
    Z, slice_thickness, precision, device
):
    """Per slice, the Gaussian integrator is documented to be the exact
    z-integral of the Peng Gaussians over the slice, plus -- in the atom's own
    slice only -- the whole infinitely projected Lobato-minus-Peng correction
    (class docstring, Notes).

    Oracle: the Peng radial potential integrated over each slab by the
    independent hat-box quadrature, plus c = F_Lobato(0) - F_Peng(0) in the
    atom's slice. Slice integrals of an FFT-built slice are its exact k = 0
    coefficient, so the only error sources are round-off/parameter storage
    and the form residual (1e-4 of F's peak); tolerance 2e-4 of F(0).
    """
    lobato, peng = LobatoParametrization(), PengParametrization()
    with config.set({"precision": precision}):
        potential = _gaussian_potential(Z, slice_thickness, device, gpts=256)
        numeric = _slice_integrals(potential)
    with config.set({"precision": "float64"}):
        V = peng.potential(chemical_symbols[Z])
        oracle = _periodic_slab_integrals(
            V, potential.slice_limits, _FINITE_Z0, _FINITE_CELL[2]
        )
    atom_slice = int(_FINITE_Z0 // slice_thickness)
    oracle[atom_slice] += _f0(lobato, Z) - _f0(peng, Z)

    f0 = _f0(lobato, Z)
    tolerance = (
        2e-4
        + 10 * _eps(precision) * np.log2(256**2)
        + _parameter_rounding(lobato, Z, precision)
        + _parameter_rounding(peng, Z, precision)
    )
    assert np.abs(numeric - oracle).max() / f0 < tolerance, (numeric - oracle) / f0


@pytest.mark.parametrize("slice_thickness", (1.0, 0.5))
@pytest.mark.parametrize("Z", ELEMENTS)
def test_gaussian_and_quadrature_slices_agree_within_the_model_bound(
    Z, slice_thickness
):
    """Gaussian vs quadrature integrator, same slab, per slice.

    The Gaussian integrator misplaces between slices whatever part of
    V_Lobato - V_Peng lies outside the atom's slice. That is bounded, per
    slice j, by B_j = int_{slab j} |V_L - V_P| d^3r (+ |c| in the atom's slice,
    c = F_L(0) - F_P(0)), computed with the independent hat-box quadrature.
    The quadrature integrator's own error is budgeted as in
    test_finite_projection_slices_match_slab_integrals (2e-3 of the slice,
    2 % in the atom's slice).
    """
    lobato, peng = LobatoParametrization(), PengParametrization()
    with config.set({"precision": "float64"}):
        gaussian = _gaussian_potential(Z, slice_thickness, "cpu", sampling=0.05)
        g = _slice_integrals(gaussian)
        atoms = Atoms(
            [Z],
            positions=[
                (_FINITE_CELL[0] / 2 + 0.013, _FINITE_CELL[1] / 2 - 0.007, _FINITE_Z0)
            ],
            cell=_FINITE_CELL,
        )
        quadrature = Potential(
            atoms,
            sampling=0.05,
            slice_thickness=slice_thickness,
            integrator=QuadratureProjectionIntegrals(cutoff_tolerance=_TIGHT_CUTOFF),
        )
        q = _slice_integrals(quadrature)
        VL = lobato.potential(chemical_symbols[Z])
        VP = peng.potential(chemical_symbols[Z])
        bound = _periodic_slab_integrals(
            lambda r: np.abs(VL(r) - VP(r)),
            quadrature.slice_limits,
            _FINITE_Z0,
            _FINITE_CELL[2],
        )
    atom_slice = int(_FINITE_Z0 // slice_thickness)
    bound[atom_slice] += abs(_f0(lobato, Z) - _f0(peng, Z))
    quadrature_budget = 2e-3 * np.abs(q)
    quadrature_budget[atom_slice] = 2e-2 * abs(q[atom_slice])
    excess = np.abs(g - q) - (bound + quadrature_budget)
    assert np.all(excess < 0), (np.abs(g - q), bound, quadrature_budget)


# --------------------------------------------------------------------------
# Task 4: ChargeDensityPotential against analytic electrostatics
# --------------------------------------------------------------------------

_CD_CELL = 8.0
_CD_GRID = 64  # charge-density grid points per axis (0.125 A)


def _charge_density_potential(Z, z0, density, slice_thickness, device="cpu"):
    from abtem.potentials.charge_density import ChargeDensityPotential

    atoms = Atoms(
        [Z], positions=[(_CD_CELL / 2, _CD_CELL / 2, z0)], cell=[_CD_CELL] * 3, pbc=True
    )
    return ChargeDensityPotential(
        atoms, density, sampling=0.05, slice_thickness=slice_thickness, device=device
    )


def _gaussian_electrons(Z, z0, sigma):
    """-Z electrons in a normalised 3D Gaussian of std `sigma` on the atom
    (minimum-image in all directions, so the density is periodic)."""
    x = np.arange(_CD_GRID) * _CD_CELL / _CD_GRID
    d = [
        (x - c + _CD_CELL / 2) % _CD_CELL - _CD_CELL / 2
        for c in (_CD_CELL / 2,) * 2 + (z0,)
    ]
    r2 = d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2
    return Z * np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2) ** 1.5


@devices
def test_charge_density_point_charges_give_the_ewald_projected_potential(device):
    """Zero electron density plus a nucleus Z: the potential is that of a
    periodic lattice of point charges in a neutralising background.

    Oracle: summed over all slices this is the kz = 0 problem -- a 2D lattice
    of line charges carrying Z per cell -- solved by an independent 2D Ewald
    split with width s = 0.7 A:

      V_p(rho) = Z/(4 pi eps0) sum_images E1(|rho - rho_n|^2 / (2 s^2))
                 + Z/(eps0 A) sum_{G != 0} exp(-2 pi^2 s^2 G^2)/(4 pi^2 G^2)
                   cos(2 pi G.(rho - rho_0))

    where the first term is the projected potential of a point charge minus a
    Gaussian charge (2D Gaussian line charge: -(Z/2 pi eps0)[ln rho +
    E1(rho^2/2s^2)/2]), and the second the Gaussian remainder from 2D Poisson,
    grad^2 V_p = -lambda/eps0. The Ewald split parameters differ from the
    code's (3D erf split, width 3 A, quadrature + 3D FFT), so nothing is shared.

    ChargeDensityPotential subtracts each slice's minimum, so the comparison
    is up to one additive constant (sum of the per-slice minima). Tolerance:
    1e-3 of the oracle's range over 0.2 < rho < 3 A -- above the order-2
    spline interpolation of the smooth 3 A-wide long-range part from the
    0.125 A density grid, ~(0.125/3)^3 ~ 7e-5, and excluding the log-singular
    core pixels.
    """
    if device == "mps":
        pytest.skip("Metal is single precision; this test runs in float64")
    from scipy.special import exp1

    from abtem.core.constants import eps0

    Z = 6
    density = np.zeros((_CD_GRID,) * 3)
    with config.set({"precision": "float64"}):
        potential = _charge_density_potential(Z, _CD_CELL / 2, density, 1.0, device)
        projected = asnumpy(potential.build(lazy=False).array).sum(0)

    gpts, sampling = potential.gpts, potential.sampling
    x = np.arange(gpts[0]) * sampling[0]
    y = np.arange(gpts[1]) * sampling[1]
    X, Y = np.meshgrid(x, y, indexing="ij")
    centre = (_CD_CELL / 2, _CD_CELL / 2)
    s = 0.7
    C = Z / (4 * np.pi * eps0)

    oracle = np.zeros_like(X)
    for i in range(-3, 4):
        for j in range(-3, 4):
            r2 = (X - centre[0] - i * _CD_CELL) ** 2 + (
                Y - centre[1] - j * _CD_CELL
            ) ** 2
            oracle += C * exp1(np.maximum(r2, 1e-12) / (2 * s**2))
    m = np.fft.fftfreq(gpts[0], 1 / gpts[0])
    Gx, Gy = m[:, None] / _CD_CELL, m[None, :] / _CD_CELL
    G2 = Gx**2 + Gy**2
    G2[0, 0] = 1.0
    coefficients = (
        Z
        / (eps0 * _CD_CELL**2)
        * np.exp(-2 * np.pi**2 * s**2 * G2)
        / (4 * np.pi**2 * G2)
    )
    coefficients[0, 0] = 0.0
    phase = np.exp(-2j * np.pi * (Gx * centre[0] + Gy * centre[1]))
    oracle += np.real(np.fft.ifft2(coefficients * phase)) * gpts[0] * gpts[1]

    R = np.hypot(X - centre[0], Y - centre[1])
    band = (R > 0.2) & (R < 3.0)
    difference = projected[band] - oracle[band]
    difference -= difference.mean()
    assert np.abs(difference).max() < 1e-3 * np.ptp(oracle[band]), (
        np.abs(difference).max(),
        np.ptp(oracle[band]),
    )


@devices
@pytest.mark.parametrize("z0", (3.3, 4.0))
def test_charge_density_neutral_atom_matches_screened_coulomb_per_slice(z0, device):
    """Nucleus Z plus -Z electrons in a Gaussian of std 0.7 A on it: a neutral
    atom, whose potential is V(r) = Z/(4 pi eps0) erfc(r / (sqrt(2) s)) / r --
    positive and decaying to zero far from the atom.

    Oracle: that V integrated over each slice along z with scipy.integrate.quad,
    at in-plane distances 0.2-3 A (plus the +-Lz images). Because V >= 0 and
    vanishes far away, the code's per-slice minimum subtraction removes ~0, so
    the comparison is absolute.

    This is a full-electron density, so issue #421 (full Z added against a
    valence-only density) does not arise here; the test does not encode it.

    Tolerance: 2e-3 of the peak. The density's kz content is cropped to the
    slice count; for 0.5 A slices the Gaussian's spectrum at the crop,
    exp(-2 pi^2 s^2 (1 A^-1)^2), is 6e-5. The remaining errors (order-2 spline
    interpolation, the 1e-4 eV Ewald cutoff) are of the same order.
    Two z0: mid-slice and on a slice boundary, where a z-offset shows as an
    asymmetry between the slices above and below.
    """
    if device == "mps":
        pytest.skip("Metal is single precision; this test runs in float64")
    from scipy.special import erfc

    from abtem.core.constants import eps0

    Z, sigma, dz = 6, 0.7, 0.5
    with config.set({"precision": "float64"}):
        potential = _charge_density_potential(
            Z, z0, _gaussian_electrons(Z, z0, sigma), dz, device
        )
        array = asnumpy(potential.build(lazy=False).array)

    C = Z / (4 * np.pi * eps0)

    def V(r):
        return C * erfc(r / (np.sqrt(2) * sigma)) / r

    s = potential.sampling
    i0 = int(round(_CD_CELL / 2 / s[0]))
    offsets = np.array([4, 10, 20, 40, 60])  # 0.2, 0.5, 1, 2, 3 A
    numeric = array[:, i0 + offsets, i0]
    oracle = np.zeros_like(numeric)
    for j, (a, b) in enumerate(potential.slice_limits):
        for k, offset in enumerate(offsets):
            rho = offset * s[0]
            for image in (-1, 0, 1):
                lo, hi = a - z0 - image * _CD_CELL, b - z0 - image * _CD_CELL
                oracle[j, k] += integrate.quad(
                    lambda z: V(np.hypot(rho, z)), lo, hi, limit=200
                )[0]
    tolerance = 2e-3 * oracle.max()
    assert np.abs(numeric - oracle).max() < tolerance, float(
        np.abs(numeric - oracle).max()
    )
    # Far from the atom a neutral atom's potential is ~0 in every slice.
    assert np.abs(numeric[:, -1]).max() < tolerance
