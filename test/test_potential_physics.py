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
        projected, pos, sampling = _infinite_projection(
            parametrization, Z, 600, device
        )
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


def _finite_potential(parametrization, Z, sampling, device, cutoff_tolerance, pos_xy=None):
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
        tail = 4 * np.pi * integrate.quad(
            lambda r: r**2 * float(V(np.array([r]))[0]), 0.85 * cutoff, 60.0
        )[0]
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
