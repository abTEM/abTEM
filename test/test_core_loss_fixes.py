"""Regression tests for correctness fixes in the core-loss machinery.

Each test here corresponds to a defect that was present and is now fixed. They
are grouped by the object they belong to rather than by symptom.
"""

from __future__ import annotations

import functools
import io
import itertools
import re
import sys

import ase
import numpy as np
import pytest

import abtem
from abtem.array import ArrayObject
from abtem.core.axes import EnergyAxis, OrdinalAxis
from abtem.core.utils import ensure_list
from abtem.inelastic.core_loss import (
    AtomicWaveFunction,
    RadialWavefunction,
    TransitionPotential,
    TransitionPotentialArray,
    _asymptotic_amplitude,
    _continuum_radial_grid,
)

try:
    import gpaw  # noqa: F401
except ImportError:
    pass

from utils import devices, synthetic_transition_potential, to_host_array  # noqa: E402

requires_gpaw = pytest.mark.skipif(
    "gpaw" not in sys.modules, reason="requires gpaw"
)

ENERGY = 100e3


def _as_array(measurement):
    return np.asarray(measurement.to_cpu().array)


def _energy_members(measurement):
    """Split an energy-ensemble measurement into its members, taken along
    whichever array axis the measurement's own ``axes_metadata`` labels as
    the ``EnergyAxis`` -- not an axis position assumed from the scan type.
    """
    axes = measurement.axes_metadata
    energy_axes = [i for i, axis in enumerate(axes) if isinstance(axis, EnergyAxis)]
    assert len(energy_axes) == 1, f"expected one EnergyAxis, got {axes}"
    array = _as_array(measurement)
    assert array.ndim == len(axes)
    axis = energy_axes[0]
    return [np.take(array, i, axis=axis) for i in range(array.shape[axis])]


def _si2_atoms():
    return ase.Atoms(
        "Si2", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)], cell=(8, 8, 8), pbc=True,
    )


def test_radial_wavefunction_bound():
    """``bound`` compared n to 0, which raises for a continuum state (n=None);
    ``AtomicWaveFunction`` delegates to it."""

    def wavefunction(n, energy):
        return RadialWavefunction(
            n=n,
            l=1,
            energy=energy,
            radial_grid=np.linspace(1e-9, 1.0, 10),
            radial_values=np.zeros(10),
        )

    assert wavefunction(n=None, energy=25.0).bound is False
    assert wavefunction(n=2, energy=-1839.0).bound is True
    assert AtomicWaveFunction(wavefunction(None, 25.0), ml=0).bound is False


class TestAsymptoticAmplitude:
    """The continuum amplitude was read as max(u), which is wrong twice over."""

    def test_recovers_the_amplitude_of_a_pure_sinusoid(self):
        k = 1.3
        r = np.linspace(1e-12, 40.0, 200000)
        for amplitude in [0.5, 1.0, 7.25]:
            u = amplitude * np.sin(k * r + 0.4)
            assert _asymptotic_amplitude(r, u, k) == pytest.approx(amplitude, rel=1e-4)

    def test_ignores_a_larger_transient_inside(self):
        # A big inner excursion, as produced near a centrifugal turning point,
        # must not be mistaken for the asymptotic amplitude.
        k = 1.3
        r = np.linspace(1e-12, 40.0, 200000)
        crest = (np.pi / 2 + 2 * np.pi) / k
        u = np.sin(k * r) * (1.0 + 5.0 * np.exp(-((r - crest) ** 2)))
        assert _asymptotic_amplitude(r, u, k) == pytest.approx(1.0, rel=1e-3)
        assert u.max() > 3.0  # max(u) would have been badly wrong

    def test_warns_when_the_envelope_is_not_flat(self):
        k = 1.3
        r = np.linspace(1e-12, 40.0, 200000)
        u = np.sin(k * r) * r  # envelope still growing
        with pytest.warns(RuntimeWarning, match="free-particle"):
            _asymptotic_amplitude(r, u, k)

    def test_vanishing_wavefunction_raises(self):
        r = np.linspace(1e-12, 40.0, 1000)
        with pytest.raises(RuntimeError, match="vanishes"):
            _asymptotic_amplitude(r, np.zeros_like(r), 1.3)


class TestContinuumGrid:
    """A fixed 20 Bohr grid cannot resolve the asymptotic region at low energy."""

    def test_grid_grows_at_low_energy(self):
        from ase import units

        low = _continuum_radial_grid(1.0 / units.Rydberg, lprime=0)
        high = _continuum_radial_grid(400.0 / units.Rydberg, lprime=0)
        assert low[-1] > high[-1]

    def test_grid_grows_with_angular_momentum(self):
        from ase import units

        ef = 25.0 / units.Rydberg
        assert (
            _continuum_radial_grid(ef, lprime=3)[-1]
            >= _continuum_radial_grid(ef, lprime=0)[-1]
        )

    def test_high_energy_keeps_the_original_grid(self):
        from ase import units

        grid = _continuum_radial_grid(400.0 / units.Rydberg, lprime=0)
        assert grid[-1] == pytest.approx(20.0)

    def test_grid_is_capped_and_warns(self):
        from ase import units

        with pytest.warns(RuntimeWarning, match="asymptotic form"):
            grid = _continuum_radial_grid(1e-4 / units.Rydberg, lprime=3)
        assert grid[-1] <= 150.0


@requires_gpaw
class TestRadialEquation:
    """``radial_schroedinger_equation`` is the operator the continuum states
    are solved with; it must be the one GPAW's own bound states solve."""

    @staticmethod
    def _bound_eigenvalue(n, l, rv, rmax=25.0, h=2e-4):
        """Shooting with node counting: the eigenvalue [Rydberg] of the state
        with n - l - 1 radial nodes, by bisection on the energy."""
        from abtem.inelastic.core_loss import numerov, radial_schroedinger_equation

        # u(0) = 0 at the origin itself (r = 0 would divide by zero).
        r = np.arange(int(rmax / h) + 1) * h
        r[0] = 1e-12
        inner = int(0.9 * len(r))  # ignore the divergent tail's last node
        lower, upper = -1.2 * 14**2, -1e-4
        for _ in range(80):
            energy = 0.5 * (lower + upper)
            u = numerov(radial_schroedinger_equation(energy, l, r, rv), 0.0, 1e-12, h)
            signs = np.sign(u[:inner])
            signs = signs[signs != 0]
            if np.sum(signs[1:] != signs[:-1]) > n - l - 1:
                upper = energy
            else:
                lower = energy
        return 0.5 * (lower + upper)

    @pytest.mark.parametrize("n, l", [(2, 1), (3, 0), (3, 1)])
    def test_reproduces_gpaw_bound_eigenvalues(self, n, l):
        """In the potential of a non-relativistic GPAW atom (whose kinetic
        operator is the same non-relativistic one) the equation must give
        GPAW's eigenvalues. It did not: both the centrifugal and the
        potential term were multiplied by an unexplained 1.02, which put
        these eigenvalues 4-10% off. Measured without it: <= 1e-5."""
        from gpaw.atom.aeatom import AllElectronAtom
        from scipy.interpolate import interp1d

        ae = AllElectronAtom(
            "Si", xc="PBE", scalar_relativistic=False, log=io.StringIO()
        )
        ae.run()
        ae.refine()
        rv = interp1d(
            ae.rgd.r_g, -2 * ae.vr_sg[0], fill_value=(28.0, 0.0), bounds_error=False
        )
        expected = 2 * ae.channels[l].e_n[n - l - 1]  # Hartree -> Rydberg

        assert self._bound_eigenvalue(n, l, rv) == pytest.approx(expected, rel=1e-4)


@requires_gpaw
class TestAtomicPotentialTail:
    """GPAW evaluates vxc at a floored density (1e-10), so in vacuum it sat
    at a constant -8.5e-4 Hartree: r*V grew linearly to the edge of the grid
    (r*V = 0.083 Ry Bohr at 49 Bohr for every element) and, extrapolated,
    left a constant potential that shifted the continuum's asymptotic
    wavenumber (q/k = 1.012 at 1 eV). Extending GPAW's grid did not help --
    the constant was the same at rcut = 150 Bohr."""

    @pytest.mark.parametrize("xc", ["LDA", "PBE"])
    def test_potential_vanishes_in_vacuum_and_is_unchanged_inside(self, xc):
        from gpaw.atom.aeatom import AllElectronAtom

        from abtem.inelastic.core_loss import _atomic_rv

        rv = _atomic_rv(14, xc=xc)
        for r in (40.0, 49.0, 100.0, 150.0):
            # Was 1.7e-3 Ry; measured <= 2e-7 Ry now.
            assert abs(rv(r) / r) < 1e-6, f"V({r} Bohr) = {rv(r) / r:.2e} Ry"

        ae = AllElectronAtom("Si", xc=xc, log=io.StringIO())
        ae.run()
        ae.scalar_relativistic = True
        ae.refine()
        dense = ae.n_sg.sum(0) >= 1e-10
        np.testing.assert_allclose(
            rv(ae.rgd.r_g[dense]), -2 * ae.vr_sg[0][dense], rtol=1e-12, atol=0
        )


@requires_gpaw
class TestContinuumNormalisation:
    """The continuum state must be energy-normalised: u -> sin(kr+d)/sqrt(pi k)."""

    # Each way _continuum_radial_grid sizes the grid, and every l' (l' >= 2 is
    # where the envelope overshoots near the barrier): at 1 eV the grid holds
    # one period (l' = 0, 2) or clears the centrifugal barrier (l' = 3, the
    # largest grid, 120 Bohr); at 25 eV it is the default 20 Bohr (l' = 1) or
    # clears the barrier (l' = 3); at 400 eV it is the default (l' = 2).
    @pytest.mark.parametrize(
        "epsilon, lprime",
        [(1.0, 0), (1.0, 2), (1.0, 3), (25.0, 1), (25.0, 3), (400.0, 2)],
    )
    def test_asymptotic_amplitude_is_one_over_sqrt_pi_k(self, epsilon, lprime):
        """Two checks on one wavefunction (each costs a GPAW all-electron run
        and a Numerov integration on up to ~7.5e6 points, so it is computed
        once): the envelope amplitude, and an independent WKB fit."""
        from ase import units

        from abtem.inelastic.core_loss import (
            calculate_continuum_radial_wavefunction,
        )

        wavefunction = calculate_continuum_radial_wavefunction(
            Z=14, n=1, l=0, lprime=lprime, epsilon=epsilon
        )
        r = wavefunction.radial_grid
        u = wavefunction._radial_values
        ef = epsilon / units.Rydberg
        k = np.sqrt(ef)

        outer = r > 0.75 * r[-1]
        du = np.gradient(u, r)
        amplitude = float(np.median(np.sqrt(u[outer] ** 2 + (du[outer] / k) ** 2)))

        assert amplitude * np.sqrt(np.pi * k) == pytest.approx(1.0, rel=1e-3)

        self._assert_outer_region_fits_an_energy_normalised_wkb_wave(
            r, u, ef, lprime
        )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _si_rv():
        """r*V(r) of the Si atom [Rydberg * Bohr]: the very potential
        calculate_continuum_radial_wavefunction solves in (the physical
        input to the problem, not the code under test)."""
        from abtem.inelastic.core_loss import _atomic_rv

        return _atomic_rv(14, xc="PBE")

    def _assert_outer_region_fits_an_energy_normalised_wkb_wave(
        self, r, u, ef, lprime
    ):
        """Independent of ``_asymptotic_amplitude``'s envelope estimator:
        least-squares fit the outer region to the WKB form of the solution.

        The potential is the neutral ground-state atom's (no core hole), so
        there is no Coulomb log-phase term, and it vanishes in vacuum (see
        TestAtomicPotentialTail) -- but not by the start of the outer region
        at every energy, so rather than a bare free-wave fit (Riccati-Bessel
        functions at fixed k), fit
        u = A sqrt(k/q) [a sin(phi) + b cos(phi)], phi = int q dr, with
        q(r)^2 = -f(r) from the very radial equation u'' = f u that is
        solved (``radial_schroedinger_equation``, which defines the
        problem). A = hypot(a, b) is then the r -> infinity amplitude.

        Because q(r) comes from that same equation, this checks the
        normalisation of whatever equation is solved; whether the equation
        itself is right is TestRadialEquation's job.

        Energy normalisation, derived: u -> A sin(kr + delta) gives
        integral u_k u_k' dr = A^2 (pi/2) delta(k - k'), and with E = k^2
        (Rydberg units), delta(k - k') = 2k delta(E - E'), so
        <E|E'> = delta(E - E') requires A = 1/sqrt(pi k).

        Measured |A sqrt(pi k) - 1| <= 4e-4 (l'=3, 25 eV, the case the WKB
        form describes worst: 0.2% fit residual); <= 2.4e-4 elsewhere. (With
        the potential's spurious vacuum constant, before TestAtomicPotential
        Tail's fix: 1.1e-3, and a 1% residual.)
        """
        from scipy.integrate import cumulative_trapezoid

        from abtem.inelastic.core_loss import radial_schroedinger_equation

        k = np.sqrt(ef)

        # Outer 40% of the grid, and outside the Si atom.
        outer = (r > 0.6 * r[-1]) & (r > 10.0)
        r_outer = r[outer]
        q = np.sqrt(-radial_schroedinger_equation(ef, lprime, r_outer, self._si_rv()))
        phi = cumulative_trapezoid(q, r_outer, initial=0.0)
        basis = np.sqrt(k / q)[:, None] * np.stack([np.sin(phi), np.cos(phi)], axis=1)
        (a, b), *_ = np.linalg.lstsq(basis, u[outer], rcond=None)
        amplitude = np.hypot(a, b)

        # The WKB form must actually describe the wave out there ...
        residual = np.abs(u[outer] - basis @ np.array([a, b])).max()
        assert residual < 1e-2 * amplitude
        # ... with the energy-normalised amplitude.
        assert amplitude * np.sqrt(np.pi * k) == pytest.approx(1.0, rel=1e-3)


def test_no_hardcoded_dtypes_remain_in_core_loss():
    """The transition potential ignored ``config['precision']``: it was
    allocated complex64 and then promoted back to complex128 by a division by
    a float64 numpy scalar (NEP 50), so every transition potential was silently
    double precision.

    This greps the source for dtype literals, which a promotion does not leave;
    the built dtype and the single-vs-double agreement are checked by
    test_core_loss.py's test_build_dtype_follows_precision and
    test_build_matches_double_precision_reference, which need GPAW. Until those
    run without it, this is the only check of the defect in CI.
    """
    from pathlib import Path

    import abtem.inelastic.core_loss as module

    source = Path(module.__file__).read_text()
    offenders = [
        line
        for line in source.splitlines()
        if re.search(
            r"(dtype\s*=\s*(np|xp)\.(float|complex)\d+|"
            r"(np|xp)\.(float|complex)\d+\s*\()",
            line,
        )
    ]
    assert not offenders, (
        "use get_dtype() so abtem.config['precision'] is honoured:\n"
        + "\n".join(offenders)
    )


@devices
def test_entrance_exit_plane_carries_no_core_loss_signal(device):
    """At t = 0 nothing has been traversed, so the core-loss signal is zero.

    The driver used to detect the incident *elastic* wave there, writing the
    full unscattered intensity into the t = 0 bin.
    """
    atoms = ase.build.bulk("Si", cubic=True) * (1, 1, 3)
    potential = abtem.Potential(
        atoms, gpts=(64, 64), slice_thickness=1.4, exit_planes=3, device=device
    )
    assert potential.exit_planes[0] == -1

    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)

    got = probe.transition_potential_scan(
        potential=potential,
        transition_potentials=synthetic_transition_potential(
            gpts=potential.gpts, extent=potential.extent, n_transitions=3,
            device=device,
        ),
        scan=np.array([[0.0, 0.0]]),
        detectors=abtem.AnnularDetector(inner=0.0, outer=None),
        double_channel=False,
        lazy=False,
        sites=atoms,
    ).compute()

    values = np.asarray(abtem.core.backend.asnumpy(got.array)).ravel()
    assert values[0] == 0.0
    assert np.all(np.diff(values) > 0)


class TestPrismEelsBuiltTransitionPotentialGrid:
    """A built transition potential on the wrong grid ran silently.

    ``_prism_eels_common_setup`` matches the transition potential's grid to the
    S-matrix waves. For an UNBUILT one that is the fix in the preceding commit:
    it has no array yet, so the match sets the grid and ``build()`` then
    evaluates the form factors at the right gpts.

    For an already-BUILT one the array shape is fixed, so the match cannot
    change it -- it overwrites the *grid* to agree with the waves and leaves the
    array alone, producing an object whose grid lies about its own contents
    (gpts (64, 64) over a (4, 32, 32) array). The scan then completed and
    returned a result on the wrong grid: 18.1 % low against the matched
    reference, with an identical output shape, so nothing downstream could
    notice.

    The guard therefore has to run BEFORE the match, while the grid still
    reports what the array really is; checking afterwards is useless because
    the match is what destroys the evidence.
    """

    def _s_matrix(self, gpts):
        potential = abtem.Potential(_si2_atoms(), gpts=gpts, slice_thickness=4.0)
        return abtem.SMatrix(
            potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=1
        )

    def _scan(self, s_matrix, transition_potentials, **kwargs):
        return s_matrix.transition_potential_scan(
            transition_potentials=transition_potentials,
            scan=abtem.GridScan(start=(0, 0), end=(1, 1), gpts=(2, 2), fractional=True,
                                potential=s_matrix.potential),
            detectors=abtem.FlexibleAnnularDetector(),
            sites=_si2_atoms(), lazy=False, **kwargs,
        )

    def test_a_built_potential_on_the_wrong_grid_is_refused(self):
        s_matrix = self._s_matrix((64, 64))
        mismatched = synthetic_transition_potential(
            extent=(8.0, 8.0), gpts=(32, 32), n_transitions=2
        )
        with pytest.raises(RuntimeError, match="Inconsistent grid"):
            self._scan(s_matrix, mismatched)

    def test_a_built_potential_on_the_right_grid_still_runs(self):
        """The guard must not fire on the case it is meant to allow."""
        s_matrix = self._s_matrix((64, 64))
        matched = synthetic_transition_potential(
            extent=(8.0, 8.0), gpts=(64, 64), n_transitions=2
        )
        measurement = self._scan(s_matrix, matched)
        assert measurement.shape[:2] == (2, 2)

    def test_a_mismatched_extent_is_refused_too(self):
        """check_match compares extent as well as gpts, and an extent mismatch
        is the same class of error -- the array cannot be re-gridded either."""
        s_matrix = self._s_matrix((64, 64))
        wrong_extent = synthetic_transition_potential(
            extent=(6.0, 6.0), gpts=(64, 64), n_transitions=2
        )
        with pytest.raises(RuntimeError, match="Inconsistent grid"):
            self._scan(s_matrix, wrong_extent)

    def test_a_mismatched_energy_is_refused_too(self):
        """A grid match is not enough: build() bakes self.energy into the
        array's form factors (k0, kn, the relativistic mass correction, the
        interaction parameter), so a built array on the right grid but the
        wrong energy is exactly as stale as a gpts/extent mismatch. Without
        the accelerator check, _task_local's match silently overwrites
        .energy to agree with the S-matrix while those baked-in form factors
        stay computed at the old one, and the scan completes with a plausible
        but wrong result."""
        s_matrix = self._s_matrix((64, 64))
        wrong_energy = synthetic_transition_potential(
            extent=(8.0, 8.0), gpts=(64, 64), n_transitions=2
        )
        wrong_energy.accelerator._energy = 2 * ENERGY
        with pytest.raises(RuntimeError, match="Inconsistent energies"):
            self._scan(s_matrix, wrong_energy)

    def test_an_unbuilt_potential_is_still_regridded(self):
        """An unbuilt potential has no array, so a grid mismatch is not an
        error: it is matched and then built at the S-matrix's gpts, giving the
        result of one created on the S-matrix's grid."""
        pytest.importorskip("sympy")
        s_matrix = self._s_matrix((64, 64))
        regridded, reference = (
            _as_array(
                self._scan(
                    s_matrix,
                    _synthetic_unbuilt_transition_potential(
                        ENERGY, extent=(8.0, 8.0), gpts=gpts, Z=14
                    ),
                )
            )
            for gpts in ((32, 32), (64, 64))
        )
        assert regridded.shape[:2] == (2, 2)
        assert np.abs(reference).max() > 0
        np.testing.assert_array_equal(regridded, reference)


class TestMultisliceBuiltTransitionPotentialEnergy:
    """The sibling of TestPrismEelsBuiltTransitionPotentialGrid's energy
    check above, for the plain multislice EELS driver
    (transition_potential_multislice_and_detect, abtem/multislice.py) rather
    than the PRISM-EELS one. Same guard, same reason, same place: before
    _task_local's match, which would otherwise silently overwrite a built
    transition potential's energy to agree with the waves while its baked-in
    form factors stay computed at the old one.
    """

    def test_a_mismatched_energy_is_refused_too(self):
        atoms = _si2_atoms()
        potential = abtem.Potential(atoms, gpts=(64, 64), slice_thickness=4.0)
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
        probe.grid.match(potential)

        wrong_energy = synthetic_transition_potential(
            extent=(8.0, 8.0), gpts=(64, 64), n_transitions=2
        )
        wrong_energy.accelerator._energy = 2 * ENERGY

        with pytest.raises(RuntimeError, match="Inconsistent energies"):
            probe.transition_potential_scan(
                potential=potential, transition_potentials=wrong_energy,
                scan=abtem.GridScan(start=(0, 0), end=(1, 1), gpts=(2, 2),
                                    fractional=True, potential=potential),
                detectors=abtem.FlexibleAnnularDetector(), sites=atoms,
                double_channel=False, lazy=False,
            )


def _synthetic_unbuilt_transition_potential(
    energy, extent=(8.0, 8.0), gpts=(64, 64), Z=5
):
    """An unbuilt ``TransitionPotential`` with a synthetic (non-physical)
    1s -> continuum-p transition. Building it exercises the real
    `_calculate_form_factor` path (needs sympy for the Wigner-3j symbols),
    unlike `synthetic_transition_potential`'s already-built
    `TransitionPotentialArray`, which never calls `build()` again -- exactly
    the case where a genuinely per-member energy can (and, after the fix,
    does) drive a fresh build.
    """
    r = np.linspace(1e-3, 20.0, 4000)
    bound = RadialWavefunction(
        n=1, l=0, energy=-188.0, radial_grid=r, radial_values=r * np.exp(-r)
    )
    excited = RadialWavefunction(
        n=None, l=1, energy=15.0, radial_grid=r, radial_values=r * np.exp(-0.1 * r)
    )
    transitions = [
        (AtomicWaveFunction(bound, ml=0), AtomicWaveFunction(excited, ml=-1)),
        (AtomicWaveFunction(bound, ml=0), AtomicWaveFunction(excited, ml=1)),
    ]
    return TransitionPotential(
        Z=Z, transitions=transitions, extent=extent, gpts=gpts, energy=energy,
        double_channel=False,
    )


def _bn_atoms():
    return ase.Atoms(
        "BN", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 1.0)], cell=(8, 8, 4), pbc=True,
    )


def _scan_of_kind(scan_kind, potential):
    if scan_kind == "custom":
        return abtem.CustomScan([[0.0, 0.0], [1.0, 1.0]])
    if scan_kind == "grid":
        return abtem.GridScan(
            start=(0, 0), end=(1, 1), gpts=(2, 2), fractional=True, potential=potential
        )
    if scan_kind == "line":
        return abtem.LineScan(
            start=(0, 0), end=(1, 1), gpts=3, fractional=True, potential=potential
        )
    raise ValueError(scan_kind)


def _energy_ensemble_scan(
    driver,
    energy,
    lazy,
    *,
    scan_kind="grid",
    transition_potential="unbuilt",
    interpolation=1,
    **kwargs,
):
    """A core-loss scan of the B site of a BN cell by the probe ("multislice")
    or the S-matrix ("prism") driver, at one energy or an energy ensemble.

    ``transition_potential`` is "unbuilt" (built per energy by the driver) or
    "built" (a ``TransitionPotentialArray`` baked at the first energy). Without
    a ``detectors`` keyword the driver's default detector is used.
    """
    atoms = _bn_atoms()
    potential = abtem.Potential(atoms, gpts=(64, 64), slice_thickness=2.0)
    first = energy[0] if isinstance(energy, (list, tuple)) else energy
    if transition_potential == "unbuilt":
        pytest.importorskip("sympy")
        transition_potential = _synthetic_unbuilt_transition_potential(
            first, extent=potential.extent, gpts=potential.gpts
        )
    elif transition_potential == "built":
        transition_potential = synthetic_transition_potential(
            Z=5, gpts=potential.gpts, extent=potential.extent, energy=first
        )
    scan = _scan_of_kind(scan_kind, potential)
    sites = atoms[atoms.numbers == 5]

    if driver == "multislice":
        probe = abtem.Probe(
            semiangle_cutoff=20, energy=energy, extent=potential.extent,
            gpts=potential.gpts,
        )
        measurement = probe.transition_potential_scan(
            scan=scan, potential=potential, transition_potentials=transition_potential,
            double_channel=False, sites=sites, threshold=1.0, lazy=lazy, **kwargs,
        )
    elif driver == "prism":
        s_matrix = abtem.SMatrix(
            potential=potential, energy=energy, semiangle_cutoff=20,
            interpolation=interpolation,
        )
        measurement = s_matrix.transition_potential_scan(
            transition_potential, scan=scan, sites=sites, lazy=lazy, **kwargs,
        )
    else:
        raise ValueError(driver)

    if lazy:
        measurement = measurement.compute(progress_bar=False)
    return measurement


class TestEnergyEnsembleCoreLoss:
    """Core-loss scans of an energy ensemble (a probe or an S-matrix with a list
    of energies), by both drivers.

    Each member must reproduce a separate single-energy run of its own energy.
    The defects behind this were each a silently wrong member: the eager path
    used the first member's energy for the whole ensemble, the transition
    potential's energy overwrote the members' own, the energy stack landed on a
    different array axis than its metadata, the lazy chunk bookkeeping assumed
    the detector keeps the ensemble axes in order, and the S-matrix driver ran
    the first energy only. One energy order suffices: the energy is resolved
    per member, and the refusals below do not depend on order (see
    ``FlexibleAnnularDetector._outer_for``). The descending (200, 100) keV order
    is the stricter one for the reproduction test, as sorting the energies or
    taking any one member's energy for all fails it.
    """

    @pytest.mark.parametrize(
        "driver, scan_kind, energies, lazy, options",
        [
            # CustomScan keeps the ensemble axes in order, isolating the
            # per-member energy. max_batch=1 spreads the waves ensemble over
            # several dask blocks, which raised "too many values to unpack"
            # lazily while partitioning a potential with no ensemble of its own.
            ("multislice", "custom", (200e3, 100e3), False, {"max_batch": 1}),
            ("multislice", "custom", (200e3, 100e3), True, {"max_batch": 1}),
            # The detector moves the scan axes behind the energy axis. Three
            # energies on a 2x2 grid break the coincidence that hid the lazy
            # chunk reordering for two (both orders give shape (2, 2, 2)).
            ("multislice", "grid", (200e3, 100e3, 150e3), False, {}),
            ("multislice", "grid", (200e3, 100e3, 150e3), True, {}),
            # LineScan: the only scan with exactly one scan axis.
            ("multislice", "line", (200e3, 100e3), False, {}),
            ("multislice", "line", (200e3, 100e3), True, {}),
            # PRISM splits the S-matrix into its energies; interpolation 2
            # crops a window around each position.
            ("prism", "grid", (200e3, 100e3), False, {"interpolation": 1}),
            ("prism", "grid", (200e3, 100e3), True, {"interpolation": 2}),
        ],
        ids=[
            "multislice-custom-eager", "multislice-custom-lazy",
            "multislice-grid3-eager", "multislice-grid3-lazy",
            "multislice-line-eager", "multislice-line-lazy",
            "prism-interp1-eager", "prism-interp2-lazy",
        ],
    )
    def test_each_member_reproduces_its_own_single_energy_run(
        self, driver, scan_kind, energies, lazy, options
    ):
        def run(energy, lazy):
            detectors = [abtem.AnnularDetector(inner=0.0, outer=30.0)]
            if driver == "prism":
                # both measurements must be split and stacked
                detectors.append(abtem.PixelatedDetector(max_angle="cutoff"))
            return ensure_list(
                _energy_ensemble_scan(
                    driver, energy, lazy, scan_kind=scan_kind, detectors=detectors,
                    **options,
                )
            )

        references = {e: run(e, lazy=False) for e in energies}
        ensemble = run(list(energies), lazy)
        assert len(ensemble) == len(references[energies[0]])

        for j, measurement in enumerate(ensemble):
            reference = {e: _as_array(references[e][j]) for e in energies}
            # Core-loss intensities are far below the default tolerances, so
            # the atol is tied to the signal's own scale.
            scale = max(np.abs(r).max() for r in reference.values())

            (energy_axis,) = [
                a for a in measurement.axes_metadata if isinstance(a, EnergyAxis)
            ]
            assert tuple(energy_axis.values) == energies

            # The members are taken along the axis the metadata labels the
            # EnergyAxis, so data stacked on another axis fails here.
            members = _energy_members(measurement)
            assert len(members) == len(energies)
            for member, e in zip(members, energies):
                np.testing.assert_allclose(
                    member, reference[e], rtol=1e-5, atol=scale * 1e-6,
                    err_msg=f"detector {j}, member at {e:.0f} eV",
                )
            # The energies must give different results, or every member equal
            # to one energy's run would pass.
            for a, b in itertools.combinations(energies, 2):
                assert not np.allclose(
                    reference[a], reference[b], rtol=1e-3, atol=scale * 1e-6
                )

    @pytest.mark.parametrize(
        "driver, energies, transition_potential, detectors, match",
        [
            # An auto outer angle cannot fit the cutoff of every energy; the
            # eager path used to size it from the first member, lazy from the
            # highest energy.
            (
                "multislice", [200e3, 100e3], "unbuilt",
                lambda: abtem.FlexibleAnnularDetector(),
                "cannot auto-size its outer angle",
            ),
            # The S-matrix default detector is that FlexibleAnnularDetector.
            (
                "prism", [100e3, 200e3], "built", None,
                "cannot auto-size its outer angle",
            ),
            # A built array has its form factors baked in at one energy, here the
            # first member's, the only one the old code checked against. An
            # explicit outer gets past the detector's own refusal.
            (
                "multislice", [100e3, 200e3], "built",
                lambda: abtem.FlexibleAnnularDetector(outer=60.0),
                "Inconsistent energies",
            ),
            (
                "prism", [100e3, 200e3], "built",
                lambda: abtem.AnnularDetector(inner=0.0, outer=30.0),
                "Inconsistent energies",
            ),
        ],
        ids=[
            "multislice-auto-outer", "prism-default-detector",
            "multislice-built-array", "prism-built-array",
        ],
    )
    def test_an_energy_ensemble_is_refused(
        self, driver, energies, transition_potential, detectors, match
    ):
        for lazy in (False, True):
            kwargs = {} if detectors is None else {"detectors": detectors()}
            try:
                _energy_ensemble_scan(
                    driver, energies, lazy,
                    transition_potential=transition_potential, **kwargs,
                )
            except RuntimeError as error:
                assert re.search(match, str(error)), f"lazy={lazy}: {error}"
            else:
                pytest.fail(f"lazy={lazy}: the energy ensemble was not refused")

    def test_a_single_energy_list_is_a_single_energy(self):
        """A length-1 energy list is not an ensemble: with an auto-sized
        detector it runs like the scalar energy, eagerly and lazily alike."""
        results = {
            (form, lazy): _as_array(
                _energy_ensemble_scan(
                    "multislice", energy, lazy, transition_potential="built",
                    detectors=abtem.FlexibleAnnularDetector(),
                )
            )
            for form, energy in (("scalar", 100e3), ("list", [100e3]))
            for lazy in (False, True)
        }
        reference = results["scalar", False]
        for case, result in results.items():
            assert np.array_equal(result, reference), case

    def test_an_explicit_outer_lets_an_ensemble_through(self):
        """Pinning ``outer`` is the documented escape from the auto-sizing
        refusal; the detector keeps its outer angle when it is reused."""
        detector = abtem.FlexibleAnnularDetector(outer=60.0)
        ensemble = _energy_ensemble_scan(
            "multislice", [200e3, 100e3], False, detectors=detector
        )
        single = _energy_ensemble_scan("multislice", 150e3, False, detectors=detector)
        assert detector.outer == 60.0
        for member in _energy_members(ensemble):
            assert member.shape == _as_array(single).shape

    def test_the_prism_default_detector_is_a_flexible_annular_detector(self):
        """The documented default; ``validate_detectors`` used to turn
        ``detectors=None`` into a ``WavesDetector``."""
        for lazy in (False, True):
            default = _energy_ensemble_scan(
                "prism", ENERGY, lazy, transition_potential="built"
            )
            explicit = _energy_ensemble_scan(
                "prism", ENERGY, lazy, transition_potential="built",
                detectors=abtem.FlexibleAnnularDetector(),
            )
            assert type(default) is type(explicit), f"lazy={lazy}"
            assert default.axes_metadata == explicit.axes_metadata, f"lazy={lazy}"
            np.testing.assert_array_equal(
                _as_array(default), _as_array(explicit), err_msg=f"lazy={lazy}"
            )


class TestPrismScanAxisSqueeze:
    """The scan position axis was squeezed per dask block, not once at the end.

    ``validate_scan`` turns any non-``BaseScan`` scan -- ``(x, y)``,
    ``[(x, y)]``, ``np.array([[x, y]])`` -- into a ``CustomScan`` whose position
    axis is tagged ``_squeeze=True``. The only reader of that flag is the
    module-level ``reduce_ensemble`` in ``waves.py``.

    ``Waves.transition_potential_multislice`` calls it **once, on the assembled
    result**, after the graph is built, so blocks and declared chunks both keep
    the axis and only the finished object loses it. The PRISM driver
    ``prism_transition_potential_scan`` called it **inside every block**, below
    the level that declares ``chunks`` and below the level that accumulates the
    potential ensemble -- so two consumers were handed a shape the blocks did
    not produce:

    * the lazy branch declares ``chunks += scan.shape``;
    * the eager branch pre-allocates from ``dummy_probes(scan)``.

    The result was an output whose *rank* depended on whether the potential was
    a ``FrozenPhonons`` ensemble and on whether the call was lazy -- silently,
    with correct values, since a wrong axis count is only fatal where array rank
    is paired with metadata.

    The oracle throughout is the multislice path, which gets every case right.
    """

    def _potential(self, ensemble):
        atoms = _si2_atoms()
        if ensemble:
            atoms = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.0, seed=1)
        return abtem.Potential(atoms, gpts=(32, 32), slice_thickness=4.0)

    def _prism(self, scan, ensemble, lazy):
        potential = self._potential(ensemble)
        s_matrix = abtem.SMatrix(
            potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=1
        )
        return s_matrix.transition_potential_scan(
            transition_potentials=synthetic_transition_potential(
                gpts=potential.gpts, extent=potential.extent, n_transitions=2
            ),
            scan=scan, detectors=abtem.FlexibleAnnularDetector(),
            sites=_si2_atoms(), lazy=lazy,
        )

    def _multislice(self, scan, ensemble, lazy):
        potential = self._potential(ensemble)
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
        probe.grid.match(potential)
        return probe.transition_potential_scan(
            scan=scan, potential=potential,
            transition_potentials=synthetic_transition_potential(
                gpts=potential.gpts, extent=potential.extent, n_transitions=2
            ),
            detectors=abtem.FlexibleAnnularDetector(),
            sites=_si2_atoms(), lazy=lazy,
        )

    @staticmethod
    def _axes(measurement):
        """The ensemble/scan axes and their count -- not the detector bins.

        PRISM and multislice legitimately disagree on the detector base shape
        (32 vs 50 radial bins here), because an S-matrix has a different
        reciprocal sampling than a probe on the full grid. That difference is
        algorithmic and pre-existing; what this defect moved is the *number*
        of axes and which ones they are, so that is what to compare.
        """
        return (
            len(measurement.shape),
            len(measurement.axes_metadata),
            tuple(type(a).__name__ for a in measurement.axes_metadata),
        )

    @pytest.mark.parametrize("ensemble", [False, True])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_a_single_position_scan_matches_the_multislice_oracle(
        self, ensemble, lazy
    ):
        """Both potentials, both modes; the rank depended on each.

        ``ArrayObject.squeeze`` only drops length-1 axes, so the defect needed
        exactly one position to show. The tuple, list and array spellings of a
        position all become the same squeezed ``CustomScan`` in
        ``validate_scan``, so the tuple stands for all three.
        """
        scan = (1.0, 1.0)
        got = self._prism(scan, ensemble, lazy)
        assert self._axes(got) == self._axes(self._multislice(scan, ensemble, lazy))

        # The same call must not change rank with how it is scheduled or with
        # the potential being a FrozenPhonons ensemble (the silent half of the
        # defect: with one exit plane it did not crash).
        plain_eager = self._prism(scan, False, lazy=False)
        assert self._axes(got) == self._axes(plain_eager)
        assert got.shape == plain_eager.shape

    def test_a_grid_scan_keeps_its_position_axes(self):
        """The squeeze must fire only for ``_squeeze`` axes. A ``BaseScan``
        carries no such flag and must be untouched -- this is the regression
        guard for the fix itself."""
        scan = abtem.GridScan(
            start=(0, 0), end=(1, 1), gpts=(2, 2), fractional=True,
            potential=self._potential(False),
        )
        measurement = self._prism(scan, False, lazy=False)
        assert measurement.shape[:2] == (2, 2)
        assert len(measurement.axes_metadata) == len(measurement.shape)

    def test_the_single_position_value_equals_that_point_of_a_grid_scan(self):
        """Shapes agreeing is not enough: the surviving axis must carry the
        same numbers it did before."""
        potential = self._potential(False)
        grid = abtem.GridScan(
            start=(1.0, 1.0), end=(1.0, 1.0), gpts=(1, 1), endpoint=False,
            potential=potential,
        )
        from_grid = np.squeeze(np.asarray(self._prism(grid, False, lazy=False).array))
        from_point = np.squeeze(np.asarray(self._prism((1.0, 1.0), False, lazy=False).array))
        assert from_grid.shape == from_point.shape
        assert np.allclose(from_point, from_grid, rtol=1e-6, atol=0.0)


class TestMultipleDetectorsInOnePass:
    """Passing several detectors raised AssertionError instead of working.

    The elastic multislice and the scattered waves are shared, so filling two
    detectors in one pass is both possible and much cheaper than two runs.
    """

    @staticmethod
    def _setup():
        atoms = ase.build.bulk("Si", cubic=True) * (1, 1, 2)
        potential = abtem.Potential(atoms, gpts=(32, 32), slice_thickness=1.4)
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
        probe.grid.match(potential)
        return atoms, potential, probe

    def _run(self, detectors, transition_potentials=None):
        atoms, potential, probe = self._setup()
        if transition_potentials is None:
            transition_potentials = synthetic_transition_potential(
                gpts=potential.gpts, extent=potential.extent, n_transitions=3
            )
        return probe.transition_potential_scan(
            potential=potential,
            transition_potentials=transition_potentials,
            scan=abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2)),
            detectors=detectors,
            double_channel=False,
            lazy=False,
            sites=atoms,
        )

    def test_two_detectors_return_a_list(self):
        got = self._run(
            [
                abtem.AnnularDetector(0.0, 30.0),
                abtem.AnnularDetector(30.0, 60.0),
            ]
        )
        assert isinstance(got, list) and len(got) == 2

    def test_each_matches_its_own_single_detector_run(self):
        inner = abtem.AnnularDetector(0.0, 30.0)
        outer = abtem.AnnularDetector(30.0, 60.0)

        both = self._run([inner, outer])
        np.testing.assert_allclose(
            np.asarray(both[0].compute().array),
            np.asarray(self._run(inner).compute().array),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(both[1].compute().array),
            np.asarray(self._run(outer).compute().array),
            rtol=1e-6,
        )

    def test_a_single_detector_still_returns_one_measurement(self):
        got = self._run(abtem.AnnularDetector(0.0, 30.0))
        assert not isinstance(got, list)

    def test_multiple_detectors_with_multiple_edges(self):
        atoms, potential, _ = self._setup()
        potentials = [
            synthetic_transition_potential(
                gpts=potential.gpts, extent=potential.extent, n_transitions=3,
                seed=seed,
            )
            for seed in (0, 1)
        ]
        got = self._run(
            [abtem.AnnularDetector(0.0, 30.0), abtem.AnnularDetector(30.0, 60.0)],
            transition_potentials=potentials,
        )
        assert isinstance(got, list) and len(got) == 2
        for measurement in got:
            assert measurement.compute().shape[0] == 2


class TestFilterByIntensity:
    """It sorted by intensity, then sliced the *unsorted* list."""

    def test_keeps_the_strongest_transitions(self):
        # Synthetic 1s -> p transitions (see _synthetic_unbuilt_transition_potential)
        # whose continuum states differ only in amplitude, so the intensities are
        # known without building: proportional to 0.1**2, 1 and 0.5**2, that is
        # 0.8 %, 79 % and 20 % of the total. The weakest comes first, so the
        # unsorted slice keeps it.
        pytest.importorskip("sympy")
        r = np.linspace(1e-3, 20.0, 4000)
        bound = AtomicWaveFunction(
            RadialWavefunction(
                n=1, l=0, energy=-188.0, radial_grid=r, radial_values=r * np.exp(-r)
            ),
            ml=0,
        )
        weak, strong, medium = (
            (
                bound,
                AtomicWaveFunction(
                    RadialWavefunction(
                        n=None, l=1, energy=15.0, radial_grid=r,
                        radial_values=scale * r * np.exp(-0.1 * r),
                    ),
                    ml=0,
                ),
            )
            for scale in (0.1, 1.0, 0.5)
        )
        potential = TransitionPotential(
            Z=5, transitions=[weak, strong, medium], extent=(8.0, 8.0),
            gpts=(32, 32), energy=ENERGY, double_channel=False,
        )

        for threshold, expected in ((0.5, [strong]), (0.9, [strong, medium])):
            filtered = potential.filter_by_intensity(threshold)
            assert [id(t) for t in filtered.transitions] == [
                id(t) for t in expected
            ], threshold


def test_a_single_transition_potential_equals_a_one_element_list():
    """Waves.transition_potential_multislice wraps a non-list/tuple transition
    potential into a list; the two spellings must give the same measurement."""
    atoms, potential, probe = _expansion_scope_setup()
    single = synthetic_transition_potential(
        gpts=potential.gpts, extent=potential.extent, n_transitions=3
    )

    from_single, from_list = (
        probe.transition_potential_scan(
            potential=potential, transition_potentials=transition_potentials,
            scan=np.array([[0.0, 0.0]]), detectors=abtem.FlexibleAnnularDetector(),
            double_channel=False, lazy=False, sites=atoms,
        )
        for transition_potentials in (single, [single])
    )
    assert np.array_equal(
        np.asarray(abtem.core.backend.asnumpy(from_single.array)),
        np.asarray(abtem.core.backend.asnumpy(from_list.array)),
    )


class TestPotentialEnsembleAccumulation:
    """The single-channel driver indexed only the exit-plane axis.

    The measurement is allocated with the potential's ensemble axes before the
    exit-plane axis, so indexing the plane axis alone addressed the ensemble
    axis instead. With one exit plane the index was empty and every
    configuration's contribution was broadcast across all configurations, so
    the result came out ``num_configs`` times too large; with several exit
    planes the plane slice landed on the configuration axis. On a length-1
    configuration axis -- what the lazy path produces for every configuration
    count, and what eager produces for num_configs == 1 -- it selected nothing
    and the series came back zero; for eager with num_configs > 1 it selected
    real configuration slots instead, giving a flat series carrying signal at
    zero thickness. Wrong either way, zeros only in the first case.

    The double-channel branch goes through ``_update_loss_measurements`` and
    was always correct -- it is used here as a reference.
    """

    @staticmethod
    def _setup(num_configs=None, exit_planes=None, seed=7):
        atoms = ase.Atoms(
            "Si2",
            positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 1.0)],
            cell=(8, 8, 4),
            pbc=True,
        )
        if num_configs is None:
            ensemble = atoms
        else:
            ensemble = abtem.FrozenPhonons(
                atoms, num_configs=num_configs, sigmas=0.05, seed=seed
            )
        potential = abtem.Potential(
            ensemble, gpts=(64, 64), slice_thickness=2.0, exit_planes=exit_planes
        )
        probe = abtem.Probe(semiangle_cutoff=32, energy=ENERGY, extent=(8.0, 8.0))
        probe.grid.match(potential)
        return atoms, potential, probe

    def _run(self, potential, probe, sites, lazy, double_channel=False):
        scan = abtem.GridScan(
            start=(0, 0), end=(1, 1), gpts=(2, 2), fractional=True, potential=potential
        )
        measurement = probe.transition_potential_scan(
            scan=scan,
            potential=potential,
            detectors=abtem.FlexibleAnnularDetector(),
            transition_potentials=synthetic_transition_potential(
                gpts=potential.gpts, extent=potential.extent, n_transitions=2
            ),
            double_channel=double_channel,
            sites=sites,
            threshold=1.0,
            lazy=lazy,
        )
        if lazy:
            measurement = measurement.compute(progress_bar=False)
        return np.asarray(abtem.core.backend.asnumpy(measurement.array))

    @pytest.mark.parametrize("num_configs", [2, 3])
    @pytest.mark.parametrize("double_channel", [False, True])
    def test_eager_matches_lazy_over_frozen_phonons(self, num_configs, double_channel):
        """Eager summed over configurations where lazy averaged."""
        atoms, potential, probe = self._setup(num_configs=num_configs)
        sites = atoms
        eager = self._run(potential, probe, sites, lazy=False,
                          double_channel=double_channel)
        lazy = self._run(potential, probe, sites, lazy=True,
                         double_channel=double_channel)
        # Core-loss intensities are ~1e-9, far below np.allclose's default
        # atol of 1e-8, which would call a factor-of-num_configs error
        # "close". Compare against the data's own scale instead.
        scale = max(np.abs(eager).max(), np.abs(lazy).max())
        np.testing.assert_allclose(eager, lazy, rtol=1e-6, atol=1e-9 * scale)

    @pytest.mark.parametrize("num_configs", [1, 2])
    def test_configuration_count_does_not_scale_the_result(self, num_configs):
        """The ensemble reduction averages, so the total must not follow n."""
        atoms, one, probe = self._setup(num_configs=1)
        sites = atoms
        reference = self._run(one, probe, sites, lazy=True)

        _, potential, probe = self._setup(num_configs=num_configs)
        got = self._run(potential, probe, sites, lazy=False)
        assert got.sum() == pytest.approx(reference.sum(), rel=1e-2)

    @pytest.mark.parametrize("lazy", [False, True])
    def test_thickness_series_survives_a_potential_ensemble(self, lazy):
        """A frozen-phonon thickness series came back identically zero."""
        atoms, plain, probe = self._setup(exit_planes=1)
        sites = atoms
        reference = self._run(plain, probe, sites, lazy=lazy)

        _, potential, probe = self._setup(num_configs=1, exit_planes=1)
        got = self._run(potential, probe, sites, lazy=lazy)

        assert got.shape == reference.shape
        # The entrance plane at t = 0 is zero by construction; every later
        # plane must carry signal, and must match the un-wrapped potential to
        # within the frozen-phonon displacement.
        assert got[0].sum() == 0.0
        assert np.all([got[i].sum() > 0.0 for i in range(1, got.shape[0])])
        assert got.sum() == pytest.approx(reference.sum(), rel=1e-2)

    def test_each_configuration_slot_holds_its_own_configuration(self):
        """Every slot held the sum over all configurations, not its own.

        Asserting only that the slots differ is too weak -- a permuted
        configuration index passes that. With ``ensemble_mean=False`` the
        per-configuration slots are public, so compare each against its own
        independently built single-configuration run, which goes through the
        plain-atoms path that was correct all along.
        """
        atoms = ase.Atoms(
            "Si2",
            positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)],
            cell=(8, 8, 8),
            pbc=True,
        )
        phonons = abtem.FrozenPhonons(
            atoms, num_configs=3, sigmas=0.05, seed=7, ensemble_mean=False
        )
        potential = abtem.Potential(phonons, gpts=(64, 64), slice_thickness=2.0)
        probe = abtem.Probe(semiangle_cutoff=32, energy=ENERGY, extent=(8.0, 8.0))
        probe.grid.match(potential)
        got = self._run(potential, probe, atoms, lazy=False)
        assert got.shape[0] == 3

        for index, configuration in enumerate(phonons):
            single = abtem.Potential(
                configuration, gpts=(64, 64), slice_thickness=2.0
            )
            reference = self._run(single, probe, atoms, lazy=False)
            assert np.array_equal(got[index], reference), (
                f"slot {index} does not hold configuration {index}"
            )


class TestPrismPotentialEnsembleAccumulation:
    """The PRISM core-loss driver had the same axis-ordering defect.

    ``prism_transition_potential_scan`` shares
    ``_potential_ensemble_shape_and_metadata`` with the regular multislice
    driver, so its measurement also carries the potential's ensemble axes
    before the exit-plane axis -- but it indexed the plane part alone. It runs
    once per configuration with a length-1 ensemble axis, so an exit-plane
    slice starting at 1 or beyond selected nothing and a whole thickness
    series came back zero.

    Separately, ``SMatrix._build_ensemble_shape_metadata`` described that
    exit-plane axis with the per-*slice* ``ThicknessAxis`` (length
    ``num_slices``) instead of the per-exit-plane one, which raised from the
    measurement constructor and masked the zeros above (checked by
    TestPrismLazyExitPlanes, whose lazy axes come from the per-exit-plane one).
    """

    @staticmethod
    def _setup(num_configs=None, exit_planes=None):
        atoms = _si2_atoms()
        ensemble = (
            atoms
            if num_configs is None
            else abtem.FrozenPhonons(
                atoms, num_configs=num_configs, sigmas=0.05, seed=7
            )
        )
        potential = abtem.Potential(
            ensemble, gpts=(64, 64), slice_thickness=2.0, exit_planes=exit_planes
        )
        return atoms, potential

    def _run(self, atoms, potential, double_channel=False):
        s_matrix = abtem.SMatrix(
            potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=1
        )
        scan = abtem.GridScan(
            start=(0, 0), end=(1, 1), gpts=(2, 2), fractional=True, potential=potential
        )
        measurement = s_matrix.transition_potential_scan(
            transition_potentials=synthetic_transition_potential(
                gpts=potential.gpts, extent=potential.extent, n_transitions=2
            ),
            scan=scan,
            detectors=abtem.FlexibleAnnularDetector(),
            sites=atoms,
            double_channel=double_channel,
            lazy=False,
        )
        return np.asarray(abtem.core.backend.asnumpy(measurement.array))

    @pytest.mark.parametrize("num_configs", [1, 3])
    @pytest.mark.parametrize("double_channel", [False, True])
    def test_thickness_series_survives_a_potential_ensemble(
        self, num_configs, double_channel
    ):
        atoms, plain = self._setup(exit_planes=1)
        reference = self._run(atoms, plain, double_channel)

        _, potential = self._setup(num_configs=num_configs, exit_planes=1)
        got = self._run(atoms, potential, double_channel)

        assert got.shape == reference.shape
        assert got[0].sum() == 0.0  # entrance plane, nothing traversed yet
        assert all(got[i].sum() > 0.0 for i in range(1, got.shape[0]))
        # Frozen phonons displace the atoms, so this matches the un-displaced
        # reference only in aggregate -- the point is that the series carries
        # signal at all, and at the right magnitude, rather than being zeroed.
        for plane in range(1, got.shape[0]):
            assert got[plane].sum() == pytest.approx(
                reference[plane].sum(), rel=1e-3
            )

@devices
def test_ensemble_indices_handle_a_multi_axis_potential_ensemble(device):
    """A potential over an ``EnergyResolvedAtomsEnsemble`` has two ensemble
    axes, (energy, configuration), and multislice with exit planes indexes
    them through ``_validate_potential_ensemble_indices`` with 2-tuples.

    Every member must land in its own (energy, configuration, plane) cell, so
    each is compared against that configuration run as a plain potential.
    """
    from abtem.core.axes import EnergyLossAxis, FrozenPhononsAxis, ThicknessAxis

    def snapshot(energy_index, config_index):
        # A distinct atom position per member, so a mis-addressed member
        # differs from its reference.
        x = 1.0 + 0.4 * energy_index + 0.15 * config_index
        return ase.Atoms(
            "Si2",
            positions=[(x, 2.0, 1.0), (3.0, 3.0, 3.0)],
            cell=(5, 5, 4),
            pbc=True,
        )

    num_energies, num_configs = 2, 2
    snapshots = [
        [snapshot(e, c) for c in range(num_configs)] for e in range(num_energies)
    ]
    potential_kwargs = dict(gpts=32, slice_thickness=1.0, exit_planes=2, device=device)
    potential = abtem.Potential(
        abtem.EnergyResolvedAtomsEnsemble(snapshots, energies=[0.0, 0.05]),
        **potential_kwargs,
    )
    assert potential.ensemble_shape == (num_energies, num_configs)
    plane_wave = abtem.PlaneWave(energy=100e3, device=device)

    waves = plane_wave.multislice(potential, lazy=False)

    num_planes = len(potential.exit_planes)
    assert num_planes > 1
    assert waves.shape == (num_energies, num_configs, num_planes, 32, 32)
    assert [type(axis) for axis in waves.ensemble_axes_metadata] == [
        EnergyLossAxis,
        FrozenPhononsAxis,
        ThicknessAxis,
    ]

    got = to_host_array(waves)
    for e, c in itertools.product(range(num_energies), range(num_configs)):
        reference = plane_wave.multislice(
            abtem.Potential(snapshots[e][c], **potential_kwargs), lazy=False
        )
        # Same slices through the same code, so agreement is to rounding.
        np.testing.assert_allclose(
            got[e, c], to_host_array(reference), rtol=1e-5, atol=1e-6
        )


def test_prism_driver_refuses_a_multi_configuration_potential():
    """Called directly it would silently return 1/num_configurations.

    The SMatrix entry points always hand it a single-configuration
    sub-potential, but nothing enforced that, and after the indexing fix the
    wrong answer looks like a plausible monotone thickness series rather than
    obviously broken output.
    """
    from abtem.inelastic.core_loss import prism_transition_potential_scan

    atoms = ase.Atoms(
        "Si2", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)], cell=(8, 8, 8), pbc=True
    )
    potential = abtem.Potential(
        abtem.FrozenPhonons(atoms, num_configs=3, sigmas=0.05, seed=7),
        gpts=(64, 64),
        slice_thickness=2.0,
    )
    s_matrix = abtem.SMatrix(
        potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=1
    )
    scan = abtem.GridScan(
        start=(0, 0), end=(1, 1), gpts=(2, 2), fractional=True, potential=potential
    )
    with pytest.raises(NotImplementedError, match="one potential"):
        prism_transition_potential_scan(
            s_matrix,
            transition_potentials=synthetic_transition_potential(
                gpts=potential.gpts, extent=potential.extent, n_transitions=2
            ),
            scan=scan,
            detectors=[abtem.FlexibleAnnularDetector()],
            sites=atoms,
        )


class TestPrismEelsReductionChunking:
    """The per-site reduction cropped a bounding box spanning the *whole*
    scan and ran one ``tensordot`` over every position at once, so peak
    memory was ``n_positions * (scan_span + window)**2`` -- growing with the
    scan's spatial extent rather than with the output window. A
    production-sized PRISM-EELS scan demanded a single allocation in the
    hundreds of GB (abtem_issues/prism_eels_reduction_allocates_whole_scan.md).

    The reduction is now chunked over spatially contiguous blocks of scan
    rows, sized from the same memory-budget heuristic
    ``estimate_scan_batch_size`` already uses for the probe batch elsewhere,
    so each block's bounding box shrinks along with the block.
    """

    @staticmethod
    def _setup(n_rows, n_cols, interpolation=1):
        atoms = ase.Atoms(
            "Si2",
            positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)],
            cell=(8, 8, 8),
            pbc=True,
        )
        potential = abtem.Potential(
            atoms, gpts=(64, 64), slice_thickness=2.0, exit_planes=1
        )
        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20,
            interpolation=interpolation,
        )
        scan = abtem.GridScan(
            start=(0, 0),
            end=(n_rows, n_cols),
            gpts=(n_rows, n_cols),
            fractional=False,
            potential=potential,
        )
        return atoms, potential, s_matrix, scan

    @staticmethod
    def _run(atoms, s_matrix, scan, double_channel=False):
        measurement = s_matrix.transition_potential_scan(
            transition_potentials=synthetic_transition_potential(
                gpts=s_matrix.potential.gpts,
                extent=s_matrix.potential.extent,
                n_transitions=2,
            ),
            scan=scan,
            detectors=abtem.FlexibleAnnularDetector(),
            sites=atoms,
            double_channel=double_channel,
            lazy=False,
        )
        return np.asarray(abtem.core.backend.asnumpy(measurement.array))

    @pytest.mark.parametrize("double_channel", [False, True])
    # With this test's n_T=2 and 5 columns, the forced "position" budget 1
    # resolves (guess, then verified against the actual box) to single-row
    # batches at both interpolations. At interpolation 1 the reduction is on
    # the cell itself, whose box is the same for every batch, so 207 gives row
    # batches of 3, 3 and 1 (7 rows is not a multiple of 3, so the last batch
    # is uneven) and 350 a single batch. At interpolation 2 the box is cropped
    # around each batch and grows with it, so 350 gives batches of 3, 3 and 1.
    # Checked directly by recording the batch sizes of the reduction.
    @pytest.mark.parametrize(
        "interpolation, forced_budget", [(1, 1), (1, 207), (1, 350), (2, 1), (2, 350)]
    )
    def test_chunked_reduction_matches_a_single_whole_scan_batch(
        self, monkeypatch, double_channel, forced_budget, interpolation
    ):
        atoms, _, s_matrix, scan = self._setup(
            n_rows=7, n_cols=5, interpolation=interpolation
        )

        monkeypatch.setattr(
            "abtem.inelastic.core_loss.estimate_scan_batch_size",
            lambda *a, **k: 10**9,
            raising=False,
        )
        reference = self._run(atoms, s_matrix, scan, double_channel)

        monkeypatch.setattr(
            "abtem.inelastic.core_loss.estimate_scan_batch_size",
            lambda *a, **k: forced_budget,
            raising=False,
        )
        got = self._run(atoms, s_matrix, scan, double_channel)

        scale = np.abs(reference).max()
        assert scale > 0
        assert got.shape == reference.shape
        assert np.allclose(got, reference, rtol=1e-5, atol=scale * 1e-6)

    def test_the_reduction_never_crops_around_the_whole_scan(self, monkeypatch):
        """A budget of one position per batch collapses every block to a
        single scan row, so no crop call should ever see the full scan.

        On unfixed code the bounding box is computed once, globally, before
        the site loop -- so ``minimum_crop`` sees every position in that one
        call regardless of how small a budget is forced here, and this
        assertion catches that directly.
        """
        from abtem.prism.utils import minimum_crop as _real_minimum_crop

        n_rows, n_cols = 9, 6
        # A window smaller than the cell is cropped around each batch.
        atoms, _, s_matrix, scan = self._setup(
            n_rows=n_rows, n_cols=n_cols, interpolation=2
        )
        n_positions = n_rows * n_cols

        call_sizes = []

        def _recording_minimum_crop(positions, shape):
            call_sizes.append(int(positions.shape[0]))
            return _real_minimum_crop(positions, shape)

        monkeypatch.setattr(
            "abtem.inelastic.core_loss.estimate_scan_batch_size",
            lambda *a, **k: 1,
            raising=False,
        )
        monkeypatch.setattr(
            "abtem.prism.utils.minimum_crop", _recording_minimum_crop
        )

        self._run(atoms, s_matrix, scan)

        assert call_sizes
        assert max(call_sizes) < n_positions, (
            f"minimum_crop saw {max(call_sizes)} of {n_positions} positions "
            "in one call -- the reduction still crops around the whole scan"
        )

    def test_the_reduction_still_shrinks_when_the_naive_guess_already_covers_the_whole_scan(
        self, monkeypatch
    ):
        """The sizing guess assumes no bounding-box growth; verifying and
        shrinking it against the actual box only ran when the guess landed
        BELOW the row count (``rows_per_batch < n_rows``). Capping the guess
        at n_rows and then skipping verification because the capped value no
        longer compares less than itself reproduced the original defect
        exactly for that case -- a whole-scan block, never checked -- and it
        is not a corner case: any scan whose physical span exceeds its
        output window (i.e. most of them) grows the real box past the
        no-growth estimate, so this triggers whenever the naive guess merely
        reaches the row count, not only when it wildly overshoots it.

        A scan physically wider than the window (unlike this class's other
        tests, whose scans are drawn in the same few Angstrom as the window
        and never exercise real box growth) with a budget picked to land the
        naive guess exactly at the row count reproduces this directly. The
        window is smaller than the cell (interpolation 2), so the box is
        cropped around each batch and grows with it.
        """
        from abtem.prism.utils import minimum_crop as _real_minimum_crop

        atoms = ase.Atoms(
            "Si2", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)], cell=(8, 8, 8),
            pbc=True,
        )
        potential = abtem.Potential(
            atoms, gpts=(64, 64), slice_thickness=2.0, exit_planes=1
        )
        s_matrix = abtem.SMatrix(
            potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=2
        )
        n_rows, n_cols = 9, 6
        n_positions = n_rows * n_cols
        # Span (40 x 30 A) well past the ~8 A window -- unlike _setup's scans,
        # whose end == gpts puts every position within the window itself.
        scan = abtem.GridScan(
            start=(0, 0), end=(40, 30), gpts=(n_rows, n_cols), fractional=False,
            potential=potential,
        )

        call_sizes = []

        def _recording_minimum_crop(positions, shape):
            call_sizes.append(int(positions.shape[0]))
            return _real_minimum_crop(positions, shape)

        # n_T=2, row_cols=6: guess = budget // 2 // 6. 300 -> 25, capped to
        # n_rows=9 -- the exact "guess already covers everything" case.
        monkeypatch.setattr(
            "abtem.inelastic.core_loss.estimate_scan_batch_size",
            lambda *a, **k: 300,
            raising=False,
        )
        monkeypatch.setattr(
            "abtem.prism.utils.minimum_crop", _recording_minimum_crop
        )

        self._run(atoms, s_matrix, scan)

        # The sizing pass itself legitimately probes the full-size candidate
        # first (it has to, to find out it is too big) -- max(call_sizes)
        # would see that probe regardless of whether the fix works. The
        # sizing pass always finishes before any site's real reduction call,
        # so the LAST recorded call is from that real work; on unfixed code
        # (no sizing pass at all when the guess lands at n_rows) every call,
        # including the last, is the unchunked whole-scan size.
        assert call_sizes
        assert call_sizes[-1] < n_positions, (
            f"the last minimum_crop call saw {call_sizes[-1]} of "
            f"{n_positions} positions -- the reduction itself is still "
            f"unchunked (all calls: {call_sizes})"
        )

    def test_the_batch_size_is_estimated_from_the_box_the_reduction_uses(
        self, monkeypatch
    ):
        """At interpolation 1 the window is the cell, and the reduction runs on
        the cell for every batch, so the box does not grow with the number of
        rows. The batch size is estimated from that box: a budget that holds
        the whole scan on the cell gives one batch, and one position less gives
        more. Sizing the box as ``minimum_crop`` does, around the batch's
        positions and growing with their span, splits this wide scan into
        several batches under the same budget.
        """
        from abtem.prism import utils

        n_rows, n_cols, n_T = 9, 6, 2
        atoms, potential, s_matrix, _ = self._setup(n_rows=1, n_cols=1)
        scan = abtem.GridScan(
            start=(0, 0),
            end=(40, 30),
            gpts=(n_rows, n_cols),
            fractional=False,
            potential=potential,
        )

        batch_rows = []
        real_batch_crop = utils.batch_crop_2d

        def _recording_batch_crop(array, corners, shape):
            batch_rows.append(len(corners) // n_cols)
            return real_batch_crop(array, corners, shape)

        monkeypatch.setattr("abtem.prism.utils.batch_crop_2d", _recording_batch_crop)

        def _run_with_budget(budget):
            monkeypatch.setattr(
                "abtem.inelastic.core_loss.estimate_scan_batch_size",
                lambda *a, **k: budget,
                raising=False,
            )
            batch_rows.clear()
            return self._run(atoms, s_matrix, scan)

        reference = _run_with_budget(10**9)
        assert set(batch_rows) == {n_rows}

        # One batch needs (n_T * n_k + n_positions * n_T) cell-sized planes: the
        # scattered window of every transition and plane wave, and the reduced
        # waves of every position.
        fitting_budget = n_T * (len(s_matrix) + n_rows * n_cols)
        got = _run_with_budget(fitting_budget)
        assert set(batch_rows) == {n_rows}
        assert np.array_equal(got, reference)

        _run_with_budget(fitting_budget - 1)
        assert max(batch_rows) < n_rows

    def test_minimum_crop_does_not_scale_with_the_number_of_sites(
        self, monkeypatch
    ):
        """minimum_crop's result for a row batch depends only on that
        batch's own positions, never on which site or exit plane is being
        recorded -- but it was called from inside _reduce_and_record, which
        runs once per (site, exit plane). Computing it there recomputed the
        identical box on every one of those calls, scaling the call count
        with the site count for no reason.
        """
        from abtem.prism.utils import minimum_crop as _real_minimum_crop

        monkeypatch.setattr(
            "abtem.inelastic.core_loss.estimate_scan_batch_size",
            lambda *a, **k: 1,
            raising=False,
        )

        def _call_count(n_sites):
            atoms = ase.Atoms(
                numbers=[14] * n_sites,
                positions=[(i * 1.0, i * 1.0, i * 0.5) for i in range(n_sites)],
                cell=(8, 8, 8),
                pbc=True,
            )
            potential = abtem.Potential(
                atoms, gpts=(64, 64), slice_thickness=1.0, exit_planes=1
            )
            s_matrix = abtem.SMatrix(
                potential=potential, energy=ENERGY, semiangle_cutoff=20,
                interpolation=2,
            )
            scan = abtem.GridScan(
                start=(0, 0), end=(7, 5), gpts=(7, 5), fractional=False,
                potential=potential,
            )

            calls = []

            def _recording_minimum_crop(positions, shape):
                calls.append(int(positions.shape[0]))
                return _real_minimum_crop(positions, shape)

            monkeypatch.setattr(
                "abtem.prism.utils.minimum_crop", _recording_minimum_crop
            )
            self._run(atoms, s_matrix, scan)
            return len(calls)

        assert _call_count(n_sites=2) == _call_count(n_sites=8)

    @pytest.mark.parametrize(
        "interpolation, cell_frame", [(1, True), ((1, 2), False), (2, False)]
    )
    def test_elastic_and_core_loss_reductions_share_the_cell_frame_rule(
        self, monkeypatch, interpolation, cell_frame
    ):
        """A window as large as the grid in both directions is the cell for
        every position, in the elastic reduction and in the core-loss one. A
        window smaller along one direction only, ``interpolation=(1, 2)``, is
        centred on each position in both."""
        from abtem.prism import s_matrix as s_matrix_module
        from abtem.prism import utils

        real_window_crop = utils.window_crop
        boxes = {"elastic": [], "core loss": []}

        def _recording(name):
            def window_crop(positions, window_gpts, gpts):
                boxes[name].append(real_window_crop(positions, window_gpts, gpts))
                return boxes[name][-1]

            return window_crop

        monkeypatch.setattr(s_matrix_module, "window_crop", _recording("elastic"))
        monkeypatch.setattr(utils, "window_crop", _recording("core loss"))

        atoms, potential, s_matrix, _ = self._setup(
            n_rows=1, n_cols=1, interpolation=interpolation
        )
        scan = abtem.GridScan(
            start=(1, 1),
            end=(3, 3),
            gpts=(2, 2),
            fractional=False,
            potential=potential,
        )
        s_matrix.reduce(scan=scan, detectors=abtem.WavesDetector(), lazy=False)
        self._run(atoms, s_matrix, scan)

        for name, recorded in boxes.items():
            assert recorded, name
            for crop_corner, size, corners in recorded:
                assert (not np.any(corners)) == cell_frame, name
                if cell_frame:
                    assert crop_corner == (0, 0), name
                    assert size == s_matrix.downsampled_gpts, name

    def test_matches_reference_with_a_custom_scans_positions_axis(
        self, monkeypatch
    ):
        """``CustomScan.ensemble_axes_metadata`` is a ``PositionsAxis``, which
        carries an explicit per-position ``values`` tuple -- unlike
        ``GridScan``'s linear ``ScanAxis``, which every other test here uses.
        Reusing that tuple unchanged for a batch covering fewer positions
        than the full scan raises inside ``Waves.__init__`` (it validates an
        ordinal axis's ``values`` length against the array), so this needs
        the axis restricted to the batch's own row range -- the same
        restriction dask's own ensemble partitioning already applies per
        block via ``AxisMetadata.__getitem__``.
        """
        atoms, potential, s_matrix, _ = self._setup(n_rows=1, n_cols=1)
        rng = np.random.default_rng(3)
        # A deliberately non-uniform layout: two tight clusters far apart,
        # so the sizing heuristic's "first batch is representative" guess
        # (built for a regular raster) does not hold -- correctness must
        # not depend on it, only the chosen batch size might be suboptimal.
        positions = np.concatenate(
            [
                rng.uniform(0.1, 0.3, size=(4, 2)),
                rng.uniform(7.0, 7.9, size=(4, 2)),
            ]
        ).astype(np.float32)
        scan = abtem.scan.CustomScan(positions)

        monkeypatch.setattr(
            "abtem.inelastic.core_loss.estimate_scan_batch_size",
            lambda *a, **k: 10**9,
            raising=False,
        )
        reference = self._run(atoms, s_matrix, scan)

        for forced_budget in (1, 3):
            monkeypatch.setattr(
                "abtem.inelastic.core_loss.estimate_scan_batch_size",
                lambda *a, _f=forced_budget, **k: _f,
                raising=False,
            )
            got = self._run(atoms, s_matrix, scan)

            scale = np.abs(reference).max()
            assert scale > 0
            assert got.shape == reference.shape
            assert np.allclose(got, reference, rtol=1e-5, atol=scale * 1e-6)

    @pytest.mark.parametrize("double_channel", [False, True])
    def test_matches_reference_with_multiple_exit_planes(
        self, monkeypatch, double_channel
    ):
        """Multiple exit planes add a broadcast slice (single-channel) or a
        plain int (double-channel) ahead of the scan axes in the
        measurement's leading indices (see
        ``TestPrismPotentialEnsembleAccumulation``) -- composing that with
        the new row slice is the trickiest indexing case this change adds,
        and no other test here uses more than one exit plane.
        """
        atoms = ase.Atoms(
            "Si2",
            positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)],
            cell=(8, 8, 8),
            pbc=True,
        )
        potential = abtem.Potential(
            atoms, gpts=(64, 64), slice_thickness=2.0, exit_planes=1
        )
        assert len(potential.exit_planes) > 1
        s_matrix = abtem.SMatrix(
            potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=1
        )
        scan = abtem.GridScan(
            start=(0, 0), end=(7, 5), gpts=(7, 5), fractional=False, potential=potential
        )

        monkeypatch.setattr(
            "abtem.inelastic.core_loss.estimate_scan_batch_size",
            lambda *a, **k: 10**9,
            raising=False,
        )
        reference = self._run(atoms, s_matrix, scan, double_channel)

        for forced_budget in (1, 25, 40):
            monkeypatch.setattr(
                "abtem.inelastic.core_loss.estimate_scan_batch_size",
                lambda *a, _f=forced_budget, **k: _f,
                raising=False,
            )
            got = self._run(atoms, s_matrix, scan, double_channel)

            scale = np.abs(reference).max()
            assert scale > 0
            assert got.shape == reference.shape
            assert np.allclose(got, reference, rtol=1e-5, atol=scale * 1e-6)


class TestPrismLazyExitPlanes:
    """The lazy PRISM path omitted the exit-plane axis from its block shape.

    ``SMatrix.transition_potential_scan(lazy=True)`` declared ``chunks`` and
    ``new_axis`` from the S-matrix ensemble and the scan only, while each block
    carries a measurement that also has an exit-plane axis whenever the
    potential has more than one. The declared block shape therefore disagreed
    with the computed one: the result came back with one more array dimension
    than axes metadata, so every method pairing the two raised, and with
    ``ensemble_mean=False`` it failed outright during compute.

    The eager branch got this right through ``_build_ensemble_shape_metadata``;
    the two computed their bookkeeping separately, which is how they diverged.
    """

    def _run(self, potential, lazy, double_channel=False):
        atoms = _si2_atoms()
        s_matrix = abtem.SMatrix(
            potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=1
        )
        scan = abtem.GridScan(
            start=(0, 0), end=(1, 1), gpts=(2, 2), fractional=True, potential=potential
        )
        measurement = s_matrix.transition_potential_scan(
            transition_potentials=synthetic_transition_potential(
                gpts=potential.gpts, extent=potential.extent, n_transitions=2
            ),
            scan=scan,
            detectors=abtem.FlexibleAnnularDetector(),
            sites=atoms,
            double_channel=double_channel,
            lazy=lazy,
        )
        if lazy:
            # The shape dask was told to expect must equal what the blocks
            # actually produce -- a chunks declaration that is never honoured
            # (single block) would otherwise go unnoticed, and that mismatch is
            # the whole defect.
            declared = measurement.array.shape
            measurement = measurement.compute(progress_bar=False)
            assert declared == np.asarray(
                abtem.core.backend.asnumpy(measurement.array)
            ).shape, f"dask declared {declared}, blocks produced a different shape"
        return measurement

    def _potential(self, num_configs=None, exit_planes=None, ensemble_mean=True):
        atoms = _si2_atoms()
        ensemble = (
            atoms
            if num_configs is None
            else abtem.FrozenPhonons(
                atoms,
                num_configs=num_configs,
                sigmas=0.05,
                seed=7,
                ensemble_mean=ensemble_mean,
            )
        )
        return abtem.Potential(
            ensemble, gpts=(64, 64), slice_thickness=2.0, exit_planes=exit_planes
        )

    # The declared chunks do not depend on double_channel (see
    # SMatrix.transition_potential_scan); the per-block measurement does, as
    # the two channels index the ensemble and exit-plane axes differently, so
    # the double channel runs on the two cases with both axes.
    @pytest.mark.parametrize(
        "num_configs, exit_planes, ensemble_mean, double_channel",
        [
            (None, None, True, False),
            (None, 1, True, False),
            (1, 1, True, False),
            (3, 1, True, False),
            (3, 1, False, False),
            (None, [1, 3], True, False),
            (3, 1, True, True),
            (3, 1, False, True),
        ],
        ids=[
            "plain-1", "plain-5", "phonons1-5", "phonons3-5", "no-mean", "explicit",
            "phonons3-5-double", "no-mean-double",
        ],
    )
    def test_lazy_matches_eager(self, num_configs, exit_planes, ensemble_mean,
                                double_channel):
        potential = self._potential(num_configs, exit_planes, ensemble_mean)
        eager = self._run(potential, lazy=False, double_channel=double_channel)
        lazy = self._run(potential, lazy=True, double_channel=double_channel)

        # Compare the axes themselves, not their type names: the defect an
        # earlier commit in this PR fixed was two *ThicknessAxis* objects with
        # different values (per-slice vs per-exit-plane), which a type-name
        # comparison cannot see.
        assert eager.axes_metadata == lazy.axes_metadata

        eager_array = np.asarray(abtem.core.backend.asnumpy(eager.array))
        lazy_array = np.asarray(abtem.core.backend.asnumpy(lazy.array))
        assert eager_array.shape == lazy_array.shape
        assert np.array_equal(eager_array, lazy_array)

    @pytest.mark.parametrize("lazy", [False, True])
    def test_array_dimensions_match_the_axes_metadata(self, lazy):
        """The failure users hit: every method pairing the two raised."""
        measurement = self._run(self._potential(exit_planes=1), lazy=lazy)
        array = np.asarray(abtem.core.backend.asnumpy(measurement.array))
        assert array.ndim == len(measurement.axes_metadata)
        measurement.to_cpu()  # raised before the fix


class TestTransitionPotentialDeviceMemo:
    """``copy_to_device`` rebuilds through ``__init__``, which recomputes
    ``_local_potential`` from the array even when the array is already on
    the target device (the free ``copy_to_device`` being a no-op there is
    invisible to ``__init__``). Both core-loss drivers call it once per task
    on a transition potential that arrives as one graph node shared by every
    task on a worker, so the recomputation -- and, on GPU, the host-to-device
    upload beneath it -- happened once per task rather than once per worker.

    The memo must not hand out the same wrapper object twice: ``scatter``
    mutates what it is given (``self._array = ...``, ``self.grid.match``),
    so two tasks sharing one instance would race on those mutations exactly
    as ``BaseTransitionPotential._task_local`` exists to prevent one step
    earlier. What is cached is the immutable-in-practice array data; each
    call still returns a fresh, independently-mutable wrapper.
    """

    @staticmethod
    def _make():
        rng = np.random.default_rng(0)
        array = (
            rng.standard_normal((2, 32, 32)) + 1j * rng.standard_normal((2, 32, 32))
        ).astype(np.complex64)
        return TransitionPotentialArray(
            Z=14,
            array=array,
            energy=ENERGY,
            extent=(8.0, 8.0),
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
            metadata={"Z": 14, "n": 1, "l": 0},
        )

    def test_repeated_calls_share_the_uploaded_array(self):
        tp = self._make()
        a = tp.copy_to_device("cpu")
        b = tp.copy_to_device("cpu")

        assert a is not b, "each call must return an independently-mutable wrapper"
        assert a.array is b.array, "the array data itself should be memoized"
        assert a._local_potential is b._local_potential

    def test_mutating_one_wrapper_does_not_corrupt_the_cache(self):
        """The defect this guards against: if the memo cached the *wrapper*
        rather than its array data, mutating one task's copy (as ``scatter``
        does) would corrupt what the next task pulls from the cache.
        """
        tp = self._make()
        a = tp.copy_to_device("cpu")
        untouched_array = a.array.copy()

        a._array = np.zeros_like(a._array)  # exactly what scatter() does

        c = tp.copy_to_device("cpu")
        assert np.array_equal(c.array, untouched_array)

    def test_sibling_task_local_views_share_one_upload(self, monkeypatch):
        """The mechanism the fix relies on: _task_local's shallow copy shares
        the cache dict *reference*, so two per-task views spawned from the
        same shared node see the same memo -- the shape every real driver
        call takes (_task_local, then copy_to_device, per task).
        """
        import abtem.inelastic.core_loss as cl

        tp = self._make()
        calls = []
        real_copy_to_device = cl.copy_to_device

        def counting_copy_to_device(array, device):
            calls.append(device)
            return real_copy_to_device(array, device)

        monkeypatch.setattr(cl, "copy_to_device", counting_copy_to_device)

        view1 = tp._task_local()
        view2 = tp._task_local()
        view1.copy_to_device("cpu")
        view2.copy_to_device("cpu")

        assert len(calls) == 1, (
            f"expected one upload shared across sibling task-local views, "
            f"got {len(calls)}"
        )

    def test_matches_an_unmemoized_rebuild(self):
        tp = self._make()
        memoized = tp.copy_to_device("cpu")
        plain = ArrayObject.copy_to_device(tp, "cpu")

        assert np.array_equal(
            np.asarray(memoized.array), np.asarray(plain.array)
        )
        assert np.array_equal(
            np.asarray(memoized._local_potential), np.asarray(plain._local_potential)
        )

    def test_pickling_still_drops_both_device_caches(self):
        """__copy__ exists so copy.copy shares _device_array_cache; pickling
        must still go through __getstate__ and drop it (and the older
        _local_potential_device_cache), same as before __copy__ existed --
        a cupy array riding through pickle would break unpickling on a
        CPU-only worker.
        """
        import pickle

        tp = self._make()
        tp.copy_to_device("cpu")
        assert tp._device_array_cache

        restored = pickle.loads(pickle.dumps(tp))
        assert restored._device_array_cache == {}
        assert restored._local_potential_device_cache is None
        assert np.array_equal(np.asarray(restored.array), np.asarray(tp.array))


class TestScatterBatchBucketing:
    """generate_scattered_waves pads each site chunk at or above
    core_loss._SCATTER_BATCH_BUCKET up to a multiple of it before
    scattering, so the ifft2 batch shape depends on the bucket rather than
    the data-dependent surviving-site count. The padding is capped at the
    chunk's own site ceiling, so a VRAM-constrained run's small max_batch
    chunks are never rounded up past their memory budget.
    """

    @staticmethod
    def _make_tp(n_transitions=2, gpts=(32, 32)):
        rng = np.random.default_rng(0)
        array = (
            rng.standard_normal((n_transitions, *gpts))
            + 1j * rng.standard_normal((n_transitions, *gpts))
        ).astype(np.complex64)
        return TransitionPotentialArray(
            Z=5, array=array, energy=ENERGY, extent=(8.0, 8.0),
            ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_transitions)))],
            metadata={"Z": 5, "n": 1, "l": 0},
        )

    @staticmethod
    def _make_waves():
        probe = abtem.Probe(
            energy=ENERGY, extent=(8.0, 8.0), gpts=(32, 32), semiangle_cutoff=20
        )
        return probe.build(lazy=False)

    @staticmethod
    def _record_ifft2_batches(monkeypatch):
        import abtem.inelastic.core_loss as cl

        real_ifft2 = cl.ifft2
        batch_shapes = set()

        def recording_ifft2(x, *args, **kwargs):
            batch_shapes.add(x.shape[0])
            return real_ifft2(x, *args, **kwargs)

        monkeypatch.setattr(cl, "ifft2", recording_ifft2)
        return batch_shapes

    def test_ifft2_batch_shapes_are_bucketed(self, monkeypatch):
        tp = self._make_tp(n_transitions=2)
        waves = self._make_waves()
        batch_shapes = self._record_ifft2_batches(monkeypatch)

        rng = np.random.default_rng(1)
        for n_sites in (3, 5, 7, 11, 17, 23):
            sites = rng.uniform(0, 8, size=(n_sites, 2)).astype(np.float32)
            for _ in tp.generate_scattered_waves(waves, sites, max_batch=10**9):
                pass

        # ifft2's batch axis is the site axis alone -- the transitions axis
        # (length n_transitions=2 here) is a separate, un-folded dimension,
        # e.g. (11, 2, 32, 32) for 11 sites, not (22, 32, 32). Unbucketed,
        # this reproduces the site counts exactly: {3, 5, 7, 11, 17, 23}, one
        # distinct shape per call. Bucketed to multiples of 8, every one
        # collapses onto one of 3.
        assert batch_shapes == {8, 16, 24}

    def test_matches_a_direct_unchunked_unpadded_scatter(self):
        """Below, at and just above a bucket boundary (the bucket is 8): the
        padded path must match scatter() called directly on the real sites,
        which pads and chunks nothing."""
        tp = self._make_tp(n_transitions=2)
        waves = self._make_waves()
        rng = np.random.default_rng(2)

        for n_sites in (1, 8, 9, 17):
            sites = rng.uniform(0, 8, size=(n_sites, 2)).astype(np.float32)
            chunks = list(
                tp.generate_scattered_waves(
                    waves, sites, max_batch=10**9, threshold=0.0
                )
            )
            assert len(chunks) == 1  # max_batch huge enough for one chunk
            sites_chunk, padded = chunks[0]
            # the yielded chunk is never padded
            assert len(sites_chunk) == n_sites, f"{n_sites} sites"

            direct = tp.scatter(waves, sites)
            assert np.array_equal(
                np.asarray(padded.array), np.asarray(direct.array)
            ), f"{n_sites} sites"

    def test_padding_never_exceeds_a_small_explicit_max_batch(self, monkeypatch):
        """A memory-tight run (max_batch below the bucket) must not have its
        per-chunk budget multiplied by padding: without the site-ceiling cap
        the last chunk of 37 sites in chunks of 6 (one real site) would pad up
        to a full bucket of 8. Padding is per real chunk, so the several
        chunks must still reassemble to the direct, unchunked scatter."""
        tp = self._make_tp(n_transitions=2)
        waves = self._make_waves()
        rng = np.random.default_rng(3)
        sites = rng.uniform(0, 8, size=(37, 2)).astype(np.float32)

        batch_shapes = self._record_ifft2_batches(monkeypatch)
        chunks = list(
            tp.generate_scattered_waves(waves, sites, max_batch=6, threshold=0.0)
        )
        assert batch_shapes == {6}
        assert len(chunks) > 1
        assert sum(len(c[0]) for c in chunks) == 37

        reassembled = np.concatenate([np.asarray(c[1].array) for c in chunks], axis=0)
        direct = tp.scatter(waves, sites)
        assert np.array_equal(reassembled, np.asarray(direct.array))

    def test_the_site_ceiling_probe_runs_once_per_waves_shape(self, monkeypatch):
        """The site ceiling for max_batch="auto" needs an extra validate_chunks
        call with a sentinel site count. Running it on every
        generate_scattered_waves call measured as a ~10x regression on the
        "auto" path in a real per-slice driver loop, so it is memoized -- but
        per waves shape, so a stale ceiling never leaks into another grid. An
        explicit int max_batch already is the ceiling and never probes."""
        from abtem.core.chunks import validate_chunks as real_validate_chunks
        from abtem.inelastic.core_loss import _SCATTER_BATCH_CEILING_PROBE_SITES

        tp = self._make_tp(n_transitions=2)
        rng = np.random.default_rng(5)
        sites = rng.uniform(0, 8, size=(5, 2)).astype(np.float32)

        probes = []

        def counting_validate_chunks(*args, **kwargs):
            shape = kwargs.get("shape", args[0] if args else None)
            if shape[0] == _SCATTER_BATCH_CEILING_PROBE_SITES:
                probes.append(shape)
            return real_validate_chunks(*args, **kwargs)

        monkeypatch.setattr(
            "abtem.inelastic.core_loss.validate_chunks", counting_validate_chunks
        )

        def scatter_all(waves, max_batch):
            for _ in tp.generate_scattered_waves(
                waves, sites, max_batch=max_batch, threshold=0.0
            ):
                pass

        waves = self._make_waves()
        for _ in range(5):
            scatter_all(waves, "auto")
        assert len(probes) == 1

        # two probe positions: waves of shape (2, 32, 32) rather than (32, 32)
        probe = abtem.Probe(
            energy=ENERGY, extent=(8.0, 8.0), gpts=(32, 32), semiangle_cutoff=20
        )
        scatter_all(probe.build(scan=[(0.0, 0.0), (1.0, 1.0)], lazy=False), "auto")
        assert len(probes) == 2, "a new waves shape must probe its own ceiling"

        scatter_all(waves, 7)
        assert len(probes) == 2, "an int max_batch must not probe"


def test_ceil_to_multiple_rounds_up_never_down():
    """Contrast case for _nearest_power_of_two (core/chunks.py), which rounds
    *down* when the upper power would overshoot by more than 25% -- safe for a
    VRAM budget, unsafe for a batch bucket that must keep every item."""
    from abtem.core.chunks import _ceil_to_multiple

    table = {0: 0, 1: 8, 7: 8, 8: 8, 9: 16, 16: 16, 17: 24, 100: 104}
    got = {n: _ceil_to_multiple(n, 8) for n in table}
    assert got == table


class TestScatteringSitesReadReciprocalSpaceWaves:
    """``Waves`` with ``reciprocal_space=True`` give the same result as the same
    waves in real space in every public method that reads the waves as a real-space
    wave function."""

    @staticmethod
    def _setup(lazy):
        # Three waves on a 16 x 20 grid with two transitions and five sites: the
        # ensemble, transition and site counts and both grid sides all differ.
        gpts = (16, 20)
        x, y = np.meshgrid(*(np.arange(n) * 0.1 for n in gpts), indexing="ij")
        centers = [(0.4, 0.5), (0.8, 1.0), (1.2, 1.5)]
        psi = np.stack(
            [np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.05) for cx, cy in centers]
        ).astype(np.complex128)
        real = abtem.Waves(
            psi,
            energy=ENERGY,
            sampling=0.1,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1, 2))],
        )
        if lazy:
            real = real.ensure_lazy()
        recip = real.ensure_reciprocal_space()
        assert recip.reciprocal_space

        tp = synthetic_transition_potential(
            gpts=gpts, extent=(1.6, 2.0), energy=ENERGY, n_transitions=2
        )
        sites = np.array(
            [[0.4, 0.5], [0.8, 1.0], [1.2, 1.5], [0.1, 1.9], [1.5, 0.1]]
        )
        return tp, real, recip, sites

    @pytest.fixture(autouse=True)
    def _precision(self):
        with abtem.config.set({"precision": "float64", "fft": "numpy"}):
            yield

    @staticmethod
    def _assert_close(actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=1e-10 * np.abs(expected).max()
        )

    @pytest.mark.parametrize("lazy", [False, True])
    def test_scatter(self, lazy):
        tp, real, recip, sites = self._setup(lazy)

        expected = tp.scatter(real, sites)
        result = tp.scatter(recip, sites)

        assert not result.reciprocal_space
        self._assert_close(result.compute().array, expected.compute().array)

    @pytest.mark.parametrize("lazy", [False, True])
    def test_generate_scattered_waves(self, lazy):
        tp, real, recip, sites = self._setup(lazy)

        def collect(waves):
            chunks = list(tp.generate_scattered_waves(waves, sites, max_batch=2))
            assert len(chunks) > 1
            assert not any(w.reciprocal_space for _, w in chunks)
            return np.concatenate([w.compute().array for _, w in chunks])

        self._assert_close(collect(recip), collect(real))

    @pytest.mark.parametrize("lazy", [False, True])
    def test_absolute_threshold(self, lazy):
        tp, real, recip, _ = self._setup(lazy)

        expected = tp.absolute_threshold(real, 0.5)
        assert expected > 0
        assert tp.absolute_threshold(recip, 0.5) == pytest.approx(expected, rel=1e-8)

    @pytest.mark.parametrize("lazy", [False, True])
    def test_filter_sites(self, lazy):
        tp, real, recip, sites = self._setup(lazy)

        cut = tp.absolute_threshold(real, 0.5)
        expected = tp.filter_sites(real, sites, cut)
        assert 0 < len(expected) < len(sites)
        np.testing.assert_array_equal(tp.filter_sites(recip, sites, cut), expected)

    def test_scatter_with_a_threshold(self):
        tp, real, recip, sites = self._setup(lazy=False)

        cut = tp.absolute_threshold(real, 0.5)
        expected = tp.scatter(real, sites, threshold=cut)
        assert 0 < expected.shape[0] < len(sites) * len(tp)
        self._assert_close(
            tp.scatter(recip, sites, threshold=cut).array, expected.array
        )


def _expansion_scope_setup():
    """Si cell with one seeded displacement and three slices."""
    atoms = list(
        abtem.FrozenPhonons(ase.build.bulk("Si", cubic=True), 1, sigmas=0.1, seed=1)
    )[0]
    potential = abtem.Potential(
        atoms, gpts=(32, 32), slice_thickness=atoms.cell[2, 2] / 3
    )
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
    probe.grid.match(potential)
    return atoms, potential, probe


def _core_loss_scan(lazy, double_channel=False, **kwargs):
    atoms, potential, probe = _expansion_scope_setup()
    return probe.transition_potential_scan(
        potential=potential,
        transition_potentials=synthetic_transition_potential(
            gpts=potential.gpts, extent=potential.extent, n_transitions=2
        ),
        scan=np.array([[0.0, 0.0]]),
        detectors=abtem.AnnularDetector(inner=0.0, outer=None),
        double_channel=double_channel,
        lazy=lazy,
        sites=atoms,
        **kwargs,
    )


@pytest.mark.parametrize(
    "option, match",
    [
        ("detectors_elastic", "detectors_elastic"),
        ("algorithm", "expansion_scope='full'"),
    ],
)
def test_unsupported_core_loss_options_are_refused_up_front(option, match, monkeypatch):
    """``detectors_elastic`` was declared and never read, so elastic detectors
    were accepted in silence and only the inelastic measurement came back;
    ``RealSpaceMultislice(expansion_scope="full")`` is not supported by the
    core-loss multislice.

    Both are refused when the call is made, before the probe is built at every
    scan position, eager or lazy: abTEM is lazy by default, and a check in the
    per-chunk worker alone would fail only at compute time, inside a dask
    traceback. The public driver refuses them too. The refusal precedes every
    use of ``double_channel``, so one channel stands for both.
    """
    from abtem.multislice import (
        RealSpaceMultislice,
        transition_potential_multislice_and_detect,
    )

    refused = {
        "detectors_elastic": [abtem.AnnularDetector(inner=50, outer=150)],
        "algorithm": RealSpaceMultislice(expansion_scope="full"),
    }[option]

    atoms, potential, probe = _expansion_scope_setup()
    with pytest.raises(NotImplementedError, match=match):
        transition_potential_multislice_and_detect(
            probe.build(scan=np.array([[0.0, 0.0]]), lazy=False),
            potential.build(),
            synthetic_transition_potential(
                gpts=potential.gpts, extent=potential.extent, n_transitions=2
            ),
            detectors=[abtem.AnnularDetector(inner=0.0, outer=None)],
            double_channel=False,
            sites=atoms,
            **{option: refused},
        )

    builds = []
    monkeypatch.setattr(
        abtem.Probe, "build", lambda self, *args, **kwargs: builds.append(kwargs)
    )
    for lazy in (False, True):
        try:
            _core_loss_scan(lazy, **{option: refused})
        except NotImplementedError as error:
            assert re.search(match, str(error)), f"lazy={lazy}: {error}"
        else:
            pytest.fail(f"lazy={lazy}: {option} was not refused")
        assert builds == [], f"lazy={lazy}: the probe was built before the refusal"


def test_no_elastic_detectors_are_not_refused():
    """An empty list asks for no elastic detectors, so nothing is dropped;
    raising on it, or on an explicit ``None`` a caller forwards, would turn a
    working call into a crash for no benefit."""
    for detectors_elastic in (None, []):
        for lazy in (False, True):
            measurement = _core_loss_scan(lazy, detectors_elastic=detectors_elastic)
            if lazy:
                measurement = measurement.compute(progress_bar=False)
            assert np.asarray(abtem.core.backend.asnumpy(measurement.array)).size, (
                f"detectors_elastic={detectors_elastic}, lazy={lazy}"
            )


@pytest.mark.parametrize("double_channel", [False, True])
def test_propagator_expansion_scope_runs_in_the_core_loss_multislice(double_channel):
    """``RealSpaceMultislice(expansion_scope="propagator")`` is the supported
    real-space choice."""
    from abtem.multislice import RealSpaceMultislice

    default = _core_loss_scan(False, double_channel)
    realspace = _core_loss_scan(
        False, double_channel,
        algorithm=RealSpaceMultislice(expansion_scope="propagator"),
    )
    reference = np.asarray(abtem.core.backend.asnumpy(default.array))
    result = np.asarray(abtem.core.backend.asnumpy(realspace.array))
    assert result.shape == reference.shape
    # Both propagators solve the same equation; relative to the maximum they differ
    # by 9e-5 (single channel) and 1e-3 (double channel).
    assert np.abs(result - reference).max() < 1e-2 * np.abs(reference).max()
