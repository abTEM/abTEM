"""Tests for the continuum-energy quadrature and the continuum normalisation.

The quadrature and the bookkeeping around it are pure arithmetic and need no
GPAW; the tests that actually solve for a continuum state are skipped without it.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

try:
    import gpaw  # noqa: F401
except ImportError:
    pass

from abtem.inelastic.core_loss import (
    EnergyIntegral,
    SubshellTransitions,
    TransitionPotential,
    _asymptotic_amplitude,
    _continuum_radial_grid,
)

requires_gpaw = pytest.mark.skipif(
    "gpaw" not in sys.modules, reason="requires gpaw"
)


class TestEnergyIntegral:
    def test_nodes_lie_inside_the_interval(self):
        integral = EnergyIntegral(stop=4000.0, start=1.0, num=16)
        assert np.all(integral.energies > integral.start)
        assert np.all(integral.energies < integral.stop)

    def test_len_is_the_node_count(self):
        assert len(EnergyIntegral(stop=100.0, num=7)) == 7
        assert EnergyIntegral(stop=100.0, num=7).energies.shape == (7,)

    def test_weights_sum_to_the_interval_width(self):
        # Integrating f = 1 must give the width of the interval.
        integral = EnergyIntegral(stop=4000.0, start=1.0, num=16)
        assert integral.weights.sum() == pytest.approx(3999.0, rel=1e-12)

    @pytest.mark.parametrize(
        "f, exact",
        [
            (lambda e: np.ones_like(e), lambda a, b: b - a),
            (lambda e: 1 / e, lambda a, b: np.log(b / a)),
            (lambda e: e**-1.2, lambda a, b: (b**-0.2 - a**-0.2) / -0.2),
            (lambda e: e**-3.0, lambda a, b: (b**-2.0 - a**-2.0) / -2.0),
        ],
    )
    def test_quadrature_is_exact_for_power_laws(self, f, exact):
        # The integrand of an ionisation edge is flat near threshold and then a
        # slow power law, which is why the nodes are placed in log(epsilon).
        a, b = 1.0, 4000.0
        integral = EnergyIntegral(stop=b, start=a, num=16)
        got = float((integral.weights * f(integral.energies)).sum())
        assert got == pytest.approx(exact(a, b), rel=1e-8)

    def test_more_nodes_do_not_change_a_converged_integral(self):
        coarse = EnergyIntegral(stop=4000.0, num=8)
        fine = EnergyIntegral(stop=4000.0, num=32)
        f = lambda e: e**-1.2  # noqa: E731
        assert float((coarse.weights * f(coarse.energies)).sum()) == pytest.approx(
            float((fine.weights * f(fine.energies)).sum()), rel=1e-6
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"stop": 100.0, "start": 100.0},
            {"stop": 100.0, "start": 200.0},
            {"stop": 100.0, "start": 0.0},
            {"stop": 100.0, "start": -1.0},
            {"stop": 100.0, "num": 0},
        ],
    )
    def test_invalid_parameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            EnergyIntegral(**kwargs)

    def test_copy_preserves_parameters(self):
        integral = EnergyIntegral(stop=4000.0, start=2.0, num=9)
        clone = integral.copy()
        assert (clone.start, clone.stop, clone.num) == (2.0, 4000.0, 9)


class TestSubshellTransitionsEnergyAxis:
    """Bookkeeping only -- no continuum states are solved for here."""

    def test_single_energy_is_unweighted(self):
        transitions = SubshellTransitions(14, 1, 0, epsilon=25.0)
        assert transitions.continuum_energies == pytest.approx([25.0])
        assert transitions.continuum_weights == pytest.approx([1.0])
        assert not transitions.energy_integrated

    def test_energy_integral_exposes_its_nodes(self):
        integral = EnergyIntegral(stop=1000.0, num=8)
        transitions = SubshellTransitions(14, 1, 0, epsilon=integral)
        assert transitions.continuum_energies == pytest.approx(integral.energies)
        assert transitions.continuum_weights == pytest.approx(integral.weights)
        assert transitions.energy_integrated

    @pytest.mark.parametrize("n, l", [(1, 0), (2, 1), (3, 2)])
    def test_transition_count_scales_with_the_number_of_nodes(self, n, l):
        single = SubshellTransitions(29, n, l, epsilon=25.0)
        integrated = SubshellTransitions(
            29, n, l, epsilon=EnergyIntegral(stop=1000.0, num=8)
        )
        assert len(integrated) == 8 * len(single)

    @pytest.mark.parametrize("n, l", [(1, 0), (2, 1), (3, 2)])
    def test_weights_align_with_the_quantum_numbers(self, n, l):
        integral = EnergyIntegral(stop=1000.0, num=5)
        transitions = SubshellTransitions(29, n, l, epsilon=integral)

        weights = transitions.get_transition_weights()
        assert weights.shape == (len(transitions),)

        # Every transition of a given l' appears once per node, so each weight
        # occurs exactly (2l+1) * (2l'+1) times summed over l'.
        multiplicity = (2 * l + 1) * sum(2 * lp + 1 for lp in transitions.lprimes)
        for weight in integral.weights:
            assert np.isclose(weights, weight).sum() == multiplicity

    def test_single_energy_weights_are_all_one(self):
        transitions = SubshellTransitions(14, 1, 0, epsilon=25.0)
        weights = transitions.get_transition_weights()
        assert weights.shape == (len(transitions),)
        assert np.all(weights == 1.0)


class TestTransitionPotentialWeights:
    @staticmethod
    def _fake_transitions(n):
        class _Excited:
            l = 1
            ml = 0
            energy = 25.0
            quantum_numbers = (None, 1, 0)

        class _Bound:
            l = 0
            ml = 0
            energy = -1839.0
            quantum_numbers = (1, 0, 0)

        return [(_Bound(), _Excited()) for _ in range(n)]

    def test_weights_default_to_one(self):
        potential = TransitionPotential(14, self._fake_transitions(3))
        assert potential.weights == pytest.approx(np.ones(3))

    def test_mismatched_weights_raise(self):
        with pytest.raises(ValueError, match="one weight per transition"):
            TransitionPotential(14, self._fake_transitions(3), weights=[1.0, 2.0])


@requires_gpaw
class TestContinuumNormalisation:
    """The continuum state must be energy-normalised: u -> sin(kr + d)/sqrt(pi k)."""

    @staticmethod
    def _amplitude(wavefunction, epsilon):
        from ase import units

        r = wavefunction.radial_grid
        u = wavefunction._radial_values
        k = np.sqrt(epsilon / units.Rydberg)

        outer = r > 0.75 * r[-1]
        du = np.gradient(u, r)
        return float(np.median(np.sqrt(u[outer] ** 2 + (du[outer] / k) ** 2)))

    @pytest.mark.parametrize("epsilon", [1.0, 25.0, 400.0])
    @pytest.mark.parametrize("lprime", [0, 1, 2, 3])
    def test_asymptotic_amplitude_is_one_over_sqrt_pi_k(self, epsilon, lprime):
        from ase import units

        from abtem.inelastic.core_loss import (
            calculate_continuum_radial_wavefunction,
        )

        wavefunction = calculate_continuum_radial_wavefunction(
            Z=14, n=1, l=0, lprime=lprime, epsilon=epsilon
        )
        k = np.sqrt(epsilon / units.Rydberg)
        amplitude = self._amplitude(wavefunction, epsilon)

        assert amplitude * np.sqrt(np.pi * k) == pytest.approx(1.0, rel=1e-3)


class TestContinuumGrid:
    def test_grid_grows_at_low_energy(self):
        from ase import units

        low = _continuum_radial_grid(1.0 / units.Rydberg, lprime=0)
        high = _continuum_radial_grid(400.0 / units.Rydberg, lprime=0)
        assert low[-1] > high[-1]

    def test_grid_grows_with_angular_momentum(self):
        from ase import units

        ef = 25.0 / units.Rydberg
        assert _continuum_radial_grid(ef, lprime=3)[-1] >= _continuum_radial_grid(
            ef, lprime=0
        )[-1]

    def test_grid_is_capped_and_warns(self):
        from ase import units

        # Far below threshold the asymptotic region is unreachable.
        with pytest.warns(RuntimeWarning, match="asymptotic form"):
            grid = _continuum_radial_grid(1e-4 / units.Rydberg, lprime=3)
        assert grid[-1] <= 150.0

    def test_grid_starts_near_the_origin(self):
        from ase import units

        grid = _continuum_radial_grid(25.0 / units.Rydberg, lprime=1)
        assert 0.0 < grid[0] < 1e-6


class TestAsymptoticAmplitude:
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
        # Centre the excursion on a crest of the sine so max(u) really is large.
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


class TestRadialWavefunctionBound:
    """``bound`` used to compare n to 0, which raised for continuum states."""

    @staticmethod
    def _wavefunction(n, energy):
        from abtem.inelastic.core_loss import RadialWavefunction

        return RadialWavefunction(
            n=n,
            l=1,
            energy=energy,
            radial_grid=np.linspace(1e-9, 1.0, 10),
            radial_values=np.zeros(10),
        )

    def test_continuum_state_is_not_bound(self):
        assert self._wavefunction(n=None, energy=25.0).bound is False

    def test_bound_state_is_bound(self):
        assert self._wavefunction(n=2, energy=-1839.0).bound is True

    def test_atomic_wavefunction_delegates(self):
        from abtem.inelastic.core_loss import AtomicWaveFunction

        continuum = AtomicWaveFunction(self._wavefunction(None, 25.0), ml=0)
        assert continuum.bound is False


class TestPrecisionConfig:
    """The transition potential must follow ``config['precision']``."""

    @staticmethod
    def _fake_transitions():
        return TestTransitionPotentialWeights._fake_transitions(2)

    @pytest.mark.parametrize(
        "precision, expected",
        [("float32", np.complex64), ("float64", np.complex128)],
    )
    def test_build_allocates_at_the_configured_precision(self, precision, expected):
        import abtem
        from abtem.core.utils import get_dtype

        with abtem.config.set({"precision": precision}):
            assert get_dtype(complex=True) is expected

    @requires_gpaw
    @pytest.mark.parametrize(
        "precision, expected",
        [("float32", np.complex64), ("float64", np.complex128)],
    )
    def test_built_array_honours_precision(self, precision, expected):
        import abtem

        with abtem.config.set({"precision": precision}):
            transitions = SubshellTransitions(14, 1, 0, epsilon=25.0)
            potential = transitions.get_transition_potentials(
                extent=6.0, gpts=64, energy=100e3
            ).build()
            assert potential.array.dtype == expected


def test_dead_set_threshold_is_gone():
    # It computed two values, discarded both and returned None.
    from abtem.inelastic.core_loss import TransitionPotentialArray

    assert not hasattr(TransitionPotentialArray, "set_threshold")


@requires_gpaw
class TestKinematicLimit:
    """The energy loss cannot exceed the beam energy."""

    def test_excessive_epsilon_range_raises_clearly(self):
        # Cu K threshold is ~8.8 keV; stop = 60 keV puts the loss past a 30 keV beam.
        transitions = SubshellTransitions(
            29, 1, 0, epsilon=EnergyIntegral(stop=60000.0, num=4)
        )
        with pytest.raises(ValueError, match="exceeds the beam energy"):
            transitions.get_transition_potentials(
                extent=6.0, gpts=64, energy=30e3
            ).build()

    def test_message_names_the_threshold_and_the_remedy(self):
        transitions = SubshellTransitions(
            29, 1, 0, epsilon=EnergyIntegral(stop=60000.0, num=4)
        )
        with pytest.raises(ValueError) as excinfo:
            transitions.get_transition_potentials(
                extent=6.0, gpts=64, energy=30e3
            ).build()
        message = str(excinfo.value)
        assert "ionisation threshold" in message
        assert "EnergyIntegral" in message

    def test_a_workable_range_still_builds(self):
        transitions = SubshellTransitions(
            29, 1, 0, epsilon=EnergyIntegral(stop=20000.0, num=4)
        )
        potential = transitions.get_transition_potentials(
            extent=6.0, gpts=64, energy=200e3
        ).build()
        assert np.isfinite(potential.array).all()


@requires_gpaw
class TestDelocalisationWarning:
    """A cell smaller than a few delocalisation lengths truncates the potential."""

    @staticmethod
    def _build(extent, energy, Z=6):
        return SubshellTransitions(Z, 1, 0, epsilon=25.0).get_transition_potentials(
            extent=extent, gpts=128, energy=energy
        ).build()

    def test_small_cell_warns(self):
        with pytest.warns(RuntimeWarning, match="delocalisation"):
            self._build(extent=8.0, energy=300e3)

    def test_large_cell_is_quiet(self):
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            self._build(extent=32.0, energy=300e3)
        assert not [w for w in caught if "delocalisation" in str(w.message)]

    def test_a_tightly_bound_edge_needs_a_smaller_cell(self):
        # Delocalisation scales as 1/dE, so Cu K is far more localised than C K
        # at the same beam energy and does not warn in a cell that C would.
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            self._build(extent=8.0, energy=300e3, Z=29)
        assert not [w for w in caught if "delocalisation" in str(w.message)]
