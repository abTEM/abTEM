"""Regression tests for correctness fixes in the core-loss machinery.

Each test here corresponds to a defect that was present and is now fixed. They
are grouped by the object they belong to rather than by symptom.
"""

from __future__ import annotations

import sys

import ase
import numpy as np
import pytest

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import (
    AtomicWaveFunction,
    RadialWavefunction,
    TransitionPotentialArray,
    _asymptotic_amplitude,
    _continuum_radial_grid,
)

try:
    import gpaw  # noqa: F401
except ImportError:
    pass

from utils import gpu  # noqa: E402

requires_gpaw = pytest.mark.skipif(
    "gpaw" not in sys.modules, reason="requires gpaw"
)

ENERGY = 100e3


def _synthetic_transition_potential(extent, gpts, device="cpu", n=3, seed=0):
    try:
        import cupy as cp
    except ImportError:
        cp = None

    xp = cp if device == "gpu" else np
    rng = np.random.default_rng(seed)
    array = (
        rng.standard_normal((n, *gpts)) + 1j * rng.standard_normal((n, *gpts))
    ).astype(np.complex64)
    return TransitionPotentialArray(
        Z=14,
        array=xp.asarray(array),
        energy=ENERGY,
        extent=extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n)))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )


class TestRadialWavefunctionBound:
    """``bound`` compared n to 0, which raises for a continuum state."""

    @staticmethod
    def _wavefunction(n, energy):
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
        continuum = AtomicWaveFunction(self._wavefunction(None, 25.0), ml=0)
        assert continuum.bound is False


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
class TestContinuumNormalisation:
    """The continuum state must be energy-normalised: u -> sin(kr+d)/sqrt(pi k)."""

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
        r = wavefunction.radial_grid
        u = wavefunction._radial_values
        k = np.sqrt(epsilon / units.Rydberg)

        outer = r > 0.75 * r[-1]
        du = np.gradient(u, r)
        amplitude = float(np.median(np.sqrt(u[outer] ** 2 + (du[outer] / k) ** 2)))

        assert amplitude * np.sqrt(np.pi * k) == pytest.approx(1.0, rel=1e-3)


class TestPrecisionConfig:
    """The transition potential ignored ``config['precision']`` twice over.

    It was allocated complex64, and then the closing division by a float64
    numpy scalar promoted the whole array back to complex128 under NEP 50 --
    so every transition potential was silently double precision at twice the
    memory, whatever the configuration said.
    """

    @requires_gpaw
    @pytest.mark.parametrize(
        "precision, expected",
        [("float32", np.complex64), ("float64", np.complex128)],
    )
    def test_built_array_honours_precision(self, precision, expected):
        from abtem.inelastic.core_loss import SubshellTransitions

        with abtem.config.set({"precision": precision}):
            potential = SubshellTransitions(14, 1, 0, epsilon=25.0)
            built = potential.get_transition_potentials(
                extent=6.0, gpts=64, energy=ENERGY
            ).build()
            assert built.array.dtype == expected

    @requires_gpaw
    def test_single_and_double_precision_agree(self):
        from abtem.inelastic.core_loss import SubshellTransitions

        values = {}
        for precision in ("float32", "float64"):
            with abtem.config.set({"precision": precision}):
                built = SubshellTransitions(
                    14, 1, 0, epsilon=25.0
                ).get_transition_potentials(
                    extent=10.0, gpts=128, energy=ENERGY
                ).build()
                values[precision] = float(
                    np.abs(built.array).sum(dtype=np.float64)
                )

        assert values["float32"] == pytest.approx(values["float64"], rel=1e-5)

    def test_no_hardcoded_dtypes_remain(self):
        import re
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


def test_dead_set_threshold_is_gone():
    # It computed two values, discarded both and returned None, and was never
    # called from anywhere in abTEM or the tests.
    assert not hasattr(TransitionPotentialArray, "set_threshold")


@pytest.mark.parametrize("device", ["cpu", gpu])
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
        transition_potentials=_synthetic_transition_potential(
            potential.extent, potential.gpts, device=device
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
            transition_potentials = _synthetic_transition_potential(
                potential.extent, potential.gpts
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
            _synthetic_transition_potential(
                potential.extent, potential.gpts, seed=seed
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


@requires_gpaw
class TestFilterByIntensity:
    """It sorted by intensity, then sliced the *unsorted* list."""

    def test_keeps_the_strongest_transitions(self):
        from abtem.inelastic.core_loss import SubshellTransitions

        potential = SubshellTransitions(14, 1, 0, epsilon=25.0).get_transition_potentials(
            extent=8.0, gpts=64, energy=ENERGY
        )

        intensities = potential.integrated_intensities()
        order = np.argsort(-intensities)

        filtered = potential.filter_by_intensity(0.5)
        kept = {id(t) for t in filtered.transitions}

        # Everything kept must be at least as strong as everything dropped.
        strongest = [potential.transitions[i] for i in order]
        kept_ranks = [i for i, t in enumerate(strongest) if id(t) in kept]
        assert kept_ranks == list(range(len(kept_ranks)))
