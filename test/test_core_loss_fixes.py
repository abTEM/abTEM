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
from abtem.array import ArrayObject
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

    @staticmethod
    def _atoms():
        return ase.Atoms(
            "Si2", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)],
            cell=(8, 8, 8), pbc=True,
        )

    def _s_matrix(self, gpts):
        potential = abtem.Potential(self._atoms(), gpts=gpts, slice_thickness=4.0)
        return abtem.SMatrix(
            potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=1
        )

    def _scan(self, s_matrix, transition_potentials, **kwargs):
        return s_matrix.transition_potential_scan(
            transition_potentials=transition_potentials,
            scan=abtem.GridScan(start=(0, 0), end=(1, 1), gpts=(2, 2), fractional=True,
                                potential=s_matrix.potential),
            detectors=abtem.FlexibleAnnularDetector(),
            sites=self._atoms(), lazy=False, **kwargs,
        )

    def test_a_built_potential_on_the_wrong_grid_is_refused(self):
        s_matrix = self._s_matrix((64, 64))
        mismatched = _synthetic_transition_potential((8.0, 8.0), (32, 32), n=2)
        with pytest.raises(RuntimeError, match="Inconsistent grid"):
            self._scan(s_matrix, mismatched)

    def test_a_built_potential_on_the_right_grid_still_runs(self):
        """The guard must not fire on the case it is meant to allow."""
        s_matrix = self._s_matrix((64, 64))
        matched = _synthetic_transition_potential((8.0, 8.0), (64, 64), n=2)
        measurement = self._scan(s_matrix, matched)
        assert measurement.shape[:2] == (2, 2)

    def test_a_mismatched_extent_is_refused_too(self):
        """check_match compares extent as well as gpts, and an extent mismatch
        is the same class of error -- the array cannot be re-gridded either."""
        s_matrix = self._s_matrix((64, 64))
        wrong_extent = _synthetic_transition_potential((6.0, 6.0), (64, 64), n=2)
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
        wrong_energy = _synthetic_transition_potential((8.0, 8.0), (64, 64), n=2)
        wrong_energy.accelerator._energy = 2 * ENERGY
        with pytest.raises(RuntimeError, match="Inconsistent energies"):
            self._scan(s_matrix, wrong_energy)

    def test_an_unbuilt_potential_is_still_regridded(self):
        """Regression guard for the preceding commit: an unbuilt potential has
        no array, so a grid mismatch is not an error -- it is matched and then
        built at the S-matrix's gpts."""
        pytest.importorskip("gpaw")
        from abtem.inelastic.core_loss import SubshellTransitions, TransitionPotential

        s_matrix = self._s_matrix((64, 64))
        transitions = SubshellTransitions(Z=14, n=1, l=0, xc="PBE").get_transitions()
        unbuilt = TransitionPotential(
            14, transitions, extent=8.0, gpts=32, energy=ENERGY, double_channel=False
        )
        measurement = self._scan(s_matrix, unbuilt)
        assert measurement.shape[:2] == (2, 2)


class TestMultisliceBuiltTransitionPotentialEnergy:
    """The sibling of TestPrismEelsBuiltTransitionPotentialGrid's energy
    check above, for the plain multislice EELS driver
    (transition_potential_multislice_and_detect, abtem/multislice.py) rather
    than the PRISM-EELS one. Same guard, same reason, same place: before
    _task_local's match, which would otherwise silently overwrite a built
    transition potential's energy to agree with the waves while its baked-in
    form factors stay computed at the old one.
    """

    @staticmethod
    def _atoms():
        return ase.Atoms(
            "Si2", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)],
            cell=(8, 8, 8), pbc=True,
        )

    def test_a_mismatched_energy_is_refused_too(self):
        atoms = self._atoms()
        potential = abtem.Potential(atoms, gpts=(64, 64), slice_thickness=4.0)
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
        probe.grid.match(potential)

        wrong_energy = _synthetic_transition_potential((8.0, 8.0), (64, 64), n=2)
        wrong_energy.accelerator._energy = 2 * ENERGY

        with pytest.raises(RuntimeError, match="Inconsistent energies"):
            probe.transition_potential_scan(
                potential=potential, transition_potentials=wrong_energy,
                scan=abtem.GridScan(start=(0, 0), end=(1, 1), gpts=(2, 2),
                                    fractional=True, potential=potential),
                detectors=abtem.FlexibleAnnularDetector(), sites=atoms,
                double_channel=False, lazy=False,
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

    @staticmethod
    def _atoms():
        return ase.Atoms(
            "Si2", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)],
            cell=(8, 8, 8), pbc=True,
        )

    def _potential(self, ensemble):
        atoms = self._atoms()
        if ensemble:
            atoms = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.0, seed=1)
        return abtem.Potential(atoms, gpts=(32, 32), slice_thickness=4.0)

    def _prism(self, scan, ensemble, lazy):
        potential = self._potential(ensemble)
        s_matrix = abtem.SMatrix(
            potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=1
        )
        return s_matrix.transition_potential_scan(
            transition_potentials=_synthetic_transition_potential(
                potential.extent, potential.gpts, n=2
            ),
            scan=scan, detectors=abtem.FlexibleAnnularDetector(),
            sites=self._atoms(), lazy=lazy,
        )

    def _multislice(self, scan, ensemble, lazy):
        potential = self._potential(ensemble)
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
        probe.grid.match(potential)
        return probe.transition_potential_scan(
            scan=scan, potential=potential,
            transition_potentials=_synthetic_transition_potential(
                potential.extent, potential.gpts, n=2
            ),
            detectors=abtem.FlexibleAnnularDetector(),
            sites=self._atoms(), lazy=lazy,
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
    @pytest.mark.parametrize(
        "scan",
        [(1.0, 1.0), [(1.0, 1.0)], np.array([[1.0, 1.0]])],
        ids=["tuple", "list", "array"],
    )
    def test_a_single_position_scan_matches_the_multislice_oracle(
        self, scan, ensemble, lazy
    ):
        """Every single-position non-BaseScan form, both ensembles, both modes.

        ``ArrayObject.squeeze`` only drops length-1 axes, so all three of these
        forms squeeze and a multi-position list does not -- which is why the
        defect needed exactly one position to show.
        """
        got = self._axes(self._prism(scan, ensemble, lazy))
        expected = self._axes(self._multislice(scan, ensemble, lazy))
        assert got == expected

    @pytest.mark.parametrize("ensemble", [False, True])
    def test_the_rank_does_not_depend_on_laziness(self, ensemble):
        """The sharpest form of the bug: eager and lazy disagreed with each
        other, so the same call returned different ranks depending only on how
        it was scheduled."""
        eager = self._prism((1.0, 1.0), ensemble, lazy=False)
        lazy = self._prism((1.0, 1.0), ensemble, lazy=True)
        assert self._axes(eager) == self._axes(lazy)
        assert eager.shape == lazy.shape

    def test_the_rank_does_not_depend_on_the_potential_being_an_ensemble(self):
        """With one exit plane the mechanism did not crash -- it returned a
        result whose rank depended on whether the potential was a
        ``FrozenPhonons`` ensemble, which is the silent half of the defect."""
        plain = self._prism((1.0, 1.0), False, lazy=False)
        ensemble = self._prism((1.0, 1.0), True, lazy=False)
        assert self._axes(plain) == self._axes(ensemble)
        assert plain.shape == ensemble.shape

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


@pytest.mark.parametrize("lazy", [False, True])
def test_detectors_elastic_is_refused_rather_than_ignored(lazy):
    """It was declared in the signature and never read.

    Elastic detectors passed through Probe.transition_potential_scan were
    accepted in silence and only the inelastic measurement came back. Every
    other unsupported keyword reaching the driver through
    **multislice_func_kwargs raises TypeError; this one spelling quietly
    absorbed the caller's intent.

    Parametrised over ``lazy`` because abTEM is lazy by default: a check
    inside the per-chunk worker would let the caller build a whole measurement
    object without complaint and only fail later, inside a dask traceback.
    """
    atoms, potential, probe = _detectors_elastic_setup()

    with pytest.raises(NotImplementedError, match="detectors_elastic"):
        probe.transition_potential_scan(
            potential=potential,
            transition_potentials=_synthetic_transition_potential(
                potential.extent, potential.gpts
            ),
            scan=np.array([[0.0, 0.0]]),
            detectors=abtem.FlexibleAnnularDetector(),
            detectors_elastic=[abtem.AnnularDetector(inner=50, outer=150)],
            double_channel=False,
            lazy=lazy,
            sites=atoms,
        )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "kwargs", [{}, {"detectors_elastic": None}, {"detectors_elastic": []}],
    ids=["omitted", "none", "empty"],
)
def test_detectors_elastic_guard_has_no_false_positives(lazy, kwargs):
    """An empty list asks for no elastic detectors, so nothing is dropped.

    Raising on it would turn a working call into a crash for no benefit, and
    callers that forward an explicit ``None`` must keep working.
    """
    atoms, potential, probe = _detectors_elastic_setup()

    measurement = probe.transition_potential_scan(
        potential=potential,
        transition_potentials=_synthetic_transition_potential(
            potential.extent, potential.gpts
        ),
        scan=np.array([[0.0, 0.0]]),
        detectors=abtem.FlexibleAnnularDetector(),
        double_channel=False,
        lazy=lazy,
        sites=atoms,
        **kwargs,
    )
    if lazy:
        measurement = measurement.compute(progress_bar=False)
    assert np.asarray(abtem.core.backend.asnumpy(measurement.array)).size


def _detectors_elastic_setup():
    atoms = ase.build.bulk("Si", cubic=True)
    potential = abtem.Potential(atoms, gpts=(32, 32), slice_thickness=1.4)
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
    probe.grid.match(potential)
    return atoms, potential, probe


def test_transition_potentials_are_wrapped_by_the_live_isinstance_check():
    """The deleted duck-typed validator was not an equivalent spelling.

    It wrapped on ``hasattr(x, "scatter")``; the live check in
    Waves.transition_potential_multislice is ``isinstance(x, (list, tuple))``.
    They diverge on a generator, so asserting only that the dead one is gone
    would not stop it being "restored" later as a synonym.
    """
    import abtem.inelastic.core_loss as core_loss

    assert not hasattr(core_loss, "_validate_transition_potentials")

    atoms, potential, probe = _detectors_elastic_setup()
    single = _synthetic_transition_potential(potential.extent, potential.gpts)

    from_single = probe.transition_potential_scan(
        potential=potential, transition_potentials=single,
        scan=np.array([[0.0, 0.0]]), detectors=abtem.FlexibleAnnularDetector(),
        double_channel=False, lazy=False, sites=atoms,
    )
    from_list = probe.transition_potential_scan(
        potential=potential, transition_potentials=[single],
        scan=np.array([[0.0, 0.0]]), detectors=abtem.FlexibleAnnularDetector(),
        double_channel=False, lazy=False, sites=atoms,
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
            transition_potentials=_synthetic_transition_potential(
                potential.extent, potential.gpts, n=2
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
    measurement constructor and masked the zeros above.
    """

    @staticmethod
    def _setup(num_configs=None, exit_planes=None):
        atoms = ase.Atoms(
            "Si2",
            positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)],
            cell=(8, 8, 8),
            pbc=True,
        )
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
            transition_potentials=_synthetic_transition_potential(
                potential.extent, potential.gpts, n=2
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

    def test_exit_plane_axis_metadata_has_one_value_per_exit_plane(self):
        """It was built from the per-slice ThicknessAxis instead."""
        _, potential = self._setup(num_configs=2, exit_planes=1)
        s_matrix = abtem.SMatrix(
            potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=1
        )
        shape, metadata = s_matrix._build_ensemble_shape_metadata()
        assert shape[-1] == len(potential.exit_planes)
        assert len(metadata[-1].values) == len(potential.exit_planes)


def test_ensemble_indices_handle_a_multi_axis_potential_ensemble():
    """A potential with two ensemble axes is not constructible through the
    public API today, so exercise the index builder directly."""
    from abtem.multislice import _validate_potential_ensemble_indices

    class _FakePotential:
        ensemble_shape = (2, 3)
        exit_planes = (-1, 0, 1)

    potential = _FakePotential()
    indices = _validate_potential_ensemble_indices((1, 2), slice(1, 3), potential)
    assert indices == (1, 2, slice(1, 3))
    assert sum(isinstance(i, slice) for i in indices) == 1

    class _SinglePlane(_FakePotential):
        exit_planes = (-1,)

    assert _validate_potential_ensemble_indices((1, 2), slice(0, 1), _SinglePlane()) == (
        1,
        2,
    )


def test_ensemble_index_length_is_enforced():
    """Every defect in this family was a caller passing too few indices.

    The helper used to accept a short tuple silently, which let the exit-plane
    part land on an ensemble axis.
    """
    from abtem.multislice import _validate_potential_ensemble_indices

    class _Potential:
        ensemble_shape = (2, 3)
        exit_planes = (-1, 0, 1)

    potential = _Potential()
    assert _validate_potential_ensemble_indices((1, 2), slice(1, 3), potential) == (
        1,
        2,
        slice(1, 3),
    )
    for short in ((1,), ()):
        with pytest.raises(ValueError, match="entries for an ensemble"):
            _validate_potential_ensemble_indices(short, slice(1, 3), potential)


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
            transition_potentials=_synthetic_transition_potential(
                potential.extent, potential.gpts, n=2
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
    def _setup(n_rows, n_cols):
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
            potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=1
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
            transition_potentials=_synthetic_transition_potential(
                s_matrix.potential.extent, s_matrix.potential.gpts, n=2
            ),
            scan=scan,
            detectors=abtem.FlexibleAnnularDetector(),
            sites=atoms,
            double_channel=double_channel,
            lazy=False,
        )
        return np.asarray(abtem.core.backend.asnumpy(measurement.array))

    @pytest.mark.parametrize("double_channel", [False, True])
    # With this test's n_T=2 and 5 columns, these forced "position" budgets
    # resolve (guess, then verified against the actual crop box) to row
    # batches of 1, 2 and 4 respectively -- checked directly by recording
    # minimum_crop's call sizes for each value. 7 rows is not a multiple of
    # 2 or 4, so two of the three exercise an uneven last batch.
    @pytest.mark.parametrize("forced_budget", [1, 25, 40])
    def test_chunked_reduction_matches_a_single_whole_scan_batch(
        self, monkeypatch, double_channel, forced_budget
    ):
        atoms, _, s_matrix, scan = self._setup(n_rows=7, n_cols=5)

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
        atoms, _, s_matrix, scan = self._setup(n_rows=n_rows, n_cols=n_cols)
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

        A scan physically wider than the ~8 A window (unlike this class's
        other tests, whose scans are drawn in the same few Angstrom as the
        window and never exercise real box growth) with a budget picked to
        land the naive guess exactly at the row count reproduces this
        directly: found by running the fix's own benchmark script against a
        real GPU, where interpolation=1 (a large, undownsampled window) hit
        it immediately.
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
            potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=1
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
                interpolation=1,
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

    @staticmethod
    def _atoms():
        return ase.Atoms(
            "Si2",
            positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)],
            cell=(8, 8, 8),
            pbc=True,
        )

    def _run(self, potential, lazy, double_channel=False):
        atoms = self._atoms()
        s_matrix = abtem.SMatrix(
            potential=potential, energy=ENERGY, semiangle_cutoff=20, interpolation=1
        )
        scan = abtem.GridScan(
            start=(0, 0), end=(1, 1), gpts=(2, 2), fractional=True, potential=potential
        )
        measurement = s_matrix.transition_potential_scan(
            transition_potentials=_synthetic_transition_potential(
                potential.extent, potential.gpts, n=2
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
        atoms = self._atoms()
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

    @pytest.mark.parametrize(
        "num_configs, exit_planes, ensemble_mean",
        [
            (None, None, True),
            (None, 1, True),
            (1, 1, True),
            (3, 1, True),
            (3, 1, False),
            (None, [1, 3], True),
        ],
        ids=["plain-1", "plain-5", "phonons1-5", "phonons3-5", "no-mean", "explicit"],
    )
    @pytest.mark.parametrize("double_channel", [False, True])
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
