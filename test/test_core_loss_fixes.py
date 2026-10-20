"""Regression tests for correctness fixes in the core-loss machinery.

Each test here corresponds to a defect that was present and is now fixed. They
are grouped by the object they belong to rather than by symptom.
"""

from __future__ import annotations

import io
import sys

import ase
import numpy as np
import pytest

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import (
    AtomicWaveFunction,
    RadialWavefunction,
    TransitionPotential,
    TransitionPotentialArray,
)

try:
    import gpaw  # noqa: F401
except ImportError:
    pass

from utils import gpu  # noqa: E402

requires_gpaw = pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")

# Building a transition potential evaluates spherical harmonics (with
# SciPy < 1.17 through the deprecated `sph_harm`) and Wigner 3j symbols (which
# trigger a deprecation warning in some mpmath versions).
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:`scipy.special.sph_harm` is deprecated:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:bitcount function is deprecated:DeprecationWarning"
    ),
]

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


def _synthetic_transition_potential_from_radial_functions():
    # A hydrogen-like 1s bound state and a free p-wave, so that the transition
    # potential can be built without a DFT code.
    r = np.linspace(1e-12, 20.0, 20000)
    k = 1.3
    bound = RadialWavefunction(
        n=1, l=0, energy=-1839.0, radial_grid=r, radial_values=2 * r * np.exp(-r)
    )
    continuum = RadialWavefunction(
        n=None,
        l=1,
        energy=25.0,
        radial_grid=r,
        radial_values=np.sin(k * r) / np.sqrt(np.pi * k),
    )
    transitions = [
        (AtomicWaveFunction(bound, 0), AtomicWaveFunction(continuum, ml))
        for ml in (-1, 0, 1)
    ]
    return TransitionPotential(14, transitions, extent=6.0, gpts=32, energy=ENERGY)


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
        from abtem.inelastic.core_loss import _asymptotic_amplitude

        k = 1.3
        r = np.linspace(1e-12, 40.0, 200000)
        for amplitude in [0.5, 1.0, 7.25]:
            u = amplitude * np.sin(k * r + 0.4)
            assert _asymptotic_amplitude(r, u, k) == pytest.approx(amplitude, rel=1e-4)

    def test_ignores_a_larger_transient_inside(self):
        from abtem.inelastic.core_loss import _asymptotic_amplitude

        # A big inner excursion, as produced near a centrifugal turning point,
        # must not be mistaken for the asymptotic amplitude.
        k = 1.3
        r = np.linspace(1e-12, 40.0, 200000)
        crest = (np.pi / 2 + 2 * np.pi) / k
        u = np.sin(k * r) * (1.0 + 5.0 * np.exp(-((r - crest) ** 2)))
        assert _asymptotic_amplitude(r, u, k) == pytest.approx(1.0, rel=1e-3)
        assert u.max() > 3.0  # max(u) would have been badly wrong

    def test_warns_when_the_envelope_is_not_flat(self):
        from abtem.inelastic.core_loss import _asymptotic_amplitude

        k = 1.3
        r = np.linspace(1e-12, 40.0, 200000)
        u = np.sin(k * r) * r  # envelope still growing
        with pytest.warns(RuntimeWarning, match="free-particle"):
            _asymptotic_amplitude(r, u, k)

    def test_vanishing_wavefunction_raises(self):
        from abtem.inelastic.core_loss import _asymptotic_amplitude

        r = np.linspace(1e-12, 40.0, 1000)
        with pytest.raises(RuntimeError, match="vanishes"):
            _asymptotic_amplitude(r, np.zeros_like(r), 1.3)


class TestContinuumGrid:
    """A fixed 20 Bohr grid cannot resolve the asymptotic region at low energy."""

    def test_grid_grows_at_low_energy(self):
        from ase import units

        from abtem.inelastic.core_loss import _continuum_radial_grid

        low = _continuum_radial_grid(1.0 / units.Rydberg, lprime=0)
        high = _continuum_radial_grid(400.0 / units.Rydberg, lprime=0)
        assert low[-1] > high[-1]

    def test_grid_grows_with_angular_momentum(self):
        from ase import units

        from abtem.inelastic.core_loss import _continuum_radial_grid

        ef = 25.0 / units.Rydberg
        assert (
            _continuum_radial_grid(ef, lprime=3)[-1]
            >= _continuum_radial_grid(ef, lprime=0)[-1]
        )

    def test_high_energy_keeps_the_original_grid(self):
        from ase import units

        from abtem.inelastic.core_loss import _continuum_radial_grid

        grid = _continuum_radial_grid(400.0 / units.Rydberg, lprime=0)
        assert grid[-1] == pytest.approx(20.0)

    def test_grid_is_capped_and_warns(self):
        from ase import units

        from abtem.inelastic.core_loss import _continuum_radial_grid

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
class TestContinuumNormalisation:
    """The continuum state must be energy-normalised: u -> sin(kr+d)/sqrt(pi k)."""

    @pytest.mark.parametrize("epsilon", [1.0, 25.0, 400.0])
    @pytest.mark.parametrize("lprime", [0, 1, 2, 3])
    def test_asymptotic_amplitude_is_one_over_sqrt_pi_k(self, epsilon, lprime):
        from ase import units

        from abtem.inelastic.core_loss import calculate_continuum_radial_wavefunction

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


@requires_gpaw
class TestContinuumPotentialVacuumTail:
    """The atomic potential the continuum states are solved in must vanish in vacuum.

    GPAW's xc kernels clamp the density from below, which leaves vxc at a
    constant -8.5e-4 Hartree in the vacuum tail. r*V then grew linearly to the
    edge of the radial grid, which shifted the asymptotic wavenumber of the
    continuum and with it the energy normalisation below about 10 eV.
    """

    @pytest.mark.parametrize("Z", [6, 14])
    def test_neutral_atom_potential_vanishes_in_vacuum(self, Z):
        from abtem.inelastic.core_loss import _atomic_rv

        rv = _atomic_rv(Z)
        # r * V [Rydberg * Bohr] of a neutral atom goes to zero far from the
        # nucleus; with the uncorrected vxc it was 0.25 at 150 Bohr.
        assert abs(float(rv(150.0))) < 1e-3
        assert abs(float(rv(60.0))) < 1e-3


class TestPrecisionConfig:
    """The transition potential ignored ``config['precision']`` twice over.

    It was allocated complex64, and then the closing division by a float64
    numpy scalar promoted the whole array back to complex128 under NEP 50 --
    so every transition potential was silently double precision at twice the
    memory, whatever the configuration said.
    """

    @pytest.mark.parametrize(
        "precision, expected",
        [("float32", np.complex64), ("float64", np.complex128)],
    )
    def test_built_array_honours_precision(self, precision, expected):
        with abtem.config.set({"precision": precision}):
            potential = _synthetic_transition_potential_from_radial_functions()
            assert potential.build().array.dtype == expected

    @requires_gpaw
    @pytest.mark.parametrize(
        "precision, expected",
        [("float32", np.complex64), ("float64", np.complex128)],
    )
    def test_dft_built_array_honours_precision(self, precision, expected):
        from abtem.inelastic.core_loss import SubshellTransitions

        with abtem.config.set({"precision": precision}):
            potential = SubshellTransitions(14, 1, 0, epsilon=25.0)
            built = potential.get_transition_potentials(
                extent=6.0, gpts=64, energy=ENERGY
            ).build()
            assert built.array.dtype == expected

    def test_single_and_double_precision_agree(self):
        values = {}
        for precision in ("float32", "float64"):
            with abtem.config.set({"precision": precision}):
                built = _synthetic_transition_potential_from_radial_functions().build()
                values[precision] = float(np.abs(built.array).sum(dtype=np.float64))

        assert values["float32"] == pytest.approx(values["float64"], rel=1e-5)


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
            _synthetic_transition_potential(potential.extent, potential.gpts, seed=seed)
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
        potential = _synthetic_transition_potential_from_radial_functions()

        # The ml' = 0 final state is the strongest of the three, but is not
        # the first in the list.
        intensities = potential.integrated_intensities()
        assert np.argmax(intensities) == 1

        filtered = potential.filter_by_intensity(0.4)

        assert len(filtered.transitions) == 1
        assert filtered.transitions[0] is potential.transitions[1]


class TestFilterSites:
    """The survival mask was built over the element-subset array but applied to
    the caller's object."""

    def test_mask_is_aligned_with_mixed_element_atoms(self):
        potential = _synthetic_transition_potential((8.0, 8.0), (32, 32))
        waves = abtem.Probe(
            energy=ENERGY, extent=(8.0, 8.0), gpts=(32, 32), semiangle_cutoff=30
        ).build(lazy=False)
        atoms = ase.Atoms(
            "SiNSi",
            positions=[(4.0, 4.0, 0.0), (1.0, 1.0, 0.0), (2.0, 6.0, 0.0)],
            cell=(8, 8, 4),
        )

        filtered = potential.filter_sites(waves, atoms, threshold=1e-12)

        assert isinstance(filtered, np.ndarray)
        assert filtered.shape[1:] == (2,)
        silicon = atoms[atoms.numbers == 14].positions[:, :2]
        assert all(np.isclose(silicon, site).all(axis=1).any() for site in filtered)


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
        eager = self._run(
            potential, probe, sites, lazy=False, double_channel=double_channel
        )
        lazy = self._run(
            potential, probe, sites, lazy=True, double_channel=double_channel
        )
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
            single = abtem.Potential(configuration, gpts=(64, 64), slice_thickness=2.0)
            reference = self._run(single, probe, atoms, lazy=False)
            assert np.array_equal(
                got[index], reference
            ), f"slot {index} does not hold configuration {index}"


class TestMultisliceBuiltTransitionPotentialMatch:
    """A built transition potential cannot be matched to the waves.

    ``build()`` bakes the energy into the array's form factors (k0, kn, the
    relativistic mass correction, the interaction parameter) and fixes its
    shape, so the driver's ``match`` could only overwrite the grid or energy
    label without recomputing anything. A mismatched energy used to run to
    completion silently; a mismatched grid failed with an opaque broadcast
    error. Both are now refused with a clear error before the match.
    """

    @staticmethod
    def _scan(transition_potential):
        atoms = ase.Atoms(
            "Si2",
            positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)],
            cell=(8, 8, 8),
            pbc=True,
        )
        potential = abtem.Potential(atoms, gpts=(64, 64), slice_thickness=4.0)
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
        probe.grid.match(potential)
        return probe.transition_potential_scan(
            potential=potential,
            transition_potentials=transition_potential,
            scan=abtem.GridScan(
                start=(0, 0),
                end=(1, 1),
                gpts=(2, 2),
                fractional=True,
                potential=potential,
            ),
            detectors=abtem.FlexibleAnnularDetector(),
            sites=atoms,
            double_channel=False,
            lazy=False,
        )

    def test_a_mismatched_energy_is_refused(self):
        wrong_energy = _synthetic_transition_potential((8.0, 8.0), (64, 64), n=2)
        wrong_energy.accelerator._energy = 2 * ENERGY

        with pytest.raises(RuntimeError, match="Inconsistent energies"):
            self._scan(wrong_energy)

        assert wrong_energy.energy == 2 * ENERGY

    def test_a_mismatched_grid_is_refused(self):
        wrong_grid = _synthetic_transition_potential((8.0, 8.0), (32, 32), n=2)

        with pytest.raises(RuntimeError, match="Inconsistent grid gpts"):
            self._scan(wrong_grid)

    def test_a_matching_built_transition_potential_is_accepted(self):
        matching = _synthetic_transition_potential((8.0, 8.0), (64, 64), n=2)

        assert self._scan(matching).array.shape[:2] == (2, 2)


class TestExplicitSitesAreWrapped:
    """Explicit ``sites`` outside the cell were binned without being wrapped.

    ``np.digitize`` returns 0 for any z below the first slice edge, so a site
    at negative z was assigned to the first slice whatever its true wrapped
    depth, i.e. it ionised at the wrong depth; one above the last edge was
    dropped. Sites that differ from in-cell ones by a lattice vector are the
    same sites in a periodic potential, and must give the same result.
    """

    @staticmethod
    def _run(sites):
        atoms = ase.Atoms(
            "Si2",
            positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 5.0)],
            cell=(8, 8, 8),
            pbc=True,
        )
        potential = abtem.Potential(atoms, gpts=(64, 64), slice_thickness=2.0)
        probe = abtem.Probe(semiangle_cutoff=32, energy=ENERGY, extent=(8.0, 8.0))
        probe.grid.match(potential)
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
            sites=sites,
            threshold=1.0,
            double_channel=False,
            lazy=False,
        )
        return np.asarray(abtem.core.backend.asnumpy(measurement.array))

    def test_sites_a_cell_vector_away_give_the_same_result(self):
        inside = ase.Atoms(
            "Si2",
            positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 5.0)],
            cell=(8, 8, 8),
            pbc=True,
        )
        outside = inside.copy()
        # The same two sites, one cell below and one cell above.
        outside.positions[0, 2] -= 8.0
        outside.positions[1, 2] += 8.0

        reference = self._run(inside)
        got = self._run(outside)

        assert reference.sum() > 0.0
        np.testing.assert_allclose(
            got, reference, rtol=1e-6, atol=1e-9 * reference.max()
        )


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

        tp = _synthetic_transition_potential((1.6, 2.0), gpts, n=2)
        sites = np.array([[0.4, 0.5], [0.8, 1.0], [1.2, 1.5], [0.1, 1.9], [1.5, 0.1]])
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


def _core_loss_scan(algorithm, lazy, double_channel):
    atoms, potential, probe = _expansion_scope_setup()
    kwargs = {} if algorithm is None else {"algorithm": algorithm}
    return probe.transition_potential_scan(
        potential=potential,
        transition_potentials=_synthetic_transition_potential(
            potential.extent, potential.gpts, n=2
        ),
        scan=np.array([[0.0, 0.0]]),
        detectors=abtem.AnnularDetector(inner=0.0, outer=None),
        double_channel=double_channel,
        lazy=lazy,
        sites=atoms,
        **kwargs,
    )


@pytest.mark.parametrize("double_channel", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_full_expansion_scope_is_refused_at_call_time(lazy, double_channel):
    """The core-loss multislice does not support ``expansion_scope="full"``.

    The step returned a (waves, backscattered) pair, which the driver used as the
    wave function and failed on with an AttributeError at the first slice. The
    refusal happens when the measurement is built, not when it is computed: a lazy
    call must not return an object that fails later inside a dask task.
    """
    from abtem.multislice import RealSpaceMultislice

    with pytest.raises(NotImplementedError, match="expansion_scope='full'"):
        _core_loss_scan(
            RealSpaceMultislice(expansion_scope="full"), lazy, double_channel
        )


@pytest.mark.parametrize("lazy", [False, True])
def test_core_loss_scan_refuses_before_building_the_probe(lazy, monkeypatch):
    """An unsupported algorithm is refused before the probe is built at every scan
    position."""
    from abtem.multislice import RealSpaceMultislice

    builds = []
    monkeypatch.setattr(
        abtem.Probe, "build", lambda self, *args, **kwargs: builds.append(kwargs)
    )

    with pytest.raises(NotImplementedError):
        _core_loss_scan(RealSpaceMultislice(expansion_scope="full"), lazy, False)
    assert builds == []


def test_full_expansion_scope_is_refused_by_the_driver():
    """The public driver refuses it too, before any work."""
    from abtem.multislice import (
        RealSpaceMultislice,
        transition_potential_multislice_and_detect,
    )

    atoms, potential, probe = _expansion_scope_setup()
    with pytest.raises(NotImplementedError, match="expansion_scope='full'"):
        transition_potential_multislice_and_detect(
            probe.build(scan=np.array([[0.0, 0.0]]), lazy=False),
            potential.build(),
            _synthetic_transition_potential(potential.extent, potential.gpts, n=2),
            detectors=[abtem.AnnularDetector(inner=0.0, outer=None)],
            double_channel=False,
            sites=atoms,
            algorithm=RealSpaceMultislice(expansion_scope="full"),
        )


@pytest.mark.parametrize("double_channel", [False, True])
def test_propagator_expansion_scope_runs_in_the_core_loss_multislice(double_channel):
    """``RealSpaceMultislice(expansion_scope="propagator")`` is the supported
    real-space choice."""
    from abtem.multislice import RealSpaceMultislice

    default = _core_loss_scan(None, False, double_channel)
    realspace = _core_loss_scan(
        RealSpaceMultislice(expansion_scope="propagator"), False, double_channel
    )
    reference = np.asarray(abtem.core.backend.asnumpy(default.array))
    result = np.asarray(abtem.core.backend.asnumpy(realspace.array))
    assert result.shape == reference.shape
    # Both propagators solve the same equation; relative to the maximum they differ
    # by 9e-5 (single channel) and 1e-3 (double channel).
    assert np.abs(result - reference).max() < 1e-2 * np.abs(reference).max()
