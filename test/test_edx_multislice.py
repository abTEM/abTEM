"""Tests for the effective-local-potential route to the ionisation probability.

For a measurement that collects all scattering angles -- X-ray emission -- the
sum over final states of ``int |H_f psi|^2`` equals ``int |psi|^2 mu`` with
``mu = sigma_e^2 sum_f |H_f|^2`` (Parseval). That turns the ionisation signal
into one elementwise product per slice, instead of scattering and propagating a
wave per site per final state.

These tests check that identity, and that the multislice driver built on it
reproduces the existing scattered-wave route.
"""

from __future__ import annotations

import ase
import numpy as np
import pytest

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.core.energy import energy2sigma
from abtem.core.fft import ifft2
from abtem.detectors import IonizationDetector
from abtem.inelastic.core_loss import (
    TransitionPotentialArray,
    effective_ionization_multislice_and_detect,
)

try:
    import cupy as cp
except ImportError:
    cp = None

from utils import gpu  # noqa: E402


ENERGY = 100e3


def _transition_potential(
    extent, gpts, device="cpu", n_transitions=3, seed=0, band_limited=False
):
    """A synthetic transition potential, avoiding a GPAW solve.

    ``band_limited`` applies a Gaussian falloff in k. White noise puts about
    two thirds of its intensity outside the antialiasing disc -- the disc covers
    only pi/9 of the square grid -- which makes the angle-unrestricted
    scattered-wave route truncate massively. A real transition potential decays
    with k, so band-limiting the fixture is what makes the two routes
    comparable.
    """
    xp = cp if device == "gpu" else np
    rng = np.random.default_rng(seed)
    array = (
        rng.standard_normal((n_transitions, *gpts))
        + 1j * rng.standard_normal((n_transitions, *gpts))
    ).astype(np.complex64)

    if band_limited:
        extent_xy = (extent, extent) if np.isscalar(extent) else extent
        kx = np.fft.fftfreq(gpts[0], d=extent_xy[0] / gpts[0])
        ky = np.fft.fftfreq(gpts[1], d=extent_xy[1] / gpts[1])
        k2 = kx[:, None] ** 2 + ky[None] ** 2
        k_cut = 0.2 * np.abs(kx).max()
        array = (array * np.exp(-k2 / (2 * k_cut**2))).astype(np.complex64)

    return TransitionPotentialArray(
        Z=14,
        array=xp.asarray(array),
        energy=ENERGY,
        extent=extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_transitions)))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )


class TestEffectiveIonizationPotential:
    def test_matches_the_sum_over_transitions_at_the_origin(self):
        extent, gpts = 10.0, (64, 64)
        tp = _transition_potential(extent, gpts)

        expected = (np.abs(ifft2(tp.array)) ** 2).sum(0) * energy2sigma(ENERGY) ** 2
        np.testing.assert_allclose(
            tp.effective_ionization_potential(), expected, rtol=1e-6
        )

    def test_is_real_and_non_negative(self):
        tp = _transition_potential(10.0, (64, 64))
        mu = tp.effective_ionization_potential(np.array([[3.0, 7.0]]))
        assert np.isrealobj(mu)
        # mu is a sum of squared moduli, so it cannot go negative. Shifting the
        # summed intensity instead of the amplitudes would ring it to about
        # -9 % of the maximum.
        assert mu.min() >= 0.0

    def test_no_sites_gives_zero(self):
        tp = _transition_potential(10.0, (64, 64))
        empty = ase.Atoms("O", positions=[(1.0, 1.0, 0.0)], cell=(10, 10, 10))
        mu = tp.effective_ionization_potential(empty)
        assert np.allclose(mu, 0.0)

    def test_sites_add_linearly(self):
        tp = _transition_potential(10.0, (64, 64))
        a = np.array([[3.0, 7.0]])
        b = np.array([[6.0, 2.0]])

        both = tp.effective_ionization_potential(np.concatenate([a, b]))
        separate = tp.effective_ionization_potential(
            a
        ) + tp.effective_ionization_potential(b)
        np.testing.assert_allclose(
            both, separate, rtol=0, atol=1e-5 * float(both.max())
        )

    def test_site_batching_does_not_change_the_result(self):
        tp = _transition_potential(10.0, (64, 64))
        sites = np.array([[3.0, 7.0], [6.0, 2.0], [1.0, 1.0], [8.5, 4.25]])
        one_batch = tp.effective_ionization_potential(sites, max_batch=16)
        many = tp.effective_ionization_potential(sites, max_batch=1)
        np.testing.assert_allclose(
            one_batch, many, rtol=0, atol=1e-6 * float(one_batch.max())
        )

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_parseval_identity_against_scattered_waves(self, device):
        """The load-bearing claim: sum_f int |H_f psi|^2 == int |psi|^2 mu."""
        extent, gpts = 10.0, (128, 128)
        tp = _transition_potential(extent, (128, 128), device=device)

        probe = abtem.Probe(
            energy=ENERGY,
            semiangle_cutoff=25,
            extent=extent,
            gpts=gpts,
            device=device,
        )
        waves = probe.build(np.array([[extent / 2, extent / 2]])).compute()

        sites = np.array([[extent / 2, extent / 2], [3.0, 7.0]])

        scattered = tp.scatter(waves, sites)
        direct = float(
            abtem.core.backend.asnumpy(
                (np.abs(scattered.array) ** 2).sum() * np.prod(waves.sampling)
            )
        )

        mu = tp.effective_ionization_potential(sites)
        intensity = np.abs(np.squeeze(abtem.core.backend.asnumpy(waves.array))) ** 2
        shortcut = float(
            (intensity * abtem.core.backend.asnumpy(mu)).sum()
            * np.prod(waves.sampling)
        )

        assert shortcut == pytest.approx(direct, rel=1e-5)


class TestIonizationDetectorBasics:
    def test_no_mu_weights_uniformly(self):
        # The default is the X-ray detector: total intensity, no weighting.
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=25, extent=10.0, gpts=64)
        waves = probe.build(np.array([[5.0, 5.0]])).compute()
        plain = IonizationDetector().detect(waves)
        unit = IonizationDetector(mu=1.0).detect(waves)
        np.testing.assert_allclose(
            np.asarray(plain.array), np.asarray(unit.array), rtol=1e-6
        )

    def test_output_matches_annular_detector_shape(self):
        extent, gpts = 10.0, (64, 64)
        tp = _transition_potential(extent, gpts)
        probe = abtem.Probe(
            energy=ENERGY, semiangle_cutoff=25, extent=extent, gpts=gpts
        )
        scan = abtem.GridScan(start=(0, 0), end=(5, 5), gpts=(3, 4))
        waves = probe.build(scan).compute()

        detector = IonizationDetector(mu=tp.effective_ionization_potential())
        got = detector.detect(waves)
        reference = abtem.AnnularDetector(inner=0.0, outer=None).detect(waves)

        assert got.shape == reference.shape
        assert type(got) is type(reference)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("n_exit_planes", [1, 3])
def test_matches_the_scattered_wave_route(device, n_exit_planes):
    """The effective-potential driver must reproduce the existing EELS driver
    with an angle-unrestricted detector, up to the antialiasing truncation that
    only the scattered-wave route suffers."""
    atoms = ase.build.bulk("Si", cubic=True) * (1, 1, 3)

    exit_planes = None if n_exit_planes == 1 else n_exit_planes
    potential = abtem.Potential(
        atoms, gpts=(64, 64), slice_thickness=1.4, exit_planes=exit_planes,
        device=device,
    )

    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)

    def make_tp():
        return _transition_potential(
            potential.extent, potential.gpts, device=device, band_limited=True
        )

    scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2))

    reference = probe.transition_potential_scan(
        potential=potential,
        transition_potentials=make_tp(),
        scan=scan,
        detectors=abtem.AnnularDetector(inner=0.0, outer=None),
        double_channel=False,
        lazy=False,
    ).compute()

    waves = probe.build(scan).compute()
    got = effective_ionization_multislice_and_detect(waves, potential, make_tp())[0]

    assert got.shape == reference.shape

    a = np.asarray(abtem.core.backend.asnumpy(got.array))
    b = np.asarray(abtem.core.backend.asnumpy(reference.array))

    # The scattered-wave route integrates in reciprocal space out to the
    # antialiasing cutoff and so loses the aliased tail; the real-space route
    # does not. The difference is small but systematic for a band-limited
    # transition potential, and the real-space value is the more accurate.
    assert np.all(a >= b * (1 - 1e-6))
    np.testing.assert_allclose(a, b, rtol=5e-3, atol=0)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_entrance_exit_plane_carries_no_ionisation(device):
    """At t = 0 nothing has been traversed, so the core-loss signal is zero.

    The scattered-wave driver used to detect the incident *elastic* wave here,
    writing the full unscattered intensity into the t = 0 bin.
    """
    atoms = ase.build.bulk("Si", cubic=True) * (1, 1, 3)
    potential = abtem.Potential(
        atoms, gpts=(64, 64), slice_thickness=1.4, exit_planes=3, device=device
    )
    assert potential.exit_planes[0] == -1

    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)
    scan = np.array([[0.0, 0.0]])

    def make_tp():
        return _transition_potential(
            potential.extent, potential.gpts, device=device, band_limited=True
        )

    reference = probe.transition_potential_scan(
        potential=potential,
        transition_potentials=make_tp(),
        scan=scan,
        detectors=abtem.AnnularDetector(inner=0.0, outer=None),
        double_channel=False,
        lazy=False,
    ).compute()
    got = effective_ionization_multislice_and_detect(
        probe.build(scan).compute(), potential, make_tp()
    )[0]

    a = np.asarray(abtem.core.backend.asnumpy(got.array)).ravel()
    b = np.asarray(abtem.core.backend.asnumpy(reference.array)).ravel()

    assert a[0] == 0.0
    assert b[0] == 0.0
    # and the signal must grow from there
    assert np.all(np.diff(a) > 0)
    assert np.all(np.diff(b) > 0)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_signal_grows_monotonically_with_thickness(device):
    atoms = ase.build.bulk("Si", cubic=True) * (1, 1, 4)
    potential = abtem.Potential(
        atoms, gpts=(64, 64), slice_thickness=1.4, exit_planes=4, device=device
    )
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)

    waves = probe.build(np.array([[0.0, 0.0]])).compute()
    got = effective_ionization_multislice_and_detect(
        waves, potential,
        _transition_potential(potential.extent, potential.gpts, device=device),
    )[0]

    values = np.asarray(abtem.core.backend.asnumpy(got.array)).ravel()
    assert np.all(np.diff(values) >= -1e-12), values


def test_no_matching_sites_raises():
    atoms = ase.build.bulk("Cu", cubic=True)
    potential = abtem.Potential(atoms, gpts=(64, 64), slice_thickness=1.8)
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
    probe.grid.match(potential)
    waves = probe.build(np.array([[0.0, 0.0]])).compute()

    with pytest.raises(RuntimeError, match="No scattering sites"):
        effective_ionization_multislice_and_detect(
            waves, potential,
            _transition_potential(potential.extent, potential.gpts),
        )


def _si_setup(device="cpu", exit_planes=None, nz=3):
    atoms = ase.build.bulk("Si", cubic=True) * (1, 1, nz)
    potential = abtem.Potential(
        atoms, gpts=(64, 64), slice_thickness=1.4, exit_planes=exit_planes,
        device=device,
    )
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)
    return potential, probe


class TestPublicEntryPoints:
    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_probe_scan_matches_the_driver(self, device):
        potential, probe = _si_setup(device)
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2))

        def make_tp():
            return _transition_potential(
                potential.extent, potential.gpts, device=device, band_limited=True
            )

        got = probe.ionization_scan(
            potential, make_tp(), scan=scan, lazy=False
        ).compute()
        expected = effective_ionization_multislice_and_detect(
            probe.build(scan).compute(), potential, make_tp()
        )[0]

        np.testing.assert_allclose(
            np.asarray(abtem.core.backend.asnumpy(got.array)),
            np.asarray(abtem.core.backend.asnumpy(expected.array)),
            rtol=1e-6,
        )

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_lazy_matches_eager(self, device):
        potential, probe = _si_setup(device, exit_planes=3)
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 3))

        def make_tp():
            return _transition_potential(
                potential.extent, potential.gpts, device=device, band_limited=True
            )

        eager = probe.ionization_scan(potential, make_tp(), scan=scan, lazy=False)
        lazy = probe.ionization_scan(potential, make_tp(), scan=scan, lazy=True)

        assert lazy.shape == eager.shape
        np.testing.assert_allclose(
            np.asarray(abtem.core.backend.asnumpy(lazy.compute().array)),
            np.asarray(abtem.core.backend.asnumpy(eager.compute().array)),
            rtol=1e-6,
        )

    def test_output_is_an_images_object_over_the_scan(self):
        potential, probe = _si_setup()
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(3, 4))
        got = probe.ionization_scan(
            potential,
            _transition_potential(potential.extent, potential.gpts, band_limited=True),
            scan=scan,
            lazy=False,
        ).compute()
        assert isinstance(got, abtem.Images)
        assert got.shape[-2:] == (3, 4)

    def test_multiple_edges_are_stacked(self):
        potential, probe = _si_setup()
        scan = np.array([[0.0, 0.0]])
        tps = [
            _transition_potential(
                potential.extent, potential.gpts, seed=seed, band_limited=True
            )
            for seed in (0, 1)
        ]
        got = probe.ionization_scan(potential, tps, scan=scan, lazy=False).compute()
        assert got.shape[0] == 2

    def test_angle_resolved_detector_is_rejected(self):
        potential, probe = _si_setup()
        waves = probe.build(np.array([[0.0, 0.0]])).compute()
        with pytest.raises(RuntimeError, match="only accepts IonizationDetector"):
            effective_ionization_multislice_and_detect(
                waves,
                potential,
                _transition_potential(potential.extent, potential.gpts),
                detectors=[abtem.AnnularDetector(inner=0.0, outer=None)],
            )

    def test_xray_detector_converts_the_scan(self):
        potential, probe = _si_setup()
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2))
        ionisation = probe.ionization_scan(
            potential,
            _transition_potential(potential.extent, potential.gpts, band_limited=True),
            scan=scan,
            lazy=False,
        ).compute()

        detector = abtem.XrayDetector(solid_angle=0.7)
        counts = detector.to_counts(ionisation, "Si", 1, 0)

        expected = detector.total_yield("Si", 1, 0)
        np.testing.assert_allclose(
            np.asarray(counts.array),
            np.asarray(ionisation.array) * expected,
            rtol=1e-6,
        )
        assert counts.metadata["units"] == "photons / electron"


class TestJointEelsAndEdx:
    """One scattered-wave pass can fill an EELS aperture and an EDX detector.

    The elastic multislice and the scattered waves are the expensive part and
    are shared; the two detectors differ only in their angular acceptance.
    """

    @staticmethod
    def _run(potential, probe, scan, detectors):
        return probe.transition_potential_scan(
            potential=potential,
            transition_potentials=_transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            scan=scan,
            detectors=detectors,
            double_channel=False,
            lazy=False,
        )

    def test_two_detectors_return_a_list(self):
        potential, probe = _si_setup()
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2))

        eels = abtem.AnnularDetector(inner=0.0, outer=30.0)
        edx = abtem.AnnularDetector(inner=0.0, outer=None)

        both = self._run(potential, probe, scan, [eels, edx])
        assert isinstance(both, list) and len(both) == 2

    def test_joint_pass_matches_separate_passes(self):
        potential, probe = _si_setup()
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2))

        eels = abtem.AnnularDetector(inner=0.0, outer=30.0)
        edx = abtem.AnnularDetector(inner=0.0, outer=None)

        both = self._run(potential, probe, scan, [eels, edx])
        alone_eels = self._run(potential, probe, scan, eels)
        alone_edx = self._run(potential, probe, scan, edx)

        np.testing.assert_allclose(
            np.asarray(both[0].compute().array),
            np.asarray(alone_eels.compute().array),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(both[1].compute().array),
            np.asarray(alone_edx.compute().array),
            rtol=1e-6,
        )

    def test_aperture_captures_less_than_all_angles(self):
        potential, probe = _si_setup()
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2))

        both = self._run(
            potential,
            probe,
            scan,
            [
                abtem.AnnularDetector(inner=0.0, outer=30.0),
                abtem.AnnularDetector(inner=0.0, outer=None),
            ],
        )
        eels = np.asarray(both[0].compute().array)
        edx = np.asarray(both[1].compute().array)
        assert np.all(eels < edx)

    def test_multiple_detectors_with_multiple_edges(self):
        potential, probe = _si_setup()
        scan = np.array([[0.0, 0.0]])
        tps = [
            _transition_potential(
                potential.extent, potential.gpts, seed=seed, band_limited=True
            )
            for seed in (0, 1)
        ]
        out = probe.transition_potential_scan(
            potential=potential,
            transition_potentials=tps,
            scan=scan,
            detectors=[
                abtem.AnnularDetector(inner=0.0, outer=30.0),
                abtem.AnnularDetector(inner=0.0, outer=None),
            ],
            double_channel=False,
            lazy=False,
        )
        assert isinstance(out, list) and len(out) == 2
        # each is stacked over the two edges
        for measurement in out:
            assert measurement.compute().shape[0] == 2


class TestIonizationDetector:
    """No-FFT, non-truncating replacement for AnnularDetector(0, None)."""

    def test_matches_the_unrestricted_annular_detector(self):
        potential, probe = _si_setup()
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2))

        def run(detector):
            return probe.transition_potential_scan(
                potential=potential,
                transition_potentials=_transition_potential(
                    potential.extent, potential.gpts, band_limited=True
                ),
                scan=scan,
                detectors=detector,
                double_channel=False,
                lazy=False,
            ).compute()

        reciprocal = np.asarray(run(abtem.AnnularDetector(0.0, None)).array)
        real_space = np.asarray(run(abtem.IonizationDetector()).array)

        # The reciprocal route truncates at the antialiasing cutoff, so the
        # real-space total is always the larger of the two.
        assert np.all(real_space >= reciprocal)
        np.testing.assert_allclose(real_space, reciprocal, rtol=5e-3)

    def test_scalar_mu_weights_uniformly(self):
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=25, extent=10.0, gpts=64)
        waves = probe.build(np.array([[5.0, 5.0]])).compute()

        one = IonizationDetector(mu=1.0).detect(waves)
        three = IonizationDetector(mu=3.0).detect(waves)
        np.testing.assert_allclose(
            np.asarray(three.array), 3 * np.asarray(one.array), rtol=1e-6
        )

    def test_unit_weight_recovers_the_incident_intensity(self):
        # abTEM detectors read 1.0 on an unscattered probe.
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=25, extent=10.0, gpts=64)
        waves = probe.build(np.array([[5.0, 5.0]])).compute()
        got = float(np.squeeze(np.asarray(IonizationDetector(mu=1.0).detect(waves).array)))
        assert got == pytest.approx(1.0, rel=1e-4)

    def test_exported_from_the_top_level(self):
        assert abtem.IonizationDetector is IonizationDetector
        assert "IonizationDetector" in abtem.__all__


class TestPrecisionConfig:
    """Everything must follow abtem.config['precision'], not a literal dtype.

    The real-valued geometry arrays -- site coordinates, sampling, wave vectors,
    sub-pixel shifts -- used to be pinned to np.float32. Because
    ``fft_shift_kernel`` takes its output dtype from the positions handed to it,
    that forced the whole downstream chain back to single precision even under
    ``precision="float64"``.
    """

    @staticmethod
    def _setup():
        atoms = ase.build.bulk("Si", cubic=True) * (1, 1, 3)
        potential = abtem.Potential(atoms, gpts=(64, 64), slice_thickness=1.4)
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
        probe.grid.match(potential)
        return potential, probe

    @pytest.mark.parametrize(
        "precision, real, complex_",
        [("float32", np.float32, np.complex64), ("float64", np.float64, np.complex128)],
    )
    def test_effective_potential_follows_the_config(self, precision, real, complex_):
        with abtem.config.set({"precision": precision}):
            potential, _ = self._setup()
            tp = _transition_potential(
                potential.extent, potential.gpts, band_limited=True
            )
            tp._array = tp.array.astype(complex_)
            mu = tp.effective_ionization_potential(np.array([[1.0, 2.0]]))
            assert mu.dtype == real

    @pytest.mark.parametrize(
        "precision, real, complex_",
        [("float32", np.float32, np.complex64), ("float64", np.float64, np.complex128)],
    )
    def test_both_drivers_follow_the_config(self, precision, real, complex_):
        with abtem.config.set({"precision": precision}):
            potential, probe = self._setup()
            scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2))

            def make_tp():
                tp = _transition_potential(
                    potential.extent, potential.gpts, band_limited=True
                )
                tp._array = tp.array.astype(complex_)
                return tp

            waves = probe.build(scan).compute()
            assert waves.array.dtype == complex_

            effective = effective_ionization_multislice_and_detect(
                waves, potential, make_tp()
            )[0]
            assert effective.array.dtype == real

            scattered = probe.transition_potential_scan(
                potential=potential,
                transition_potentials=make_tp(),
                scan=scan,
                detectors=abtem.AnnularDetector(inner=0.0, outer=None),
                double_channel=False,
                lazy=False,
            ).compute()
            assert scattered.array.dtype == real

    def test_single_and_double_precision_agree(self):
        results = {}
        for precision, complex_ in [
            ("float32", np.complex64),
            ("float64", np.complex128),
        ]:
            with abtem.config.set({"precision": precision}):
                potential, probe = self._setup()
                scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2))
                tp = _transition_potential(
                    potential.extent, potential.gpts, band_limited=True
                )
                tp._array = tp.array.astype(complex_)
                got = effective_ionization_multislice_and_detect(
                    probe.build(scan).compute(), potential, tp
                )[0]
                results[precision] = np.asarray(got.array, dtype=np.float64)

        np.testing.assert_allclose(
            results["float32"], results["float64"], rtol=1e-5
        )

    def test_no_hardcoded_dtypes_remain_in_core_loss(self):
        import re
        from pathlib import Path

        import abtem.inelastic.core_loss as module

        source = Path(module.__file__).read_text()
        offenders = [
            line
            for line in source.splitlines()
            if re.search(r"(dtype\s*=\s*(np|xp)\.(float|complex)\d+|"
                         r"(np|xp)\.(float|complex)\d+\s*\()", line)
        ]
        assert not offenders, (
            "use get_dtype() so abtem.config['precision'] is honoured:\n"
            + "\n".join(offenders)
        )


class TestXrayDetectorAsDetector:
    """XrayDetector models the experiment, so it goes straight into a scan.

    That needs the waves to carry the identity of the ionised edge, which the
    transition potential now stamps on them in ``scatter``.
    """

    @staticmethod
    def _setup():
        unit = ase.build.bulk("Si", cubic=True)
        atoms = unit * (1, 1, 2)
        potential = abtem.Potential(
            atoms, gpts=(32, 32), slice_thickness=float(unit.cell[2, 2])
        )
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
        probe.grid.match(potential)
        return atoms, potential, probe

    def test_scatter_stamps_the_edge_on_the_waves(self):
        _, potential, probe = self._setup()
        tp = _transition_potential(potential.extent, potential.gpts)
        waves = probe.build(np.array([[0.0, 0.0]])).compute()

        assert "Z" not in waves.metadata
        scattered = tp.scatter(waves, np.array([[0.0, 0.0]]))
        assert scattered.metadata["Z"] == 14
        assert scattered.metadata["n"] == 1
        assert scattered.metadata["l"] == 0
        # and the wave's own metadata is preserved
        assert scattered.metadata["energy"] == waves.metadata["energy"]

    def test_counts_are_the_ionisation_probability_times_the_yield(self):
        atoms, potential, probe = self._setup()
        scan = np.array([[0.0, 0.0]])

        xray = abtem.XrayDetector.from_sdd(solid_angle=0.7, window=("Be", 8.0))
        ionization = abtem.IonizationDetector()

        both = probe.transition_potential_scan(
            potential=potential,
            transition_potentials=_transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            scan=scan,
            detectors=[ionization, xray],
            double_channel=False,
            lazy=False,
            sites=atoms,
        )
        ion = np.asarray(both[0].compute().array)
        counts = np.asarray(both[1].compute().array)

        np.testing.assert_allclose(
            counts, ion * xray.total_yield("Si", 1, 0), rtol=1e-6
        )

    def test_direct_detection_matches_the_two_step_route(self):
        atoms, potential, probe = self._setup()
        scan = np.array([[0.0, 0.0]])
        xray = abtem.XrayDetector(solid_angle=0.7)

        direct = probe.transition_potential_scan(
            potential=potential,
            transition_potentials=_transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            scan=scan,
            detectors=xray,
            double_channel=False,
            lazy=False,
            sites=atoms,
        ).compute()

        ionization = probe.ionization_scan(
            potential,
            _transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            scan=scan,
            sites=atoms,
            lazy=False,
        ).compute()
        two_step = xray.to_counts(ionization, "Si", 1, 0)

        np.testing.assert_allclose(
            np.asarray(direct.array), np.asarray(two_step.array), rtol=5e-3
        )

    def test_raises_on_waves_without_an_edge(self):
        _, _, probe = self._setup()
        waves = probe.build(np.array([[0.0, 0.0]])).compute()
        with pytest.raises(RuntimeError, match="which edge was ionised"):
            abtem.XrayDetector(solid_angle=0.7).detect(waves)


class TestPlaneWaveEntryPoint:
    """A plane wave of unit intensity gives the cross-section per unit area."""

    @staticmethod
    def _setup():
        unit = ase.build.bulk("Si", cubic=True)
        atoms = unit * (1, 1, 2)
        potential = abtem.Potential(
            atoms, gpts=(32, 32), slice_thickness=float(unit.cell[2, 2])
        )
        return atoms, potential

    def test_matches_the_waves_level_driver(self):
        atoms, potential = self._setup()
        plane_wave = abtem.PlaneWave(energy=ENERGY)
        plane_wave.grid.match(potential)

        def make_tp():
            return _transition_potential(
                potential.extent, potential.gpts, band_limited=True
            )

        got = plane_wave.ionization_multislice(
            potential, make_tp(), sites=atoms, lazy=False
        )

        waves = plane_wave.build(lazy=False)
        expected = effective_ionization_multislice_and_detect(
            waves, potential, make_tp(), sites=atoms
        )[0]

        np.testing.assert_allclose(
            np.asarray(got.array), np.asarray(expected.array), rtol=1e-6
        )

    def test_lazy_matches_eager(self):
        atoms, potential = self._setup()
        plane_wave = abtem.PlaneWave(energy=ENERGY)
        plane_wave.grid.match(potential)

        def run(lazy):
            return plane_wave.ionization_multislice(
                potential,
                _transition_potential(
                    potential.extent, potential.gpts, band_limited=True
                ),
                sites=atoms,
                lazy=lazy,
            )

        np.testing.assert_allclose(
            np.asarray(run(True).compute().array),
            np.asarray(run(False).compute().array),
            rtol=1e-6,
        )

    def test_signal_is_positive_and_scales_with_thickness(self):
        unit = ase.build.bulk("Si", cubic=True)
        atoms = unit * (1, 1, 4)
        potential = abtem.Potential(
            atoms, gpts=(32, 32), slice_thickness=1.4, exit_planes=4
        )
        plane_wave = abtem.PlaneWave(energy=ENERGY)
        plane_wave.grid.match(potential)

        got = plane_wave.ionization_multislice(
            potential,
            _transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            sites=atoms,
            lazy=False,
        )
        values = np.asarray(got.array).ravel()
        assert np.all(values >= 0.0)
        assert np.all(np.diff(values) >= -1e-12), values

    def test_xray_detector_converts_the_result(self):
        atoms, potential = self._setup()
        plane_wave = abtem.PlaneWave(energy=ENERGY)
        plane_wave.grid.match(potential)

        ionization = plane_wave.ionization_multislice(
            potential,
            _transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            sites=atoms,
            lazy=False,
        )
        detector = abtem.XrayDetector(solid_angle=0.7)
        counts = detector.to_counts(ionization, "Si", 1, 0)
        np.testing.assert_allclose(
            np.asarray(counts.array),
            np.asarray(ionization.array) * detector.total_yield("Si", 1, 0),
            rtol=1e-6,
        )


