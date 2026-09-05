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


class TestPrismEffectiveIonization:
    """PRISM port: I(p) = c_p^dagger M c_p with M = S^dagger diag(mu) S.

    M does not depend on the probe position, so it is accumulated once as the
    S-matrix propagates and the scan is a quadratic form per position.
    """

    @staticmethod
    def _setup(device="cpu", nz=2, exit_planes=None, slice_thickness=None):
        unit = ase.build.bulk("Si", cubic=True)
        atoms = unit * (1, 1, nz)
        if slice_thickness is None:
            slice_thickness = float(unit.cell[2, 2])
        potential = abtem.Potential(
            atoms,
            gpts=(32, 32),
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
        )
        return atoms, potential

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_matches_multislice_at_interpolation_one(self, device):
        atoms, potential = self._setup(device)
        cutoff = 20.0

        def make_tp():
            return _transition_potential(
                potential.extent, potential.gpts, device=device, band_limited=True
            )

        probe = abtem.Probe(
            energy=ENERGY, semiangle_cutoff=cutoff, device=device
        )
        probe.grid.match(potential)
        scan = np.array([[0.0, 0.0], [1.3, 2.1]])

        reference = effective_ionization_multislice_and_detect(
            probe.build(scan).compute(), potential, make_tp(), sites=atoms
        )[0]

        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=cutoff,
            interpolation=1,
            downsample=False,
            device=device,
        )
        got = s_matrix.ionization_scan(make_tp(), scan=scan, sites=atoms)

        np.testing.assert_allclose(
            np.asarray(abtem.core.backend.asnumpy(got.array)),
            np.asarray(abtem.core.backend.asnumpy(reference.array)),
            rtol=1e-4,
        )

    def test_exit_planes_are_resolved_and_monotonic(self):
        # `exit_planes=3` counts slices, so the cell needs many thin slices to
        # produce several planes; with one slice per cell it yields just one.
        atoms, potential = self._setup(nz=3, exit_planes=3, slice_thickness=1.4)
        assert len(potential.exit_planes) > 1
        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20.0,
            interpolation=1,
            downsample=False,
        )
        got = s_matrix.ionization_scan(
            _transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            scan=np.array([[0.0, 0.0]]),
            sites=atoms,
        )
        values = np.asarray(got.array).ravel()
        assert values[0] == 0.0  # entrance plane, nothing traversed
        assert np.all(np.diff(values) >= -1e-12), values

    def test_signal_is_real_and_non_negative(self):
        atoms, potential = self._setup()
        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20.0,
            interpolation=1,
            downsample=False,
        )
        got = s_matrix.ionization_scan(
            _transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            scan=abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2)),
            sites=atoms,
        )
        array = np.asarray(got.array)
        assert np.isrealobj(array)
        assert np.all(array >= 0.0)

    def test_scan_shape_is_preserved(self):
        atoms, potential = self._setup()
        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20.0,
            interpolation=1,
            downsample=False,
        )
        got = s_matrix.ionization_scan(
            _transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            scan=abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(3, 4)),
            sites=atoms,
        )
        assert got.shape[-2:] == (3, 4)

    def test_angle_resolved_detector_is_rejected(self):
        from abtem.inelastic.core_loss import prism_effective_ionization_scan

        atoms, potential = self._setup()
        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20.0,
            interpolation=1,
            downsample=False,
        )
        with pytest.raises(RuntimeError, match="only accepts IonizationDetector"):
            prism_effective_ionization_scan(
                s_matrix,
                _transition_potential(potential.extent, potential.gpts),
                scan=np.array([[0.0, 0.0]]),
                detectors=abtem.AnnularDetector(inner=0.0, outer=None),
                sites=atoms,
            )

    @pytest.mark.parametrize("interpolation", [1, 2])
    def test_interpolation_matches_multislice(self, interpolation):
        """Interpolation replicates the PRISM wave function across the cell.

        Two corrections are needed and both are exact: each probe may only see
        the sites inside its own window, and the reduction normalises every
        replica to unit intensity so the cell carries prod(interpolation) times
        one probe's worth.
        """
        atoms, potential = self._setup()
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20.0)
        probe.grid.match(potential)
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2))

        def make_tp():
            return _transition_potential(
                potential.extent, potential.gpts, band_limited=True
            )

        reference = effective_ionization_multislice_and_detect(
            probe.build(scan).compute(), potential, make_tp(), sites=atoms
        )[0]

        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20.0,
            interpolation=interpolation,
            downsample=False,
        )
        got = s_matrix.ionization_scan(make_tp(), scan=scan, sites=atoms)

        # The band-limited fixture has a far more compact mu than a real
        # transition potential, so this tolerance is optimistic: on a real Si K
        # potential interpolation 2 runs about 4 % low with 15 % at the worst
        # position, which is the PRISM interpolation error for a delocalised
        # potential rather than a defect here.
        rtol = 1e-6 if interpolation == 1 else 2e-2
        np.testing.assert_allclose(
            np.asarray(got.array), np.asarray(reference.array), rtol=rtol
        )

    def test_interpolation_does_not_scale_the_signal(self):
        """A missing 1/prod(interpolation) showed up as an exact factor of 4."""
        atoms, potential = self._setup()
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2))

        def run(interpolation):
            s_matrix = abtem.SMatrix(
                potential=potential,
                energy=ENERGY,
                semiangle_cutoff=20.0,
                interpolation=interpolation,
                downsample=False,
            )
            return np.asarray(
                s_matrix.ionization_scan(
                    _transition_potential(
                        potential.extent, potential.gpts, band_limited=True
                    ),
                    scan=scan,
                    sites=atoms,
                ).array
            )

        np.testing.assert_allclose(run(2), run(1), rtol=2e-2)

    def test_window_assigns_a_boundary_site_to_one_probe_only(self):
        """A symmetric window double-counted sites sitting on the boundary."""
        atoms, potential = self._setup()
        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20.0,
            interpolation=2,
            downsample=False,
        )
        # A scan straddling the window edge is where double counting showed.
        window = np.asarray(s_matrix.window_extent)
        scan = np.array([[0.0, 0.0], [window[0] / 2, 0.0], [window[0], 0.0]])
        got = np.asarray(
            s_matrix.ionization_scan(
                _transition_potential(
                    potential.extent, potential.gpts, band_limited=True
                ),
                scan=scan,
                sites=atoms,
            ).array
        )
        # Positions one window apart are equivalent by the PRISM periodicity.
        assert got[0] == pytest.approx(got[2], rel=1e-5)

    def test_interpolated_smatrix_still_works_via_the_scattered_wave_route(self):
        # The documented fallback must actually run.
        atoms, potential = self._setup()
        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20.0,
            interpolation=2,
            downsample=False,
        )
        got = s_matrix.transition_potential_scan(
            transition_potentials=_transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            scan=np.array([[0.0, 0.0]]),
            detectors=abtem.IonizationDetector(),
            sites=atoms,
        )
        assert np.all(np.asarray(got.compute().array) >= 0.0)


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

    def test_prism_applies_the_yield_too(self):
        atoms, potential, probe = self._setup()
        from abtem.inelastic.core_loss import prism_effective_ionization_scan

        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20.0,
            interpolation=1,
            downsample=False,
        )
        xray = abtem.XrayDetector(solid_angle=0.7)

        bare = s_matrix.ionization_scan(
            _transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            scan=np.array([[0.0, 0.0]]),
            sites=atoms,
        )
        counts = prism_effective_ionization_scan(
            s_matrix,
            _transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            scan=np.array([[0.0, 0.0]]),
            detectors=[xray],
            sites=atoms,
        )[0]

        np.testing.assert_allclose(
            np.asarray(counts.array),
            np.asarray(bare.array) * xray.total_yield("Si", 1, 0),
            rtol=1e-6,
        )


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


class TestIonizationWindow:
    """The S^dagger diag(mu) S integral only needs a window around each site."""

    def test_window_is_a_physical_size_not_a_pixel_count(self):
        from abtem.inelastic.core_loss import (
            SubshellTransitions,
            _ionization_window_gpts,
        )

        pytest.importorskip("gpaw")

        sizes = []
        for extent, gpts in [(10.0, 128), (10.0, 256)]:
            tp = SubshellTransitions(14, 1, 0, epsilon=25.0).get_transition_potentials(
                extent=extent, gpts=gpts, energy=ENERGY
            ).build()
            window = _ionization_window_gpts(tp.effective_ionization_potential())
            sizes.append(window[0] * extent / gpts)

        # Same physical window from both grids, to within a pixel or two.
        assert sizes[0] == pytest.approx(sizes[1], rel=0.1)

    def test_window_shrinks_for_a_localised_potential(self):
        from abtem.inelastic.core_loss import (
            SubshellTransitions,
            _ionization_window_gpts,
        )

        pytest.importorskip("gpaw")

        tp = SubshellTransitions(14, 1, 0, epsilon=25.0).get_transition_potentials(
            extent=20.0, gpts=256, energy=ENERGY
        ).build()
        window = _ionization_window_gpts(tp.effective_ionization_potential())
        assert np.prod(window) < 0.25 * np.prod(tp.gpts)

    def test_delocalised_potential_keeps_the_whole_grid(self):
        from abtem.inelastic.core_loss import _ionization_window_gpts

        # White noise spread over the cell must not be clipped.
        tp = _transition_potential(10.0, (64, 64))
        window = _ionization_window_gpts(tp.effective_ionization_potential())
        assert window == (64, 64)

    def test_tighter_tolerance_gives_a_larger_window(self):
        from abtem.inelastic.core_loss import (
            SubshellTransitions,
            _ionization_window_gpts,
        )

        pytest.importorskip("gpaw")

        mu = SubshellTransitions(14, 1, 0, epsilon=25.0).get_transition_potentials(
            extent=20.0, gpts=256, energy=ENERGY
        ).build().effective_ionization_potential()

        loose = _ionization_window_gpts(mu, tolerance=1e-2)
        tight = _ionization_window_gpts(mu, tolerance=1e-4)
        assert np.prod(tight) > np.prod(loose)


class TestInelasticCropConsistency:
    """The window parameter must match SMatrix.transition_potential_scan.

    PR #289 established ``inelastic_crop`` as a real-space side length in
    Angstrom, scalar or pair, clamped to the cell with a warning. The
    effective-potential driver takes the same parameter with the same meaning
    rather than inventing a second window concept.
    """

    @staticmethod
    def _setup():
        unit = ase.build.bulk("Si", cubic=True)
        atoms = unit * (1, 1, 2)
        potential = abtem.Potential(
            atoms, gpts=(32, 32), slice_thickness=float(unit.cell[2, 2])
        )
        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20.0,
            interpolation=1,
            downsample=False,
        )
        return atoms, potential, s_matrix

    def _run(self, s_matrix, potential, atoms, **kwargs):
        return np.asarray(
            s_matrix.ionization_scan(
                _transition_potential(potential.extent, potential.gpts),
                scan=np.array([[0.0, 0.0]]),
                sites=atoms,
                **kwargs,
            ).array
        )

    def test_the_transition_potential_scan_also_takes_it(self):
        # Same name, same units, on both PRISM entry points.
        import inspect

        assert "inelastic_crop" in inspect.signature(
            abtem.SMatrix.transition_potential_scan
        ).parameters
        assert "inelastic_crop" in inspect.signature(
            abtem.SMatrix.ionization_scan
        ).parameters

    def test_oversized_crop_is_clamped_with_a_warning(self):
        atoms, potential, s_matrix = self._setup()
        with pytest.warns(RuntimeWarning, match="exceeds the PRISM cell"):
            clamped = self._run(
                s_matrix, potential, atoms, inelastic_crop=1000.0
            )
        full = self._run(s_matrix, potential, atoms)
        np.testing.assert_allclose(clamped, full, rtol=1e-6)

    def test_a_tight_crop_is_honoured_not_silently_widened(self):
        # A crop the caller asked for is an accuracy choice, so it must be
        # applied even where the full-grid path would be cheaper.
        atoms, potential, s_matrix = self._setup()
        cropped = self._run(s_matrix, potential, atoms, inelastic_crop=2.0)
        full = self._run(s_matrix, potential, atoms)
        assert cropped < full

    def test_a_scalar_and_a_pair_agree(self):
        atoms, potential, s_matrix = self._setup()
        np.testing.assert_allclose(
            self._run(s_matrix, potential, atoms, inelastic_crop=2.0),
            self._run(s_matrix, potential, atoms, inelastic_crop=(2.0, 2.0)),
            rtol=1e-6,
        )

    def test_no_double_channel_option(self):
        # Absent by design: with no angular restriction the elastic propagation
        # of the ejected-state wave is unitary, so double channelling cannot
        # change the total count. `lazy` is supported, like the sibling method.
        import inspect

        parameters = inspect.signature(abtem.SMatrix.ionization_scan).parameters
        assert "double_channel" not in parameters
        assert "lazy" in parameters


class TestLazyMatchesEagerEverywhere:
    """Every entry point and feature must give the same answer either way."""

    UNIT = ase.build.bulk("Si", cubic=True)

    @classmethod
    def _potential(cls, exit_planes=None, num_configs=None):
        atoms = cls.UNIT * (1, 1, 2)
        source = atoms
        if num_configs is not None:
            source = abtem.FrozenPhonons(
                atoms, num_configs=num_configs, sigmas=0.05, seed=1
            )
        potential = abtem.Potential(
            source,
            gpts=(32, 32),
            slice_thickness=float(cls.UNIT.cell[2, 2]),
            exit_planes=exit_planes,
        )
        return atoms, potential

    @staticmethod
    def _array(measurement):
        if hasattr(measurement, "compute"):
            measurement = measurement.compute()
        return np.asarray(measurement.array)

    def _assert_same(self, eager, lazy):
        a, b = self._array(eager), self._array(lazy)
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=0)

    @pytest.mark.parametrize(
        "exit_planes, num_configs",
        [(None, None), (1, None), (None, 2)],
        ids=["plain", "exit-planes", "frozen-phonons"],
    )
    def test_probe_scan(self, exit_planes, num_configs):
        atoms, potential = self._potential(exit_planes, num_configs)
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
        probe.grid.match(potential)
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 3))

        def run(lazy):
            return probe.ionization_scan(
                potential,
                _transition_potential(
                    potential.extent, potential.gpts, band_limited=True
                ),
                scan=scan,
                sites=atoms,
                lazy=lazy,
            )

        self._assert_same(run(False), run(True))

    def test_plane_wave(self):
        atoms, potential = self._potential()
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

        self._assert_same(run(False), run(True))

    @pytest.mark.parametrize("interpolation", [1, 2])
    def test_smatrix_scan(self, interpolation):
        atoms, potential = self._potential()
        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20.0,
            interpolation=interpolation,
            downsample=False,
        )
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 3))

        def run(lazy):
            return s_matrix.ionization_scan(
                _transition_potential(
                    potential.extent, potential.gpts, band_limited=True
                ),
                scan=scan,
                sites=atoms,
                lazy=lazy,
            )

        self._assert_same(run(False), run(True))

    @pytest.mark.parametrize(
        "kwargs_name",
        ["xray", "absorption", "inelastic-crop", "frozen-phonons"],
    )
    def test_smatrix_features(self, kwargs_name):
        from abtem.inelastic.xray import SpecimenAbsorption

        num_configs = 2 if kwargs_name == "frozen-phonons" else None
        atoms, potential = self._potential(num_configs=num_configs)

        kwargs = {}
        if kwargs_name == "xray":
            kwargs["detectors"] = abtem.XrayDetector(0.7)
        elif kwargs_name == "absorption":
            kwargs["detectors"] = abtem.XrayDetector(
                0.7, absorption=SpecimenAbsorption("Si")
            )
        elif kwargs_name == "inelastic-crop":
            kwargs["inelastic_crop"] = 2.0

        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20.0,
            interpolation=1,
            downsample=False,
        )
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 3))

        def run(lazy):
            return s_matrix.ionization_scan(
                _transition_potential(
                    potential.extent, potential.gpts, band_limited=True
                ),
                scan=scan,
                sites=atoms,
                lazy=lazy,
                **kwargs,
            )

        self._assert_same(run(False), run(True))

    def test_joint_eels_and_edx(self):
        atoms, potential = self._potential()
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
        probe.grid.match(potential)
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 3))

        def run(lazy):
            return probe.transition_potential_scan(
                potential=potential,
                transition_potentials=_transition_potential(
                    potential.extent, potential.gpts, band_limited=True
                ),
                scan=scan,
                detectors=[
                    abtem.AnnularDetector(0.0, 30.0),
                    abtem.XrayDetector(0.7),
                ],
                double_channel=False,
                lazy=lazy,
                sites=atoms,
            )

        eager, lazy = run(False), run(True)
        assert len(eager) == len(lazy) == 2
        for e, l in zip(eager, lazy):
            self._assert_same(e, l)

    def test_smatrix_lazy_is_actually_lazy(self):
        atoms, potential = self._potential()
        s_matrix = abtem.SMatrix(
            potential=potential,
            energy=ENERGY,
            semiangle_cutoff=20.0,
            interpolation=1,
            downsample=False,
        )
        got = s_matrix.ionization_scan(
            _transition_potential(
                potential.extent, potential.gpts, band_limited=True
            ),
            scan=abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 3)),
            sites=atoms,
            lazy=True,
        )
        assert got.is_lazy
