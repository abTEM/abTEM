"""Tests for abtem/transfer.py — CTF, apertures, envelopes, and utilities."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from abtem.core.energy import energy2wavelength
from abtem.transfer import (
    Aberrations,
    AnnularAperture,
    Aperture,
    Bullseye,
    CTF,
    RadialPhasePlate,
    SpatialEnvelope,
    TemporalEnvelope,
    Vortex,
    Zernike,
    cartesian2polar,
    nyquist_sampling,
    point_resolution,
    polar2cartesian,
    polar_aliases,
    scherzer_defocus,
    soft_aperture,
    hard_aperture,
)

ENERGY = 100e3  # eV
GPTS = (64, 64)
SAMPLING = (0.1, 0.1)  # Å/pixel


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestNyquistSampling:
    def test_positive(self):
        s = nyquist_sampling(semiangle_cutoff=30.0, energy=ENERGY)
        assert s > 0

    def test_smaller_cutoff_gives_larger_sampling(self):
        s1 = nyquist_sampling(10.0, ENERGY)
        s2 = nyquist_sampling(30.0, ENERGY)
        assert s1 > s2

    def test_higher_energy_smaller_sampling(self):
        s1 = nyquist_sampling(30.0, 100e3)
        s2 = nyquist_sampling(30.0, 300e3)
        assert s2 < s1


class TestScherzerDefocus:
    def test_positive_Cs_gives_positive_defocus(self):
        Cs = 1e7  # Å
        df = scherzer_defocus(Cs, ENERGY)
        assert df > 0

    def test_negative_Cs_gives_negative_defocus(self):
        df = scherzer_defocus(-1e7, ENERGY)
        assert df < 0

    def test_formula(self):
        Cs = 1e7
        wl = energy2wavelength(ENERGY)
        expected = np.sqrt(1.5 * Cs * wl)
        assert np.isclose(scherzer_defocus(Cs, ENERGY), expected, rtol=1e-6)


class TestPointResolution:
    def test_positive(self):
        assert point_resolution(1e7, ENERGY) > 0

    def test_larger_Cs_lower_resolution(self):
        r1 = point_resolution(1e6, ENERGY)
        r2 = point_resolution(1e8, ENERGY)
        assert r2 > r1

    def test_higher_energy_better_resolution(self):
        r1 = point_resolution(1e7, 100e3)
        r2 = point_resolution(1e7, 300e3)
        assert r2 < r1


class TestPolar2Cartesian:
    def test_returns_dict(self):
        result = polar2cartesian({"C10": 100.0, "C30": 1e7})
        assert isinstance(result, dict)

    def test_C10_preserved(self):
        result = polar2cartesian({"C10": 200.0})
        assert np.isclose(result["C10"], 200.0)

    def test_C30_preserved(self):
        result = polar2cartesian({"C10": 0.0, "C30": 1e7})
        assert np.isclose(result["C30"], 1e7)

    def test_zero_astigmatism(self):
        result = polar2cartesian({"C12": 0.0, "phi12": 0.0})
        assert np.isclose(result["C12a"], 0.0)
        assert np.isclose(result["C12b"], 0.0)


class TestCartesian2Polar:
    def test_returns_dict(self):
        result = cartesian2polar({"C10": 100.0})
        assert isinstance(result, dict)

    def test_C10_preserved(self):
        result = cartesian2polar({"C10": 300.0})
        assert np.isclose(result["C10"], 300.0)


# ---------------------------------------------------------------------------
# soft_aperture / hard_aperture
# ---------------------------------------------------------------------------

class TestSoftHardAperture:
    def _grid(self, n=32):
        alpha = np.linspace(0, 50e-3, n * n).reshape(n, n)
        phi = np.zeros_like(alpha)
        return alpha, phi

    def test_soft_aperture_values_between_0_and_1(self):
        alpha, phi = self._grid()
        result = soft_aperture(alpha, phi, 30e-3, (1.0, 1.0))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_hard_aperture_binary(self):
        alpha, phi = self._grid()
        result = hard_aperture(alpha, 30e-3)
        unique = np.unique(result)
        assert set(unique).issubset({0.0, 1.0})

    def test_hard_aperture_dc_is_one(self):
        alpha = np.zeros((8, 8))
        result = hard_aperture(alpha, 30e-3)
        assert np.all(result == 1.0)


# ---------------------------------------------------------------------------
# Aperture
# ---------------------------------------------------------------------------

class TestAperture:
    def test_basic_construction(self):
        ap = Aperture(semiangle_cutoff=30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        assert ap.semiangle_cutoff == 30.0

    def test_soft_property(self):
        ap = Aperture(30.0, soft=True, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        assert ap.soft is True
        ap2 = Aperture(30.0, soft=False, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        assert ap2.soft is False

    def test_evaluate_kernel_shape(self):
        ap = Aperture(30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = ap._evaluate_kernel()
        assert kernel.shape == GPTS

    def test_evaluate_values_between_0_and_1(self):
        ap = Aperture(30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = ap._evaluate_kernel()
        assert np.all(kernel >= 0.0)
        assert np.all(kernel <= 1.0)

    def test_infinite_cutoff_gives_all_ones(self):
        ap = Aperture(np.inf, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = ap._evaluate_kernel()
        assert np.all(kernel == 1.0)

    def test_nyquist_sampling(self):
        ap = Aperture(30.0, energy=ENERGY)
        s = ap.nyquist_sampling
        assert s > 0

    def test_to_diffraction_patterns(self):
        from abtem.measurements import DiffractionPatterns

        ap = Aperture(30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        dp = ap.to_diffraction_patterns()
        assert isinstance(dp, DiffractionPatterns)

    def test_distribution_semiangle(self):
        from abtem.distributions import from_values

        cutoffs = from_values([20.0, 30.0, 40.0])
        ap = Aperture(cutoffs, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = ap._evaluate_kernel()
        assert kernel.shape[0] == 3


# ---------------------------------------------------------------------------
# AnnularAperture
# ---------------------------------------------------------------------------

class TestAnnularAperture:
    def test_construction(self):
        ap = AnnularAperture(10.0, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        assert ap.inner_cutoff == 10.0
        assert ap.semiangle_cutoff == 30.0

    def test_evaluate_binary(self):
        ap = AnnularAperture(10.0, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = ap._evaluate_kernel()
        unique = np.unique(kernel)
        assert set(unique).issubset({0.0, 1.0})

    def test_dc_is_zero(self):
        """DC component (alpha=0) should be blocked by the inner cutoff."""
        ap = AnnularAperture(5.0, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = ap._evaluate_kernel()
        # The very centre should be 0 (blocked by inner cutoff)
        assert kernel[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Vortex
# ---------------------------------------------------------------------------

class TestVortex:
    def test_construction(self):
        v = Vortex(1, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        assert v.quantum_number == 1

    def test_evaluate_is_complex(self):
        v = Vortex(1, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = v._evaluate_kernel()
        assert np.iscomplexobj(kernel)

    def test_amplitude_binary(self):
        v = Vortex(1, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = v._evaluate_kernel()
        amp = np.abs(kernel)
        unique = np.unique(np.round(amp, 6))
        assert set(unique).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Bullseye
# ---------------------------------------------------------------------------

class TestBullseye:
    def test_construction(self):
        b = Bullseye(
            num_spokes=4, spoke_width=5.0, num_rings=3, ring_width=2.0,
            semiangle_cutoff=30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING,
        )
        assert b.num_spokes == 4
        assert b.num_rings == 3

    def test_evaluate_kernel_shape(self):
        b = Bullseye(4, 5.0, 3, 2.0, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = b._evaluate_kernel()
        assert kernel.shape == GPTS

    def test_soft_is_false(self):
        b = Bullseye(4, 5.0, 3, 2.0, 30.0, energy=ENERGY)
        assert b.soft is False


# ---------------------------------------------------------------------------
# Zernike
# ---------------------------------------------------------------------------

class TestZernike:
    def test_construction(self):
        z = Zernike(2.0, np.pi / 2, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        assert z.center_hole_cutoff == 2.0
        assert z.phase_shift == np.pi / 2

    def test_evaluate_is_complex(self):
        z = Zernike(2.0, np.pi / 2, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = z._evaluate_kernel()
        assert np.iscomplexobj(kernel)

    def test_amplitude_leq_1(self):
        z = Zernike(2.0, np.pi / 2, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = z._evaluate_kernel()
        assert np.all(np.abs(kernel) <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# RadialPhasePlate
# ---------------------------------------------------------------------------

class TestRadialPhasePlate:
    def test_construction(self):
        rpp = RadialPhasePlate(2, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        assert rpp.num_flips == 2

    def test_evaluate_is_complex(self):
        rpp = RadialPhasePlate(2, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = rpp._evaluate_kernel()
        assert np.iscomplexobj(kernel)

    def test_amplitude_is_one(self):
        rpp = RadialPhasePlate(2, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = rpp._evaluate_kernel()
        assert np.allclose(np.abs(kernel), 1.0)


# ---------------------------------------------------------------------------
# TemporalEnvelope
# ---------------------------------------------------------------------------

class TestTemporalEnvelope:
    def test_construction(self):
        te = TemporalEnvelope(100.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        assert te.focal_spread == 100.0

    def test_evaluate_real_valued(self):
        te = TemporalEnvelope(100.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = te._evaluate_kernel()
        assert not np.iscomplexobj(kernel)

    def test_values_between_0_and_1(self):
        te = TemporalEnvelope(100.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = te._evaluate_kernel()
        assert np.all(kernel >= 0.0)
        assert np.all(kernel <= 1.0)

    def test_zero_focal_spread_gives_ones(self):
        te = TemporalEnvelope(0.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = te._evaluate_kernel()
        assert np.allclose(kernel, 1.0)

    def test_dc_is_one(self):
        """DC component (alpha=0) always has envelope = 1."""
        te = TemporalEnvelope(500.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = te._evaluate_kernel()
        assert np.isclose(kernel[0, 0], 1.0)


# ---------------------------------------------------------------------------
# Aberrations
# ---------------------------------------------------------------------------

class TestAberrations:
    def test_construction_defocus(self):
        ab = Aberrations(energy=ENERGY, defocus=100.0)
        assert ab.defocus == 100.0
        assert np.isclose(ab.C10, -100.0)

    def test_construction_Cs(self):
        ab = Aberrations(energy=ENERGY, Cs=1e7)
        assert ab.Cs == 1e7

    def test_set_aberrations_dict(self):
        ab = Aberrations(energy=ENERGY)
        ab.set_aberrations({"C10": -200.0, "C30": 1e7})
        assert np.isclose(ab.defocus, 200.0)

    def test_scherzer_defocus_string(self):
        ab = Aberrations(energy=ENERGY, Cs=1e7)
        ab.set_aberrations({"defocus": "scherzer"})
        expected = scherzer_defocus(1e7, ENERGY)
        assert np.isclose(ab.defocus, expected, rtol=1e-5)

    def test_aberration_coefficients_dict(self):
        ab = Aberrations(energy=ENERGY, defocus=100.0, Cs=1e6)
        coeffs = ab.aberration_coefficients
        assert isinstance(coeffs, dict)
        assert "C10" in coeffs

    def test_polar_aliases(self):
        ab = Aberrations(energy=ENERGY)
        ab.astigmatism = 50.0
        assert ab.C12 == 50.0

    def test_evaluate_from_angular_grid(self):
        ab = Aberrations(energy=ENERGY, gpts=GPTS, sampling=SAMPLING, defocus=100.0)
        kernel = ab._evaluate_kernel()
        assert kernel.shape == GPTS
        assert np.iscomplexobj(kernel)

    def test_no_aberrations_gives_unit_phase(self):
        ab = Aberrations(energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = ab._evaluate_kernel()
        assert np.allclose(np.abs(kernel), 1.0)


# ---------------------------------------------------------------------------
# CTF
# ---------------------------------------------------------------------------

class TestCTF:
    def test_construction_defaults(self):
        ctf = CTF(energy=ENERGY)
        assert ctf.defocus == 0.0
        assert ctf.focal_spread == 0.0

    def test_construction_with_aberrations(self):
        ctf = CTF(energy=ENERGY, defocus=500.0, Cs=1e7)
        assert ctf.defocus == 500.0
        assert ctf.Cs == 1e7

    def test_evaluate_kernel_shape(self):
        ctf = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = ctf._evaluate_kernel()
        assert kernel.shape == GPTS

    def test_evaluate_kernel_complex(self):
        ctf = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING, defocus=100.0)
        kernel = ctf._evaluate_kernel()
        assert np.iscomplexobj(kernel)

    def test_no_aberrations_dc_is_one(self):
        ctf = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        kernel = ctf._evaluate_kernel()
        assert np.isclose(np.abs(kernel[0, 0]), 1.0)

    def test_aperture_cuts_high_angles(self):
        ctf_open = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        ctf_cut = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING, semiangle_cutoff=5.0)
        open_sum = np.sum(np.abs(ctf_open._evaluate_kernel()))
        cut_sum = np.sum(np.abs(ctf_cut._evaluate_kernel()))
        assert open_sum > cut_sum

    def test_focal_spread_attenuates_high_angles(self):
        ctf_nofs = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING, focal_spread=0.0)
        ctf_fs = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING, focal_spread=500.0)
        nofs_sum = np.sum(np.abs(ctf_nofs._evaluate_kernel()))
        fs_sum = np.sum(np.abs(ctf_fs._evaluate_kernel()))
        assert nofs_sum > fs_sum

    def test_to_diffraction_patterns(self):
        from abtem.measurements import DiffractionPatterns

        ctf = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        dp = ctf.to_diffraction_patterns()
        assert isinstance(dp, DiffractionPatterns)

    def test_profiles(self):
        from abtem.measurements import ReciprocalSpaceLineProfiles

        ctf = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING, defocus=200.0)
        profiles = ctf.profiles()
        assert isinstance(profiles, ReciprocalSpaceLineProfiles)

    def test_apply_to_waves(self):
        from abtem.waves import PlaneWave

        ctf = CTF(energy=ENERGY, semiangle_cutoff=30.0, defocus=100.0)
        wave = PlaneWave(energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        wave = wave.build(lazy=False)
        result = wave.apply_transform(ctf)
        assert result.array.shape == wave.array.shape

    @settings(max_examples=5)
    @given(defocus=st.floats(-500, 500), Cs=st.floats(0, 1e8))
    def test_hypothesis_evaluate(self, defocus, Cs):
        ctf = CTF(energy=ENERGY, gpts=(32, 32), sampling=SAMPLING,
                  defocus=defocus, Cs=Cs)
        kernel = ctf._evaluate_kernel()
        assert kernel.shape == (32, 32)
        assert np.all(np.isfinite(kernel))


# ---------------------------------------------------------------------------
# SpatialEnvelope
# ---------------------------------------------------------------------------

class TestSpatialEnvelope:
    def test_construction(self):
        se = SpatialEnvelope(angular_spread=0.5, energy=ENERGY,
                             gpts=GPTS, sampling=SAMPLING)
        assert se.angular_spread == 0.5

    def test_zero_spread_gives_ones(self):
        se = SpatialEnvelope(angular_spread=0.0, energy=ENERGY,
                             gpts=GPTS, sampling=SAMPLING)
        kernel = se._evaluate_kernel()
        assert np.allclose(kernel, 1.0)

    def test_values_between_0_and_1(self):
        se = SpatialEnvelope(angular_spread=1.0, energy=ENERGY,
                             gpts=GPTS, sampling=SAMPLING)
        kernel = se._evaluate_kernel()
        assert np.all(kernel >= 0.0 - 1e-10)
        assert np.all(kernel <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Existing test (kept for regression)
# ---------------------------------------------------------------------------

def test_point_resolution_regression():
    ctf = Aberrations(energy=200e3, Cs=1e-3 * 1e10, defocus=600)

    max_semiangle = 20
    n = 1e3
    sampling = max_semiangle / 1000.0 / n
    alpha = np.arange(0, max_semiangle / 1000.0, sampling)

    aberrations = ctf._evaluate_from_angular_grid(alpha, 0.0)

    zero_crossings = np.where(np.diff(np.sign(aberrations.imag)))[0]
    numerical_point_resolution1 = 1 / (
        zero_crossings[1] * alpha[1] / energy2wavelength(ctf.energy)
    )
    analytical_point_resolution = point_resolution(energy=200e3, Cs=1e-3 * 1e10)

    assert np.round(numerical_point_resolution1, 1) == np.round(
        analytical_point_resolution, 1
    )
