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
    hard_aperture,
    nyquist_sampling,
    point_resolution,
    polar2cartesian,
    scherzer_defocus,
    soft_aperture,
)

ENERGY = 100e3  # eV
GPTS = (64, 64)
SAMPLING = (0.1, 0.1)  # Å/pixel


# ---------------------------------------------------------------------------
# Factory functions for parametrized kernel tests
# ---------------------------------------------------------------------------

def _aperture():  return Aperture(30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
def _annular():   return AnnularAperture(10.0, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
def _vortex():    return Vortex(1, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
def _bullseye():  return Bullseye(4, 5.0, 3, 2.0, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
def _zernike():   return Zernike(2.0, np.pi / 2, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
def _rpp():       return RadialPhasePlate(2, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
def _temporal():  return TemporalEnvelope(100.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
def _spatial():   return SpatialEnvelope(0.5, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
def _ctf():       return CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING, defocus=100.0)


ALL_KERNELS = pytest.mark.parametrize("factory", [
    pytest.param(_aperture, id="Aperture"),
    pytest.param(_annular,  id="AnnularAperture"),
    pytest.param(_vortex,   id="Vortex"),
    pytest.param(_bullseye, id="Bullseye"),
    pytest.param(_zernike,  id="Zernike"),
    pytest.param(_rpp,      id="RadialPhasePlate"),
    pytest.param(_temporal, id="TemporalEnvelope"),
    pytest.param(_spatial,  id="SpatialEnvelope"),
    pytest.param(_ctf,      id="CTF"),
])

REAL_KERNELS = pytest.mark.parametrize("factory", [
    pytest.param(_aperture, id="Aperture"),
    pytest.param(_annular,  id="AnnularAperture"),
    pytest.param(_bullseye, id="Bullseye"),
    pytest.param(_temporal, id="TemporalEnvelope"),
    pytest.param(_spatial,  id="SpatialEnvelope"),
])

COMPLEX_KERNELS = pytest.mark.parametrize("factory", [
    pytest.param(_vortex,   id="Vortex"),
    pytest.param(_zernike,  id="Zernike"),
    pytest.param(_rpp,      id="RadialPhasePlate"),
    pytest.param(_ctf,      id="CTF"),
])


# ---------------------------------------------------------------------------
# Common kernel property tests (parametrized across all transfer functions)
# ---------------------------------------------------------------------------

@ALL_KERNELS
def test_kernel_shape(factory):
    assert factory()._evaluate_kernel().shape == GPTS


@ALL_KERNELS
def test_kernel_finite(factory):
    assert np.all(np.isfinite(factory()._evaluate_kernel()))


@REAL_KERNELS
def test_kernel_real_in_unit_interval(factory):
    k = factory()._evaluate_kernel()
    assert not np.iscomplexobj(k)
    assert np.all(k >= 0.0) and np.all(k <= 1.0 + 1e-10)


@COMPLEX_KERNELS
def test_kernel_complex_amplitude_leq_1(factory):
    k = factory()._evaluate_kernel()
    assert np.iscomplexobj(k)
    assert np.all(np.abs(k) <= 1.0 + 1e-6)  # float32 tolerance


# ---------------------------------------------------------------------------
# Utility functions — Hypothesis for monotonicity / sign properties
# ---------------------------------------------------------------------------

@given(
    semiangle=st.floats(5.0, 100.0),
    energy=st.floats(60e3, 300e3),
)
def test_nyquist_sampling_positive(semiangle, energy):
    assert nyquist_sampling(semiangle, energy) > 0


@given(Cs=st.floats(1e5, 1e8))
def test_scherzer_defocus_sign(Cs):
    assert scherzer_defocus(Cs, ENERGY) > 0
    assert scherzer_defocus(-Cs, ENERGY) < 0


def test_scherzer_defocus_formula():
    Cs = 1e7
    wl = energy2wavelength(ENERGY)
    assert np.isclose(scherzer_defocus(Cs, ENERGY), np.sqrt(1.5 * Cs * wl), rtol=1e-6)


@given(
    Cs_lo=st.floats(1e6, 5e6),
    Cs_hi=st.floats(5.1e6, 1e8),
)
def test_point_resolution_monotone_in_Cs(Cs_lo, Cs_hi):
    assert point_resolution(Cs_lo, ENERGY) < point_resolution(Cs_hi, ENERGY)


def test_polar2cartesian():
    result = polar2cartesian({"C10": 200.0, "C12": 0.0, "phi12": 0.0})
    assert isinstance(result, dict)
    assert np.isclose(result["C10"], 200.0)
    assert np.isclose(result.get("C12a", 0.0), 0.0) and np.isclose(result.get("C12b", 0.0), 0.0)


def test_cartesian2polar():
    result = cartesian2polar({"C10": 300.0})
    assert isinstance(result, dict) and np.isclose(result["C10"], 300.0)


# ---------------------------------------------------------------------------
# soft_aperture / hard_aperture
# ---------------------------------------------------------------------------

class TestSoftHardAperture:
    def _grid(self, n=32):
        alpha = np.linspace(0, 50e-3, n * n).reshape(n, n)
        return alpha, np.zeros_like(alpha)

    def test_soft_aperture_in_0_1(self):
        alpha, phi = self._grid()
        r = soft_aperture(alpha, phi, 30e-3, (1.0, 1.0))
        assert np.all(r >= 0.0) and np.all(r <= 1.0)

    def test_hard_aperture_binary(self):
        r = hard_aperture(self._grid()[0], 30e-3)
        assert set(np.unique(r)).issubset({0.0, 1.0})

    def test_hard_aperture_dc_is_one(self):
        assert np.all(hard_aperture(np.zeros((8, 8)), 30e-3) == 1.0)


# ---------------------------------------------------------------------------
# Aperture-specific attribute / edge-case tests
# ---------------------------------------------------------------------------

def test_aperture_attributes():
    ap = Aperture(30.0, soft=True, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
    assert ap.semiangle_cutoff == 30.0 and ap.soft is True


def test_aperture_infinite_cutoff_gives_ones():
    assert np.all(
        Aperture(np.inf, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)._evaluate_kernel() == 1.0
    )


def test_aperture_nyquist_sampling():
    assert Aperture(30.0, energy=ENERGY).nyquist_sampling > 0


def test_aperture_to_diffraction_patterns():
    from abtem.measurements import DiffractionPatterns
    dp = Aperture(30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING).to_diffraction_patterns()
    assert isinstance(dp, DiffractionPatterns)


def test_aperture_distribution_semiangle():
    from abtem.distributions import from_values
    k = Aperture(
        from_values([20.0, 30.0, 40.0]), energy=ENERGY, gpts=GPTS, sampling=SAMPLING
    )._evaluate_kernel()
    assert k.shape[0] == 3


def test_annular_aperture_attributes_and_dc():
    ap = AnnularAperture(10.0, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
    assert ap.inner_cutoff == 10.0 and ap.semiangle_cutoff == 30.0
    assert AnnularAperture(5.0, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)._evaluate_kernel()[0, 0] == 0.0


def test_vortex_quantum_number():
    assert Vortex(1, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING).quantum_number == 1


def test_bullseye_attributes():
    b = Bullseye(4, 5.0, 3, 2.0, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
    assert b.num_spokes == 4 and b.num_rings == 3 and b.soft is False


def test_zernike_attributes():
    z = Zernike(2.0, np.pi / 2, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
    assert z.center_hole_cutoff == 2.0 and z.phase_shift == np.pi / 2


def test_rpp_attributes_and_unit_amplitude():
    rpp = RadialPhasePlate(2, 30.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
    assert rpp.num_flips == 2
    assert np.allclose(np.abs(rpp._evaluate_kernel()), 1.0)


def test_temporal_envelope_attributes_and_limits():
    te = TemporalEnvelope(100.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
    assert te.focal_spread == 100.0
    assert np.allclose(
        TemporalEnvelope(0.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)._evaluate_kernel(), 1.0
    )
    assert np.isclose(
        TemporalEnvelope(500.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)._evaluate_kernel()[0, 0], 1.0
    )


def test_spatial_envelope_zero_spread_gives_ones():
    assert np.allclose(
        SpatialEnvelope(0.0, energy=ENERGY, gpts=GPTS, sampling=SAMPLING)._evaluate_kernel(), 1.0
    )


# ---------------------------------------------------------------------------
# Aberrations
# ---------------------------------------------------------------------------

class TestAberrations:
    def test_defocus_alias(self):
        ab = Aberrations(energy=ENERGY, defocus=100.0)
        assert ab.defocus == 100.0 and np.isclose(ab.C10, -100.0)

    def test_set_aberrations_dict(self):
        ab = Aberrations(energy=ENERGY)
        ab.set_aberrations({"C10": -200.0, "C30": 1e7})
        assert np.isclose(ab.defocus, 200.0)

    def test_scherzer_string(self):
        ab = Aberrations(energy=ENERGY, Cs=1e7)
        ab.set_aberrations({"defocus": "scherzer"})
        assert np.isclose(ab.defocus, scherzer_defocus(1e7, ENERGY), rtol=1e-5)

    def test_no_aberrations_unit_phase(self):
        k = Aberrations(energy=ENERGY, gpts=GPTS, sampling=SAMPLING)._evaluate_kernel()
        assert np.allclose(np.abs(k), 1.0)


# ---------------------------------------------------------------------------
# CTF
# ---------------------------------------------------------------------------

class TestCTF:
    def test_defaults(self):
        ctf = CTF(energy=ENERGY)
        assert ctf.defocus == 0.0 and ctf.focal_spread == 0.0

    def test_aperture_cuts_high_angles(self):
        open_ = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING)
        cut = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING, semiangle_cutoff=5.0)
        assert np.sum(np.abs(open_._evaluate_kernel())) > np.sum(np.abs(cut._evaluate_kernel()))

    def test_focal_spread_attenuates(self):
        nofs = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING, focal_spread=0.0)
        fs = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING, focal_spread=500.0)
        assert np.sum(np.abs(nofs._evaluate_kernel())) > np.sum(np.abs(fs._evaluate_kernel()))

    def test_to_diffraction_patterns(self):
        from abtem.measurements import DiffractionPatterns
        dp = CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING).to_diffraction_patterns()
        assert isinstance(dp, DiffractionPatterns)

    def test_profiles(self):
        from abtem.measurements import ReciprocalSpaceLineProfiles
        assert isinstance(
            CTF(energy=ENERGY, gpts=GPTS, sampling=SAMPLING, defocus=200.0).profiles(),
            ReciprocalSpaceLineProfiles,
        )

    def test_apply_to_waves(self):
        from abtem.waves import PlaneWave
        ctf = CTF(energy=ENERGY, semiangle_cutoff=30.0, defocus=100.0)
        wave = PlaneWave(energy=ENERGY, gpts=GPTS, sampling=SAMPLING).build(lazy=False)
        assert wave.apply_transform(ctf).array.shape == wave.array.shape

    @settings(max_examples=5)
    @given(defocus=st.floats(-500, 500), Cs=st.floats(0, 1e8))
    def test_hypothesis_evaluate(self, defocus, Cs):
        k = CTF(
            energy=ENERGY, gpts=(32, 32), sampling=SAMPLING, defocus=defocus, Cs=Cs
        )._evaluate_kernel()
        assert k.shape == (32, 32) and np.all(np.isfinite(k))


# ---------------------------------------------------------------------------
# Existing regression test
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

    assert np.round(numerical_point_resolution1, 1) == np.round(analytical_point_resolution, 1)
