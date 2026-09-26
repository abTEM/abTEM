"""Aperture cutoffs that describe no aperture are rejected where they are set.

A negative semiangle cutoff is never meaningful. A zero one is a parallel beam for an
Aperture or CTF (the zero-angle pixel stays open), but leaves nothing open in a
Bullseye, Vortex, AnnularAperture or Zernike aperture, whose probe then normalizes to
NaN, and turns a RadialPhasePlate into a no-op.
"""

import numpy as np
import pytest

import abtem
from abtem.transfer import (
    CTF,
    AnnularAperture,
    Aperture,
    Bullseye,
    RadialPhasePlate,
    Vortex,
    Zernike,
    nyquist_sampling,
)

ENERGY = 60e3

APERTURES = {
    "Aperture": lambda c: Aperture(semiangle_cutoff=c),
    "hard Aperture": lambda c: Aperture(semiangle_cutoff=c, soft=False),
    "CTF": lambda c: CTF(semiangle_cutoff=c, defocus=50),
    "Bullseye": lambda c: Bullseye(4, 0.1, 3, 0.5, semiangle_cutoff=c),
    "Vortex": lambda c: Vortex(1, semiangle_cutoff=c),
    "AnnularAperture": lambda c: AnnularAperture(0.0, semiangle_cutoff=c),
    "Zernike": lambda c: Zernike(0.5, np.pi / 2, semiangle_cutoff=c),
    "RadialPhasePlate": lambda c: RadialPhasePlate(2, semiangle_cutoff=c),
}
NOTHING_OPEN_AT_ZERO = (
    "Bullseye",
    "Vortex",
    "AnnularAperture",
    "Zernike",
    "RadialPhasePlate",
)


def _probe_intensity(aperture):
    probe = abtem.Probe(energy=ENERGY, aperture=aperture, extent=10, gpts=64)
    return np.abs(probe.build(lazy=False).array) ** 2


@pytest.mark.parametrize("name", list(APERTURES))
@pytest.mark.parametrize("semiangle_cutoff", [-5.0, -np.inf, np.nan])
def test_a_negative_or_nan_cutoff_is_rejected(name, semiangle_cutoff):
    with pytest.raises(ValueError, match="must be non-negative"):
        APERTURES[name](semiangle_cutoff)


@pytest.mark.parametrize("make", [lambda: Aperture(20.0), lambda: CTF(20.0)])
def test_a_negative_cutoff_is_rejected_by_the_setter(make):
    aperture = make()

    with pytest.raises(ValueError, match="must be non-negative"):
        aperture.semiangle_cutoff = -1.0

    assert aperture.semiangle_cutoff == 20.0


def test_a_negative_cutoff_is_rejected_by_probe_and_distributions():
    with pytest.raises(ValueError, match="must be non-negative"):
        abtem.Probe(energy=ENERGY, semiangle_cutoff=-5.0)

    with pytest.raises(ValueError, match=r"values \[-5.0, 10.0\]"):
        Aperture(semiangle_cutoff=abtem.distributions.from_values([-5.0, 10.0]))

    with pytest.raises(ValueError, match="must be positive"):
        nyquist_sampling(-5.0, ENERGY)


@pytest.mark.parametrize("name", NOTHING_OPEN_AT_ZERO)
def test_a_zero_cutoff_is_rejected_where_it_describes_no_aperture(name):
    with pytest.raises(ValueError, match="semiangle_cutoff=0"):
        APERTURES[name](0.0)


@pytest.mark.parametrize("name", ["Aperture", "hard Aperture", "CTF"])
def test_a_zero_cutoff_is_still_a_parallel_beam(name):
    intensity = _probe_intensity(APERTURES[name](0.0))

    assert np.all(np.isfinite(intensity))
    np.testing.assert_allclose(intensity, intensity.flat[0], rtol=1e-6)


@pytest.mark.parametrize("name", list(APERTURES))
def test_a_positive_cutoff_gives_a_normalized_probe(name):
    """Guards against rejecting valid apertures."""
    intensity = _probe_intensity(APERTURES[name](20.0))

    assert np.all(np.isfinite(intensity))
    assert intensity.sum() > 0


@pytest.mark.parametrize(
    "inner_cutoff, match",
    [
        (-1.0, "inner_cutoff must be non-negative"),
        (30.0, "no open area"),
        (40.0, "no open area"),
    ],
)
def test_annular_aperture_rejects_an_empty_annulus(inner_cutoff, match):
    with pytest.raises(ValueError, match=match):
        AnnularAperture(inner_cutoff, semiangle_cutoff=30.0)


def test_zernike_rejects_a_negative_center_hole():
    with pytest.raises(ValueError, match="center_hole_cutoff must be non-negative"):
        Zernike(-1.0, np.pi / 2, semiangle_cutoff=30.0)
