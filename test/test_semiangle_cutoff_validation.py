"""Semiangle cutoffs that describe no aperture are rejected where they are set.

A negative semiangle cutoff is never meaningful. A zero one is a parallel beam for an
Aperture or CTF (the zero-angle pixel stays open), but leaves nothing open in a
Bullseye, Vortex, AnnularAperture or Zernike aperture, whose probe would then
normalize to NaN. A direct-beam radius read from metadata must not be negative.
"""

import numpy as np
import pytest

import abtem
from abtem.measurements import DiffractionPatterns
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
NOTHING_OPEN_AT_ZERO = ("Bullseye", "Vortex", "AnnularAperture", "Zernike")


def _probe_intensity(aperture):
    probe = abtem.Probe(energy=ENERGY, aperture=aperture, extent=10, gpts=64)
    return np.abs(probe.build(lazy=False).array) ** 2


def _not_rejected(names, semiangle_cutoffs, match):
    """The (aperture, cutoff) pairs that construct without the ValueError, or
    with another message."""
    accepted = []
    for name in names:
        for semiangle_cutoff in semiangle_cutoffs:
            try:
                APERTURES[name](semiangle_cutoff)
            except ValueError as error:
                if match not in str(error):
                    accepted.append((name, semiangle_cutoff, str(error)))
            else:
                accepted.append((name, semiangle_cutoff))
    return accepted


def test_a_negative_or_nan_cutoff_is_rejected():
    # One validator for every aperture class, so one test loops over them.
    assert (
        _not_rejected(APERTURES, (-5.0, -np.inf, np.nan), "must be non-negative")
        == []
    )


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


def test_a_zero_cutoff_is_rejected_where_it_describes_no_aperture():
    assert _not_rejected(NOTHING_OPEN_AT_ZERO, (0.0,), "no open area") == []


def test_a_zero_cutoff_is_still_a_parallel_beam():
    not_flat = []
    for name in ("Aperture", "hard Aperture", "CTF"):
        intensity = _probe_intensity(APERTURES[name](0.0))
        if not (
            np.all(np.isfinite(intensity))
            and np.allclose(intensity, intensity.flat[0], rtol=1e-6, atol=0)
        ):
            not_flat.append(name)
    assert not_flat == []

    # A RadialPhasePlate's cutoff is the radius of its phase pattern: at zero
    # the plate is a uniform phase, which is accepted and leaves the probe finite.
    intensity = _probe_intensity(APERTURES["RadialPhasePlate"](0.0))
    assert np.all(np.isfinite(intensity)) and intensity.sum() > 0


def test_a_positive_cutoff_gives_a_normalized_probe():
    """Guards against rejecting valid apertures."""
    not_normalized = []
    for name, make in APERTURES.items():
        intensity = _probe_intensity(make(20.0))
        if not (np.all(np.isfinite(intensity)) and intensity.sum() > 0):
            not_normalized.append(name)
    assert not_normalized == []


def test_annular_aperture_rejects_an_empty_annulus():
    with pytest.raises(ValueError, match="inner_cutoff must be non-negative"):
        AnnularAperture(-1.0, semiangle_cutoff=30.0)

    # an inner cutoff equal to or beyond the outer one
    for inner_cutoff in (30.0, 40.0):
        with pytest.raises(ValueError, match="no open area"):
            AnnularAperture(inner_cutoff, semiangle_cutoff=30.0)


def test_annular_aperture_setter_rejects_an_empty_annulus():
    aperture = AnnularAperture(inner_cutoff=5.0, semiangle_cutoff=20.0)

    for semiangle_cutoff in (3.0, 5.0):
        with pytest.raises(ValueError, match="no open area"):
            aperture.semiangle_cutoff = semiangle_cutoff

    assert aperture.semiangle_cutoff == 20.0
    aperture.semiangle_cutoff = 10.0  # guard: a non-empty annulus is still accepted
    intensity = _probe_intensity(aperture)
    assert np.all(np.isfinite(intensity)) and intensity.sum() > 0


def test_zernike_rejects_a_negative_center_hole():
    with pytest.raises(ValueError, match="center_hole_cutoff must be non-negative"):
        Zernike(-1.0, np.pi / 2, semiangle_cutoff=30.0)


def test_block_direct_rejects_a_negative_radius():
    patterns = DiffractionPatterns(
        np.ones((32, 32), dtype=np.float32),
        sampling=0.1,
        fftshift=True,
        metadata={"energy": ENERGY, "semiangle_cutoff": -5.0},
    )

    with pytest.raises(ValueError, match="radius must be non-negative"):
        patterns.block_direct()

    with pytest.raises(ValueError, match="radius must be non-negative"):
        patterns.block_direct(radius=-1.0)


@pytest.mark.parametrize("cutoff", [np.array(10.0), np.array([10.0])])
def test_block_direct_leaves_an_array_cutoff_in_the_metadata_unchanged(cutoff):
    from abtem.waves import Waves

    rng = np.random.default_rng(0)
    array = (
        rng.standard_normal((100, 100)) + 1j * rng.standard_normal((100, 100))
    ).astype(np.complex64)

    def patterns(semiangle_cutoff):
        waves = Waves(
            array,
            energy=200e3,
            sampling=0.5,
            metadata={"semiangle_cutoff": semiangle_cutoff},
        )
        return waves, waves.diffraction_patterns()

    expected = patterns(10.0)[1].block_direct().array
    waves, diffraction_patterns = patterns(cutoff)

    for _ in range(3):
        np.testing.assert_array_equal(diffraction_patterns.block_direct().array, expected)

    assert np.asarray(waves.metadata["semiangle_cutoff"]) == 10.0
    assert np.asarray(diffraction_patterns.metadata["semiangle_cutoff"]) == 10.0
