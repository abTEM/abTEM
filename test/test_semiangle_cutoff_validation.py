"""Semiangle cutoffs that describe no aperture are rejected where they are set.

A negative semiangle cutoff is never meaningful. A zero one is a parallel beam for an
Aperture or CTF (the zero-angle pixel stays open), but leaves nothing open in a
Bullseye, Vortex, AnnularAperture or Zernike aperture, whose probe then normalizes to
NaN, and turns a RadialPhasePlate into a no-op. PRISM's scattering matrices need a
positive cutoff, and a direct-beam radius read from metadata must not be negative.
"""

import inspect

import ase.build
import numpy as np
import pytest

import abtem
from abtem.measurements import DiffractionPatterns
from abtem.prism.s_matrix import CompressedSMatrixArray, SMatrixArray
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


@pytest.mark.parametrize("semiangle_cutoff", [3.0, 5.0])
def test_annular_aperture_setter_rejects_an_empty_annulus(semiangle_cutoff):
    aperture = AnnularAperture(inner_cutoff=5.0, semiangle_cutoff=20.0)

    with pytest.raises(ValueError, match="no open area"):
        aperture.semiangle_cutoff = semiangle_cutoff

    assert aperture.semiangle_cutoff == 20.0
    aperture.semiangle_cutoff = 10.0  # guard: a non-empty annulus is still accepted
    intensity = _probe_intensity(aperture)
    assert np.all(np.isfinite(intensity)) and intensity.sum() > 0


def test_annular_aperture_compares_a_distribution_by_its_smallest_value():
    values = abtem.distributions.from_values
    AnnularAperture(inner_cutoff=5.0, semiangle_cutoff=values([10.0, 20.0]))

    with pytest.raises(ValueError, match="smallest value 3.0"):
        AnnularAperture(inner_cutoff=5.0, semiangle_cutoff=values([3.0, 20.0]))


def test_zernike_rejects_a_negative_center_hole():
    with pytest.raises(ValueError, match="center_hole_cutoff must be non-negative"):
        Zernike(-1.0, np.pi / 2, semiangle_cutoff=30.0)


@pytest.fixture(scope="module")
def potential():
    atoms = ase.build.mx2("WSe2", vacuum=2)
    return abtem.Potential(atoms, sampling=0.1, slice_thickness=2)


def _with_cutoff(obj, cls, semiangle_cutoff):
    """The constructor arguments of `obj`, with another semiangle cutoff."""
    kwargs = {}
    for name in inspect.signature(cls).parameters:
        attribute = name if hasattr(obj, name) else f"_{name}"
        kwargs[name] = getattr(obj, attribute)
    kwargs["semiangle_cutoff"] = semiangle_cutoff
    return kwargs


@pytest.mark.parametrize("semiangle_cutoff", [0.0, -5.0])
def test_s_matrix_arrays_reject_a_non_positive_cutoff(potential, semiangle_cutoff):
    s_matrix = abtem.SMatrix(potential=potential, energy=ENERGY, semiangle_cutoff=20)
    array = s_matrix.build(lazy=False)
    compressed = abtem.SMatrix(
        potential=potential,
        energy=ENERGY,
        semiangle_cutoff=20,
        interpolation=(2, 2),
        upsample=True,
    ).build(lazy=False)
    assert isinstance(array, SMatrixArray)
    assert isinstance(compressed, CompressedSMatrixArray)

    for obj, cls in ((array, SMatrixArray), (compressed, CompressedSMatrixArray)):
        with pytest.raises(ValueError, match="positive 'semiangle_cutoff'"):
            cls(**_with_cutoff(obj, cls, semiangle_cutoff))

        # the same arguments with a positive cutoff construct
        cls(**_with_cutoff(obj, cls, 20.0))


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
