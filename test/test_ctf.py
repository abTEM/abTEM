import numpy as np
import pytest

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.core.energy import energy2wavelength
from abtem.transfer import (
    CTF,
    Aberrations,
    Aperture,
    Bullseye,
    RadialPhasePlate,
    SpatialEnvelope,
    point_resolution,
    scherzer_defocus,
)
from abtem.waves import PlaneWave, Waves


def test_point_resolution():
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


@pytest.mark.parametrize("gpts", [(64, 64), (65, 64)])
@pytest.mark.parametrize("units", ["1/Å", "mrad"])
def test_aperture_to_diffraction_patterns_is_centred_on_zero_frequency(gpts, units):
    # A round aperture without aberrations is symmetric under k -> -k, so its
    # center of mass is k = 0, and it transmits the zero frequency fully. Both
    # only hold if the pattern's stored order matches its fftshift flag.
    dp = Aperture(
        30.0, energy=100e3, gpts=gpts, sampling=(0.1, 0.1)
    ).to_diffraction_patterns()

    com = complex(dp.center_of_mass(units=units).array)
    # Exact symmetry: only float32 rounding remains; a misordering moves the
    # COM by a sizeable fraction of the 30 mrad aperture.
    scale = max(dp.angular_sampling) if units == "mrad" else max(dp.sampling)
    assert abs(com) < 1e-5 * scale

    kx, ky = dp.coordinates
    zero = (int(np.argmin(np.abs(kx))), int(np.argmin(np.abs(ky))))
    assert np.asarray(dp.array)[zero] == pytest.approx(np.asarray(dp.array).max())


@pytest.mark.parametrize("lazy", [False, True])
def test_apply_ctf_does_not_take_on_the_energy_of_the_waves(lazy):
    # The energy of a CTF without one is inferred from the wave functions. It must
    # not be written into the CTF, or the CTF could not be reused at another energy.
    ctf = CTF(defocus=50, semiangle_cutoff=20)

    for energy in (200e3, 300e3):
        waves = PlaneWave(energy=energy, extent=5, gpts=32).build(lazy=lazy)
        result = waves.apply_ctf(ctf)

        assert ctf.energy is None
        reference = waves.apply_ctf(CTF(energy=energy, defocus=50, semiangle_cutoff=20))
        assert np.allclose(result.compute().array, reference.compute().array)


def test_apply_ctf_with_fixed_energy_still_checks_the_energy_of_the_waves():
    waves = PlaneWave(energy=200e3, extent=5, gpts=32).build()

    with pytest.raises(RuntimeError, match="Inconsistent energies"):
        waves.apply_ctf(CTF(energy=300e3, defocus=50))


def test_scherzer_string_via_C10_symbol():
    # "scherzer" given as C10 must produce the same lens as "scherzer" given as
    # defocus, since defocus = -C10: a positive defocus, with the focus moved
    # into the cell.
    ab = Aberrations(energy=200e3, Cs=1e7, C10="scherzer")
    assert np.isclose(ab.C10, -scherzer_defocus(1e7, 200e3), rtol=1e-6)
    assert np.isclose(ab.defocus, scherzer_defocus(1e7, 200e3), rtol=1e-6)


def test_scherzer_string_independent_of_keyword_order():
    # Scherzer defocus depends on Cs, so it must not matter whether Cs is given
    # before or after defocus="scherzer".
    expected = scherzer_defocus(1e7, 200e3)
    ab = Aberrations(energy=200e3, defocus="scherzer", Cs=1e7)
    assert np.isclose(ab.defocus, expected)
    ctf = CTF(energy=200e3, aberration_coefficients={"defocus": "scherzer", "C30": 1e7})
    assert np.isclose(ctf.defocus, expected)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "transform",
    [
        Bullseye(4, 3.0, 3, 0.5, 20),
        RadialPhasePlate(2, 30.0),
        SpatialEnvelope(angular_spread=2, C30=1e4),
    ],
    ids=["bullseye", "radial_phase_plate", "spatial_envelope"],
)
def test_single_precision_transforms_agree_with_double_precision(transform, lazy):
    # Waves transformed in single precision stay in single precision and equal those
    # transformed in double precision to rounding: in particular the phase flips of
    # the phase plate must not depend on the precision. A flipped phase or a
    # misplaced ring changes the waves by order one.
    import dask.array as da

    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 16, 20)) + 1j * rng.normal(size=(2, 16, 20))
    result = {}
    for precision, dtype in (("float32", np.complex64), ("float64", np.complex128)):
        with abtem.config.set({"precision": precision}):
            array = data.astype(dtype)
            if lazy:
                array = da.from_array(array, chunks=(1, 16, 20))
            waves = Waves(
                array,
                energy=100e3,
                sampling=0.1,
                ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
            ).apply_transform(transform)
            assert waves.array.dtype == dtype
            result[precision] = waves.compute().array
            assert result[precision].dtype == dtype

    np.testing.assert_allclose(
        result["float32"],
        result["float64"],
        rtol=0,
        atol=1e-5 * np.abs(result["float64"]).max(),
    )
