import numpy as np
import pytest

from abtem.core.energy import energy2wavelength
from abtem.transfer import CTF, Aberrations, Aperture, point_resolution
from abtem.waves import PlaneWave


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
