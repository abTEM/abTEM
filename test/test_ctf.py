import numpy as np
import pytest

from abtem.core.energy import energy2wavelength
from abtem.transfer import Aberrations, Aperture, point_resolution


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
