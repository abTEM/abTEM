import numpy as np

from abtem.core.energy import energy2wavelength
from abtem.waves.transfer import Aberrations
from abtem.waves.transfer import point_resolution


def test_point_resolution():
    ctf = Aberrations(energy=200e3, Cs=1e-3 * 1e10, defocus=600)

    max_semiangle = 20
    n = 1e3
    sampling = max_semiangle / 1000. / n
    alpha = np.arange(0, max_semiangle / 1000., sampling)

    aberrations = ctf.evaluate_with_alpha_and_phi(alpha, 0.)

    zero_crossings = np.where(np.diff(np.sign(aberrations.imag)))[0]
    numerical_point_resolution1 = 1 / (zero_crossings[1] * alpha[1] / energy2wavelength(ctf.energy))
    analytical_point_resolution = point_resolution(energy=200e3, Cs=1e-3 * 1e10)

    assert np.round(numerical_point_resolution1, 1) == np.round(analytical_point_resolution, 1)
