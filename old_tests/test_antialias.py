import numpy as np

from abtem import Waves
from abtem.core.antialias import AntialiasAperture


def test_build_antialias_aperture():
    antialias_aperture = AntialiasAperture(gpts=32, sampling=.1)
    assert np.isclose(antialias_aperture.build().sum(), 347.6798)

    antialias_aperture = AntialiasAperture(gpts=32, sampling=.2)
    assert np.isclose(antialias_aperture.build().sum(), 347.6798)

    antialias_aperture = AntialiasAperture(gpts=(32, 64), sampling=.1)
    assert np.isclose(antialias_aperture.build().sum(), 695.34924)

    antialias_aperture = AntialiasAperture(gpts=(32, 64), sampling=(.2, .1))
    assert np.isclose(antialias_aperture.build().sum(), 347.6798)


def test_bandlimit():
    array = np.zeros((2, 64, 64))
    array[0, 0, 0] = 1

    waves = Waves(array, sampling=.1, energy=80e3, axes_metadata=[{}])

    antialias_aperture = AntialiasAperture()
    antialias_aperture.match_grid(waves)
    waves = antialias_aperture.apply(waves)

    diffraction_patterns = waves.diffraction_patterns(max_angle=None, fftshift=False)
    np.all(np.abs(diffraction_patterns.array[0] - antialias_aperture.array) < .5)
