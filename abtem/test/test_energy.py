import numpy as np

from ..bases import energy2mass, energy2wavelength, energy2sigma, Energy


def test_energy2mass():
    assert np.isclose(energy2mass(300e3), 1.445736928082275e-30)


def test_energy2wavelength():
    assert np.isclose(energy2wavelength(300e3), 0.01968748889772767)


def test_energy2sigma():
    assert np.isclose(energy2sigma(300e3), 0.0006526161464700888)


def test_energy():
    energy = Energy(energy=300e3)

    assert energy.energy == 300e3
    assert energy.wavelength == energy2wavelength(300e3)

    energy.energy = 200e3
    assert energy.wavelength == energy2wavelength(200e3)

    energy1 = Energy(300e3)
    energy2 = Energy()

    energy1.match_energy(energy2)

    assert np.all(energy1.energy == energy2.energy)

    energy1.energy = 200e3
    energy2 = Energy()

    energy2.match_energy(energy1)

    assert np.all(energy1.energy == energy2.energy)