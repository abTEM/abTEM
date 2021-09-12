import numpy as np
import pytest

from abtem.base_classes import Accelerator
from abtem.utils import energy2mass, energy2wavelength, energy2sigma


def test_energy2mass():
    assert np.isclose(energy2mass(300e3), 1.445736928082275e-30)


def test_energy2wavelength():
    assert np.isclose(energy2wavelength(300e3), 0.01968748889772767)


def test_energy2sigma():
    assert np.isclose(energy2sigma(300e3), 0.0006526161464700888)


def test_energy():
    energy = Accelerator(energy=300e3)

    assert energy.energy == 300e3
    assert np.isclose(energy.wavelength, energy2wavelength(300e3))

    energy.energy = 200e3

    assert np.isclose(energy.wavelength, energy2wavelength(200e3))


def test_energy_raises():
    accelerator1 = Accelerator(300e3)
    accelerator2 = Accelerator()

    with pytest.raises(RuntimeError):
        accelerator2.check_is_defined()

    accelerator2.energy = 200e3
    with pytest.raises(RuntimeError):
        accelerator1.check_match(accelerator2)

    accelerator2.energy = accelerator1.energy
    accelerator1.check_match(accelerator2)


def test_accelerator_event():
    accelerator = Accelerator(300e3)

    accelerator.energy = 200e3
    assert accelerator.event._notify_count == 1
