import numpy as np
import pytest

from abtem.bases import Accelerator
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
    energy1 = Accelerator(300e3)
    energy2 = Accelerator()

    with pytest.raises(RuntimeError):
        energy2.check_is_defined()

    energy2.energy = 200e3
    with pytest.raises(RuntimeError):
        energy1.check_energies_can_match(energy2)

    energy2.energy = energy1.energy
    energy1.check_energies_can_match(energy2)

# def test_energy_notifies():
#     with mock.patch.object(Energy, 'notify_observers') as mock_notify:
#         energy = Energy()
#         assert mock_notify.call_count == 0
#         energy.energy = 200e3
#         assert mock_notify.call_count == 1
