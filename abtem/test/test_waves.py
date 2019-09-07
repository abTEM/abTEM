import pytest
import numpy as np
from ..waves import Waves
from ..bases import Grid


class DummyPotential(Grid):

    def __init__(self, extent=None, gpts=None, sampling=None, slice_thickness=.5):
        self.slice_thickness = slice_thickness
        Grid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, dimensions=2)

    @property
    def num_slices(self):
        return 10

    def get_slice(self, i):
        array = np.zeros(self.gpts, dtype=np.float32)
        array[:self.gpts[0] // 2] = 1
        return array


def test_waves():
    array = np.ones((128, 128), dtype=np.complex64)

    waves = Waves(array)

    with pytest.raises(RuntimeError):
        waves.check_is_grid_defined()

    waves = Waves(array, extent=5)

    with pytest.raises(RuntimeError):
        waves.check_is_energy_defined()

    waves = Waves(array, extent=5, energy=60e3)

    waves.check_is_grid_defined()
    waves.check_is_energy_defined()

    waves = Waves(array, energy=60e3)

    potential = DummyPotential(extent=5)

    waves.match_grid(potential)

    waves.check_is_grid_defined()
    waves.check_is_energy_defined()

    old_array = waves.array.copy()

    new_waves = waves.multislice(potential)

    assert np.allclose(waves.array, old_array)
    assert not np.allclose(new_waves.array, old_array)

    waves.multislice(potential, in_place=True)

    assert not np.allclose(waves.array, old_array)
