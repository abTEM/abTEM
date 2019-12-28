import numpy as np
import pytest

from abtem.bases import ArrayWithGrid, ArrayWithGridAndEnergy


def test_array_with_grid():
    array = np.zeros((100, 100), dtype=np.float32)

    array_with_grid = ArrayWithGrid(array, 2, extent=10)

    assert np.all(array_with_grid.gpts == np.array(array.shape))
    assert np.all(array_with_grid.extent == 10.)
    assert np.all(array_with_grid.sampling == .1)

    array_with_grid.extent = 20

    assert np.all(array_with_grid.sampling == .2)

    with pytest.raises(RuntimeError):
        ArrayWithGrid(array, 3)


def test_array_with_grid_and_energy():
    array = np.zeros((100, 100), dtype=np.float32)

    array_with_grid_and_energy = ArrayWithGridAndEnergy(array, 2, extent=10, energy=60e3)

    assert array_with_grid_and_energy.energy == 60e3

