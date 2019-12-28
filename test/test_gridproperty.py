import numpy as np
from abtem.bases import GridProperty
import pytest


def test_grid_property():
    grid_property = GridProperty(value=5, dtype=np.float)

    assert np.all(grid_property.value == 5.)
    assert grid_property.value.dtype == np.float

    grid_property.value = 2

    assert np.all(grid_property.value == np.array([2, 2], dtype=np.float))
    assert grid_property.value.dtype == np.float

    grid_property = GridProperty(value=[5, 5], dtype=np.int)

    assert np.all(grid_property.value == 5)
    assert grid_property.value.dtype == np.int

    grid_property = GridProperty(value=lambda _: np.array([5., 5.]), dtype=np.int)

    assert np.all(grid_property.value == 5)
    assert grid_property.value.dtype == np.int

    with pytest.raises(RuntimeError):
        grid_property.value = 2

    with pytest.raises(RuntimeError):
        GridProperty(value=[5, 5, 5], dtype=np.float)

    grid_property = GridProperty(value=None, dtype=np.float)
    grid_property.value = 2

    assert np.all(grid_property.value == 2.)
