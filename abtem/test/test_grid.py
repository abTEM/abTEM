import pytest
import numpy as np
from ..bases import GridProperty, Grid, ArrayWithGrid


def test_grid_property():
    grid_property = GridProperty(value=5, dtype=np.float64)

    assert np.all(grid_property.value == 5.)
    assert grid_property.value.dtype == np.float64

    grid_property.value = 2

    assert np.all(grid_property.value == np.array([2, 2], dtype=np.float64))
    assert grid_property.value.dtype == np.float64

    grid_property = GridProperty(value=[5, 5], dtype=np.int64)

    assert np.all(grid_property.value == 5)
    assert grid_property.value.dtype == np.int64

    with pytest.raises(RuntimeError):
        GridProperty(value=lambda _: np.array([5, 5]), dtype=np.int64, locked=False)

    grid_property = GridProperty(value=lambda _: np.array([5, 5]), dtype=np.int64, locked=True)

    assert np.all(grid_property.value == 5)
    assert grid_property.value.dtype == np.int64

    with pytest.raises(RuntimeError):
        grid_property.value = 2

    with pytest.raises(RuntimeError):
        GridProperty(value=[5, 5, 5], dtype=np.float32)

    grid_property = GridProperty(value=None, dtype=np.float64)
    grid_property.value = 2

    assert np.all(grid_property.value == 2.)


def test_create_grid():
    grid = Grid(extent=5, sampling=.2)

    assert np.all(grid.extent == 5.)
    assert np.all(grid.gpts == 25)

    assert np.all(grid.sampling == .2)

    grid = Grid(sampling=.2, gpts=10)

    assert np.all(grid.extent == 2.)

    grid = Grid(extent=(8, 6), gpts=10)
    assert np.all(grid.sampling == np.array([0.8, 0.6]))


def test_change_grid():
    grid.sampling = .2
    assert np.all(grid.extent == np.array([8., 6.]))
    assert np.all(grid.gpts == np.array([40, 30]))

    grid.gpts = 100
    assert np.all(grid.extent == np.array([8, 6]))
    assert np.all(grid.sampling == np.array([0.08, 0.06]))

    grid.extent = (16, 12)
    assert np.all(grid.gpts == np.array([100, 100]))
    assert np.all(grid.extent == np.array([16, 12]))
    assert np.all(grid.sampling == np.array([16 / 100, 12 / 100]))

    grid.extent = (10, 10)
    assert np.all(grid.sampling == grid.extent / grid.gpts)

    grid.sampling = .3
    assert np.all(grid.extent == grid.sampling * grid.gpts)

    grid.gpts = 30
    assert np.all(grid.sampling == grid.extent / grid.gpts)

    def get_gpts(obj):
        return np.array([20, 20], dtype=np.int32)

    gpts = GridProperty(value=get_gpts, dtype=np.int32, locked=True)
    grid = Grid(gpts=gpts)

    grid.sampling = .1

    assert np.all(grid.extent == np.array([2., 2.], dtype=np.float32))

    grid.sampling = .01
    assert np.all(grid.gpts == np.array([20, 20], dtype=np.int32))
    assert np.all(grid.extent == grid.sampling * np.float32(grid.gpts))

    with pytest.raises(RuntimeError):
        grid.gpts = 10

    grid = Grid()

    with pytest.raises(RuntimeError):
        grid.check_is_grid_defined()

    grid = Grid(extent=10, sampling=.1, dimensions=1, endpoint=True)
    assert grid.gpts == 101
    assert grid.sampling == .1

    grid = Grid(extent=9.9, sampling=.1, dimensions=1, endpoint=True)
    assert grid.gpts == 100

    grid = Grid(extent=10, gpts=100, dimensions=1, endpoint=True)
    assert grid.sampling == 10 / 99.

    grid = Grid(gpts=101, sampling=.1, dimensions=1, endpoint=True)
    assert grid.extent == 10.

    kwargs_list = [{'gpts': 100, 'extent': 10}, {'gpts': 100, 'sampling': .1}, {'extent': 10, 'sampling': .1}]
    for kwargs in kwargs_list:
        grid1 = Grid(**kwargs)
        grid2 = Grid()

        grid1.match_grid(grid2)

        assert np.all(grid1.gpts == grid2.gpts)
        assert np.all(grid1.sampling == grid2.sampling)
        assert np.all(grid1.extent == grid2.extent)

        grid2 = Grid()
        grid2.match_grid(grid1)

        assert np.all(grid1.gpts == grid2.gpts)
        assert np.all(grid1.sampling == grid2.sampling)
        assert np.all(grid1.extent == grid2.extent)

        grid2 = Grid(gpts=101)

        with pytest.raises(RuntimeError):
            grid1.match_grid(grid2)

        with pytest.raises(RuntimeError):
            grid2.match_grid(grid1)


def test_array_with_grid():
    array = np.zeros((100, 100), dtype=np.float32)

    array_with_grid = ArrayWithGrid(array, 2, 2, extent=10)

    assert np.all(array_with_grid.gpts == np.array(array.shape))
    assert np.all(array_with_grid.extent == 10.)
    assert np.all(array_with_grid.sampling == .1)

    array_with_grid.extent = 20

    assert np.all(array_with_grid.sampling == .2)

    with pytest.raises(RuntimeError):
        ArrayWithGrid(array, 1, 1)

    with pytest.raises(RuntimeError):
        ArrayWithGrid(array, 2, 3)
