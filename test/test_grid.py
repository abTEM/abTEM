import numpy as np
import pytest

from abtem.bases import Grid


def test_create_grid():
    grid = Grid(extent=5, sampling=.2)

    assert (grid.extent[0] == 5.) & (grid.extent[1] == 5.)
    assert (grid.gpts[0] == 25) & (grid.gpts[1] == 25)
    assert (grid.sampling[0] == .2) & (grid.sampling[1] == .2)

    grid = Grid(sampling=.2, gpts=10)
    assert (grid.extent[0] == 2.) & (grid.extent[1] == 2.)

    grid = Grid(extent=(8, 6), gpts=10)
    assert (grid.sampling[0] == .8) & (grid.sampling[1] == .6)

    grid = Grid()
    with pytest.raises(RuntimeError):
        grid.check_is_defined()


def test_change_grid():
    grid = Grid(extent=(8, 6), gpts=10)

    grid.sampling = .2
    assert (grid.extent[0] == 8.) & (grid.extent[1] == 6.)
    assert (grid.gpts[0] == 40) & (grid.gpts[1] == 30)

    grid.gpts = 100
    assert (grid.extent[0] == 8.) & (grid.extent[1] == 6.)
    assert (grid.sampling[0] == .08) & (grid.sampling[1] == .06)

    grid.extent = (16, 12)
    assert (grid.gpts[0] == 100) & (grid.gpts[1] == 100)
    assert (grid.extent[0] == 16.) & (grid.extent[1] == 12.)
    assert (grid.sampling[0] == .16) & (grid.sampling[1] == .12)

    grid.extent = (10, 10)
    assert (grid.sampling[0] == grid.extent[0] / grid.gpts[0]) & (grid.sampling[1] == grid.extent[1] / grid.gpts[1])

    grid.sampling = .3
    assert (grid.extent[0] == grid.sampling[0] * grid.gpts[0]) & (grid.extent[1] == grid.sampling[1] * grid.gpts[1])

    grid.gpts = 30
    assert (grid.sampling[0] == grid.extent[0] / grid.gpts[0]) & (grid.sampling[1] == grid.extent[1] / grid.gpts[1])


def test_grid_raises():
    with pytest.raises(RuntimeError) as e:
        Grid(extent=[5, 5, 5])

    assert str(e.value) == 'grid value length of 3 != 2'


def test_grid_event():
    grid = Grid()

    grid.extent = 5
    assert grid.changed._notify_count == 1

    grid.gpts = 100
    assert grid.changed._notify_count == 2

    grid.sampling = .1
    assert grid.changed._notify_count == 3


def test_locked_grid():
    grid = Grid(gpts=200, lock_gpts=True)

    grid.extent = 10
    assert (grid.sampling[0] == .05) & (grid.sampling[1] == .05)
    grid.extent = 20
    assert (grid.sampling[0] == .1) & (grid.sampling[1] == .1)

    with pytest.raises(RuntimeError) as e:
        grid.gpts = 100

    assert str(e.value) == 'gpts cannot be modified'


def test_grid_match():
    grid1 = Grid(extent=10, gpts=10)
    grid2 = Grid()
    grid1.match(grid2)

    grid1.check_match(grid2)
    grid2.sampling = .2

    with pytest.raises(RuntimeError) as e:
        grid1.check_match(grid2)

    assert str(e.value) == 'inconsistent grid gpts ((10, 10) != (50, 50))'

    grid1.match(grid2)
    grid1.check_match(grid2)

