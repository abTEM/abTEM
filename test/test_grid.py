import mock
import numpy as np
import pytest

from abtem.bases import Grid, Observable, GridProperty, fftfreq


def test_create_grid():
    grid = Grid(extent=5, sampling=.2)

    assert np.all(grid.extent == 5.)
    assert np.all(grid.gpts == 25)

    assert np.all(grid.sampling == .2)

    grid = Grid(sampling=.2, gpts=10)

    assert np.all(grid.extent == 2.)

    grid = Grid(extent=(8, 6), gpts=10)
    assert np.allclose(grid.sampling, np.array([0.8, 0.6]))

    grid = Grid()
    with pytest.raises(RuntimeError):
        grid.check_is_grid_defined()


def test_change_free_grid():
    grid = Grid(extent=(8, 6), gpts=10)

    grid.sampling = .2
    assert np.allclose(grid.extent, np.array([8., 6.]))
    assert np.allclose(grid.gpts, np.array([40, 30]))

    grid.gpts = 100
    assert np.allclose(grid.extent, np.array([8, 6]))
    assert np.allclose(grid.sampling, np.array([0.08, 0.06]))

    grid.extent = (16, 12)
    assert np.allclose(grid.gpts, np.array([100, 100]))
    assert np.allclose(grid.extent, np.array([16, 12]))
    assert np.allclose(grid.sampling, np.array([16 / 100, 12 / 100]))

    grid.extent = (10, 10)
    assert np.allclose(grid.sampling, grid.extent / grid.gpts)

    grid.sampling = .3
    assert np.allclose(grid.extent, grid.sampling * grid.gpts)

    grid.gpts = 30
    assert np.allclose(grid.sampling, grid.extent / grid.gpts)


def test_grid_raises():
    with pytest.raises(RuntimeError) as e:
        Grid(extent=[5, 5, 5])

    assert str(e.value) == 'grid value length of 3 != 2'


@mock.patch.object(Observable, 'notify_observers')
def test_grid_notify(mock_notify_observers):
    grid = Grid()

    grid.extent = 5
    assert mock_notify_observers.call_count == 1

    grid.gpts = 100
    assert mock_notify_observers.call_count == 2

    grid.sampling = .1
    assert mock_notify_observers.call_count == 3


def test_locked_grid():
    gpts = GridProperty(value=5, dtype=np.int, locked=True)

    grid = Grid(gpts=gpts)

    grid.extent = 10
    assert np.all(grid.sampling == 2)
    grid.extent = 20
    assert np.all(grid.sampling == 4)

    with pytest.raises(RuntimeError) as e:
        grid.gpts = 6

    assert str(e.value) == 'grid property locked'


def test_check_grid_matches():
    grid1 = Grid(extent=10, gpts=10)
    grid2 = Grid(extent=10, gpts=10)

    grid1.check_grids_can_match(grid2)

    grid2.sampling = .2

    with pytest.raises(RuntimeError) as e:
        grid1.check_grids_can_match(grid2)

    assert str(e.value) == 'inconsistent grid gpts ([10 10] != [50 50])'


def test_fourier_limits():
    grid = Grid(extent=(3, 3), gpts=(12, 13))
    assert np.isclose(fftfreq(grid)[0][5], grid.fourier_limits[0][1])
    assert np.isclose(fftfreq(grid)[0][6], grid.fourier_limits[0][0])
    assert np.isclose(fftfreq(grid)[1][6], grid.fourier_limits[1][1])
    assert np.isclose(fftfreq(grid)[1][7], grid.fourier_limits[1][0])
