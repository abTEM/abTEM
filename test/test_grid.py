import numpy as np
import pytest

from abtem.base_classes import Grid
from abtem.waves import Probe

def test_create_grid():
    grid = Grid(extent=5, sampling=.2)

    assert (grid.extent[0] == 5.) & (grid.extent[1] == 5.)
    assert (grid.gpts[0] == 25) & (grid.gpts[1] == 25)
    print(grid.sampling[0] == .2, type(grid.sampling[0]), type(.2))
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

    assert str(e.value) == 'Grid value length of 3 != 2'


def test_grid_event():
    grid = Grid()

    grid.extent = 5
    assert grid.event._notify_count == 1

    grid.gpts = 100
    assert grid.event._notify_count == 2

    grid.sampling = .1
    assert grid.event._notify_count == 3


def test_locked_grid():
    grid = Grid(gpts=200, lock_gpts=True)

    grid.extent = 10
    assert (grid.sampling[0] == .05) & (grid.sampling[1] == .05)
    grid.extent = 20
    assert (grid.sampling[0] == .1) & (grid.sampling[1] == .1)

    with pytest.raises(RuntimeError) as e:
        grid.gpts = 100

    assert str(e.value) == 'Grid gpts cannot be modified'


def test_grid_match():
    grid1 = Grid(extent=10, gpts=10)
    grid2 = Grid()
    grid1.match(grid2)

    grid1.check_match(grid2)
    grid2.sampling = .2

    with pytest.raises(RuntimeError) as e:
        grid1.check_match(grid2)

    assert str(e.value) == 'Inconsistent grid gpts ((10, 10) != (50, 50))'

# def test_gridscan_calibration():
#     gridscan = GridScan(start=[0, 0],
#                         end=[4, 4],
#                         sampling=.7,
#                         endpoint=False)
#
#     gridscan.calibrations[0].coordinates(gridscan.gpts[0]) == gridscan.get_positions()[:, 0][::7]

# def test_scattering_angle():
#     probe = Probe(extent=18, gpts=(250, 251), energy=80e3)
#     probe = probe.build()
#
#     alpha_x = np.fft.fftfreq(probe.gpts[0], probe.sampling[0]) * probe.wavelength * 1000
#     alpha_y = np.fft.fftfreq(probe.gpts[1], probe.sampling[1]) * probe.wavelength * 1000
#
#     assert np.isclose(alpha_x.min(), probe.max_scattering_angles[0][0])
#     assert np.isclose(alpha_x.max(), probe.max_scattering_angles[0][1])
#     assert np.isclose(alpha_y.min(), probe.max_scattering_angles[1][0])
#     assert np.isclose(alpha_y.max(), probe.max_scattering_angles[1][1])
    #assert np.sum(np.abs(alpha_x) < 65) == probe._resampled_gpts(65)[0]
    #assert np.sum(np.abs(alpha_y) < 65) == probe._resampled_gpts(65)[1]
