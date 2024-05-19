from abtem.waves import PlaneWave
import numpy as np


def test_tilt():
    planewave1 = PlaneWave(energy=80e3, tilt=([100], 0), extent=10, gpts=64)
    planewave2 = PlaneWave(energy=80e3, tilt=([100, 0], [0, 0]), extent=10, gpts=64)
    planewave3 = PlaneWave(energy=80e3, tilt=(100, 0), extent=10, gpts=64)
    tilt = np.array([[100.0, 0.0], [0.0, 0.0]])
    planewave4 = PlaneWave(energy=80e3, tilt=tilt, extent=10, gpts=64)

    wave1 = planewave1.multislice(potential).compute()
    wave2 = planewave2.multislice(potential).compute()
    wave3 = planewave3.multislice(potential).compute()
    wave4 = planewave4.multislice(potential).compute()

    assert wave1.shape[0] == 1
    assert wave2.shape[:2] == (2, 2)
    assert wave3.shape[:-2] == ()
    assert wave4.shape[:-2] == (2,)
    assert wave1.metadata["base_tilt_x"] == 0.0
    assert wave2.metadata["base_tilt_x"] == 0.0
    assert wave3.metadata["base_tilt_x"] == 100.0
    assert wave4.metadata["base_tilt_x"] == 0.0
    assert wave1[0].metadata["base_tilt_x"] == 100.0
    assert wave2[0, 0].metadata["base_tilt_x"] == 100.0
    assert wave2[0, 0].metadata["base_tilt_y"] == 0.0
    assert wave4[0].metadata["base_tilt_x"] == 100.0
    assert wave4[0].metadata["base_tilt_y"] == 0.0
    assert wave1[0] == wave3
    assert wave1[0] == wave2[0, 0] == wave2[0, 1]
    assert wave4[0] == wave3
