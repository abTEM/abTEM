from abtem.base_classes import Grid
from abtem.scan import GridScan
import numpy as np


def test_calibration_coordinates():
    for endpoint in (True, False):
        gridscan = GridScan(sampling=.7, start=(0, 0), end=(4, 4), endpoint=endpoint)
        calibration_coordinates = gridscan.calibrations[0].coordinates(gridscan.gpts[0])
        assert np.all(calibration_coordinates == gridscan.get_positions()[::gridscan.gpts[0], 0])
