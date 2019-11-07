import pytest
import numpy as np
from ..waves import Waves, PlaneWaves, ProbeWaves
from ..bases import Grid
from ..detect import DetectorBase
import mock


#
# @pytest.fixture(scope='session')
# def mocked_detector_base():
#     with mock.patch.object(DetectorBase, 'detect') as detect:
#         with mock.patch.object(DetectorBase, 'out_shape') as out_shape:
#             out_shape.side_effect = lambda: (1,)
#             yield DetectorBase
#
#
# def test_probe_waves_linescan(potential, mocked_detector_base):
#     array = np.zeros()
#     probe_waves = Waves(extent=10, energy=60e3)
#
#     detector = mocked_detector_base()