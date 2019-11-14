import numpy as np

from ..detect import RingDetector, FourierSpaceDetector
from ..waves import ProbeWaves


def test_ring_detector():
    ring_detector = RingDetector(None, None, extent=10, gpts=200, energy=100e3)
    ring_detector.inner = (ring_detector.fourier_extent * ring_detector.wavelength / 4)[0]
    ring_detector.outer = (ring_detector.fourier_extent * ring_detector.wavelength / 2)[0]
    assert np.round(np.sum(ring_detector.get_efficiency()) / np.prod(ring_detector.gpts), 2) == .58


# def test_ring_detector_detect():
#     probe_waves = ProbeWaves(energy=60e3, extent=10, cutoff=.03, gpts=200).build()
#
#     ring_detector = RingDetector(0.05, .2)
#
#     assert np.isclose(ring_detector.detect(probe_waves), 0)
#
#     probe_waves = ProbeWaves(energy=60e3, extent=10, cutoff=.06, gpts=200).build()
#
#     assert not np.isclose(ring_detector.detect(probe_waves), 0)
#

def test_fourier_space_detector():
    fourier_space_detector = FourierSpaceDetector(gpts=200)
    assert fourier_space_detector.out_shape == (200, 200)

    #fourier_space_detector = FourierSpaceDetector(gpts=200, extent=)
