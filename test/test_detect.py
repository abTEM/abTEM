import numpy as np
from ase.build import bulk

from abtem.measure.detect import AnnularDetector, FlexibleAnnularDetector, PixelatedDetector
from abtem.potentials import Potential
from abtem.waves.scan import GridScan
from abtem.waves.waves import Probe


def test_detector_consistency():
    atoms = bulk('Si', 'diamond', a=5.43, cubic=True)
    atoms *= (1, 1, 1)

    potential = Potential(atoms,
                          gpts=256,
                          device='cpu',
                          projection='infinite',
                          slice_thickness=6)

    probe = Probe(energy=100e3, semiangle_cutoff=20, device='cpu')
    scan = GridScan(sampling=.5)
    flexible_detector = FlexibleAnnularDetector()
    annular_detector = AnnularDetector(inner=50, outer=90)
    pixelated_detector = PixelatedDetector()

    measurements = probe.scan(scan, [pixelated_detector, flexible_detector, annular_detector], potential)
    measurements.compute()

    assert np.allclose(measurements[1].integrate_radial(50, 90).array, measurements[2].array)
    assert np.allclose(measurements[0].integrate_radial(50, 90).array, measurements[2].array)

# def test_pixelated_detector():
#     gpts = (512, 512)
#     extent = (12, 12)
#     probe = Probe(energy=60e3, extent=extent, gpts=gpts, semiangle_cutoff=80, rolloff=0.2)
#     detector = PixelatedDetector(max_angle=30, resample='uniform')
#
#     wave = probe.build().downsample(max_angle=30)
#     measurement = detector.detect(wave)
#     assert measurement.shape == wave.array.shape
#
#     detector = PixelatedDetector(max_angle='valid', resample='uniform')
#     measurement = detector.detect(wave)
#     assert measurement.shape == wave.array.shape
#
#     detector = PixelatedDetector(max_angle='limit', resample='uniform')
#     measurement = detector.detect(wave)
#     assert measurement.shape == wave.array.shape
#
#     gpts = (512, 512)
#     extent = (10, 12)
#
#     probe = Probe(energy=60e3, extent=extent, gpts=gpts, semiangle_cutoff=80, rolloff=0.2)
#     detector = PixelatedDetector(max_angle='valid', resample='uniform')
#
#     wave = probe.build()
#     measurement = detector.allocate_measurement(wave)
#     assert measurement.array.shape[0] == measurement.array.shape[1]
