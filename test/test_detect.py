import numpy as np
from ase.io import read

from abtem.detect import AnnularDetector, FlexibleAnnularDetector, PixelatedDetector
from abtem.potentials import Potential
from abtem.scan import GridScan
from abtem.waves import Probe


def test_detector_consistency():
    atoms = read('data/srtio3_100.cif')
    atoms *= (4, 4, 1)

    potential = Potential(atoms, gpts=256, projection='infinite', slice_thickness=.5,
                          parametrization='kirkland', ).build(pbar=False)

    probe = Probe(energy=300e3, semiangle_cutoff=9.4, rolloff=0.05)

    flexible_detector = FlexibleAnnularDetector()
    annular_detector = AnnularDetector(inner=20, outer=40)
    pixelated_detector = PixelatedDetector()

    end = (potential.extent[0] / 4, potential.extent[1] / 4)

    gridscan = GridScan(start=[0, 0], end=end, sampling=.5)

    measurements = probe.scan(gridscan, [flexible_detector, pixelated_detector, annular_detector], potential,
                              pbar=False)

    assert np.allclose(measurements[0].integrate(20, 40).array, measurements[2].array)
    assert np.allclose(annular_detector.integrate(measurements[1]).array,
                       measurements[2].array)


def test_pixelated_detector():
    gpts = (512, 512)
    extent = (12, 12)
    probe = Probe(energy=60e3, extent=extent, gpts=gpts, semiangle_cutoff=80, rolloff=0.2)
    detector = PixelatedDetector(max_angle=30, resample='uniform')

    wave = probe.build().downsample(max_angle=30)
    measurement = detector.detect(wave)
    assert measurement.shape == wave.array.shape

    detector = PixelatedDetector(max_angle='valid', resample='uniform')
    measurement = detector.detect(wave)
    assert measurement.shape == wave.array.shape

    detector = PixelatedDetector(max_angle='limit', resample='uniform')
    measurement = detector.detect(wave)
    assert measurement.shape == wave.array.shape

    gpts = (512, 512)
    extent = (10, 12)

    probe = Probe(energy=60e3, extent=extent, gpts=gpts, semiangle_cutoff=80, rolloff=0.2)
    detector = PixelatedDetector(max_angle='valid', resample='uniform')

    wave = probe.build()
    measurement = detector.allocate_measurement(wave)
    assert measurement.array.shape[0] == measurement.array.shape[1]


def test_segmented_detector():
    # TODO : fill in test
    pass