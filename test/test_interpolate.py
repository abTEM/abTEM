import numpy as np
from abtem.utils import spatial_frequencies
from scipy.interpolate import interpn
from abtem.detect import PixelatedDetector
from abtem.interpolate import interpolate_bilinear_cpu
from abtem.waves import Probe


def test_interpolate():
    detector = PixelatedDetector()

    array = np.random.rand(2, 32, 32)

    gpts = array.shape[-2:]
    new_gpts = (16, 9)

    v, u, vw, uw = detector._bilinear_nodes_and_weight(gpts, new_gpts, (1/32, 1/32), (1/16, 1/9), np)

    interpolated = interpolate_bilinear_cpu(array, v, u, vw, uw)

    kx, ky = spatial_frequencies(gpts, (1, 1))
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)

    kx_new, ky_new = spatial_frequencies(new_gpts, (1, 1))
    kx_new = np.fft.fftshift(kx_new)
    ky_new = np.fft.fftshift(ky_new)

    px, py = np.meshgrid(kx_new, ky_new, indexing='ij')
    p = np.array([px.ravel(), py.ravel()]).T
    interpolated2 = interpn((kx, ky), array[0], p)
    interpolated2 = interpolated2.reshape(new_gpts)

    assert np.allclose(interpolated[0], interpolated2)


def test_resample_diffraction_patterns():
    for extent in [(5, 10), (5, 8)]:
        for gpts in [(256, 256), (256, 296)]:
            probe = Probe(energy=60e3, extent=extent, gpts=gpts, semiangle_cutoff=80, rolloff=0.2)
            detector = PixelatedDetector(max_angle='valid', resample='uniform')

            wave = probe.build()
            measurement = detector.detect(wave)
            measurement /= measurement.max()

            probe = Probe(energy=60e3, extent=(5, 5), gpts=(256, 256), semiangle_cutoff=80, rolloff=0.2)
            wave = probe.build()
            measurement2 = detector.detect(wave)
            measurement2 /= measurement2.max()

            s1 = (measurement2.shape[1] - measurement.shape[1]) // 2
            s2 = (measurement2.shape[2] - measurement.shape[2]) // 2
            measurement2 = measurement2[:, s1:s1 + measurement.shape[2], s2:s2 + measurement.shape[2]]
            assert np.all(np.abs(measurement2[0] - measurement[0]) < .5)
