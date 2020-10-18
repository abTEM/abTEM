from abtem.base_classes import AntialiasFilter
import numpy as np
from ase import Atoms
from abtem.waves import Probe


# def test_crop():
#     f = AntialiasFilter()
#
#     assert np.all(f.crop(f.get_mask((128, 128), (.1, .1), np), (.1, .1), 'valid'))
#     assert np.all(f.crop(f.get_mask((128, 128), (.1, .1), np), (.1, .1), 'valid'))
#
#     f = AntialiasFilter(rolloff=0.)
#     mask_sum = f.get_mask((128, 128), (.1, .1), np).sum()
#     assert f.crop(f.get_mask((128, 128), (.1, .1), np), (.1, .1), 'limit').sum() == mask_sum
#     assert f.crop(f.get_mask((128, 128), (.1, .1), np), (.1, .1), 'limit').sum() == mask_sum


def test_bandlimit():
    atoms = Atoms(cell=(10, 10, 2))

    probe = Probe(energy=300e3, semiangle_cutoff=30, rolloff=0.0, gpts=100)

    #alpha_x = np.fft.fftfreq(probe.gpts[0], probe.sampling[0]) * probe.wavelength * 1000
    #alpha_y = np.fft.fftfreq(probe.gpts[1], probe.sampling[1]) * probe.wavelength * 1000

    #alpha = np.sqrt(alpha_x[:, None] ** 2 + alpha_y[None] ** 2)

    #np.sum(alpha < 30)

    waves = probe.multislice((0, 0), atoms, pbar=False)
    diffraction_pattern = waves.diffraction_pattern(max_angle=30)

    assert diffraction_pattern.shape == (1, 31, 31)

    probe = Probe(energy=300e3, semiangle_cutoff=1e3, rolloff=0.0, gpts=100)
    waves = probe.multislice((0, 0), atoms, pbar=False)

    diffraction_pattern = waves.diffraction_pattern('limit')
    assert not np.allclose(diffraction_pattern.array / diffraction_pattern.array.max(), 1.)
    assert diffraction_pattern.shape == (1, 67, 67)

    diffraction_pattern = waves.diffraction_pattern('valid')

    assert diffraction_pattern.shape == (1, 45, 45)
    assert np.allclose(diffraction_pattern.array / diffraction_pattern.array.max(), 1.)
