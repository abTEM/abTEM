from abtem.base_classes import AntialiasFilter
import numpy as np


def test_crop():
    f = AntialiasFilter(extent=20, gpts=64)
    assert np.all(f.crop(f.get_mask(np), 'valid'))
    f = AntialiasFilter(extent=20, gpts=65)
    assert np.all(f.crop(f.get_mask(np), 'valid'))

    f = AntialiasFilter(extent=20, gpts=64, rolloff=0)
    assert f.crop(f.get_mask(np), 'limit').sum() == f.get_mask(np).sum()
    f = AntialiasFilter(extent=20, gpts=65, rolloff=0)
    assert f.crop(f.get_mask(np), 'limit').sum() == f.get_mask(np).sum()


def test_bandlimit():
    f = AntialiasFilter(extent=20, gpts=64)
    array = np.exp(1.j * np.random.rand(64, 64))
    f.bandlimit(array)
    array = np.fft.fft2(array)

    assert np.allclose(np.abs(array)[array.shape[0] // 2], 0.)
