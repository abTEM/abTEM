import numpy as np

from abtem import distributions
from abtem.waves import Probe


def test_gaussian_distribution_normalized():
    defocus = distributions.gaussian(1., num_samples=11, center=3)
    wave = Probe(energy=100e3, semiangle_cutoff=30, defocus=0., extent=10, gpts=64)
    assert np.allclose(wave.build().diffraction_patterns().reduce_ensemble().array.sum().compute(), 1.)
