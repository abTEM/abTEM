import numpy as np
import pytest

from abtem import distributions
from abtem.waves import Probe


def test_gaussian_distribution_normalized():
    defocus = distributions.gaussian(1.0, num_samples=11, center=3)
    wave = Probe(energy=100e3, semiangle_cutoff=30, defocus=0.0, extent=10, gpts=64)
    assert np.allclose(
        wave.build().diffraction_patterns().reduce_ensemble().array.sum().compute(), 1.0
    )


def test_lorentzian_distribution_normalized():
    defocus = distributions.lorentzian(1.0, num_samples=21, center=3)
    wave = Probe(energy=100e3, semiangle_cutoff=30, defocus=defocus, extent=10, gpts=64)
    assert np.allclose(
        wave.build().diffraction_patterns().reduce_ensemble().array.sum().compute(), 1.0
    )


def test_lorentzian_distribution_shape():
    dist = distributions.lorentzian(2.0, num_samples=15)
    assert dist.shape == (15,)
    assert len(dist.values) == 15
    assert len(dist.weights) == 15
    # peak at center
    center_idx = len(dist.values) // 2
    assert dist.weights[center_idx] == dist.weights.max()


def test_lorentzian_distribution_multidimensional():
    dist = distributions.lorentzian(1.0, num_samples=11, dimension=2)
    assert dist.dimensions == 2
    assert dist.shape == (11, 11)


def test_voigtian_distribution_normalized():
    defocus = distributions.voigtian(1.0, 0.5, num_samples=21, center=3)
    wave = Probe(energy=100e3, semiangle_cutoff=30, defocus=defocus, extent=10, gpts=64)
    assert np.allclose(
        wave.build().diffraction_patterns().reduce_ensemble().array.sum().compute(), 1.0
    )


def test_voigtian_distribution_shape():
    dist = distributions.voigtian(1.0, 0.5, num_samples=15)
    assert dist.shape == (15,)
    assert len(dist.values) == 15
    assert len(dist.weights) == 15
    # peak at center
    center_idx = len(dist.values) // 2
    assert dist.weights[center_idx] == dist.weights.max()


def test_voigtian_distribution_multidimensional():
    dist = distributions.voigtian(1.0, 0.5, num_samples=11, dimension=2)
    assert dist.dimensions == 2
    assert dist.shape == (11, 11)


def test_voigtian_pure_gaussian_limit():
    # With gamma=0, weights must be proportional to a Gaussian at the sampled points
    sigma = 1.5
    v = distributions.voigtian(sigma, 0.0, num_samples=31)
    expected = np.exp(-0.5 * v.values**2 / sigma**2)
    expected /= np.sqrt((expected**2).sum())
    assert np.allclose(v.weights, expected, atol=1e-6)


def test_voigtian_pure_lorentzian_limit():
    # With sigma=0, weights must be proportional to a Lorentzian at the sampled points
    gamma = 1.5
    v = distributions.voigtian(0.0, gamma, num_samples=31)
    expected = 1.0 / (1.0 + (v.values / gamma) ** 2)
    expected /= np.sqrt((expected**2).sum())
    assert np.allclose(v.weights, expected, atol=1e-6)


def test_voigtian_both_zero_raises():
    with pytest.raises(ValueError, match="non-zero"):
        distributions.voigtian(0.0, 0.0, num_samples=11)


def test_pseudo_voigtian_distribution_normalized():
    defocus = distributions.pseudo_voigtian(1.0, 0.5, eta=0.4, num_samples=21, center=3)
    wave = Probe(energy=100e3, semiangle_cutoff=30, defocus=defocus, extent=10, gpts=64)
    assert np.allclose(
        wave.build().diffraction_patterns().reduce_ensemble().array.sum().compute(), 1.0
    )


def test_pseudo_voigtian_distribution_shape():
    dist = distributions.pseudo_voigtian(1.0, 0.5, eta=0.4, num_samples=15)
    assert dist.shape == (15,)
    assert len(dist.values) == 15
    assert len(dist.weights) == 15
    # peak at center (symmetric profile)
    center_idx = len(dist.values) // 2
    assert dist.weights[center_idx] == dist.weights.max()


def test_pseudo_voigtian_distribution_multidimensional():
    dist = distributions.pseudo_voigtian(1.0, 0.5, eta=0.4, num_samples=11, dimension=2)
    assert dist.dimensions == 2
    assert dist.shape == (11, 11)


def test_pseudo_voigtian_pure_gaussian_limit():
    # eta=0 must give a pure Gaussian
    sigma = 1.5
    pv = distributions.pseudo_voigtian(sigma, 1.0, eta=0.0, num_samples=31)
    expected = np.exp(-0.5 * pv.values**2 / sigma**2)
    expected /= np.sqrt((expected**2).sum())
    assert np.allclose(pv.weights, expected, atol=1e-6)


def test_pseudo_voigtian_pure_lorentzian_limit():
    # eta=1 must give a pure Lorentzian
    gamma = 1.5
    pv = distributions.pseudo_voigtian(1.0, gamma, eta=1.0, num_samples=31)
    expected = 1.0 / (1.0 + (pv.values / gamma) ** 2)
    expected /= np.sqrt((expected**2).sum())
    assert np.allclose(pv.weights, expected, atol=1e-6)


def test_pseudo_voigtian_both_zero_raises():
    with pytest.raises(ValueError, match="non-zero"):
        distributions.pseudo_voigtian(0.0, 0.0, eta=0.5, num_samples=11)
