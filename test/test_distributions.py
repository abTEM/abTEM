import numpy as np
import pytest

from abtem import distributions
from abtem.transfer import CTF
from abtem.waves import PlaneWave, Probe


def test_gaussian_distribution_normalized():
    defocus = distributions.gaussian(1.0, num_samples=11, center=3)
    wave = Probe(energy=100e3, semiangle_cutoff=30, defocus=0.0, extent=10, gpts=64)
    assert np.allclose(
        wave.build().diffraction_patterns().reduce_ensemble().array.sum().compute(), 1.0
    )


def test_focal_series_with_incoherent_spread():
    # Answers https://github.com/abTEM/abTEM/issues/168: a focal series (kept as
    # its own axis) where each defocus step is itself an incoherent
    # (temporal-coherence) average. Composing two apply_ctf() calls works because
    # the defocus phase term is linear in defocus, so applying CTF twice with
    # independent defocus distributions equals a single application with summed
    # defocus.
    #
    # Both distributions use ensemble_mean=False and the incoherent spread axis is
    # reduced with a plain .sum(): abTEM pre-multiplies the wave amplitude by the
    # L2-normalized distribution weight (so I_i = w_i**2 * I(x_i) with sum w_i**2 =
    # 1), which makes .sum() the correct incoherent average. This is the same
    # reduction the partial-coherence tutorial performs by hand.
    import ase

    atoms = ase.build.mx2(vacuum=2)
    exit_wave = PlaneWave(energy=80e3, sampling=0.1).multislice(atoms).compute()

    focal_series_values = np.array([-100.0, 0.0, 100.0])
    focal_series = distributions.from_values(
        focal_series_values, ensemble_mean=False
    )
    spread = distributions.gaussian(
        20.0, num_samples=7, sampling_limit=2, ensemble_mean=False
    )

    images = (
        exit_wave.apply_ctf(CTF(energy=80e3, defocus=focal_series))
        .apply_ctf(CTF(energy=80e3, defocus=spread))
        .intensity()
        .compute()
    )

    # locate the spread (7 values) and series (3 values) axes and reduce the spread
    spread_axis = next(
        i
        for i, ax in enumerate(images.axes_metadata)
        if getattr(ax, "values", None) is not None and len(ax.values) == 7
    )
    result = images.sum(spread_axis)
    series_axis = next(
        i
        for i, ax in enumerate(result.axes_metadata)
        if getattr(ax, "values", None) is not None and len(ax.values) == 3
    )
    reduced = np.moveaxis(result.array, series_axis, 0)

    # brute-force reference: per focus step, weighted incoherent average over the
    # spread (weights pre-multiplied as w_i**2, matching abTEM's convention)
    raw = distributions.gaussian(20.0, num_samples=7, sampling_limit=2)
    deltas, weights = np.array(raw.values), np.array(raw.weights)
    ref = np.zeros((len(focal_series_values),) + exit_wave.array.shape)
    for i, series_val in enumerate(focal_series_values):
        acc = np.zeros(exit_wave.array.shape)
        for delta, w in zip(deltas, weights):
            acc += (w**2) * (
                exit_wave.apply_ctf(CTF(energy=80e3, defocus=series_val + delta))
                .intensity()
                .compute()
                .array
            )
        ref[i] = acc

    assert reduced.shape[0] == 3  # focal series axis survives
    assert np.allclose(reduced, ref, atol=1e-4)


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
