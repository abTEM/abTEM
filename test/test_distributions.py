import numpy as np

from abtem import distributions
from abtem.transfer import CTF
from abtem.waves import PlaneWave, Probe


def _assert_reduced_probe_normalized(defocus):
    # Oracle: every member is an ordinary probe normalized to unit total
    # intensity, and the probability-weighted mean of unit-intensity members has
    # unit intensity (Σ p_i · 1 / Σ p_i = 1). The stored weights are probabilities
    # summing to one (the default normalize="probability").
    assert np.isclose(np.sum(defocus.weights), 1.0)
    wave = Probe(energy=100e3, semiangle_cutoff=30, defocus=defocus, extent=10, gpts=64)
    diffraction_patterns = wave.build().diffraction_patterns()
    assert diffraction_patterns.shape[0] == len(defocus)  # really an ensemble
    reduced = diffraction_patterns.reduce_ensemble()
    assert reduced.shape == diffraction_patterns.shape[1:]
    assert np.allclose(reduced.array.sum().compute(), 1.0)


def test_gaussian_distribution_normalized():
    _assert_reduced_probe_normalized(
        distributions.gaussian(1.0, num_samples=11, center=3)
    )


def test_focal_series_with_incoherent_spread():
    # Answers https://github.com/abTEM/abTEM/issues/168: a focal series (kept as
    # its own axis) where each defocus step is itself an incoherent
    # (temporal-coherence) average. Composing two apply_ctf() calls works because
    # the defocus phase term is linear in defocus, so applying CTF twice with
    # independent defocus distributions equals a single application with summed
    # defocus.
    #
    # Distribution weights are probabilities applied at ensemble reduction, so the
    # spread axis is reduced to Σ p_i I(x_i) / Σ p_i either automatically
    # (ensemble_mean=True) or on request (reduce_ensemble(axis=...) on an axis
    # kept with ensemble_mean=False), while the focal-series axis survives.
    import ase

    atoms = ase.build.mx2(vacuum=2)
    exit_wave = PlaneWave(energy=80e3, sampling=0.1).multislice(atoms).compute()

    focal_series_values = np.array([-100.0, 0.0, 100.0])
    focal_series = distributions.from_values(
        focal_series_values, ensemble_mean=False
    )

    def images_with_spread(ensemble_mean):
        spread = distributions.gaussian(
            20.0, num_samples=7, sampling_limit=2, ensemble_mean=ensemble_mean
        )
        return (
            exit_wave.apply_ctf(CTF(energy=80e3, defocus=focal_series))
            .apply_ctf(CTF(energy=80e3, defocus=spread))
            .intensity()
            .compute()
        )

    automatic = images_with_spread(ensemble_mean=True).reduce_ensemble()

    kept = images_with_spread(ensemble_mean=False)
    spread_axis = next(
        i
        for i, ax in enumerate(kept.ensemble_axes_metadata)
        if len(getattr(ax, "values", ())) == 7
    )
    manual = kept.reduce_ensemble(axis=spread_axis)

    # Brute-force oracle: per focus step, an explicit weighted incoherent average
    # of ordinary single-defocus images, with the Gaussian probabilities written
    # out from the definition p ∝ exp(-δ²/2σ²) on the ±2σ, 7-point grid.
    deltas = np.linspace(-40.0, 40.0, 7)
    probabilities = np.exp(-(deltas**2) / (2 * 20.0**2))
    ref = np.zeros((len(focal_series_values),) + exit_wave.array.shape)
    for i, series_val in enumerate(focal_series_values):
        for delta, p in zip(deltas, probabilities):
            ref[i] += p * (
                exit_wave.apply_ctf(CTF(energy=80e3, defocus=series_val + delta))
                .intensity()
                .compute()
                .array
            )
        ref[i] /= probabilities.sum()

    # float32 images of magnitude ~1: rtol 1e-5 of the maximum is ~100 ulp.
    for reduced in (automatic, manual):
        assert reduced.shape[0] == 3  # focal series axis survives
        # the axis is labelled C10 = -defocus
        assert reduced.ensemble_axes_metadata[0].values == tuple(-focal_series_values)
        np.testing.assert_allclose(reduced.array, ref, rtol=0, atol=1e-5 * ref.max())
