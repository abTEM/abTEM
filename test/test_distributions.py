import numpy as np
import pytest
from ase.build import bulk
from utils import gpu

from abtem import distributions
from abtem.core.backend import asnumpy
from abtem.transfer import CTF, Aperture, Vortex
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


# One factory per profile; each computes its weights in its own code.
_PROFILES = {
    "gaussian": lambda **kw: distributions.gaussian(1.0, **kw),
    "lorentzian": lambda **kw: distributions.lorentzian(1.0, **kw),
    "voigtian": lambda **kw: distributions.voigtian(1.0, 0.5, **kw),
    "pseudo_voigtian": lambda **kw: distributions.pseudo_voigtian(
        1.0, 0.5, eta=0.4, **kw
    ),
}


def test_distribution_normalized():
    # The profiles differ only in their weights; the reduction checked here is
    # one code path, so they are looped over rather than parametrized.
    for name, factory in _PROFILES.items():
        try:
            _assert_reduced_probe_normalized(factory(num_samples=21, center=3))
        except AssertionError as error:
            raise AssertionError(name) from error


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


@pytest.mark.parametrize("name", ["lorentzian", "voigtian", "pseudo_voigtian"])
def test_distribution_shape(name):
    dist = _PROFILES[name](num_samples=15)
    assert dist.shape == (15,)
    assert len(dist.values) == 15
    assert len(dist.weights) == 15
    # peak at center (symmetric profiles)
    center_idx = len(dist.values) // 2
    assert dist.weights[center_idx] == dist.weights.max()

    dist = _PROFILES[name](num_samples=11, dimension=2)
    assert dist.dimensions == 2
    assert dist.shape == (11, 11)


def _gaussian(values, sigma=1.5):
    return np.exp(-0.5 * values**2 / sigma**2)


def _lorentzian(values, gamma=1.5):
    return 1.0 / (1.0 + (values / gamma) ** 2)


@pytest.mark.parametrize(
    "factory, profile",
    [
        # voigtian takes its own branch for a zero width
        (lambda: distributions.voigtian(1.5, 0.0, num_samples=31), _gaussian),
        (lambda: distributions.voigtian(0.0, 1.5, num_samples=31), _lorentzian),
        # pseudo_voigtian mixes the two with eta
        (
            lambda: distributions.pseudo_voigtian(1.5, 1.0, eta=0.0, num_samples=31),
            _gaussian,
        ),
        (
            lambda: distributions.pseudo_voigtian(1.0, 1.5, eta=1.0, num_samples=31),
            _lorentzian,
        ),
    ],
    ids=["voigt_gamma0", "voigt_sigma0", "pseudo_eta0", "pseudo_eta1"],
)
def test_pure_limits(factory, profile):
    # The weights must be proportional to the pure profile at the sampled points
    # (probability weights sum to one).
    dist = factory()
    expected = profile(dist.values)
    expected /= expected.sum()
    assert np.allclose(dist.weights, expected, atol=1e-6)




def test_both_widths_zero_raises():
    with pytest.raises(ValueError, match="non-zero"):
        distributions.voigtian(0.0, 0.0, num_samples=11)
    with pytest.raises(ValueError, match="non-zero"):
        distributions.pseudo_voigtian(0.0, 0.0, eta=0.5, num_samples=11)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_vortex_quantum_number_ensemble(device):
    # Vortex(quantum_number=[...]) should build a l-ensemble in one Waves object,
    # matching per-l scalar Vortex runs bit-for-bit through multislice -- same
    # ensemble mechanism Aperture(semiangle_cutoff=[...]) already uses.
    from abtem import Potential

    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    potential = Potential(atoms, sampling=0.1, slice_thickness=1.0, device=device)

    ls = (-1, 0, 1)
    probe = Probe(energy=100e3, semiangle_cutoff=20.0, device=device)
    probe.aperture = Vortex(quantum_number=ls, semiangle_cutoff=20.0, soft=True)
    probe.grid.match(potential)

    exit_waves = asnumpy(probe.build().multislice(potential).compute().array)
    assert exit_waves.shape[0] == len(ls)

    for i, l in enumerate(ls):
        probe_scalar = Probe(energy=100e3, semiangle_cutoff=20.0, device=device)
        probe_scalar.aperture = Vortex(
            quantum_number=l, semiangle_cutoff=20.0, soft=True
        )
        probe_scalar.grid.match(potential)
        exit_scalar = probe_scalar.build().multislice(potential).compute()
        assert np.allclose(exit_waves[i], asnumpy(exit_scalar.array), atol=1e-8)


def _probe_array(aperture, energy=100e3, gpts=64, extent=10.0, device="cpu"):
    probe = Probe(
        energy=energy, semiangle_cutoff=20.0, gpts=gpts, extent=extent, device=device
    )
    probe.aperture = aperture
    return asnumpy(probe.build().compute().array)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("soft", [False, True])
def test_vortex_zeroes_the_optical_axis(device, soft):
    # A vortex has no amplitude on its axis; the k = 0 bin used to keep the
    # arbitrary arctan2(0, 0) phase, breaking the +l / -l mirror relation.
    plane_wave = PlaneWave(energy=100e3, gpts=64, extent=10.0, device=device)
    waves = plane_wave.build(lazy=False)
    kernel = asnumpy(
        Vortex(quantum_number=(-2, 0, 1), semiangle_cutoff=20.0, soft=soft)
        ._evaluate_kernel(waves)
    )
    assert np.all(kernel[[0, 2], 0, 0] == 0.0)
    assert kernel[1, 0, 0] != 0.0


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_vortex_l0_soft_matches_plain_soft_aperture(device):
    # The ensemble rewrite inlines soft_aperture()'s formula (it has to: that
    # helper sets its centre point by in-place index assignment, which does not
    # generalize to broadcast ensemble axes). For l=0 the vortex phase is unity,
    # so the result must be identical to the ordinary soft aperture.
    vortex = _probe_array(
        Vortex(quantum_number=0, semiangle_cutoff=20.0, soft=True), device=device
    )
    plain = _probe_array(Aperture(semiangle_cutoff=20.0, soft=True), device=device)
    # the same formula, evaluated in a different order: equal to rounding
    tolerance = 10 * np.finfo(plain.dtype).eps * np.abs(plain).max()
    assert np.allclose(vortex, plain, rtol=0, atol=tolerance)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_vortex_hard_aperture_ensemble_matches_scalars(device):
    ls = (-2, 0, 3)
    ens = _probe_array(
        Vortex(quantum_number=ls, semiangle_cutoff=20.0, soft=False), device=device
    )
    assert ens.shape[0] == len(ls)
    for i, l in enumerate(ls):
        scalar = _probe_array(
            Vortex(quantum_number=l, semiangle_cutoff=20.0, soft=False), device=device
        )
        assert np.allclose(ens[i], scalar, atol=1e-12)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_vortex_quantum_number_and_semiangle_ensembles_combine(device):
    ls, cutoffs = (-1, 1), (15.0, 20.0)
    ens = _probe_array(
        Vortex(quantum_number=ls, semiangle_cutoff=cutoffs, soft=True), device=device
    )
    assert ens.shape[:2] == (len(ls), len(cutoffs))
    for i, l in enumerate(ls):
        for j, cutoff in enumerate(cutoffs):
            scalar = _probe_array(
                Vortex(quantum_number=l, semiangle_cutoff=cutoff, soft=True),
                device=device,
            )
            assert np.allclose(ens[i, j], scalar, atol=1e-12)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_vortex_ensemble_preserves_mirror_symmetry(device):
    # +l and -l must remain exact mirror images: magnetic signals are extracted
    # as the difference between them, so any asymmetry becomes a spurious signal.
    # Compared on the angular grid, where the relation is plain conjugation --
    # in real space it would additionally involve the r -> -r inversion that
    # conjugating a Fourier transform induces.
    aperture = Vortex(quantum_number=(1, -1), semiangle_cutoff=20.0, soft=True)
    waves = PlaneWave(energy=100e3, gpts=64, extent=10.0, device=device).build(
        lazy=False
    )
    kernel = asnumpy(aperture._evaluate_kernel(waves))
    assert kernel.shape[0] == 2
    assert np.allclose(kernel[0], np.conj(kernel[1]), atol=1e-12)


def test_vortex_ensemble_axes_metadata():
    aperture = Vortex(
        quantum_number=(-1, 0, 1), semiangle_cutoff=(15.0, 20.0), soft=True
    )
    axes = aperture.ensemble_axes_metadata
    assert [ax.label for ax in axes] == ["quantum_number", "semiangle_cutoff"]
    assert [len(ax.values) for ax in axes] == [3, 2]
    # OAM is dimensionless; without an explicit empty string it inherits
    # NonLinearAxis's "unknown" default and panel titles read "l = 1 unknown"
    assert axes[0].units == ""
    assert axes[1].units == "mrad"
    assert "unknown" not in axes[0].format_title()
