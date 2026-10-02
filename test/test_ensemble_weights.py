"""Oracle tests for probability-weighted ensemble reduction (abTEM issue #305).

Convention under test: a distribution's weights p_i are *probabilities*. Every
ensemble member is the physically correct, unweighted wave function for its
parameter value x_i, and ``reduce_ensemble()`` returns the weighted mean
Σ p_i I_i / Σ p_i. Frozen-phonon (and other unweighted) axes keep the plain mean.

Every expected value below comes from an oracle that is independent of the
ensemble machinery under test: a brute-force loop over single-valued
(non-ensemble) simulations, an explicit numpy weighted sum, a closed-form moment,
or the analytic Gaussian characteristic function (Kirkland's temporal envelope).

Tolerances: abTEM computes in float32 by default (``abtem.config['precision']``).
A weighted mean of float32 values of magnitude ~max(I) carries a relative
rounding error of a few ulp·log2(N) ≲ 1e-6; the brute-force oracle carries the
same, so ``rtol=1e-5`` relative to the maximum of the reference is a ~10x margin.
Tests that set float64 use correspondingly tighter bounds (stated in place).
"""

import ase
import numpy as np
import pytest
from utils import gpu

import abtem
from abtem import distributions
from abtem.core.backend import get_array_module
from abtem.core.energy import energy2wavelength
from abtem.transfer import CTF
from abtem.waves import PlaneWave, Probe

# Asymmetric values AND weights, so a reversal or misalignment of weights relative
# to values, or a dropped/duplicated member, changes the answer.
ASYM_VALUES = np.array([-40.0, 0.0, 25.0, 60.0])
ASYM_WEIGHTS = np.array([0.1, 0.5, 0.3, 0.1])

PROBE_KW = dict(energy=100e3, semiangle_cutoff=20, extent=8, gpts=48)


def _to_numpy(array):
    array = array.compute() if hasattr(array, "compute") else array
    xp = get_array_module(array)
    return xp.asnumpy(array) if xp is not np else np.asarray(array)


def _assert_close_to(actual, reference, rtol=1e-5):
    actual = _to_numpy(actual)
    reference = _to_numpy(reference)
    assert actual.shape == reference.shape
    scale = np.abs(reference).max()
    np.testing.assert_allclose(actual, reference, rtol=0, atol=rtol * scale)


def _gaussian_oracle(sigma, num_samples, sampling_limit=3.0, center=0.0):
    """Values and probability weights of a truncated, sampled Gaussian, written out
    from the definition (independently of abtem.distributions)."""
    x = np.linspace(-sampling_limit * sigma, sampling_limit * sigma, num_samples)
    return x + center, np.exp(-(x**2) / (2 * sigma**2))


def _distribution_cases():
    # (label, distribution, oracle values, oracle weights)
    values, weights = _gaussian_oracle(15.0, 7, center=10.0)
    return {
        "gaussian": (
            distributions.gaussian(15.0, num_samples=7, center=10.0),
            values,
            weights,
        ),
        "asymmetric": (
            distributions.from_values(
                ASYM_VALUES, weights=ASYM_WEIGHTS, ensemble_mean=True
            ),
            ASYM_VALUES,
            ASYM_WEIGHTS,
        ),
    }


@pytest.fixture(scope="module")
def tiny_potential():
    atoms = ase.Atoms(
        "CO", positions=[(2.0, 2.0, 1.0), (5.5, 4.0, 2.5)], cell=(8.0, 8.0, 4.0)
    )
    return abtem.Potential(atoms, gpts=48, slice_thickness=2.0, projection="finite")


# ---------------------------------------------------------------------------------
# (a) STEM: probe ensembles reduce to the brute-force weighted average
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["gaussian", "asymmetric"])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_stem_probe_defocus_ensemble_matches_brute_force(case, lazy, device):
    dist, values, weights = _distribution_cases()[case]

    probe = Probe(defocus=dist, device=device, **PROBE_KW)
    reduced = probe.build(lazy=lazy).intensity().reduce_ensemble()

    # Oracle: one ordinary single-defocus probe per value, weighted explicitly.
    reference = sum(
        p * _to_numpy(Probe(defocus=x, device=device, **PROBE_KW).build().intensity().array)
        for x, p in zip(values, weights)
    ) / np.sum(weights)

    assert reduced.shape == reference.shape
    _assert_close_to(reduced.array, reference)


@pytest.mark.parametrize("case", ["gaussian", "asymmetric"])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_stem_scan_defocus_ensemble_matches_brute_force(
    case, lazy, device, tiny_potential
):
    dist, values, weights = _distribution_cases()[case]
    scan = abtem.GridScan((0, 0), (4, 4), gpts=(3, 3))
    detectors = [abtem.AnnularDetector(10, 60), abtem.PixelatedDetector()]

    probe = Probe(defocus=dist, device=device, **PROBE_KW)
    measurements = probe.scan(tiny_potential, scan=scan, detectors=detectors, lazy=lazy)
    measurements = [m.compute() if m.is_lazy else m for m in measurements]

    # Oracle: independent scans with ordinary single-defocus probes.
    references = [0.0, 0.0]
    for x, p in zip(values, weights):
        single = Probe(defocus=x, device=device, **PROBE_KW).scan(
            tiny_potential, scan=scan, detectors=detectors, lazy=False
        )
        for i in range(2):
            references[i] = references[i] + p * _to_numpy(single[i].array)
    references = [r / np.sum(weights) for r in references]

    for measurement, reference in zip(measurements, references):
        _assert_close_to(measurement.array, reference)


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_ctf_defocus_ensemble_matches_brute_force(lazy, device, tiny_potential):
    # PRISM evaluates the CTF on the S-matrix plane waves itself, a separate code
    # path from Probe; its ensemble must reduce with the same weights.
    dist = distributions.from_values(
        ASYM_VALUES, weights=ASYM_WEIGHTS, ensemble_mean=True
    )
    scan = abtem.GridScan((0, 0), (4, 4), gpts=(3, 3))
    detector = abtem.AnnularDetector(10, 60)
    s_matrix = abtem.SMatrix(
        potential=tiny_potential, energy=100e3, semiangle_cutoff=20, device=device
    )

    ctf = CTF(semiangle_cutoff=20, defocus=dist)
    reduced = s_matrix.scan(scan=scan, detectors=detector, ctf=ctf, lazy=lazy)

    reference = sum(
        p
        * _to_numpy(
            s_matrix.scan(
                scan=scan,
                detectors=detector,
                ctf=CTF(semiangle_cutoff=20, defocus=x),
                lazy=False,
            ).array
        )
        for x, p in zip(ASYM_VALUES, ASYM_WEIGHTS)
    ) / ASYM_WEIGHTS.sum()

    _assert_close_to(reduced.array, reference)


# ---------------------------------------------------------------------------------
# (b) Width convention: the intensity sees the documented distribution
# ---------------------------------------------------------------------------------


def _effective_weights(images):
    """Weight with which each ensemble member enters ``reduce_ensemble()``:
    reduce(I·e_j)/reduce(I), with e_j the j'th one-hot vector along the axis."""
    images = images.compute()
    n = images.shape[0]
    total = _to_numpy(images.reduce_ensemble().array).sum()
    effective = np.zeros(n)
    for j in range(n):
        one_hot = np.zeros((n, 1, 1), dtype=images.array.dtype)
        one_hot[j] = 1.0
        xp = get_array_module(images.array)
        masked = images * xp.asarray(one_hot)
        effective[j] = _to_numpy(masked.reduce_ensemble().array).sum() / total
    return effective


def _images_for(kind, dist, device):
    if kind == "plane_wave_ctf":
        # HRTEM: an unscattered plane wave only has the k=0 beam, where every CTF
        # is exactly 1, so each member's intensity is independent of defocus.
        waves = PlaneWave(energy=100e3, extent=8, gpts=32, device=device).build()
        return waves.apply_ctf(CTF(defocus=dist)).intensity()
    else:
        # STEM: every member is an ordinary normalized probe.
        return Probe(defocus=dist, device=device, **PROBE_KW).build().intensity()


@pytest.mark.parametrize("kind", ["plane_wave_ctf", "probe"])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_gaussian_width_seen_by_intensity_is_sigma(kind, device):
    sigma = 20.0
    # sampling_limit=6: the truncated mass is 2e-9 and, for a Gaussian sampled
    # with spacing h = 0.12σ, the Poisson-summation error of the discrete second
    # moment is ~exp(-2π²σ²/h²) ≈ 0 -- so the discrete variance equals σ² to
    # far below float32 resolution of the effective weights (~1e-7).
    dist = distributions.gaussian(sigma, num_samples=101, sampling_limit=6)
    values = np.array(dist.values)

    p = _effective_weights(_images_for(kind, dist, device))

    # Oracle: closed form. Old convention gave σ²/2 (HRTEM, weights squared) or
    # the variance of a ±6σ top-hat, 12σ² (STEM, weights erased).
    np.testing.assert_allclose(p.sum(), 1.0, rtol=1e-5)
    np.testing.assert_allclose((p * values**2).sum(), sigma**2, rtol=1e-4)
    # and the effective weights are the Gaussian itself, p ∝ exp(-x²/2σ²)
    expected = np.exp(-(values**2) / (2 * sigma**2))
    np.testing.assert_allclose(p, expected / expected.sum(), rtol=0, atol=1e-6)


def test_gaussian_default_truncation_variance():
    # With the default sampling_limit=3 the sampled distribution is a truncated
    # Gaussian. Oracle: variance of a Gaussian truncated at ±Lσ,
    # σ²(1 - 2Lφ(L)/(2Φ(L)-1)) (a textbook result), evaluated with scipy. The
    # sum over 201 evenly spaced samples (endpoints included) differs from the
    # integral by O(h·x²φ(L)/σ) ≈ 1e-3 relative, so rtol=3e-3.
    from scipy.stats import norm

    sigma, L = 20.0, 3.0
    dist = distributions.gaussian(sigma, num_samples=201)
    x, p = np.array(dist.values), np.array(dist.weights)
    truncated_var = sigma**2 * (1 - 2 * L * norm.pdf(L) / (2 * norm.cdf(L) - 1))
    np.testing.assert_allclose((p * x**2).sum() / p.sum(), truncated_var, rtol=3e-3)


@pytest.mark.parametrize("kind", ["plane_wave_ctf", "probe"])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_lorentzian_effective_weights(kind, device):
    gamma = 10.0
    dist = distributions.lorentzian(gamma, num_samples=21)
    values = np.array(dist.values)

    p = _effective_weights(_images_for(kind, dist, device))

    # Oracle: p_i ∝ 1/(1+(x_i/γ)²) -- a Lorentzian, not Lorentzian² (old HRTEM
    # convention: x⁻⁴ tails, HWHM 0.64γ) and not a top-hat (old STEM).
    expected = 1.0 / (1.0 + (values / gamma) ** 2)
    np.testing.assert_allclose(p, expected / expected.sum(), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------------
# (c) Physics cross-check: incoherent defocus average == Kirkland temporal envelope
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("precision, atol", [("float32", 2e-5), ("float64", 1e-9)])
def test_defocus_average_reproduces_kirkland_temporal_envelope(precision, atol):
    # For a Gaussian defocus distribution with standard deviation σ about d0, the
    # probability-weighted average of the coherent phase factor is the Gaussian
    # characteristic function,
    #     Σ p_i exp(-iπλ x_i k²) → exp(-iπλ d0 k²) · exp(-(πλ k² σ)²/2),
    # and Kirkland's temporal envelope exp(-(πλΔk²/2)²) equals the second factor
    # when Δ = √2·σ (Δ is the 1/e width of the focal spread). So averaging the
    # CTF's imaginary part sin(πλ(d0+x_i)k²) over the ensemble must give
    # sin(πλd0k²)·E_Kirkland(k; Δ=√2σ) -- which pins the probability-weight
    # convention of the ensemble reduction and the 1/e width convention of
    # TemporalEnvelope in one test.
    #
    # Discretisation: 161 samples to ±8σ (h = 0.1σ). Truncated mass 1e-15 (at
    # ±6σ it would be 2e-9, which float64 resolves); aliasing error of the
    # discrete characteristic function ~exp(-(2π/h)²σ²/2), negligible since the
    # largest phase slope here, πα²/λ ≈ 0.11/Å, is far below 2π/h ≈ 3.1/Å. What
    # remains is rounding: the float32 phase χ reaches ~24 rad, i.e. an absolute
    # phase error ~24·6e-8 ≈ 1.5e-6 per sample, so atol=2e-5 (float32); in float64
    # the same estimate gives ~3e-15, so atol=1e-9 is a wide margin.
    energy, sigma, d0 = 200e3, 20.0, 50.0
    focal_spread = np.sqrt(2) * sigma

    with abtem.config.set({"precision": precision}):
        dist = distributions.gaussian(
            sigma, num_samples=161, sampling_limit=8, center=d0
        )
        averaged = CTF(energy=energy, defocus=dist).profiles(max_angle=30, gpts=200)
        averaged = _to_numpy(averaged.reduce_ensemble().array)

        # The documented quasi-coherent alternative: a single CTF at d0 with the
        # temporal envelope. Its profiles stack the components
        # ("ctf" = sin χ·E, "temporal envelope" = E) along the first axis.
        quasi_coherent = CTF(energy=energy, defocus=d0, focal_spread=focal_spread)
        quasi_coherent = quasi_coherent.profiles(max_angle=30, gpts=200)
        components = quasi_coherent.ensemble_axes_metadata[-1].values
        assert components == ("ctf", "temporal envelope")
        quasi_coherent_ctf, envelope = _to_numpy(quasi_coherent.array)

    # Analytic oracle on the same angular grid as CTF.profiles.
    wavelength = energy2wavelength(energy)
    alpha = np.linspace(0, 30e-3, 200)
    kirkland = np.exp(-((np.pi * wavelength * focal_spread * (alpha / wavelength) ** 2 / 2) ** 2))
    coherent_at_d0 = np.sin(np.pi * alpha**2 * d0 / wavelength)

    assert kirkland[-1] < 0.2  # the envelope is actually probed, not ~1 everywhere
    np.testing.assert_allclose(envelope, kirkland, rtol=0, atol=atol)
    np.testing.assert_allclose(averaged, coherent_at_d0 * kirkland, rtol=0, atol=atol)
    np.testing.assert_allclose(averaged, quasi_coherent_ctf, rtol=0, atol=atol)


# ---------------------------------------------------------------------------------
# (d) HRTEM: exit wave through an ensemble CTF
# ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exit_wave():
    atoms = ase.Atoms(
        "CO", positions=[(2.0, 2.0, 1.0), (5.5, 4.0, 2.5)], cell=(8.0, 8.0, 4.0)
    )
    return PlaneWave(energy=100e3, gpts=48).multislice(atoms).compute()


@pytest.mark.parametrize("case", ["gaussian", "asymmetric"])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_hrtem_ctf_defocus_ensemble_matches_brute_force(case, lazy, device, exit_wave):
    dist, values, weights = _distribution_cases()[case]
    waves = exit_wave.copy_to_device(device)
    if lazy:
        waves = waves.ensure_lazy()

    reduced = waves.apply_ctf(CTF(defocus=dist, Cs=-5e4)).intensity().reduce_ensemble()

    reference = sum(
        p * _to_numpy(waves.apply_ctf(CTF(defocus=x, Cs=-5e4)).intensity().array)
        for x, p in zip(values, weights)
    ) / np.sum(weights)

    _assert_close_to(reduced.array, reference)


# ---------------------------------------------------------------------------------
# (e) Frozen phonons: still the plain mean
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_frozen_phonon_reduction_is_plain_mean(lazy, device):
    atoms = ase.Atoms(
        "CO", positions=[(2.0, 2.0, 1.0), (5.5, 4.0, 2.5)], cell=(8.0, 8.0, 4.0)
    )

    def potential(ensemble_mean):
        fp = abtem.FrozenPhonons(
            atoms, num_configs=4, sigmas=0.1, seed=7, ensemble_mean=ensemble_mean
        )
        return abtem.Potential(fp, gpts=48, slice_thickness=2.0)

    wave = PlaneWave(energy=100e3, device=device)
    detector = abtem.PixelatedDetector()
    reduced = wave.multislice(potential(True), detector, lazy=lazy).compute()
    configs = wave.multislice(potential(False), detector, lazy=lazy).compute()

    assert configs.shape[0] == 4
    # Oracle: explicit, equal-weight numpy mean over the four configurations.
    reference = _to_numpy(configs.array).mean(axis=0)
    _assert_close_to(reduced.array, reference, rtol=1e-6)


# ---------------------------------------------------------------------------------
# (f) Two weighted axes: outer product of the per-axis weights
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_two_weighted_axes_match_double_sum(lazy, device):
    defocus = distributions.from_values(
        ASYM_VALUES, weights=ASYM_WEIGHTS, ensemble_mean=True
    )
    cs_values = np.array([-3e4, 1e4, 8e4])
    cs_weights = np.array([0.6, 0.15, 0.25])
    cs = distributions.from_values(cs_values, weights=cs_weights, ensemble_mean=True)

    probe = Probe(defocus=defocus, Cs=cs, device=device, **PROBE_KW)
    reduced = probe.build(lazy=lazy).intensity().reduce_ensemble()

    reference = 0.0
    for x, px in zip(ASYM_VALUES, ASYM_WEIGHTS):
        for c, pc in zip(cs_values, cs_weights):
            single = Probe(defocus=x, Cs=c, device=device, **PROBE_KW).build()
            reference = reference + px * pc * _to_numpy(single.intensity().array)
    reference = reference / (ASYM_WEIGHTS.sum() * cs_weights.sum())

    _assert_close_to(reduced.array, reference)


# ---------------------------------------------------------------------------------
# (g) Weights survive chunking, slicing, concatenation, zarr, ensemble_mean=False
# ---------------------------------------------------------------------------------


def _asym_reference(device, members=slice(None)):
    values, weights = ASYM_VALUES[members], ASYM_WEIGHTS[members]
    return sum(
        p * _to_numpy(Probe(defocus=x, device=device, **PROBE_KW).build().intensity().array)
        for x, p in zip(values, weights)
    ) / weights.sum()


@pytest.mark.parametrize("chunks", [(1, 3), (2, 2), (1, 1, 1, 1), (3, 1)])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_weights_survive_lazy_chunking(chunks, device):
    dist = distributions.from_values(
        ASYM_VALUES, weights=ASYM_WEIGHTS, ensemble_mean=True
    )
    images = Probe(defocus=dist, device=device, **PROBE_KW).build(lazy=True).intensity()
    images = images.rechunk((chunks, (48,), (48,)))
    assert images.array.chunks[0] == chunks

    reduced = images.reduce_ensemble()
    assert reduced.is_lazy  # the reduction itself must not compute eagerly
    _assert_close_to(reduced.array, _asym_reference(device))


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_weights_survive_build_with_ensemble_chunks(device):
    # Chunking the *distribution* when building: each dask block is built from a
    # sub-distribution (distributions.divide), and the assembled axis metadata
    # must still carry the full, aligned weights.
    dist = distributions.from_values(
        ASYM_VALUES, weights=ASYM_WEIGHTS, ensemble_mean=True
    )
    probe = Probe(defocus=dist, device=device, **PROBE_KW)
    images = probe.build(lazy=True, max_batch=1).intensity()
    assert images.array.chunks[0] == (1, 1, 1, 1)
    assert images.ensemble_axes_metadata[0].weights == tuple(ASYM_WEIGHTS)
    _assert_close_to(images.reduce_ensemble().array, _asym_reference(device))


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_weights_follow_indexing(device):
    dist = distributions.from_values(
        ASYM_VALUES, weights=ASYM_WEIGHTS, ensemble_mean=True
    )
    images = Probe(defocus=dist, device=device, **PROBE_KW).build().intensity()

    # Reversed and sliced views must re-align the weights with the members.
    _assert_close_to(images[::-1].reduce_ensemble().array, _asym_reference(device))
    _assert_close_to(
        images[1:].reduce_ensemble().array, _asym_reference(device, slice(1, None))
    )
    _assert_close_to(
        images[[3, 0]].reduce_ensemble().array, _asym_reference(device, [3, 0])
    )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_weights_follow_concatenation(device):
    dist = distributions.from_values(
        ASYM_VALUES, weights=ASYM_WEIGHTS, ensemble_mean=True
    )
    images = Probe(defocus=dist, device=device, **PROBE_KW).build().intensity()
    joined = abtem.concatenate([images[:1], images[1:]], axis=0)
    assert joined.ensemble_axes_metadata[0].weights == tuple(ASYM_WEIGHTS)
    _assert_close_to(joined.reduce_ensemble().array, _asym_reference(device))


@pytest.mark.parametrize("lazy", [True, False])
def test_weights_survive_zarr_round_trip(lazy, tmp_path):
    dist = distributions.from_values(
        ASYM_VALUES, weights=ASYM_WEIGHTS, ensemble_mean=True
    )
    images = Probe(defocus=dist, **PROBE_KW).build(lazy=lazy).intensity()
    url = str(tmp_path / "ensemble.zarr.zip")
    images.to_zarr(url)
    loaded = abtem.from_zarr(url)

    assert loaded.ensemble_axes_metadata[0].weights == tuple(ASYM_WEIGHTS)
    _assert_close_to(loaded.reduce_ensemble().array, _asym_reference("cpu"))


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_manual_reduction_of_kept_ensemble(device):
    # ensemble_mean=False keeps the axis; reduce_ensemble(axis=...) then applies
    # the same probability weights on request, while a plain .sum()/.mean()
    # warns that it ignores them.
    dist = distributions.from_values(
        ASYM_VALUES, weights=ASYM_WEIGHTS, ensemble_mean=False
    )
    images = Probe(defocus=dist, device=device, **PROBE_KW).build().intensity()
    assert images.shape[0] == 4
    assert images.reduce_ensemble().shape[0] == 4  # not flagged: not reduced

    _assert_close_to(images.reduce_ensemble(axis=0).array, _asym_reference(device))

    with pytest.warns(UserWarning, match="ignores the weights"):
        images.sum(0)
    with pytest.warns(UserWarning, match="ignores the weights"):
        images.mean(0)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_weighted_tilt_axis(device):
    # Beam-tilt distributions are reduced with their weights too (previously the
    # weights were discarded -- a plain mean over the sampled tilts).
    # A 12 Å Au column and tilts up to 50 mrad, so that the exit-wave images of
    # the members differ at the O(1) level (the weighted and plain means then
    # differ by ~25% of the maximum, far above the tolerance).
    tilts, weights = [0.0, 20.0, 50.0], [0.2, 0.7, 0.1]
    tilt_x = distributions.from_values(tilts, weights=weights, ensemble_mean=True)
    atoms = ase.Atoms(
        "Au4",
        positions=[(3.0, 3.0, z) for z in (1.0, 4.0, 7.0, 10.0)],
        cell=(6.0, 6.0, 12.0),
    )
    potential = abtem.Potential(atoms, gpts=48, slice_thickness=1.0)

    wave = PlaneWave(energy=100e3, tilt=(tilt_x, 0.0), device=device)
    reduced = wave.multislice(potential).intensity().reduce_ensemble().compute()

    reference = sum(
        p
        * _to_numpy(
            PlaneWave(energy=100e3, tilt=(t, 0.0), device=device)
            .multislice(potential)
            .intensity()
            .array
        )
        for t, p in zip(tilts, weights)
    )
    _assert_close_to(reduced.array, reference)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_weighted_energy_axis(device, tiny_potential):
    # Energy distributions carry their weights on the EnergyAxis (previously a
    # plain mean over the sampled energies; here 1.1% of the maximum off).
    energies, weights = [80e3, 100e3, 200e3], [0.2, 0.7, 0.1]
    energy = distributions.from_values(energies, weights=weights, ensemble_mean=True)

    images = PlaneWave(energy=energy, device=device).multislice(tiny_potential)
    reduced = images.intensity().reduce_ensemble()

    reference = sum(
        p
        * _to_numpy(
            PlaneWave(energy=e, device=device)
            .multislice(tiny_potential)
            .intensity()
            .array
        )
        for e, p in zip(energies, weights)
    )
    _assert_close_to(reduced.array, reference)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_weighted_reduction_dtype_and_direct_oracle(dtype, lazy, device):
    from abtem.core.axes import ParameterAxis
    from abtem.core.backend import get_array_module as xp_for

    rng = np.random.default_rng(0)
    host = rng.integers(0, 1000, size=(4, 3, 5, 6)).astype(dtype)
    xp = xp_for(device)
    array = xp.asarray(host)
    if lazy:
        import dask.array as da

        array = da.from_array(array, chunks=(1, 2, -1, -1))

    axes = [
        ParameterAxis(values=tuple(ASYM_VALUES), weights=tuple(ASYM_WEIGHTS), _ensemble_mean=True),
        ParameterAxis(values=(1.0, 2.0, 3.0), weights=(0.6, 0.15, 0.25), _ensemble_mean=True),
    ]
    images = abtem.Images(array, sampling=0.1, ensemble_axes_metadata=axes)
    reduced = images.reduce_ensemble()
    assert reduced.is_lazy == lazy

    # Oracle: explicit float64 double sum with the weights written out.
    w1 = ASYM_WEIGHTS / ASYM_WEIGHTS.sum()
    w2 = np.array([0.6, 0.15, 0.25])
    reference = np.einsum("i,j,ijkl->kl", w1, w2 / w2.sum(), host.astype(np.float64))

    result = _to_numpy(reduced.array)
    expected_dtype = np.dtype(dtype) if dtype != "int32" else np.dtype(
        abtem.config.get("precision")
    )
    assert result.dtype == expected_dtype
    # float32: ~1e-7 relative rounding per operation on values up to 1e3.
    rtol = 1e-12 if dtype == "float64" else 1e-6
    _assert_close_to(result, reference, rtol=rtol)


def test_distribution_weights_are_validated():
    with pytest.raises(ValueError, match=">= 0"):
        distributions.from_values([1.0, 2.0], weights=[0.5, -0.1])
    with pytest.raises(ValueError, match="one weight per value"):
        distributions.from_values([1.0, 2.0], weights=[1.0])
    with pytest.raises(ValueError, match="all be zero"):
        distributions.from_values([1.0, 2.0], weights=[0.0, 0.0])


def test_deprecated_normalize_options_warn_but_do_not_change_reduction():
    # The deprecated options behave as 'probability': the stored weights sum to
    # one (not Σp² = 1, which no longer means anything), and the reduction is
    # unchanged.
    new = distributions.gaussian(15.0, num_samples=7)
    with pytest.warns(FutureWarning, match="deprecated"):
        old = distributions.gaussian(15.0, num_samples=7, normalize="intensity")
    with pytest.warns(FutureWarning, match="deprecated"):
        amplitude = distributions.gaussian(15.0, num_samples=7, normalize="amplitude")

    np.testing.assert_allclose(np.sum(new.weights), 1.0)
    np.testing.assert_array_equal(old.weights, new.weights)
    np.testing.assert_array_equal(amplitude.weights, new.weights)

    a = Probe(defocus=new, **PROBE_KW).build().intensity().reduce_ensemble()
    b = Probe(defocus=old, **PROBE_KW).build().intensity().reduce_ensemble()
    _assert_close_to(a.array, _to_numpy(b.array), rtol=1e-6)

    with pytest.raises(ValueError, match="Unknown normalization"):
        distributions.gaussian(15.0, num_samples=7, normalize="bogus")


# ---------------------------------------------------------------------------------
# (h) Every axis built from a distribution carries its weights; reductions that
#     would ignore them warn or apply them
# ---------------------------------------------------------------------------------


def test_transfer_function_energy_axis_carries_weights():
    # The transfer-function energy axis used to be built without weights (and
    # without the ensemble_mean flag), unlike the Probe/PlaneWave energy axis.
    energy = distributions.from_values(
        [100e3, 200e3, 300e3], weights=[0.2, 0.7, 0.1], ensemble_mean=True
    )
    for transfer_function in (
        CTF(energy=energy, defocus=50.0, semiangle_cutoff=20.0),
        abtem.transfer.Aperture(semiangle_cutoff=20.0, energy=energy),
        abtem.transfer.TemporalEnvelope(focal_spread=10.0, energy=energy),
    ):
        (axis,) = [
            axis
            for axis in transfer_function.ensemble_axes_metadata
            if isinstance(axis, abtem.core.axes.EnergyAxis)
        ]
        assert axis.weights == (0.2, 0.7, 0.1)
        assert axis._ensemble_mean
        # identical to the axis of a wave function with the same energy spread
        wave_axis = PlaneWave(energy=energy).ensemble_axes_metadata[0]
        assert axis == wave_axis


def test_reduction_over_all_axes_warns_for_weighted_axes():
    dist = distributions.from_values(
        ASYM_VALUES, weights=ASYM_WEIGHTS, ensemble_mean=False
    )
    images = Probe(defocus=dist, **PROBE_KW).build().intensity()
    with pytest.warns(UserWarning, match="ignores the weights"):
        images.sum()
    with pytest.warns(UserWarning, match="ignores the weights"):
        images.mean()


def test_concatenating_weighted_with_unweighted_axis_raises():
    from abtem.core.axes import ParameterAxis

    weighted = ParameterAxis(label="C10", values=(0.0, 1.0), weights=(0.2, 0.8))
    unweighted = ParameterAxis(label="C10", values=(2.0,))
    with pytest.raises(ValueError, match="relative weighting is undefined"):
        weighted.concatenate(unweighted)
    with pytest.raises(ValueError, match="relative weighting is undefined"):
        unweighted.concatenate(weighted)

    # equal weights on the weighted side: the plain mean is exact either way
    equal = ParameterAxis(label="C10", values=(0.0, 1.0), weights=(0.5, 0.5))
    assert equal.concatenate(unweighted).weights is None
    assert weighted.concatenate(weighted).weights == (0.2, 0.8, 0.2, 0.8)


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_weighted_and_unweighted_axes_reduced_together(lazy, device):
    # Weighted, unweighted, weighted: the unweighted axis is averaged first, which
    # renumbers the weighted axes behind it.
    from abtem.core.axes import ParameterAxis
    from abtem.core.backend import get_array_module as xp_for

    rng = np.random.default_rng(1)
    host = rng.random((4, 2, 3, 5, 6))
    array = xp_for(device).asarray(host)
    if lazy:
        import dask.array as da

        array = da.from_array(array, chunks=(1, 1, 2, -1, -1))

    w1, w3 = ASYM_WEIGHTS, np.array([0.6, 0.15, 0.25])
    axes = [
        ParameterAxis(
            values=tuple(ASYM_VALUES), weights=tuple(w1), _ensemble_mean=True
        ),
        ParameterAxis(values=(1.0, 2.0), _ensemble_mean=True),
        ParameterAxis(values=(1.0, 2.0, 3.0), weights=tuple(w3), _ensemble_mean=True),
    ]
    images = abtem.Images(array, sampling=0.1, ensemble_axes_metadata=axes)
    reduced = images.reduce_ensemble()
    assert reduced.is_lazy == lazy

    reference = np.einsum(
        "i,j,k,ijklm->lm", w1 / w1.sum(), np.full(2, 0.5), w3 / w3.sum(), host
    )
    _assert_close_to(reduced.array, reference, rtol=1e-12)


def test_visualization_range_sum_applies_weights():
    # Summing a range of ensemble members for display weights a weighted axis
    # (scaled so that equal weights would give the plain sum) without warning.
    from abtem.visualize.visualizations import _sum_ensemble_range

    dist = distributions.from_values(
        ASYM_VALUES, weights=ASYM_WEIGHTS, ensemble_mean=False
    )
    images = Probe(defocus=dist, **PROBE_KW).build().intensity().compute()

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summed = _sum_ensemble_range(images[1:], (0,))

    members = _to_numpy(images.array)[1:]
    weights = ASYM_WEIGHTS[1:]
    reference = len(weights) * np.einsum("i,ijk->jk", weights / weights.sum(), members)
    _assert_close_to(summed.array, reference)

    # unweighted axes keep the plain sum
    unweighted = abtem.Images(
        images.array,
        sampling=images.sampling,
        ensemble_axes_metadata=[
            abtem.core.axes.ParameterAxis(values=tuple(ASYM_VALUES))
        ],
    )
    _assert_close_to(
        _sum_ensemble_range(unweighted, (0,)).array, _to_numpy(images.array).sum(0)
    )
