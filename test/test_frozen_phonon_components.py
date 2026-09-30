"""Elastic, diffuse and total intensity from frozen-phonon exit waves.

The reference is formed independently of the implementation: from each
configuration's complex diffraction amplitudes, in float64, as the mean amplitude m,
the total T = mean |a|², the elastic E = |m|² and the diffuse D = T - E. The sizes all
differ (3 configurations, 2 members of a second ensemble axis, a 16 x 20 grid), so a
component or axis in the wrong place shows up as a wrong shape or wrong values.
"""

import dask.array as da
import numpy as np
import pytest

import abtem
from abtem.core.axes import FrozenPhononsAxis, OrdinalAxis
from abtem.measurements import elastic_diffuse_diffraction_patterns
from abtem.waves import Waves

N_CONFIGS = 3
N_MEMBERS = 2
GPTS = (16, 20)
HELPER_KWARGS = dict(max_angle="cutoff", parity="odd")
# The helper always returns centred (fftshift) patterns.
DP_KWARGS = dict(**HELPER_KWARGS, fftshift=True)


def _exit_waves(diffuse_fraction=1.0, n_configs=N_CONFIGS, lazy=False, seed=1):
    """Configurations ψ_j = μ + noise_j, complex64, with a frozen-phonon axis followed
    by a second ensemble axis; ``diffuse_fraction`` scales the noise."""
    rng = np.random.default_rng(seed)
    shape = (N_MEMBERS,) + GPTS

    def normal(size):
        return rng.normal(size=size) + 1j * rng.normal(size=size)

    mean = normal(shape)
    noise = normal((n_configs,) + shape) * diffuse_fraction
    array = (mean[None] + noise).astype(np.complex64)
    if lazy:
        array = da.from_array(array, chunks=(1, 1) + GPTS)
    return Waves(
        array,
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[
            FrozenPhononsAxis(_ensemble_mean=False),
            OrdinalAxis(label="member", values=tuple(range(N_MEMBERS))),
        ],
    )


def _reference(waves, unbiased=False):
    """(total, elastic, diffuse) from the complex amplitudes, in float64."""
    amplitudes = waves.diffraction_patterns(return_complex=True, **DP_KWARGS).array
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    n = amplitudes.shape[0]
    mean = amplitudes.mean(axis=0)
    total = (np.abs(amplitudes) ** 2).mean(axis=0)
    elastic = np.abs(mean) ** 2
    diffuse = total - elastic
    if unbiased:
        diffuse = (np.abs(amplitudes - mean) ** 2).sum(axis=0) / (n - 1)
        elastic = total - diffuse
    return {"total": total, "elastic": elastic, "diffuse": diffuse}


def _float64(waves):
    return Waves(
        np.asarray(waves.array).astype(np.complex128),
        energy=waves.energy,
        sampling=waves.sampling,
        ensemble_axes_metadata=waves.ensemble_axes_metadata,
    )


def _max_error(result, reference):
    return float(np.abs(result - reference).max() / np.abs(reference).max())


def test_components_match_the_reference():
    waves = _exit_waves()
    result = elastic_diffuse_diffraction_patterns(waves, **HELPER_KWARGS)
    reference = _reference(waves)

    assert result.shape == (3, N_MEMBERS) + result.base_shape
    assert result.ensemble_axes_metadata[0].values == ("total", "elastic", "diffuse")
    assert isinstance(result.ensemble_axes_metadata[1], OrdinalAxis)
    for i, name in enumerate(("total", "elastic", "diffuse")):
        assert _max_error(result.array[i], reference[name]) < 1e-5


def test_each_member_of_another_ensemble_axis_gets_its_own_components():
    waves = _exit_waves()
    result = elastic_diffuse_diffraction_patterns(waves, **HELPER_KWARGS)
    for member in range(N_MEMBERS):
        alone = elastic_diffuse_diffraction_patterns(waves[:, member], **HELPER_KWARGS)
        np.testing.assert_allclose(result.array[:, member], alone.array, rtol=1e-6)


def test_components_follow_the_order_given_and_a_single_name_has_no_axis():
    waves = _exit_waves()
    stacked = elastic_diffuse_diffraction_patterns(
        waves, components=("diffuse", "total")
    )
    single = elastic_diffuse_diffraction_patterns(waves, components="elastic")
    everything = elastic_diffuse_diffraction_patterns(waves)

    assert stacked.ensemble_axes_metadata[0].values == ("diffuse", "total")
    np.testing.assert_array_equal(stacked.array[0], everything.array[2])
    np.testing.assert_array_equal(stacked.array[1], everything.array[0])

    assert single.shape == everything.shape[1:]
    np.testing.assert_array_equal(single.array, everything.array[1])
    assert not any(
        isinstance(axis, OrdinalAxis) and axis.label == "component"
        for axis in single.ensemble_axes_metadata
    )


def test_metadata_records_the_components_and_the_estimator():
    waves = _exit_waves()
    everything = elastic_diffuse_diffraction_patterns(waves)
    single = elastic_diffuse_diffraction_patterns(
        waves, components="diffuse", unbiased=True
    )

    assert everything.metadata["frozen_phonon_component"] == [
        "total",
        "elastic",
        "diffuse",
    ]
    assert everything.metadata["num_configurations"] == N_CONFIGS
    assert everything.metadata["unbiased"] is False
    assert single.metadata["frozen_phonon_component"] == "diffuse"
    assert single.metadata["unbiased"] is True


def test_unbiased_estimator_matches_the_sample_variance():
    waves = _float64(_exit_waves())
    result = elastic_diffuse_diffraction_patterns(waves, unbiased=True, **HELPER_KWARGS)
    biased = elastic_diffuse_diffraction_patterns(waves, **HELPER_KWARGS)
    reference = _reference(waves, unbiased=True)
    n = N_CONFIGS

    for i, name in enumerate(("total", "elastic", "diffuse")):
        assert _max_error(result.array[i], reference[name]) < 1e-12
    total, elastic, _ = biased.array
    np.testing.assert_allclose(
        result.array[1],
        (n * elastic - total) / (n - 1),
        rtol=0,
        atol=1e-12 * total.max(),
    )


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_float64_reduction_keeps_a_small_diffuse_part(lazy):
    """Where the diffuse part is small next to the total, it is a difference of two
    nearly equal intensities: float32 moments lose most of it, float64 ones keep it.
    The reference is formed in float64 from the same float32 waves."""
    eager = _exit_waves(diffuse_fraction=3e-3)
    reference = _reference(_float64(eager))["diffuse"]
    assert reference.max() / _reference(_float64(eager))["total"].max() < 1e-4
    waves = _exit_waves(diffuse_fraction=3e-3, lazy=lazy)

    default = elastic_diffuse_diffraction_patterns(waves, components="diffuse")
    float64 = elastic_diffuse_diffraction_patterns(
        waves, components="diffuse", reduction_dtype="float64"
    )

    assert float64.array.dtype == np.float64
    assert default.array.dtype == np.float32
    float64_array, default_array = (
        np.asarray(x.array.compute() if lazy else x.array) for x in (float64, default)
    )
    assert float64_array.dtype == np.float64
    float64_error = _max_error(float64_array, reference)
    assert float64_error < 1e-9
    assert _max_error(default_array, reference) > 100 * float64_error


def test_lazy_exit_waves_give_the_eager_result():
    eager = elastic_diffuse_diffraction_patterns(_exit_waves())
    lazy = elastic_diffuse_diffraction_patterns(_exit_waves(lazy=True))

    assert isinstance(lazy.array, da.core.Array)
    np.testing.assert_allclose(lazy.array.compute(), eager.array, rtol=1e-5)


def test_waves_method_matches_the_function():
    waves = _exit_waves()
    np.testing.assert_array_equal(
        waves.elastic_diffuse_diffraction_patterns(components="diffuse").array,
        elastic_diffuse_diffraction_patterns(waves, components="diffuse").array,
    )


@pytest.mark.parametrize(
    "components, error, match",
    [
        ({"total"}, TypeError, "components must be one of"),
        (3, TypeError, "components must be one of"),
        ((), ValueError, "at least one"),
        (("total", "total"), ValueError, "twice"),
        (("all", "total"), ValueError, "'all' cannot be combined"),
        (("total", 1), TypeError, "must be strings"),
        ("bogus", ValueError, "components must be one of"),
        ("tds", ValueError, "'tds' is now 'diffuse'"),
        (("coherent",), ValueError, "'coherent' is now 'elastic'"),
        ("incoherent", ValueError, "'incoherent' is now 'total'"),
        (None, TypeError, "components must be one of"),
    ],
)
def test_invalid_components_raise(components, error, match):
    with pytest.raises(error, match=match):
        elastic_diffuse_diffraction_patterns(_exit_waves(), components=components)


@pytest.mark.parametrize("unbiased", [1, None, "yes"])
def test_unbiased_must_be_a_bool(unbiased):
    with pytest.raises(TypeError, match="unbiased must be a bool"):
        elastic_diffuse_diffraction_patterns(_exit_waves(), unbiased=unbiased)


@pytest.mark.parametrize(
    "reduction_dtype, error",
    [
        ("int32", ValueError),
        ("float16", ValueError),
        (-1, TypeError),
        (object(), TypeError),
    ],
)
def test_reduction_dtype_must_be_float32_or_float64(reduction_dtype, error):
    with pytest.raises(error, match="reduction_dtype must be float32 or float64"):
        elastic_diffuse_diffraction_patterns(
            _exit_waves(), reduction_dtype=reduction_dtype
        )


def test_exit_waves_need_a_frozen_phonon_axis():
    waves = _exit_waves()
    without = Waves(
        np.asarray(waves.array)[0],
        energy=waves.energy,
        sampling=waves.sampling,
        ensemble_axes_metadata=waves.ensemble_axes_metadata[1:],
    )
    with pytest.raises(ValueError, match="FrozenPhononsAxis"):
        elastic_diffuse_diffraction_patterns(without)


@pytest.mark.parametrize(
    "kwargs", [dict(components="diffuse"), dict(components="all"), dict(unbiased=True)]
)
def test_a_single_configuration_raises_where_the_diffuse_part_is_needed(kwargs):
    with pytest.raises(ValueError, match="at least 2 frozen-phonon"):
        elastic_diffuse_diffraction_patterns(_exit_waves(n_configs=1), **kwargs)


def test_a_single_configuration_gives_elastic_equal_to_total():
    waves = _exit_waves(n_configs=1)
    elastic = elastic_diffuse_diffraction_patterns(waves, components="elastic")
    total = elastic_diffuse_diffraction_patterns(waves, components="total")
    np.testing.assert_allclose(elastic.array, total.array, rtol=1e-6)


def test_total_of_a_frozen_phonon_scan_matches_the_pixelated_detector():
    """An independent path for the total: the ensemble mean of a PixelatedDetector
    over the same frozen-phonon configurations."""
    import ase.build

    abtem.config.set({"device": "cpu"})
    atoms = ase.build.mx2("MoS2", vacuum=2)
    phonons = abtem.FrozenPhonons(
        atoms, num_configs=N_CONFIGS, sigmas=0.1, seed=3, ensemble_mean=False
    )
    potential = abtem.Potential(phonons, sampling=0.1, slice_thickness=2)
    probe = abtem.Probe(energy=80e3, semiangle_cutoff=20)
    probe.grid.match(potential)

    exit_waves = probe.multislice(
        potential, detectors=abtem.WavesDetector(), lazy=False
    )
    total = elastic_diffuse_diffraction_patterns(
        exit_waves, components="total", max_angle=60, parity="same"
    )

    phonons_mean = abtem.FrozenPhonons(atoms, num_configs=N_CONFIGS, sigmas=0.1, seed=3)
    potential_mean = abtem.Potential(phonons_mean, sampling=0.1, slice_thickness=2)
    detected = probe.multislice(
        potential_mean, detectors=abtem.PixelatedDetector(max_angle=60), lazy=False
    )

    assert total.shape == detected.shape
    assert _max_error(np.asarray(total.array), np.asarray(detected.array)) < 1e-5
