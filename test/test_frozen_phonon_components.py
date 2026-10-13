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
from abtem.core.axes import EnergyLossAxis, FrozenPhononsAxis, OrdinalAxis
from abtem.measurements import (
    elastic_diffuse_diffraction_patterns,
    phonon_loss_diffraction_patterns,
)
from abtem.waves import Waves

N_CONFIGS = 3
N_MEMBERS = 2
GPTS = (16, 20)
HELPER_KWARGS = dict(max_angle="cutoff", parity="odd")
# The helper always returns centred (fftshift) patterns.
DP_KWARGS = dict(**HELPER_KWARGS, fftshift=True)


def _exit_waves(
    diffuse_fraction=1.0,
    n_configs=N_CONFIGS,
    lazy=False,
    seed=1,
    gpts=GPTS,
    metadata=None,
):
    """Configurations ψ_j = μ + noise_j, complex64, with a frozen-phonon axis followed
    by a second ensemble axis; ``diffuse_fraction`` scales the noise."""
    rng = np.random.default_rng(seed)
    shape = (N_MEMBERS,) + gpts

    def normal(size):
        return rng.normal(size=size) + 1j * rng.normal(size=size)

    mean = normal(shape)
    noise = normal((n_configs,) + shape) * diffuse_fraction
    array = (mean[None] + noise).astype(np.complex64)
    if lazy:
        array = da.from_array(array, chunks=(1, 1) + gpts)
    return Waves(
        array,
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[
            FrozenPhononsAxis(_ensemble_mean=False),
            OrdinalAxis(label="member", values=tuple(range(N_MEMBERS))),
        ],
        metadata=dict(metadata or {}),
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


@pytest.mark.parametrize("components", ["all", "total", "elastic", "diffuse"])
def test_lazy_exit_waves_give_the_eager_result(components):
    eager = elastic_diffuse_diffraction_patterns(_exit_waves(), components=components)
    lazy = elastic_diffuse_diffraction_patterns(
        _exit_waves(lazy=True), components=components
    )

    assert isinstance(lazy.array, da.core.Array)
    np.testing.assert_allclose(lazy.array.compute(), eager.array, rtol=1e-5)


@pytest.mark.parametrize(
    "kwargs, formed",
    [
        (dict(components="elastic"), ["mean"]),
        (dict(components="total"), ["configurations"]),
        (dict(components=("total",), unbiased=True), ["configurations"]),
        (dict(components="diffuse"), ["mean", "configurations"]),
        (dict(components="all"), ["mean", "configurations"]),
        (dict(components="elastic", unbiased=True), ["mean", "configurations"]),
    ],
)
def test_only_the_moments_the_components_need_are_formed(monkeypatch, kwargs, formed):
    """The elastic component needs only the patterns of the mean wave, the total only
    those of the configurations; the diffuse component, and the elastic one with
    unbiased, need both."""
    waves = _exit_waves()
    everything = elastic_diffuse_diffraction_patterns(
        waves, unbiased=kwargs.get("unbiased", False)
    )
    names = {(N_MEMBERS,): "mean", (N_CONFIGS, N_MEMBERS): "configurations"}
    calls = []
    diffraction_patterns = Waves.diffraction_patterns

    def spy(self, *args, **spy_kwargs):
        calls.append(names[self.ensemble_shape])
        return diffraction_patterns(self, *args, **spy_kwargs)

    monkeypatch.setattr(Waves, "diffraction_patterns", spy)
    result = elastic_diffuse_diffraction_patterns(waves, **kwargs)
    monkeypatch.undo()

    assert calls == formed
    components = kwargs["components"]
    order = ("total", "elastic", "diffuse")
    if isinstance(components, str) and components != "all":
        np.testing.assert_array_equal(
            result.array, everything.array[order.index(components)]
        )
    else:
        selected = order if components == "all" else components
        np.testing.assert_array_equal(
            result.array, everything.array[[order.index(c) for c in selected]]
        )


def test_metadata_does_not_depend_on_the_moments_formed():
    waves = _exit_waves(metadata={"semiangle_cutoff": 20.0})
    results = {
        name: elastic_diffuse_diffraction_patterns(waves, components=name)
        for name in ("total", "elastic", "diffuse")
    }
    for name, result in results.items():
        metadata = dict(result.metadata)
        assert metadata.pop("frozen_phonon_component") == name
        assert metadata == {
            **waves.metadata,
            "label": "intensity",
            "units": "arb. unit",
            "num_configurations": N_CONFIGS,
            "unbiased": False,
        }
        assert result.sampling == results["diffuse"].sampling
        assert result.fftshift


def test_waves_method_matches_the_function():
    waves = _exit_waves()
    np.testing.assert_array_equal(
        waves.elastic_diffuse_diffraction_patterns(components="diffuse").array,
        elastic_diffuse_diffraction_patterns(waves, components="diffuse").array,
    )


@pytest.mark.parametrize("block_direct", [True, np.True_], ids=["bool", "numpy_bool"])
def test_block_direct_true_blocks_up_to_the_semiangle_cutoff(block_direct):
    """block_direct=True blocks what DiffractionPatterns.block_direct() blocks: with a
    20 mrad semiangle cutoff in the metadata, the bright-field disk and a margin of one
    pixel. Per pattern, a radius of 1 mrad (True taken as a number) blocks 5 pixels
    here, the default 51 (angular sampling 7.7 x 6.2 mrad)."""
    waves = _exit_waves(gpts=(48, 60), metadata={"semiangle_cutoff": 20.0})
    unblocked = elastic_diffuse_diffraction_patterns(waves, components="total")
    expected = unblocked.block_direct()
    one_mrad = unblocked.block_direct(radius=1.0)
    assert (expected.array == 0).sum() > 5 * (one_mrad.array == 0).sum()

    blocked = elastic_diffuse_diffraction_patterns(
        waves, components="total", block_direct=block_direct
    )

    np.testing.assert_array_equal(blocked.array, expected.array)


def _zeroed_pixels(blocked, unblocked):
    """The pixels (row, column) zeroed by blocking, in any pattern of the stack."""
    blocked, unblocked = np.asarray(blocked.array), np.asarray(unblocked.array)
    zeroed = (blocked == 0) & (unblocked != 0)
    zeroed = zeroed.reshape((-1,) + zeroed.shape[-2:]).any(axis=0)
    return [tuple(int(i) for i in index) for index in np.argwhere(zeroed)]


@pytest.mark.parametrize("block_direct", [True, np.True_], ids=["bool", "numpy_bool"])
@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"semiangle_cutoff": 0.0},
        {"semiangle_cutoff": 1e-6},
        {"semiangle_cutoff": 1e-3},
        {"semiangle_cutoff": 1.0},
        {"semiangle_cutoff": np.inf},
        {"semiangle_cutoff": np.array([10.0, 20.0])},
    ],
    ids=["no_cutoff", "parallel_beam", "1e-6", "1e-3", "1.0", "no_aperture", "array"],
)
def test_block_direct_true_without_a_cutoff_blocks_the_zero_angle_pixel(
    metadata, block_direct
):
    """Without a semiangle cutoff, or with one smaller than the angular sampling
    (7.7 x 6.2 mrad here) or an infinite one, the direct beam is the zero-angle pixel
    alone, and only that pixel is blocked."""
    waves = _exit_waves(gpts=(48, 60), metadata=metadata)
    unblocked = elastic_diffuse_diffraction_patterns(waves)
    center = tuple(n // 2 for n in unblocked.base_shape)

    blocked = elastic_diffuse_diffraction_patterns(waves, block_direct=block_direct)

    assert _zeroed_pixels(blocked, unblocked) == [center]
    np.testing.assert_array_equal(
        np.asarray(blocked.array)[..., center[0], center[1]], 0.0
    )
    keep = np.ones(unblocked.base_shape, dtype=bool)
    keep[center] = False
    np.testing.assert_array_equal(
        np.asarray(blocked.array)[..., keep], np.asarray(unblocked.array)[..., keep]
    )


@pytest.mark.parametrize("semiangle_cutoff", [float("nan"), -5.0])
def test_block_direct_true_with_an_invalid_cutoff_raises(semiangle_cutoff):
    waves = _exit_waves(metadata={"semiangle_cutoff": semiangle_cutoff})
    with pytest.raises(ValueError, match="must be non-negative"):
        elastic_diffuse_diffraction_patterns(waves, block_direct=True)


def _sweep_grids():
    """(extent, gpts, max_angle): one SrTiO3 cell and a 5 x 5 supercell over
    gpts 32 to 300, and random grids."""
    grids = [
        ((extent, extent), (n, n), "cutoff")
        for extent in (3.905, 19.525)
        for n in range(32, 301)
    ]
    rng = np.random.default_rng(0)
    max_angles = ["cutoff", "valid", "full", 10.0, 50.0]
    while len(grids) < 2 * 269 + 30:
        extent = tuple(float(x) for x in rng.uniform(2.0, 40.0, 2))
        gpts = tuple(int(x) for x in rng.integers(16, 200, 2))
        grids.append((extent, gpts, max_angles[rng.integers(len(max_angles))]))
    return grids


def _sweep_waves(extent, gpts, energy_axis=False):
    """Two configurations, non-zero everywhere, without a semiangle cutoff."""
    rng = np.random.default_rng(sum(gpts))
    axes = [FrozenPhononsAxis(_ensemble_mean=False)]
    shape = (2,) + gpts
    if energy_axis:
        axes = [EnergyLossAxis(values=(0.02,))] + axes
        shape = (1,) + shape
    array = (5 + rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex64
    )
    return Waves(array, energy=200e3, extent=extent, ensemble_axes_metadata=axes)


_SWEEP_FUNCTIONS = {
    "waves": lambda waves, **kwargs: waves[0].diffraction_patterns(**kwargs),
    "elastic_diffuse": lambda waves, **kwargs: elastic_diffuse_diffraction_patterns(
        waves, components="total", **kwargs
    ),
    "phonon_loss": lambda waves, **kwargs: phonon_loss_diffraction_patterns(
        waves, components="total", **kwargs
    ),
}


@pytest.mark.parametrize("function", list(_SWEEP_FUNCTIONS))
def test_block_direct_true_blocks_the_zero_angle_pixel_on_any_grid(function):
    """Without a semiangle cutoff, block_direct=True blocks exactly the zero-angle
    pixel. Its float32 coordinate carries roundoff on many grids (e.g. -2.8e-14
    mrad), so a radius of 0 would leave it, and a radius of one sampling step would
    reach its neighbours."""
    patterns_of = _SWEEP_FUNCTIONS[function]
    wrong = []
    # NumPy's FFT needs no plan for each of the several hundred grid shapes.
    with abtem.config.set({"fft": "numpy"}):
        for extent, gpts, max_angle in _sweep_grids():
            waves = _sweep_waves(extent, gpts, energy_axis=function == "phonon_loss")
            unblocked = patterns_of(waves, max_angle=max_angle)
            if min(unblocked.base_shape) < 3:
                continue
            blocked = patterns_of(waves, max_angle=max_angle, block_direct=True)
            center = tuple(n // 2 for n in unblocked.base_shape)
            if _zeroed_pixels(blocked, unblocked) != [center]:
                wrong.append((extent, gpts, max_angle))
    assert wrong == []


def _probe_waves(extent, gpts, semiangle_cutoff, energy_axis=False):
    """Two identical configurations of a probe in vacuum (soft aperture), whose
    diffraction patterns are the direct beam alone."""
    probe = abtem.Probe(
        energy=200e3, extent=extent, gpts=gpts, semiangle_cutoff=semiangle_cutoff
    )
    with abtem.config.set({"device": "cpu"}):
        built = probe.build(lazy=False)
    array = np.asarray(built.array)
    axes = [FrozenPhononsAxis(_ensemble_mean=False)]
    array = np.stack([array, array])
    if energy_axis:
        axes = [EnergyLossAxis(values=(0.02,))] + axes
        array = array[None]
    return Waves(
        array,
        energy=200e3,
        extent=extent,
        ensemble_axes_metadata=axes,
        metadata=dict(built.metadata),
    )


@pytest.mark.parametrize("fraction", [0.3, 0.5, 0.7, 0.9, 0.99, 1.0, 1.5])
@pytest.mark.parametrize(
    "extent, gpts",
    [((19.525, 19.525), (128, 128)), ((19.525, 9.7625), (128, 64))],
    ids=["isotropic", "anisotropic"],
)
@pytest.mark.parametrize("function", list(_SWEEP_FUNCTIONS))
def test_block_direct_true_blocks_a_small_soft_aperture(
    function, extent, gpts, fraction
):
    """A soft aperture reaches the nearest pixels once the cutoff exceeds half the
    angular sampling; block_direct=True must still leave no direct-beam intensity.
    The cutoff is a fraction of the smaller angular sampling (1.28 mrad)."""
    sampling = abtem.Probe(energy=200e3, extent=extent, gpts=gpts).angular_sampling
    waves = _probe_waves(
        extent,
        gpts,
        fraction * min(sampling),
        energy_axis=function == "phonon_loss",
    )
    patterns_of = _SWEEP_FUNCTIONS[function]
    with abtem.config.set({"device": "cpu"}):
        unblocked = np.asarray(patterns_of(waves, max_angle="full").array)
        blocked = np.asarray(
            patterns_of(waves, max_angle="full", block_direct=True).array
        )
    assert np.abs(blocked).sum() < 1e-6 * np.abs(unblocked).sum()


def test_block_direct_true_keeps_the_first_order_reflections_of_a_plane_wave():
    """In a one-unit-cell plane-wave pattern the pixels next to the zero-angle pixel
    are the (100) and (010) reflections; block_direct=True keeps them."""
    from ase import Atoms

    a = 3.905
    atoms = Atoms(
        "SrTiO3",
        scaled_positions=[
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ],
        cell=[a, a, a],
        pbc=True,
    ) * (1, 1, 4)
    with abtem.config.set({"device": "cpu"}):
        phonons = abtem.FrozenPhonons(
            atoms, num_configs=2, sigmas=0.05, seed=1, ensemble_mean=False
        )
        potential = abtem.Potential(phonons, gpts=(48, 48), slice_thickness=a / 2)
        exit_waves = abtem.PlaneWave(energy=200e3).multislice(potential, lazy=False)
    unblocked = elastic_diffuse_diffraction_patterns(exit_waves, components="total")
    center = tuple(n // 2 for n in unblocked.base_shape)
    first_order = [(center[0] + 1, center[1]), (center[0], center[1] + 1)]
    unblocked_array = np.asarray(unblocked.array)
    assert min(unblocked_array[index] for index in first_order) > 0

    blocked = elastic_diffuse_diffraction_patterns(
        exit_waves, components="total", block_direct=True
    )

    assert _zeroed_pixels(blocked, unblocked) == [center]
    for index in first_order:
        assert np.asarray(blocked.array)[index] == unblocked_array[index]


@pytest.mark.parametrize("components", [("all",), ["all"]], ids=["tuple", "list"])
def test_a_sequence_of_all_alone_is_all(components):
    waves = _exit_waves()
    result = elastic_diffuse_diffraction_patterns(waves, components=components)
    everything = elastic_diffuse_diffraction_patterns(waves, components="all")
    assert result.ensemble_axes_metadata[0].values == ("total", "elastic", "diffuse")
    assert result.metadata["frozen_phonon_component"] == [
        "total",
        "elastic",
        "diffuse",
    ]
    np.testing.assert_array_equal(result.array, everything.array)


def test_component_names_are_recorded_as_plain_str():
    waves = _exit_waves()
    single = elastic_diffuse_diffraction_patterns(waves, components=np.str_("total"))
    stacked = elastic_diffuse_diffraction_patterns(
        waves, components=(np.str_("diffuse"), np.str_("total"))
    )

    assert type(single.metadata["frozen_phonon_component"]) is str
    assert [type(name) for name in stacked.metadata["frozen_phonon_component"]] == [
        str,
        str,
    ]
    assert [type(name) for name in stacked.ensemble_axes_metadata[0].values] == [
        str,
        str,
    ]


@pytest.mark.parametrize(
    "components, error, match",
    [
        ({"total"}, TypeError, "components must be one of"),
        (3, TypeError, "components must be one of"),
        ((), ValueError, "at least one"),
        (("total", "total"), ValueError, "twice"),
        (("all", "total"), ValueError, "'all' cannot be combined"),
        (("all", "all"), ValueError, "'all' more than once"),
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


def test_unbiased_accepts_a_numpy_bool():
    waves = _exit_waves()
    result = elastic_diffuse_diffraction_patterns(waves, unbiased=np.True_)
    expected = elastic_diffuse_diffraction_patterns(waves, unbiased=True)
    np.testing.assert_array_equal(result.array, expected.array)
    assert result.metadata["unbiased"] is True


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
    "kwargs",
    [
        dict(components="diffuse"),
        dict(components="all"),
        dict(unbiased=True),
        dict(components="elastic", unbiased=True),
        dict(components=("total", "elastic"), unbiased=True),
    ],
)
def test_a_single_configuration_raises_where_the_diffuse_part_is_needed(kwargs):
    with pytest.raises(ValueError, match="at least 2 frozen-phonon"):
        elastic_diffuse_diffraction_patterns(_exit_waves(n_configs=1), **kwargs)


def test_unbiased_total_of_a_single_configuration_is_the_total():
    """unbiased changes only the elastic and diffuse components, so the total of a
    single configuration is available with it."""
    waves = _exit_waves(n_configs=1)
    total = elastic_diffuse_diffraction_patterns(waves, components="total")
    unbiased = elastic_diffuse_diffraction_patterns(
        waves, components="total", unbiased=True
    )
    np.testing.assert_array_equal(unbiased.array, total.array)


def test_a_single_configuration_gives_elastic_equal_to_total():
    waves = _exit_waves(n_configs=1)
    elastic = elastic_diffuse_diffraction_patterns(waves, components="elastic")
    total = elastic_diffuse_diffraction_patterns(waves, components="total")
    np.testing.assert_allclose(elastic.array, total.array, rtol=1e-6)


def test_total_of_a_frozen_phonon_scan_matches_the_pixelated_detector():
    """An independent path for the total: the ensemble mean of a PixelatedDetector
    over the same frozen-phonon configurations."""
    import ase.build

    atoms = ase.build.mx2("MoS2", vacuum=2)
    with abtem.config.set({"device": "cpu"}):
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

        phonons_mean = abtem.FrozenPhonons(
            atoms, num_configs=N_CONFIGS, sigmas=0.1, seed=3
        )
        potential_mean = abtem.Potential(phonons_mean, sampling=0.1, slice_thickness=2)
        detected = probe.multislice(
            potential_mean, detectors=abtem.PixelatedDetector(max_angle=60), lazy=False
        )

    assert total.shape == detected.shape
    assert _max_error(np.asarray(total.array), np.asarray(detected.array)) < 1e-5
