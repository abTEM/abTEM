"""Tests for phonon_loss_diffraction_patterns."""

import dask.array as da
import numpy as np
import pytest
from ase import units

from abtem.core.axes import OrdinalAxis, EnergyLossAxis, FrozenPhononsAxis, PhononParityAxis
from abtem.measurements import phonon_loss_diffraction_patterns
from abtem.waves import Waves


def _make_exit_waves(e_values, n_configs=6, gpts=24, seed=0, lazy=False, metadata=None):
    rng = np.random.default_rng(seed)
    n_energies = len(e_values)
    array = (
        rng.normal(size=(n_energies, n_configs, gpts, gpts))
        + 1j * rng.normal(size=(n_energies, n_configs, gpts, gpts))
    ).astype(np.complex64)
    if lazy:
        array = da.from_array(array, chunks=(1, 1, gpts, gpts))
    return Waves(
        array,
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[
            EnergyLossAxis(values=tuple(float(e) for e in e_values)),
            FrozenPhononsAxis(_ensemble_mean=False),
        ],
        metadata=dict(metadata or {}),
    )


def _make_parity_exit_waves(
    e_values, n_configs=6, gpts=24, seed=0, lazy=False, real=None, twin=None,
):
    """Build exit_waves carrying a ("real", "twin") PhononParityAxis, as
    `multislice()` on a parity_projection=True ensemble returns them."""
    rng = np.random.default_rng(seed)
    n_energies = len(e_values)
    shape = (n_energies, n_configs, gpts, gpts)

    def _random_complex():
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex64
        )

    if real is None:
        real = _random_complex()
    if twin is None:
        twin = _random_complex()

    array = np.stack([real, twin], axis=0)
    if lazy:
        array = da.from_array(array, chunks=(1, 1, 1, gpts, gpts))
    return Waves(
        array,
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[
            PhononParityAxis(values=("real", "twin")),
            EnergyLossAxis(values=tuple(float(e) for e in e_values)),
            FrozenPhononsAxis(_ensemble_mean=False),
        ],
    )


def test_components_are_consistent():
    waves = _make_exit_waves([0.02, 0.05, 0.10])

    dp_diffuse = phonon_loss_diffraction_patterns(waves, components="diffuse")
    dp_elastic = phonon_loss_diffraction_patterns(waves, components="elastic")
    dp_total = phonon_loss_diffraction_patterns(waves, components="total")
    dp_all = phonon_loss_diffraction_patterns(waves, components="all")

    assert dp_all.ensemble_axes_metadata[0].values == ("total", "elastic", "diffuse")
    assert np.allclose(dp_all.array[0], dp_total.array)
    assert np.allclose(dp_all.array[1], dp_elastic.array)
    assert np.allclose(dp_all.array[2], dp_diffuse.array)
    assert np.allclose(dp_diffuse.array, dp_total.array - dp_elastic.array)

    for dp, name in [
        (dp_diffuse, "diffuse"),
        (dp_elastic, "elastic"),
        (dp_total, "total"),
    ]:
        assert dp.metadata["frozen_phonon_component"] == name
        assert dp.metadata["energy"] == 100e3
    assert dp_all.metadata["frozen_phonon_component"] == ["total", "elastic", "diffuse"]


def test_default_is_the_diffuse_component():
    waves = _make_exit_waves([0.02, 0.05, 0.10])
    np.testing.assert_array_equal(
        phonon_loss_diffraction_patterns(waves).array,
        phonon_loss_diffraction_patterns(waves, components="diffuse").array,
    )


def test_invalid_component_raises():
    waves = _make_exit_waves([0.02, 0.05, 0.10])
    with pytest.raises(ValueError, match="components must be one of"):
        phonon_loss_diffraction_patterns(waves, components="bogus")


def test_invalid_components_raise_for_parity_projected_input():
    """components is ignored (not applicable) once exit_waves carries a
    PhononParityAxis, but an invalid or renamed value must still be rejected
    rather than silently dropped -- the parity path must not bypass this check
    just because it doesn't use the value."""
    waves = _make_parity_exit_waves([0.02, 0.05])
    with pytest.raises(ValueError, match="components must be one of"):
        phonon_loss_diffraction_patterns(waves, components="bogus")
    with pytest.raises(ValueError, match="'tds' is now 'diffuse'"):
        phonon_loss_diffraction_patterns(waves, components="tds")


@pytest.mark.parametrize("kwargs", [{"unbiased": True}, {"reduction_dtype": "float64"}])
def test_component_options_raise_for_parity_projected_input(kwargs):
    """unbiased and reduction_dtype shape the elastic/diffuse split, which the
    parity path does not form; passing them must raise rather than be ignored."""
    waves = _make_parity_exit_waves([0.02, 0.05])
    with pytest.raises(ValueError, match="do not apply to a parity-projected"):
        phonon_loss_diffraction_patterns(waves, **kwargs)


@pytest.mark.parametrize(
    "old, new", [("tds", "diffuse"), ("coherent", "elastic"), ("incoherent", "total")]
)
def test_earlier_component_names_point_to_the_new_ones(old, new):
    waves = _make_exit_waves([0.02, 0.05, 0.10])
    with pytest.raises(ValueError, match=f"'{old}' is now '{new}'"):
        phonon_loss_diffraction_patterns(waves, components=old)


@pytest.mark.parametrize("components", ["diffuse", "all"])
def test_single_config_diffuse_raises_instead_of_returning_zeros(components):
    """With one frozen-phonon configuration, the total and the elastic intensity
    are identical by construction, so the diffuse intensity is identically zero
    everywhere -- not a bug, but silently returning an all-zero array looks
    exactly like one. This must raise instead."""
    waves = _make_exit_waves([0.02, 0.05, 0.10], n_configs=1)
    with pytest.raises(ValueError, match="at least 2 frozen-phonon"):
        phonon_loss_diffraction_patterns(waves, components=components)


@pytest.mark.parametrize("components", ["elastic", "total"])
def test_single_config_elastic_and_total_still_work(components):
    """elastic/total are well-defined (if trivial) for a single configuration
    and must not be blocked by the N>=2 check."""
    waves = _make_exit_waves([0.02, 0.05, 0.10], n_configs=1)
    dp = phonon_loss_diffraction_patterns(waves, components=components)
    assert not np.allclose(dp.array, 0.0)


def test_two_configs_diffuse_does_not_raise():
    waves = _make_exit_waves([0.02, 0.05, 0.10], n_configs=2)
    dp = phonon_loss_diffraction_patterns(waves, components="diffuse")
    assert dp.array.shape[0] == 3


def _computed(measurement):
    array = measurement.array
    return np.asarray(array.compute() if isinstance(array, da.core.Array) else array)


def test_the_last_frozen_phonons_axis_is_reduced():
    """With two FrozenPhononsAxis, the last one is reduced and the first is kept."""
    rng = np.random.default_rng(0)
    shape = (2, 3, 4, 16, 16)
    array = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    waves = Waves(
        array,
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[
            EnergyLossAxis(values=(0.02, 0.05)),
            FrozenPhononsAxis(_ensemble_mean=False),
            FrozenPhononsAxis(_ensemble_mean=False),
        ],
    )

    result = phonon_loss_diffraction_patterns(waves)

    assert result.shape[:2] == (2, 3)
    assert [type(axis) for axis in result.ensemble_axes_metadata] == [
        EnergyLossAxis,
        FrozenPhononsAxis,
    ]
    for k in range(3):
        alone = Waves(
            array[:, k],
            energy=100e3,
            sampling=0.1,
            ensemble_axes_metadata=[
                EnergyLossAxis(values=(0.02, 0.05)),
                FrozenPhononsAxis(_ensemble_mean=False),
            ],
        )
        np.testing.assert_array_equal(
            np.asarray(result.array)[:, k],
            np.asarray(phonon_loss_diffraction_patterns(alone).array),
        )


class TestThermalWeighting:
    def test_signed_axis_and_zero_bin_passthrough(self):
        e_values = [0.0, 0.02, 0.05, 0.10]
        waves = _make_exit_waves(e_values)

        dp_unweighted = phonon_loss_diffraction_patterns(waves, components="diffuse")
        dp_weighted = phonon_loss_diffraction_patterns(
            waves, components="diffuse", temperature=300.0
        )

        energy_axis = next(
            ax
            for ax in dp_weighted.ensemble_axes_metadata
            if isinstance(ax, EnergyLossAxis)
        )
        signed_e = np.array(energy_axis.values)

        expected = np.array([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10])
        assert np.allclose(signed_e, expected)
        assert dp_weighted.array.shape[0] == 2 * len(e_values) - 1

        zero_old = e_values.index(0.0)
        zero_new = list(signed_e).index(0.0)
        assert np.allclose(dp_weighted.array[zero_new], dp_unweighted.array[zero_old])

    def test_detailed_balance_conservation(self):
        e_values = [0.0, 0.02, 0.05, 0.10]
        T = 300.0
        waves = _make_exit_waves(e_values)

        dp_unweighted = phonon_loss_diffraction_patterns(waves, components="diffuse")
        dp_weighted = phonon_loss_diffraction_patterns(
            waves, components="diffuse", temperature=T
        )

        energy_axis = next(
            ax
            for ax in dp_weighted.ensemble_axes_metadata
            if isinstance(ax, EnergyLossAxis)
        )
        signed_e = list(energy_axis.values)

        beta = 1.0 / (units.kB * T)
        for i, E in enumerate(e_values):
            if E == 0.0:
                continue
            n_occ = 1.0 / (np.exp(E * beta) - 1.0)
            loss_weight = (n_occ + 1.0) / (2 * n_occ + 1.0)
            gain_weight = n_occ / (2 * n_occ + 1.0)

            idx_loss = signed_e.index(E)
            idx_gain = signed_e.index(-E)

            assert np.allclose(
                dp_weighted.array[idx_loss], dp_unweighted.array[i] * loss_weight
            )
            assert np.allclose(
                dp_weighted.array[idx_gain], dp_unweighted.array[i] * gain_weight
            )
            # loss + gain must reconstruct the original (unweighted) signal
            assert np.allclose(
                dp_weighted.array[idx_loss] + dp_weighted.array[idx_gain],
                dp_unweighted.array[i],
            )

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    @pytest.mark.parametrize(
        "metadata",
        [
            {},
            {"semiangle_cutoff": 0.0},
            {"semiangle_cutoff": 1e-3},
            {"semiangle_cutoff": np.inf},
        ],
        ids=["no_cutoff", "parallel_beam", "1e-3", "no_aperture"],
    )
    def test_block_direct_true_without_a_cutoff_blocks_the_zero_angle_pixel(
        self, metadata, lazy
    ):
        """Without a semiangle cutoff, or with one smaller than the angular sampling
        or an infinite one, block_direct=True blocks the zero-angle pixel of every
        unfolded pattern and nothing else."""
        waves = _make_exit_waves(
            [0.0, 0.02, 0.05, 0.10], gpts=48, lazy=lazy, metadata=metadata
        )
        unfolded = _computed(phonon_loss_diffraction_patterns(waves, temperature=300.0))
        center = tuple(n // 2 for n in unfolded.shape[-2:])

        blocked = _computed(
            phonon_loss_diffraction_patterns(
                waves, temperature=300.0, block_direct=True
            )
        )

        np.testing.assert_array_equal(blocked[..., center[0], center[1]], 0.0)
        keep = np.ones(unfolded.shape[-2:], dtype=bool)
        keep[center] = False
        np.testing.assert_array_equal(blocked[..., keep], unfolded[..., keep])

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    @pytest.mark.parametrize(
        "reduction_dtype, dtype",
        [(None, np.float32), ("float32", np.float32), ("float64", np.float64)],
    )
    def test_unfolding_keeps_the_precision_of_the_patterns(
        self, reduction_dtype, dtype, lazy
    ):
        """complex64 exit waves give float32 patterns, and reduction_dtype sets the
        precision; the loss/gain weights must not raise it to float64."""
        e_values = [0.0, 0.02, 0.05, 0.10]
        waves = _make_exit_waves(e_values, lazy=lazy)
        diffuse = _computed(
            phonon_loss_diffraction_patterns(waves, reduction_dtype=reduction_dtype)
        )
        assert diffuse.dtype == dtype

        unfolded = phonon_loss_diffraction_patterns(
            waves, temperature=300.0, reduction_dtype=reduction_dtype
        )

        assert unfolded.array.dtype == dtype
        array = _computed(unfolded)
        assert array.dtype == dtype
        n_occ = 1.0 / np.expm1(np.array(e_values[1:]) / (units.kB * 300.0))
        loss = (n_occ + 1.0) / (2.0 * n_occ + 1.0)
        gain = n_occ / (2.0 * n_occ + 1.0)
        diffuse = diffuse.astype(np.float64)
        expected = np.concatenate(
            [
                (diffuse[1:] * gain[:, None, None])[::-1],
                diffuse[:1],
                diffuse[1:] * loss[:, None, None],
            ]
        )
        np.testing.assert_allclose(
            array, expected, rtol=0, atol=1e-6 * np.abs(expected).max()
        )

    @pytest.mark.parametrize("components", ["elastic", "all", ("diffuse",)])
    def test_requires_the_diffuse_component(self, components):
        waves = _make_exit_waves([0.0, 0.02, 0.05])
        with pytest.raises(ValueError, match="components='diffuse'"):
            phonon_loss_diffraction_patterns(
                waves, components=components, temperature=300.0
            )

    @pytest.mark.parametrize(
        "old, new",
        [("tds", "diffuse"), ("coherent", "elastic"), ("incoherent", "total")],
    )
    def test_earlier_component_names_point_to_the_new_ones(self, old, new):
        waves = _make_exit_waves([0.0, 0.02, 0.05])
        with pytest.raises(ValueError, match=f"'{old}' is now '{new}'"):
            phonon_loss_diffraction_patterns(waves, components=old, temperature=300.0)

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    @pytest.mark.parametrize("block_direct", [True, 15.0])
    def test_block_direct_gives_the_unfolded_patterns_blocked(self, block_direct, lazy):
        """The direct beam is blocked before the unfolding, a per-pixel mask that
        gives the same result as blocking the unfolded patterns; True blocks up to
        the semiangle cutoff, as DiffractionPatterns.block_direct() does."""
        waves = _make_exit_waves(
            [0.0, 0.02, 0.05, 0.10],
            gpts=48,
            lazy=lazy,
            metadata={"semiangle_cutoff": 20.0},
        )
        unfolded = phonon_loss_diffraction_patterns(waves, temperature=300.0)
        radius = None if block_direct is True else block_direct
        expected = _computed(unfolded.block_direct(radius=radius))

        blocked = phonon_loss_diffraction_patterns(
            waves, temperature=300.0, block_direct=block_direct
        )

        np.testing.assert_array_equal(_computed(blocked), expected)

    def test_zero_temperature_puts_the_whole_signal_on_the_loss_side(self):
        """The limit T -> 0 has no thermal phonons (n = 0): loss weight 1, gain
        weight 0."""
        e_values = [0.0, 0.02, 0.05, 0.10]
        waves = _make_exit_waves(e_values)
        diffuse = np.asarray(phonon_loss_diffraction_patterns(waves).array)

        unfolded = phonon_loss_diffraction_patterns(waves, temperature=0.0)

        energies = unfolded.ensemble_axes_metadata[0].values
        np.testing.assert_allclose(
            energies, [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
        )
        array = np.asarray(unfolded.array)
        np.testing.assert_array_equal(array[:3], 0.0)
        np.testing.assert_array_equal(array[3:], diffuse)

    @pytest.mark.parametrize(
        "temperature", [-300.0, float("nan"), float("inf"), -float("inf")]
    )
    def test_a_negative_or_non_finite_temperature_raises(self, temperature):
        waves = _make_exit_waves([0.0, 0.02, 0.05])
        with pytest.raises(ValueError, match="must be finite and non-negative"):
            phonon_loss_diffraction_patterns(waves, temperature=temperature)

    @pytest.mark.parametrize(
        "temperature", [True, np.True_], ids=["bool", "numpy_bool"]
    )
    def test_a_bool_temperature_raises(self, temperature):
        waves = _make_exit_waves([0.0, 0.02, 0.05])
        with pytest.raises(TypeError, match="temperature must be a number"):
            phonon_loss_diffraction_patterns(waves, temperature=temperature)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_the_temperature_is_used_in_double_precision(self, dtype):
        """k_B T in the precision of a float16 or float32 scalar puts the gain
        weight off by up to 7.6e-4 or 1.0e-9 (relative; 0.02 and 0.05 eV at 300 K);
        the float64 unfolding shows either."""
        waves = _make_exit_waves([0.0, 0.02, 0.05])
        expected = phonon_loss_diffraction_patterns(
            waves, temperature=300.0, reduction_dtype="float64"
        )
        result = phonon_loss_diffraction_patterns(
            waves, temperature=dtype(300.0), reduction_dtype="float64"
        )
        np.testing.assert_array_equal(
            np.asarray(result.array), np.asarray(expected.array)
        )

    @pytest.mark.parametrize("temperature", [-300.0, float("nan"), True])
    def test_the_weights_check_the_temperature(self, temperature):
        from abtem.measurements import _thermal_weight_tds

        with pytest.raises((TypeError, ValueError), match="temperature must be"):
            _thermal_weight_tds(
                np.ones((3, 4, 4)), np.array([0.0, 0.02, 0.05]), 0, temperature
            )

    def test_a_temperature_whose_k_b_t_underflows_is_the_zero_temperature_limit(
        self,
    ):
        waves = _make_exit_waves([0.0, 0.02, 0.05])
        zero = phonon_loss_diffraction_patterns(waves, temperature=0.0)
        tiny = phonon_loss_diffraction_patterns(waves, temperature=5e-324)
        np.testing.assert_array_equal(np.asarray(tiny.array), np.asarray(zero.array))

    def test_a_very_high_temperature_splits_the_signal_evenly(self):
        """n -> infinity: loss and gain weights both tend to 1/2."""
        waves = _make_exit_waves([0.0, 0.02, 0.05])
        diffuse = np.asarray(phonon_loss_diffraction_patterns(waves).array)

        array = np.asarray(
            phonon_loss_diffraction_patterns(waves, temperature=1e300).array
        )

        np.testing.assert_allclose(array[3:], diffuse[1:] / 2, rtol=1e-6)
        np.testing.assert_allclose(array[:2], diffuse[:0:-1] / 2, rtol=1e-6)

    def test_a_low_temperature_unfolds_without_overflow_warnings(self):
        """E / k_B T overflows the exponential (0.1 eV at 1 K), where n -> 0 is the
        correct limit: the whole signal is loss, and nothing warns."""
        import warnings

        e_values = [0.0, 0.1, 0.2]
        waves = _make_exit_waves(e_values)
        diffuse = np.asarray(phonon_loss_diffraction_patterns(waves).array)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            unfolded = phonon_loss_diffraction_patterns(waves, temperature=1.0)

        array = np.asarray(unfolded.array)
        np.testing.assert_array_equal(array[:2], 0.0)
        np.testing.assert_array_equal(array[2:], diffuse)

    def test_requires_energies_start_at_zero_and_ascending(self):
        waves_no_zero = _make_exit_waves([0.01, 0.02, 0.05])
        with pytest.raises(ValueError, match="starting at 0"):
            phonon_loss_diffraction_patterns(
                waves_no_zero, components="diffuse", temperature=300.0
            )

        waves_unsorted = _make_exit_waves([0.0, 0.05, 0.02])
        with pytest.raises(ValueError, match="starting at 0"):
            phonon_loss_diffraction_patterns(
                waves_unsorted, components="diffuse", temperature=300.0
            )


class TestMomentumResolvedSpectrum:
    """momentum_resolved_spectrum builds S(q, E) from the diffuse component only."""

    @staticmethod
    def _spectrum(patterns):
        from abtem.detectors import SpectralSlitDetector
        from abtem.measurements import momentum_resolved_spectrum

        detector = SpectralSlitDetector(q_max=60.0, width=20.0)
        return np.asarray(momentum_resolved_spectrum(patterns, detector).array)

    def test_a_component_axis_gives_the_diffuse_spectrum(self):
        waves = _make_exit_waves([0.02, 0.05, 0.10])
        diffuse = phonon_loss_diffraction_patterns(waves, components="diffuse")
        stacked = phonon_loss_diffraction_patterns(waves, components="all")
        np.testing.assert_array_equal(self._spectrum(stacked), self._spectrum(diffuse))

    @pytest.mark.parametrize("components", ["total", "elastic"])
    def test_a_single_other_component_raises(self, components):
        waves = _make_exit_waves([0.02, 0.05, 0.10])
        patterns = phonon_loss_diffraction_patterns(waves, components=components)
        with pytest.raises(ValueError, match=f"is the '{components}' component"):
            self._spectrum(patterns)

    def test_a_reduced_stack_of_the_diffuse_component_alone_is_used(self):
        waves = _make_exit_waves([0.02, 0.05, 0.10])
        stacked = phonon_loss_diffraction_patterns(waves, components=("diffuse",))
        diffuse = phonon_loss_diffraction_patterns(waves, components="diffuse")
        np.testing.assert_allclose(
            self._spectrum(stacked.sum(axis=0)), self._spectrum(diffuse), rtol=1e-6
        )

    def test_another_recorded_component_is_used_as_given(self):
        """Only the components with the elastic intensity are rejected; another
        name, such as a parity-projected one-phonon channel, is used as given."""
        waves = _make_exit_waves([0.02, 0.05, 0.10])
        diffuse = phonon_loss_diffraction_patterns(waves, components="diffuse")
        kwargs = diffuse._copy_kwargs(exclude=("metadata",))
        kwargs["metadata"] = {**diffuse.metadata, "frozen_phonon_component": "one"}
        one = diffuse.__class__(**kwargs)
        np.testing.assert_array_equal(self._spectrum(one), self._spectrum(diffuse))

    def test_patterns_without_the_component_metadata_are_used_as_given(self):
        waves = _make_exit_waves([0.02, 0.05, 0.10])
        diffuse = phonon_loss_diffraction_patterns(waves, components="diffuse")
        kwargs = diffuse._copy_kwargs(exclude=("metadata",))
        kwargs["metadata"] = dict(diffuse.metadata)
        del kwargs["metadata"]["frozen_phonon_component"]
        plain = diffuse.__class__(**kwargs)
        np.testing.assert_array_equal(self._spectrum(plain), self._spectrum(diffuse))


class TestLazyExitWaves:
    """exit_waves may be a lazy (dask-backed) Waves object -- e.g. built with
    multislice(..., lazy=True) and fed straight into to_zarr(). CuPy's own
    concatenate/flip/stack do not accept a dask array (unlike NumPy, which
    dispatches to dask via __array_function__), so on GPU these must route
    through dask's own implementations rather than get_array_module's xp
    directly. Exercised here via a dask-wrapped numpy array, which hits the
    same "still a da.core.Array" code path independent of the device."""

    def test_thermal_weighting_matches_eager(self):
        e_values = [0.0, 0.02, 0.05, 0.10]
        waves_eager = _make_exit_waves(e_values, lazy=False)
        waves_lazy = _make_exit_waves(e_values, lazy=True)

        dp_eager = phonon_loss_diffraction_patterns(
            waves_eager, components="diffuse", temperature=300.0
        )
        dp_lazy = phonon_loss_diffraction_patterns(
            waves_lazy, components="diffuse", temperature=300.0
        )

        assert isinstance(dp_lazy.array, da.core.Array), (
            "result should stay lazy when exit_waves was lazy"
        )
        np.testing.assert_allclose(dp_lazy.array.compute(), dp_eager.array, rtol=1e-4)

    def test_component_all_matches_eager(self):
        e_values = [0.0, 0.02, 0.05]
        waves_eager = _make_exit_waves(e_values, lazy=False)
        waves_lazy = _make_exit_waves(e_values, lazy=True)

        dp_eager = phonon_loss_diffraction_patterns(waves_eager, components="all")
        dp_lazy = phonon_loss_diffraction_patterns(waves_lazy, components="all")

        assert isinstance(dp_lazy.array, da.core.Array)
        np.testing.assert_allclose(dp_lazy.array.compute(), dp_eager.array, rtol=1e-4)


class TestClassicalStatistics:
    """Unfolding of snapshots sampled with classical (equipartition)
    amplitudes, e.g. from molecular dynamics."""

    def test_classical_weights_apply_the_quantum_correction(self):
        """classical weights = quantum weights * x coth x, x = E / 2kT: the
        standard correction of a classical spectrum, and detailed balance
        holds in both modes."""
        from ase import units

        from abtem.measurements import _loss_gain_weights

        e = np.array([0.01, 0.05, 0.2])
        T = 300.0
        x = e / (2 * units.kB * T)
        loss_q, gain_q = _loss_gain_weights(e, T, "quantum")
        loss_c, gain_c = _loss_gain_weights(e, T, "classical")

        np.testing.assert_allclose(loss_q + gain_q, 1.0)
        np.testing.assert_allclose(loss_c + gain_c, x / np.tanh(x))
        np.testing.assert_allclose(loss_c, loss_q * x / np.tanh(x))
        np.testing.assert_allclose(gain_c, gain_q * x / np.tanh(x))
        for loss, gain in ((loss_q, gain_q), (loss_c, gain_c)):
            np.testing.assert_allclose(loss / gain, np.exp(e / (units.kB * T)))

        # high temperature: the classical split is even and unscaled
        loss_hot, gain_hot = _loss_gain_weights(np.array([0.001]), 5000.0, "classical")
        np.testing.assert_allclose([loss_hot[0], gain_hot[0]], [0.5, 0.5], atol=1e-3)  # deviation is x/2 ~ 6e-4
        # low temperature: zero-point motion restored, gain switched off
        loss_cold, gain_cold = _loss_gain_weights(np.array([0.1]), 10.0, "classical")
        assert gain_cold[0] < 1e-40
        np.testing.assert_allclose(loss_cold[0], 0.1 / (2 * units.kB * 10.0))

    @pytest.mark.parametrize("temperature", [0.0, 5e-324])
    def test_classical_weights_at_zero_temperature_raise(self, temperature):
        """The quantum weights have the n -> 0 limit at T = 0, but the
        classical ones, x (n + 1) and x n with x = E / (2 k_B T), diverge:
        equipartition amplitudes vanish there. They raise rather than return
        inf and NaN, also where k_B T underflows."""
        from abtem.measurements import _loss_gain_weights

        e = np.array([0.02, 0.05])
        loss, gain = _loss_gain_weights(e, temperature, "quantum")
        np.testing.assert_array_equal(loss, [1.0, 1.0])
        np.testing.assert_array_equal(gain, [0.0, 0.0])
        with pytest.raises(ValueError, match="needs a temperature above 0 K"):
            _loss_gain_weights(e, temperature, "classical")

    def test_classical_mode_in_four_run_rest_parity(self):
        """With rest fields and no static reference the one-phonon channel is
        returned alone and may be unfolded; the classical weights must reach
        that path too."""
        from ase import units

        from abtem.core.axes import PhononRestParityAxis

        e_values = [0.0, 0.02, 0.05]
        n_configs, gpts = 4, 16
        rng = np.random.default_rng(3)
        shape = (2, 2, len(e_values), n_configs, gpts, gpts)
        members = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex64
        )
        waves = Waves(
            members, energy=100e3, sampling=0.1,
            ensemble_axes_metadata=[
                PhononParityAxis(values=("real", "twin")),
                PhononRestParityAxis(values=("plus", "minus")),
                EnergyLossAxis(values=tuple(e_values)),
                FrozenPhononsAxis(_ensemble_mean=False),
            ],
        )
        T = 300.0
        quantum = phonon_loss_diffraction_patterns(waves, temperature=T, max_angle="full")
        classical = phonon_loss_diffraction_patterns(
            waves, temperature=T, snapshot_statistics="classical", max_angle="full"
        )
        x = np.array(e_values[1:]) / (2 * units.kB * T)
        for k, f in enumerate(x / np.tanh(x)):
            np.testing.assert_allclose(classical.array[3 + k], quantum.array[3 + k] * f, rtol=1e-5)
            np.testing.assert_allclose(classical.array[1 - k], quantum.array[1 - k] * f, rtol=1e-5)

    def test_invalid_statistics_rejected(self):
        from abtem.measurements import _loss_gain_weights, unfold_loss_gain

        with pytest.raises(ValueError, match="snapshot_statistics"):
            _loss_gain_weights(np.array([0.05]), 300.0, "bogus")
        waves = _make_exit_waves([0.0, 0.02, 0.05], n_configs=4)
        with pytest.raises(ValueError, match="snapshot_statistics"):
            phonon_loss_diffraction_patterns(
                waves, temperature=300.0, snapshot_statistics="bogus"
            )
        dp = phonon_loss_diffraction_patterns(waves)
        with pytest.raises(ValueError, match="snapshot_statistics"):
            unfold_loss_gain(dp, 300.0, snapshot_statistics="md")

        # also without temperature: the argument is only consumed by the
        # unfolding, so an unvalidated typo would be silently ignored
        with pytest.raises(ValueError, match="snapshot_statistics"):
            phonon_loss_diffraction_patterns(waves, snapshot_statistics="clasical")

    def test_classical_mode_threads_through_both_entry_points(self):
        from ase import units

        from abtem.measurements import unfold_loss_gain

        e_values = [0.0, 0.02, 0.05]
        waves = _make_exit_waves(e_values, n_configs=4)
        T = 300.0
        via_call = phonon_loss_diffraction_patterns(
            waves, temperature=T, snapshot_statistics="classical", max_angle="full"
        )
        via_helper = unfold_loss_gain(
            phonon_loss_diffraction_patterns(waves, max_angle="full"),
            T, snapshot_statistics="classical",
        )
        quantum = phonon_loss_diffraction_patterns(waves, temperature=T, max_angle="full")
        np.testing.assert_allclose(via_call.array, via_helper.array, rtol=1e-6)

        # classical / quantum = x coth x on every non-zero energy, both sides
        x = np.array(e_values[1:]) / (2 * units.kB * T)
        factor = x / np.tanh(x)
        for k, f in enumerate(factor):
            loss_ratio = via_call.array[3 + k] / quantum.array[3 + k]
            gain_ratio = via_call.array[1 - k] / quantum.array[1 - k]
            np.testing.assert_allclose(loss_ratio, f, rtol=1e-5)
            np.testing.assert_allclose(gain_ratio, f, rtol=1e-5)
        np.testing.assert_allclose(via_call.array[2], quantum.array[2])  # zero bin


class TestParityProjection:
    """Tests for the "Phonon order"=("all", "one", "multi") ensemble output
    when exit_waves carries a PhononParityAxis (issue #373).
    """

    def test_block_direct_true_infers_radius_from_metadata(self):
        waves = _make_parity_exit_waves([0.02, 0.05], n_configs=6)
        waves.metadata["semiangle_cutoff"] = 15.0

        dp_auto = phonon_loss_diffraction_patterns(waves, block_direct=True)
        dp_explicit = phonon_loss_diffraction_patterns(waves, block_direct=15.0)
        dp_wrong = phonon_loss_diffraction_patterns(waves, block_direct=1.0)

        np.testing.assert_array_equal(dp_auto.array, dp_explicit.array)
        assert not np.array_equal(dp_auto.array, dp_wrong.array)

    def test_shape_and_axes(self):
        e_values = [0.02, 0.05, 0.10]
        waves = _make_parity_exit_waves(e_values, n_configs=6)

        dp = phonon_loss_diffraction_patterns(waves)

        from abtem.core.axes import OrdinalAxis

        phonon_order_axis = next(
            ax for ax in dp.ensemble_axes_metadata
            if isinstance(ax, OrdinalAxis) and ax.label == "Phonon order"
        )
        assert phonon_order_axis.values == ("all", "one", "multi")
        assert dp.array.shape[0] == 3

        energy_axis = next(
            ax for ax in dp.ensemble_axes_metadata if isinstance(ax, EnergyLossAxis)
        )
        assert len(energy_axis.values) == 3
        # FrozenPhononsAxis and PhononParityAxis must both be gone
        assert not any(
            isinstance(ax, FrozenPhononsAxis) for ax in dp.ensemble_axes_metadata
        )
        assert not any(
            isinstance(ax, PhononParityAxis) for ax in dp.ensemble_axes_metadata
        )

    def test_all_equals_ordinary_tds_over_full_parity_set(self):
        """"all" is one + multi, and for the symmetric (real, twin) set that
        is exactly the ordinary total-minus-elastic (diffuse) estimator over all
        2N members -- not a statistical agreement, an identity."""
        e_values = [0.02, 0.05, 0.10]
        waves = _make_parity_exit_waves(e_values, n_configs=6)

        dp = phonon_loss_diffraction_patterns(waves, max_angle="full")

        full_set = Waves(
            np.concatenate([waves.array[0], waves.array[1]], axis=1),
            energy=100e3, sampling=0.1,
            ensemble_axes_metadata=waves.ensemble_axes_metadata[1:],
        )
        dp_full = phonon_loss_diffraction_patterns(
            full_set, components="diffuse", max_angle="full"
        )

        scale = np.abs(dp.array[1]).max()
        np.testing.assert_allclose(dp.array[0], dp_full.array, atol=1e-5 * scale, rtol=0)
        np.testing.assert_allclose(
            dp.array[0], dp.array[1] + dp.array[2], atol=1e-6 * scale, rtol=0
        )

    def test_one_phonon_and_multi_phonon_isolate_known_signals(self):
        """Construct real/twin so that psi_odd and psi_even are exactly
        known, independently-verifiable signals."""
        e_values = [0.02, 0.05]
        n_configs, gpts = 4, 16
        shape = (len(e_values), n_configs, gpts, gpts)
        rng = np.random.default_rng(1)

        static = (
            rng.normal(size=(gpts, gpts)) + 1j * rng.normal(size=(gpts, gpts))
        ).astype(np.complex64)
        delta = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex64
        )
        eps = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex64
        )

        # real = static + delta + eps, twin = static - delta + eps
        #   => psi_odd  = (real - twin) / 2 = delta          (one-phonon)
        #   => psi_even = (real + twin) / 2 = static + eps   (multi-phonon =
        #      its variance over configs; the constant static drops out)
        real = static + delta + eps
        twin = static - delta + eps

        waves = _make_parity_exit_waves(
            e_values, n_configs=n_configs, gpts=gpts, real=real, twin=twin,
        )
        dp = phonon_loss_diffraction_patterns(waves, max_angle="full")

        # independent reference: FFT delta/eps directly; axis=1 of shape
        # (n_e, n_c, gpts, gpts) is the frozen-phonon axis
        delta_waves = Waves(
            delta, energy=100e3, sampling=0.1,
            ensemble_axes_metadata=waves.ensemble_axes_metadata[1:],
        )
        eps_waves = Waves(
            eps, energy=100e3, sampling=0.1,
            ensemble_axes_metadata=waves.ensemble_axes_metadata[1:],
        )
        I_one_ref = delta_waves.diffraction_patterns(max_angle="full").array.mean(axis=1)
        I_multi_ref = (
            eps_waves.diffraction_patterns(max_angle="full").array.mean(axis=1)
            - eps_waves.sum(axis=1).diffraction_patterns(max_angle="full").array
            / n_configs**2
        )

        np.testing.assert_allclose(dp.array[1], I_one_ref, rtol=1e-4)
        np.testing.assert_allclose(dp.array[2], I_multi_ref, rtol=1e-4, atol=1e-4 * I_multi_ref.max())

    def test_requires_real_twin_parity_axis_values(self):
        e_values = [0.02, 0.05]
        waves = _make_exit_waves(e_values, n_configs=4)
        array = np.stack([waves.array, waves.array, waves.array], axis=0)
        bad_waves = Waves(
            array, energy=100e3, sampling=0.1,
            ensemble_axes_metadata=[
                PhononParityAxis(values=("real", "twin", "static"))
            ]
            + waves.ensemble_axes_metadata,
        )
        with pytest.raises(ValueError, match="'real', 'twin'"):
            phonon_loss_diffraction_patterns(bad_waves)

    def test_temperature_is_rejected(self):
        """One-phonon Bose weights at +/-E are wrong for the multi channel
        (its two-phonon processes sit at +2E, 0, -2E), so the combined call
        must refuse rather than mislabel."""
        e_values = [0.0, 0.02, 0.05]
        waves = _make_parity_exit_waves(e_values, n_configs=6)
        with pytest.raises(ValueError, match="unfold_loss_gain"):
            phonon_loss_diffraction_patterns(waves, temperature=300.0)

    def test_unfold_loss_gain_selects_the_one_phonon_channel_of_a_stack(self):
        """One-phonon Bose weights apply to the 'one' channel alone, so
        unfolding a whole "Phonon order" stack has only one possible meaning
        and picks that channel itself rather than making the caller index the
        axis by hand. The stack survives into the spectrum, so both forms
        must do it, and both must equal passing the 'one' slot explicitly."""
        from abtem.detectors import SpectralSlitDetector
        from abtem.measurements import momentum_resolved_spectrum, unfold_loss_gain

        e_values = [0.0, 0.02, 0.05]
        waves = _make_parity_exit_waves(e_values, n_configs=6)
        dp = phonon_loss_diffraction_patterns(waves, max_angle="full")
        order = list(dp.ensemble_axes_metadata[0].values)
        one = dp[order.index("one")]

        auto = unfold_loss_gain(dp, 300.0)
        manual = unfold_loss_gain(one, 300.0)
        np.testing.assert_array_equal(auto.array, manual.array)
        assert auto.metadata["frozen_phonon_component"] == "one"
        # the stacking axis is consumed, not carried through
        assert not any(
            getattr(ax, "label", None) == "Phonon order"
            for ax in auto.ensemble_axes_metadata
        )
        assert auto.array.shape[0] == 5  # signed energies, no channel axis

        # and the same through a MomentumResolvedSpectrum
        detector = SpectralSlitDetector(width=4.0, q_min=0.0, q_max=20.0)
        auto_spec = unfold_loss_gain(momentum_resolved_spectrum(dp, detector), 300.0)
        manual_spec = unfold_loss_gain(
            momentum_resolved_spectrum(one, detector), 300.0
        )
        np.testing.assert_array_equal(auto_spec.array, manual_spec.array)
        assert auto_spec.array.shape[-1] == 5

    def test_unfold_loss_gain_on_one_slot_matches_internal_unfolding(self):
        from abtem.measurements import unfold_loss_gain

        e_values = [0.0, 0.02, 0.05]
        waves = _make_parity_exit_waves(e_values, n_configs=6)
        dp = phonon_loss_diffraction_patterns(waves, max_angle="full")
        one = dp[1]
        assert not any(
            isinstance(ax, OrdinalAxis) and ax.label == "Phonon order"
            for ax in one.ensemble_axes_metadata
        )

        unfolded = unfold_loss_gain(one, 300.0)
        assert isinstance(unfolded, type(one))
        energy_axis = next(
            ax for ax in unfolded.ensemble_axes_metadata
            if isinstance(ax, EnergyLossAxis)
        )
        assert energy_axis.values == (-0.05, -0.02, 0.0, 0.02, 0.05)
        assert unfolded.array.shape[0] == 5

        # identical to what the non-parity path does with temperature=,
        # applied to the same (unweighted) input
        waves_odd = Waves(
            (waves.array[0] - waves.array[1]) / 2,
            energy=100e3, sampling=0.1,
            ensemble_axes_metadata=waves.ensemble_axes_metadata[1:],
        )
        ref = phonon_loss_diffraction_patterns(
            waves_odd, components="total", max_angle="full"
        )
        from abtem.measurements import _thermal_weight_tds
        ref_array, _ = _thermal_weight_tds(
            ref.array, np.asarray(e_values), 0, 300.0
        )
        np.testing.assert_allclose(unfolded.array, ref_array, rtol=1e-5)

        # detailed balance: loss/gain = exp(E / kT) at every nonzero energy
        from ase import units
        loss, gain = unfolded.array[3:], unfolded.array[:2][::-1]
        ratio = loss.sum(axis=(-2, -1)) / gain.sum(axis=(-2, -1))
        np.testing.assert_allclose(
            ratio, np.exp(np.array([0.02, 0.05]) / (units.kB * 300.0)), rtol=1e-5
        )

    def test_unfold_loss_gain_commutes_with_momentum_resolved_spectrum(self):
        """unfold_loss_gain must accept a MomentumResolvedSpectrum (energy is
        its last base axis) and give the same result as unfolding the
        diffraction patterns before building the spectrum."""
        from abtem.detectors import SpectralSlitDetector
        from abtem.measurements import momentum_resolved_spectrum, unfold_loss_gain

        e_values = [0.0, 0.02, 0.05]
        waves = _make_parity_exit_waves(e_values, n_configs=6)
        dp_one = phonon_loss_diffraction_patterns(waves, max_angle="full")[1]
        detector = SpectralSlitDetector(width=20.0, q_min=0.0, q_max=60.0)

        spectrum_then_unfold = unfold_loss_gain(
            momentum_resolved_spectrum(dp_one, detector), 300.0
        )
        unfold_then_spectrum = momentum_resolved_spectrum(
            unfold_loss_gain(dp_one, 300.0), detector
        )
        assert spectrum_then_unfold.e_values == (-0.05, -0.02, 0.0, 0.02, 0.05)
        assert spectrum_then_unfold.e_values == unfold_then_spectrum.e_values
        np.testing.assert_allclose(
            spectrum_then_unfold.array, unfold_then_spectrum.array, rtol=1e-6
        )

    def test_unfold_loss_gain_requires_energy_axis(self):
        from abtem.measurements import unfold_loss_gain

        waves = _make_exit_waves([0.02, 0.05], n_configs=4)
        dp = waves.diffraction_patterns(max_angle="full")
        no_energy = dp.sum(axis=0)  # drops the EnergyLossAxis
        with pytest.raises(ValueError, match="EnergyLossAxis"):
            unfold_loss_gain(no_energy, 300.0)

    def test_rest_parity_axis_is_averaged_out_before_projection(self):
        """With a rest-parity axis the waves are first averaged over the two
        rest signs and the static member is subtracted from the even part.
        Build psi(s, t) = static + s*delta + t*rho + chi + eps (eps absent
        from the static member) so that the rest-odd part rho and the
        rest-even, bin-independent part chi must both drop out exactly,
        leaving one = mean|delta|^2 and multi = variance(eps)."""
        from abtem.core.axes import PhononRestParityAxis

        e_values = [0.02, 0.05]
        n_configs, gpts = 4, 16
        shape = (len(e_values), n_configs, gpts, gpts)
        rng = np.random.default_rng(7)

        def rc(size):
            return (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
                np.complex64
            )

        static, delta, eps = rc((gpts, gpts)), rc(shape), rc(shape)
        rho = 5.0 * rc(shape)  # deliberately large rest-odd part

        # members: real/twin/static x rest sign; the static member carries the
        # rest-odd part rho and a rest-realization-dependent even part chi
        # that must cancel against the same chi in real and twin
        chi = 3.0 * rc(shape)
        members = np.empty((3, 2) + shape, dtype=np.complex64)
        for parity, s_sign in enumerate((1, -1, 0)):
            for rest, t_sign in enumerate((1, -1)):
                members[parity, rest] = (
                    static + s_sign * delta + t_sign * rho + chi + (eps if s_sign else 0)
                )

        waves = Waves(
            members, energy=100e3, sampling=0.1,
            ensemble_axes_metadata=[
                PhononParityAxis(values=("real", "twin", "static")),
                PhononRestParityAxis(values=("plus", "minus")),
                EnergyLossAxis(values=tuple(e_values)),
                FrozenPhononsAxis(_ensemble_mean=False),
            ],
        )
        dp = phonon_loss_diffraction_patterns(waves, max_angle="full")
        assert dp.array.shape[0] == 3

        reference = _make_parity_exit_waves(
            e_values, n_configs=n_configs, gpts=gpts,
            real=static + delta + eps, twin=static - delta + eps,
        )
        dp_ref = phonon_loss_diffraction_patterns(reference, max_angle="full")
        scale = np.abs(dp_ref.array[1]).max()
        np.testing.assert_allclose(dp.array, dp_ref.array, atol=1e-5 * scale, rtol=0)

        lazy_waves = Waves(
            da.from_array(members, chunks=(1, 1, 1, 1, gpts, gpts)),
            energy=100e3, sampling=0.1,
            ensemble_axes_metadata=waves.ensemble_axes_metadata,
        )
        dp_lazy = phonon_loss_diffraction_patterns(lazy_waves, max_angle="full")
        assert isinstance(dp_lazy.array, da.core.Array)
        np.testing.assert_allclose(
            dp_lazy.array.compute(), dp.array, atol=1e-5 * scale, rtol=0
        )

        # without the static member (the four-run default) only the
        # one-phonon channel is returned, as a plain DiffractionPatterns
        waves_four = Waves(
            members[:2], energy=100e3, sampling=0.1,
            ensemble_axes_metadata=[
                PhononParityAxis(values=("real", "twin")),
                PhononRestParityAxis(values=("plus", "minus")),
                EnergyLossAxis(values=tuple(e_values)),
                FrozenPhononsAxis(_ensemble_mean=False),
            ],
        )
        dp_four = phonon_loss_diffraction_patterns(waves_four, max_angle="full")
        assert dp_four.metadata["frozen_phonon_component"] == "one"
        assert not any(
            getattr(ax, "label", None) == "Phonon order"
            for ax in dp_four.ensemble_axes_metadata
        )
        np.testing.assert_allclose(dp_four.array, dp.array[1], atol=1e-5 * scale, rtol=0)

        # ... and temperature unfolding is then allowed (one-phonon weights)
        waves_four_t = Waves(
            members[:2], energy=100e3, sampling=0.1,
            ensemble_axes_metadata=[
                PhononParityAxis(values=("real", "twin")),
                PhononRestParityAxis(values=("plus", "minus")),
                EnergyLossAxis(values=(0.0, 0.05)),
                FrozenPhononsAxis(_ensemble_mean=False),
            ],
        )
        unfolded = phonon_loss_diffraction_patterns(
            waves_four_t, max_angle="full", temperature=300.0
        )
        assert unfolded.array.shape[0] == 3
        with pytest.raises(ValueError, match="unfold_loss_gain"):
            phonon_loss_diffraction_patterns(waves, max_angle="full", temperature=300.0)

    def test_lazy_matches_eager(self):
        e_values = [0.02, 0.05, 0.10]
        waves_eager = _make_parity_exit_waves(e_values, n_configs=6, lazy=False)
        waves_lazy = _make_parity_exit_waves(e_values, n_configs=6, lazy=True)

        dp_eager = phonon_loss_diffraction_patterns(waves_eager)
        dp_lazy = phonon_loss_diffraction_patterns(waves_lazy)

        assert isinstance(dp_lazy.array, da.core.Array)
        np.testing.assert_allclose(dp_lazy.array.compute(), dp_eager.array, rtol=1e-4)
