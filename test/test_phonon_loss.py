"""Tests for phonon_loss_diffraction_patterns."""

import dask.array as da
import numpy as np
import pytest
from ase import units

from abtem.core.axes import EnergyLossAxis, FrozenPhononsAxis
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

    @pytest.mark.parametrize(
        "temperature",
        [
            "300",
            b"300",
            bytearray(b"300"),
            300j,
            np.complex64(300),
            np.complex128(300 + 1j),
            np.clongdouble(300),
            np.array(True),
            np.array("300"),
            np.array(b"300"),
        ],
        ids=[
            "str",
            "bytes",
            "bytearray",
            "complex",
            "complex64",
            "complex128",
            "clongdouble",
            "bool_array",
            "str_array",
            "bytes_array",
        ],
    )
    def test_a_temperature_that_is_not_a_number_raises(self, temperature):
        waves = _make_exit_waves([0.0, 0.02, 0.05])
        with pytest.raises(TypeError, match="temperature must be a number"):
            phonon_loss_diffraction_patterns(waves, temperature=temperature)

    @pytest.mark.parametrize(
        "kind", ["decimal", "fraction", "object_array", "float_only"]
    )
    def test_any_real_number_is_a_temperature(self, kind):
        """Real numbers other than float, int and the NumPy scalars give the same
        result as the float."""
        from decimal import Decimal
        from fractions import Fraction

        class FloatOnly:
            def __float__(self):
                return 300.0

        temperature = {
            "decimal": Decimal(300),
            "fraction": Fraction(300),
            "object_array": np.array(300.0, dtype=object),
            "float_only": FloatOnly(),
        }[kind]
        waves = _make_exit_waves([0.0, 0.02, 0.05])
        expected = phonon_loss_diffraction_patterns(waves, temperature=300.0)

        result = phonon_loss_diffraction_patterns(waves, temperature=temperature)

        np.testing.assert_array_equal(
            np.asarray(result.array), np.asarray(expected.array)
        )

    def test_a_masked_temperature_raises(self):
        waves = _make_exit_waves([0.0, 0.02, 0.05])
        with pytest.raises(ValueError, match="masked"):
            phonon_loss_diffraction_patterns(waves, temperature=np.ma.masked)

    def test_an_array_of_temperatures_raises(self):
        waves = _make_exit_waves([0.0, 0.02, 0.05])
        with pytest.raises(ValueError, match="must be a single number"):
            phonon_loss_diffraction_patterns(
                waves, temperature=np.array([300.0, 310.0])
            )

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
