"""Tests for phonon_loss_diffraction_patterns."""

import dask.array as da
import numpy as np
import pytest
from ase import units

from abtem.core.axes import OrdinalAxis, EnergyLossAxis, FrozenPhononsAxis, PhononParityAxis
from abtem.measurements import phonon_loss_diffraction_patterns
from abtem.waves import Waves


def _make_exit_waves(e_values, n_configs=6, gpts=24, seed=0, lazy=False):
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

    dp_tds = phonon_loss_diffraction_patterns(waves, component="tds")
    dp_coh = phonon_loss_diffraction_patterns(waves, component="coherent")
    dp_inc = phonon_loss_diffraction_patterns(waves, component="incoherent")
    dp_all = phonon_loss_diffraction_patterns(waves, component="all")

    assert np.allclose(dp_all.array[0], dp_coh.array)
    assert np.allclose(dp_all.array[1], dp_inc.array)
    assert np.allclose(dp_all.array[2], dp_tds.array)
    assert np.allclose(dp_tds.array, dp_inc.array - dp_coh.array)

    for dp, name in [(dp_tds, "tds"), (dp_coh, "coherent"), (dp_inc, "incoherent")]:
        assert dp.metadata["phonon_loss_component"] == name
        assert dp.metadata["energy"] == 100e3


def test_invalid_component_raises():
    waves = _make_exit_waves([0.02, 0.05, 0.10])
    with pytest.raises(ValueError, match="component must be one of"):
        phonon_loss_diffraction_patterns(waves, component="bogus")


@pytest.mark.parametrize("component", ["tds", "all"])
def test_single_config_tds_raises_instead_of_returning_zeros(component):
    """With one frozen-phonon configuration, I_incoherent == I_coherent by
    construction, so I_tds is identically zero everywhere -- not a bug, but
    silently returning an all-zero array looks exactly like one. This must
    raise instead."""
    waves = _make_exit_waves([0.02, 0.05, 0.10], n_configs=1)
    with pytest.raises(ValueError, match="at least 2 frozen-phonon"):
        phonon_loss_diffraction_patterns(waves, component=component)


@pytest.mark.parametrize("component", ["coherent", "incoherent"])
def test_single_config_non_tds_components_still_work(component):
    """coherent/incoherent are well-defined (if trivial) for a single
    configuration and must not be blocked by the N>=2 check."""
    waves = _make_exit_waves([0.02, 0.05, 0.10], n_configs=1)
    dp = phonon_loss_diffraction_patterns(waves, component=component)
    assert not np.allclose(dp.array, 0.0)


def test_two_configs_tds_does_not_raise():
    waves = _make_exit_waves([0.02, 0.05, 0.10], n_configs=2)
    dp = phonon_loss_diffraction_patterns(waves, component="tds")
    assert dp.array.shape[0] == 3


class TestThermalWeighting:
    def test_signed_axis_and_zero_bin_passthrough(self):
        e_values = [0.0, 0.02, 0.05, 0.10]
        waves = _make_exit_waves(e_values)

        dp_unweighted = phonon_loss_diffraction_patterns(waves, component="tds")
        dp_weighted = phonon_loss_diffraction_patterns(
            waves, component="tds", temperature=300.0
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

        dp_unweighted = phonon_loss_diffraction_patterns(waves, component="tds")
        dp_weighted = phonon_loss_diffraction_patterns(
            waves, component="tds", temperature=T
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

    def test_requires_tds_component(self):
        waves = _make_exit_waves([0.0, 0.02, 0.05])
        with pytest.raises(ValueError, match="component='tds'"):
            phonon_loss_diffraction_patterns(
                waves, component="coherent", temperature=300.0
            )

    def test_requires_energies_start_at_zero_and_ascending(self):
        waves_no_zero = _make_exit_waves([0.01, 0.02, 0.05])
        with pytest.raises(ValueError, match="starting at 0"):
            phonon_loss_diffraction_patterns(
                waves_no_zero, component="tds", temperature=300.0
            )

        waves_unsorted = _make_exit_waves([0.0, 0.05, 0.02])
        with pytest.raises(ValueError, match="starting at 0"):
            phonon_loss_diffraction_patterns(
                waves_unsorted, component="tds", temperature=300.0
            )


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
            waves_eager, component="tds", temperature=300.0
        )
        dp_lazy = phonon_loss_diffraction_patterns(
            waves_lazy, component="tds", temperature=300.0
        )

        assert isinstance(dp_lazy.array, da.core.Array), (
            "result should stay lazy when exit_waves was lazy"
        )
        np.testing.assert_allclose(dp_lazy.array.compute(), dp_eager.array, rtol=1e-4)

    def test_component_all_matches_eager(self):
        e_values = [0.0, 0.02, 0.05]
        waves_eager = _make_exit_waves(e_values, lazy=False)
        waves_lazy = _make_exit_waves(e_values, lazy=True)

        dp_eager = phonon_loss_diffraction_patterns(waves_eager, component="all")
        dp_lazy = phonon_loss_diffraction_patterns(waves_lazy, component="all")

        assert isinstance(dp_lazy.array, da.core.Array)
        np.testing.assert_allclose(dp_lazy.array.compute(), dp_eager.array, rtol=1e-4)


class TestParityProjection:
    """Tests for the "Phonon order"=("all", "one", "multi") ensemble output
    when exit_waves carries a PhononParityAxis (issue #373).
    """

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
        is exactly the ordinary I_incoherent - I_coherent estimator over all
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
            full_set, component="tds", max_angle="full"
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
            waves_odd, component="incoherent", max_angle="full"
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

    def test_unfold_loss_gain_requires_energy_axis(self):
        from abtem.measurements import unfold_loss_gain

        waves = _make_exit_waves([0.02, 0.05], n_configs=4)
        dp = waves.diffraction_patterns(max_angle="full")
        no_energy = dp.sum(axis=0)  # drops the EnergyLossAxis
        with pytest.raises(ValueError, match="EnergyLossAxis"):
            unfold_loss_gain(no_energy, 300.0)

    def test_lazy_matches_eager(self):
        e_values = [0.02, 0.05, 0.10]
        waves_eager = _make_parity_exit_waves(e_values, n_configs=6, lazy=False)
        waves_lazy = _make_parity_exit_waves(e_values, n_configs=6, lazy=True)

        dp_eager = phonon_loss_diffraction_patterns(waves_eager)
        dp_lazy = phonon_loss_diffraction_patterns(waves_lazy)

        assert isinstance(dp_lazy.array, da.core.Array)
        np.testing.assert_allclose(dp_lazy.array.compute(), dp_eager.array, rtol=1e-4)
