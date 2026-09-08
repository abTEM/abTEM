"""Tests for phonon_loss_diffraction_patterns."""

import dask.array as da
import numpy as np
import pytest
from ase import units

from abtem.core.axes import EnergyLossAxis, FrozenPhononsAxis, PhononParityAxis
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
    static=None,
):
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
    if static is None:
        static = (
            rng.normal(size=(gpts, gpts)) + 1j * rng.normal(size=(gpts, gpts))
        ).astype(np.complex64)
        static = np.broadcast_to(static, shape)

    array = np.stack([real, twin, static], axis=0)
    if lazy:
        array = da.from_array(array, chunks=(1, 1, 1, gpts, gpts))
    return Waves(
        array,
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[
            PhononParityAxis(values=("real", "twin", "static")),
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
    """Tests for the phonon_order=("all", "one_phonon", "multi_phonon")
    ensemble output when exit_waves carries a PhononParityAxis (issue #373).
    """

    def test_shape_and_axes(self):
        e_values = [0.02, 0.05, 0.10]
        waves = _make_parity_exit_waves(e_values, n_configs=6)

        dp = phonon_loss_diffraction_patterns(waves)

        from abtem.core.axes import OrdinalAxis

        phonon_order_axis = next(
            ax for ax in dp.ensemble_axes_metadata
            if isinstance(ax, OrdinalAxis) and ax.label == "phonon_order"
        )
        assert phonon_order_axis.values == ("all", "one_phonon", "multi_phonon")
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

    def test_all_slot_matches_direct_tds_call(self):
        e_values = [0.02, 0.05, 0.10]
        waves = _make_parity_exit_waves(e_values, n_configs=6)

        dp = phonon_loss_diffraction_patterns(waves)

        waves_real = waves[(0, slice(None), slice(None))]
        dp_direct = phonon_loss_diffraction_patterns(waves_real, component="tds")

        np.testing.assert_allclose(dp.array[0], dp_direct.array)

    def test_one_phonon_and_multi_phonon_isolate_known_signals(self):
        """Construct real/twin/static so that psi_odd and psi_diff are
        exactly known, independently-verifiable signals."""
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
        static_b = np.broadcast_to(static, shape)

        # real = static + delta + eps, twin = static - delta + eps
        #   => psi_odd  = (real - twin) / 2 = delta          (one-phonon)
        #   => psi_diff = (real + twin) / 2 - static = eps   (multi-phonon)
        real = static_b + delta + eps
        twin = static_b - delta + eps

        waves = _make_parity_exit_waves(
            e_values, n_configs=n_configs, gpts=gpts,
            real=real, twin=twin, static=static_b,
        )
        dp = phonon_loss_diffraction_patterns(waves, max_angle="full")

        # independent reference: FFT delta/eps directly and average |.|^2
        # over the frozen-phonon axis (axis=1 of shape (n_e, n_c, gpts, gpts))
        delta_waves = Waves(
            delta, energy=100e3, sampling=0.1,
            ensemble_axes_metadata=waves.ensemble_axes_metadata[1:],
        )
        eps_waves = Waves(
            eps, energy=100e3, sampling=0.1,
            ensemble_axes_metadata=waves.ensemble_axes_metadata[1:],
        )
        I_one_phonon_ref = delta_waves.diffraction_patterns(
            max_angle="full"
        ).array.mean(axis=1)
        I_multi_phonon_ref = eps_waves.diffraction_patterns(
            max_angle="full"
        ).array.mean(axis=1)

        np.testing.assert_allclose(dp.array[1], I_one_phonon_ref, rtol=1e-4)
        np.testing.assert_allclose(dp.array[2], I_multi_phonon_ref, rtol=1e-4)

    def test_requires_fully_expanded_parity_axis(self):
        e_values = [0.02, 0.05]
        waves = _make_exit_waves(e_values, n_configs=4)
        # manually attach a not-yet-expanded (length 2) PhononParityAxis
        array = np.stack([waves.array, waves.array], axis=0)
        bad_waves = Waves(
            array, energy=100e3, sampling=0.1,
            ensemble_axes_metadata=[PhononParityAxis(values=("real", "twin"))]
            + waves.ensemble_axes_metadata,
        )
        with pytest.raises(ValueError, match="fully expanded"):
            phonon_loss_diffraction_patterns(bad_waves)

    def test_temperature_unfolds_all_three_slots(self):
        e_values = [0.0, 0.02, 0.05]
        waves = _make_parity_exit_waves(e_values, n_configs=6)

        dp = phonon_loss_diffraction_patterns(waves, temperature=300.0)

        energy_axis = next(
            ax for ax in dp.ensemble_axes_metadata if isinstance(ax, EnergyLossAxis)
        )
        assert len(energy_axis.values) == 2 * len(e_values) - 1
        assert dp.array.shape[0] == 3
        assert dp.array.shape[1] == 2 * len(e_values) - 1

    def test_lazy_matches_eager(self):
        e_values = [0.02, 0.05, 0.10]
        waves_eager = _make_parity_exit_waves(e_values, n_configs=6, lazy=False)
        waves_lazy = _make_parity_exit_waves(e_values, n_configs=6, lazy=True)

        dp_eager = phonon_loss_diffraction_patterns(waves_eager)
        dp_lazy = phonon_loss_diffraction_patterns(waves_lazy)

        assert isinstance(dp_lazy.array, da.core.Array)
        np.testing.assert_allclose(dp_lazy.array.compute(), dp_eager.array, rtol=1e-4)
