"""Tests for phonon_loss_diffraction_patterns."""

import dask.array as da
import numpy as np
import pytest
from ase import units

from abtem.core.axes import EnergyLossAxis, FrozenPhononsAxis
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
