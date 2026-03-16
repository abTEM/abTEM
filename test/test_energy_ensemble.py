"""Tests for energy ensemble support in PlaneWave, Probe, and Waves."""

import numpy as np
import os
import tempfile

from abtem.core.axes import EnergyAxis
from abtem.waves import PlaneWave, Probe, Waves

ENERGIES = [80e3, 200e3, 300e3]


class TestPlaneWaveEnergyEnsemble:
    def test_ensemble_shape(self):
        pw = PlaneWave(energy=ENERGIES, gpts=32, sampling=0.1)
        assert pw.ensemble_shape == (3,)

    def test_build_eager(self):
        pw = PlaneWave(energy=ENERGIES, gpts=32, sampling=0.1)
        waves = pw.build(lazy=False)
        assert waves.shape == (3, 32, 32)
        assert isinstance(waves.ensemble_axes_metadata[0], EnergyAxis)
        assert waves.ensemble_axes_metadata[0].values == tuple(ENERGIES)

    def test_build_lazy(self):
        pw = PlaneWave(energy=ENERGIES, gpts=32, sampling=0.1)
        waves = pw.build(lazy=True)
        assert waves.shape == (3, 32, 32)
        assert isinstance(waves.ensemble_axes_metadata[0], EnergyAxis)
        waves_computed = waves.compute()
        assert waves_computed.shape == (3, 32, 32)

    def test_scalar_energy_unchanged(self):
        pw = PlaneWave(energy=100e3, gpts=32, sampling=0.1)
        assert pw.accelerator.energy == 100e3
        assert pw.ensemble_shape == ()
        waves = pw.build(lazy=False)
        assert waves.shape == (32, 32)
        assert waves.energy == 100e3

    def test_numpy_array_energy(self):
        pw = PlaneWave(energy=np.array(ENERGIES), gpts=32, sampling=0.1)
        assert pw.ensemble_shape == (3,)
        waves = pw.build(lazy=False)
        assert waves.shape == (3, 32, 32)

    def test_to_zarr_from_zarr(self):
        pw = PlaneWave(energy=ENERGIES, gpts=32, sampling=0.1)
        waves = pw.build(lazy=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "waves.zarr")
            waves.to_zarr(path)
            loaded = Waves.from_zarr(path)
        assert loaded.shape == (3, 32, 32)
        assert isinstance(loaded.ensemble_axes_metadata[0], EnergyAxis)
        assert loaded.ensemble_axes_metadata[0].values == tuple(ENERGIES)

    def test_to_zarr_zip_from_zarr(self):
        pw = PlaneWave(energy=ENERGIES, gpts=32, sampling=0.1)
        waves = pw.build(lazy=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "waves.zarr.zip")
            waves.to_zarr(path)
            loaded = Waves.from_zarr(path)
        assert loaded.shape == (3, 32, 32)
        assert isinstance(loaded.ensemble_axes_metadata[0], EnergyAxis)
        assert loaded.ensemble_axes_metadata[0].values == tuple(ENERGIES)


class TestProbeEnergyEnsemble:
    def test_ensemble_shape(self):
        probe = Probe(energy=ENERGIES, gpts=32, sampling=0.1, semiangle_cutoff=30)
        assert probe.ensemble_shape == (3,)

    def test_build_eager(self):
        probe = Probe(energy=ENERGIES, gpts=32, sampling=0.1, semiangle_cutoff=30)
        waves = probe.build(lazy=False)
        assert waves.shape == (3, 32, 32)
        assert isinstance(waves.ensemble_axes_metadata[0], EnergyAxis)

    def test_scalar_energy_unchanged(self):
        probe = Probe(energy=100e3, gpts=32, sampling=0.1, semiangle_cutoff=30)
        assert probe.accelerator.energy == 100e3
        assert probe.ensemble_shape == ()

    def test_to_zarr_from_zarr(self):
        probe = Probe(energy=ENERGIES, gpts=32, sampling=0.1, semiangle_cutoff=30)
        waves = probe.build(lazy=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "probe_waves.zarr")
            waves.to_zarr(path)
            loaded = Waves.from_zarr(path)
        assert loaded.shape == (3, 32, 32)
        assert isinstance(loaded.ensemble_axes_metadata[0], EnergyAxis)
        assert loaded.ensemble_axes_metadata[0].values == tuple(ENERGIES)


class TestWavesEnergyEnsemble:
    def test_list_energy(self):
        arr = np.ones((3, 32, 32), dtype=complex)
        w = Waves(arr, energy=[80e3, 200e3, 300e3], sampling=0.1)
        assert w.shape == (3, 32, 32)
        assert isinstance(w.ensemble_axes_metadata[0], EnergyAxis)
        assert w.ensemble_axes_metadata[0].values == (80e3, 200e3, 300e3)

    def test_scalar_energy(self):
        arr = np.ones((32, 32), dtype=complex)
        w = Waves(arr, energy=100e3, sampling=0.1)
        assert w.shape == (32, 32)
        assert w.energy == 100e3
        assert w.ensemble_axes_metadata == []

    def test_per_chunk_valid_energy(self):
        """Single-element EnergyAxis in ensemble metadata resolves via _valid_energy."""
        arr = np.ones((1, 32, 32), dtype=complex)
        energy_axis = EnergyAxis(values=(100e3,))
        w = Waves(arr, sampling=0.1, ensemble_axes_metadata=[energy_axis])
        assert w._valid_energy == 100e3

    def test_to_zarr_from_zarr(self):
        arr = np.ones((3, 32, 32), dtype=complex)
        w = Waves(arr, energy=[80e3, 200e3, 300e3], sampling=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "waves.zarr")
            w.to_zarr(path)
            loaded = Waves.from_zarr(path)
        assert loaded.shape == (3, 32, 32)
        assert isinstance(loaded.ensemble_axes_metadata[0], EnergyAxis)
        assert loaded.ensemble_axes_metadata[0].values == (80e3, 200e3, 300e3)
