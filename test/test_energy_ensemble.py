"""Tests for energy ensemble support in PlaneWave, Probe, Waves, BlochWaves, and SMatrix."""

import numpy as np
import os
import tempfile

import ase
import abtem
from abtem.bloch.dynamical import BlochWaves
from abtem.core.axes import EnergyAxis, ThicknessAxis
from abtem.measurements import IndexedDiffractionPatterns
from abtem.prism.s_matrix import SMatrix, SMatrixArray
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

    def test_multislice_eager(self):
        """PlaneWave with list energy completes multislice without EnergyUndefinedError."""
        unit_cell = ase.Atoms(
            symbols="SrTiO3",
            scaled_positions=[
                [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0], [0.0, 0.5, 0.5],
            ],
            cell=[3.9127, 3.9127, 3.9127],
            pbc=True,
        )
        potential_unit = abtem.Potential(unit_cell, sampling=0.1, projection="finite")
        potential = abtem.CrystalPotential(potential_unit, repetitions=(1, 1, 2))
        pw = abtem.PlaneWave(energy=[100e3, 200e3, 300e3])
        pw.grid.match(potential)
        result = pw.multislice(potential).compute()
        assert result.array.shape[0] == 3


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


# ---------------------------------------------------------------------------
# SrTiO3 fixture used by BlochWaves tests
# ---------------------------------------------------------------------------

def _srtio3_atoms():
    return ase.Atoms(
        symbols="SrTiO3",
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ],
        cell=[3.9127, 3.9127, 3.9127],
        pbc=True,
    )


BLOCH_ENERGIES = [100e3, 200e3, 300e3]
BLOCH_SG_MAX = 0.1
BLOCH_G_MAX = 8.0
BLOCH_THICKNESS = [20.0, 40.0]


class TestBlochWavesEnergyEnsemble:
    """Verify that BlochWaves accepts a list of energies and returns correctly
    shaped, stacked results without requiring manual loops."""

    def test_scalar_energy_unchanged(self):
        """Scalar energy still works as before (no regression)."""
        atoms = _srtio3_atoms()
        bw = BlochWaves(atoms, energy=100e3, sg_max=BLOCH_SG_MAX, g_max=BLOCH_G_MAX)
        assert bw._energy_hkl_masks is None
        result = bw.calculate_diffraction_patterns(BLOCH_THICKNESS[0])
        assert isinstance(result, IndexedDiffractionPatterns)

    def test_union_mask_is_superset(self):
        """The union _hkl_mask must be at least as large as each per-energy mask."""
        atoms = _srtio3_atoms()
        bw = BlochWaves(
            atoms, energy=BLOCH_ENERGIES, sg_max=BLOCH_SG_MAX, g_max=BLOCH_G_MAX
        )
        assert bw._energy_hkl_masks is not None
        n_union = int(bw._hkl_mask.sum())
        for sub in bw._energy_hkl_masks:
            assert int(sub.sum()) <= n_union

    def test_diffraction_patterns_shape(self):
        """calculate_diffraction_patterns returns (n_energies, n_beams) array."""
        atoms = _srtio3_atoms()
        bw = BlochWaves(
            atoms, energy=BLOCH_ENERGIES, sg_max=BLOCH_SG_MAX, g_max=BLOCH_G_MAX
        )
        result = bw.calculate_diffraction_patterns(BLOCH_THICKNESS[0]).compute()
        assert isinstance(result, IndexedDiffractionPatterns)
        n_energies = len(BLOCH_ENERGIES)
        n_union = int(bw._hkl_mask.sum())
        assert result.array.shape == (n_energies, n_union)
        assert isinstance(result.ensemble_axes_metadata[0], EnergyAxis)
        assert result.ensemble_axes_metadata[0].values == tuple(BLOCH_ENERGIES)

    def test_diffraction_patterns_multi_thickness_shape(self):
        """calculate_diffraction_patterns with thicknesses: (n_energies, n_thick, n_beams)."""
        atoms = _srtio3_atoms()
        bw = BlochWaves(
            atoms, energy=BLOCH_ENERGIES, sg_max=BLOCH_SG_MAX, g_max=BLOCH_G_MAX
        )
        result = bw.calculate_diffraction_patterns(BLOCH_THICKNESS).compute()
        n_energies = len(BLOCH_ENERGIES)
        n_union = int(bw._hkl_mask.sum())
        n_thick = len(BLOCH_THICKNESS)
        assert result.array.shape == (n_energies, n_thick, n_union)
        assert isinstance(result.ensemble_axes_metadata[0], EnergyAxis)
        assert isinstance(result.ensemble_axes_metadata[1], ThicknessAxis)

    def test_exit_waves_shape(self):
        """calculate_exit_waves returns (n_energies, ny, nx) Waves."""
        atoms = _srtio3_atoms()
        bw = BlochWaves(
            atoms, energy=BLOCH_ENERGIES, sg_max=BLOCH_SG_MAX, g_max=BLOCH_G_MAX
        )
        result = bw.calculate_exit_waves(BLOCH_THICKNESS[0]).compute()
        assert isinstance(result, Waves)
        n_energies = len(BLOCH_ENERGIES)
        assert result.array.shape[0] == n_energies
        assert isinstance(result.ensemble_axes_metadata[0], EnergyAxis)
        assert result.ensemble_axes_metadata[0].values == tuple(BLOCH_ENERGIES)

    def test_exit_waves_multi_thickness_shape(self):
        """calculate_exit_waves with thicknesses: (n_energies, n_thick, ny, nx)."""
        atoms = _srtio3_atoms()
        bw = BlochWaves(
            atoms, energy=BLOCH_ENERGIES, sg_max=BLOCH_SG_MAX, g_max=BLOCH_G_MAX
        )
        result = bw.calculate_exit_waves(BLOCH_THICKNESS).compute()
        n_energies = len(BLOCH_ENERGIES)
        n_thick = len(BLOCH_THICKNESS)
        assert result.array.shape[0] == n_energies
        assert result.array.shape[1] == n_thick

    def test_inactive_beams_are_zero(self):
        """Beams inactive at a given energy must have zero intensity in the output."""
        atoms = _srtio3_atoms()
        bw = BlochWaves(
            atoms, energy=BLOCH_ENERGIES, sg_max=BLOCH_SG_MAX, g_max=BLOCH_G_MAX
        )
        result = bw.calculate_diffraction_patterns(BLOCH_THICKNESS[0]).compute()
        for i, sub in enumerate(bw._energy_hkl_masks):
            inactive = ~sub  # positions active in union but NOT at energy i
            if inactive.any():
                np.testing.assert_array_equal(
                    result.array[i, inactive], 0.0,
                    err_msg=f"Inactive beams non-zero at energy index {i}",
                )


# ---------------------------------------------------------------------------
# SMatrix (PRISM) energy ensemble tests
# ---------------------------------------------------------------------------

PRISM_ENERGIES = [100e3, 200e3, 300e3]
PRISM_SEMIANGLE = 20.0
PRISM_GPTS = 64
PRISM_SAMPLING = 0.1


class TestSMatrixEnergyEnsemble:
    def test_scalar_energy_unchanged(self):
        s = SMatrix(semiangle_cutoff=PRISM_SEMIANGLE, energy=100e3,
                    gpts=PRISM_GPTS, sampling=PRISM_SAMPLING)
        assert s.energy == 100e3
        assert s.ensemble_shape == ()
        assert s.ensemble_axes_metadata == []

    def test_multi_energy_ensemble_shape(self):
        s = SMatrix(semiangle_cutoff=PRISM_SEMIANGLE, energy=PRISM_ENERGIES,
                    gpts=PRISM_GPTS, sampling=PRISM_SAMPLING)
        assert s.ensemble_shape == (3,)
        assert len(s.ensemble_axes_metadata) == 1
        assert isinstance(s.ensemble_axes_metadata[0], EnergyAxis)
        assert s.ensemble_axes_metadata[0].values == tuple(PRISM_ENERGIES)

    def test_build_eager_shape(self):
        s = SMatrix(semiangle_cutoff=PRISM_SEMIANGLE, energy=PRISM_ENERGIES,
                    gpts=PRISM_GPTS, sampling=PRISM_SAMPLING)
        sma = s.build(lazy=False)
        assert isinstance(sma, SMatrixArray)
        assert sma.array.shape[0] == len(PRISM_ENERGIES)
        assert isinstance(sma.ensemble_axes_metadata[0], EnergyAxis)
        assert sma.ensemble_axes_metadata[0].values == tuple(PRISM_ENERGIES)

    def test_build_lazy_shape(self):
        s = SMatrix(semiangle_cutoff=PRISM_SEMIANGLE, energy=PRISM_ENERGIES,
                    gpts=PRISM_GPTS, sampling=PRISM_SAMPLING)
        sma = s.build(lazy=True)
        assert sma.array.shape[0] == len(PRISM_ENERGIES)
        sma_computed = sma.compute()
        assert sma_computed.array.shape[0] == len(PRISM_ENERGIES)
        assert isinstance(sma_computed.ensemble_axes_metadata[0], EnergyAxis)
