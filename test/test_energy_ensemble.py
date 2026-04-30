"""Tests for energy ensemble support in PlaneWave, Probe, Waves, BlochWaves, and SMatrix."""

import numpy as np
import os
import tempfile

import ase
import pytest
import abtem
from abtem.bloch.dynamical import BlochWaves
from abtem.core.axes import EnergyAxis, ThicknessAxis
from abtem.measurements import IndexedDiffractionPatterns
from abtem.prism.s_matrix import SMatrix, SMatrixArray
from abtem.multislice import RealSpaceMultislice
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

    def test_realspace_multislice_eager(self):
        """PlaneWave with list energy completes RealSpaceMultislice without EnergyUndefinedError."""
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
        result = pw.multislice(
            potential, algorithm=RealSpaceMultislice(order=1)
        ).compute()
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

    def test_multislice_eager(self):
        """Probe.multislice(lazy=False) with energy ensemble produces per-energy exit waves."""
        import ase.build
        atoms = ase.build.mx2(vacuum=2)
        atoms = abtem.orthogonalize_cell(atoms)
        potential = abtem.Potential(atoms, sampling=0.1)
        repeated = abtem.CrystalPotential(potential, (2, 1, 1))
        probe = Probe(
            sampling=0.1, extent=10, energy=[40e3, 60e3, 80e3], semiangle_cutoff=20
        ).match_grid(repeated)
        result = probe.multislice(repeated, lazy=False)
        assert isinstance(result, Waves)
        energy_axes = [
            ax for ax in result.ensemble_axes_metadata if isinstance(ax, EnergyAxis)
        ]
        assert len(energy_axes) == 1
        assert tuple(energy_axes[0].values) == (40e3, 60e3, 80e3)


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

    def test_angular_sampling_uses_max_energy(self):
        """Waves.angular_sampling uses max energy for energy-ensemble waves."""
        from abtem.core.energy import energy2wavelength
        arr = np.ones((3, 32, 32), dtype=complex)
        w = Waves(arr, energy=[80e3, 100e3, 120e3], sampling=0.1)
        sampling = w.angular_sampling
        assert len(sampling) == 2 and all(a > 0 for a in sampling)
        wl = energy2wavelength(120e3)  # max energy
        expected = (
            w.reciprocal_space_sampling[0] * wl * 1e3,
            w.reciprocal_space_sampling[1] * wl * 1e3,
        )
        assert np.allclose(sampling, expected)

    def test_indexing_propagates_energy_to_metadata(self):
        """Indexing a member from an energy ensemble populates metadata['energy']."""
        arr = np.ones((3, 32, 32), dtype=complex)
        w = Waves(arr, energy=[80e3, 100e3, 120e3], sampling=0.1)
        assert w.metadata.get("energy") is None  # full ensemble: no scalar energy
        assert w[0].metadata["energy"] == 80e3
        assert w[1].metadata["energy"] == 100e3
        assert w[2].metadata["energy"] == 120e3

    def test_indexed_member_valid_energy(self):
        """_valid_energy on an indexed member returns the correct per-member energy."""
        arr = np.ones((3, 32, 32), dtype=complex)
        w = Waves(arr, energy=[80e3, 100e3, 120e3], sampling=0.1)
        assert w[0]._valid_energy == 80e3
        assert w[1]._valid_energy == 100e3
        assert w[2]._valid_energy == 120e3

    def test_indexed_member_angular_sampling(self):
        """angular_sampling on an indexed member uses the exact per-member energy."""
        from abtem.core.energy import energy2wavelength
        arr = np.ones((3, 32, 32), dtype=complex)
        w = Waves(arr, energy=[80e3, 100e3, 120e3], sampling=0.1)
        for i, energy in enumerate([80e3, 100e3, 120e3]):
            wl = energy2wavelength(energy)
            expected = (
                w.reciprocal_space_sampling[0] * wl * 1e3,
                w.reciprocal_space_sampling[1] * wl * 1e3,
            )
            assert np.allclose(w[i].angular_sampling, expected)


class TestWavesEnergyEnsembleDiffractionPatterns:
    """Regression tests for three failures reported by a user working with an
    energy-ensemble exit-wave stack produced by RealSpaceMultislice.

    All three failures share the same root cause: when energy is stored as an
    EnergyAxis in ensemble_axes_metadata, metadata["energy"] is None, causing
    energy-dependent geometry calculations to raise TypeError / EnergyUndefinedError.
    """

    def _exit_waves(self):
        """Minimal energy-ensemble Waves (3 energies, 32×32 grid)."""
        arr = np.ones((3, 32, 32), dtype=complex)
        return Waves(arr, energy=[80e3, 100e3, 120e3], sampling=0.1)

    def test_diffraction_patterns_default_angle(self):
        """diffraction_patterns() without max_angle works for energy ensemble."""
        dp = self._exit_waves().diffraction_patterns()
        assert dp.array.shape[0] == 3
        assert isinstance(dp.ensemble_axes_metadata[0], EnergyAxis)

    def test_diffraction_patterns_max_angle(self):
        """diffraction_patterns(max_angle=30) no longer raises EnergyUndefinedError."""
        dp = self._exit_waves().diffraction_patterns(max_angle=30)
        assert dp.array.shape[0] == 3
        # angular_sampling should be computable without error
        assert all(a > 0 for a in dp.angular_sampling)

    def test_dp_get_energy_fallback(self):
        """Full ensemble: _get_energy() falls back to the first EnergyAxis value."""
        dp = self._exit_waves().diffraction_patterns()
        assert dp.metadata.get("energy") is None   # no scalar energy on full ensemble
        assert dp._get_energy() == 80e3            # resolved from EnergyAxis

    def test_indexed_member_has_correct_energy(self):
        """Indexing a member propagates the per-member energy into metadata."""
        dp = self._exit_waves().diffraction_patterns()
        assert dp[0].metadata["energy"] == 80e3
        assert dp[1].metadata["energy"] == 100e3
        assert dp[2].metadata["energy"] == 120e3

    def test_indexed_member_get_energy(self):
        """_get_energy() on an indexed member returns the exact per-member energy."""
        dp = self._exit_waves().diffraction_patterns()
        assert dp[0]._get_energy() == 80e3
        assert dp[2]._get_energy() == 120e3

    def test_indexed_member_angular_sampling(self):
        """angular_sampling on an indexed DiffractionPatterns uses exact per-member energy."""
        from abtem.core.energy import energy2wavelength
        dp = self._exit_waves().diffraction_patterns()
        for i, energy in enumerate([80e3, 100e3, 120e3]):
            wl = energy2wavelength(energy)
            expected = (
                dp.sampling[0] * wl * 1e3,
                dp.sampling[1] * wl * 1e3,
            )
            assert np.allclose(dp[i].angular_sampling, expected)

    def test_block_direct(self):
        """block_direct() no longer raises TypeError on energy-ensemble patterns."""
        dp = self._exit_waves().diffraction_patterns()
        blocked = dp.block_direct()
        assert blocked.array.shape == dp.array.shape

    def test_angular_sampling_and_max_angles(self):
        """angular_sampling and max_angles work on energy-ensemble DiffractionPatterns."""
        dp = self._exit_waves().diffraction_patterns()
        assert all(a > 0 for a in dp.angular_sampling)
        assert all(a > 0 for a in dp.max_angles)

    def test_index_diffraction_spots(self):
        """index_diffraction_spots() no longer raises TypeError on energy-ensemble patterns."""
        from ase.build import bulk
        atoms = bulk("Al", cubic=True)
        dp = self._exit_waves().diffraction_patterns()
        result = dp.index_diffraction_spots(cell=atoms.cell)
        assert result is not None


class TestCTFEnergyEnsemble:
    """Regression tests for CTF / Aperture with energy-ensemble Probe."""

    def _probe(self):
        return Probe(
            sampling=0.05, extent=20, energy=[40e3, 60e3, 80e3], semiangle_cutoff=20
        )

    def test_probe_ctf_is_energy_ensemble(self):
        """probe.ctf must not raise and must expose EnergyAxis."""
        ctf = self._probe().ctf
        assert len(ctf.ensemble_axes_metadata) == 1
        assert isinstance(ctf.ensemble_axes_metadata[0], EnergyAxis)
        assert tuple(ctf.ensemble_axes_metadata[0].values) == (40e3, 60e3, 80e3)

    def test_ctf_scalar_energy_no_ensemble_axis(self):
        """A single-energy CTF has no energy ensemble axis."""
        from abtem.transfer import CTF
        ctf = CTF(energy=60e3, semiangle_cutoff=20)
        assert ctf.energy == 60e3
        assert ctf._energy_distribution is None
        energy_axes = [a for a in ctf.ensemble_axes_metadata if isinstance(a, EnergyAxis)]
        assert energy_axes == []

    def test_ctf_to_diffraction_patterns(self):
        """to_diffraction_patterns() on an energy-ensemble CTF produces ensemble output."""
        dp = self._probe().ctf.to_diffraction_patterns(max_angle=20)
        assert dp.array.shape[0] == 3
        assert isinstance(dp.ensemble_axes_metadata[0], EnergyAxis)

    def test_aperture_energy_ensemble(self):
        """Aperture also accepts energy as a distribution via the same path."""
        from abtem.transfer import Aperture
        from abtem.distributions import DistributionFromValues

        ap = Aperture(semiangle_cutoff=20, energy=DistributionFromValues([40e3, 60e3, 80e3]))
        assert len(ap.ensemble_axes_metadata) == 1
        assert isinstance(ap.ensemble_axes_metadata[0], EnergyAxis)


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


@pytest.mark.slow
@pytest.mark.xdist_group("bloch_energy_ensemble")
class TestBlochWavesEnergyEnsemble:
    """Verify that BlochWaves accepts a list of energies and returns correctly
    shaped, stacked results without requiring manual loops."""

    @pytest.fixture(scope="class")
    def bw_multi(self):
        """Shared BlochWaves object with multiple energies (expensive to create)."""
        atoms = _srtio3_atoms()
        return BlochWaves(
            atoms, energy=BLOCH_ENERGIES, sg_max=BLOCH_SG_MAX, g_max=BLOCH_G_MAX
        )

    @pytest.fixture(scope="class")
    def dp_single(self, bw_multi):
        """Cached single-thickness diffraction patterns (used by multiple tests)."""
        return bw_multi.calculate_diffraction_patterns(BLOCH_THICKNESS[0]).compute()

    @pytest.fixture(scope="class")
    def dp_multi(self, bw_multi):
        """Cached multi-thickness diffraction patterns."""
        return bw_multi.calculate_diffraction_patterns(BLOCH_THICKNESS).compute()

    @pytest.fixture(scope="class")
    def ew_single(self, bw_multi):
        """Cached single-thickness exit waves."""
        return bw_multi.calculate_exit_waves(BLOCH_THICKNESS[0]).compute()

    @pytest.fixture(scope="class")
    def ew_multi(self, bw_multi):
        """Cached multi-thickness exit waves."""
        return bw_multi.calculate_exit_waves(BLOCH_THICKNESS).compute()

    def test_scalar_energy_unchanged(self):
        """Scalar energy still works as before (no regression)."""
        atoms = _srtio3_atoms()
        bw = BlochWaves(atoms, energy=100e3, sg_max=BLOCH_SG_MAX, g_max=BLOCH_G_MAX)
        assert bw._energy_hkl_masks is None
        result = bw.calculate_diffraction_patterns(BLOCH_THICKNESS[0])
        assert isinstance(result, IndexedDiffractionPatterns)

    def test_union_mask_is_superset(self, bw_multi):
        """The union _hkl_mask must be at least as large as each per-energy mask."""
        assert bw_multi._energy_hkl_masks is not None
        n_union = int(bw_multi._hkl_mask.sum())
        for sub in bw_multi._energy_hkl_masks:
            assert int(sub.sum()) <= n_union

    def test_diffraction_patterns_shape(self, bw_multi, dp_single):
        """calculate_diffraction_patterns returns (n_energies, n_beams) array."""
        assert isinstance(dp_single, IndexedDiffractionPatterns)
        n_energies = len(BLOCH_ENERGIES)
        n_union = int(bw_multi._hkl_mask.sum())
        assert dp_single.array.shape == (n_energies, n_union)
        assert isinstance(dp_single.ensemble_axes_metadata[0], EnergyAxis)
        assert dp_single.ensemble_axes_metadata[0].values == tuple(BLOCH_ENERGIES)

    def test_diffraction_patterns_multi_thickness_shape(self, bw_multi, dp_multi):
        """calculate_diffraction_patterns with thicknesses: (n_energies, n_thick, n_beams)."""
        n_energies = len(BLOCH_ENERGIES)
        n_union = int(bw_multi._hkl_mask.sum())
        n_thick = len(BLOCH_THICKNESS)
        assert dp_multi.array.shape == (n_energies, n_thick, n_union)
        assert isinstance(dp_multi.ensemble_axes_metadata[0], EnergyAxis)
        assert isinstance(dp_multi.ensemble_axes_metadata[1], ThicknessAxis)

    def test_exit_waves_shape(self, ew_single):
        """calculate_exit_waves returns (n_energies, ny, nx) Waves."""
        assert isinstance(ew_single, Waves)
        n_energies = len(BLOCH_ENERGIES)
        assert ew_single.array.shape[0] == n_energies
        assert isinstance(ew_single.ensemble_axes_metadata[0], EnergyAxis)
        assert ew_single.ensemble_axes_metadata[0].values == tuple(BLOCH_ENERGIES)

    def test_exit_waves_multi_thickness_shape(self, ew_multi):
        """calculate_exit_waves with thicknesses: (n_energies, n_thick, ny, nx)."""
        n_energies = len(BLOCH_ENERGIES)
        n_thick = len(BLOCH_THICKNESS)
        assert ew_multi.array.shape[0] == n_energies
        assert ew_multi.array.shape[1] == n_thick

    def test_inactive_beams_are_zero(self, bw_multi, dp_single):
        """Beams inactive at a given energy must have zero intensity in the output."""
        for i, sub in enumerate(bw_multi._energy_hkl_masks):
            inactive = ~sub  # positions active in union but NOT at energy i
            if inactive.any():
                np.testing.assert_array_equal(
                    dp_single.array[i, inactive], 0.0,
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
