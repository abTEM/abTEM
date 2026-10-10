"""Tests verifying that potential chunking does not affect numerical results."""

import sys
import types

import numpy as np
import pytest

from utils import devices, requires_gpu, si_cubic_atoms
from ase.build import bulk

import abtem
from abtem import FrozenPhonons, PlaneWave, Potential
from abtem.core import config as abtem_config
from abtem.core.backend import asnumpy
from abtem.core.chunks import (
    _nearest_power_of_two,
    estimate_potential_chunk_size,
    estimate_scan_batch_size,
)
from abtem.core.complex import complex_exponential
from abtem.core.grid import Grid
from abtem.magnetism.iam import (
    MagneticField,
    MagneticFieldArray,
    VectorPotential,
    VectorPotentialArray,
)
from abtem.potentials.iam import BasePotential, CrystalPotential, PotentialArray


@pytest.fixture
def si_potential():
    """A silicon potential with enough slices to exercise chunking."""
    atoms = bulk("Si", cubic=True) * (2, 2, 6)
    return Potential(atoms, gpts=(64, 64), slice_thickness=1.0)


@pytest.fixture
def si_potential_with_exit_planes():
    """A silicon potential with multiple exit planes."""
    atoms = bulk("Si", cubic=True) * (2, 2, 6)
    return Potential(atoms, gpts=(64, 64), slice_thickness=1.0, exit_planes=5)


class TestNearestPowerOfTwo:
    """Unit tests for the _nearest_power_of_two helper."""

    @pytest.mark.parametrize(
        "n, expected",
        [
            (1, 1),    # already a power of two
            (2, 2),    # already a power of two
            (4, 4),    # already a power of two
            (8, 8),    # already a power of two
            (3, 2),    # int(3 * 1.25)=3; ceil=4 > 3 → lower
            (14, 16),  # int(14 * 1.25)=17; ceil=16 <= 17 → upper
            (59, 64),  # int(59 * 1.25)=73; ceil=64 <= 73 → upper
            (20, 16),  # int(20 * 1.25)=25; ceil=32 > 25 → lower
            (100, 64),  # int(100 * 1.25)=125; ceil=128 > 125 → lower
            (65, 64),  # int(65 * 1.25)=81; ceil=128 > 81 → lower
        ],
    )
    def test_rounding(self, n, expected):
        assert _nearest_power_of_two(n) == expected


def _install_fake_cupy(monkeypatch, free, total, pool_used=0):
    """Install a minimal cupy stand-in exposing the memory-probing API."""
    fake = types.ModuleType("cupy")
    fake.get_default_memory_pool = lambda: types.SimpleNamespace(
        used_bytes=lambda: pool_used
    )
    fake.cuda = types.SimpleNamespace(
        Device=lambda: types.SimpleNamespace(mem_info=(free, total))
    )
    monkeypatch.setitem(sys.modules, "cupy", fake)


class TestEstimatePotentialChunkSize:
    """Unit tests for estimate_potential_chunk_size."""

    def test_config_override(self):
        """The potential.slice-chunk-size config key must short-circuit estimation."""
        from abtem.core import config

        with config.set({"potential.slice-chunk-size": 7}):
            assert estimate_potential_chunk_size((64, 64)) == 7

    def test_cpu_returns_positive_int(self):
        assert estimate_potential_chunk_size((64, 64), device="cpu") >= 1

    def test_larger_gpts_gives_smaller_chunk(self):
        small = estimate_potential_chunk_size((64, 64), device="cpu")
        large = estimate_potential_chunk_size((512, 512), device="cpu")
        assert large <= small

    def test_gpu_chunk_size_depends_only_on_slice_bytes(self, monkeypatch):
        """An FFT-unfriendly grid must not shrink the potential chunk.

        Unlike the probe batch, potential slices are built and bandlimited one
        at a time, so the Bluestein workspace is constant in the chunk size --
        see the comment in ``estimate_potential_chunk_size``.  2623 = 43*61 and
        2271 = 3*757 force the Bluestein fallback; 2625 = 3*5^3*7 and
        2268 = 2^2*3^4*7 do not, and the two grids differ in area by 0.06 %.
        Both must give the same chunk, fixed by bytes alone:
        int(0.35 * 40 GB / (2623*2271*4 * 5)) = 117.  A doubled overhead like
        the one in ``estimate_scan_batch_size`` would give 58.
        """
        _install_fake_cupy(monkeypatch, free=40_000_000_000, total=40_000_000_000)
        dtype = np.dtype(np.float32)
        bluestein = estimate_potential_chunk_size((2623, 2271), "gpu", dtype)
        fast = estimate_potential_chunk_size((2625, 2268), "gpu", dtype)
        assert bluestein == fast == 117

    def test_device_chunk_size_counts_every_element_of_a_slice(self, monkeypatch):
        """The shape of a slice may carry a component axis: a magnetic slice of
        (3, 2623, 2271) takes three times the bytes of a (2623, 2271) one.
        int(0.35 * 40 GB / (3*2623*2271*4 * 5)) = 39."""
        _install_fake_cupy(monkeypatch, free=40_000_000_000, total=40_000_000_000)
        dtype = np.dtype(np.float32)
        assert estimate_potential_chunk_size((3, 2623, 2271), "gpu", dtype) == 39


class TestEstimateScanBatchSize:
    """Unit tests for the VRAM-aware scan-batch estimator (GPU path mocked)."""

    def test_fast_radix_grid(self, monkeypatch):
        _install_fake_cupy(monkeypatch, free=40_000_000_000, total=40_000_000_000)
        # budget = 20 GB; per probe = 2048² x 16 B x 6 -> 49 probes -> pow2 32
        assert estimate_scan_batch_size((2048, 2048), np.complex128, "gpu") == 32

    def test_bluestein_grid_uses_doubled_overhead(self, monkeypatch):
        _install_fake_cupy(monkeypatch, free=40_000_000_000, total=40_000_000_000)
        # 2623 = 43*61 and 2271 = 3*757 force the Bluestein FFT fallback;
        # per probe = 2623*2271 x 16 B x 12 -> 17 probes -> pow2 16.
        # (The 6x factor would have given 34 -> 32.)
        assert estimate_scan_batch_size((2623, 2271), np.complex128, "gpu") == 16

    def test_pool_usage_reduces_batch(self, monkeypatch):
        _install_fake_cupy(
            monkeypatch,
            free=40_000_000_000,
            total=40_000_000_000,
            pool_used=30_000_000_000,
        )
        # effective free = min(free, total - pool_used) = 10 GB -> budget 5 GB
        assert estimate_scan_batch_size((2048, 2048), np.complex128, "gpu") <= 16

    def test_cpu_falls_back_to_chunk_size(self):
        assert estimate_scan_batch_size((2048, 2048), np.complex128, "cpu") >= 1


class TestChunkedSlicesCorrectness:
    """Verify that generate_chunked_slices reproduces generate_slices exactly."""

    def test_single_chunk_matches_build(self, si_potential):
        """Chunk size larger than total slices should match build()."""
        built = si_potential.build(lazy=False)
        chunks = list(
            si_potential.generate_chunked_slices(chunk_size=len(si_potential) + 1)
        )
        assert len(chunks) == 1
        assert np.allclose(chunks[0].array, built.array)

    def test_chunk_size_1_matches_generate_slices(self, si_potential):
        """Chunk size 1 should yield identical slices to generate_slices."""
        ref_slices = list(si_potential.generate_slices())
        chunked_slices = list(si_potential.generate_chunked_slices(chunk_size=1))

        assert len(ref_slices) == len(chunked_slices)
        for ref, chunked in zip(ref_slices, chunked_slices):
            assert np.allclose(ref.array, chunked.array)
            assert ref.slice_thickness == chunked.slice_thickness

    @pytest.mark.parametrize("chunk_size", [2, 3, 5, 7])
    def test_various_chunk_sizes_match_build(self, si_potential, chunk_size):
        """All chunk sizes should reconstruct the same full potential."""
        built = si_potential.build(lazy=False)
        chunks = list(si_potential.generate_chunked_slices(chunk_size=chunk_size))

        reconstructed = np.concatenate([c.array for c in chunks], axis=0)
        assert reconstructed.shape == built.array.shape
        assert np.allclose(reconstructed, built.array)

    def test_exit_planes_preserved_across_chunks(self, si_potential_with_exit_planes):
        """Exit planes must be correctly assigned regardless of chunk boundaries."""
        potential = si_potential_with_exit_planes
        ref_exit_plane_after = potential._exit_plane_after

        for chunk_size in [2, 3, 5]:
            offset = 0
            for chunk in potential.generate_chunked_slices(chunk_size=chunk_size):
                n = len(chunk)
                expected = np.where(ref_exit_plane_after[offset : offset + n])[0]
                assert tuple(expected) == chunk.exit_planes, (
                    f"Exit planes mismatch at offset {offset} with "
                    f"chunk_size={chunk_size}"
                )
                offset += n
            assert offset == len(potential)

    def test_slice_thicknesses_preserved(self, si_potential):
        """Slice thicknesses must match the original across all chunks."""
        ref_thickness = si_potential.slice_thickness
        for chunk_size in [2, 4]:
            thicknesses = []
            for chunk in si_potential.generate_chunked_slices(chunk_size=chunk_size):
                thicknesses.extend(chunk.slice_thickness)
            assert tuple(thicknesses) == ref_thickness


class TestPotentialArrayChunkedSlices:
    """Verify chunked slices on pre-built PotentialArray."""

    def test_views_match_original(self, si_potential):
        """Chunks from PotentialArray should be views into the original array."""
        built = si_potential.build(lazy=False)

        for chunk_size in [3, 5]:
            reconstructed = np.concatenate(
                [c.array for c in built.generate_chunked_slices(chunk_size=chunk_size)],
                axis=0,
            )
            assert np.allclose(reconstructed, built.array)

    def test_dask_backed_potential_array(self, si_potential):
        """Chunked slices should also work on dask-backed PotentialArray."""
        built_lazy = si_potential.build(lazy=True)
        built_eager = si_potential.build(lazy=False)

        chunks_lazy = list(built_lazy.generate_chunked_slices(chunk_size=3))
        reconstructed = np.concatenate(
            [c.compute().array if hasattr(c, "compute") else c.array for c in chunks_lazy],
            axis=0,
        )
        assert np.allclose(reconstructed, built_eager.array)


class TestMultisliceWithChunking:
    """Verify that multislice results are identical with different chunk sizes."""

    @pytest.mark.parametrize("chunk_size", [1, 2, 5, 100])
    def test_plane_wave_chunked_vs_unchunked(self, si_potential, chunk_size):
        """PlaneWave multislice must give identical results for all chunk sizes."""
        waves = PlaneWave(energy=200e3, gpts=si_potential.gpts)
        waves.grid.match(si_potential)

        # Reference: large chunk (all slices at once)
        ref = waves.multislice(
            si_potential, potential_chunk_size=len(si_potential) + 1, lazy=False
        )

        # Test: specific chunk size
        result = waves.multislice(
            si_potential, potential_chunk_size=chunk_size, lazy=False
        )

        assert np.allclose(ref.array, result.array), (
            f"Mismatch with chunk_size={chunk_size}, "
            f"max diff={np.abs(ref.array - result.array).max()}"
        )

    def test_chunked_matches_prebuilt_potential_array(self, si_potential):
        """Chunked unbuilt potential must match passing a pre-built PotentialArray.

        This is stronger than comparing two chunked runs: the pre-built path
        exercises ``FieldArray.generate_chunked_slices`` (array-view slicing)
        while the unbuilt path exercises ``_FieldBuilder.generate_chunked_slices``
        (on-the-fly build). Agreement between the two confirms that neither
        chunker introduces numerical error relative to the underlying atom
        integration.
        """
        waves = PlaneWave(energy=200e3, gpts=si_potential.gpts)
        waves.grid.match(si_potential)

        prebuilt = si_potential.build(lazy=False)
        ref = waves.multislice(prebuilt, lazy=False)

        result = waves.multislice(si_potential, potential_chunk_size=3, lazy=False)

        assert np.allclose(ref.array, result.array, atol=1e-6), (
            f"Chunked unbuilt potential differs from pre-built reference; "
            f"max diff={np.abs(ref.array - result.array).max()}"
        )

    def test_exit_planes_with_chunking(self, si_potential_with_exit_planes):
        """Thickness series measurements must be identical with chunking."""
        potential = si_potential_with_exit_planes
        waves = PlaneWave(energy=200e3, gpts=potential.gpts)
        waves.grid.match(potential)

        ref = waves.multislice(
            potential, potential_chunk_size=len(potential) + 1, lazy=False
        )
        chunked = waves.multislice(potential, potential_chunk_size=3, lazy=False)

        assert ref.array.shape == chunked.array.shape
        assert np.allclose(ref.array, chunked.array), (
            f"Exit plane mismatch, max diff={np.abs(ref.array - chunked.array).max()}"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    def test_lazy_vs_eager_with_chunking(self, si_potential, lazy):
        """Lazy and eager paths should give identical results with chunking."""
        waves = PlaneWave(energy=200e3, gpts=si_potential.gpts)
        waves.grid.match(si_potential)

        result = waves.multislice(
            si_potential, potential_chunk_size=3, lazy=lazy
        )
        if hasattr(result, "compute"):
            result = result.compute()

        ref = waves.multislice(
            si_potential, potential_chunk_size=len(si_potential) + 1, lazy=False
        )

        assert np.allclose(ref.array, result.array)


class TestDiskMeshgridIter:
    """Verify that disk_meshgrid_iter produces the same indices as disk_meshgrid."""

    @pytest.mark.parametrize("r", [0, 1, 5, 20, 100])
    def test_matches_disk_meshgrid(self, r):
        from abtem.core.grid import disk_meshgrid, disk_meshgrid_iter

        reference = disk_meshgrid(r)
        # Concatenate all chunks from the iterator.
        chunks = list(disk_meshgrid_iter(r, chunk_size=500))
        if len(chunks) == 0:
            result = np.empty((0, 2), dtype=np.int32)
        else:
            result = np.concatenate(chunks)

        assert result.dtype == np.int32
        assert result.shape[1] == 2

        # Sort both by (row, col) for comparison.
        ref_sorted = reference[np.lexsort((reference[:, 1], reference[:, 0]))]
        res_sorted = result[np.lexsort((result[:, 1], result[:, 0]))]
        np.testing.assert_array_equal(ref_sorted, res_sorted)

    def test_chunk_size_respected(self):
        from abtem.core.grid import disk_meshgrid_iter

        # r=50 → π*50² ≈ 7854 indices.
        # With chunk_size=1000 we expect ~8 chunks, each ≤ ~1100 entries
        # (one row can add up to 101 entries which may slightly exceed chunk_size).
        chunks = list(disk_meshgrid_iter(50, chunk_size=1000))
        assert len(chunks) >= 7
        # No chunk should be much larger than chunk_size + max row width.
        for c in chunks:
            assert c.shape[0] <= 1000 + 2 * 50 + 1


class TestFiniteProjectionChunked:
    """Verify finite projection builds correctly and is deterministic."""

    def test_finite_build_obeys_the_sum_rule(self):
        """The finite projection of 64 Si atoms must integrate to 64 F_Si(0).

        Oracle: the k = 0 sum rule int V d^3r = F(0), with F(0) the Lobato
        ``projected_scattering_factor`` at k = 0 (see test_potential_physics).
        Budget: default-cutoff truncation (<= 0.2 % for Si) plus the core-pixel
        discretisation, which test_potential_physics measures as <= 1.8 % of
        the atom's slice at dx = 0.1 A and shows to scale as dx^2 -- so
        <= 1.3 % of the total at dx = 0.085 A. Truncation only removes
        potential, so the result must be a deficit.
        """
        from abtem.parametrizations import LobatoParametrization

        atoms = si_cubic_atoms() * (2, 2, 2)
        pot = Potential(atoms, gpts=(128, 128), slice_thickness=2.0,
                        projection="finite")
        result = pot.build(lazy=False)
        total = float(result.array.sum()) * np.prod(pot.sampling)
        f0 = float(
            LobatoParametrization().projected_scattering_factor("Si")(
                np.array([0.0])
            )[0]
        )
        deficit = 1 - total / (len(atoms) * f0)
        assert -1e-3 < deficit < 1.5e-2, deficit

    def test_finite_chunked_slices_equal_the_unchunked_build(self):
        """Chunked slice generation must reproduce the one-shot build and the
        per-slice generator exactly, and a second build must be identical."""
        atoms = si_cubic_atoms() * (2, 2, 2)
        pot = Potential(atoms, gpts=(64, 64), slice_thickness=1.0,
                        projection="finite")
        reference = pot.build(lazy=False).array
        per_slice = np.concatenate([s.array for s in pot.generate_slices()])
        np.testing.assert_array_equal(per_slice, reference)
        for chunk_size in (1, 3, 5):
            chunked = np.concatenate(
                [c.array for c in pot.generate_chunked_slices(chunk_size=chunk_size)]
            )
            np.testing.assert_array_equal(chunked, reference)
        np.testing.assert_array_equal(pot.build(lazy=False).array, reference)
        lazy = pot.build(lazy=True).compute().array
        np.testing.assert_allclose(lazy, reference, rtol=0, atol=0)

    def test_finite_multislice_chunked(self):
        """Finite-projection multislice must be identical across chunk sizes."""
        atoms = si_cubic_atoms() * (2, 2, 4)
        pot = Potential(atoms, gpts=(64, 64), slice_thickness=2.0,
                        projection="finite")
        waves = PlaneWave(energy=200e3, gpts=pot.gpts)
        waves.grid.match(pot)

        ref = waves.multislice(pot, potential_chunk_size=len(pot) + 1, lazy=False)
        chunked = waves.multislice(pot, potential_chunk_size=2, lazy=False)
        np.testing.assert_allclose(ref.array, chunked.array, atol=1e-10)


def _frozen_phonon_crystal(
    num_configs, repetitions, unit_seed=1, device="cpu", **kwargs
):
    atoms = bulk("Si", cubic=True)
    unit = Potential(
        abtem.FrozenPhonons(atoms, num_configs, sigmas=0.1, seed=unit_seed),
        gpts=(16, 16),
        slice_thickness=atoms.cell[2, 2] / 4,  # 4 slices per unit
        device=device,
    )
    return CrystalPotential(unit, repetitions, **kwargs)


class TestCrystalPotentialChunking:
    """Verify that CrystalPotential generates the correct slices when chunked."""

    @pytest.fixture
    def crystal_potential(self):
        """Si CrystalPotential: 4×4 xy tiles, 10 z-reps → 30 slices."""
        atoms = si_cubic_atoms()
        unit = Potential(atoms, gpts=(32, 32), slice_thickness=2.0)
        return CrystalPotential(unit, repetitions=(4, 4, 10))

    def test_total_slice_count(self, crystal_potential):
        """CrystalPotential must report the correct total slice count."""
        unit_slices = len(crystal_potential.potential_unit)
        assert len(crystal_potential) == unit_slices * crystal_potential.repetitions[2]

    def test_chunked_slices_correct_total(self, crystal_potential):
        """Sum of chunk lengths must equal len(potential), not a multiple."""
        for chunk_size in [1, 3, 5, len(crystal_potential)]:
            chunks = list(crystal_potential.generate_chunked_slices(chunk_size=chunk_size))
            total = sum(len(c) for c in chunks)
            assert total == len(crystal_potential), (
                f"chunk_size={chunk_size}: got {total} slices, "
                f"expected {len(crystal_potential)}"
            )

    @pytest.mark.parametrize("chunk_size", [1, 3, 5])
    def test_chunked_multislice_matches_full(self, crystal_potential, chunk_size):
        """Multislice through CrystalPotential must be identical for all chunk sizes."""
        waves = PlaneWave(energy=200e3, gpts=crystal_potential.gpts)
        waves.grid.match(crystal_potential)

        ref = waves.multislice(
            crystal_potential,
            potential_chunk_size=len(crystal_potential) + 1,
            lazy=False,
        )
        chunked = waves.multislice(
            crystal_potential,
            potential_chunk_size=chunk_size,
            lazy=False,
        )
        assert np.allclose(ref.array, chunked.array), (
            f"chunk_size={chunk_size}: "
            f"max diff={np.abs(ref.array - chunked.array).max()}"
        )

    def test_crystal_matches_explicit_supercell(self):
        """CrystalPotential multislice result must match an explicit supercell Potential."""
        atoms_unit = si_cubic_atoms()
        unit = Potential(atoms_unit, gpts=(32, 32), slice_thickness=2.0)
        crys = CrystalPotential(unit, repetitions=(2, 2, 3))

        atoms_full = si_cubic_atoms() * (2, 2, 3)
        full = Potential(atoms_full, gpts=(64, 64), slice_thickness=2.0)

        waves = PlaneWave(energy=200e3, gpts=crys.gpts)
        waves.grid.match(crys)

        ref = waves.multislice(full, lazy=False)
        result = waves.multislice(crys, lazy=False)

        # Numerical agreement: same physics, different code path.
        assert np.allclose(ref.array, result.array, atol=1e-5), (
            f"max diff={np.abs(ref.array - result.array).max()}"
        )

    def test_slice_range_within_z_rep(self, crystal_potential):
        """generate_slices with first/last slice within one z-rep must work."""
        unit_slices = len(crystal_potential.potential_unit)
        first, last = 1, unit_slices  # second slice onward in first z-rep
        slices = list(crystal_potential.generate_slices(first, last))
        assert len(slices) == last - first

    def test_slice_range_spanning_z_reps(self, crystal_potential):
        """generate_slices spanning multiple z-reps must return the right count."""
        unit_slices = len(crystal_potential.potential_unit)
        first = unit_slices - 1
        last = unit_slices + 2
        slices = list(crystal_potential.generate_slices(first, last))
        assert len(slices) == last - first

    def test_generate_chunked_slices_no_list_accumulation(self, crystal_potential):
        """Each chunk must be a single contiguous array, not a concatenation artefact.

        The override pre-allocates and fills in-place, so each chunk array
        must have exactly chunk_size slices — never more.
        """
        chunk_size = 3
        for chunk in crystal_potential.generate_chunked_slices(chunk_size=chunk_size):
            assert chunk.array.shape[0] <= chunk_size

    def test_single_configuration_chunks_hold_no_tiled_unit(self):
        """A unit with one configuration is tiled into each chunk: no tiled copy
        of the unit's slices is kept across z-repetitions."""
        import tracemalloc

        unit = Potential(
            si_cubic_atoms(), gpts=(24, 32), slice_thickness=0.5
        ).build(lazy=False)
        crystal = CrystalPotential(unit, repetitions=(4, 3, 3))
        slice_bytes = np.prod(crystal.gpts) * unit.array.dtype.itemsize

        tracemalloc.start()
        for chunk in crystal.generate_chunked_slices(chunk_size=1):
            del chunk
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # two chunks and the temporaries of one tile; the 11 tiled unit slices
        # would add 11
        assert peak < 6 * slice_bytes, peak / slice_bytes

    def test_dtype_follows_precision_config(self, crystal_potential):
        """Chunk dtype must reflect the abtem precision config (float32 / float64)."""
        for precision, expected in [("float32", np.float32), ("float64", np.float64)]:
            with abtem_config.set({"precision": precision}):
                # Re-build inside the config context so the unit is built at
                # the configured precision.
                atoms = si_cubic_atoms()
                unit = Potential(atoms, gpts=(32, 32), slice_thickness=2.0)
                crys = CrystalPotential(unit, repetitions=(4, 4, 10))
                chunks = list(crys.generate_chunked_slices(chunk_size=4))
                for chunk in chunks:
                    assert chunk.array.dtype == expected, (
                        f"precision={precision}: got {chunk.array.dtype}"
                    )

    def test_multislice_float64_matches_float32(self, crystal_potential):
        """Multislice at float64 must agree with float32 to within tolerances."""
        waves32 = PlaneWave(energy=200e3, gpts=crystal_potential.gpts)
        waves32.grid.match(crystal_potential)

        with abtem_config.set({"precision": "float32"}):
            atoms = si_cubic_atoms()
            unit32 = Potential(atoms, gpts=(32, 32), slice_thickness=2.0)
            crys32 = CrystalPotential(unit32, repetitions=(4, 4, 10))
            result32 = waves32.multislice(crys32, lazy=False)

        with abtem_config.set({"precision": "float64"}):
            atoms = si_cubic_atoms()
            unit64 = Potential(atoms, gpts=(32, 32), slice_thickness=2.0)
            crys64 = CrystalPotential(unit64, repetitions=(4, 4, 10))
            result64 = waves32.multislice(crys64, lazy=False)

        assert np.allclose(
            np.abs(result32.array), np.abs(result64.array), atol=1e-4
        ), f"max diff: {np.abs(np.abs(result32.array) - np.abs(result64.array)).max()}"

    # A pool of 3 is smaller than the 6 lateral tiles of (2, 3, .), which
    # enlarges it.
    @pytest.mark.filterwarnings("ignore:frozen-phonon pool .* is smaller:UserWarning")
    @pytest.mark.parametrize("num_configs", [1, 3, 12])
    @pytest.mark.parametrize("repetitions", [(2, 3, 2), (3, 1, 3)])
    @pytest.mark.parametrize("chunk_size", [1, 3, 5, 100])
    @pytest.mark.parametrize("slice_range", [(0, None), (3, 7)])
    @devices
    def test_frozen_phonon_chunks_equal_generate_slices(
        self, num_configs, repetitions, chunk_size, slice_range, device
    ):
        """The chunks hold the slices of generate_slices: the member's reseeded
        pool, the balanced draws and the lateral mosaic."""
        crystal = _frozen_phonon_crystal(
            num_configs, repetitions, device=device, seeds=(5,)
        )
        expected = np.stack(
            [asnumpy(s.array[0]) for s in crystal.generate_slices(*slice_range)]
        )
        chunks = list(
            crystal.generate_chunked_slices(*slice_range, chunk_size=chunk_size)
        )

        assert all(len(chunk) <= chunk_size for chunk in chunks)
        np.testing.assert_array_equal(
            np.concatenate([asnumpy(chunk.array) for chunk in chunks]), expected
        )

    @pytest.mark.filterwarnings("ignore:frozen-phonon pool .* is smaller:UserWarning")
    def test_frozen_phonon_chunks_come_from_one_generate_slices_call(
        self, monkeypatch
    ):
        """One generator serves all chunks: the pool is built once, and an
        unseeded crystal is not drawn anew for every chunk."""
        crystal = _frozen_phonon_crystal(3, (2, 3, 2))
        calls = []
        generate_slices = CrystalPotential.generate_slices

        def counting(self, *args, **kwargs):
            calls.append(args)
            return generate_slices(self, *args, **kwargs)

        monkeypatch.setattr(CrystalPotential, "generate_slices", counting)

        chunks = list(crystal.generate_chunked_slices(chunk_size=1))

        assert len(chunks) == len(crystal) == 8
        assert len(calls) == 1

    @pytest.mark.parametrize("lazy", [False, True])
    def test_frozen_phonon_multislice_matches_the_built_crystal(self, lazy):
        """Simulating the crystal directly and simulating crystal.build() use
        the same crystal, for every member and exit plane, with chunk
        boundaries that do not align with the unit cell."""

        def crystal():
            return _frozen_phonon_crystal(
                6,
                (2, 3, 2),
                num_frozen_phonons=2,
                seeds=(5, 6),
                exit_planes=3,
                ensemble_mean=False,
            )

        def run(potential, lazy):
            waves = PlaneWave(energy=100e3).multislice(
                potential, lazy=lazy, potential_chunk_size=3
            )
            return waves.compute(scheduler="synchronous").array if lazy else waves.array

        expected = run(crystal().build(lazy=False), lazy=False)
        result = run(crystal(), lazy=lazy)

        # 2 members, the entrance plane and 3 exit planes, 2 x 16 by 3 x 16 points
        assert result.shape == expected.shape == (2, 4, 32, 48)
        np.testing.assert_allclose(
            result, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
        )

    def test_frozen_phonon_prism_matches_the_built_crystal(self):
        """PRISM builds its S-matrix through the same chunked slices."""

        def scan(potential):
            return (
                abtem.SMatrix(
                    potential=potential,
                    energy=100e3,
                    semiangle_cutoff=15,
                    interpolation=1,
                )
                .scan(
                    scan=abtem.GridScan((0, 0), (2, 2), gpts=(2, 3)),
                    detectors=abtem.AnnularDetector(10, 30),
                    lazy=False,
                )
                .array
            )

        def crystal():
            return _frozen_phonon_crystal(
                6, (2, 3, 2), num_frozen_phonons=2, seeds=(5, 6)
            )

        expected = scan(crystal().build(lazy=False))
        np.testing.assert_allclose(
            scan(crystal()), expected, rtol=0, atol=1e-5 * np.abs(expected).max()
        )

    @pytest.mark.parametrize(
        "root", ["seeds", "frozen phonons", "built ensemble", "lazy built ensemble"]
    )
    def test_a_crystal_is_one_crystal_in_every_scan_block(self, root):
        """Identical probe positions in different lazy blocks see one crystal,
        whichever of the crystal's seeds, the unit's frozen phonons, or a seed
        drawn at construction sets its mosaic."""
        crystal = _frozen_phonon_crystal(
            8, (2, 3, 3), seeds=(5,) if root == "seeds" else None
        )
        if root == "built ensemble":
            crystal = CrystalPotential(
                crystal.potential_unit.build(lazy=False), crystal.repetitions
            )
        elif root == "lazy built ensemble":
            crystal = CrystalPotential(
                crystal.potential_unit.build(lazy=True), crystal.repetitions
            )
        scan = abtem.CustomScan(np.array([[1.0, 1.0]] * 4))

        # a slice budget below the crystal's 12 slices: one multislice per block
        with abtem.config.set({"potential.slice-chunk-size": 2}):
            measurement = abtem.Probe(energy=100e3, semiangle_cutoff=20).scan(
                crystal,
                scan=scan,
                detectors=abtem.AnnularDetector(10, 30),
                lazy=True,
                max_batch=1,
            )
            assert measurement.array.numblocks[-1] == 4
            values = np.asarray(measurement.compute(scheduler="synchronous").array)

        assert values.shape == (4,)
        np.testing.assert_allclose(
            values, values[0], rtol=0, atol=1e-6 * np.abs(values).max()
        )

    @pytest.mark.parametrize("lazy", [False, True])
    def test_an_unseeded_crystal_backscatters_as_it_builds(self, lazy):
        """The backward pass of full-expansion backscattering, which calls
        generate_slices again, sees the crystal of the forward pass."""
        from abtem.multislice import RealSpaceMultislice

        def crystal():
            return _frozen_phonon_crystal(6, (2, 3, 2), exit_planes=1)

        def run(potential):
            result = PlaneWave(energy=100e3).multislice(
                potential,
                lazy=lazy,
                algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
                return_backscattered=True,
            )
            if lazy:
                result = [r.compute(scheduler="synchronous") for r in result]
            return [np.asarray(r.array) for r in result]

        expected = run(crystal().build(lazy=False))
        result = run(crystal())

        assert len(result) == len(expected) == 2
        for r, e in zip(result, expected):
            assert r.shape == e.shape
            np.testing.assert_allclose(r, e, rtol=0, atol=1e-6 * np.abs(e).max())

    @pytest.mark.parametrize(
        "kwargs", [{}, {"num_frozen_phonons": 2}], ids=["single", "ensemble"]
    )
    def test_a_seeded_unit_makes_the_crystal_reproducible(self, kwargs):
        """Separately constructed crystals of a seeded unit give one result,
        eager or lazy, for any number of scan blocks."""

        def scan(lazy, max_batch):
            crystal = _frozen_phonon_crystal(
                6, (2, 3, 2), ensemble_mean=False, **kwargs
            )
            with abtem.config.set({"potential.slice-chunk-size": 2}):
                measurement = abtem.Probe(energy=100e3, semiangle_cutoff=20).scan(
                    crystal,
                    scan=abtem.GridScan((0, 0), (3, 3), gpts=(3, 2)),
                    detectors=abtem.AnnularDetector(10, 30),
                    lazy=lazy,
                    max_batch=max_batch,
                )
                if lazy:
                    measurement = measurement.compute(scheduler="threads")
            return np.asarray(measurement.array)

        expected = scan(False, 6)
        for lazy, max_batch in [(False, 6), (True, 1), (True, 4)]:
            result = scan(lazy, max_batch)
            assert result.shape == expected.shape
            np.testing.assert_allclose(
                result, expected, rtol=0, atol=1e-6 * np.abs(expected).max()
            )

    @pytest.mark.parametrize(
        "num_configs, repetitions, kwargs, expected",
        [
            (
                6,
                (2, 3, 2),
                dict(seeds=(5, 6)),
                [
                    [[[1, 4, 2], [3, 5, 0]], [[4, 0, 1], [3, 2, 5]]],
                    [[[2, 3, 0], [5, 4, 1]], [[4, 0, 2], [1, 5, 3]]],
                ],
            ),
            (
                12,
                (2, 3, 2),
                dict(num_frozen_phonons=2, seeds=11),
                [
                    [[[7, 11, 5], [8, 2, 3]], [[4, 10, 1], [9, 6, 0]]],
                    [[[0, 10, 2], [6, 3, 9]], [[7, 11, 5], [8, 4, 1]]],
                ],
            ),
            (
                4,
                (3, 1, 3),
                dict(seeds=(7,)),
                [[[[0], [3], [1]], [[0], [2], [3]], [[1], [2], [0]]]],
            ),
        ],
        ids=["seeds", "master seed", "one seed"],
    )
    def test_explicit_seeds_draw_the_same_mosaic(
        self, num_configs, repetitions, kwargs, expected
    ):
        """The pool configuration of every tile, pinned for explicit seeds."""
        crystal = _frozen_phonon_crystal(num_configs, repetitions, **kwargs)
        np.testing.assert_array_equal(_drawn_tiles(crystal), expected)

    @pytest.mark.parametrize(
        "num_configs, repetitions, kwargs, expected",
        [
            (
                6,
                (2, 3, 2),
                {},
                [[[[5, 4, 2], [0, 1, 3]], [[5, 2, 0], [3, 1, 4]]]],
            ),
            (
                12,
                (2, 3, 2),
                dict(num_frozen_phonons=2),
                [
                    [[[8, 3, 7], [5, 2, 1]], [[11, 9, 4], [0, 10, 6]]],
                    [[[1, 0, 6], [8, 7, 9]], [[2, 11, 10], [3, 5, 4]]],
                ],
            ),
        ],
        ids=["single", "ensemble"],
    )
    def test_a_seeded_unit_draws_a_pinned_mosaic(
        self, num_configs, repetitions, kwargs, expected
    ):
        """Without seeds on the crystal, the seed of the unit's frozen phonons
        fixes the pool configuration of every tile, in every session."""
        crystal = _frozen_phonon_crystal(num_configs, repetitions, **kwargs)
        np.testing.assert_array_equal(_drawn_tiles(crystal), expected)

    def test_explicit_seeds_take_precedence_over_the_unit_seed(self):
        a = _frozen_phonon_crystal(6, (2, 3, 2), unit_seed=1, seeds=(5, 6))
        b = _frozen_phonon_crystal(6, (2, 3, 2), unit_seed=2, seeds=(5, 6))
        np.testing.assert_array_equal(
            a.build(lazy=False).array, b.build(lazy=False).array
        )

        unit = _frozen_phonon_crystal(6, (1, 1, 1)).potential_unit.build(lazy=False)
        a = CrystalPotential(unit, (2, 3, 2), seeds=(5, 6))
        b = CrystalPotential(unit, (2, 3, 2), seeds=(5, 6))
        np.testing.assert_array_equal(
            a.build(lazy=False).array, b.build(lazy=False).array
        )

    @pytest.mark.parametrize(
        "seeds, expected", [(5, (5,)), ([5, 6], (5, 6))], ids=["int", "list"]
    )
    def test_seeds_take_an_int_or_a_list(self, seeds, expected):
        crystal = _frozen_phonon_crystal(6, (2, 3, 2), seeds=seeds)

        assert crystal.seeds == expected
        np.testing.assert_array_equal(
            crystal.build(lazy=False).array,
            _frozen_phonon_crystal(6, (2, 3, 2), seeds=expected)
            .build(lazy=False)
            .array,
        )

    def test_two_crystals_of_one_built_ensemble_are_different_crystals(self):
        """A crystal of a unit without frozen phonons draws its seed when it is
        created: a copy is the same crystal, another construction is not."""
        from dask.base import tokenize

        unit = _frozen_phonon_crystal(6, (1, 1, 1)).potential_unit.build(lazy=False)
        a = CrystalPotential(unit, (2, 3, 2))
        b = CrystalPotential(unit, (2, 3, 2))

        assert a == a.copy()
        assert tokenize(a) == tokenize(a.copy())
        assert a != b
        assert tokenize(a) != tokenize(b)
        assert not np.array_equal(a.build(lazy=False).array, b.build(lazy=False).array)

    def test_two_crystals_of_one_frozen_phonon_unit_are_one_crystal(self):
        unit = _frozen_phonon_crystal(6, (1, 1, 1)).potential_unit
        a = CrystalPotential(unit, (2, 3, 2))
        b = CrystalPotential(unit, (2, 3, 2))

        assert a == b
        np.testing.assert_array_equal(
            a.build(lazy=False).array, b.build(lazy=False).array
        )

    def test_roots_that_differ_by_a_small_relative_amount_are_different_crystals(self):
        """The root seed is compared exactly, not to a relative tolerance."""
        unit = _frozen_phonon_crystal(6, (1, 1, 1)).potential_unit.build(lazy=False)
        a = CrystalPotential(unit, (2, 3, 2))
        b = a.copy()
        b._root_seed = a._root_seed + a._root_seed // 10**9

        assert a != b
        assert a == a.copy()

    def test_a_rebuilt_crystal_takes_its_root_seed_without_drawing_one(
        self, monkeypatch
    ):
        """A lazy block rebuilds the crystal with the root seed of the original,
        and draws no fresh one."""
        unit = _frozen_phonon_crystal(6, (1, 1, 1)).potential_unit.build(lazy=False)
        crystal = CrystalPotential(unit, (2, 3, 2))
        args = crystal._partition_args(lazy=False)

        drawn = []
        seed_sequence = np.random.SeedSequence

        def counting_seed_sequence(entropy=None, **kwargs):
            if entropy is None:
                drawn.append(None)
            return seed_sequence(entropy, **kwargs)

        monkeypatch.setattr(np.random, "SeedSequence", counting_seed_sequence)
        rebuilt = crystal._from_partitioned_args()(*args).item()

        assert rebuilt == crystal
        assert drawn == []

    @pytest.mark.parametrize("seeds", [None, (5, 6)], ids=["unseeded", "seeded"])
    def test_a_crystal_whose_pickle_lacks_the_root_seed_equals_a_fresh_one(self, seeds):
        """A crystal whose pickle lacks the root seed and the shared pool."""
        import copy
        import pickle

        fresh = _frozen_phonon_crystal(4, (2, 2, 1), seeds=seeds)
        stripped = copy.copy(fresh)
        del stripped.__dict__["_root_seed"], stripped.__dict__["_shared_pool"]
        loaded = pickle.loads(pickle.dumps(stripped))

        eager = loaded.build(lazy=False).array

        assert loaded == fresh
        assert fresh == loaded
        np.testing.assert_array_equal(loaded.build(lazy=False).array, eager)
        np.testing.assert_array_equal(loaded.build(lazy=True).compute().array, eager)
        np.testing.assert_array_equal(fresh.build(lazy=False).array, eager)

    def test_the_member_seeds_of_a_crystal_are_distinct(self, monkeypatch):
        """A member seed that repeats an earlier one is replaced."""
        values = iter([7, 7, 8, 9])

        class Child:
            def generate_state(self, n):
                return np.array([next(values)], dtype=np.uint32)

        class SeedSequence:
            def __init__(self, entropy=None):
                pass

            def spawn(self, n):
                return [Child() for _ in range(n)]

        monkeypatch.setattr(np.random, "SeedSequence", SeedSequence)
        crystal = _frozen_phonon_crystal(6, (2, 3, 2), num_frozen_phonons=3)

        assert crystal.seeds == (7, 8, 9)

    @pytest.mark.filterwarnings("ignore:frozen-phonon pool .* is smaller:UserWarning")
    @pytest.mark.parametrize(
        "frozen_phonons, repetitions, kwargs",
        [
            (False, (1, 1, 2), dict(num_frozen_phonons=3)),
            # a pool of 2 for 4 lateral tiles, enlarged in the pool's task
            (True, (2, 2, 2), dict()),
            # one pool per member, each reseeded in its own task
            (True, (1, 1, 2), dict(seeds=(1, 2, 3))),
        ],
        ids=["unit", "enlarged pool", "reseeded pools"],
    )
    def test_the_lazy_graph_carries_the_unit_once(
        self, frozen_phonons, repetitions, kwargs
    ):
        """The task that builds the shared pool derives it from the graph's own
        copy of the unit, so the tasks of the graph, serialized one by one as a
        distributed scheduler receives them, hold the unit once."""
        import cloudpickle

        atoms = bulk("Si", cubic=True) * (6, 6, 2)
        if frozen_phonons:
            atoms = abtem.FrozenPhonons(atoms, 2, sigmas=0.1, seed=1)
        unit = Potential(atoms, gpts=(64, 64), slice_thickness=2.0)
        crystal = CrystalPotential(unit, repetitions, **kwargs)

        graph = dict(crystal._partition_args(lazy=True)[0].__dask_graph__())
        size = sum(len(cloudpickle.dumps(task)) for task in graph.values())

        assert size < 1.5 * len(cloudpickle.dumps(unit))

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize(
        "case, expected",
        [
            ("plain unit, seeds", 1),
            ("unseeded frozen phonons, scan blocks", 1),
            ("reseeded frozen phonons", 3),
        ],
    )
    def test_an_unbuilt_unit_is_built_once(self, monkeypatch, lazy, case, expected):
        """Members and scan blocks share one build of the unit, except members
        that reseed their own frozen-phonon pool."""
        from abtem.potentials.iam import _FieldBuilder

        builds = []
        build = _FieldBuilder.build

        def counting(self, *args, **kwargs):
            builds.append(type(self).__name__)
            return build(self, *args, **kwargs)

        atoms = bulk("Si", cubic=True)
        if case == "plain unit, seeds":
            unit = Potential(
                atoms, gpts=(16, 16), slice_thickness=atoms.cell[2, 2] / 4
            )
            crystal = CrystalPotential(unit, (2, 3, 2), seeds=(1, 2, 3))
        elif case == "reseeded frozen phonons":
            crystal = _frozen_phonon_crystal(6, (2, 3, 2), seeds=(1, 2, 3))
        else:
            crystal = _frozen_phonon_crystal(6, (2, 3, 2))

        monkeypatch.setattr(_FieldBuilder, "build", counting)
        with abtem.config.set({"potential.slice-chunk-size": 2}):
            if crystal.seeds is None:
                # four scan blocks of one crystal
                result = abtem.Probe(energy=100e3, semiangle_cutoff=20).scan(
                    crystal,
                    scan=abtem.GridScan((0, 0), (3, 3), gpts=(3, 2)),
                    detectors=abtem.AnnularDetector(10, 30),
                    lazy=lazy,
                    max_batch=2,
                )
            else:
                result = PlaneWave(energy=100e3).multislice(crystal, lazy=lazy)
            if lazy:
                result.compute(scheduler="synchronous")

        assert len(builds) == expected

    @staticmethod
    def _count_pool_builds(monkeypatch):
        """The frozen-phonon seeds of every pool built, one entry per build."""
        builds = []
        build = Potential.build

        def counting(self, *args, **kwargs):
            builds.append(tuple(int(seed) for seed in self.frozen_phonons.seed))
            return build(self, *args, **kwargs)

        monkeypatch.setattr(Potential, "build", counting)
        return builds

    @pytest.mark.parametrize("scheduler", ["threads", "synchronous"])
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(seeds=(1, 2, 3)),
            dict(num_frozen_phonons=3),
            dict(seeds=(1, 2, 3), ensemble_mean=False),
        ],
        ids=["seeds", "num_frozen_phonons", "no ensemble mean"],
    )
    def test_a_reseeded_pool_is_built_once_in_a_lazy_scan(
        self, monkeypatch, kwargs, scheduler
    ):
        """Every block of a lazy scan rebuilds its members, and each member that
        reseeds its pool built it again, so 3 members over 4 scan blocks built 12
        pools. Each member's pool is built once, however many blocks it runs in,
        and the result is that of the eager scan."""
        probe = abtem.Probe(energy=100e3, semiangle_cutoff=20)
        scan_kwargs = dict(
            scan=abtem.GridScan((0, 0), (3, 3), gpts=(3, 2)),
            detectors=abtem.AnnularDetector(10, 30),
        )
        # Slices in chunks of 2, fewer than the crystal has, so that the crystal is
        # not built for all blocks first (_prebuild_reused_potential) and each
        # block runs its own multislice through it.
        with abtem.config.set({"potential.slice-chunk-size": 2}):
            crystal = _frozen_phonon_crystal(6, (2, 3, 3), **kwargs)
            expected = probe.scan(crystal, lazy=False, max_batch=2, **scan_kwargs)

            builds = self._count_pool_builds(monkeypatch)
            num_blocks = {}
            pools = {}
            results = {}
            for max_batch in (6, 2):
                builds.clear()
                lazy = probe.scan(
                    crystal, lazy=True, max_batch=max_batch, **scan_kwargs
                )
                num_blocks[max_batch] = lazy.array.npartitions
                results[max_batch] = lazy.compute(
                    scheduler=scheduler, progress_bar=False
                )
                pools[max_batch] = list(builds)

        # 1 and 4 scan blocks, each for every member without the ensemble mean
        assert num_blocks[2] == 4 * num_blocks[6]
        # one pool per member, each built once
        assert len(pools[2]) == len(set(pools[2])) == 3
        assert sorted(pools[2]) == sorted(pools[6])
        for result in results.values():
            np.testing.assert_array_equal(result.array, expected.array)

    @pytest.mark.parametrize("scheduler", ["threads", "synchronous"])
    def test_a_reseeded_pool_is_built_once_in_a_lazy_transition_potential_scan(
        self, monkeypatch, scheduler
    ):
        """The core-loss multislice takes its slices from `generate_slices`."""
        from abtem.core.axes import OrdinalAxis
        from abtem.inelastic.core_loss import TransitionPotentialArray

        crystal = _frozen_phonon_crystal(6, (2, 2, 2), seeds=(1, 2, 3))
        rng = np.random.default_rng(0)
        array = (
            rng.standard_normal((2, 32, 32)) + 1j * rng.standard_normal((2, 32, 32))
        ).astype(np.complex64)
        transition_potentials = TransitionPotentialArray(
            Z=14,
            array=array,
            energy=100e3,
            extent=crystal.extent,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
            metadata={"Z": 14, "n": 1, "l": 0},
        )

        def scan(lazy):
            return abtem.Probe(
                energy=100e3, semiangle_cutoff=20
            ).transition_potential_scan(
                potential=crystal,
                transition_potentials=transition_potentials,
                scan=abtem.GridScan((0, 0), (3, 3), gpts=(2, 2)),
                detectors=abtem.PixelatedDetector(max_angle=40),
                lazy=lazy,
                max_batch=1,
            )

        # Slices in chunks of 2, as in the test above.
        with abtem.config.set({"potential.slice-chunk-size": 2}):
            expected = scan(lazy=False)

            builds = self._count_pool_builds(monkeypatch)
            lazy = scan(lazy=True)
            assert lazy.array.npartitions == 4
            result = lazy.compute(scheduler=scheduler, progress_bar=False)

        assert len(builds) == len(set(builds)) == 3
        np.testing.assert_array_equal(result.array, expected.array)

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("seeds", [None, (1, 2)])
    def test_prism_eels_extracts_the_sites_of_a_crystal(self, lazy, seeds):
        """Members keep their unit, so PRISM-EELS can take the sites from it."""
        from abtem.core.axes import OrdinalAxis
        from abtem.inelastic.core_loss import TransitionPotentialArray

        atoms = bulk("Si", cubic=True)
        unit = Potential(atoms, gpts=(32, 32), slice_thickness=atoms.cell[2, 2])
        crystal = CrystalPotential(unit, (2, 2, 3), seeds=seeds)
        rng = np.random.default_rng(0)
        array = rng.standard_normal((2, 64, 64)) + 1j * rng.standard_normal(
            (2, 64, 64)
        )

        def run(sites):
            transition_potentials = TransitionPotentialArray(
                Z=14,
                array=array.astype(np.complex64),
                energy=100e3,
                extent=crystal.extent,
                ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
                metadata={"Z": 14, "n": 1, "l": 0},
            )
            result = abtem.SMatrix(
                potential=crystal, energy=100e3, semiangle_cutoff=20, interpolation=1
            ).transition_potential_scan(
                transition_potentials,
                scan=abtem.GridScan((0, 0), (2, 2), gpts=(2, 3)),
                detectors=abtem.AnnularDetector(0, 40),
                sites=sites,
                lazy=lazy,
            )
            if lazy:
                result = result.compute(scheduler="synchronous")
            return np.asarray(result.array)

        expected = run(crystal.get_sliced_atoms())
        result = run(None)
        assert result.shape == expected.shape
        np.testing.assert_allclose(
            result, expected, rtol=0, atol=1e-6 * np.abs(expected).max()
        )


def _drawn_tiles(crystal):
    """The pool configuration of every lateral tile of every z-repetition of
    the built crystal, found by exact match against the member's pool; axes:
    member, z-repetition, repetitions[0], repetitions[1]."""
    repetitions = crystal.repetitions
    built = crystal.build(lazy=False).array
    if crystal.seeds is None:
        built, members = built[None], [None]
    else:
        members = [int(seed) for seed in crystal.seeds]

    drawn = np.empty((len(members), repetitions[2]) + repetitions[:2], dtype=int)
    for m, seed in enumerate(members):
        pool = crystal._pool_unit_for_member(seed).build(lazy=False).array
        n_sub, uy, ux = pool.shape[1:]
        for i in range(repetitions[2]):
            layer = built[m, n_sub * i : n_sub * (i + 1)].reshape(
                n_sub, repetitions[0], uy, repetitions[1], ux
            )
            for a in range(repetitions[0]):
                for b in range(repetitions[1]):
                    d = np.abs(pool - layer[None, :, a, :, b, :]).max(axis=(1, 2, 3))
                    assert d.min() == 0
                    drawn[m, i, a, b] = np.argmin(d)
    return drawn


class TestComplexExponential:
    """Verify the fused GPU complex_exponential kernel against the CPU reference."""

    @pytest.mark.parametrize("dtype,expected_cdtype", [
        (np.float32, np.complex64),
        (np.float64, np.complex128),
    ])
    def test_cpu_matches_reference(self, dtype, expected_cdtype):
        x = np.linspace(-np.pi, np.pi, 64, dtype=dtype)
        result = complex_exponential(x)
        expected = np.exp(1j * x).astype(expected_cdtype)
        assert result.dtype == expected_cdtype
        assert np.allclose(result, expected, atol=1e-6)

    @pytest.mark.parametrize("dtype,expected_cdtype", [
        (np.float32, np.complex64),
        (np.float64, np.complex128),
    ])
    @requires_gpu
    def test_gpu_matches_cpu(self, dtype, expected_cdtype):
        cp = pytest.importorskip("cupy")
        x_cpu = np.linspace(-np.pi, np.pi, 64, dtype=dtype)
        x_gpu = cp.asarray(x_cpu)

        result_cpu = complex_exponential(x_cpu)
        result_gpu = complex_exponential(x_gpu)

        assert isinstance(result_gpu, cp.ndarray)
        assert result_gpu.dtype == expected_cdtype
        assert np.allclose(cp.asnumpy(result_gpu), result_cpu, atol=1e-6)


def _fe_atoms_with_moments():
    atoms = bulk("Fe", cubic=True) * (1, 1, 2)
    moments = np.tile([[0.0, 0.0, 2.0], [0.5, 0.0, 1.0]], (len(atoms) // 2, 1))
    atoms.set_array("magnetic_moments", moments)
    return atoms


def _transmission_function():
    atoms = bulk("Si", cubic=True) * (1, 1, 2)
    potential = Potential(atoms, gpts=(32, 32), slice_thickness=2.0)
    return potential.build(lazy=False).transmission_function(100e3)


def _local_exit_planes(global_exit_planes, offset, length):
    return tuple(
        int(i) - offset for i in global_exit_planes if offset <= i < offset + length
    )


class TestPotentialSubclassWithoutAChunker:
    """A potential class written against v1.0.10 implements only that version's
    abstract members, and has no generate_chunked_slices."""

    @staticmethod
    def _minimal_potential(array, extent):
        class Minimal(BasePotential):
            def __init__(self):
                self._grid = Grid(extent=extent, gpts=array.shape[-2:])

            num_configurations = 1
            base_axes_metadata = []
            ensemble_axes_metadata = []
            ensemble_shape = ()
            device = "cpu"
            slice_thickness = (1.0,) * len(array)
            exit_planes = (len(array) - 1,)

            def generate_slices(self, first_slice=0, last_slice=None):
                for i in range(first_slice, last_slice or len(array)):
                    yield PotentialArray(array[i : i + 1], (1.0,), extent=extent)

            def build(self, first_slice=0, last_slice=None, chunks=1, lazy=None):
                return PotentialArray(array, self.slice_thickness, extent=extent)

            def _partition_args(self, chunks=1, lazy=True):
                return ()

            def _from_partitioned_args(self):
                def from_partitioned_args(*args, **kwargs):
                    members = np.empty((), dtype=object)
                    members[()] = self
                    return members

                return from_partitioned_args

        return Minimal()

    @pytest.mark.parametrize("lazy", [False, True])
    def test_multislice_runs_slice_by_slice(self, lazy):
        array = np.random.default_rng(0).random((3, 32, 32)).astype(np.float32)
        potential = self._minimal_potential(array, extent=8.0)
        waves = PlaneWave(energy=100e3, gpts=32, extent=8.0)

        result = waves.multislice(potential, lazy=lazy).compute()
        expected = waves.multislice(
            PotentialArray(array, (1.0,) * 3, extent=8.0), lazy=False
        )

        np.testing.assert_allclose(result.array, expected.array, rtol=0, atol=1e-6)

    @pytest.mark.parametrize("chunk_size", [1, 2, "auto"])
    @pytest.mark.parametrize("first_slice, last_slice", [(0, None), (1, 3)])
    def test_the_inherited_chunker_yields_each_slice_as_a_chunk(
        self, chunk_size, first_slice, last_slice
    ):
        array = np.random.default_rng(0).random((4, 32, 32)).astype(np.float32)
        potential = self._minimal_potential(array, extent=8.0)

        chunks = list(
            potential.generate_chunked_slices(first_slice, last_slice, chunk_size)
        )
        slices = list(potential.generate_slices(first_slice, last_slice))

        assert len(chunks) == len(slices) == (last_slice or 4) - first_slice
        for chunk, slic in zip(chunks, slices):
            assert type(chunk) is PotentialArray
            np.testing.assert_array_equal(chunk.array, slic.array)
            assert chunk.slice_thickness == slic.slice_thickness
            assert chunk.exit_planes == slic.exit_planes


class TestTransmissionFunctionSlices:
    """Slices and chunks of a transmission function carry its energy."""

    def test_slices_keep_the_energy(self):
        t = _transmission_function()
        slices = list(t.generate_slices())

        assert len(slices) == len(t) > 1
        assert [s.energy for s in slices] == [t.energy] * len(t)
        assert slices[0].transmission_function(100e3) is slices[0]

    def test_chunks_keep_the_energy(self):
        t = _transmission_function()
        chunks = list(t.generate_chunked_slices(chunk_size=1))

        assert len(chunks) == len(t) > 1
        assert [c.energy for c in chunks] == [t.energy] * len(t)
        assert chunks[0].transmission_function(100e3) is chunks[0]


class TestSlicesKeepTheMetadata:
    """Slices and chunks carry the metadata of their array, as indexing it does."""

    @pytest.mark.parametrize(
        "cls, shape",
        [
            (PotentialArray, (3, 8, 10)),
            (MagneticFieldArray, (3, 3, 8, 10)),
            (VectorPotentialArray, (3, 3, 8, 10)),
        ],
    )
    def test_arrays(self, cls, shape):
        array = cls(
            np.ones(shape, dtype=np.float32),
            slice_thickness=(1.0, 2.0, 1.5),
            extent=(4.0, 5.0),
            metadata={"note": "kept"},
        )

        slices = list(array.generate_slices())
        chunks = list(array.generate_chunked_slices(chunk_size=2))

        assert [len(c) for c in chunks] == [1, 2]
        assert array.metadata["note"] == "kept"
        for i, s in enumerate(slices):
            assert s.metadata == array[i : i + 1].metadata
        for chunk, (start, stop) in zip(chunks, [(0, 1), (1, 3)]):
            assert chunk.metadata == array[start:stop].metadata

    def test_transmission_functions(self):
        t = _transmission_function()

        for i, s in enumerate(t.generate_slices()):
            assert s.metadata == t.get_chunk(i, i + 1).metadata
        for chunk, (start, stop) in zip(
            t.generate_chunked_slices(chunk_size=2), [(0, 2), (2, 4), (4, 6)]
        ):
            assert chunk.metadata == t.get_chunk(start, stop).metadata


class TestAutoChunkSizeCountsTheComponentAxis:
    """chunk_size="auto" prices a slice at its own shape."""

    @staticmethod
    def _record_estimates(monkeypatch):
        shapes = []

        def estimate(gpts, device="cpu", dtype=None):
            shapes.append(tuple(gpts))
            return 2

        monkeypatch.setattr("abtem.core.chunks.estimate_potential_chunk_size", estimate)
        return shapes

    @pytest.mark.parametrize("builder", [MagneticField, VectorPotential])
    def test_field_builder_and_array(self, builder, monkeypatch):
        shapes = self._record_estimates(monkeypatch)
        field = builder(_fe_atoms_with_moments(), gpts=(16, 20), slice_thickness=1.5)
        built = field.build()

        assert field.base_shape[1:] == built.base_shape[1:] == (3, 16, 20)
        for chunked in (field, built):
            assert [len(c) for c in chunked.generate_chunked_slices()] == [2, 2]
        assert shapes == [(3, 16, 20)] * 2

    def test_potentials_keep_their_shape(self, monkeypatch):
        shapes = self._record_estimates(monkeypatch)
        potential = Potential(
            bulk("Si", cubic=True), gpts=(16, 20), slice_thickness=1.5
        )
        built = potential.build()

        for chunked in (potential, built):
            list(chunked.generate_chunked_slices())
        assert shapes == [(16, 20)] * 2


class TestAutoChunkSizeCountsTheEnsembleAxis:
    """A builder's chunk holds every member of its ensemble, so chunk_size="auto"
    prices a slice at ensemble_shape + base_shape[1:]."""

    @staticmethod
    def _record_estimates(monkeypatch):
        shapes = []

        def estimate(slice_shape, device="cpu", dtype=None):
            shapes.append(tuple(slice_shape))
            return 2

        monkeypatch.setattr("abtem.core.chunks.estimate_potential_chunk_size", estimate)
        return shapes

    @pytest.mark.parametrize("cls", [Potential, MagneticField, VectorPotential])
    @pytest.mark.parametrize("num_configurations", [1, 4])
    def test_the_slice_shape_includes_the_ensemble_axis(
        self, cls, num_configurations, monkeypatch
    ):
        shapes = self._record_estimates(monkeypatch)
        phonons = FrozenPhonons(
            _fe_atoms_with_moments(), num_configurations, sigmas=0.05, seed=3
        )
        field = cls(phonons, gpts=(16, 20), slice_thickness=1.5)

        list(field.generate_chunked_slices())

        assert shapes == [(num_configurations,) + field.base_shape[1:]]

    def test_a_chunk_shrinks_with_the_number_of_configurations(self, monkeypatch):
        """The estimate of a 4-configuration builder is a quarter of the
        estimate of a single member, on a device with a fixed amount of free
        memory."""
        _install_fake_cupy(monkeypatch, free=150_000, total=1_000_000)
        monkeypatch.setattr(
            "abtem.core.chunks.estimate_potential_chunk_size",
            lambda slice_shape, device="cpu", dtype=None: estimate_potential_chunk_size(
                slice_shape, "gpu", dtype
            ),
        )

        def chunk_lengths(num_configurations):
            phonons = FrozenPhonons(
                bulk("Si", cubic=True), num_configurations, sigmas=0.05, seed=3
            )
            builder = Potential(phonons, gpts=(16, 20), slice_thickness=0.34)
            return [len(c) for c in builder.generate_chunked_slices()]

        assert max(chunk_lengths(1)) == 8
        assert max(chunk_lengths(4)) == 2

    def test_multislice_prices_a_slice_of_one_configuration(self, monkeypatch):
        """Multislice hands each configuration over with an ensemble axis of
        length 1, so its chunk size does not change."""
        shapes = self._record_estimates(monkeypatch)
        phonons = FrozenPhonons(bulk("Si", cubic=True), 3, sigmas=0.05, seed=3)
        potential = Potential(phonons, gpts=(16, 20), slice_thickness=2.0)

        PlaneWave(energy=100e3, gpts=(16, 20), extent=potential.extent).multislice(
            potential, lazy=False
        )

        assert len(shapes) == 3
        assert {int(np.prod(shape)) for shape in shapes} == {16 * 20}


class TestBuiltFieldWithAnEnsembleAxis:
    """Iterating or chunking a built array visits its first ensemble member only,
    and says so."""

    @staticmethod
    def _built(cls, num_configurations=2):
        phonons = FrozenPhonons(
            _fe_atoms_with_moments(), num_configurations, sigmas=0.05, seed=3
        )
        return cls(phonons, gpts=(16, 20), slice_thickness=1.5).build()

    @pytest.mark.parametrize("cls", [MagneticField, VectorPotential])
    @pytest.mark.parametrize("method", ["generate_slices", "generate_chunked_slices"])
    def test_warns(self, cls, method):
        built = self._built(cls)
        assert built.shape == (2, 4, 3, 16, 20)

        with pytest.warns(UserWarning, match="ensemble"):
            slices = list(getattr(built, method)())

        np.testing.assert_array_equal(
            np.concatenate([s.array for s in slices]), built.array[0]
        )

    def test_warns_for_a_potential_too(self):
        phonons = FrozenPhonons(bulk("Si", cubic=True), 2, sigmas=0.05, seed=3)
        built = Potential(phonons, gpts=(16, 20), slice_thickness=2.0).build()

        with pytest.warns(UserWarning, match="ensemble"):
            slices = list(built.generate_slices())

        np.testing.assert_array_equal(
            np.concatenate([s.array for s in slices]), built.array[0]
        )

    @pytest.mark.parametrize("cls", [MagneticField, VectorPotential])
    @pytest.mark.parametrize(
        "how", ["generate_slices", "generate_chunked_slices", "list", "for"]
    )
    def test_the_warning_points_at_the_caller(self, cls, how):
        built = self._built(cls)

        with pytest.warns(UserWarning, match="ensemble") as record:
            if how == "list":
                list(built)
            elif how == "for":
                for _ in built:
                    break
            else:
                list(getattr(built, how)())

        assert [w.filename for w in record] == [__file__]

    @pytest.mark.parametrize("cls", [MagneticField, VectorPotential])
    @pytest.mark.parametrize("method", ["generate_slices", "generate_chunked_slices"])
    @pytest.mark.parametrize("num_configurations", [None, 1])
    def test_a_single_member_does_not_warn(
        self, cls, method, num_configurations, recwarn
    ):
        atoms = _fe_atoms_with_moments()
        if num_configurations is not None:
            atoms = FrozenPhonons(atoms, num_configurations, sigmas=0.05, seed=3)
        built = cls(atoms, gpts=(16, 20), slice_thickness=1.5).build()
        assert built.ensemble_shape == (() if num_configurations is None else (1,))

        list(getattr(built, method)())
        assert not [w for w in recwarn if "ensemble" in str(w.message)]


@pytest.mark.parametrize("cls", [MagneticField, VectorPotential])
class TestAtomBasedFieldsWithFrozenPhonons:
    """The slices of a field with an ensemble axis, against slicing the full
    build's array."""

    @staticmethod
    def _field(cls):
        phonons = FrozenPhonons(
            _fe_atoms_with_moments(), 2, sigmas=0.05, seed=3
        )
        return cls(phonons, gpts=(16, 20), slice_thickness=1.5, exit_planes=2)

    def test_built_array_iterates_and_chunks_its_first_member(self, cls):
        full = self._field(cls).build()
        assert full.shape == (2, 4, 3, 16, 20)

        with pytest.warns(UserWarning, match="ensemble"):
            slices = list(full.generate_slices())
        assert len(slices) == 4
        for i, s in enumerate(slices):
            assert type(s) is type(full)
            np.testing.assert_array_equal(s.array, full.array[0, i : i + 1])
            assert s.slice_thickness == full.slice_thickness[i : i + 1]

        with pytest.warns(UserWarning, match="ensemble"):
            chunks = list(full.generate_chunked_slices(1, 4, chunk_size=2))
        assert [len(c) for c in chunks] == [1, 2]
        np.testing.assert_array_equal(
            np.concatenate([c.array for c in chunks]), full.array[0, 1:4]
        )
        assert full.exit_planes == (-1, 1, 3)
        assert [c.exit_planes for c in chunks] == [
            _local_exit_planes(full.exit_planes, 1, 1),
            _local_exit_planes(full.exit_planes, 2, 2),
        ]

    def test_builder_chunks_hold_every_member(self, cls):
        field = self._field(cls)
        full = field.build()

        chunks = list(field.generate_chunked_slices(chunk_size=3))

        assert [len(c) for c in chunks] == [2, 2]
        for chunk, (start, stop) in zip(chunks, [(0, 2), (2, 4)]):
            assert type(chunk) is type(full)
            assert chunk.shape == (2, stop - start, 3, 16, 20)
            np.testing.assert_array_equal(chunk.array, full.array[:, start:stop])
            assert chunk.ensemble_axes_metadata == full.ensemble_axes_metadata
            assert chunk.sampling == full.sampling
            assert chunk.slice_thickness == full.slice_thickness[start:stop]
        assert full.exit_planes == (-1, 1, 3)
        assert [c.exit_planes for c in chunks] == [
            _local_exit_planes(full.exit_planes, 0, 2),
            _local_exit_planes(full.exit_planes, 2, 2),
        ]
