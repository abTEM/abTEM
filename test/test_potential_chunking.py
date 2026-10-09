"""Tests verifying that potential chunking does not affect numerical results."""

import sys
import types

import numpy as np
import pytest

from utils import requires_gpu, si_cubic_atoms
from ase.build import bulk

from abtem import FrozenPhonons, PlaneWave, Potential
from abtem.core import config as abtem_config
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
