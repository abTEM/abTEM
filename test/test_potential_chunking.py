"""Tests verifying that potential chunking does not affect numerical results."""

import numpy as np
import pytest
from ase.build import bulk

from abtem import PlaneWave, Potential
from abtem.core.chunks import _nearest_power_of_two, estimate_potential_chunk_size
from abtem.core.complex import complex_exponential
from abtem.potentials.iam import CrystalPotential, PotentialArray


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
        while the unbuilt path exercises ``_FieldBuilderFromAtoms.generate_chunked_slices``
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

    def test_finite_builds_without_error(self):
        """Finite projection should build without error on CPU."""
        atoms = bulk("Si", cubic=True) * (2, 2, 2)
        pot = Potential(atoms, gpts=(64, 64), slice_thickness=2.0,
                        projection="finite")
        result = pot.build(lazy=False)
        # Should produce non-zero potential slices.
        assert len(result) > 0
        assert any(np.any(s.array != 0) for s in result)

    def test_finite_deterministic(self):
        """Two builds of the same finite potential must be identical."""
        atoms = bulk("Si", cubic=True) * (2, 2, 2)
        pot = Potential(atoms, gpts=(64, 64), slice_thickness=2.0,
                        projection="finite")
        arr1 = pot.build(lazy=False)
        arr2 = pot.build(lazy=False)
        for s1, s2 in zip(arr1, arr2):
            np.testing.assert_array_equal(s1.array, s2.array)

    def test_finite_multislice_chunked(self):
        """Finite-projection multislice must be identical across chunk sizes."""
        atoms = bulk("Si", cubic=True) * (2, 2, 4)
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
        atoms = bulk("Si", cubic=True)
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
        atoms_unit = bulk("Si", cubic=True)
        unit = Potential(atoms_unit, gpts=(32, 32), slice_thickness=2.0)
        crys = CrystalPotential(unit, repetitions=(2, 2, 3))

        atoms_full = bulk("Si", cubic=True) * (2, 2, 3)
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
    def test_gpu_matches_cpu(self, dtype, expected_cdtype):
        cp = pytest.importorskip("cupy")
        x_cpu = np.linspace(-np.pi, np.pi, 64, dtype=dtype)
        x_gpu = cp.asarray(x_cpu)

        result_cpu = complex_exponential(x_cpu)
        result_gpu = complex_exponential(x_gpu)

        assert isinstance(result_gpu, cp.ndarray)
        assert result_gpu.dtype == expected_cdtype
        assert np.allclose(cp.asnumpy(result_gpu), result_cpu, atol=1e-6)
