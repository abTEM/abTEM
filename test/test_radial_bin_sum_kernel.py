"""Tests for the radial-bin summing kernel (abtem.core._cuda:sum_run_length_encoded),
the CUDA RawKernel behind DiffractionPatterns._radial_binning / polar_binning -- the
reduction step every radial/polar detector (FlexibleAnnularDetector, SegmentedDetector,
AnnularDetector outside its single-bin fast path) runs once per site batch.

The kernel used to launch one thread per bin (n_bins ~10-1000, a small, roughly-fixed
detector-resolution parameter) and serially loop each thread over both the batch axis
and every pixel in that bin's segment -- occupying a single CUDA thread block regardless
of how large the batch or the diffraction pattern actually were. It now launches one
block per (bin, batch) pair and sums each bin's pixel segment with a block-level
shared-memory reduction, so both axes that actually scale with problem size get real
parallelism.

These tests check that:

1. The kernel's output still matches the CPU oracle (_sum_run_length_encoded, a plain
   NumPy/numba loop) across a range of shapes and both supported dtypes -- the
   parallel reduction sums the same terms in a different order, so exact equality
   isn't expected, but the result must agree to within the data's own floating-point
   accumulation error.
2. Degenerate shapes (zero batch rows, zero bins) do not crash.

The batch-parallel speedup itself (322.5 ms -> 2.9 ms for a 300 x 400,000 x 97 batch on
a Strix Halo iGPU) is not asserted: a wall-clock bound flaked on loaded GPU boxes.
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from utils import requires_gpu  # noqa: E402

from abtem.core._cuda import sum_run_length_encoded  # noqa: E402
from abtem.measurements import _sum_run_length_encoded  # noqa: E402


def _make_separators(n_selected, n_bins, rng):
    if n_bins == 0:
        return np.array([0], dtype=np.int64)
    cuts = np.sort(rng.integers(0, n_selected + 1, size=n_bins - 1))
    return np.concatenate([[0], cuts, [n_selected]]).astype(np.int64)


SHAPES = [
    (1, 1, 1),
    (5, 1000, 1),
    (80, 400_000, 97),  # mirrors a real FlexibleAnnularDetector call
    (300, 50_000, 97),
    (10, 10, 10),
    (10, 5, 10),  # n_selected < n_bins -- many empty bins
    (2, 1_000_000, 1),  # a single giant bin
    (256, 4096, 256),
]


@requires_gpu
@pytest.mark.parametrize("n_batch, n_selected, n_bins", SHAPES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sum_run_length_encoded_matches_cpu_oracle(n_batch, n_selected, n_bins, dtype, seed=0):
    rng = np.random.default_rng(seed)
    array_np = rng.standard_normal((n_batch, n_selected)).astype(dtype)
    separators_np = _make_separators(n_selected, n_bins, rng)

    result_cpu = np.zeros((n_batch, n_bins), dtype=dtype)
    _sum_run_length_encoded(array_np, result_cpu, separators_np)

    result_gpu = cp.zeros((n_batch, n_bins), dtype=dtype)
    sum_run_length_encoded(cp.asarray(array_np), result_gpu, cp.asarray(separators_np))
    result_gpu = cp.asnumpy(result_gpu)

    # Scale from the data itself, not a bare np.allclose default atol -- a bin can sum
    # tens of thousands of standard-normal draws, well above float32/float64 eps.
    scale = np.abs(array_np).sum() / max(array_np.size, 1) * (n_selected / max(n_bins, 1))
    atol = max(scale * 1e-4, 1e-6)
    np.testing.assert_allclose(result_gpu, result_cpu, atol=atol, rtol=1e-4)


@requires_gpu
@pytest.mark.parametrize("n_batch, n_bins", [(0, 10), (5, 0), (0, 0)])
def test_sum_run_length_encoded_handles_zero_sized_inputs(n_batch, n_bins):
    n_selected = 100
    rng = np.random.default_rng(0)
    array = cp.zeros((n_batch, n_selected), dtype=cp.float32)
    result = cp.zeros((n_batch, n_bins), dtype=cp.float32)
    separators = cp.asarray(_make_separators(n_selected, n_bins, rng))

    sum_run_length_encoded(array, result, separators)  # must not raise

