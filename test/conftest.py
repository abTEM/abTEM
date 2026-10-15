import os
import sys
import warnings
import zlib

# A non-interactive backend, before anything imports matplotlib (abtem does): on
# macOS the default (TkAgg) runs figure cleanup off the main thread under
# pytest-xdist, which raised a tkinter error in one run and hung another. An
# MPLBACKEND that is already set wins.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402
from hypothesis import HealthCheck, Phase, settings

from abtem import config
from utils import _ACCELERATOR
from utils import gpu as _gpu_param
from utils import requires_gpu as _requires_gpu
from utils import requires_multigpu as _requires_multigpu

# FFTW_ESTIMATE rather than abTEM's default FFTW_MEASURE: measuring pays off
# for a user who transforms one shape many times, but the tests use most of
# their (often hypothesis-drawn) shapes only a few times, and measuring each
# new shape cost more than the transforms (8 of 16 s in one Bloch-wave test).
config.set(
    {"diagnostics.progress_bar": False, "fftw.planning_effort": "FFTW_ESTIMATE"}
)

# The literal device string that actually exercises an accelerator -- "gpu"
# with CuPy, "mps" on Apple silicon -- whatever `gpu` in test/utils.py
# currently resolves to.
_GPU_DEVICE = _gpu_param.values[0]

# Only tests on a real device need to share one xdist worker (see
# pytest_collection_modifyitems): with no accelerator every `gpu` case is a
# skip, and on torch's CPU device (ABTEM_TORCH__DEVICE=cpu, the CI torch job)
# there is no device memory to exhaust -- grouping would only run the whole
# `-m torch` selection on a single worker.
_SERIALIZE_GPU_TESTS = _ACCELERATOR == "gpu" or (
    _ACCELERATOR == "mps" and str(config.get("torch.device")).lower() == "mps"
)

# How many xdist workers may run GPU tests at the same time. 1 (the default)
# is safe on any card; on one with memory to spare, e.g. an A100, more
# (ABTEM_TEST_GPU_WORKERS=4) shortens a run whose wall time is otherwise that
# of the GPU tests one after another.
_GPU_WORKERS = max(int(os.environ.get("ABTEM_TEST_GPU_WORKERS", "1")), 1)


def _gpu_group(item) -> str:
    """The xdist group of a GPU test: one of _GPU_WORKERS, chosen by its test
    function, so the cases of one function (and their fixtures) stay on one
    worker. Multi-GPU tests share a group of their own."""
    if _GPU_WORKERS == 1 or "multigpu" in item.keywords:
        return "gpu"
    function = item.nodeid.split("[")[0]
    return f"gpu{zlib.crc32(function.encode()) % _GPU_WORKERS}"

# requires_gpu/requires_multigpu both apply a dedicated `gpu` marker
# alongside their skipif (see their definitions in test/utils.py) -- that
# marker's mere presence is what the check below prefers, since it
# survives a reword of the skipif's `reason=` text that a string match
# would not.
#
# The reason-string check stays as a fallback, imported rather than
# duplicated, for anything not going through those two helpers: a test
# with its own hand-rolled `skipif(..., reason=...)` that happens to reuse
# the same text carries no `gpu` marker at all, so without this fallback
# it would silently fall out of the group.
_GPU_SKIP_REASONS = {
    _requires_gpu.mark.kwargs.get("reason"),
    _requires_multigpu.mark.kwargs.get("reason"),
}

settings.register_profile(
    "dev",
    max_examples=10,
    print_blob=True,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much),
    # Not just Phase.generate: without `explicit` every @example is silently
    # skipped, without `reuse` a failure found once is never replayed from the
    # example database, and without `shrink` it is reported unminimised.
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)
settings.load_profile("dev")


@pytest.fixture(autouse=True)
def _close_figures():
    # Warnings are errors here, and matplotlib warns once more than 20 pyplot
    # figures are open. Without this, the figures that tests leave open pile up
    # over a serial run until a test that opens one more fails; CI runs with
    # `-n auto`, so no worker gets that far. Only a test that loaded pyplot can
    # have opened one, so the others do not import it.
    yield
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        plt.close("all")


@pytest.fixture
def float64_config(device):
    # Its 'mps' case never gets here: `float64_devices` in test/utils.py also
    # applies the `float64` marker, which pytest_collection_modifyitems below
    # skips on Metal.
    with config.set({"device": device, "precision": "float64", "fft": "numpy"}):
        yield


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def _cap_threads_per_worker(config):
    """Share the cores between pytest-xdist workers instead of letting each
    start one thread per core for BLAS/OpenMP, Numba, dask's threaded
    scheduler and abTEM's interpolation pool -- with `-n auto` that is
    cores**2 threads, and the context switching more than doubled the CPU
    time of the suite. Runs in the controller before the workers are started,
    so they inherit the environment before importing numba, dask or BLAS.
    Variables already set are left alone.

    BLAS/OpenMP get one thread per worker: the arrays in the tests are small,
    and GPAW warns (an error here) when OMP_NUM_THREADS > 1 in PW mode. Numba
    and dask share the cores, keeping at least 2 threads, so tests on the
    threaded scheduler still run concurrently.
    """
    workers = getattr(config.option, "numprocesses", None)
    if not isinstance(workers, int) or workers < 2:
        return
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
    threads = max((os.cpu_count() or 1) // workers, 2)
    for name in ("NUMBA_NUM_THREADS", "DASK_NUM_WORKERS"):
        os.environ.setdefault(name, str(threads))


def pytest_collection_finish(session):
    # Whether collecting the tests imported PyTorch, which can break CuPy's
    # kernel compilation in the same process; checked, before any test can
    # have imported it, by test_backend.py's
    # test_collecting_the_tests_does_not_import_torch.
    session.config._abtem_torch_imported_by_collection = (
        "torch" in sys.modules or "abtem.core._torch" in sys.modules
    )


def pytest_configure(config):
    _cap_threads_per_worker(config)
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    # Ignore specific warnings globally
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )
    config.addinivalue_line(
        "markers",
        "multigpu: requires >=2 GPUs and dask-cuda; skipped otherwise",
    )
    config.addinivalue_line(
        "markers",
        "gpu: runs on an accelerator (CuPy or Metal); applied by the `gpu` "
        "device parameter and by requires_gpu/requires_multigpu, not meant "
        "to be applied directly. Deselect with -m 'not gpu'.",
    )
    for name in ("torch", "mps", "metal"):
        config.addinivalue_line(
            "markers",
            f"{name}: the torch backend (device 'mps'): Apple Metal, or torch's CPU "
            "device with ABTEM_TORCH__DEVICE=cpu; see `gpu` in test/utils.py.",
        )
    config.addinivalue_line(
        "markers",
        "float64: the test runs in double precision, set in its body or by a "
        "fixture rather than by a parameter; its 'mps' case is skipped.",
    )


# Metal is a single-precision backend -- torch refuses a float64 tensor on the
# MPS device outright -- so a test parametrized on both the 'mps' device and
# double precision is asking for something the hardware cannot do. Skipping is
# the honest outcome; letting it fail would bury real Metal regressions under
# noise that no amount of backend work can clear. Double precision is seen in
# the test's parameters, or declared with the `float64` marker by a test that
# sets it in its body or a fixture, which no parameter shows.
_DOUBLE_PRECISION_PARAMS = frozenset({"float64", "complex128"})
_METAL_DOUBLE_SKIP_REASON = (
    "Metal (MPS) is single precision; float64 cannot run on this device"
)


def _is_double_precision_on_metal(item) -> bool:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False

    # Only string parameters are of interest, and restricting to them also
    # keeps unhashable ones (arrays, Atoms) away from the set membership test.
    values = [value for value in callspec.params.values() if isinstance(value, str)]
    if "mps" not in values:
        return False
    return item.get_closest_marker("float64") is not None or any(
        value in _DOUBLE_PRECISION_PARAMS for value in values
    )


# tryfirst=True is required, not stylistic: pytest-xdist's own
# pytest_collection_modifyitems (xdist/remote.py) reads each item's
# xdist_group marker to build the nodeid suffix its scheduler groups on.
# Without tryfirst, xdist's copy runs before this one adds the marker, so
# grouping silently never happens for any test.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Skip slow and impossible tests, and confine every GPU-touching test to
    a single pytest-xdist worker.

    ``-n auto`` sizes the worker pool from the CPU core count, with no idea
    that a "device" parametrization means real VRAM. Each worker that lands a
    GPU-parametrized test brings up its own CUDA/CuPy context, and enough of
    them running at once exhausts the card -- the actual failure mode is an
    OOM deep inside a kernel launch, not a clean skip or a clear message.

    Routing every such test into one ``xdist_group`` makes xdist schedule
    them onto the same worker, so at most one runs at a time regardless of
    ``-n`` -- while CPU-only tests still fan out across every worker
    (ABTEM_TEST_GPU_WORKERS spreads them over more groups, see _gpu_group). This
    only takes effect together with ``--dist=loadgroup`` (see pyproject.toml);
    without it, xdist's default scheduler ignores ``xdist_group`` entirely.
    A test is "GPU-touching" if any of its parametrized values -- direct or
    indirect, whatever the parameter's name -- is the resolved `gpu` device
    string, if it carries a `requires_gpu`/`requires_multigpu` skip (bare or
    mixed with an unrelated parametrize, however it's applied), or if it
    carries the `multigpu` marker (real multi-GPU tests want exclusive
    access even more, not less).
    """
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_metal_double = pytest.mark.skip(reason=_METAL_DOUBLE_SKIP_REASON)
    runslow = config.getoption("--runslow")

    for item in items:
        if not runslow and "slow" in item.keywords:
            item.add_marker(skip_slow)

        if _is_double_precision_on_metal(item):
            item.add_marker(skip_metal_double)

        callspec = getattr(item, "callspec", None)
        is_gpu_param = callspec is not None and any(
            isinstance(v, str) and v == _GPU_DEVICE
            for v in callspec.params.values()
        )
        # "gpu" is the dedicated marker requires_gpu/requires_multigpu both
        # apply (see test/utils.py); the reason-string match is a fallback
        # for anything not going through those two helpers.
        is_gpu_marked = "gpu" in item.keywords or any(
            mark.name == "skipif" and mark.kwargs.get("reason") in _GPU_SKIP_REASONS
            for mark in item.iter_markers()
        )
        if _SERIALIZE_GPU_TESTS and (
            is_gpu_param or is_gpu_marked or "multigpu" in item.keywords
        ):
            item.add_marker(pytest.mark.xdist_group(_gpu_group(item)))
