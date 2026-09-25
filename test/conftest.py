import warnings

import pytest
from hypothesis import HealthCheck, Phase, settings

from abtem import config
from utils import gpu as _gpu_param
from utils import requires_gpu as _requires_gpu
from utils import requires_multigpu as _requires_multigpu

config.set({"diagnostics.progress_bar": False})

# The literal device string that actually exercises an accelerator -- "gpu"
# with CuPy, "mps" on Apple silicon -- whatever `gpu` in test/utils.py
# currently resolves to.
_GPU_DEVICE = _gpu_param.values[0]

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
    phases=[Phase.generate],
)
settings.load_profile("dev")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
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
        "gpu: applied by requires_gpu/requires_multigpu; a presence-only "
        "marker for the GPU-worker-grouping hook below, not meant to be "
        "applied directly",
    )


# Metal is a single-precision backend -- torch refuses a float64 tensor on the
# MPS device outright -- so a test parametrized on both the 'mps' device and
# double precision is asking for something the hardware cannot do. Skipping is
# the honest outcome; letting it fail would bury real Metal regressions under
# noise that no amount of backend work can clear.
_DOUBLE_PRECISION_PARAMS = frozenset({"float64", "complex128"})


def _is_double_precision_on_metal(item) -> bool:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False

    # Only string parameters are of interest, and restricting to them also
    # keeps unhashable ones (arrays, Atoms) away from the set membership test.
    values = [value for value in callspec.params.values() if isinstance(value, str)]
    return "mps" in values and any(
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
    ``-n`` -- while CPU-only tests still fan out across every worker. This
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
    skip_metal_double = pytest.mark.skip(
        reason="Metal (MPS) is single precision; float64 cannot run on this device"
    )
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
        if is_gpu_param or is_gpu_marked or "multigpu" in item.keywords:
            item.add_marker(pytest.mark.xdist_group("gpu"))
