import os
import warnings
import zlib

import pytest
from hypothesis import HealthCheck, Phase, settings

from abtem import config
from abtem.core.backend import cp

config.set({"diagnostics.progress_bar": False})

settings.register_profile(
    "dev",
    max_examples=20,
    print_blob=True,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    phases=[Phase.generate],
)
settings.load_profile("dev")


def pytest_configure(config):
    # Ignore specific warnings globally
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )

    # Under pytest-xdist the tests on the GPU share one worker (see
    # pytest_collection_modifyitems), which xdist only honours with the
    # loadgroup scheduler. An explicitly chosen scheduler is left alone.
    if config.pluginmanager.hasplugin("xdist") and (
        getattr(config.option, "dist", None) == "load"
    ):
        config.option.dist = "loadgroup"


# How many xdist workers may run GPU tests at the same time. 1 (the default) is
# safe on any card; on one with memory to spare, more (ABTEM_TEST_GPU_WORKERS=4)
# shortens a run whose wall time is otherwise that of the GPU tests one after
# another.
_GPU_WORKERS = max(int(os.environ.get("ABTEM_TEST_GPU_WORKERS", "1")), 1)


def _gpu_group(item) -> str:
    """The xdist group of a GPU test: one of _GPU_WORKERS, chosen by its test
    function, so the cases of one function (and their fixtures) stay on one
    worker."""
    if _GPU_WORKERS == 1:
        return "gpu"
    function = item.nodeid.split("[")[0]
    return f"gpu{zlib.crc32(function.encode()) % _GPU_WORKERS}"


def pytest_collection_modifyitems(config, items):
    """Confine the tests on the GPU to a single pytest-xdist worker.

    ``-n auto`` sizes the pool from the CPU core count, with no idea that the
    "gpu" device parametrization means real device memory. Each worker that
    lands a GPU test brings up its own CUDA context, and enough of them at once
    exhaust the card; the failure is an error deep inside a kernel or cuFFT
    launch (``CUFFT_EXEC_FAILED``), not a clean skip. Routing every such test
    into one ``xdist_group`` runs at most one at a time whatever ``-n`` is,
    while the CPU tests still fan out over all workers.
    """
    if cp is None or not config.pluginmanager.hasplugin("xdist"):
        return

    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is not None and any(
            isinstance(value, str) and value == "gpu"
            for value in callspec.params.values()
        ):
            item.add_marker(pytest.mark.xdist_group(_gpu_group(item)))


# def pytest_addoption(parser):
#     parser.addoption(
#         "--runslow", action="store_true", default=False, help="run slow tests"
#     )


# def pytest_configure(config):
#     config.addinivalue_line("markers", "slow: mark test as slow to run")


# def pytest_collection_modifyitems(config, items):
#     if config.getoption("--runslow"):
#         return

#     skip_slow = pytest.mark.skip(reason="need --runslow option to run")
#     for item in items:
#         if "slow" in item.keywords:
#             item.add_marker(skip_slow)
