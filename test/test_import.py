"""
Smoke tests for import-time safety.

The key invariant: ``import abtem`` must never trigger CUDA device
initialization, so it works on machines where CuPy is installed but no
GPU is present (e.g. a login node with the GPU conda env active, or a
CI runner).

Setting CUDA_VISIBLE_DEVICES="" hides all physical GPUs from the CUDA
runtime without uninstalling CuPy, reproducing that environment on any
machine.  If any module-level code calls a CUDA API (e.g. via
cp.RawModule(...).get_function(...)), the subprocess exits non-zero with
CUDARuntimeError: cudaErrorNoDevice.
"""

import os
import subprocess
import sys

import pytest


def _run_import(extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", "import abtem"],
        env=env,
        capture_output=True,
        timeout=60,
    )


def test_import_succeeds():
    """Baseline: import abtem works in the current environment."""
    result = _run_import()
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.skipif(
    subprocess.run(
        [sys.executable, "-c", "import cupy"],
        capture_output=True,
    ).returncode != 0,
    reason="CuPy not installed",
)
def test_import_with_cupy_but_no_gpu():
    """import abtem must not call any CUDA API at module level.

    CUDA_VISIBLE_DEVICES='' makes the CUDA runtime report no devices even
    when physical GPUs are present.  Any eager cp.RawModule.get_function()
    or similar call will raise CUDARuntimeError: cudaErrorNoDevice here.
    """
    result = _run_import(extra_env={"CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0, (
        "import abtem failed when CuPy is installed but no GPU is visible.\n"
        "This usually means a cp.RawModule / get_function() or similar CUDA "
        "call was added at module level.\n\n"
        + result.stderr.decode()
    )
