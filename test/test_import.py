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

import importlib.util
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


# find_spec, not a subprocess import: this runs at collection in every xdist
# worker and the controller.
requires_cupy = pytest.mark.skipif(
    importlib.util.find_spec("cupy") is None, reason="CuPy not installed"
)


@requires_cupy
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


@requires_cupy
def test_import_does_not_load_cublas():
    """import abtem must not load CuPy's BLAS bindings.

    On CUDA, loading them maps cuBLAS and costs about 150 MB of resident memory
    in every process, whatever device the work later runs on. They are loaded
    by modules such as cupyx.scipy.signal, which abTEM imports where it uses
    them.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, abtem; "
            "print(sorted(m for m in ('cupy_backends.cuda.libs.cublas', "
            "'cupyx.scipy.signal') if m in sys.modules))",
        ],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().strip() == "[]"


_MISSING = "_abtem_test_missing_dependency"

# What raises once a backend's package could not be imported, keyed by package.
_USE = {
    "cupy": "from abtem.core.backend import get_array_module; get_array_module('gpu')",
    "pyfftw": "abtem.config.set({'fft': 'fftw'}); "
    "from abtem.core.fft import fft2; fft2(np.ones((4, 4), np.complex64))",
    "mkl_fft": "abtem.config.set({'fft': 'mkl'}); "
    "from abtem.core.fft import fft2; fft2(np.ones((4, 4), np.complex64))",
    "torch": "from abtem.core.backend import get_array_module; "
    "get_array_module('mps')",
}

# Every backend is used in turn; each prints the error raised and its cause.
_PROBE = """
import warnings
warnings.simplefilter("always")
with warnings.catch_warnings(record=True) as caught:
    import numpy as np
    import abtem
print("WARNED", any("could not be imported" in str(w.message) for w in caught))
for package, use in {use!r}.items():
    try:
        exec(use)
    except RuntimeError as error:
        cause = error.__cause__
        print(package, type(error).__name__, type(cause).__name__,
              getattr(cause, "name", None))
"""

# Makes the packages not installed: a finder ahead of the others that answers "No
# module named ...". (sys.modules[package] = None would do, but scipy then finds
# None where it looks for torch.)
_HIDE = """
import sys
class _NotInstalled:
    def find_spec(self, name, path=None, target=None):
        if name in {packages!r}:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
sys.meta_path.insert(0, _NotInstalled())
"""


def _run_with_every_backend(tmp_path, init=None):
    """Run _PROBE in a fresh process where every backend package either is not
    installed (`init` is None) or is installed but its import runs `init`."""
    code = _PROBE.format(use=_USE)
    if init is None:
        code = _HIDE.format(packages=tuple(_USE)) + code
    else:
        for package in _USE:
            (tmp_path / package).mkdir()
            (tmp_path / package / "__init__.py").write_text(init)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path)] + [p for p in [env.get("PYTHONPATH")] if p]
    )
    env["ABTEM_DEVICE"] = "gpu"
    env["ABTEM_TORCH__DEVICE"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().split("\n")


def test_a_broken_optional_backend_reports_its_import_error(tmp_path):
    lines = _run_with_every_backend(tmp_path, init=f"import {_MISSING}\n")
    for package in _USE:
        assert f"{package} RuntimeError ModuleNotFoundError {_MISSING}" in lines
    # CuPy is the configured device
    assert "WARNED True" in lines


def test_a_plain_import_error_of_an_optional_backend_is_reported(tmp_path):
    # as raised by a missing shared library; a broken mkl_fft must not stop
    # `import abtem` either
    lines = _run_with_every_backend(
        tmp_path, init='raise ImportError("libx.so: cannot open shared object file")\n'
    )
    for package in _USE:
        assert f"{package} RuntimeError ImportError None" in lines
    assert "WARNED True" in lines


def test_a_missing_optional_backend_raises_a_runtime_error_when_used(tmp_path):
    lines = _run_with_every_backend(tmp_path)
    for package in _USE:
        assert f"{package} RuntimeError ModuleNotFoundError {package}" in lines
    # a package that is not installed is not a warning at import
    assert "WARNED False" in lines
