"""`import abtem` survives a broken optional dependency."""

import os
import subprocess
import sys


def _import_abtem_with_packages(tmp_path, packages: dict[str, str]) -> str:
    """Import abtem in a fresh process in which each package of `packages` is
    installed but its import runs the given code."""
    for name, init in packages.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text(init)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path)] + [p for p in [env.get("PYTHONPATH")] if p]
    )
    code = "import abtem\nfrom abtem.core import fft\nprint(fft.mkl_fft)\n"
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().strip()


def test_a_broken_mkl_fft_does_not_stop_import_abtem(tmp_path):
    # as raised by a missing shared library: a plain ImportError, which the guard
    # did not catch, so `import abtem` failed whichever FFT library was configured
    out = _import_abtem_with_packages(
        tmp_path,
        {"mkl_fft": 'raise ImportError("libx.so: cannot open shared object file")\n'},
    )
    assert out == "None"
