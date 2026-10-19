import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("cupy") is not None,
    reason="the stub cupyx would shadow the real one",
)
def test_import_survives_cupyx_ndimage_failing_to_load(tmp_path):
    """cupyx can be importable while cupyx.scipy.ndimage is not (it eagerly
    loads the CUDA/ROCm libraries). That used to abort `import abtem` with an
    AssertionError, for CPU-only users too."""
    stub = tmp_path / "stub"
    (stub / "cupyx" / "scipy").mkdir(parents=True)
    (stub / "cupyx" / "__init__.py").write_text("")
    (stub / "cupyx" / "scipy" / "__init__.py").write_text("")
    (stub / "cupyx" / "scipy" / "ndimage.py").write_text(
        "raise ImportError('libcublas.so could not be loaded')\n"
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(stub)] + [p for p in [env.get("PYTHONPATH")] if p]
    )
    code = (
        "import abtem\n"
        "from abtem.core import backend\n"
        "assert backend.cupyx_ndimage is None\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
