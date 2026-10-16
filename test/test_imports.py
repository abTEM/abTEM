"""Every module under abtem/ imports.

A module whose import is broken is unreachable, and nothing notices when no other code
imports it. A module that needs an optional package that is not installed is skipped,
not failed.
"""

import importlib
import pkgutil

import pytest

import abtem
from abtem.core.backend import cp

MODULES = sorted(
    module.name for module in pkgutil.walk_packages(abtem.__path__, "abtem.")
)

# Modules that abTEM imports only when CuPy is available (``if cp is not None``) and
# that, without CuPy, fail at import with an AttributeError on ``cp`` rather than a
# ModuleNotFoundError.
NEEDS_CUPY = {"abtem.bloch.matrix_exponential"}


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    if name in NEEDS_CUPY and cp is None:
        pytest.skip("needs cupy, which is not installed")
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] != "abtem":
            pytest.skip(f"needs {error.name}, which is not installed")
        raise


def test_the_cupy_only_modules_exist():
    """A module that is renamed or removed is dropped from NEEDS_CUPY too."""
    assert NEEDS_CUPY <= set(MODULES)
