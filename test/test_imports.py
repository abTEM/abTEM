"""Every module under abtem/ imports.

A module whose import is broken is unreachable, and nothing notices when no other code
imports it. A module that needs an optional package that is not installed is skipped, not
failed.
"""

import importlib
import pkgutil

import pytest

import abtem

MODULES = sorted(module.name for module in pkgutil.walk_packages(abtem.__path__, "abtem."))


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] != "abtem":
            pytest.skip(f"needs {error.name}, which is not installed")
        raise
