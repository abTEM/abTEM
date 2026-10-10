# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging"]
# ///
"""Print pip constraints that pin each dependency in pyproject.toml to its lower bound.

Run as ``uv run --script ci/minimum_versions.py > constraints.txt``. A requirement
with no ``>=`` or ``==`` bound is left unpinned, and exclusions such as ``!=2025.12.*``
are ignored. The constraints at the end are for the test environment only.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

# matplotlib before 3.10.7 uses pyparsing names that 3.3 deprecates, and the test
# suite turns warnings into errors.
TEST_ONLY = ["pyparsing<3.3"]

pyproject = Path(__file__).parent.parent / "pyproject.toml"
dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]

for requirement in map(Requirement, dependencies):
    for specifier in requirement.specifier:
        if specifier.operator in (">=", "=="):
            print(f"{requirement.name}=={specifier.version}")

print(*TEST_ONLY, sep="\n")
