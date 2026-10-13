# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging"]
# ///
"""Print pip constraints that pin each dependency in pyproject.toml to its lower bound.

Run as ``uv run --script ci/minimum_versions.py > constraints.txt``. Only the runtime
``[project].dependencies`` are read, and a requirement whose environment marker does
not hold for the running interpreter is skipped. A ``>=``, ``~=`` or ``==`` bound is
pinned; a requirement with no lower bound is left unpinned, and upper bounds and
exclusions such as ``!=2025.12.*`` are ignored. Any other bound (``>``, a wildcard
``==3.*``) has no single floor and stops the script. The constraints at the end are
for the test environment only.
"""

import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

# matplotlib before 3.10.7 uses pyparsing names that 3.3 deprecates, and the test
# suite turns warnings into errors.
TEST_ONLY = ["pyparsing<3.3"]

pyproject = Path(__file__).parent.parent / "pyproject.toml"
dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]

floors = {}
for requirement in map(Requirement, dependencies):
    if requirement.marker and not requirement.marker.evaluate():
        continue
    for specifier in requirement.specifier:
        if specifier.operator in ("!=", "<", "<="):
            continue
        if specifier.operator not in (">=", "~=", "==") or "*" in specifier.version:
            sys.exit(f"{requirement}: no single floor to pin for {specifier}")
        floors[requirement.name] = specifier.version
        print(f"{requirement.name}=={specifier.version}")

if "matplotlib" not in floors or Version(floors["matplotlib"]) >= Version("3.10.7"):
    sys.exit("The pyparsing<3.3 pin is only for a matplotlib floor below 3.10.7.")

print(*TEST_ONLY, sep="\n")
