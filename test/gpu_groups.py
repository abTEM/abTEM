"""Balance the GPU tests over the ABTEM_TEST_GPU_WORKERS xdist groups.

Every test that touches the accelerator runs in an ``xdist_group`` (see
``pytest_collection_modifyitems`` in conftest.py), and the tests of one group
run one after another on one worker, so the busiest group sets the wall time
of a GPU run. The cases of one test function share a group, because they
share its fixtures.

``gpu_test_costs.json`` maps each test function (its nodeid without
parameters) to its measured GPU seconds: setup + call + teardown, summed over
its GPU cases, from a ``pytest --durations=0`` run on the GPU machine with
ABTEM_TEST_GPU_WORKERS above 1 (the grouped nodeids end in ``@gpuN``). The
functions are placed greedily, the most expensive first, each in the group with
the least work so far. A function missing from the table counts as the median
cost of the ones in it; with an empty table, every function goes by the hash of
its name. The placement depends only on the table and on the set of function
names, never on collection order, so every xdist worker computes the same
groups.
"""

from __future__ import annotations

import json
import statistics
import zlib
from pathlib import Path

COSTS_PATH = Path(__file__).with_name("gpu_test_costs.json")


def function_id(nodeid: str) -> str:
    """The test function of a nodeid: the nodeid without its parameters or the
    ``@group`` suffix that xdist adds."""
    return nodeid.split("@")[0].split("[")[0]


def load_costs(path: Path = COSTS_PATH) -> dict[str, float]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def assign_groups(functions, n_groups: int, costs: dict[str, float]) -> dict[str, int]:
    """Place each test function in one of ``n_groups`` groups, the most expensive
    first, each in the group with the least work so far (ties go to the lower
    group). Functions not in ``costs`` count as its median. With no costs at all,
    each function goes by the CRC32 hash of its name, as before the table."""
    if not costs:
        return {f: zlib.crc32(f.encode()) % n_groups for f in set(functions)}
    default = statistics.median(costs.values())
    loads = [0.0] * n_groups
    groups = {}
    for function in sorted(set(functions), key=lambda f: (-costs.get(f, default), f)):
        group = min(range(n_groups), key=lambda g: (loads[g], g))
        groups[function] = group
        loads[group] += costs.get(function, default)
    return groups
