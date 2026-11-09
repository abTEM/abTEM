"""Balance the GPU tests over the ABTEM_TEST_GPU_WORKERS xdist groups.

Every test that touches the accelerator runs in an ``xdist_group`` (see
``pytest_collection_modifyitems`` in conftest.py), and the tests of one group
run one after another on one worker, so the busiest group sets the wall time
of a GPU run. The cases of one test function share a group, because they
share its fixtures.

``gpu_test_costs.json`` maps the most expensive test functions (the nodeid
without parameters) to their measured GPU seconds. The functions are placed
greedily, the most expensive first, each in the group with the least work so
far; one that is not in the table counts as DEFAULT_COST. The placement depends
only on the table and on the set of collected function names, never on the
order of collection, so every xdist worker computes the same groups. It does
depend on which functions are collected: a run of a few files places them
differently than a run of the whole suite, so a failure that depends on which
tests share a worker may not recur with a subset.

Refreshing the table
--------------------
The table is a measurement of one machine (europa, an RTX 4090). The order of
the expensive functions transfers to other cards, their absolute seconds do
not, and it only needs refreshing when a function that dominates a group is
added or changes. On the GPU machine, run the suite with durations and with
more than one GPU worker, so that the nodeids of the grouped tests end in
``@gpuN``::

    ABTEM_TEST_GPU_WORKERS=4 python -m pytest -n auto --runslow --durations=0 \\
        -vv test 2>&1 | tee gpu_durations.log
    python test/gpu_groups.py gpu_durations.log

Without ``-vv`` pytest hides the durations under 0.005 s, which only shortens
the tail of the table. Only the ``@gpuN`` lines are read: the CPU cases of a
function run outside the groups, and the multi-GPU tests are in a group of their
own (``@gpu``). Per function, the seconds of setup, call and teardown of all its
GPU cases are summed, the ``@gpuN`` suffix is dropped, and the TABLE_SIZE most
expensive functions are written to ``gpu_test_costs.json``. The script prints
the seconds of each group as measured and as the new table places them, so a
second run of the command above shows the effect.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

COSTS_PATH = Path(__file__).with_name("gpu_test_costs.json")

# The seconds of a function that is not in the table: about the median of all
# the GPU functions. The 80 functions of the table hold 74 % of the seconds. On
# europa (4 workers) the groups then took 96 to 120 s, the slowest 7 % above the
# mean; a table of all 500 functions gave 114 to 122 s.
DEFAULT_COST = 0.25
TABLE_SIZE = 80

# "12.34s call     test/test_x.py::test_y[gpu-a]@gpu1"
_DURATION_LINE = re.compile(
    r"^\s*(?P<seconds>[\d.]+)s\s+(?:setup|call|teardown)\s+"
    r"(?P<nodeid>\S+)@(?P<group>gpu\d+)\s*$"
)


def function_id(nodeid: str) -> str:
    """The test function of a nodeid: the nodeid without its parameters."""
    return nodeid.split("[")[0]


def load_costs(path: Path = COSTS_PATH) -> dict[str, float]:
    return json.loads(path.read_text())


def assign_groups(functions, n_groups: int, costs: dict[str, float]) -> dict[str, int]:
    """Place each test function in one of ``n_groups`` groups, the most expensive
    first, each in the group with the least work so far (ties go to the lower
    group). Functions not in ``costs`` count as DEFAULT_COST."""
    loads = [0.0] * n_groups
    groups = {}
    for function in sorted(
        set(functions), key=lambda f: (-costs.get(f, DEFAULT_COST), f)
    ):
        group = min(range(n_groups), key=lambda g: (loads[g], g))
        groups[function] = group
        loads[group] += costs.get(function, DEFAULT_COST)
    return groups


def assign_gpu_groups(items, n_groups: int, costs: dict[str, float]) -> dict[str, str]:
    """The xdist group of each GPU test item, by nodeid. With one group every
    item is in ``gpu``. Otherwise the test functions are spread over the groups
    ``gpu0`` ... by their cost, so the cases of one function (and their
    fixtures) stay on one worker, and multi-GPU tests (the ``multigpu`` keyword)
    share the group ``gpu`` of their own."""
    items = list(items)
    if n_groups == 1:
        return {item.nodeid: "gpu" for item in items}
    single = [item for item in items if "multigpu" not in item.keywords]
    placed = assign_groups((function_id(i.nodeid) for i in single), n_groups, costs)
    groups = {item.nodeid: "gpu" for item in items}
    groups.update({i.nodeid: f"gpu{placed[function_id(i.nodeid)]}" for i in single})
    return groups


def parse_durations(lines) -> tuple[dict[str, float], dict[str, float]]:
    """The seconds of each test function and of each group, from the lines of a
    ``pytest --durations=0`` run, counting only the grouped (``@gpuN``) tests."""
    functions = defaultdict(float)
    groups = defaultdict(float)
    for line in lines:
        match = _DURATION_LINE.match(line)
        if match is None:
            continue
        seconds = float(match["seconds"])
        functions[function_id(match["nodeid"])] += seconds
        groups[match["group"]] += seconds
    return dict(functions), dict(groups)


def top_costs(functions: dict[str, float], size: int = TABLE_SIZE) -> dict[str, float]:
    """The ``size`` most expensive functions, sorted by name."""
    top = sorted(functions, key=lambda f: (-functions[f], f))[:size]
    return {f: round(functions[f], 2) for f in sorted(top)}


def _print_groups(title: str, seconds: dict[str, float]) -> None:
    mean = sum(seconds.values()) / len(seconds)
    print(title)
    for group in sorted(seconds):
        print(f"  {group}: {seconds[group]:7.1f} s ({seconds[group] / mean - 1:+.0%})")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    functions, measured = parse_durations(Path(argv[1]).read_text().splitlines())
    if not functions:
        print(f"no grouped (@gpuN) tests in {argv[1]}; was --durations=0 given?")
        return 1
    costs = top_costs(functions)
    COSTS_PATH.write_text(json.dumps(costs, indent=1) + "\n")
    print(f"wrote {len(costs)} of {len(functions)} functions to {COSTS_PATH}")

    _print_groups("measured:", measured)
    placed = defaultdict(float)
    for function, group in assign_groups(functions, len(measured), costs).items():
        placed[f"gpu{group}"] += functions[function]
    _print_groups("placed by the new table:", dict(placed))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
