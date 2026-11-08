"""Refresh test/gpu_test_costs.json from a GPU run with per-test durations.

On the GPU machine::

    ABTEM_TEST_GPU_WORKERS=4 python -m pytest -n auto --runslow --durations=0 -q test \\
        2>&1 | tee gpu_durations.log
    python test/refresh_gpu_test_costs.py gpu_durations.log

Only the nodeids that end in ``@gpuN`` are read: the CPU cases of a function
run outside the groups, and the multi-GPU tests in a group of their own
(``@gpu``). ABTEM_TEST_GPU_WORKERS must be more than 1, or every test is in that
single ``@gpu`` group. The script rewrites the table and prints the seconds of
each group as measured and as the new table would place them.

Temporary: to be removed before the branch is merged.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from gpu_groups import COSTS_PATH, assign_groups, function_id

# "12.34s call     test/test_x.py::test_y[gpu-a]@gpu1"
_DURATION_LINE = re.compile(
    r"^\s*(?P<seconds>[\d.]+)s\s+(?:setup|call|teardown)\s+"
    r"(?P<nodeid>\S+)@(?P<group>gpu\d+)\s*$"
)


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


def print_groups(title: str, seconds: dict[str, float]) -> None:
    mean = sum(seconds.values()) / len(seconds)
    print(title)
    for group in sorted(seconds):
        print(f"  {group}: {seconds[group]:7.1f} s ({seconds[group] / mean - 1:+.0%})")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    costs, measured = parse_durations(Path(argv[1]).read_text().splitlines())
    if not costs:
        print(f"no grouped (@gpuN) tests in {argv[1]}; was --durations=0 given?")
        return 1
    COSTS_PATH.write_text(
        json.dumps({f: round(s, 2) for f, s in sorted(costs.items())}, indent=1) + "\n"
    )
    print(f"wrote {len(costs)} functions, {sum(costs.values()):.1f} s, to {COSTS_PATH}")

    print_groups("measured:", measured)
    placed = assign_groups(costs, len(measured), costs)
    predicted = defaultdict(float)
    for function, group in placed.items():
        predicted[f"gpu{group}"] += costs[function]
    print_groups("placed by the new table:", dict(predicted))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
