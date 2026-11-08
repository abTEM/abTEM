"""Print the xdist groups conftest.py gives the GPU tests, without running them.

    ABTEM_TEST_GPU_WORKERS=4 python test/check_gpu_groups.py [pytest args]

Collects the suite (``test`` unless other paths are given) and prints, for each
group, the number of test functions and their seconds in gpu_test_costs.json
(``?`` counts the functions missing from it). On a machine with an accelerator
it reads the ``xdist_group`` markers conftest.py adds. Without one there are no
GPU tests, so it stands the ``cpu`` cases of the device-parametrized functions
in for them and places those through conftest's ``_gpu_groups``. Either way it
fails if the cases of one function land in different groups.

Temporary: to be removed before the branch is merged.
"""

import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gpu_groups import function_id, load_costs  # noqa: E402


def _string_params(item):
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return []
    return [v for v in callspec.params.values() if isinstance(v, str)]


class _GroupReport:
    def pytest_collection_finish(self, session):
        groups = {}
        for item in session.items:
            marker = item.get_closest_marker("xdist_group")
            if marker is not None and str(marker.args[0]).startswith("gpu"):
                groups[item.nodeid] = marker.args[0]
        if not groups:
            print("\nno accelerator: placing the cpu cases of device tests instead")
            stand_ins = [
                item for item in session.items if "cpu" in _string_params(item)
            ]
            groups = sys.modules["conftest"]._gpu_groups(stand_ins)

        per_function = defaultdict(set)
        for nodeid, group in groups.items():
            per_function[function_id(nodeid)].add(group)
        split = {f: g for f, g in per_function.items() if len(g) > 1}
        if split:
            raise pytest.UsageError(f"functions split over groups: {split}")

        costs = load_costs()
        seconds, counts, unknown = (
            defaultdict(float),
            defaultdict(int),
            defaultdict(int),
        )
        for function, (group,) in per_function.items():
            counts[group] += 1
            if function in costs:
                seconds[group] += costs[function]
            else:
                unknown[group] += 1
        print()
        for group in sorted(counts):
            print(
                f"{group}: {counts[group]} functions, {seconds[group]:.1f} s"
                f" (+ {unknown[group]} ?)"
            )


if __name__ == "__main__":
    args = sys.argv[1:] or ["test"]
    sys.exit(
        pytest.main(
            ["--collect-only", "-q", "-n", "0", *args], plugins=[_GroupReport()]
        )
    )
