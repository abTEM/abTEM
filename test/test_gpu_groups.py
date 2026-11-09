from types import SimpleNamespace

from gpu_groups import (
    DEFAULT_COST,
    assign_gpu_groups,
    assign_groups,
    function_id,
    load_costs,
    parse_durations,
    top_costs,
)


def _item(nodeid, *keywords):
    return SimpleNamespace(nodeid=nodeid, keywords=set(keywords))


def test_function_id_drops_the_parameters():
    assert function_id("test/test_a.py::TestB::test_c[gpu-1-x]") == (
        "test/test_a.py::TestB::test_c"
    )
    assert function_id("test/test_a.py::test_d") == "test/test_a.py::test_d"


def test_the_shipped_table_holds_positive_costs_of_whole_functions():
    costs = load_costs()
    assert costs
    assert all(isinstance(s, float) and s > 0 for s in costs.values())
    assert all("::" in f and "[" not in f for f in costs)


def test_assign_groups_places_the_most_expensive_first_in_the_emptiest_group():
    costs = {"a": 8, "b": 5, "c": 4, "d": 3, "e": 2}
    groups = assign_groups(costs, 2, costs)
    # a -> 0 (8); b -> 1 (5); c -> 1 (9); d -> 0 (11); e -> 1 (11)
    assert groups == {"a": 0, "b": 1, "c": 1, "d": 0, "e": 1}


def test_assign_groups_does_not_depend_on_the_order_of_the_functions():
    costs = {f"f{i}": float(i % 7) + 1 for i in range(30)}
    functions = list(costs) + ["unknown1", "unknown2"]
    assert assign_groups(functions, 4, costs) == assign_groups(
        reversed(functions), 4, costs
    )


def test_assign_groups_counts_an_unknown_function_as_the_default_cost():
    costs = {"a": 100 * DEFAULT_COST, "b": 100 * DEFAULT_COST}
    groups = assign_groups(["a", "b", "x", "y", "z"], 2, costs)
    # the two known functions fill one group each; the unknown ones, each
    # DEFAULT_COST, alternate from the lower group
    assert groups == {"a": 0, "b": 1, "x": 0, "y": 1, "z": 0}


def test_assign_groups_with_an_empty_table_spreads_the_functions_by_count():
    functions = [f"f{i}" for i in range(8)]
    groups = assign_groups(functions, 4, {})
    assert sorted(groups.values()) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_the_cases_of_one_function_share_a_group():
    items = [
        _item(f"test/t.py::test_{name}[{case}]")
        for name in "abcdef"
        for case in ("cpu", "gpu-1", "gpu-2")
    ]
    groups = assign_gpu_groups(items, 3, {})
    for name in "abcdef":
        cases = {groups[i.nodeid] for i in items if f"test_{name}[" in i.nodeid}
        assert len(cases) == 1
    assert set(groups.values()) <= {"gpu0", "gpu1", "gpu2"}


def test_multi_gpu_tests_keep_a_group_of_their_own():
    items = [
        _item("test/t.py::test_a[gpu]"),
        _item("test/t.py::test_multi[1]", "multigpu"),
        _item("test/t.py::test_multi[2]", "multigpu"),
    ]
    groups = assign_gpu_groups(items, 4, {})
    assert groups["test/t.py::test_multi[1]"] == "gpu"
    assert groups["test/t.py::test_multi[2]"] == "gpu"
    assert groups["test/t.py::test_a[gpu]"] in {"gpu0", "gpu1", "gpu2", "gpu3"}


def test_one_gpu_worker_keeps_the_single_gpu_group():
    items = [
        _item("test/t.py::test_a[gpu]"),
        _item("test/t.py::test_b[gpu]"),
        _item("test/t.py::test_multi[1]", "multigpu"),
    ]
    assert set(assign_gpu_groups(items, 1, load_costs()).values()) == {"gpu"}


def test_parse_durations_sums_the_grouped_phases_of_each_function():
    lines = [
        "============ slowest durations ============",
        "3.00s call     test/t.py::test_a[gpu-x]@gpu1",
        "1.50s setup    test/t.py::test_a[gpu-y]@gpu1",
        "0.25s teardown test/t.py::test_a[gpu-y]@gpu1",
        "2.00s call     test/t.py::test_b[gpu]@gpu0",
        # CPU cases and multi-GPU tests are not in a numbered group
        "9.00s call     test/t.py::test_a[cpu-x]",
        "9.00s call     test/t.py::test_multi@gpu",
        "(3 durations < 0.005s hidden.)",
    ]
    functions, groups = parse_durations(lines)
    assert functions == {"test/t.py::test_a": 4.75, "test/t.py::test_b": 2.0}
    assert groups == {"gpu1": 4.75, "gpu0": 2.0}


def test_top_costs_keeps_the_most_expensive_functions_sorted_by_name():
    functions = {"c": 3.0, "a": 1.0, "d": 4.0, "b": 2.0}
    assert top_costs(functions, 3) == {"b": 2.0, "c": 3.0, "d": 4.0}
