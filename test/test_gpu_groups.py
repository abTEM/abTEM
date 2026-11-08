import zlib

from gpu_groups import assign_groups, function_id


def test_function_id_drops_the_parameters_and_the_group():
    nodeid = "test/test_a.py::TestB::test_c[gpu-1-x]@gpu3"
    assert function_id(nodeid) == "test/test_a.py::TestB::test_c"


def test_assign_groups_places_the_most_expensive_first_in_the_emptiest_group():
    costs = {"a": 8, "b": 5, "c": 4, "d": 3, "e": 2}
    groups = assign_groups(costs, 2, costs)
    # a -> 0 (8); b -> 1 (5); c -> 1 (9); d -> 0 (11); e -> 1 (11)
    assert groups == {"a": 0, "b": 1, "c": 1, "d": 0, "e": 1}


def test_assign_groups_does_not_depend_on_the_order_of_the_functions():
    costs = {f"f{i}": float(i % 7) for i in range(30)}
    functions = list(costs) + ["unknown1", "unknown2"]
    assert assign_groups(functions, 4, costs) == assign_groups(
        reversed(functions), 4, costs
    )


def test_assign_groups_counts_an_unknown_function_as_the_median_cost():
    costs = {"a": 1.0, "b": 2.0, "c": 10.0}
    # the median is 2: "new" and "b" tie, and "b" sorts first
    groups = assign_groups(["a", "b", "c", "new"], 3, costs)
    assert groups == {"c": 0, "b": 1, "new": 2, "a": 1}


def test_assign_groups_without_costs_keeps_the_hash_of_the_function():
    functions = [f"test/t.py::test_{i}" for i in range(20)]
    assert assign_groups(functions, 4, {}) == {
        f: zlib.crc32(f.encode()) % 4 for f in functions
    }
