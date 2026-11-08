# Temporary, with refresh_gpu_test_costs.py: to be removed before merging.
from refresh_gpu_test_costs import parse_durations


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
