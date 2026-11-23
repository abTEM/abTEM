"""Importing GPAW must not change the NumPy the other tests run against.

GPAW's debug mode patches NumPy process-wide; `conftest._restore_numpy` explains
when the mode is on and undoes it. These tests can fail only where the mode is
on: GPAW 25, or GPAW 26.7 with `GPAW_DEBUG` set.
"""

import numpy as np
import pytest

# imported at collection, as the GPAW test modules do
pytest.importorskip("gpaw")


def test_empty_object_arrays_hold_none_after_gpaw_is_imported():
    import matplotlib.pyplot as plt

    assert all(value is None for value in np.empty((2, 2), dtype=object).flat)
    assert all(value is None for value in np.empty_like(np.zeros(3, object)).flat)
    try:
        plt.subplots(1, 1, sharey=True)
    finally:
        plt.close("all")


def test_numpy_floating_point_errors_warn_after_gpaw_is_imported():
    assert np.geterr()["invalid"] == "warn"
    with pytest.warns(RuntimeWarning):
        np.log(np.array([-1.0]))
