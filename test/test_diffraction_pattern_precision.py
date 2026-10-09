"""Lazy diffraction patterns keep the precision of the waves, as eager ones do."""

import dask.array as da
import numpy as np
import pytest

from abtem.core.axes import FrozenPhononsAxis
from abtem.waves import Waves


def _waves(lazy):
    rng = np.random.default_rng(0)
    array = rng.normal(size=(3, 16, 20)) + 1j * rng.normal(size=(3, 16, 20))
    if lazy:
        array = da.from_array(array, chunks=(1, 16, 20))
    return Waves(
        array,
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[FrozenPhononsAxis(_ensemble_mean=False)],
    )


@pytest.mark.parametrize(
    "return_complex, dtype", [(False, np.float64), (True, np.complex128)]
)
def test_lazy_diffraction_patterns_keep_the_precision_of_the_waves(
    return_complex, dtype
):
    """complex128 waves give float64 patterns under the default float32 setting,
    lazily as well as eagerly. A lazy result declared as float32 would make later
    dask reductions accumulate in float32."""
    eager = _waves(lazy=False).diffraction_patterns(return_complex=return_complex)
    lazy = _waves(lazy=True).diffraction_patterns(return_complex=return_complex)

    assert lazy.array.dtype == eager.array.dtype == dtype
    np.testing.assert_array_equal(lazy.array.compute(), eager.array)


def test_a_lazy_sum_over_patterns_accumulates_in_their_precision():
    eager = _waves(lazy=False).diffraction_patterns().array.sum(axis=0)
    lazy = _waves(lazy=True).diffraction_patterns().array.sum(axis=0)

    assert lazy.dtype == np.float64
    np.testing.assert_allclose(lazy.compute(), eager, rtol=1e-13)
