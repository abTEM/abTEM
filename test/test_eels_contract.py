"""Numerical regression tests for the two EELS contraction associations."""

import numpy as np
import pytest
from utils import gpu

from abtem.core.backend import asnumpy, get_array_module


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("order", ["auto", "beam", "probe"])
@pytest.mark.parametrize(
    "shape",
    [(3, 7, 2, 2, 4, 5), (2, 3, 25, 3, 3, 4), (1, 1, 1, 1, 1, 1), (3, 4, 0, 2, 3, 5)],
)
def test_contract_eels_matches_reference(device, dtype, order, shape):
    from abtem.prism._eels_contract import contract_eels

    xp = get_array_module(device)
    n_s2, n_k, n_active, n_t, wy, wx = shape
    rng = np.random.default_rng(20260909)

    def random_array(shape):
        return xp.asarray(
            (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
        )

    # Slice a larger allocation to also exercise noncontiguous spatial crops.
    s2 = random_array((n_s2, wy, wx * 2))[..., ::2]
    s1 = random_array((n_k, wy, wx * 2))[..., ::2]
    h = random_array((n_t, wy, wx * 2))[..., ::2]
    coefficients = random_array((n_active, n_k))
    scale = 0.037
    s2_flat = s2.conj().reshape(n_s2, -1)
    hs1 = h[:, None] * s1[None]
    expected = xp.stack(
        [
            (scale * (s2_flat @ hs1[t].reshape(n_k, -1).T)) @ coefficients.T
            for t in range(n_t)
        ]
    )

    actual = contract_eels(s2, h, s1, coefficients, scale, order=order)

    assert actual.shape == (n_t, n_s2, n_active)
    assert actual.dtype == dtype
    assert get_array_module(actual) is xp
    tolerance = 3e-5 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(
        asnumpy(actual), asnumpy(expected), rtol=tolerance, atol=tolerance
    )


def test_contract_order_uses_probe_only_for_clear_arithmetic_advantage():
    from abtem.prism._eels_contract import _choose_contract_order

    assert _choose_contract_order(32, 256, 4, 1024, 3) == "probe"
    assert _choose_contract_order(32, 256, 4096, 1024, 3) == "beam"
    # Almost equal estimates retain the established beam contraction.
    assert _choose_contract_order(32, 32, 16, 1024, 1) == "beam"


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("order", ["auto", "beam", "probe"])
def test_contract_eels_zero_transitions(dtype, order):
    from abtem.prism._eels_contract import contract_eels

    actual = contract_eels(
        np.ones((3, 4, 5), dtype=dtype),
        np.empty((0, 4, 5), dtype=dtype),
        np.ones((7, 4, 5), dtype=dtype),
        np.ones((2, 7), dtype=dtype),
        0.037,
        order=order,
    )

    assert actual.shape == (0, 3, 2)
    assert actual.dtype == dtype
    assert get_array_module(actual) is np
    assert actual.size == 0


def test_contract_eels_rejects_invalid_order():
    from abtem.prism._eels_contract import contract_eels

    a = np.ones((1, 1, 1), dtype=np.complex64)
    with pytest.raises(ValueError, match="order"):
        contract_eels(a, a, a, a.reshape(1, 1), 1.0, order="invalid")
