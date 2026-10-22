import numpy as np
import pytest

from abtem.inelastic.plasmons import excitations_weights


@pytest.mark.parametrize("thickness, mean_free_path", [(10.0, 1000.0), (500.0, 800.0)])
def test_excitations_weights_are_poisson_probabilities(thickness, mean_free_path):
    # np.math was removed in NumPy 2, which made excitations_weights raise
    # AttributeError for every input.
    from scipy.stats import poisson

    n = np.arange(0, 20)
    weights = np.array([excitations_weights(i, thickness, mean_free_path) for i in n])

    assert np.allclose(weights, poisson.pmf(n, thickness / mean_free_path))
    assert excitations_weights(0, thickness, mean_free_path) == pytest.approx(
        np.exp(-thickness / mean_free_path)
    )
