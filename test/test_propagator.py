import numpy as np
import pytest

from abtem.waves import Propagator


@pytest.fixture
def propagator():
    return Propagator(extent=5, gpts=16, energy=60e3)


def test_propagator(propagator):
    assert np.allclose(propagator.build(-.5) * propagator.build(.5), 1.)


def test_propagator_cache(propagator):
    assert propagator.build(.5) is propagator.build(.5) is propagator.cache['build'][(0.5,)]
