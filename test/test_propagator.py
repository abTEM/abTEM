from abtem.waves import FresnelPropagator, PlaneWave


def test_propagator_cache():
    wave = PlaneWave(extent=10, gpts=100, energy=60e3).build()

    propagator = FresnelPropagator()
    propagator.propagate(wave, .5)
    propagator.propagate(wave, .5)

    assert propagator.cache._hits == 1
    assert propagator.cache._misses == 1
