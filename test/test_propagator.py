from abtem.waves import FresnelPropagator, Probe
import numpy as np

def test_propagator_cache():
    wave = Probe(extent=10, gpts=100, energy=60e3, semiangle_cutoff=30).build()

    initial_wave_array = wave.array.copy()

    propagator = FresnelPropagator()
    propagator.propagate(wave, .5)
    propagator.propagate(wave, .5)

    propagator.propagate(wave, -1)

    assert np.allclose(wave.array.real[0], initial_wave_array.real[0], atol=5e-5)
    assert np.allclose(wave.array.imag[0], initial_wave_array.imag[0], atol=5e-5)

    assert propagator._cache._hits == 1
    assert propagator._cache._misses == 2
