import numpy as np

from abtem import Probe, FresnelPropagator


def test_probe_tilt():
    tilted_probe = Probe(extent=5, gpts=128, energy=60e3, semiangle_cutoff=30, tilt=(200, 200))
    shifted_probe = Probe(extent=5, gpts=128, energy=60e3, semiangle_cutoff=30, tilt=(0, 0))

    tilted_wave = tilted_probe.build((0, 0))
    assert tilted_wave.tilt == tilted_probe.tilt

    propagator = FresnelPropagator()
    dz = 5
    tilted_wave = propagator.propagate(tilted_wave, dz)
    shifted_wave = shifted_probe.build([np.tan(tilted_wave.tilt[0] / 1e3) * dz, np.tan(tilted_wave.tilt[1] / 1e3) * dz])
    shifted_wave = propagator.propagate(shifted_wave, 5)
    assert np.allclose(shifted_wave.array, tilted_wave.array, atol=1e-5)


def test_probe_ctf():
    probe = Probe(extent=5, gpts=128, energy=60e3, semiangle_cutoff=30)
    defocused_probe = Probe(extent=5, gpts=128, energy=60e3, semiangle_cutoff=30, defocus=50)

    assert defocused_probe.ctf.defocus == 50

    assert not np.allclose(probe.build().array, defocused_probe.build().array)

    probe.ctf.defocus = 50
    assert np.allclose(probe.build().array, defocused_probe.build().array)
