import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from abtem.core.blockwise import concatenate_blocks
from abtem.core.distributions import ParameterSeries
from abtem.core.energy import energy2wavelength
from abtem.waves.transfer import Aberrations, polar_symbols, Aperture
from abtem.waves.transfer import point_resolution
from abtem.waves.waves import PlaneWave


def test_point_resolution():
    ctf = Aberrations(energy=200e3, Cs=1e-3 * 1e10, defocus=600)

    max_semiangle = 20
    n = 1e3
    sampling = max_semiangle / 1000. / n
    alpha = np.arange(0, max_semiangle / 1000., sampling)

    aberrations = ctf.evaluate(alpha, 0.)

    zero_crossings = np.where(np.diff(np.sign(aberrations.imag)))[0]
    numerical_point_resolution1 = 1 / (zero_crossings[1] * alpha[1] / energy2wavelength(ctf.energy))
    analytical_point_resolution = point_resolution(energy=200e3, Cs=1e-3 * 1e10)

    assert np.round(numerical_point_resolution1, 1) == np.round(analytical_point_resolution, 1)


@given(data=st.data(), chunks=st.integers(min_value=1, max_value=10))
def test_aberrations_ensemble(data, chunks):
    n = data.draw(st.integers(min_value=1, max_value=3))
    symbols = data.draw(st.permutations(polar_symbols).map(lambda x: x[:n]))

    parameters = {}
    for symbol in symbols:
        parameters[symbol] = ParameterSeries(np.linspace(0, 100, 10))

    aberrations = Aberrations(parameters=parameters)
    blocks = aberrations._ensemble_blockwise(chunks).compute()

    waves = PlaneWave(energy=200e3, extent=10, gpts=64)
    for i in np.ndindex(blocks.shape):
        blocks[i] = blocks[i].evaluate_for_waves(waves)

    assert np.allclose(concatenate_blocks(blocks), aberrations.evaluate_for_waves(waves))


@given(chunks=st.integers(min_value=1, max_value=10))
def test_aperture_ensemble(chunks):
    semiangle_cutoff = ParameterSeries(np.linspace(5, 50, 5))

    aperture = Aperture(semiangle_cutoff=semiangle_cutoff)

    blocks = aperture._ensemble_blockwise(chunks).compute()

    waves = PlaneWave(energy=200e3, extent=10, gpts=64)
    for i in np.ndindex(blocks.shape):
        blocks[i] = blocks[i].evaluate_for_waves(waves)

    assert np.allclose(concatenate_blocks(blocks), aperture.evaluate_for_waves(waves))


@given(data=st.data(), chunks=st.integers(min_value=1, max_value=10))
def test_compound_wave_transfer_function(data, chunks):
    n = data.draw(st.integers(min_value=1, max_value=3))
    symbols = data.draw(st.permutations(polar_symbols).map(lambda x: x[:n]))

    parameters = {}
    for symbol in symbols:
        parameters[symbol] = ParameterSeries(np.linspace(0, 100, 10))

    aberrations = Aberrations(parameters=parameters)

    semiangle_cutoff = ParameterSeries(np.linspace(5, 50, 5))

    aperture = Aperture(semiangle_cutoff=semiangle_cutoff, taper=5)

    ctf = aperture * aberrations

    blocks = ctf._ensemble_blockwise(chunks).compute()

    waves = PlaneWave(energy=200e3, extent=10, gpts=64)
    for i in np.ndindex(blocks.shape):
        blocks[i] = blocks[i].evaluate_for_waves(waves)

    np.allclose(concatenate_blocks(blocks), ctf.evaluate_for_waves(waves))
