import mock
import numpy as np
import pytest

from ..bases import energy2wavelength
from ..transfer import calculate_polar_aberrations, calculate_aperture, CTFBase
from ..transfer import calculate_polar_chi, calculate_symmetric_chi, polar_symbols
from ..transfer import calculate_temporal_envelope, CTF, polar_aliases

energy = 300e3
gpts = 64
sampling = .1


@pytest.fixture(scope='session')
def wavelength():
    return energy2wavelength(energy)


@pytest.fixture
def empty_parameters():
    return dict(zip(polar_symbols, [0.] * len(polar_symbols)))


@pytest.fixture
def symmetric_parameters(empty_parameters):
    empty_parameters.update({'C10': np.random.rand(),
                             'C30': np.random.rand(),
                             'C50': np.random.rand()})
    return empty_parameters


@pytest.fixture
def random_parameters(empty_parameters):
    empty_parameters.update(dict(zip(polar_symbols,
                                     np.random.rand(len(polar_symbols)))))
    return empty_parameters


@pytest.fixture(scope='session')
def alpha(wavelength):
    return np.fft.fftfreq(gpts, sampling)[:gpts // 2] * wavelength


@pytest.fixture(scope='session')
def symmetric_phi():
    return np.zeros(gpts // 2)


def test_polar_chi_symmetric(symmetric_parameters, alpha):
    assert np.allclose(calculate_polar_chi(alpha, 0., symmetric_parameters),
                       calculate_symmetric_chi(alpha, symmetric_parameters))


def test_polar_chi_scherzer(empty_parameters, wavelength):
    Cs = np.random.rand() * 1e5
    defocus = np.sqrt(1.5 * Cs * wavelength)
    empty_parameters.update({'C10': defocus, 'C30': Cs})
    alpha = (6 * wavelength / Cs) ** (1 / 4.)

    assert np.isclose(calculate_polar_aberrations(alpha, 0, wavelength, empty_parameters).imag, 0)


def test_calculate_aperture(alpha):
    rolloff = 0.
    cutoff = 2
    n = len(alpha)

    alpha = np.linspace(0, 4, n)
    aperture = calculate_aperture(alpha, cutoff, rolloff)
    assert np.all(aperture[n // 2:] == 0.)
    assert np.all(aperture[:n // 2] == 1.)

    rolloff = 0.5
    aperture = calculate_aperture(alpha, cutoff, rolloff)
    assert np.all(aperture[:n // 4] == 1.)
    assert np.all(aperture[n // 4:] != 1.)
    assert np.all(aperture[n // 4:n // 2] != 0.)


def test_calculate_temporal_envelope():
    # TODO : Make test
    pass


def test_ctf_base_raises():
    with pytest.raises(ValueError):
        CTFBase(not_a_parameter=1)


@pytest.fixture(scope='session')
def mocked_ctf_base(alpha, symmetric_phi):
    with mock.patch.object(CTFBase, 'get_alpha') as get_alpha:
        with mock.patch.object(CTFBase, 'get_phi') as get_phi:
            get_alpha.side_effect = lambda: alpha
            get_phi.side_effect = lambda: symmetric_phi
            yield CTFBase


def test_ctf_base_parameters(mocked_ctf_base, random_parameters):
    mocked_ctf_base(parameters=random_parameters)
    mocked_ctf_base(**random_parameters)


def test_ctf_base_aliases(mocked_ctf_base, random_parameters):
    parameter_aliases = {alias: random_parameters[key] for alias, key in polar_aliases.items()}

    mocked_ctf_base(parameters=parameter_aliases)
    mocked_ctf_base(**parameter_aliases)


def test_ctf_base_calculate_aberrations(mocked_ctf_base, symmetric_parameters, alpha, symmetric_phi, wavelength):
    ctf_base = mocked_ctf_base(energy=energy, **symmetric_parameters)

    assert np.all(ctf_base.get_aberrations() == calculate_polar_aberrations(alpha,
                                                                            symmetric_phi,
                                                                            wavelength,
                                                                            symmetric_parameters))


def test_ctf_base_calculate_aperture(mocked_ctf_base, alpha):
    ctf_base = mocked_ctf_base(energy=energy, cutoff=5., rolloff=0.1)
    assert np.allclose(ctf_base.get_aperture(), calculate_aperture(alpha, 5., 0.1))


def test_ctf_base_calculate_temporal_envelope(mocked_ctf_base, alpha, wavelength):
    ctf_base = mocked_ctf_base(energy=energy, focal_spread=30)
    assert np.allclose(ctf_base.get_temporal_envelope(), calculate_temporal_envelope(alpha, wavelength, 30.))


@pytest.fixture
def ctf():
    return CTF(gpts=gpts, sampling=sampling, energy=energy)


def test_ctf_get_alpha(ctf, alpha):
    ctf = CTF(gpts=gpts, sampling=sampling, energy=energy)
    assert np.allclose(ctf.get_alpha()[:gpts // 2, 0], alpha)


def test_ctf_get_phi():
    # TODO : Make test
    pass


def test_ctf_raises():
    with pytest.raises(ValueError) as e:
        CTF(not_a_parameter=10)

    assert str(e.value) == 'not_a_parameter not a recognized parameter'

    ctf = CTF()
    with pytest.raises(RuntimeError) as e:
        ctf.build()

    assert str(e.value) == 'extent is not defined'

    ctf.extent = 10
    ctf.gpts = 100
    with pytest.raises(RuntimeError) as e:
        ctf.build()

    assert str(e.value) == 'energy is not defined'

    ctf.energy = 200e3
    ctf.build()


def test_ctf_self_observe(ctf):
    assert ctf.observers == [ctf]


def test_ctf_cache(ctf):
    ctf.cutoff = 1
    ctf.focal_spread = 30

    aperture = ctf.get_aperture()
    assert set(ctf.cache.keys()) == {'get_alpha', 'get_aperture'}
    alpha = ctf.get_alpha()

    ctf.cutoff = 2
    assert set(ctf.cache.keys()) == {'get_alpha'}

    ctf.get_array()
    assert set(ctf.cache.keys()) == {'get_alpha', 'get_phi', 'get_aperture',
                                     'get_temporal_envelope', 'get_aberrations',
                                     'get_array'}

    assert alpha is ctf.get_alpha()
    assert aperture is not ctf.get_aperture()

    ctf.C10 = 10

    assert set(ctf.cache.keys()) == {'get_alpha', 'get_phi', 'get_aperture', 'get_temporal_envelope'}
