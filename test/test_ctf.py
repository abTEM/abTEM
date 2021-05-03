import numpy as np
import pytest

from abtem.base_classes import energy2wavelength
from abtem.transfer import CTF, polar_aliases, scherzer_defocus, polar_symbols


def test_scherzer():
    Cs = np.random.rand() * 1e5
    defocus = scherzer_defocus(Cs, 80e3)
    parameters = {'C10': -defocus, 'C30': Cs}
    alpha = (6 * energy2wavelength(80e3) / Cs) ** (1 / 4.)

    ctf = CTF(energy=80e3, parameters=parameters)
    assert np.isclose(ctf.evaluate_chi(alpha, 0.), 0, atol=1e-6)


def test_scherzer_kwargs():
    Cs = np.random.rand() * 1e5
    defocus = scherzer_defocus(Cs, 80e3)
    alpha = (6 * energy2wavelength(80e3) / Cs) ** (1 / 4.)

    ctf = CTF(energy=80e3, defocus=defocus, Cs=Cs)
    assert np.isclose(ctf.evaluate_chi(alpha, 0.), 0, atol=1e-6)


def test_aperture():
    ctf = CTF(semiangle_cutoff=20, rolloff=0., energy=80e3)
    assert ctf.evaluate_aperture(.021) == 0
    assert ctf.evaluate_aperture(.019) == 1


def test_ctf_base_aliases():
    random_parameters = dict(zip(polar_symbols, np.random.rand(len(polar_symbols))))
    parameter_aliases = {alias: random_parameters[key] for alias, key in polar_aliases.items()}

    CTF(parameters=random_parameters)
    CTF(**random_parameters)

    CTF(parameters=parameter_aliases)
    CTF(**parameter_aliases)


def test_ctf_raises():
    with pytest.raises(ValueError) as e:
        CTF(not_a_parameter=10)

    assert str(e.value) == 'not_a_parameter not a recognized parameter'

    ctf = CTF()
    with pytest.raises(RuntimeError) as e:
        ctf.evaluate(np.array([0]), np.array([0]))

    assert str(e.value) == 'Energy is not defined'

    ctf.energy = 200e3
    ctf.evaluate(np.array([0]), np.array([0]))


def test_ctf_event():
    ctf = CTF(energy=80e3)
    ctf.semiangle_cutoff = 1
    assert ctf.event.notify_count == 2
