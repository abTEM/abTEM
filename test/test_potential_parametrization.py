import sys

import hypothesis.strategies as st
import numpy as np
import pytest
from ase.data import chemical_symbols
from hypothesis import given, settings
from utils import array_is_close

from abtem.parametrizations import KirklandParametrization, LobatoParametrization

try:
    from gpaw import GPAW

    from abtem.potentials.gpaw import GPAWParametrization
except ImportError:
    GPAW = None
    GPAWParametrization = None


@settings(deadline=None, max_examples=1)
@given(atomic_number=st.integers(1, 102))
@pytest.mark.parametrize(
    "func",
    [
        "potential",
        "scattering_factor",
        "projected_potential",
        "projected_scattering_factor",
    ],
)
def test_lobato_kirkland_match(atomic_number, func):
    r = np.linspace(0.01, 4.0, 10)
    kirkland = KirklandParametrization()
    lobato = LobatoParametrization()
    f1 = getattr(kirkland, func)(chemical_symbols[atomic_number])(r)
    f2 = getattr(lobato, func)(chemical_symbols[atomic_number])(r)
    assert array_is_close(f1, f2, rel_tol=0.05, check_above_rel=0.02)


# @settings(deadline=None, max_examples=2)
# @given(atomic_number=st.integers(1, 102))
# @pytest.mark.parametrize("func", ['potential'])
# @pytest.mark.skipif('gpaw' not in sys.modules, reason="requires gpaw")
# def test_lobato_gpaw_match(atomic_number, func):
#
#     r = np.linspace(0.01, 4., 10)
#     gpaw = GPAWParametrization()
#     lobato = LobatoParametrization()
#     f1 = getattr(gpaw, func)(chemical_symbols[atomic_number])(r)
#     f2 = getattr(lobato, func)(chemical_symbols[atomic_number])(r)
#     assert array_is_close(f1, f2, rel_tol=0.05, check_above_rel=.02)


@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
@pytest.mark.parametrize(
    "symbol, charge, num_electrons",
    [("Na", 0.0, 11), ("Na", 1.0, 10), ("Fe", 2.0, 24)],
)
def test_gpaw_parametrization_charge_integrates_to_number_of_electrons(
    symbol, charge, num_electrons
):
    n = GPAWParametrization().charge(symbol, charge)
    r = np.linspace(0, 20, 20001)
    assert np.isclose(
        np.trapezoid(4 * np.pi * r**2 * n(r), r), num_electrons, atol=1e-2
    )


@pytest.mark.parametrize("symbol", ["Sm", "Pu"])
@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
def test_gpaw_all_electron_atom_falls_back_to_unpolarized(symbol):
    """GPAW's spin-polarized radial solver fails for some open-shell f-element
    atoms; the density should then come from a non-spin-polarized calculation
    (with a warning) rather than raising an AssertionError.
    """
    gpaw = GPAWParametrization()
    with pytest.warns(UserWarning, match="non-spin-polarized"):
        ae = gpaw._get_all_electron_atom(symbol)

    assert ae.nspins == 1
    assert np.isclose(ae.rgd.integrate(ae.n_sg.sum(0)), ae.Z, atol=1e-3)
