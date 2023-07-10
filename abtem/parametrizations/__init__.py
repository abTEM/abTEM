"""Module to describe analytical potential parametrizations."""
from abtem.parametrizations.lobato import LobatoParametrization
from abtem.parametrizations.kirkland import KirklandParametrization
from abtem.parametrizations.peng import PengParametrization
from abtem.parametrizations.ewald import EwaldParametrization

named_parametrizations = {
    "ewald": EwaldParametrization,
    "lobato": LobatoParametrization,
    "peng": PengParametrization,
    "kirkland": KirklandParametrization,
}


def validate_parametrization(parametrization):
    if isinstance(parametrization, str):
        parametrization = named_parametrizations[parametrization]()

    return parametrization
