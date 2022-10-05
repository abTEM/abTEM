from abtem.core.parametrizations.lobato import LobatoParametrization
from abtem.core.parametrizations.kirkland import KirklandParametrization
from abtem.core.parametrizations.peng import PengParametrization
from abtem.core.parametrizations.ewald import EwaldParametrization

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
