from abtem.potentials.parametrizations.lobato import LobatoParametrization
from abtem.potentials.parametrizations.kirkland import KirklandParametrization
from abtem.potentials.parametrizations.peng import PengParametrization
from abtem.potentials.parametrizations.ewald import EwaldParametrization

named_parametrizations = {'ewald': EwaldParametrization,
                          'lobato': LobatoParametrization,
                          'peng': PengParametrization,
                          'kirkland': KirklandParametrization}


def validate_parametrization(parametrization):
    if isinstance(parametrization, str):
        parametrization = named_parametrizations[parametrization]()

    return parametrization