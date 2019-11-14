from abtem.parametrizations import lobato_parameters, lobato, convert_lobato
from ase.data import chemical_symbols


def radial_potential_function():
    potential_func = lobato
    convert_param_func = convert_lobato
    parameters = lobato_parameters
    d = dict(zip(chemical_symbols, list(range(len(chemical_symbols)))))
    return lambda Z, r: potential_func(r, *convert_param_func(parameters[d[Z]]))
