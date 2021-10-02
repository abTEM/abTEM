from ase.data import chemical_symbols

from abtem.utils import _set_path

azimuthal_number = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
azimuthal_letter = {value: key for key, value in azimuthal_number.items()}


def load_electronic_configurations():
    configurations = {}
    with open(_set_path('electron_configurations.txt')) as f:
        for i, line in enumerate(f):
            line = line.strip()
            prefix_start = line.find('[')
            prefix_end = line.find(']')
            if prefix_start > -1:
                line = configurations[line[prefix_start + 1:prefix_end]] + line[prefix_end + 1:]

            configurations[chemical_symbols[i + 1]] = line
    return configurations


def config_str_to_config_tuples(config_str):
    config_tuples = []
    for subshell_string in config_str.split(' '):
        config_tuples.append((int(subshell_string[0]), azimuthal_number[subshell_string[1]], int(subshell_string[2])))
    return config_tuples


def config_tuples_to_config_str(config_tuples):
    config_str = []
    for n, ell, occ in config_tuples:
        config_str.append(str(n) + azimuthal_letter[ell] + str(occ))
    return ' '.join(config_str)


def remove_electron_from_config_str(config_str, n, ell):
    config_tuples = []
    for shell in config_str_to_config_tuples(config_str):
        if shell[:2] == (n, ell):
            config_tuples.append(shell[:2] + (shell[2] - 1,))
        else:
            config_tuples.append(shell)
    return config_tuples_to_config_str(config_tuples)


def check_valid_quantum_number(Z, n, ell):
    symbol = chemical_symbols[Z]
    config_tuple = config_str_to_config_tuples(load_electronic_configurations()[symbol])

    if not any([shell[:2] == (n, ell) for shell in config_tuple]):
        raise RuntimeError(f'Quantum numbers (n, ell) = ({n}, {ell}) not valid for element {symbol}')
