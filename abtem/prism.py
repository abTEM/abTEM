from ase import Atoms
import numpy as np


def import_prism_xyz(filename):
    with open(filename) as f:
        lines = f.readlines()
        cell = list(map(float, lines[1].split()))
        positions = []
        numbers = []
        for line in lines[2:-1]:
            line = line.split()
            numbers.append(int(line[0]))
            positions.append(list(map(float, line[1:4])))

    return Atoms(positions=positions, numbers=numbers, cell=cell)


def export_prism_xyz(filename, atoms):
    with open(filename, 'w') as f:
        f.write('test \n')
        f.write('    {} {} {} \n'.format(*np.diag(atoms.cell)))
        for number, position in zip(atoms.numbers, atoms.positions):
            f.write('{}  {}  {}  {}  {}  {} \n'.format(number, *position, 1, 0.00))
        f.write('-1 \n')
