import numpy as np
from ase import Atoms

from abtem.structures import standardize_cell


def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.math.atan2(np.cross(v1, v2), np.dot(v1, v2))


def graphene_bilayer(n, a=2.46, d=3):
    m = n - 1
    gcd = np.gcd(n, m)
    n = int(n / gcd)
    m = int(m / gcd)

    positions = np.array([[0, 0], [0, a / np.sqrt(3)]])
    cell = np.array([[a, 0], [0.5 * a, 1.5 * a / np.sqrt(3)]])

    for i in range(2 * n - 3):
        if (i == 0) or (i % 2 == 1):
            positions = np.append(positions, (positions[i * 2] + cell[1]).reshape(1, -1), axis=0)
            positions = np.append(positions, (positions[i * 2 + 1] + cell[1]).reshape(1, -1), axis=0)
        if (i == 0) or (i % 2 == 0):
            positions = np.append(positions, (positions[i * 2] + cell[0] - cell[1]).reshape(1, -1), axis=0)
            positions = np.append(positions, (positions[i * 2 + 1] + cell[0] - cell[1]).reshape(1, -1), axis=0)

    for i in range(2 * (n - 1) * (2 * n - 1)):
        positions = np.append(positions, (positions[i] + cell[0]).reshape(1, -1), axis=0)

    last = 2 * (n - 1) * (2 * n - 1)
    side = 2 * (2 * n - 1)
    for i in range(n - 1):
        for j in range(last, last + side - 4):
            positions = np.append(positions, (positions[j] + cell[0]).reshape(1, -1), axis=0)
        last = last + side
        side = side - 4

    v1 = m * cell[0] + n * cell[1]
    v2 = n * cell[0] + m * cell[1]
    theta = angle_between(v2, v1)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    v2 = np.dot(R, v2)
    v3 = np.dot(R, -m * cell[0] + (n + m) * cell[1])

    super_cell = np.array([np.append(v2, 0), np.append(v3, 0), [0, 0, 2 * d]])

    rotated_positions = positions + np.array([-0.5 * a, 0.5 * a / np.sqrt(3)])
    rotated_positions = np.dot(R, rotated_positions.T).T

    positions = np.hstack((positions, np.ones(len(positions)).reshape(-1, 1)))
    rotated_positions = np.hstack((rotated_positions, d * np.ones(len(rotated_positions)).reshape(-1, 1)))

    atoms = Atoms('C' * len(positions), positions=positions, cell=super_cell, pbc=True)
    rotated_atoms = Atoms('C' * len(rotated_positions), positions=rotated_positions, cell=super_cell, pbc=True)

    atoms += rotated_atoms
    atoms.wrap()
    standardize_cell(atoms)
    return atoms
