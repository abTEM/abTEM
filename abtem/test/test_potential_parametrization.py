from itertools import combinations

import numpy as np

from ..potentials import LobatoPotential, KirklandPotential, kappa


def test_similar():
    potentials = (LobatoPotential(), KirklandPotential())

    r = np.linspace(.1, 2, 5)

    for potential_a, potential_b in combinations(potentials, 2):
        assert np.all(np.abs(potential_a.get_potential(47)(r) -
                             potential_b.get_potential(47)(r)) / potential_b.get_potential(47)(r) < .1)


# def test_values():
#     potentials = {LobatoPotential(): [10.877785, 3.5969334, 1.1213292, 0.29497656, 0.05587856],
#                   KirklandPotential(): [10.966407, 3.7869546, 1.1616056, 0.2839873, 0.04958321]}
#
#     r = np.array([1., 1.316074, 1.7320508, 2.2795072, 3.])
#
#     for potential, values in potentials.items():
#         assert np.all(np.isclose(potential.get_potential(47)(r) / kappa, values))
