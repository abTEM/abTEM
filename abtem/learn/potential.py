import numpy as np
import torch
import torch.nn as nn


class GrapheneOptimizer(nn.Module):

    def __init__(self, positions, constants):
        super().__init__()

        self.angles = torch.nn.Parameter(data=torch.zeros(len(positions)), requires_grad=True)
        self.positions = torch.nn.Parameter(data=positions, requires_grad=True)
        self.constants = {'bond_length': 1.35,
                          'bond_scale': .3,
                          'angular_scale': 1.,
                          'repulsive_strength': 7,
                          'repulsive_scale': .5,
                          'cutoff': 4}

    def optimize(self, lr, iterations, print_every=None):
        if print_every is not None:
            print_every = int(iterations * print_every)

        optimizer_1 = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer_2 = torch.optim.Adam([test.angles], lr=lr)

        for j in range(20):
            energy = self()
            energy.backward()
            optimizer_2.step()
            optimizer_2.zero_grad()

        for i in range(iterations):
            energy = self()
            energy.backward()
            optimizer_2.step()
            optimizer_2.zero_grad()

            energy = self()
            energy.backward()
            optimizer_1.step()
            optimizer_1.zero_grad()

            if print_every is not None:
                if i % print_every == 0:
                    print('{}: energy = {}'.format(i, energy))

    @staticmethod
    def pairwise_squared_distances(x, y):
        x_norm = (x ** 2).sum(1)[:, None]
        y_norm = (y ** 2).sum(1)[None, :]
        return torch.clamp(x_norm + y_norm - 2.0 * torch.mm(x, y.T), 0, np.inf)

    def forward(self):
        vectors = self.positions[None] - self.positions[:, None]
        angles = torch.atan2(vectors[:, :, 1], vectors[:, :, 0] + 1e-7)
        distances = torch.norm(vectors, dim=2)

        mask = distances < 4
        distances = distances[mask]
        angles = (angles * 3 - self.angles[None])[mask]

        energies = -torch.exp(-(distances - self.constants['bond_length']) ** 2 / self.constants['bond_scale'])
        energies *= (torch.exp(self.constants['angular_scale'] * torch.cos(angles)) - torch.exp(
            self.constants['angular_scale'] * torch.cos(angles + np.pi)))
        energies += self.constants['repulsive_strength'] * torch.exp(-self.constants['repulsive_scale'] * distances)
        return torch.sum(energies)
