import cupy as cp
import numpy as np
import scipy.signal
import scipy.spatial
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from abtem.noise import bandpass_noise
from abtem.points import Points, fill_rectangle, inside_cell
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster


def graphene_like(a=2.46, n=1, m=1, pointwise_attributes=None):
    basis = [(0, 0), (2 / 3., 1 / 3.)]
    cell = [[a, 0], [-a / 2., a * 3 ** 0.5 / 2.]]
    positions = np.dot(np.array(basis), np.array(cell))

    points = Points(positions, cell=cell, pointwise_attributes=pointwise_attributes)
    points = points.repeat(n, m)
    return points


def random_swap_labels(points, label, new_label, probability):
    idx = np.where(points.labels == label)[0]
    points.labels[idx[np.random.rand(len(idx)) < probability]] = new_label
    return points


class StructureModifier:

    def __init__(self):
        pass

    def apply(self, points):
        raise NotImplementedError()


class SequentialStructureModifier:

    def __init__(self, modifiers):
        self.modifiers = modifiers

    def __call__(self, points):
        for modifier in self.modifiers:
            points = modifier(points)

        return points


class Sometimes:

    def __init__(self, probability, modifier):
        self.probability = probability
        self.modifier = modifier

    def __call__(self, points):
        if np.random.rand() < self.probability:
            return self.modifier(points)
        return points


class RandomRotation:

    def __init__(self, rotate_cell=True):
        self.rotate_cell = rotate_cell

    def __call__(self, points):
        return points.rotate(np.random.rand() * 360., rotate_cell=True)


class FillRectangle:

    def __init__(self, extent, margin, origin=None, periodic=True):
        self.extent = extent
        self.margin = margin
        self.origin = origin
        self.periodic = periodic

    def __call__(self, points):
        if self.periodic:
            return fill_rectangle(points, self.extent, margin=self.margin)
        else:
            points.cell = self.extent
            del points[inside_cell(points, 0.) == 0]
            return points


class RandomDelete:

    def __init__(self, fraction):
        self.fraction = fraction

    def __call__(self, points):
        n = int(np.round((1 - self.fraction) * len(points)))
        return points[np.random.choice(len(points), n, replace=False)]


class RandomSetAttributes:

    def __init__(self, fraction, values, default_value=0):
        self.fraction = fraction
        self.values = values
        self.default_value = default_value

    def __call__(self, points):
        n = int(np.round(self.fraction * len(points)))
        idx = np.random.choice(len(points), n, replace=False)

        for name, value in self.values.items():
            try:
                attributes = points.get_attributes(name)
                attributes[idx] = value
            except KeyError:
                attributes = np.full(fill_value=self.default_value, shape=len(points))
                attributes[idx] = value
                points.set_attributes(name, attributes)

        return points


class RandomRotateNeighbors:

    def __init__(self, fraction, cutoff):
        self.R = np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2)],
                           [np.sin(np.pi / 2), np.cos(np.pi / 2)]])

        self.fraction = fraction
        self.cutoff = cutoff

    def __call__(self, points):
        point_tree = scipy.spatial.cKDTree(points.positions)
        n = int(np.round(self.fraction * len(points)))
        first_indices = np.random.choice(len(points), n, replace=False)

        marked = np.zeros(len(points), dtype=np.bool)
        pairs = []
        for first_index in first_indices:
            marked[first_index] = True
            second_indices = point_tree.query_ball_point(points.positions[first_index], 1.5)

            if len(second_indices) < 2:
                continue

            np.random.shuffle(second_indices)

            for second_index in second_indices:
                if not marked[second_index]:
                    break

            marked[second_index] = True
            pairs += [(first_index, second_index)]

        for i, j in pairs:
            center = np.mean(points.positions[[i, j]], axis=0)
            points.positions[[i, j]] = np.dot(self.R, (points.positions[[i, j]] - center).T).T + center

        return points


class RandomStrain:

    def __init__(self, scale, amount):
        self.scale = scale
        self.amount = amount

    def __call__(self, points):
        xp = cp.get_array_module(points.positions)
        shape = xp.array((128, 128))
        noise = bandpass_noise(inner=0, outer=self.scale, shape=shape, xp=xp)
        indices = xp.floor((points.scaled_positions % 1.) * shape).astype(xp.int)
        for i in [0, 1]:
            points.positions[:, i] += self.amount * noise[indices[:, 0], indices[:, 1]]
        return points


# def smooth_random_loop():
#     points = np.random.rand(10, 2)
#     points = points[scipy.spatial.ConvexHull(points).vertices]
#     x = scipy.signal.resample(points[:, 0], 50)
#     y = scipy.signal.resample(points[:, 1], 50)
#     loop = np.array([x, y]).T
#     return (loop - loop.min(axis=0)) / loop.ptp(axis=0)


class RandomAddBlob:

    def __init__(self, fill_fraction, mean_density, periodicity, pointwise_attributes=None, adjacent_distance=None):
        self.fill_fraction = fill_fraction
        self.periodicity = periodicity
        self.mean_density = mean_density
        self.pointwise_attributes = pointwise_attributes
        self.adjacent_distance = adjacent_distance

    def __call__(self, points):
        shape = np.ceil(np.linalg.norm(points.cell, axis=0) / self.mean_density).astype(np.int)
        noise = bandpass_noise(0, self.periodicity, shape)
        noise = (noise - noise.min()) / noise.ptp()
        noise -= (1 - self.fill_fraction)
        noise[noise < 0] = 0

        positions = np.array(np.where(np.random.rand(*noise.shape) < noise)).T * self.mean_density

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points.positions)

        distances, indices = nbrs.radius_neighbors(positions, self.adjacent_distance)
        indices = np.unique(np.concatenate(indices))

        for name, value in self.pointwise_attributes.items():
            new_values = points.get_attributes(name)
            new_values[indices] = value
            points.set_attributes(name, new_values)

        contamination = Points(positions, pointwise_attributes=self.pointwise_attributes)
        points.extend(contamination)
        return points


class MergeClose:

    def __init__(self, distance):
        self.distance = distance

    def __call__(self, points):
        clusters = fcluster(linkage(pdist(points.positions), method='complete'), .2, criterion='distance')
        cluster_labels, counts = np.unique(clusters, return_counts=True)

        to_delete = []
        for cluster_label in cluster_labels[counts > 1]:
            cluster_members = np.where(clusters == cluster_label)[0][1:]
            to_delete.append(cluster_members)

        if len(to_delete) > 0:
            del points[np.concatenate(to_delete)]
        return points


class OptimizeGraphene:

    def __init__(self, iterations):
        self.iterations = iterations

    def __call__(self, points):
        positions = torch.tensor(points.positions)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        optimizer = GrapheneOptimizer(positions)
        optimizer = optimizer.to(device)
        optimizer.optimize(lr=.05, iterations=self.iterations)
        points.positions = optimizer.positions.detach().cpu().numpy()
        return points


class GrapheneOptimizer(nn.Module):

    def __init__(self, positions, constants=None):
        super().__init__()
        self.angles = torch.nn.Parameter(data=torch.zeros(len(positions)), requires_grad=True)
        self.positions = torch.nn.Parameter(data=positions, requires_grad=True)
        self.constants = {'bond_length': 1.3,
                          'bond_scale': .3,
                          'angular_scale': 1.,
                          'repulsive_strength': 5,
                          'repulsive_scale': .4,
                          'cutoff': 4}

    def optimize(self, lr, iterations, print_every=None):
        if print_every is not None:
            print_every = int(iterations * print_every)

        optimizer_1 = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer_2 = torch.optim.Adam([self.angles], lr=lr)

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

        mask = distances < self.constants['cutoff']
        distances = distances[mask]
        angles = (angles * 3 - self.angles[None])[mask]

        energies = -torch.exp(-(distances - self.constants['bond_length']) ** 2 / self.constants['bond_scale'])
        energies *= (torch.exp(self.constants['angular_scale'] * torch.cos(angles)) - torch.exp(
            self.constants['angular_scale'] * torch.cos(angles + np.pi)))
        energies += self.constants['repulsive_strength'] * torch.exp(-self.constants['repulsive_scale'] * distances)
        return torch.sum(energies)

# def select_blob(points, blob):
#     path = Path(blob)
#     return path.contains_points(points.positions)
#
#
# def random_select_blob(points, size):
#     blob = size * (make_blob() - .5)
#     position = np.random.rand() * points.cell[0] + np.random.rand() * points.cell[1]
#     return select_blob(points, position + blob)
#
#
# def select_close(points, mask, distance):
#     nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points.positions[mask])
#
#     distances, indices = nbrs.kneighbors(points.positions)
#     distances = distances.ravel()
#
#     return (distances < distance) * (mask == 0)
#
#
