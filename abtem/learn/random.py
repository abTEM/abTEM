import numpy as np
import scipy.spatial as spatial

from abtem.points import Points, fill_rectangle


def graphene_like(a=2.46, n=1, m=1, pointwise_properties=None):
    basis = [(0, 0), (2 / 3., 1 / 3.)]
    cell = [[a, 0], [-a / 2., a * 3 ** 0.5 / 2.]]
    positions = np.dot(np.array(basis), np.array(cell))

    points = Points(positions, cell=cell, pointwise_properties=pointwise_properties)
    points = points.repeat(n, m)
    return points


class Sequential:

    def __init__(self, modifiers):
        self.modifiers = modifiers

    def __call__(self, points):
        for modifier in self.modifiers:
            points = modifier(points)

        return points


class RandomRotation:

    def __init__(self, rotate_cell=True):
        self.rotate_cell = rotate_cell

    def __call__(self, points):
        return points.rotate(np.random.rand() * 360., rotate_cell=True)


class FillRectangle:

    def __init__(self, extent, margin):
        self.extent = extent
        self.margin = margin

    def __call__(self, points):
        return fill_rectangle(points, self.extent, margin=self.margin)


class RandomDelete:

    def __init__(self, fraction):
        self.fraction = fraction

    def __call__(self, points):
        n = int(np.round((1 - self.fraction) * len(points)))
        return points[np.random.choice(len(points), n, replace=False)]


class RandomSetProperties:

    def __init__(self, fraction, name, new_value, default_value=0):
        self.fraction = fraction
        self.name = name
        self.new_value = new_value
        self.default_value = default_value

    def __call__(self, points):
        n = int(np.round(self.fraction * len(points)))
        idx = np.random.choice(len(points), n, replace=False)

        try:
            properties = points.get_properties(self.name)
            properties[idx] = self.new_value
        except KeyError:
            properties = np.full(fill_value=self.default_value, shape=len(points))
            properties[idx] = self.new_value
            points.set_properties(self.name, properties)

        return points


class RandomRotateNeighbors:

    def __init__(self, fraction, cutoff):
        self.R = np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2)],
                           [np.sin(np.pi / 2), np.cos(np.pi / 2)]])

        self.fraction = fraction
        self.cutoff = cutoff

    def __call__(self, points):
        point_tree = spatial.cKDTree(points.positions)
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
