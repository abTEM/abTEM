import numpy as np
from scipy.spatial import Voronoi


def repeat(points, n, m):
    N = len(points)

    n0, n1 = 0, n
    m0, m1 = 0, m
    new_positions = np.zeros((n * m * len(points), 2), dtype=np.float)

    positions = points.positions.copy()
    new_positions[:N] = points.positions

    k = N
    for i in range(n0, n1):
        for j in range(m0, m1):
            if i + j != 0:
                l = k + N
                new_positions[k:l] = positions + np.dot((i, j), points.cell)
                k = l

    labels = np.tile(points.labels, (n * m,))
    cell = points.cell.copy() * (n, m)

    return LabelledPoints(new_positions, cell=cell, labels=labels)


def inside_cell(points, margin=0):
    scaled_positions = points.scaled_positions
    return ((scaled_positions[:, 0] >= -margin) & (scaled_positions[:, 1] >= -margin) &
            (scaled_positions[:, 0] <= 1 + margin) & (scaled_positions[:, 1] <= 1 + margin))


def paint_outside(points, new_label):
    mask = inside_cell(points) == 0
    points.labels[mask] = new_label


def lloyds_relaxation(points, n, mask=None):
    def voronoi_centroids(positions):

        def area_centroid(positions):
            positions = np.vstack((positions, positions[0]))
            A = 0
            C = np.zeros(2)
            for i in range(0, len(positions) - 1):
                s = positions[i, 0] * positions[i + 1, 1] - positions[i + 1, 0] * positions[i, 1]
                A = A + s
                C = C + (positions[i, :] + positions[i + 1, :]) * s
            return (1 / (3. * A)) * C

        vor = Voronoi(positions)

        for i, region in enumerate(vor.point_region):
            if all(np.array(vor.regions[region]) > -1):
                vertices = vor.vertices[vor.regions[region]]
                positions[i] = area_centroid(vertices)

        return positions

    original_positions = points.positions.copy()
    for i in range(n):
        centroids = voronoi_centroids(points.positions)
        if mask is not None:
            centroids[mask] = original_positions[mask]

        points.positions = centroids

    return points


def fill_rectangle(points, extent, origin=None, margin=0., eps=1e-12):
    if origin is None:
        origin = np.zeros(2)
    else:
        origin = np.array(origin)

    extent = np.array(extent)
    original_cell = points.cell.copy()

    P_inv = np.linalg.inv(original_cell)

    origin_t = np.dot(origin, P_inv)
    origin_t = origin_t % 1.0

    lower_corner = np.dot(origin_t, original_cell)
    upper_corner = lower_corner + extent

    corners = np.array([[-margin - eps, -margin - eps],
                        [upper_corner[0] + margin + eps, -margin - eps],
                        [upper_corner[0] + margin + eps, upper_corner[1] + margin + eps],
                        [-margin - eps, upper_corner[1] + margin + eps]])

    n0, m0 = 0, 0
    n1, m1 = 0, 0
    for corner in corners:
        new_n, new_m = np.ceil(np.dot(corner, P_inv)).astype(np.int)
        n0 = max(n0, new_n)
        m0 = max(m0, new_m)
        new_n, new_m = np.floor(np.dot(corner, P_inv)).astype(np.int)
        n1 = min(n1, new_n)
        m1 = min(m1, new_m)

    repeated = repeat(points, 1 + n0 - n1, 1 + m0 - m1)

    positions = repeated.positions.copy()
    positions = positions + original_cell[0] * n1 + original_cell[1] * m1

    inside = ((positions[:, 0] > lower_corner[0] - eps - margin) &
              (positions[:, 1] > lower_corner[1] - eps - margin) &
              (positions[:, 0] < upper_corner[0] + margin) &
              (positions[:, 1] < upper_corner[1] + margin))
    new_positions = positions[inside] - lower_corner
    new_labels = repeated.labels[inside]

    return LabelledPoints(new_positions, cell=extent, labels=new_labels)


def wrap(points, center=(0.5, 0.5), eps=1e-7):
    if not hasattr(center, '__len__'):
        center = (center,) * 2

    shift = np.asarray(center) - 0.5 - eps

    fractional = np.linalg.solve(points.cell.T, np.asarray(points.positions).T).T - shift

    for i in range(2):
        fractional[:, i] %= 1.0
        fractional[:, i] += shift[i]

    points.positions = np.dot(fractional, points.cell)

    return points


def rotate(points, angle, center=None, rotate_cell=False):
    if center is None:
        center = points.cell.sum(axis=1) / 2

    angle = angle / 180. * np.pi

    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    points.positions = np.dot(R, points.positions.T - np.array(center)[:, None]).T + center

    if rotate_cell:
        points.cell = np.dot(R, points.cell.T).T

    return points


class LabelledPoints(object):

    def __init__(self, positions=None, cell=None, labels=None, dimensions=2):

        if positions is None:
            positions = np.zeros((0, dimensions), dtype=np.float)

        positions = np.array(positions, dtype=np.float)

        if (len(positions.shape) != dimensions) | (positions.shape[1] != dimensions):
            raise RuntimeError()

        self._positions = positions

        self._cell = np.zeros((dimensions, dimensions), np.float)

        if cell is not None:
            self.cell = cell

        if labels is None:
            labels = np.zeros(len(positions), dtype=np.int)

        labels = np.array(labels, dtype=np.int)

        if (len(labels.shape) != 1) | (labels.shape[0] != positions.shape[0]):
            raise RuntimeError()

        self._labels = labels

    def __len__(self):
        return len(self.positions)

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, positions):
        positions = np.array(positions, dtype=np.float)

        self._positions[:] = positions

    @property
    def scaled_positions(self):
        return np.dot(self.positions, np.linalg.inv(self.cell))

    @property
    def cell(self):
        return self._cell

    @cell.setter
    def cell(self, cell):
        cell = np.array(cell, dtype=np.float)

        if cell.shape == (2,):
            cell = np.diag(cell)

        self._cell[:] = cell

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        labels = np.array(labels, dtype=np.int)
        self._labels[:] = labels

    def extend(self, points):
        self._positions = np.vstack((self._positions, points.positions))
        self._labels = np.hstack((self._labels, points.labels))

    def __getitem__(self, i):
        return self.__class__(positions=self.positions[i].copy(), labels=self.labels[i].copy(), cell=self.cell.copy())

    def __delitem__(self, i):

        if isinstance(i, list) and len(i) > 0:
            i = np.array(i)

        mask = np.ones(len(self), bool)
        mask[i] = False

        self._positions = self._positions[mask]
        self._labels = self._labels[mask]

    def copy(self):
        return self.__class__(self.positions.copy(), self.cell.copy(), self.labels.copy())
