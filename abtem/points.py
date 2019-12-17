import numpy as np
from scipy.spatial import Voronoi
from typing import Union
import numbers


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

    repeated = points.repeat(1 + n0 - n1, 1 + m0 - m1)

    positions = repeated.positions.copy()

    positions = positions + original_cell[0] * n1 + original_cell[1] * m1

    inside = ((positions[:, 0] > lower_corner[0] - eps - margin) &
              (positions[:, 1] > lower_corner[1] - eps - margin) &
              (positions[:, 0] < upper_corner[0] + margin) &
              (positions[:, 1] < upper_corner[1] + margin))
    new_positions = positions[inside] - lower_corner

    arrays = {}
    for name, array in repeated._arrays.items():
        if name != 'positions':
            arrays[name] = array[inside]

    masks = {}
    for name, mask in repeated._masks.items():
        masks[name] = mask[inside]

    return LabelledPoints(new_positions, cell=extent, arrays=arrays, masks=masks)


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


class LabelledPoints:

    def __init__(self, positions=None, cell=None, labels=None, dimensions=2, arrays=None, masks=None):
        self._arrays = {}
        self._masks = {}

        if positions is None:
            positions = np.zeros((0, dimensions), dtype=np.float)

        positions = np.array(positions, dtype=np.float)

        if (len(positions.shape) != dimensions) | (positions.shape[1] != dimensions):
            raise RuntimeError()

        self.create_array('positions', positions, dtype=np.float)

        if labels is None:
            labels = np.zeros(len(positions), dtype=np.int)
        else:
            labels = np.array(labels).astype(np.int)

        self.create_array('labels', labels, dtype=np.int)

        self._cell = np.zeros((dimensions, dimensions), np.float)

        if cell is not None:
            self.cell = cell

        if arrays is not None:
            for name, array in arrays.items():
                self.create_array(name, array)

        if masks is not None:
            for name, mask in masks.items():
                self.create_mask(name, mask)

    def __len__(self):
        return len(self.positions)

    def create_mask(self, name, mask):
        if isinstance(mask, (int, bool)):
            mask = np.full(self.positions.shape[0], mask, dtype=np.bool)

        self._masks[name] = mask

    def set_mask(self, name, mask):
        if name in self._arrays.keys():
            self._masks[name][:] = mask
        else:
            self.create_mask(name, mask)

    def get_mask(self, name):
        return self._masks[name].copy()

    def get_masked(self, name):
        return self[self._masks[name]]

    def create_array(self, name, values, dtype=None):
        if isinstance(values, (int, float, complex, bool)):
            values = np.full(self.positions.shape[0], values, dtype=dtype)

        elif dtype is not None:
            values = np.array(values)

            if values.dtype != dtype:
                raise RuntimeError()

        self._arrays[name] = values

    def set_array(self, name, values):
        if name in self._arrays.keys():
            self._arrays[name][:] = values
        else:
            self.create_array(name, values)

    def get_array(self, name):
        return self._arrays[name]

    @property
    def positions(self):
        return self.get_array('positions')

    @positions.setter
    def positions(self, positions):
        self.set_array('positions', positions)

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
        return self.get_array('labels')

    @labels.setter
    def labels(self, labels):
        self.set_array('labels', labels)

    def extend(self, other):
        if len(self._arrays) != len(other._arrays):
            raise RuntimeError()

        for name, values in self._arrays.items():
            if not name in other._arrays.keys():
                raise RuntimeError()

            self._arrays[name] = np.concatenate((values, other.get_array(name)))

        for name, mask in self._masks.items():
            if name in other._masks.keys():
                other_mask = other.get_mask(name)
            else:
                other_mask = np.zeros(len(other), dtype=bool)

            self._masks[name] = np.concatenate((mask, other_mask))

        for name, mask in other._masks.items():
            if name not in self._masks.keys():
                self._masks[name] = np.concatenate((np.zeros(len(self), dtype=bool), other.get_mask(name)))

    def repeat(self, n, m):
        N = len(self)

        n0, n1 = 0, n
        m0, m1 = 0, m
        new_positions = np.zeros((n * m * N, 2), dtype=np.float)

        positions = self.positions.copy()
        new_positions[:N] = self.positions

        k = N
        for i in range(n0, n1):
            for j in range(m0, m1):
                if i + j != 0:
                    l = k + N
                    new_positions[k:l] = positions + np.dot((i, j), self.cell)
                    k = l

        arrays = {}
        for name, array in self._arrays.items():
            if name != 'positions':
                arrays[name] = np.tile(array, (n * m,))

        masks = {}
        for name, array in self._masks.items():
            masks[name] = np.tile(array, (n * m,))

        cell = self.cell.copy() * (n, m)

        return LabelledPoints(new_positions, cell=cell, arrays=arrays, masks=masks)

    def __add__(self, other):
        new = self.copy()
        new.extend(other)
        return new

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):
            if i < -len(self) or i >= len(self):
                raise IndexError('Index out of range.')

            return Point(i, self)

        arrays = {key: values[i] for key, values in self._arrays.items() if not key == 'positions'}
        masks = {key: values[i] for key, values in self._masks.items()}

        return self.__class__(positions=self.positions[i].copy(), cell=self.cell.copy(), arrays=arrays, masks=masks)

    def __delitem__(self, i):
        if isinstance(i, list) and len(i) > 0:
            i = np.array(i)

        mask = np.ones(len(self), bool)
        mask[i] = False

        for key in self._arrays.keys():
            self._arrays[key] = self._arrays[key][mask]

        for key in self._masks.keys():
            self._masks[key] = self._masks[key][mask]

    def copy(self):
        arrays = {key: values.copy() for key, values in self._arrays.items() if not key == 'positions'}
        masks = {key: values.copy() for key, values in self._masks.items()}
        return self.__class__(self.positions.copy(), self.cell.copy(), self.labels.copy(), arrays=arrays, masks=masks)


class Point:

    def __init__(self, index, points):
        self.index = index
        self._points = points

    @property
    def label(self):
        return self._points.labels[self.index]

    @property
    def position(self):
        return self._points.positions[self.index]
