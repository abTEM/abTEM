import numbers

import numpy as np
import cupy as cp


def inside_cell(points, margin=0):
    margin = margin / np.linalg.norm(points.cell, axis=0)
    scaled_positions = points.scaled_positions
    return ((scaled_positions[:, 0] >= -margin[0]) & (scaled_positions[:, 1] >= -margin[1]) &
            (scaled_positions[:, 0] <= 1 + margin[0]) & (scaled_positions[:, 1] <= 1 + margin[1]))


def fill_rectangle(points, extent, origin=None, margin=0., eps=1e-12):
    xp = cp.get_array_module(points.positions)

    if origin is None:
        origin = xp.zeros(2)
    else:
        origin = xp.array(origin)

    extent = xp.array(extent)
    original_cell = points.cell.copy()

    P_inv = xp.linalg.inv(original_cell)

    origin_t = xp.dot(origin, P_inv)
    origin_t = origin_t % 1.0

    lower_corner = xp.dot(origin_t, original_cell)
    upper_corner = lower_corner + extent

    corners = xp.array([[-margin - eps, -margin - eps],
                        [upper_corner[0].item() + margin + eps, -margin - eps],
                        [upper_corner[0].item() + margin + eps, upper_corner[1].item() + margin + eps],
                        [-margin - eps, upper_corner[1].item() + margin + eps]])
    n0, m0 = 0, 0
    n1, m1 = 0, 0
    for corner in corners:
        new_n, new_m = xp.ceil(xp.dot(corner, P_inv)).astype(xp.int)
        n0 = max(n0, new_n)
        m0 = max(m0, new_m)
        new_n, new_m = xp.floor(xp.dot(corner, P_inv)).astype(xp.int)
        n1 = min(n1, new_n)
        m1 = min(m1, new_m)

    repeated = points.repeat((1 + n0 - n1).item(), (1 + m0 - m1).item())
    positions = repeated.positions.copy()

    positions = positions + original_cell[0] * n1 + original_cell[1] * m1

    inside = ((positions[:, 0] > lower_corner[0] - eps - margin) &
              (positions[:, 1] > lower_corner[1] - eps - margin) &
              (positions[:, 0] < upper_corner[0] + margin) &
              (positions[:, 1] < upper_corner[1] + margin))
    new_positions = positions[inside] - lower_corner

    pointwise_attributes = {}
    for name, array in repeated._arrays.items():
        if name != 'positions':
            pointwise_attributes[name] = array[inside]

    return Points(new_positions, cell=extent, pointwise_attributes=pointwise_attributes)


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


class Points:

    def __init__(self, positions=None, cell=None, pointwise_attributes=None, dimensions=2):
        self._arrays = {}

        if positions is None:
            positions = np.zeros((0, dimensions), dtype=np.float)

        self._positions = np.array(positions, dtype=np.float)

        if (len(self._positions.shape) != 2) | (self._positions.shape[1] != dimensions):
            raise RuntimeError()

        if pointwise_attributes is not None:
            for name, values in pointwise_attributes.items():
                self.create_attributes(name, values)

        self._cell = np.zeros((dimensions, dimensions), np.float)

        if cell is not None:
            self.cell = cell

    def __len__(self):
        return len(self.positions)

    def as_cupy(self):
        pointwise_attributes = {key: cp.asarray(values) for key, values in self._arrays.items()}
        return self.__class__(cp.array(self.positions), cp.asarray(self.cell),
                              pointwise_attributes=pointwise_attributes)

    def create_attributes(self, name, values, dtype=None):
        if isinstance(values, (int, float, complex, bool)):
            values = np.full(self.positions.shape[0], values, dtype=dtype)

        elif dtype is not None:
            values = np.array(values)

            if values.dtype != dtype:
                raise RuntimeError()

        self._arrays[name] = values

    def set_attributes(self, name, values):
        if name in self._arrays.keys():
            self._arrays[name][:] = values
        else:
            self.create_attributes(name, values)

    def get_attributes(self, name):
        return self._arrays[name]

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, positions):
        self.positions[:] = positions

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

    def get_filtered_positions(self, attribute_name, value):
        return self.positions[self.get_attributes(attribute_name) == value]

    def extend(self, other):
        if len(self._arrays) != len(other._arrays):
            raise RuntimeError()

        for name, values in self._arrays.items():
            if not name in other._arrays.keys():
                raise RuntimeError()

            self._arrays[name] = np.concatenate((values, other.get_attributes(name)))

        self._positions = np.concatenate((self.positions, other.positions))

    def rotate(self, angle, center=None, rotate_cell=False):
        if center is None:
            center = self.cell.sum(axis=1) / 2

        angle = angle / 180. * np.pi
        R = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.positions = np.dot(R, self.positions.T - np.array(center)[:, None]).T + center
        if rotate_cell:
            self.cell = np.dot(R, self.cell.T).T

        return self

    def repeat(self, n, m):
        N = len(self)

        n0, n1 = 0, n
        m0, m1 = 0, m
        new_positions = np.zeros((n * m * N, 2), dtype=cp.float)

        positions = self.positions.copy()
        new_positions[:N] = self.positions

        k = N
        for i in range(n0, n1):
            for j in range(m0, m1):
                if i + j != 0:
                    l = k + N
                    new_positions[k:l] = positions + np.dot(np.array((i, j)), self.cell)
                    k = l

        pointwise_attributes = {}
        for name, array in self._arrays.items():
            pointwise_attributes[name] = np.tile(array, (n * m,))

        cell = self.cell.copy() * np.array((n, m))

        return self.__class__(new_positions, cell=cell, pointwise_attributes=pointwise_attributes)

    def filter_by_attribute(self, name, value):
        return self[self.get_attributes(name) == value]

    def __add__(self, other):
        new = self.copy()
        new.extend(other)
        return new

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):
            if i < -len(self) or i >= len(self):
                raise IndexError('Index out of range.')

            return Point(i, self)

        pointwise_attributes = {key: values[i] for key, values in self._arrays.items()}

        return self.__class__(positions=self.positions[i].copy(), cell=self.cell.copy(),
                              pointwise_attributes=pointwise_attributes)

    def __delitem__(self, i):
        if isinstance(i, (list, tuple)) and len(i) > 0:
            i = np.array(i)

        mask = np.ones(len(self), bool)
        mask[i] = False

        for key in self._arrays.keys():
            self._arrays[key] = self._arrays[key][mask]

        self._positions = self._positions[mask]

    def copy(self):
        pointwise_attributes = {key: values.copy() for key, values in self._arrays.items()}
        return self.__class__(self.positions.copy(), self.cell.copy(), pointwise_attributes=pointwise_attributes)


class Point:

    def __init__(self, index, points):
        self.index = index
        self._points = points

    def get_attribute(self, name):
        return self._points.get_attributes(name)[self.index]

    @property
    def position(self):
        return self._points.positions[self.index]
