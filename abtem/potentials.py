import csv
import os

import networkx as nx
import numpy as np
from ase import units
from numba import jit, prange
from scipy.optimize import brentq

from .bases import Grid, HasCache, cached_method, cached_method_with_args
from .graph import get_overlap_graph

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)

kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)


class PotentialParameterization(HasCache):

    def __init__(self, filename=None, parameters=None, tolerance=1e-3):

        self._tolerance = tolerance

        if parameters is None:
            self.load_parameters(filename)
        else:
            self._parameters = parameters

        HasCache.__init__(self)

    @property
    def parameters(self):
        return self._parameters

    @property
    def tolerance(self):
        return self._tolerance

    def load_parameters(self, filename):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        parameters = {}
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            keys = next(reader)
            for _, row in enumerate(reader):
                values = list(map(float, row))
                parameters[int(row[0])] = dict(zip(keys, values))
        self._parameters = parameters

    @cached_method_with_args
    def get_scattering_factor(self, atomic_number):
        raise RuntimeError()

    @cached_method_with_args
    def get_projected_potential(self, atomic_number):
        raise RuntimeError()

    @cached_method_with_args
    def get_potential(self, atomic_number):
        raise RuntimeError()

    @cached_method_with_args
    def get_soft_potential(self, atomic_number):
        raise RuntimeError()

    @cached_method_with_args
    def get_cutoff(self, atomic_number):
        return np.float32(brentq(lambda x: (self.get_potential(atomic_number)(x)) - self.tolerance, 1e-7, 1000))


class LobatoPotential(PotentialParameterization):

    def __init__(self, tolerance=1e-3):
        PotentialParameterization.__init__(self, filename='data/lobato.txt', tolerance=tolerance)

    @cached_method_with_args
    def get_scattering_factor(self, atomic_number):
        a = np.array([self.parameters[atomic_number][key_a] for key_a in ('a1', 'a2', 'a3', 'a4', 'a5')])
        b = np.array([self.parameters[atomic_number][key_b] for key_b in ('b1', 'b2', 'b3', 'b4', 'b5')])

        @jit(nopython=True)
        def func(k2):
            return ((a[0] * (2. + b[0] * k2) / (1. + b[0] * k2) ** 2) +
                    (a[1] * (2. + b[1] * k2) / (1. + b[1] * k2) ** 2) +
                    (a[2] * (2. + b[2] * k2) / (1. + b[2] * k2) ** 2) +
                    (a[3] * (2. + b[3] * k2) / (1. + b[3] * k2) ** 2) +
                    (a[4] * (2. + b[4] * k2) / (1. + b[4] * k2) ** 2))

        return func

    @cached_method_with_args
    def get_potential(self, atomic_number):
        a = np.array(
            [np.pi ** 2 * self.parameters[atomic_number][key_a] / self.parameters[atomic_number][key_b] ** (3 / 2.)
             for key_a, key_b in zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))])

        b = np.array(
            [2 * np.pi / np.sqrt(self.parameters[atomic_number][key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5')])

        @jit(nopython=True)
        def func(r):
            return (a[0] * (2. / (b[0] * r) + 1.) * np.exp(-b[0] * r) +
                    a[1] * (2. / (b[1] * r) + 1.) * np.exp(-b[1] * r) +
                    a[2] * (2. / (b[2] * r) + 1.) * np.exp(-b[2] * r) +
                    a[3] * (2. / (b[3] * r) + 1.) * np.exp(-b[3] * r) +
                    a[4] * (2. / (b[4] * r) + 1.) * np.exp(-b[4] * r))

        return func

    @cached_method_with_args
    def get_soft_potential(self, atomic_number):
        r_cut = self.get_cutoff(atomic_number)

        a = np.array(
            [np.pi ** 2 * self.parameters[atomic_number][key_a] / self.parameters[atomic_number][key_b] ** (3 / 2.)
             for key_a, key_b in zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))])

        b = np.array(
            [2 * np.pi / np.sqrt(self.parameters[atomic_number][key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5')])

        dvdr_cut = - (a[0] * (2. / (b[0] * r_cut ** 2) + 2. / r_cut + b[0]) * np.exp(-b[0] * r_cut) +
                      a[1] * (2. / (b[1] * r_cut ** 2) + 2. / r_cut + b[1]) * np.exp(-b[1] * r_cut) +
                      a[2] * (2. / (b[2] * r_cut ** 2) + 2. / r_cut + b[2]) * np.exp(-b[2] * r_cut) +
                      a[3] * (2. / (b[3] * r_cut ** 2) + 2. / r_cut + b[3]) * np.exp(-b[3] * r_cut) +
                      a[4] * (2. / (b[4] * r_cut ** 2) + 2. / r_cut + b[4]) * np.exp(-b[4] * r_cut))

        v_cut = (a[0] * (2. / (b[0] * r_cut) + 1.) * np.exp(-b[0] * r_cut) +
                 a[1] * (2. / (b[1] * r_cut) + 1.) * np.exp(-b[1] * r_cut) +
                 a[2] * (2. / (b[2] * r_cut) + 1.) * np.exp(-b[2] * r_cut) +
                 a[3] * (2. / (b[3] * r_cut) + 1.) * np.exp(-b[3] * r_cut) +
                 a[4] * (2. / (b[4] * r_cut) + 1.) * np.exp(-b[4] * r_cut))

        @jit(nopython=True)
        def func(r):
            v = (a[0] * (2. / (b[0] * r) + 1.) * np.exp(-b[0] * r) +
                 a[1] * (2. / (b[1] * r) + 1.) * np.exp(-b[1] * r) +
                 a[2] * (2. / (b[2] * r) + 1.) * np.exp(-b[2] * r) +
                 a[3] * (2. / (b[3] * r) + 1.) * np.exp(-b[3] * r) +
                 a[4] * (2. / (b[4] * r) + 1.) * np.exp(-b[4] * r))

            return v - v_cut - (r - r_cut) * dvdr_cut

        return func

    @cached_method_with_args
    def get_projected_potential(self, atomic_number):
        raise NotImplementedError


class KirklandPotential(PotentialParameterization):

    def __init__(self, tolerance=1e-3):
        PotentialParameterization.__init__(self, filename='data/kirkland.txt', tolerance=tolerance)

    @cached_method_with_args
    def get_potential(self, atomic_number):
        a = np.array([np.pi * self.parameters[atomic_number][key] for key in ('a1', 'a2', 'a3')])
        b = np.array([2. * np.pi * np.sqrt(self.parameters[atomic_number][key]) for key in ('b1', 'b2', 'b3')])
        c = np.array(
            [np.pi ** (3. / 2.) * self.parameters[atomic_number][key_c] / self.parameters[atomic_number][key_d] ** (
                    3. / 2.) for key_c, key_d in zip(('c1', 'c2', 'c3'), ('d1', 'd2', 'd3'))])
        d = np.array([np.pi ** 2 / self.parameters[atomic_number][key] for key in ('d1', 'd2', 'd3')])

        @jit(nopython=True)
        def func(r):
            return (a[0] * np.exp(-b[0] * r) / r + c[0] * np.exp(-d[0] * r ** 2.) +
                    a[1] * np.exp(-b[1] * r) / r + c[1] * np.exp(-d[1] * r ** 2.) +
                    a[2] * np.exp(-b[2] * r) / r + c[2] * np.exp(-d[2] * r ** 2.))

        return func

    @cached_method_with_args
    def get_soft_potential(self, atomic_number):
        r_cut = self.get_cutoff(atomic_number)

        a = np.array([np.pi * self.parameters[atomic_number][key] for key in ('a1', 'a2', 'a3')])
        b = np.array([2. * np.pi * np.sqrt(self.parameters[atomic_number][key]) for key in ('b1', 'b2', 'b3')])
        c = np.array(
            [np.pi ** (3. / 2.) * self.parameters[atomic_number][key_c] / self.parameters[atomic_number][key_d] ** (
                    3. / 2.) for key_c, key_d in zip(('c1', 'c2', 'c3'), ('d1', 'd2', 'd3'))])
        d = np.array([np.pi ** 2 / self.parameters[atomic_number][key] for key in ('d1', 'd2', 'd3')])

        dvdr_cut = (- a[0] * (1 / r_cut + b[0]) * np.exp(-b[0] * r_cut) / r_cut -
                    2 * c[0] * d[0] * r_cut * np.exp(-d[0] * r_cut ** 2)
                    - a[1] * (1 / r_cut + b[1]) * np.exp(-b[1] * r_cut) / r_cut -
                    2 * c[1] * d[1] * r_cut * np.exp(-d[1] * r_cut ** 2)
                    - a[2] * (1 / r_cut + b[2]) * np.exp(-b[2] * r_cut) / r_cut -
                    2 * c[2] * d[2] * r_cut * np.exp(-d[2] * r_cut ** 2))

        v_cut = (a[0] * np.exp(-b[0] * r_cut) / r_cut + c[0] * np.exp(-d[0] * r_cut ** 2.) +
                 a[1] * np.exp(-b[1] * r_cut) / r_cut + c[1] * np.exp(-d[1] * r_cut ** 2.) +
                 a[2] * np.exp(-b[2] * r_cut) / r_cut + c[2] * np.exp(-d[2] * r_cut ** 2.))

        @jit(nopython=True)
        def func(r):
            v = (a[0] * np.exp(-b[0] * r) / r + c[0] * np.exp(-d[0] * r ** 2.) +
                 a[1] * np.exp(-b[1] * r) / r + c[1] * np.exp(-d[1] * r ** 2.) +
                 a[2] * np.exp(-b[2] * r) / r + c[2] * np.exp(-d[2] * r ** 2.))

            return v - v_cut - (r - r_cut) * dvdr_cut

        return func

    @cached_method_with_args
    def get_projected_potential(self, atomic_number):
        from scipy.special import kn

        a = np.array([np.pi * self.parameters[atomic_number][key] for key in ('a1', 'a2', 'a3')])
        b = np.array([2. * np.pi * np.sqrt(self.parameters[atomic_number][key]) for key in ('b1', 'b2', 'b3')])
        c = np.array(
            [np.pi ** (3. / 2.) * self.parameters[atomic_number][key_c] / self.parameters[atomic_number][key_d] ** (
                    3. / 2.) for key_c, key_d in zip(('c1', 'c2', 'c3'), ('d1', 'd2', 'd3'))])
        d = np.array([np.pi ** 2 / self.parameters[atomic_number][key] for key in ('d1', 'd2', 'd3')])

        def func(r):
            v = (2 * a[0] * kn(0, b[0] * r) + np.sqrt(np.pi / d[0]) * c[0] * np.exp(-d[0] * r ** 2.) +
                 2 * a[1] * kn(0, b[1] * r) + np.sqrt(np.pi / d[1]) * c[1] * np.exp(-d[1] * r ** 2.) +
                 2 * a[2] * kn(0, b[2] * r) + np.sqrt(np.pi / d[2]) * c[2] * np.exp(-d[2] * r ** 2.))

            return v

        return func


class PotentialBase(object):
    @property
    def num_slices(self):
        raise NotImplementedError()

    @property
    def thickness(self):
        raise NotImplementedError()

    @property
    def slice_thickness(self):
        raise NotImplementedError()

    def get_slice(self, i):
        raise NotImplementedError


@jit(nopython=True, nogil=True, parallel=True)
def project_riemann(func, r, r_cut, a, b, num_samples):
    projected = np.zeros((len(a), len(r)))

    for i in prange(a.shape[0]):
        xk = np.linspace(a[i], b[i], num_samples)
        wk = (b[i] - a[i]) / num_samples

        for j in prange(len(r)):
            rxy = np.sqrt(r[j] ** 2 + xk ** 2)
            projected[i, j] = np.sum(func(rxy[rxy < r_cut])) * wk

    return projected


@jit(nopython=True, nogil=True, parallel=True)
def interpolation_kernel(v, r, vr, r_cut, corner_positions, block_positions, x, y, colors):
    for color in np.unique(colors):
        for i in prange(len(corner_positions)):
            if colors[i] == color:
                for j in range(len(x)):
                    for k in range(len(y)):
                        r_interp = np.sqrt((x[j] - block_positions[i, 0]) ** np.float32(2.)
                                           + (y[k] - block_positions[i, 1]) ** np.float32(2.))

                        if r_interp < r_cut:
                            l = int(np.floor(r_interp / r_cut * (len(r) - 1)))

                            if l < (len(r) - 1):
                                y0 = vr[i, l]
                                y1 = vr[i, l + 1]
                                x0 = r[l]
                                x1 = r[l + 1]

                            else:
                                y0 = vr[i, l]
                                y1 = 0.
                                x0 = r[l]
                                x1 = r_cut

                            value = y0 + (r_interp - x0) * (y1 - y0) / (x1 - x0)

                            v[corner_positions[i, 0] + j, corner_positions[i, 1] + k] += value


class Potential(Grid, HasCache, PotentialBase):

    def __init__(self, atoms, origin=None, extent=None, gpts=None, sampling=None, slice_thickness=.5, num_slices=None,
                 parametrization='lobato', periodic=True, method='interpolation', tolerance=1e-2):

        if np.abs(atoms.cell[0, 0]) < 1e-12:
            raise RuntimeError('atoms has no thickness')

        if (np.abs(atoms.cell[0, 0]) < 1e-12) | (np.abs(atoms.cell[1, 1]) < 1e-12):
            raise RuntimeError('atoms has no width')

        if np.any(np.abs(atoms.cell[~np.eye(3, dtype=bool)]) > 1e-12):
            raise RuntimeError('non-diagonal unit cell not supported')

        if periodic is False:
            raise NotImplementedError()

        self._atoms = atoms.copy()
        self._atoms.wrap()

        self._origin = None

        if num_slices is None:
            if slice_thickness is None:
                raise RuntimeError()

            self._num_slices = int(np.floor(atoms.cell[2, 2] / slice_thickness))
        else:
            self._num_slices = num_slices

        self._periodic = periodic
        self._method = method

        if isinstance(parametrization, str):
            if parametrization == 'lobato':
                self._parametrization = LobatoPotential(tolerance=tolerance)

            elif parametrization == 'kirkland':
                self._parametrization = KirklandPotential(tolerance=tolerance)

            else:
                raise RuntimeError()

        if extent is None:
            extent = np.diag(atoms.cell)[:2]

        Grid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, dimensions=2)
        HasCache.__init__(self)

    @property
    def num_slices(self):
        return self._num_slices

    @property
    def atoms(self):
        return self._atoms

    @property
    def thickness(self):
        return self.atoms.cell[2, 2]

    @property
    def parametrization(self):
        return self._parametrization

    @property
    def slice_thickness(self):
        return self.thickness / self.num_slices

    def slice_entrance(self, i):
        return i * self.slice_thickness

    def slice_exit(self, i):
        return (i + 1) * self.slice_thickness

    @cached_method
    def _prepare_interpolation(self):
        positions = self.atoms.get_positions()
        atomic_numbers = self.atoms.get_atomic_numbers()
        unique_atomic_numbers = np.unique(self._atoms.get_atomic_numbers())

        positions = {atomic_number: positions[np.where(atomic_numbers == atomic_number)] for atomic_number in
                     unique_atomic_numbers}

        margin = max([self._parametrization.get_cutoff(atomic_number) for atomic_number in unique_atomic_numbers])
        margin = int(np.ceil(margin / min(self.sampling)))
        padded_gpts = self.gpts + 2 * margin
        v = np.zeros(padded_gpts, dtype=np.float32)

        return v, margin, positions

    def _evaluate_interpolation(self, i, num_spline_nodes=100, num_integration_samples=100):
        v, margin, positions_dict = self._prepare_interpolation()
        v[:, :] = 0.

        for atomic_number, positions in positions_dict.items():
            r_cut = self.parametrization.get_cutoff(atomic_number)
            r = np.linspace(min(self.sampling), r_cut, num_spline_nodes, dtype=np.float32)

            block_margin = int(self._parametrization.get_cutoff(atomic_number) / min(self.sampling))
            block_size = 2 * block_margin + 1

            positions = positions[np.abs(self.slice_entrance(i) + self.slice_thickness / 2 - positions[:, 2]) <
                                  (r_cut + self.slice_thickness / 2)]

            positions = positions.astype(np.float32)

            corner_positions = np.round(positions[:, :2] / self.sampling).astype(np.int32) - block_margin + margin
            block_positions = positions[:, :2] + self.sampling * margin - self.sampling * corner_positions.astype(
                np.float32)

            x = np.linspace(0., block_size * self.sampling[0] - self.sampling[0], block_size, dtype=np.float32)
            y = np.linspace(0., block_size * self.sampling[1] - self.sampling[1], block_size, dtype=np.float32)

            a = self.slice_entrance(i) - positions[:, 2]
            b = self.slice_exit(i) - positions[:, 2]

            func = self.parametrization.get_soft_potential(atomic_number)

            vr = project_riemann(func, r, r_cut, a, b, num_integration_samples)

            graph = nx.from_numpy_array(get_overlap_graph(corner_positions, block_size))
            colors = nx.algorithms.coloring.greedy_color(graph)
            colors = np.array(list(colors.values()))[np.argsort(list(colors.keys()))]

            interpolation_kernel(v, r, vr, r_cut, corner_positions, block_positions, x, y, colors)

        v[margin:2 * margin] += v[-margin:]
        v[-2 * margin:-margin] += v[:margin]
        v[:, margin:2 * margin] += v[:, -margin:]
        v[:, -2 * margin:-margin] += v[:, :margin]
        v = v[margin:-margin, margin:-margin]

        return v / kappa

    def get_slice(self, i, origin=None, extent=None):
        if self._method == 'interpolation':
            return self._evaluate_interpolation(i, origin, extent)
        elif self._method == 'fourier':
            raise RuntimeError()
        else:
            raise NotImplementedError('method {} not implemented')
