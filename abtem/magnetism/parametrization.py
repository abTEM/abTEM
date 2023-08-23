from __future__ import annotations

import json
import os

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.optimize import brentq

from abtem.core.ensemble import Ensemble
from abtem.core.grid import disc_meshgrid, HasGridMixin
from abtem.core.utils import get_data_path, EqualityMixin, CopyMixin
from abtem.integrals import cutoff_taper


def get_parameters():
    path = os.path.join(get_data_path(__file__), "lyon.json")

    with open(path, "r") as f:
        parameters = json.load(f)

    return parameters


def radial_prefactor_b1(r, parameters):
    r = r[:, None]
    a = parameters[None, :, 0]
    b = parameters[None, :, 1]
    ni = (np.arange(0, 4) / 2 + 3)[None]
    b1 = a * ni * r ** (ni - 2) / (r**ni + b) ** 2
    b1 = b1.sum(-1)
    b1 = b1 * cutoff_taper(r[:, 0], np.max(r), 0.85)
    b1 = interp1d(r[:, 0], b1, fill_value=0.0, bounds_error=False)
    return b1


def radial_prefactor_b2(r, parameters):
    r = r[:, None]
    a = parameters[None, :, 0]
    b = parameters[None, :, 1]
    ni = (np.arange(0, 4) / 2 + 3)[None]
    b2 = a * (2 * b - (ni - 2) * r**ni) / (r**ni + b) ** 2
    b2 = b2.sum(-1)
    b2 = b2 * cutoff_taper(r[:, 0], np.max(r), 0.85)
    b2 = interp1d(r[:, 0], b2, fill_value=0.0, bounds_error=False)
    return b2


def unit_vector_from_angles(theta, phi):
    R = np.sin(theta)
    m = np.array([R * np.cos(phi), R * np.sin(phi), np.sqrt(1 - R**2)]).T
    return m


def coordinate_grid(
    extent: tuple[float, ...],
    gpts: tuple[int, ...],
    origin: tuple[float, ...],
    endpoint=True,
):
    coordinates = ()
    for r, n, o in zip(extent, gpts, origin):
        coordinates += (np.linspace(0, r, n, endpoint=endpoint) - o,)
    return np.meshgrid(*coordinates, indexing="ij")


def magnetic_field_on_grid(
    extent: tuple[float, float, float],
    gpts: tuple[int, int, int],
    origin: tuple[float, float, float],
    magnetic_moment: np.ndarray,
    parameters: np.ndarray,
    cutoff,
) -> np.ndarray:
    magnetic_moment = np.array(magnetic_moment)

    x, y, z = coordinate_grid(extent, gpts, origin)

    r = np.sqrt(x**2 + y**2 + z**2)
    r_vec = np.stack([x, y, z], -1)

    r_interp = np.linspace(0, cutoff, 100)
    b1 = radial_prefactor_b1(r_interp, parameters)
    b2 = radial_prefactor_b2(r_interp, parameters)

    mr = np.sum(r_vec * magnetic_moment[None, None, None], axis=-1)

    B = (
        b1(r)[..., None] * r_vec * mr[..., None] * 2
        + b2(r)[..., None] * 2 * magnetic_moment[None, None, None]
    )
    return B


def radial_cutoff(func, tolerance=1e-3):
    return brentq(lambda x: func(x) - tolerance, a=1e-3, b=1e3)


def polar2cartesian(polar):
    return np.stack(
        [polar[:, 0] * np.cos(polar[:, 1]), polar[:, 0] * np.sin(polar[:, 1])], axis=-1
    )


def cartesian2polar(cartesian):
    return np.stack(
        [
            np.linalg.norm(cartesian, axis=1),
            np.arctan2(cartesian[:, 1], cartesian[:, 0]),
        ],
        axis=-1,
    )


class CartesianGridInterpolator:
    def __init__(self, points, values, method="linear"):
        self.limits = np.array([[min(x), max(x)] for x in points])
        self.values = np.asarray(values, dtype=float)
        self.order = {"nearest": 0, "linear": 1, "cubic": 3, "quintic": 5}[method]

    def __call__(self, xi):
        """
        `xi` here is an array-like (an array or a list) of points.

        Each "point" is an ndim-dimensional array_like, representing
        the coordinates of a point in ndim-dimensional space.
        """
        # transpose the xi array into the ``map_coordinates`` convention
        # which takes coordinates of a point along columns of a 2D array.
        # xi = np.asarray(xi)

        # convert from data coordinates to pixel coordinates
        ns = self.values.shape

        coords = [
            (val - lo) * (n - 1) / (hi - lo)
            for val, n, (lo, hi) in zip(xi.T, ns, self.limits)
        ]

        # a = (np.array(ns) - 1) / np.diff(self.limits, axis=1)[None, :, 0]
        # xi = xi.T
        # xi -= self.limits[:, 0, None]
        # coords = xi - self.limits[None, :, 0]

        return map_coordinates(self.values, coords, order=self.order, cval=0.0)


class ParametrizedMagneticFieldInterpolator:
    def __init__(
        self, slice_limits, x, y, theta, b1_integrals_xy, b1_integrals_z, b2_integrals
    ):
        method = "linear"

        self._x = x
        self._y = y
        self._theta = theta
        self._b1_integrals_xy = b1_integrals_xy
        self._b1_integrals_z = b1_integrals_z
        self._b2_integrals = b2_integrals
        self._slice_limits = slice_limits
        self._cutoff = np.max([np.max(np.abs(x)), np.max(np.abs(y))])

    def integrate_on_grid(self, position, a, b, theta, phi, gpts, sampling):
        pixel_position = np.floor(position / sampling).astype(int)
        subpixel_position = position / sampling - pixel_position

        radius_out = int(np.floor(self._cutoff / np.min(sampling)))
        radius_in = len(self._x) // 2

        indices = disc_meshgrid(radius_out)

        R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
        points = np.zeros((len(indices), 3))
        points[:, :2] = R.dot(indices.T).T
        points[:, 2] = (
            (theta - self._theta.min()) / self._theta.ptp() * (len(self._theta) - 1)
        )
        points[:, 0] = (
            points[:, 0] * radius_in / radius_out + radius_in + subpixel_position[0]
        )
        points[:, 1] = (
            points[:, 1] * radius_in / radius_out + radius_in + subpixel_position[1]
        )

        slice_index_a = np.searchsorted(self._slice_limits, a)
        slice_index_b = np.searchsorted(self._slice_limits, b)

        b1_integrals_xy = (
            self._b1_integrals_xy[slice_index_b] - self._b1_integrals_xy[slice_index_a]
        )
        b1_term_xy = map_coordinates(b1_integrals_xy, points.T, cval=0.0, order=1)

        b1_integrals_z = (
            self._b1_integrals_z[slice_index_b] - self._b1_integrals_z[slice_index_a]
        )
        b1_term_z = map_coordinates(b1_integrals_z, points.T, cval=0.0, order=1)

        b2_integrals = (
            self._b2_integrals[slice_index_b] - self._b2_integrals[slice_index_a]
        )
        b2_term = map_coordinates(b2_integrals, points[:, :-1].T, cval=0.0, order=1)

        magnetic_moment = unit_vector_from_angles(theta, phi)
        x = (indices[:, 0] + subpixel_position[0]) * sampling[0]
        y = (indices[:, 1] + subpixel_position[1]) * sampling[1]

        Bx = b1_term_xy * x + b2_term * magnetic_moment[0]
        By = b1_term_xy * y + b2_term * magnetic_moment[1]
        Bz = b1_term_z + b2_term * magnetic_moment[2]

        indices[:, 0] = indices[:, 0] + pixel_position[0]
        indices[:, 1] = indices[:, 1] + pixel_position[1]

        mask = (
            (indices[:, 0] >= 0)
            * (indices[:, 1] >= 0)
            * (indices[:, 0] < gpts[0])
            * (indices[:, 1] < gpts[1])
        )
        indices = indices[mask]

        B = np.zeros((3, gpts[0], gpts[1]))
        B[(slice(None), indices[:, 0], indices[:, 1])] = [Bx[mask], By[mask], Bz[mask]]
        return B


class IntegratedParametrizedMagneticField:
    def __init__(
        self,
        parametrization: str = "lyon",
        cutoff_tolerance: float = 1e-3,
        step_size: float = 0.01,
        slice_thickness: float = 0.1,
        gpts: int = 64,
        inclination_gpts: int = 32,
        radial_gpts: int = 100,
    ):
        self._parametrization = parametrization
        self._cutoff_tolerance = cutoff_tolerance
        self._step_size = step_size
        self._slice_thickness = slice_thickness
        self._gpts = gpts
        self._inclination_gpts = inclination_gpts
        self._radial_gpts = radial_gpts

    @property
    def cutoff(self):
        return 6

    @property
    def gpts(self):
        return self._gpts

    @property
    def parameters(self):
        return get_parameters()

    def _radial_prefactor_b1(self, symbol):
        r = np.linspace(0, self.cutoff, self._radial_gpts)
        return radial_prefactor_b1(r, np.array(self.parameters[symbol]))

    def _radial_prefactor_b2(self, symbol):
        r = np.linspace(0, self.cutoff, self._radial_gpts)
        return radial_prefactor_b2(r, np.array(self.parameters[symbol]))

    def _grid_coordinates(self):
        return (np.linspace(-self.cutoff, self.cutoff, self.gpts),) * 2

    def _integration_coordinates(self, a, b):
        z = np.arange(a, b + self._step_size / 2, self._step_size)
        return z

    def _b1_integrals(self, symbol, a, b, x, y, theta):
        b1 = self._radial_prefactor_b1(symbol)
        magnetic_moment = unit_vector_from_angles(theta, 0.0)
        z = self._integration_coordinates(a, b)
        r = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None] ** 2)
        mr = (
            x[:, None, None, None] * magnetic_moment[None, None, None, :, 0]
            + z[None, None, :, None] * magnetic_moment[None, None, None, :, 2]
        )
        integrals_xy = trapezoid(b1(r)[..., None] * mr * 2, x=z, axis=-2)

        integrals_z = trapezoid(
            b1(r)[..., None] * mr * 2 * z[None, None, :, None],
            x=z,
            axis=-2,
        )
        return integrals_xy, integrals_z

    def _b2_integrals(self, symbol, a, b, x, y):
        b2 = self._radial_prefactor_b2(symbol)
        z = self._integration_coordinates(a, b)
        r = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None] ** 2)
        integrals = trapezoid(2 * b2(r), x=z, axis=-1)
        return integrals

    def build(self, symbol):
        x, y = self._grid_coordinates()
        theta = np.linspace(0, np.pi / 2, self._inclination_gpts)

        n = np.ceil(self.cutoff / self._slice_thickness)
        slice_cutoff = n * self._slice_thickness
        slice_limits = np.linspace(-slice_cutoff, slice_cutoff, int(n) * 2 + 1)

        b1_integrals_xy = np.zeros((len(slice_limits), len(x), len(y), len(theta)))
        b1_integrals_z = np.zeros((len(slice_limits), len(x), len(y), len(theta)))
        b2_integrals = np.zeros((len(slice_limits), len(x), len(y)))

        for i, (a, b) in enumerate(zip(slice_limits[:-1], slice_limits[1:]), start=1):
            b1_integrals = self._b1_integrals(symbol, a, b, x, y, theta)
            b1_integrals_xy[i] = b1_integrals_xy[i - 1] + b1_integrals[0]
            b1_integrals_z[i] = b1_integrals_z[i - 1] + b1_integrals[1]

            b2_integrals[i] = b2_integrals[i - 1] + self._b2_integrals(
                symbol, a, b, x, y
            )

        return ParametrizedMagneticFieldInterpolator(
            slice_limits,
            x,
            y,
            theta,
            b1_integrals_xy,
            b1_integrals_z,
            b2_integrals,
        )

    def integrate_magnetic_field(self, symbol, a, b, theta, phi):
        b1 = self._radial_prefactor_b1(symbol)
        b2 = self._radial_prefactor_b2(symbol)

        x, y = self._grid_coordinates()
        z = self._integration_coordinates(a, b)
        magnetic_moment = unit_vector_from_angles(theta, phi)

        r = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None] ** 2)
        mr = (
            x[:, None, None] * magnetic_moment[0]
            + y[None, :, None] * magnetic_moment[1]
            + z[None, None] * magnetic_moment[2]
        )

        integrals = trapezoid(b1(r) * mr * 2, x=z, axis=-1)
        integrals2 = trapezoid(2 * b2(r), x=z, axis=-1)
        Bx = integrals * x[:, None] + magnetic_moment[0] * integrals2
        By = integrals * y[None, :] + magnetic_moment[1] * integrals2

        integrals = trapezoid(b1(r) * mr * 2 * z[None, None], x=z, axis=-1)
        Bz = integrals + magnetic_moment[2] * integrals2
        return Bx, By, Bz


class BaseField(
    Ensemble,
    HasGridMixin,
    EqualityMixin,
    CopyMixin,
):
    @property
    def base_shape(self):
        """Shape of the base axes of the potential."""
        return (self.num_slices,) + (3,) + self.gpts


class ParametrizedMagneticField:
    def __init__(self, atoms, gpts, sampling, slice_thickness, parametrization="lyon"):
        self._atoms = atoms
        self._
