from __future__ import annotations

from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import fsolve, brentq

from abtem.core.utils import get_data_path
import os
import json
import numpy as np
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
    m = np.array([R * np.cos(phi), R * np.sin(phi), np.sqrt(1 - R**2)])
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


def perpendicular_b1_integrals(
    x, y, integration_step, magnetic_moment, parameters, r_cut
):
    r_interp = np.linspace(0, r_cut, 100)
    b1 = radial_prefactor_b1(r_interp, parameters)
    nz = int(np.ceil(r_cut * 2 / integration_step))
    dz = r_cut * 2 / nz

    x = x[:, None, None]
    y = y[None, :, None]
    z = np.linspace(-r_cut, r_cut, nz)[None, None]

    r = np.sqrt(x**2 + y**2 + z**2)
    mr = x * magnetic_moment[0] + y * magnetic_moment[1] + z * magnetic_moment[2]

    integrals = b1(r) * mr * 2
    integrals = cumulative_trapezoid(integrals, dx=dz, axis=-1, initial=0)
    integrals = integrals * x
    return integrals


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


class ParametrizedMagneticFieldInterpolator:
    def __init__(self, x, y, z, b1_integrals_xy, b1_integrals_z, b2_integrals):
        self._x = x
        self._y = y

        method = "linear"

        b1_interpolator_xy = RegularGridInterpolator(
            (x, y, z),
            b1_integrals_xy,
            bounds_error=False,
            fill_value=0.0,
            method=method
        )

        b1_interpolator_z = RegularGridInterpolator(
            (x, y, z),
            b1_integrals_z,
            bounds_error=False,
            fill_value=0.0,
            method=method
        )

        b2_interpolator = RegularGridInterpolator(
            (x, y, z),
            b2_integrals,
            bounds_error=False,
            fill_value=0.0,
            method=method
        )

        self._b1_interpolator_xy = b1_interpolator_xy
        self._b1_interpolator_z = b1_interpolator_z
        self._b2_interpolator = b2_interpolator

        xi, yi = np.meshgrid(self._x, self._y, indexing="ij")
        cartesian = np.array([xi.ravel(), yi.ravel()]).T
        self._polar = cartesian2polar(cartesian)

    def integrate_on_grid(self, phi, a, b):  # , positions, phi, gpts, sampling, device):
        theta = np.pi / 4
        magnetic_moment = unit_vector_from_angles(theta, phi)

        n = len(self._x) * len(self._y) * 2
        coords = np.zeros((n, 3))

        coords[: n // 2, :2] = self._polar
        coords[: n // 2, 1] -= phi
        coords[: n // 2, :2] = polar2cartesian(coords[: n // 2, :2])
        coords[n // 2 :, :2] = coords[: n // 2, :2]
        coords[: n // 2, 2] = a
        coords[n // 2:, 2] = b

        shape = len(self._x), len(self._y)

        b1_term_xy = self._b1_interpolator_xy(coords)
        b1_term_xy = (b1_term_xy[n // 2 :] - b1_term_xy[:n // 2]).reshape(shape)

        b1_term_z = self._b1_interpolator_z(coords)
        b1_term_z = (b1_term_z[n // 2:] - b1_term_z[:n // 2]).reshape(shape)

        b2_term = self._b2_interpolator(coords)
        b2_term = (b2_term[n // 2:] - b2_term[:n // 2]).reshape(shape)

        Bx = b1_term_xy * self._x[:, None] + b2_term * magnetic_moment[0]
        By = b1_term_xy * self._y[None] + b2_term * magnetic_moment[1]
        Bz = b1_term_z + b2_term * magnetic_moment[2]
        return Bx, By, Bz


class IntegratedParametrizedMagneticField:
    def __init__(
        self,
        parametrization: str = "lyon",
        cutoff_tolerance: float = 1e-3,
        step_size: float = 0.1,
        gpts: int = 64,
        radial_gpts: int = 100,
    ):
        self._parametrization = parametrization
        self._cutoff_tolerance = cutoff_tolerance
        self._step_size = step_size
        self._gpts = gpts
        self._radial_gpts = radial_gpts

    @property
    def cutoff(self):
        return 4

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
        nz = int(np.ceil((b - a) / self._step_size))
        z = np.linspace(a, b, nz)
        return z

    def _b1_integrals(self, symbol, x, y, z, theta):
        b1 = self._radial_prefactor_b1(symbol)
        magnetic_moment = unit_vector_from_angles(theta, 0.0)

        r = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None] ** 2)
        mr = x[:, None, None] * magnetic_moment[0] + z[None, None] * magnetic_moment[2]

        perpendicular_integrals = cumulative_trapezoid(
            b1(r) * mr * 2, x=z, axis=-1, initial=0.0
        )

        parallel_integrals = cumulative_trapezoid(
            b1(r) * mr * 2 * z[None, None], x=z, axis=-1, initial=0.0
        )
        return perpendicular_integrals, parallel_integrals

    def _b2_integrals(self, symbol, x, y, z):
        b2 = self._radial_prefactor_b2(symbol)
        r = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None] ** 2)
        integrals = cumulative_trapezoid(2 * b2(r), x=z, axis=-1, initial=0.0)
        return integrals

    def build(self, symbol):
        x, y = self._grid_coordinates()
        z = self._integration_coordinates(-self.cutoff, 0)
        theta = np.pi / 4
        parallel_b1_integrals, perpendicular_b1_integrals = self._b1_integrals(
            symbol, x, y, z, theta
        )
        b2_integrals = self._b2_integrals(symbol, x, y, z)
        return ParametrizedMagneticFieldInterpolator(
            x, y, z, parallel_b1_integrals, perpendicular_b1_integrals, b2_integrals
        )

    def integrate_magnetic_field(self, symbol, limits, magnetic_moment):
        b1 = self._radial_prefactor_b1(symbol)
        b2 = self._radial_prefactor_b2(symbol)

        x, y = self._grid_coordinates()
        z = self._integration_coordinates(*limits)

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


class ParametrizedMagneticField:
    def __init__(self, atoms, gpts, sampling, slice_thickness, parametrization="lyon"):
        pass
