from __future__ import annotations

from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
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


def radial_prefactor_b2(parameters, cutoff=10, nodes=1000):
    r = np.linspace(0, cutoff, nodes)[:, None]
    a = parameters[None, :, 0]
    b = parameters[None, :, 1]
    ni = (np.arange(0, 4) / 2 + 3)[None]
    b2 = a * (2 * b - (ni - 2) * r**ni) / (r**ni + b) ** 2
    b2 = b2.sum(-1)
    b2 = interp1d(r[:, 0], b2, fill_value="extrapolate")
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
    b2 = radial_prefactor_b2(parameters)

    mr = np.sum(r_vec * magnetic_moment[None, None, None], axis=-1)

    B = (
        b1(r)[..., None]
        * r_vec
        * mr[..., None]
        * 2
        # + b2(r)[..., None] * 2 * magnetic_moment[None, None, None]
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
    def __init__(self, x, y, magnetic_moments, limits, integrals):
        pass

    def integrate_on_grid(
        self, positions, magnetic_moments, a, b, gpts, sampling, device
    ):
        pass


class IntegratedParametrizedMagneticField:
    def __init__(
        self,
        parametrization="lyon",
        cutoff_tolerance: float = 1e-3,
        integration_step: float = 0.1,
        gpts: int = 64,
        radial_gpts: int = 100,
    ):
        self._parametrization = parametrization
        self._cutoff_tolerance = cutoff_tolerance
        self._integration_step = integration_step
        self._gpts = gpts
        self._radial_gpts = radial_gpts

    @property
    def cutoff(self):
        return 8

    @property
    def gpts(self):
        return self._gpts

    @property
    def parameters(self):
        return get_parameters()

    def _radial_prefactor_b1(self, symbol):
        r = np.linspace(0, self.cutoff, self._radial_gpts)
        return radial_prefactor_b1(r, np.array(self.parameters[symbol]))

    def _grid_coordinates(self):
        return np.linspace(-self.cutoff, self.cutoff, self.gpts)

    def _perpendicular_integrals(self, symbol):
        b1 = self._radial_prefactor_b1(symbol)
        nz = int(np.ceil(self.cutoff * 2 / self._integration_step))
        dz = self.cutoff * 2 / nz

        magnetic_moment = unit_vector_from_angles(np.pi / 3, 0)
        x = self._grid_coordinates()[:, None, None]
        y = self._grid_coordinates()[None, :, None]
        z = np.linspace(-self.cutoff, self.cutoff, nz)[None, None]

        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        mr = x * magnetic_moment[0] + y * magnetic_moment[1] + z * magnetic_moment[2]

        integrals = b1(r) * mr * 2
        integrals = cumulative_trapezoid(integrals, dx=dz, axis=-1, initial=0)
        Bx = integrals * x
        By = integrals * y
        return Bx, By


class ParametrizedMagneticField:
    def __init__(self, atoms, gpts, sampling, slice_thickness, parametrization="lyon"):
        pass
