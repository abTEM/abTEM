from __future__ import annotations

import dask.array as da
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.optimize import brentq

from abtem.core.axes import ThicknessAxis, RealSpaceAxis, OrdinalAxis, AxisMetadata
from abtem.core.grid import disk_meshgrid


from abtem.inelastic.phonons import BaseFrozenPhonons
from abtem.integrals import cutoff_taper
from abtem.magnetism.parametrizations import LyonParametrization

from abtem.potentials.iam import (
    BaseField,
    _FieldBuilderFromAtoms,
    FieldArray,
)


def radial_prefactor_b1(r, parameters):
    r = r[:, None]
    a = parameters[None, :, 0]
    b = parameters[None, :, 1]
    ni = (np.arange(0, 5) / 2 + 3)[None]
    b1 = a * ni * r ** (ni - 2) / (r**ni + b) ** 2
    b1 = b1.sum(-1)
    b1 = b1 * cutoff_taper(r[:, 0], np.max(r), 0.85)
    b1 = interp1d(r[:, 0], b1, fill_value=0.0, bounds_error=False)
    return b1


def radial_prefactor_b2(r, parameters):
    r = r[:, None]
    a = parameters[None, :, 0]
    b = parameters[None, :, 1]
    ni = (np.arange(0, 5) / 2 + 3)[None]
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
        b1(r)[..., None] * r_vec * mr[..., None]
        + b2(r)[..., None] * magnetic_moment[None, None, None]
    )
    return B


def radial_cutoff(func, tolerance=1e-3):
    return brentq(lambda x: func(x) - tolerance, a=1e-3, b=1e3)


def index_mask(indices, shape):
    mask = (indices[:, 0] >= 0) * (indices[:, 0] < shape[0])

    for i, n in enumerate(shape[1:], start=1):
        mask *= (indices[:, i] >= 0) * (indices[:, i] < n)

    return mask


def rotate_points_2d(points, phi):
    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    points = R.dot(points.T).T
    return points


def cartesian2polar_3d(v):
    r = np.linalg.norm(v)
    theta = np.arccos(v[2] / r)
    xy_magnitude = np.linalg.norm(v[:2])
    if xy_magnitude > 0.0:
        phi = np.sign(v[1]) * np.arccos(v[0] / xy_magnitude)
    else:
        phi = 0.0

    return [r, theta, phi]


class QuasiDipoleFieldInterpolator:
    def __init__(
        self,
        slice_limits: np.ndarray,
        sampling_xy: float,
        inclination_sampling: float,
        b1_integrals_xy: np.ndarray,
        b1_integrals_z: np.ndarray,
        b2_integrals: np.ndarray,
    ):
        self._inclination_sampling = inclination_sampling
        self._b1_integrals_xy = b1_integrals_xy
        self._b1_integrals_z = b1_integrals_z
        self._b2_integrals = b2_integrals
        self._slice_limits = slice_limits
        self._sampling_xy = sampling_xy

    def _cutoff(self):
        return (self._b1_integrals_xy.shape[1] // 2) * self._sampling_xy

    def _pixel_coordinates(self, sampling):
        cutoff = self._cutoff()
        nx = int(np.ceil(2 * cutoff / sampling[0]))
        ny = int(np.ceil(2 * cutoff / sampling[0]))
        x = np.arange(0, nx)
        y = np.arange(0, ny)
        return x, y

    def integrate_on_grid(
        self,
        atoms: Atoms,
        a: float,
        b: float,
        gpts: tuple[int, int],
        sampling: [float, float],
        device: str = "cpu",
    ):
        positions = atoms.positions
        magnetic_moments = atoms.get_array("magnetic_moments")

        x_pixel, y_pixel = self._pixel_coordinates(sampling)
        pixel_center = np.array([len(x_pixel) // 2, len(y_pixel) // 2])

        x_pixel, y_pixel = np.meshgrid(x_pixel, y_pixel, indexing="ij")
        indices = np.array([x_pixel.ravel(), y_pixel.ravel()]).T
        indices[:, :2] -= pixel_center

        x, y = indices[:, 0] * sampling[0], indices[:, 1] * sampling[1]
        pixel_center_in = np.array((self._b1_integrals_xy.shape[1] // 2,) * 2)

        points = np.zeros((len(indices), 3))
        B = np.zeros((3, gpts[0], gpts[1]))
        for position, magnetic_moment in zip(positions, magnetic_moments):
            pixel_position = np.round(position[:2] / sampling).astype(int)
            subpixel_position = position[:2] / sampling - pixel_position

            magnitude, theta, phi = cartesian2polar_3d(magnetic_moment)

            points[:, :2] = indices[:, :2] * np.array(sampling) / self._sampling_xy
            points[:, :2] = rotate_points_2d(points[:, :2], phi)
            points[:, :2] += pixel_center_in - subpixel_position
            points[:, 2] = theta / self._inclination_sampling

            shifted_a = a - position[2]
            shifted_b = b - position[2]
            ai, bi = np.searchsorted(self._slice_limits, (shifted_a, shifted_b))
            bi = min(bi, len(self._b1_integrals_xy) - 1)

            b1_integrals_xy = self._b1_integrals_xy[bi] - self._b1_integrals_xy[ai]
            b1_term_xy = map_coordinates(b1_integrals_xy, points.T, cval=0.0, order=1)

            b1_integrals_z = self._b1_integrals_z[bi] - self._b1_integrals_z[ai]
            b1_term_z = map_coordinates(b1_integrals_z, points.T, cval=0.0, order=1)

            b2_integrals = self._b2_integrals[bi] - self._b2_integrals[ai]
            b2_term = map_coordinates(b2_integrals, points[:, :-1].T, cval=0.0, order=1)

            Bx = (
                b1_term_xy * (x - subpixel_position[0] * sampling[0]) * magnitude
                + b2_term * magnetic_moment[0]
            )
            By = (
                b1_term_xy * (y - subpixel_position[0] * sampling[0]) * magnitude
                + b2_term * magnetic_moment[1]
            )
            Bz = b1_term_z * magnitude + b2_term * magnetic_moment[2]

            shifted_indices = indices + pixel_position
            mask = index_mask(shifted_indices, gpts)
            shifted_indices = shifted_indices[mask]

            B[(slice(None), shifted_indices[:, 0], shifted_indices[:, 1])] += [
                Bx[mask],
                By[mask],
                Bz[mask],
            ]

        return B


def symmetric_arange(cutoff, sampling):
    cutoff = np.ceil(cutoff / sampling) * sampling
    values = np.arange(0, cutoff + sampling / 2, sampling)
    return np.concatenate([-values[::-1][:-1], values])


class QuasiDipoleProjectionIntegrals:
    def __init__(
        self,
        parametrization: str = "lyon",
        # cutoff_tolerance: float = 1e-3,
        cutoff=4,
        step_size: float = 0.01,
        slice_thickness: float = 0.2,
        xy_sampling: int = 0.1,
        inclination_sampling: int = np.pi / 10,
        radial_gpts: int = 100,
    ):
        self._parametrization = LyonParametrization()
        # self._cutoff_tolerance = cutoff_tolerance
        self._cutoff = cutoff
        self._step_size = step_size
        self._slice_thickness = slice_thickness
        self._xy_sampling = xy_sampling
        self._inclination_sampling = inclination_sampling
        self._radial_gpts = radial_gpts

    @property
    def parametrization(self):
        return self._parametrization

    @property
    def finite(self):
        return True

    @property
    def periodic(self):
        return False

    def cutoff(self, symbol):
        return self._cutoff

    @property
    def xy_sampling(self):
        return self._xy_sampling

    def _radial_prefactor_b1(self, symbol):
        r = np.linspace(0, self.cutoff(symbol), self._radial_gpts)
        return radial_prefactor_b1(r, np.array(self.parametrization.parameters[symbol]))

    def _radial_prefactor_b2(self, symbol):
        r = np.linspace(0, self.cutoff(symbol), self._radial_gpts)
        return radial_prefactor_b2(r, np.array(self.parametrization.parameters[symbol]))

    def _xy_coordinates(self, symbol):
        cutoff = self.cutoff(symbol)
        return symmetric_arange(cutoff, self._xy_sampling)

    def _integration_coordinates(self, a, b):
        z = np.arange(a, b + self._step_size / 2, self._step_size)
        return z

    def _b1_integrals(
        self,
        symbol: str,
        a: float,
        b: float,
        x: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
    ):
        b1 = self._radial_prefactor_b1(symbol)
        magnetic_moment = unit_vector_from_angles(theta, 0.0)
        z = self._integration_coordinates(a, b)
        r = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None] ** 2)
        mr = (
            x[:, None, None, None] * magnetic_moment[None, None, None, :, 0]
            + z[None, None, :, None] * magnetic_moment[None, None, None, :, 2]
        )
        integrals_xy = trapezoid(b1(r)[..., None] * mr, x=z, axis=-2)

        integrals_z = trapezoid(
            b1(r)[..., None] * mr * z[None, None, :, None],
            x=z,
            axis=-2,
        )
        return integrals_xy, integrals_z

    def _b2_integrals(
        self, symbol: str, a: float, b: float, x: np.ndarray, y: np.ndarray
    ):
        b2 = self._radial_prefactor_b2(symbol)
        z = self._integration_coordinates(a, b)
        r = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None] ** 2)
        integrals = trapezoid(b2(r), x=z, axis=-1)
        return integrals

    def build_magnetic_field_interpolator(self, symbol: str):
        x = y = self._xy_coordinates(symbol)
        theta = np.arange(0, np.pi / 2, self._inclination_sampling)

        n = np.ceil(self.cutoff(symbol) / self._slice_thickness)
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

        return QuasiDipoleFieldInterpolator(
            slice_limits,
            self._xy_sampling,
            self._inclination_sampling,
            b1_integrals_xy,
            b1_integrals_z,
            b2_integrals,
        )

    def integrate_magnetic_field(self, symbol, a, b, magnetic_moment):
        b1 = self._radial_prefactor_b1(symbol)
        b2 = self._radial_prefactor_b2(symbol)

        x = y = self._xy_coordinates(symbol)
        z = self._integration_coordinates(a, b)

        r = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None] ** 2)
        mr = (
            x[:, None, None] * magnetic_moment[0]
            + y[None, :, None] * magnetic_moment[1]
            + z[None, None] * magnetic_moment[2]
        )

        integrals = trapezoid(b1(r) * mr, x=z, axis=-1)
        integrals2 = trapezoid(b2(r), x=z, axis=-1)
        Bx = integrals * x[:, None] + magnetic_moment[0] * integrals2
        By = integrals * y[None, :] + magnetic_moment[1] * integrals2

        integrals = trapezoid(b1(r) * mr * z[None, None], x=z, axis=-1)
        Bz = integrals + magnetic_moment[2] * integrals2
        return Bx, By, Bz


class BaseMagneticField(BaseField):
    @property
    def base_shape(self):
        """Shape of the base axes of the potential."""
        return (
            self.num_slices,
            3,
        ) + self.gpts

    @property
    def base_axes_metadata(self):
        """List of AxisMetadata for the base axes."""
        return [
            ThicknessAxis(
                label="z", values=tuple(np.cumsum(self.slice_thickness)), units="Å"
            ),
            OrdinalAxis(
                values=("Bx", "By", "Bz"),
            ),
            RealSpaceAxis(
                label="x", sampling=self.sampling[0], units="Å", endpoint=False
            ),
            RealSpaceAxis(
                label="y", sampling=self.sampling[1], units="Å", endpoint=False
            ),
        ]


class MagneticFieldArray(BaseMagneticField, FieldArray):

    _base_dims = 4

    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        slice_thickness: float | tuple[float, ...] = None,
        extent: float | tuple[float, float] = None,
        sampling: float | tuple[float, float] = None,
        exit_planes: int | tuple[int, ...] = None,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        metadata: dict = None,
    ):
        super().__init__(
            array=array,
            slice_thickness=slice_thickness,
            extent=extent,
            sampling=sampling,
            exit_planes=exit_planes,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )


class MagneticField(_FieldBuilderFromAtoms, BaseMagneticField):
    _exclude_from_copy = ("parametrization",)

    def __init__(
        self,
        atoms: Atoms | BaseFrozenPhonons = None,
        gpts: int | tuple[int, int] = None,
        sampling: float | tuple[float, float] = None,
        slice_thickness: float | tuple[float, ...] = 1,
        parametrization: str = "lyon",
        exit_planes: int | tuple[int, ...] = None,
        plane: str
        | tuple[tuple[float, float, float], tuple[float, float, float]] = "xy",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: tuple[float, float, float] = None,
        periodic: bool = True,
        integrator=None,
        device: str = None,
    ):

        if integrator is None:
            integrator = QuasiDipoleProjectionIntegrals(parametrization=parametrization)

        super().__init__(
            atoms=atoms,
            array_object=MagneticFieldArray,
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            origin=origin,
            box=box,
            periodic=periodic,
            integrator=integrator,
        )

    def _build_integrators(self):
        numbers = np.unique(self.frozen_phonons.atomic_numbers)
        integrators = {
            number: self.integrator.build(
                chemical_symbols[number],
                gpts=self.gpts,
                sampling=self.sampling,
                device=self.device,
            )
            for number in numbers
        }
        return integrators
