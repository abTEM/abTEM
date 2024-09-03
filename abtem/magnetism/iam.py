from __future__ import annotations

from abc import abstractmethod

import dask.array as da
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from numba import jit
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from abtem.core.axes import AxisMetadata, OrdinalAxis, RealSpaceAxis, ThicknessAxis
from abtem.inelastic.phonons import BaseFrozenPhonons
from abtem.integrals import cutoff_taper
from abtem.magnetism.parametrizations import LyonParametrization
from abtem.potentials.iam import (
    BaseField,
    FieldArray,
    _FieldBuilderFromAtoms,
)


def radial_prefactor_a(r, parameters):
    r = r[:, None]
    a = parameters[None, :, 0]
    b = parameters[None, :, 1]
    ni = (np.arange(0, 5) / 2 + 3)[None]
    a = a / (r**ni + b)
    a = a.sum(-1)
    a = a * cutoff_taper(r[:, 0], np.max(r), 0.85)
    a = interp1d(r[:, 0], a, fill_value=0.0, bounds_error=False)
    return a


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


def vector_potential_3d(
    extent: tuple[float, float, float],
    gpts: tuple[int, int, int],
    origin: tuple[float, float, float],
    magnetic_moment: np.ndarray,
    parameters: np.ndarray,
    cutoff: float,
):
    x, y, z = coordinate_grid(extent, gpts, origin, endpoint=False)
    parameters = np.array(parameters)
    r = np.sqrt(x**2 + y**2 + z**2)

    r_interp = np.linspace(0, cutoff, 200)

    a = radial_prefactor_a(r_interp, parameters)

    r_vec = np.stack([x, y, z], -1)

    m_cross_r = np.cross(magnetic_moment, r_vec, axis=-1)

    field = a(r)[..., None] * m_cross_r

    return field


def magnetic_field_3d(
    extent: tuple[float, float, float],
    gpts: tuple[int, int, int],
    origin: tuple[float, float, float],
    magnetic_moment: np.ndarray,
    parameters: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    magnetic_moment = np.array(magnetic_moment)
    parameters = np.array(parameters)

    x, y, z = coordinate_grid(extent, gpts, origin, endpoint=False)

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


def cartesian2polar_3d(v: np.ndarray) -> List[float]:
    r = np.linalg.norm(v)
    theta = np.arccos(v[2] / r)
    xy_magnitude = np.linalg.norm(v[:2])
    if xy_magnitude > 0.0:
        phi = np.sign(v[1]) * np.arccos(v[0] / xy_magnitude)
    else:
        phi = 0.0

    return [r, theta, phi]


def symmetric_arange(cutoff: float, sampling: float) -> np.ndarray:
    cutoff = np.ceil(cutoff / sampling) * sampling
    values = np.arange(0, cutoff + sampling / 2, sampling)
    return np.concatenate([-values[::-1][:-1], values])


@jit(nopython=True, fastmath=True, nogil=True, cache=True)
def bilinear_weighted_sum(
    array: np.ndarray, x: int, y: int, wx0: float, wx1: float, wy0: float, wy1: float
) -> float:
    return (
        array[x, y] * wx0 * wy0
        + array[x + 1, y] * wx1 * wy0
        + array[x, y + 1] * wx0 * wy1
        + array[x + 1, y + 1] * wx1 * wy1
    )


@jit(nopython=True, fastmath=True, nogil=True, cache=True)
def interpolate(array_out, array_in, position, sampling_out, sampling_in):
    nx = array_in.shape[1]
    ny = array_in.shape[2]

    scale_x = sampling_out[0] / sampling_in[0]
    scale_y = sampling_out[1] / sampling_in[1]
    region_x = int(np.floor(nx // 2 / scale_x))
    region_y = int(np.floor(ny // 2 / scale_y))

    left = max(int(round(position[0] / sampling_out[0])) - region_x, 0)
    right = min(
        int(round(position[0] / sampling_out[0])) + region_x, array_out.shape[1]
    )

    bottom = max(int(round(position[1] / sampling_out[1])) - region_y, 0)
    top = min(int(round(position[1] / sampling_out[1])) + region_y, array_out.shape[2])

    shift_x = np.float32(position[0] / sampling_in[0] - nx // 2)
    shift_y = np.float32(position[1] / sampling_in[1] - ny // 2)

    for i in range(left, right):
        x = np.float32(i * scale_x) - shift_x
        xf = np.floor(x)
        wx1 = x - xf
        wx0 = np.float32(1) - wx1

        for j in range(bottom, top):
            y = np.float32(j * scale_y) - shift_y
            yf = np.floor(y)
            wy1 = y - yf
            wy0 = np.float32(1) - wy1

            array_out[0, i, j] += bilinear_weighted_sum(
                array_in[0], int(xf), int(yf), wx0, wx1, wy0, wy1
            )
            array_out[1, i, j] += bilinear_weighted_sum(
                array_in[1], int(xf), int(yf), wx0, wx1, wy0, wy1
            )
            array_out[2, i, j] += bilinear_weighted_sum(
                array_in[2], int(xf), int(yf), wx0, wx1, wy0, wy1
            )


@jit(nopython=True, fastmath=True, nogil=True, cache=True)
def interpolate_quasi_dipole_field_projections(
    magnetic_field,
    sampling,
    positions,
    magnetic_moments,
    slice_limits,
    integral_limits,
    integral_sampling,
    tables,
):
    B = np.zeros((3, tables.shape[2], tables.shape[3]))
    for position, magnetic_moment in zip(positions, magnetic_moments):
        shifted_limits = slice_limits - position[2]
        i = np.argmin(np.abs(integral_limits - shifted_limits[0]))
        j = np.argmin(np.abs(integral_limits - shifted_limits[1]))
        j = min(tables.shape[1] - 1, j)

        b1xxi = tables[0, j] - tables[0, i]
        b1yyi = b1xxi.T
        b1xyi = tables[1, j] - tables[1, i]
        b1xzi = tables[2, j] - tables[2, i]
        b1yzi = b1xzi.T
        b1zzi = tables[3, j] - tables[3, i]
        b2i = tables[4, j] - tables[4, i]

        B[0] = (
            (b1xxi + b2i) * magnetic_moment[0]
            + b1xyi * magnetic_moment[1]
            + b1xzi * magnetic_moment[2]
        )

        B[1] = (
            b1xyi * magnetic_moment[0]
            + (b1yyi + b2i) * magnetic_moment[1]
            + b1xzi.T * magnetic_moment[2]
        )

        B[2] = (
            b1xzi * magnetic_moment[0]
            + b1yzi * magnetic_moment[1]
            + (b2i + b1zzi) * magnetic_moment[2]
        )

        interpolate(magnetic_field, B, position, sampling, integral_sampling)

    return magnetic_field


@jit(nopython=True, fastmath=True, nogil=True, cache=True)
def interpolate_quasi_dipole_vector_field_projections(
    magnetic_field,
    sampling,
    positions,
    magnetic_moments,
    slice_limits,
    integral_limits,
    integral_sampling,
    tables,
):
    A = np.zeros((3, tables.shape[2], tables.shape[3]))
    for position, magnetic_moment in zip(positions, magnetic_moments):
        shifted_limits = slice_limits - position[2]
        i = np.argmin(np.abs(integral_limits - shifted_limits[0]))
        j = np.argmin(np.abs(integral_limits - shifted_limits[1]))
        j = min(tables.shape[1] - 1, j)

        Ix = tables[0, j] - tables[0, i]
        Iy = Ix.T
        Iz = tables[0, j] - tables[0, i]

        A[0] = magnetic_moment[1] * Iz - magnetic_moment[2] * Iy
        A[1] = magnetic_moment[2] * Ix - magnetic_moment[0] * Iz
        A[2] = magnetic_moment[0] * Iy - magnetic_moment[1] * Ix

        interpolate(magnetic_field, A, position, sampling, integral_sampling)

    return magnetic_field


class QuasiDipoleProjections:
    def __init__(
        self,
        interpolation_func,
        parametrization: str = "lyon",
        # cutoff_tolerance: float = 1e-3,
        cutoff: float = 4.0,
        integration_steps: float = 0.01,
        sampling: float = 0.1,
        slice_thickness: float = 0.1,
    ):
        self._parametrization = LyonParametrization()
        self._cutoff = cutoff
        self._step_size = integration_steps
        self._slice_thickness = slice_thickness
        self._sampling = sampling
        self._interpolation_func = interpolation_func
        self._tables = {}

    @property
    def slice_thickness(self):
        return self._slice_thickness

    def cutoff(self, symbol):
        return self._cutoff

    def _xy_coordinates(self, symbol):
        cutoff = self.cutoff(symbol)
        return symmetric_arange(cutoff, self._sampling)

    def _slice_limits(self, symbol):
        n = np.ceil(self.cutoff(symbol) / self.slice_thickness)
        slice_cutoff = n * self.slice_thickness
        slice_limits = np.linspace(-slice_cutoff, slice_cutoff, int(n) * 2 + 1)
        return slice_limits

    @property
    def parametrization(self):
        return self._parametrization

    @property
    def finite(self):
        return True

    @property
    def periodic(self):
        return False

    @property
    def sampling(self):
        return self._sampling

    @abstractmethod
    def _calculate_integral_table(self, symbol):
        pass

    def get_integral_table(self, symbol: str):
        try:
            table = self._tables[symbol]
        except KeyError:
            table = self._calculate_integral_table(symbol)
            self._tables[symbol] = table

        return table

    def integrate_on_grid(
        self,
        atoms: Atoms,
        a: float,
        b: float,
        gpts: tuple[int, int],
        sampling: [float, float],
        device: str = "cpu",
    ):
        if len(atoms) == 0:
            return np.zeros((3,) + gpts, dtype=np.float32)

        positions = atoms.positions
        magnetic_moments = atoms.get_array("magnetic_moments")
        slice_limits = np.array([a, b])
        integral_sampling = (self._sampling,) * 2

        array = np.zeros((3,) + gpts, dtype=np.float32)
        for number in np.unique(atoms.numbers):
            positions = positions[atoms.numbers == number]
            symbol = chemical_symbols[number]

            integral_limits = self._slice_limits(symbol)
            tables = self.get_integral_table(symbol)

            self._interpolation_func(
                array,
                sampling,
                positions,
                magnetic_moments,
                slice_limits,
                integral_limits,
                integral_sampling,
                tables,
            )

        return array


class QuasiDipoleMagneticFieldProjections(QuasiDipoleProjections):
    def __init__(
        self,
        parametrization: str = "lyon",
        # cutoff_tolerance: float = 1e-3,
        cutoff: float = 4.0,
        integration_steps: float = 0.01,
        sampling: float = 0.1,
        slice_thickness: float = 0.1,
    ):
        super().__init__(
            interpolate_quasi_dipole_field_projections,
            parametrization=parametrization,
            cutoff=cutoff,
            integration_steps=integration_steps,
            sampling=sampling,
            slice_thickness=slice_thickness,
        )

    def _calculate_integral_table(self, symbol):
        r = np.linspace(0, self.cutoff(symbol), 100)
        parameters = np.array(self.parametrization.parameters[symbol])
        b1_radial = radial_prefactor_b1(r, parameters)
        b2_radial = radial_prefactor_b2(r, parameters)

        x = self._xy_coordinates(symbol)
        slice_limits = self._slice_limits(symbol)

        shape = (5, len(slice_limits), *(len(x),) * 2)

        tables = np.zeros(shape, dtype=np.float32)
        for i, (a, b) in enumerate(zip(slice_limits[:-1], slice_limits[1:]), start=1):
            n = int(np.round((b - a) / self._step_size)) + 1
            z = np.linspace(a, b, n)
            r = np.sqrt(
                x[:, None, None] ** 2 + x[None, :, None] ** 2 + z[None, None] ** 2
            )

            tables[0, i] = tables[0, i - 1] + trapezoid(
                b1_radial(r) * x[:, None, None] ** 2, x=z, axis=-1
            )
            tables[1, i] = tables[1, i - 1] + trapezoid(
                b1_radial(r) * x[:, None, None] * x[None, :, None], x=z, axis=-1
            )

            tables[2, i] = tables[2, i - 1] + trapezoid(
                b1_radial(r) * x[:, None, None] * z[None, None, :], x=z, axis=-1
            )
            tables[3, i] = tables[3, i - 1] + trapezoid(
                b1_radial(r) * z[None, None, :] ** 2, x=z, axis=-1
            )

            tables[4, i] = tables[4, i - 1] + trapezoid(b2_radial(r), x=z, axis=-1)

        return tables


class QuasiDipoleVectorPotentialProjections(QuasiDipoleProjections):
    def __init__(
        self,
        parametrization: str = "lyon",
        # cutoff_tolerance: float = 1e-3,
        cutoff: float = 4.0,
        integration_steps: float = 0.01,
        sampling: float = 0.1,
        slice_thickness: float = 0.1,
    ):
        super().__init__(
            interpolate_quasi_dipole_vector_field_projections,
            parametrization=parametrization,
            cutoff=cutoff,
            integration_steps=integration_steps,
            sampling=sampling,
            slice_thickness=slice_thickness,
        )

    def _calculate_integral_table(self, symbol):
        r = np.linspace(0, self.cutoff(symbol), 100)
        parameters = np.array(self.parametrization.parameters[symbol])
        a_radial = radial_prefactor_a(r, parameters)

        x = self._xy_coordinates(symbol)
        slice_limits = self._slice_limits(symbol)

        shape = (2, len(slice_limits), *(len(x),) * 2)

        tables = np.zeros(shape, dtype=np.float32)
        for i, (a, b) in enumerate(zip(slice_limits[:-1], slice_limits[1:]), start=1):
            n = int(np.round((b - a) / self._step_size)) + 1
            z = np.linspace(a, b, n)
            r = np.sqrt(
                x[:, None, None] ** 2 + x[None, :, None] ** 2 + z[None, None] ** 2
            )

            Ix = trapezoid(a_radial(r) * x[:, None, None], x=z, axis=-1)
            tables[0, i] = tables[0, i - 1] + Ix

            Iz = trapezoid(a_radial(r) * z[None, None], x=z, axis=-1)
            tables[1, i] = tables[1, i - 1] + Iz

        return tables


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
                label="Field component",
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


class VectorPotentialArray(BaseMagneticField, FieldArray):
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
            integrator = QuasiDipoleMagneticFieldProjections(
                parametrization=parametrization
            )

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


class VectorPotential(_FieldBuilderFromAtoms, BaseMagneticField):
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
            integrator = QuasiDipoleVectorFieldProjections(
                parametrization=parametrization
            )

        super().__init__(
            atoms=atoms,
            array_object=VectorPotentialArray,
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
