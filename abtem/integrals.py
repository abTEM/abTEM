"""Module to describe projection integrals of radial potential parametrizations."""

from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from numba import jit  # type: ignore
from scipy import integrate  # type: ignore
from scipy.optimize import brentq  # type: ignore
from scipy.special import erf  # type: ignore

from abtem.core.backend import (
    cp,
    cupyx,
    device_name_from_array_module,
    get_array_module,
    get_ndimage_module,
)
from abtem.core.fft import fft2, ifft2
from abtem.core.grid import (
    disk_meshgrid,
    polar_spatial_frequencies,
    spatial_frequencies,
)
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype
from abtem.parametrizations import validate_parametrization

if cp is not None:
    from abtem.core._cuda import (
        interpolate_radial_functions as interpolate_radial_functions_cuda,
    )
else:
    interpolate_radial_functions_cuda = None

if TYPE_CHECKING:
    from abtem.parametrizations import Parametrization


def _require_orthogonal_cell(cell: Optional[np.ndarray]) -> None:
    """Raise if a non-orthogonal cell is passed to an integrator that does not yet
    support skewed grids."""
    if cell is None:
        return
    cell = np.asarray(cell, dtype=float)
    if abs(cell[0, 1]) > 1e-12 or abs(cell[1, 0]) > 1e-12:
        raise NotImplementedError(
            "this projection integrator does not yet support non-orthogonal grids; "
            "use the (default) ScatteringFactorProjectionIntegrals"
        )


class FieldIntegrator(EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """Base class for projection integrator object used for calculating projection
    integrals of radial potentials.

    Parameters
    ----------
    periodic : bool
        True indicates that the projection integrals are periodic perpendicular to the
        projection direction.
    finite : bool
        True indicates that the projection integrals are finite along the projection
        direction.
    retain_data : bool, optional
        If True, intermediate calculations are kept.
    """

    def __init__(self, periodic: bool, finite: bool, retain_data: bool = False):
        self._periodic = periodic
        self._finite = finite
        self._retain_data = retain_data

    @abstractmethod
    def integrate_on_grid(
        self,
        positions: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
        cell: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Integrate radial potential between two limits at the given 2D positions on a
        grid. The integration limits are only used when the integration method is
        finite.

        ``cell`` is an optional 2x2 real-space cell (rows are the in-plane lattice
        vectors) for a non-orthogonal grid; ``None`` (default) means an orthogonal grid.

        Parameters
        ----------
        positions : np.ndarray
            2D array of xy-positions of the centers of each radial function [Å].
        a : np.ndarray
            Lower integration limit of the pr
            ojection integrals along z for each position [Å]. The limit is given
            relative to the center of the radial function.
        b : np.ndarray
            Upper integration limit of the projection integrals along z for each
            position [Å]. The limit is given relative to the center of the radial
            function.
        gpts : two int
            Number of grid points in `x` and `y` describing each slice of the potential.
        sampling : two float
            Sampling of the potential in `x` and `y` [1 / Å].
        device : str, optional
            The device used for calculating the potential, 'cpu' or 'gpu'. The default
            is determined by the user configuration file.
        """

    @property
    def periodic(self) -> bool:
        """True indicates that the created projection integrators are implemented only
        for periodic potentials."""
        return self._periodic

    @property
    def finite(self) -> bool:
        """True indicates that the created projection integrators are implemented only
        for infinite potential projections."""
        return self._finite

    @abstractmethod
    def cutoff(self, symbol: str) -> float:
        """Radial cutoff of the potential for the given chemical symbol."""


def correction_projected_scattering_factors(
    symbol, gpts, sampling, short_range="lobato", long_range="peng"
):
    short_range = validate_parametrization(short_range)
    long_range = validate_parametrization(long_range)

    k, _ = polar_spatial_frequencies(gpts, sampling)

    short_range = short_range.projected_scattering_factor(symbol)
    long_range = long_range.projected_scattering_factor(symbol)

    correction = short_range(k**2) - long_range(k**2)
    # correction /= sinc()
    return correction


def gaussian_projected_scattering_factors(
    symbol, gpts, sampling, parametrization="peng"
):
    parametrization = validate_parametrization(parametrization)

    parameters = parametrization.scaled_parameters(
        symbol, "projected_scattering_factor"
    )

    k, _ = polar_spatial_frequencies(gpts, sampling)

    a = parameters[0, :, None, None]
    b = parameters[1, :, None, None]

    projected_gaussians = a * np.exp(-b * k[None] ** 2.0)
    return projected_gaussians


def gaussian_projection_weights(symbol, a, b, parametrization="peng"):
    parametrization = validate_parametrization(parametrization)

    parameters = parametrization.scaled_parameters(
        symbol, "projected_scattering_factor"
    )

    scales = np.pi / np.sqrt(parameters[1])[:, None]

    weights = np.abs(erf(scales * b[None]) - erf(scales * a[None])) / 2
    return weights


class GaussianProjectionIntegrals(FieldIntegrator):
    """
    Parameters
    ----------
    parametrization : str or Parametrization, optional
        The correction radial potential parametrization to integrate. Used for
        correcting the dependence of the potential close to the nuclear core.
        Default is the Lobato parametrization.
    gaussian_parametrization : str or Parametrization, optional
        The Gaussian radial potential parametrization to integrate. Must be
        parametrization described by a superposition of Gaussians. Default is the Peng
        parametrization.
    cutoff_tolerance : float, optional
        The error tolerance used for deciding the radial cutoff distance of the
        potential [eV / e]. Default is 1e-3.
    """

    def __init__(
        self,
        parametrization: str | Parametrization = "lobato",
        gaussian_parametrization: str | Parametrization = "peng",
        cutoff_tolerance: float = 1e-3,
    ):
        self._gaussian_parametrization = validate_parametrization(
            gaussian_parametrization
        )

        self._correction_parametrization = validate_parametrization(parametrization)

        self._cutoff_tolerance = cutoff_tolerance

        super().__init__(periodic=True, finite=True)

        self._gaussians = {}
        self._corrections = {}

    @property
    def cutoff_tolerance(self):
        """The error tolerance used for deciding the radial cutoff distance of the
        potential [eV / e]."""
        return self._cutoff_tolerance

    @property
    def gaussian_parametrization(self):
        """The error tolerance used for deciding the radial cutoff distance of the
        potential [eV / e]."""
        return self._gaussian_parametrization

    @property
    def correction_parametrization(self):
        return self._correction_parametrization

    def cutoff(self, symbol: str) -> float:
        return optimize_cutoff(
            self.gaussian_parametrization.potential(symbol),
            self.cutoff_tolerance,
            a=1e-3,
            b=1e3,
        )  # noqa

    def get_gaussians(self, symbol, gpts, sampling):
        if symbol in self._gaussians:
            return self._gaussian[(symbol, gpts, sampling)]

        return gaussian_projected_scattering_factors(symbol, gpts, sampling)

    def get_corrections(self, symbol, gpts, sampling):
        if symbol in self._corrections:
            return self._corrections[(symbol, gpts, sampling)]

        return correction_projected_scattering_factors(symbol, gpts, sampling)

    def _integrate_gaussians(self, positions, symbol, a, b, gpts, sampling, device):
        gaussians = self.get_gaussians(symbol, gpts, sampling)

        shifted_a = a - positions[:, 2]
        shifted_b = b - positions[:, 2]

        weights = gaussian_projection_weights(symbol, shifted_a, shifted_b)

        xp = get_array_module(device)
        positions = (positions[:, :2] / sampling).astype(xp.float32)

        array = xp.zeros(gpts, dtype=xp.complex64)
        for i in range(5):
            temp = xp.zeros_like(array, dtype=xp.complex64)
            superpose_deltas(positions, temp, weights=weights[i])
            array += fft2(temp, overwrite_x=False) * gaussians[i].astype(xp.complex64)

        return array

    def _integrate_corrections(self, positions, symbol, a, b, gpts, sampling, device):
        corrections = self.get_corrections(symbol, gpts, sampling)

        xp = get_array_module(device)

        positions = positions[(positions[:, 2] >= a) * (positions[:, 2] < b)]
        positions = (positions[:, :2] / sampling).astype(xp.float32)

        array = xp.zeros(gpts, dtype=xp.complex64)

        superpose_deltas(positions, array)

        corrections = fft2(array, overwrite_x=False) * corrections

        return corrections

    def integrate_on_grid(
        self,
        atoms: Atoms,
        a: np.ndarray,
        b: np.ndarray,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
        fourier_space: bool = False,
        cell: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        _require_orthogonal_cell(cell)
        xp = get_array_module(device)

        array = xp.zeros(gpts, dtype=get_dtype(complex=True))
        for number in np.unique(atoms.numbers):
            positions = atoms.positions[atoms.numbers == number]
            symbol = chemical_symbols[number]

            array += self._integrate_gaussians(
                positions, symbol, a, b, gpts, sampling, device
            )
            array += self._integrate_corrections(
                positions, symbol, a, b, gpts, sampling, device
            )

        return ifft2(array / sinc(gpts, sampling, device)).real


def sinc(
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    device: str = "cpu",
    cell: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns an array representing a 2D sinc function centered at [0, 0]. The result is
    used to compensate for the finite size of single pixels used for representing delta
    functions.

    Parameters
    ----------
    gpts : two int
        Number of grid points in the first and second dimension to evaluate the sinc
        over.
    sampling : two float
        Size of the pixels of the grid determining the scale of the sinc.
    device : str
        The array is created on this device ('cpu' or 'gpu').

    Returns
    -------
    sinc : np.ndarray
        2D sinc function.
    """
    xp = get_array_module(device)
    kx, ky = spatial_frequencies(gpts, sampling, return_grid=False, xp=xp)
    # the interpolation rolloff lives in fractional index space (kx*dx = h1/N1) and is
    # therefore independent of the cell shape; only the pixel area dk2 changes for a
    # skewed cell.
    k = xp.sqrt((kx[:, None] * sampling[0]) ** 2 + (ky[None] * sampling[1]) ** 2)
    if cell is None:
        dk2 = sampling[0] * sampling[1]
    else:
        dk2 = abs(float(np.linalg.det(np.asarray(cell, dtype=float)))) / (
            gpts[0] * gpts[1]
        )
    k[0, 0] = 1
    sinc = xp.sin(k) / k * dk2
    sinc[0, 0] = dk2
    return sinc


def superpose_deltas(
    positions: np.ndarray,
    array: np.ndarray,
    weights: Optional[np.ndarray] = None,
    round_positions: bool = False,
) -> np.ndarray:
    """
    Add superposition of delta functions at given positions to a 2D array.

    Parameters
    ----------
    positions : np.ndarray
        Array of 2D positions as an nx2 array. The positions are given in units of
        pixels.
    array : np.ndarray
        The delta functions are added to this 2D array.
    weights : np.ndarray, optional
        If given each delta function is weighted by the given factor. Must match the
        length of `positions`.
    round_positions : bool, optional
        If True, the delta function positions are rounded to the center of the nearest
        pixel, otherwise subpixel precision is used.

    Returns
    -------
    array : np.ndarray
        The array with the delta functions added.
    """

    xp = get_array_module(array)
    shape = array.shape

    positions = xp.array(positions)

    if round_positions:
        rounded = xp.round(positions).astype(xp.int32)
        i, j = rounded[:, 0][None] % shape[0], rounded[:, 1][None] % shape[1]
        v = xp.array([1.0], dtype=get_dtype(complex=False))[:, None]
    else:
        rounded = xp.floor(positions).astype(xp.int32)
        rows, cols = rounded[:, 0], rounded[:, 1]
        x = positions[:, 0] - rows
        y = positions[:, 1] - cols
        xy = x * y
        i = xp.array([rows % shape[0], (rows + 1) % shape[0]] * 2)
        j = xp.array([cols % shape[1]] * 2 + [(cols + 1) % shape[1]] * 2)
        v = xp.array(
            [1 + xy - y - x, x - xy, y - xy, xy], dtype=get_dtype(complex=False)
        )

    if weights is not None:
        v = v * weights[None]
    
    try:
        xp.add.at(array, (i, j), v)
    except AttributeError:
        
        if device_name_from_array_module(xp) == "gpu":
            cupyx.scatter_add(array, (i, j), v)
        else:
            raise RuntimeError()

    return array


class ScatteringFactorProjectionIntegrals(FieldIntegrator):
    """
    A FieldIntegrator calculating infinite projections of radial potential
    parametrizations. The hybrid real and reciprocal space method by
    Wouter Van den Broek et al. is used.

    Parameters
    ----------
    parametrization : str or Parametrization, optional
        The radial potential parametrization to integrate. Default is the Lobato
        parametrization.

    References
    ----------
    W. Van den Broek et al. Ultramicroscopy, 158:89-97, 2015.
    doi:10.1016/j.ultramic.2015.07.005.
    """

    def __init__(self, parametrization: str | Parametrization = "lobato"):
        self._parametrization = validate_parametrization(parametrization)
        self._scattering_factors = {}
        super().__init__(periodic=True, finite=False)

    @property
    def parametrization(self) -> Parametrization:
        return self._parametrization

    def cutoff(self, symbol: str) -> float:
        return np.inf

    def _calculate_scattering_factor(
        self,
        symbol: str,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
        cell: Optional[np.ndarray] = None,
    ):
        xp = get_array_module(device)

        if cell is None:
            kx, ky = spatial_frequencies(gpts, sampling, xp=np)
            k2 = kx[:, None] ** 2 + ky[None] ** 2
        else:
            # non-orthogonal cell: |g|^2 from the reciprocal metric
            from abtem.core.grid import Grid

            k2 = Grid(gpts=gpts, sampling=sampling, cell=cell).k_squared(xp=np)

        f = self.parametrization.projected_scattering_factor(symbol)(k2)
        f = xp.asarray(f, dtype=get_dtype(complex=False))

        if symbol in self.parametrization.sigmas.keys():
            sigma = self.parametrization.sigmas[symbol]
            f = f * xp.exp(
                -xp.asarray(k2, dtype=get_dtype(complex=False))
                * (xp.pi * sigma / xp.sqrt(1 / 2)) ** 2
            )

        return f

    def get_scattering_factor(self, symbol, gpts, sampling, device, cell=None):
        try:
            scattering_factor = self.scattering_factors[symbol]
        except KeyError:
            scattering_factor = self._calculate_scattering_factor(
                symbol, gpts, sampling, device, cell=cell
            )
            self._scattering_factors[symbol] = scattering_factor

        return scattering_factor

    @property
    def scattering_factors(self) -> dict[str, np.ndarray]:
        """Projected scattering factor array on a 2D grid."""
        return self._scattering_factors

    def integrate_on_grid(
        self,
        atoms: Atoms,
        a: np.ndarray,
        b: np.ndarray,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
        cell: Optional[np.ndarray] = None,
    ):
        xp = get_array_module(device)
        if len(atoms) == 0:
            return xp.zeros(gpts, dtype=get_dtype(complex=False))

        inverse_cell = None if cell is None else np.linalg.inv(np.asarray(cell, float))

        array = xp.zeros(gpts, dtype=get_dtype(complex=False))
        for number in np.unique(atoms.numbers):
            scattering_factor = self.get_scattering_factor(
                chemical_symbols[number], gpts, sampling, device, cell=cell
            )

            positions = atoms.positions[atoms.numbers == number]

            if inverse_cell is None:
                positions = positions[:, :2] / sampling
            else:
                # skewed grid: place atoms in fractional grid coordinates
                positions = (positions[:, :2] @ inverse_cell) * np.array(gpts)
            positions = positions.astype(get_dtype(complex=False))

            temp_array = xp.zeros(gpts, dtype=get_dtype(complex=False))

            temp_array = superpose_deltas(positions, temp_array).astype(
                get_dtype(complex=True)
            )

            temp_array = fft2(temp_array, overwrite_x=True)

            temp_array *= scattering_factor / sinc(gpts, sampling, device, cell=cell)

            # if not fourier_space:
            temp_array = ifft2(temp_array, overwrite_x=True).real

            array += temp_array

        return array


# Deliberately not numba parallel=True: the workqueue threading layer (the
# fallback when neither TBB nor OpenMP is available) hard-aborts the process
# when parallel kernels are launched concurrently from multiple Python
# threads, which is exactly what happens when dask tasks build potential
# slices in a threaded scheduler. Since the kernel is nogil, thread-level
# parallelism is instead applied by the caller (see
# _threaded_interpolate_radial_functions), which is safe under any layer.
@jit(nopython=True, nogil=True)
def interpolate_radial_functions(
    array: np.ndarray,
    positions: np.ndarray,
    disk_indices: np.ndarray,
    disk_counts: np.ndarray,
    sampling: tuple[float, float],
    radial_gpts: np.ndarray,
    radial_functions: np.ndarray,
    radial_derivative: np.ndarray,
):
    n = radial_gpts.shape[0]
    dt = np.log(radial_gpts[-1] / radial_gpts[0]) / (n - 1)

    for i in range(positions.shape[0]):
        px = int(round(positions[i, 0] / sampling[0]))
        py = int(round(positions[i, 1] / sampling[1]))

        # The disk indices are sorted by radial distance, so the loop may stop
        # after the first disk_counts[i] pixels (those within the lateral
        # cutoff of atom i for the current slice).
        for j in range(disk_counts[i]):
            k = px + disk_indices[j, 0]
            m = py + disk_indices[j, 1]

            if (k < array.shape[0]) & (m < array.shape[1]) & (k >= 0) & (m >= 0):
                r_interp = np.sqrt(
                    (k * sampling[0] - positions[i, 0]) ** 2
                    + (m * sampling[1] - positions[i, 1]) ** 2
                )

                idx = int(np.floor(np.log(r_interp / radial_gpts[0] + 1e-12) / dt))

                if idx < 0:
                    array[k, m] += radial_functions[i, 0]
                elif idx < n - 1:
                    slope = radial_derivative[i, idx]
                    array[k, m] += (
                        radial_functions[i, idx] + (r_interp - radial_gpts[idx]) * slope
                    )


@jit(nopython=True, nogil=True)
def interpolate_radial_functions_skew(
    array: np.ndarray,
    positions: np.ndarray,
    disk_indices: np.ndarray,
    disk_counts: np.ndarray,
    sampling_vectors: np.ndarray,
    inv_jacobian: np.ndarray,
    radial_gpts: np.ndarray,
    radial_functions: np.ndarray,
    radial_derivative: np.ndarray,
):
    """Skew-grid version of :func:`interpolate_radial_functions`.

    Pixel ``(k, m)`` sits at Cartesian ``r = k * a1 + m * a2`` where ``a1, a2`` are the
    two real-space sampling vectors (rows of ``sampling_vectors``). ``inv_jacobian`` is
    the inverse of ``[[a1; a2]].T`` (precomputed) and maps a Cartesian position to its
    fractional pixel coordinates. ``disk_indices`` are pixel-index offsets sorted by
    Cartesian distance from the atom and covering the cutoff disk; ``disk_counts[i]``
    is how many of those (nearest-first) offsets fall within atom ``i``'s lateral
    cutoff for the current slice (the radial-out-of-range branch below is a final
    safety net, not the primary truncation).
    """
    n = radial_gpts.shape[0]
    dt = np.log(radial_gpts[-1] / radial_gpts[0]) / (n - 1)

    a1x = sampling_vectors[0, 0]
    a1y = sampling_vectors[0, 1]
    a2x = sampling_vectors[1, 0]
    a2y = sampling_vectors[1, 1]

    for i in range(positions.shape[0]):
        xi = positions[i, 0]
        yi = positions[i, 1]

        # map Cartesian -> fractional pixel coords, then round to the nearest pixel
        fk = inv_jacobian[0, 0] * xi + inv_jacobian[0, 1] * yi
        fm = inv_jacobian[1, 0] * xi + inv_jacobian[1, 1] * yi
        px = int(round(fk))
        py = int(round(fm))

        for j in range(disk_counts[i]):
            k = px + disk_indices[j, 0]
            m = py + disk_indices[j, 1]

            if (k < array.shape[0]) & (m < array.shape[1]) & (k >= 0) & (m >= 0):
                rx = k * a1x + m * a2x - xi
                ry = k * a1y + m * a2y - yi
                r_interp = np.sqrt(rx * rx + ry * ry)

                idx = int(np.floor(np.log(r_interp / radial_gpts[0] + 1e-12) / dt))

                if idx < 0:
                    array[k, m] += radial_functions[i, 0]
                elif idx < n - 1:
                    slope = radial_derivative[i, idx]
                    array[k, m] += (
                        radial_functions[i, idx] + (r_interp - radial_gpts[idx]) * slope
                    )


_interpolation_pool: Optional[ThreadPoolExecutor] = None

# Upper bound on the temporary per-thread accumulation buffers used by
# _threaded_interpolate_radial_functions. For very large grids the thread
# count is reduced so the buffers stay below this size, degrading gracefully
# to the serial kernel.
_INTERPOLATION_BUFFER_BUDGET = 256 * 1024**2


def _get_interpolation_pool() -> ThreadPoolExecutor:
    global _interpolation_pool
    if _interpolation_pool is None:
        _interpolation_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
    return _interpolation_pool


def _threaded_interpolate_radial_functions(
    array: np.ndarray,
    positions: np.ndarray,
    disk_indices: np.ndarray,
    disk_counts: np.ndarray,
    sampling: tuple[float, float],
    radial_gpts: np.ndarray,
    radial_functions: np.ndarray,
    radial_derivative: np.ndarray,
):
    """Run the (nogil) interpolation kernel across a thread pool.

    The atoms are dealt round-robin to per-thread accumulation buffers so no
    two threads ever write to the same array; the buffers are summed into
    ``array`` afterwards. Since the kernel releases the GIL, plain Python
    threads give full parallelism without involving numba's threading layer
    (whose workqueue backend aborts on concurrent launches, e.g. from dask).
    """
    num_chunks = min(
        os.cpu_count() or 1,
        len(positions),
        max(int(_INTERPOLATION_BUFFER_BUDGET // max(array.nbytes, 1)), 1),
    )

    if num_chunks <= 1:
        interpolate_radial_functions(
            array,
            positions,
            disk_indices,
            disk_counts,
            sampling,
            radial_gpts,
            radial_functions,
            radial_derivative,
        )
        return

    buffers = np.zeros((num_chunks,) + array.shape, dtype=array.dtype)

    def run_chunk(chunk: int):
        # Round-robin selection balances the load when disk sizes vary along
        # the atom order (e.g. sorted by z relative to the slice).
        selection = slice(chunk, None, num_chunks)
        interpolate_radial_functions(
            buffers[chunk],
            np.ascontiguousarray(positions[selection]),
            disk_indices,
            np.ascontiguousarray(disk_counts[selection]),
            sampling,
            radial_gpts,
            np.ascontiguousarray(radial_functions[selection]),
            np.ascontiguousarray(radial_derivative[selection]),
        )

    pool = _get_interpolation_pool()
    futures = [pool.submit(run_chunk, chunk) for chunk in range(num_chunks)]
    for future in futures:
        future.result()

    array += buffers.sum(axis=0)


class ProjectionIntegralTable:
    """
    A ProjectionIntegrator calculating finite projections of radial potential
    parametrizations. An integral table for each used to evaluate the projection
    integrals for each atom in a slice given p integral limits. The projected potential
    evaluated along the

    Parameters
    ----------
    radial_gpts : array
        The points along a radial in the `xy`-plane where the projection integrals of
        the integral table are evaluated.
    limits : array
        The points along the projection direction where the projection integrals are
        evaluated.
    """

    def __init__(self, radial_gpts: np.ndarray, limits: np.ndarray, values: np.ndarray):
        assert values.shape[0] == len(limits)
        assert values.shape[1] == len(radial_gpts)

        self._radial_gpts = radial_gpts
        self._limits = limits
        self._values = values

    @property
    def radial_gpts(self) -> np.ndarray:
        return self._radial_gpts

    @property
    def limits(self) -> np.ndarray:
        return self._limits

    @property
    def values(self) -> np.ndarray:
        return self._values

    def _interpolate(self, x: np.ndarray) -> np.ndarray:
        # Piecewise-linear interpolation along the limits axis with linear
        # extrapolation from the end segments; equivalent to
        # scipy.interpolate.interp1d(limits, values, axis=0, kind="linear",
        # fill_value="extrapolate"), but without rebuilding an interpolator
        # for every slice.
        idx = np.searchsorted(self._limits, x, side="right") - 1
        idx = np.clip(idx, 0, len(self._limits) - 2)
        x0 = self._limits[idx]
        x1 = self._limits[idx + 1]
        weights = (x - x0) / (x1 - x0)
        return self._values[idx] + weights[..., None] * (
            self._values[idx + 1] - self._values[idx]
        )

    def integrate(self, a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
        a = np.atleast_1d(np.asarray(a, dtype=self._limits.dtype))
        b = np.atleast_1d(np.asarray(b, dtype=self._limits.dtype))
        return self._interpolate(b) - self._interpolate(a)


def optimize_cutoff(func: Callable, tolerance: float, a: float, b: float) -> float:
    """
    Calculate the point where a function becomes lower than a given tolerance within a
    given bracketing interval.

    Parameters
    ----------
    func : callable
        The function to calculate the cutoff for.
    tolerance : float
        The tolerance to calculate the cutoff for.
    a : float
        One end of the bracketing interval.
    b : float
        The other end of the bracketing interval.

    Returns
    -------
    cutoff : float
    """
    f = brentq(f=lambda r: np.abs(func(r)) - tolerance, a=a, b=b)
    return f


def cutoff_taper(radial_gpts, cutoff, taper):
    taper_start = taper * cutoff
    taper_mask = radial_gpts > taper_start
    taper_values = np.ones_like(radial_gpts)
    taper_values[taper_mask] = (
        np.cos(np.pi * (radial_gpts[taper_mask] - taper_start) / (cutoff - taper_start))
        + 1.0
    ) / 2
    return taper_values


class QuadratureProjectionIntegrals(FieldIntegrator):
    """
    Projection integration plan for calculating finite projection integrals based on
    Gaussian quadrature rule.

    Parameters
    ----------
    parametrization : str or Parametrization, optional
        The potential parametrization describing the radial dependence of the potential.
        Default is 'lobato'.
    cutoff_tolerance : float, optional
        The error tolerance used for deciding the radial cutoff distance of the
        potential [eV / e]. Default is 1e-3.
    taper : float, optional
        The fraction from the cutoff of the radial distance from the core where the
        atomic potential starts tapering
        to zero. Default is 0.85.
    integration_step : float, optional
        The step size between integration limits used for calculating the integral
        table. Default is 0.02.
    quad_order : int, optional
        Order of quadrature integration passed to scipy.integrate.fixed_quad.
        Default is 8.
    """

    def __init__(
        self,
        parametrization: str | Parametrization = "lobato",
        cutoff_tolerance: float = 1e-4,
        inner_cutoff_factor: float = 2.0,
        taper: float = 0.85,
        integration_step: float = 0.02,
        quad_order: int = 8,
    ):
        self._parametrization = validate_parametrization(parametrization)
        self._taper = taper
        self._quad_order = quad_order
        self._cutoff_tolerance = cutoff_tolerance
        self._inner_cutoff_factor = inner_cutoff_factor
        self._integration_step = integration_step
        self._tables: dict[str, ProjectionIntegralTable] = {}
        self._sorted_disks: dict[
            tuple[str, tuple[float, float]], tuple[np.ndarray, np.ndarray]
        ] = {}
        # Device-resident copies of disk_indices/radial_gpts, keyed by
        # (symbol, sampling, device). Both are invariant per (symbol,
        # sampling) across all slices, but integrate_on_grid is called once
        # per slice per species; without this cache each call re-uploads
        # them to the GPU from scratch, which measurably dominated GPU
        # build time (see PR #309 discussion) despite the arrays never
        # changing between calls.
        self._device_arrays: dict[tuple[str, tuple[float, float], str], tuple] = {}

        super().__init__(periodic=False, finite=True)

    @property
    def parametrization(self):
        """The potential parametrization describing the radial dependence of the
        potential."""
        return self._parametrization

    @property
    def quad_order(self):
        """Order of quadrature integration."""
        return self._quad_order

    @property
    def tables(self):
        return self._tables

    @property
    def cutoff_tolerance(self) -> float:
        """The error tolerance used for deciding the radial cutoff distance of the
        potential [eV / e]."""
        return self._cutoff_tolerance

    @property
    def integration_step(self) -> float:
        """The step size between integration limits used for calculating the integral
        table."""
        return self._integration_step

    def cutoff(self, symbol: str) -> float:
        return optimize_cutoff(
            self.parametrization.potential(symbol), self.cutoff_tolerance, a=1e-3, b=1e3
        )

    @staticmethod
    def _radial_gpts(inner_cutoff: float, cutoff: float) -> np.ndarray:
        num_points = int(np.ceil(cutoff / inner_cutoff))
        return np.geomspace(inner_cutoff, cutoff, num_points)

    @staticmethod
    def _taper_values(radial_gpts: np.ndarray, cutoff: float, taper: float):
        taper_start = taper * cutoff
        taper_mask = radial_gpts > taper_start
        taper_values = np.ones_like(radial_gpts)
        taper_values[taper_mask] = (
            np.cos(
                np.pi * (radial_gpts[taper_mask] - taper_start) / (cutoff - taper_start)
            )
            + 1.0
        ) / 2
        return taper_values

    def _integral_limits(self, cutoff: float):
        limits = np.linspace(-cutoff, 0, int(np.ceil(cutoff / self._integration_step)))
        return np.concatenate((limits, -limits[::-1][1:]))

    def _calculate_integral_table(
        self, symbol: str, sampling: tuple[float, float]
    ) -> ProjectionIntegralTable:
        potential = self.parametrization.potential(symbol)
        cutoff = self.cutoff(symbol)

        inner_limit = min(sampling) / self._inner_cutoff_factor
        radial_gpts = self._radial_gpts(inner_limit, cutoff)
        limits = self._integral_limits(cutoff)

        # def potential_blurred(r, func):
        #     ri = np.linspace(-4, 4, 101)[(None,) * len(r.shape)]
        #     r = r[..., None]
        #     r = np.abs(r + ri)
        #     f = (
        #         func(r) * r[None, None] ** 2 * np.exp(-(ri**2) / 0.1)[None, None]
        #     ).sum(-1)
        #     return f
        #
        # potential_blurred_ = lambda r: potential_blurred(r, potential)
        #
        # projection = lambda z: potential_blurred_(
        #     np.sqrt(radial_gpts[:, None] ** 2 + z[None] ** 2)
        # )

        # projection = lambda z: potential(
        #    np.sqrt(radial_gpts[:, None] ** 2 + z[None] ** 2)
        # )

        def project_along_z(z):
            return potential(np.sqrt(radial_gpts[:, None] ** 2 + z[None] ** 2))

        # * np.exp(-(radial_gpts[:, None] ** 2) / 10000)

        table = np.zeros((len(limits) - 1, len(radial_gpts)))
        table[0, :] = integrate.fixed_quad(
            project_along_z, -limits[0] * 2, limits[0], n=self._quad_order
        )[0]

        for j, (a, b) in enumerate(zip(limits[1:-1], limits[2:])):
            table[j + 1] = (
                table[j]
                + integrate.fixed_quad(project_along_z, a, b, n=self._quad_order)[0]
            )

        table = table * self._taper_values(radial_gpts, cutoff, self._taper)[None]

        self._tables[symbol] = ProjectionIntegralTable(radial_gpts, limits[1:], table)

        return self._tables[symbol]

    def get_integral_table(self, symbol, sampling):
        """
        Build table of projection integrals of the radial atomic potential.

        Parameters
        ----------
        symbol : str
            Chemical symbol to build the integral table.
        inner_limit : float, optional
            Smallest radius from the core at which to calculate the projection integral
            [Å].

        Returns
        -------
        projection_integral_table :
            ProjectionIntegralTable
        """
        try:
            scattering_factor = self.tables[symbol]
        except KeyError:
            scattering_factor = self._calculate_integral_table(symbol, sampling)
            self._tables[symbol] = scattering_factor

        return scattering_factor

    def integrate_on_grid(
        self,
        atoms: Atoms,
        a: float,
        b: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
        cell: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Detect a skewed in-plane grid. When the cell is None or orthogonal the original
        # rectangular-grid kernel is used unchanged (bit-identical). Otherwise the radial
        # potential is stamped via a metric-aware pixel <-> Cartesian map.
        if cell is None:
            skew = False
        else:
            cell_arr = np.asarray(cell, dtype=float)
            skew = abs(cell_arr[0, 1]) > 1e-12 or abs(cell_arr[1, 0]) > 1e-12

        xp = get_array_module(device)

        if skew:
            if xp is cp:
                raise NotImplementedError(
                    "finite projection on a non-orthogonal grid is not yet supported on "
                    "the GPU; use device='cpu' or projection='infinite'"
                )
            sampling_vectors = np.stack(
                [cell_arr[0] / gpts[0], cell_arr[1] / gpts[1]]
            )
            # inv_jacobian maps a Cartesian (x, y) to fractional pixel coords (k, m)
            inv_jacobian = np.linalg.inv(sampling_vectors.T)

        array = xp.zeros(gpts, dtype=get_dtype(complex=False))
        for number in np.unique(atoms.numbers):
            table = self.get_integral_table(chemical_symbols[number], sampling)

            positions = atoms.positions[atoms.numbers == number]

            shifted_a = a - positions[:, 2]
            shifted_b = b - positions[:, 2]

            if skew:
                # the cutoff disk in Cartesian space maps to an ellipse in pixel space;
                # use a square mesh whose operator-norm bound (||inv_J||_op * R) safely
                # contains the ellipse (the radial-out-of-range branch silently skips
                # any pixels that turn out to be outside the actual Cartesian disk).
                # skew is CPU-only, so no device caching applies, but the mesh itself
                # (and its sort by true Cartesian radius) is cached like the
                # orthogonal case, and truncated per atom/slice via disk_counts below
                # -- without this a parametrization whose cutoff exceeds the cell (e.g.
                # the Ewald short-range correction) rescans the full, hugely oversized
                # mesh for every slice even though most slices are far enough from the
                # atom in z that only a tiny lateral neighborhood can contribute.
                cutoff = table.radial_gpts[-1]
                disk_key = (chemical_symbols[number], tuple(sampling_vectors.ravel()))
                if disk_key in self._sorted_disks:
                    disk, disk_radii = self._sorted_disks[disk_key]
                else:
                    op_norm = float(np.linalg.norm(inv_jacobian, ord=2))
                    disk_radius_pixels = int(np.ceil(cutoff * op_norm))
                    disk = disk_meshgrid(disk_radius_pixels)
                    cartesian = disk[:, 0:1] * sampling_vectors[0] + disk[
                        :, 1:2
                    ] * sampling_vectors[1]
                    disk_radii = np.linalg.norm(cartesian, axis=1)
                    order = np.argsort(disk_radii)
                    disk = np.ascontiguousarray(disk[order])
                    disk_radii = disk_radii[order]
                    self._sorted_disks[disk_key] = (disk, disk_radii)

                dz = np.maximum(np.maximum(shifted_a, -shifted_b), 0.0)
                lateral_cutoff = np.sqrt(np.maximum(cutoff**2 - dz**2, 0.0))
                margin = (
                    np.linalg.norm(sampling_vectors[0])
                    + np.linalg.norm(sampling_vectors[1])
                ) / 2
                disk_counts = np.searchsorted(
                    disk_radii, lateral_cutoff + margin, side="right"
                )

                disk_indices = xp.asarray(disk)
                radial_gpts_xp = table.radial_gpts
            else:
                cutoff = table.radial_gpts[-1]
                disk_key = (chemical_symbols[number], tuple(sampling))
                if disk_key in self._sorted_disks:
                    disk, disk_radii = self._sorted_disks[disk_key]
                else:
                    disk = disk_meshgrid(int(np.ceil(cutoff / np.min(sampling))))
                    # Sort the disk pixels by physical radial distance so that the
                    # interpolation can stop at each atom's lateral cutoff.
                    disk_radii = np.hypot(
                        disk[:, 0] * sampling[0], disk[:, 1] * sampling[1]
                    )
                    order = np.argsort(disk_radii)
                    disk = np.ascontiguousarray(disk[order])
                    disk_radii = disk_radii[order]
                    self._sorted_disks[disk_key] = (disk, disk_radii)

                # A pixel at lateral distance r only receives contributions from
                # the part of the radial potential at 3D distance
                # sqrt(r ** 2 + dz ** 2), with dz the distance from the atom to the
                # slice interval. Beyond r = sqrt(cutoff ** 2 - dz ** 2) the
                # potential is below the cutoff tolerance, so those pixels are
                # skipped. The margin accounts for the atom position rounding to
                # the nearest pixel.
                dz = np.maximum(np.maximum(shifted_a, -shifted_b), 0.0)
                lateral_cutoff = np.sqrt(np.maximum(cutoff**2 - dz**2, 0.0))
                margin = np.hypot(sampling[0], sampling[1]) / 2
                disk_counts = np.searchsorted(
                    disk_radii, lateral_cutoff + margin, side="right"
                )
                # Cheap host-side reduction so the GPU kernel launcher doesn't
                # need a device sync just to size its grid; unused on CPU.
                max_disk_count = int(disk_counts.max()) if len(disk_counts) else 0

                if xp is cp:
                    device_key = disk_key + (device,)
                    cached_device_arrays = self._device_arrays.get(device_key)
                    if cached_device_arrays is None:
                        cached_device_arrays = (
                            xp.asarray(disk),
                            xp.asarray(table.radial_gpts),
                        )
                        self._device_arrays[device_key] = cached_device_arrays
                    disk_indices, radial_gpts_xp = cached_device_arrays
                else:
                    disk_indices = xp.asarray(disk)
                    radial_gpts_xp = table.radial_gpts

            radial_potential = xp.asarray(table.integrate(shifted_a, shifted_b))

            positions = xp.asarray(positions, dtype=get_dtype(complex=False))

            radial_potential_derivative = xp.zeros_like(radial_potential)
            radial_potential_derivative[:, :-1] = (
                xp.diff(radial_potential, axis=1) / xp.diff(radial_gpts_xp)[None]
            )

            if len(self._parametrization.sigmas):
                temp = xp.zeros(gpts, dtype=get_dtype(complex=False))
            else:
                temp = array

            if skew:
                interpolate_radial_functions_skew(
                    array=temp,
                    positions=positions,
                    disk_indices=disk_indices,
                    disk_counts=disk_counts,
                    sampling_vectors=sampling_vectors,
                    inv_jacobian=inv_jacobian,
                    radial_gpts=table.radial_gpts,
                    radial_functions=radial_potential,
                    radial_derivative=radial_potential_derivative,
                )
            elif xp is cp:
                interpolate_radial_functions_cuda(
                    array=temp,
                    positions=positions,
                    disk_indices=disk_indices,
                    disk_counts=xp.asarray(disk_counts),
                    sampling=sampling,
                    radial_gpts=radial_gpts_xp,
                    radial_functions=radial_potential,
                    radial_derivative=radial_potential_derivative,
                    max_disk_count=max_disk_count,
                )
            else:
                _threaded_interpolate_radial_functions(
                    array=temp,
                    positions=positions,
                    disk_indices=disk_indices,
                    disk_counts=disk_counts,
                    sampling=sampling,
                    radial_gpts=radial_gpts_xp,
                    radial_functions=radial_potential,
                    radial_derivative=radial_potential_derivative,
                )

            symbol = chemical_symbols[number]

            if symbol in self._parametrization.sigmas:
                sigma = self._parametrization.sigmas[symbol] / np.array(sampling)
                temp = get_ndimage_module(temp).gaussian_filter(
                    temp, sigma=sigma, mode="wrap"
                )
                array += temp

        return array
