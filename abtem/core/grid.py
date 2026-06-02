"""Module for the Grid class and related functions."""

from __future__ import annotations

import warnings
from types import ModuleType
from typing import Callable, Iterable, Optional, Sequence, TypeVar

import dask.array as da
import numpy as np

from abtem.core import config
from abtem.core.backend import device_name_from_array_module, get_array_module
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype


def validate_gpts(gpts: tuple[int, ...]) -> tuple[int, ...]:
    """
    Ensure that the prodived grid points are valid.

    Parameters
    ----------
    gpts : tuple of int
        The tuple of integers representing the GPTs (General Purpose Tokens).

    Returns
    -------
    tuple of int
        The validated tuple of integers representing the GPTs.

    Raises
    ------
    ValueError
        If any value in the gpts tuple is not greater than 0.
    """
    gpts = tuple(gpts)

    if not all(n > 0 for n in gpts):
        raise ValueError("gpts must be greater than 0")

    return gpts


def adjusted_gpts(
    target_sampling: tuple[float, ...],
    old_sampling: tuple[float, ...],
    old_gpts: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """
    Adjust the number of grid points to match a target sampling.

    Parameters
    ----------
    target_sampling : tuple of float
        The target sampling [Å].
    old_sampling : tuple of float
        The old sampling [Å].
    old_gpts : tuple of int
        The old number of grid points.

    Returns
    -------
    tuple of float
        The new sampling [Å].
    """
    new_sampling = tuple(
        d * n / int(np.ceil(n * (d / d_target)))
        for d_target, d, n in zip(target_sampling, old_sampling, old_gpts)
    )
    new_gpts = tuple(
        int(np.ceil(n * (d / d_target)))
        for d_target, d, n in zip(target_sampling, old_sampling, old_gpts)
    )
    return new_sampling, new_gpts


class GridUndefinedError(Exception):
    """
    Exception raised when the grid is not defined.
    """


T = TypeVar("T", int, float)
U = TypeVar("U")


class Grid(CopyMixin, EqualityMixin):
    """
    The Grid object represent the simulation grid on which the wave functions and
    potential are discretized.

    Parameters
    ----------
    extent : two float
        Grid extent in each dimension [Å].
    gpts : two int
        Number of grid points in each dimension.
    sampling : two float
        Grid sampling in each dimension [Å].
    dimensions : int
        Number of dimensions represented by the grid.
    endpoint : bool
        If true include the grid endpoint. Default is False. For periodic grids the
        endpoint should not be included.
    lock_extent : bool
        If true the extent cannot be modified. Default is False.
    lock_gpts : bool
        If true the gpts cannot be modified. Default is False.
    lock_sampling : bool
        If true the sampling cannot be modified. Default is False.
    cell : np.ndarray, optional
        Optional 2x2 real-space cell whose rows are the two in-plane lattice vectors,
        allowing a non-orthogonal (skewed) grid. If ``None`` (default) the grid is
        orthogonal (axis-aligned) and behaviour is unchanged. Only valid for a
        two-dimensional grid. When given, the row lengths should match ``extent``.
    """

    def __init__(
        self,
        extent: Optional[float | Sequence[float]] = None,
        gpts: Optional[int | Sequence[int]] = None,
        sampling: Optional[float | Sequence[float]] = None,
        dimensions: int = 2,
        endpoint: bool | Sequence[bool] = False,
        lock_extent: bool = False,
        lock_gpts: bool = False,
        lock_sampling: bool = False,
        cell: Optional[np.ndarray] = None,
    ):
        self._dimensions = dimensions

        if isinstance(endpoint, bool):
            endpoint = (endpoint,) * dimensions

        self._endpoint = tuple(endpoint)

        self._extent = self._validate(extent, dtype=float)
        self._gpts = self._validate(gpts, dtype=int)
        self._sampling = self._validate(sampling, dtype=float)

        if (
            self._extent is not None
            and self._gpts is not None
            and self._sampling is not None
            and config.get("warnings.overspecified-grid")
            and not np.allclose(np.array(self._extent) / self._gpts, self._sampling)
        ):
            warnings.warn("Overspecified grid, the provided sampling is ignored")

        self._lock_extent = lock_extent
        self._lock_gpts = lock_gpts
        self._lock_sampling = lock_sampling

        if self.extent is None:
            self._adjust_extent(self.gpts, self.sampling)

        if self.gpts is None:
            self._adjust_gpts(self.extent, self.sampling)

        if sampling is None or extent is not None:
            self._adjust_sampling(self.extent, self.gpts)

        self._cell = self._validate_cell(cell)

    def _validate_cell(self, cell):
        if cell is None:
            return None

        if self._dimensions != 2:
            raise ValueError("a non-orthogonal cell is only supported for 2D grids")

        cell = np.array(cell, dtype=float)
        if cell.shape != (2, 2):
            raise ValueError(f"cell must be a 2x2 array, got shape {cell.shape}")

        if self.extent is not None:
            lengths = np.linalg.norm(cell, axis=1)
            if not np.allclose(lengths, self.extent, rtol=1e-6):
                raise ValueError(
                    f"cell row lengths {tuple(lengths)} are inconsistent with the grid "
                    f"extent {self.extent}"
                )

        # store as a nested tuple so equality/copy behave (None vs ndarray is brittle)
        return tuple(map(tuple, cell))

    def _validate(
        self, value: Optional[T | Sequence[T]], dtype: Callable[[T], U]
    ) -> Optional[tuple[U, ...]]:
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) != self.dimensions:
                raise RuntimeError(
                    f"Grid value length of {len(value)} != {self._dimensions}"
                )
            return tuple((map(dtype, value)))

        if isinstance(value, (int, float)):
            return (dtype(value),) * self.dimensions

        if value is None:
            return value

        raise RuntimeError(f"Invalid grid property ({value})")

    def __len__(self) -> int:
        return self.dimensions

    @property
    def endpoint(self) -> tuple[bool] | tuple[bool, bool] | tuple[bool, ...]:
        """Include the grid endpoint."""
        return self._endpoint

    @property
    def dimensions(self) -> int:
        """Number of dimensions represented by the grid."""
        return self._dimensions

    @property
    def extent(self) -> tuple[float, ...] | None:
        """Grid extent in each dimension [Å]."""
        return self._extent

    @extent.setter
    def extent(self, extent: float | Sequence[float] | None):
        if extent is not None:
            if (
                self._lock_extent
                and self.extent is not None
                and not np.allclose(extent, self.extent)
            ):
                raise RuntimeError("Extent cannot be modified")

            validated_extent = self._validate(extent, dtype=float)

            if self._lock_sampling or (self.gpts is None):
                self._adjust_gpts(validated_extent, self.sampling)
                self._adjust_sampling(validated_extent, self.gpts)
            elif self.gpts is not None:
                self._adjust_sampling(validated_extent, self.gpts)
        else:
            validated_extent = None

        self._extent = validated_extent

    @property
    def gpts(self) -> tuple[int, ...] | None:
        """Number of grid points in each dimension."""
        return self._gpts

    @gpts.setter
    def gpts(self, gpts: int | Sequence[int]):
        if self._lock_gpts:
            raise RuntimeError("Grid gpts cannot be modified")

        validated_gpts = self._validate(gpts, dtype=int)

        if self._lock_sampling:
            self._adjust_extent(validated_gpts, self.sampling)
        elif self.extent is not None:
            self._adjust_sampling(self.extent, validated_gpts)
        else:
            self._adjust_extent(validated_gpts, self.sampling)

        self._gpts = validated_gpts

    @property
    def sampling(self) -> tuple[float, ...] | None:
        """Grid sampling in each dimension [Å]."""
        return self._sampling

    @sampling.setter
    def sampling(self, sampling):
        if self._lock_sampling:
            raise RuntimeError("Sampling cannot be modified")

        sampling = self._validate(sampling, dtype=float)

        if self._lock_gpts:
            self._adjust_extent(self.gpts, sampling)
        elif self.extent is not None:
            self._adjust_gpts(self.extent, sampling)
        else:
            self._adjust_extent(self.gpts, sampling)

        if self.extent is None or self.gpts is None:
            self._sampling = sampling
        else:
            self._adjust_sampling(self.extent, self.gpts)

    @property
    def reciprocal_space_sampling(self) -> tuple[float, ...]:
        """Reciprocal-space sampling [1/Å]."""
        self.check_is_defined()
        assert (
            self.sampling is not None
            and self.gpts is not None
            and self.extent is not None
        )
        return tuple(1 / (n * d) for n, d in zip(self.gpts, self.sampling))

    @property
    def cell(self) -> np.ndarray | None:
        """Real-space cell (2x2, rows are the in-plane lattice vectors). ``None`` for
        an orthogonal (axis-aligned) grid."""
        if self._cell is None:
            return None
        return np.array(self._cell, dtype=float)

    @property
    def is_orthogonal(self) -> bool:
        """Whether the grid is orthogonal (axis-aligned)."""
        if self._cell is None:
            return True
        return bool(
            abs(self._cell[0][1]) < 1e-12 and abs(self._cell[1][0]) < 1e-12
        )

    def _effective_cell(self) -> np.ndarray:
        """The 2x2 cell, falling back to ``diag(extent)`` for an orthogonal grid."""
        if self._cell is not None:
            return np.array(self._cell, dtype=float)
        return np.diag(np.array(self._valid_extent, dtype=float))

    @property
    def reciprocal_metric(self) -> np.ndarray:
        """Reciprocal metric tensor ``M = (C C^T)^-1`` of the 2D cell ``C``, with
        ``M[i, j] = b_i . b_j`` [1/Å^2]. The squared length of the Fourier component
        with integer indices ``(h1, h2)`` is ``[h1 h2] M [h1 h2]^T``."""
        cell = self._effective_cell()
        return np.linalg.inv(cell @ cell.T)

    def k_squared(self, xp=np) -> np.ndarray:
        """Squared spatial frequency ``|g|^2`` of every Fourier component [1/Å^2].

        Uses the reciprocal metric for a skewed cell; for an orthogonal grid this is
        the sum of the squared per-axis spatial frequencies (identical to
        ``spatial_frequencies``)."""
        self.check_is_defined()
        xp = get_array_module(xp)
        gpts = self._valid_gpts
        dtype = get_dtype(complex=False)

        if self.is_orthogonal:
            freqs = spatial_frequencies(gpts, self._valid_sampling, xp=xp)
            k2 = xp.zeros(gpts, dtype=dtype)
            for i, k in enumerate(freqs):
                shape = [1] * len(gpts)
                shape[i] = -1
                k2 = k2 + xp.reshape(k, shape).astype(dtype) ** 2
            return k2

        metric = self.reciprocal_metric
        h1, h2 = _integer_frequencies(gpts, xp)
        h1, h2 = h1[:, None].astype(dtype), h2[None, :].astype(dtype)
        return (
            metric[0, 0] * h1**2
            + 2.0 * metric[0, 1] * h1 * h2
            + metric[1, 1] * h2**2
        ).astype(dtype)

    def k_components(self, xp=np) -> tuple[np.ndarray, np.ndarray]:
        """Physical reciprocal-space components ``(gx, gy)`` of every Fourier component
        [1/Å]. ``gx**2 + gy**2`` equals :meth:`k_squared`. Two-dimensional grids only."""
        self.check_is_defined()
        xp = get_array_module(xp)
        if self._dimensions != 2:
            raise ValueError("k_components is only defined for 2D grids")

        if self.is_orthogonal:
            return spatial_frequencies(
                self._valid_gpts, self._valid_sampling, return_grid=True, xp=xp
            )

        dtype = get_dtype(complex=False)
        reciprocal = np.linalg.inv(self._effective_cell()).T  # rows b1, b2
        h1, h2 = _integer_frequencies(self._valid_gpts, xp)
        h1, h2 = h1[:, None].astype(dtype), h2[None, :].astype(dtype)
        gx = h1 * reciprocal[0, 0] + h2 * reciprocal[1, 0]
        gy = h1 * reciprocal[0, 1] + h2 * reciprocal[1, 1]
        return gx, gy

    def polar_spatial_frequencies(self, xp=np) -> tuple[np.ndarray, np.ndarray]:
        """Physical polar spatial frequencies ``(k, phi)`` [1/Å, rad] of every Fourier
        component. For a skewed cell these use the reciprocal metric; for an orthogonal
        grid this reduces exactly to :func:`polar_spatial_frequencies`. 2D grids only."""
        self.check_is_defined()
        xp = get_array_module(xp)
        if self.is_orthogonal:
            return polar_spatial_frequencies(
                self._valid_gpts, self._valid_sampling, xp=xp
            )
        gx, gy = self.k_components(xp=xp)
        k = xp.sqrt(gx**2 + gy**2)
        phi = xp.arctan2(gy, gx)
        return k, phi

    def _adjust_extent(
        self, gpts: tuple[int, ...] | None, sampling: tuple[float, ...] | None
    ):
        if gpts is not None and sampling is not None:
            self._extent = tuple(
                (n - 1) * d if e else n * d
                for n, d, e in zip(gpts, sampling, self._endpoint)
            )
            self._extent = self._validate(self._extent, float)

    def _adjust_gpts(
        self, extent: tuple[float, ...] | None, sampling: tuple[float, ...] | None
    ):
        if extent is not None and sampling is not None:
            self._gpts = tuple(
                int(np.ceil(r / d)) + 1 if e else int(np.ceil(r / d))
                for r, d, e in zip(extent, sampling, self._endpoint)
            )

    def _adjust_sampling(
        self, extent: tuple[float, ...] | None, gpts: tuple[int, ...] | None
    ):
        def _safe_divide(a: float, b: float) -> float:
            if b == 0.0:
                return 0.0
            else:
                return a / b

        if extent is not None and gpts is not None:
            self._sampling = tuple(
                _safe_divide(r, (n - 1)) if e else _safe_divide(r, n)
                for r, n, e in zip(extent, gpts, self._endpoint)
            )
            self._sampling = self._validate(self._sampling, float)

    def check_is_defined(self, raise_error: bool = True):
        """
        Raise error if the grid is not defined.
        """
        is_defined = True
        if self.extent is None:
            is_defined = False

        elif self.gpts is None:
            is_defined = False

        if raise_error and not is_defined:
            raise GridUndefinedError("grid is not defined")

        return is_defined

    def match(self, other: Grid | HasGrid2DMixin, check_match: bool = False):
        """
        Set the parameters of this grid to match another grid.

        Parameters
        ----------
        other : Grid object
            The grid that should be matched.
        check_match : bool
            If true check whether grids can match without overriding already defined
            grid parameters.
        """

        if check_match:
            self.check_match(other)

        # if (self.extent is None) & (other.extent is None):
        #    raise RuntimeError('Grid extent cannot be inferred')

        if other.extent is None:
            other.extent = self.extent
        elif np.any(
            np.array(self.extent, np.float32) != np.array(other.extent, np.float32)
        ):
            self.extent = other.extent

        # if (self.gpts is None) & (other.gpts is None):
        #    raise RuntimeError('Grid gpts cannot be inferred')

        if other.gpts is None:
            other.gpts = self.gpts
        elif np.any(self.gpts != other.gpts):
            self.gpts = other.gpts

        if other.sampling is None:
            other.sampling = self.sampling
        elif not np.allclose(
            np.array(self.sampling, np.float32), np.array(other.sampling, np.float32)
        ):
            self.sampling = other.sampling

        # propagate a non-orthogonal cell between matched grids
        other_grid = other if isinstance(other, Grid) else getattr(other, "grid", other)
        other_cell = getattr(other_grid, "_cell", None)
        if other_cell is not None and self._cell is None:
            self._cell = other_cell
        elif other_cell is None and self._cell is not None:
            other_grid._cell = self._cell

    def check_match(self, other: Grid | HasGrid2DMixin):
        """
        Raise error if the grid of another object is different from this object.

        Parameters
        ----------
        other : Grid object
            The grid that should be checked.
        """

        if self.extent is not None and other.extent is not None:
            if not np.all(np.isclose(self.extent, other.extent)):
                raise RuntimeError(
                    f"Inconsistent grid extent ({self.extent} != {other.extent})"
                )

        if self.gpts is not None and other.gpts is not None:
            if not np.all(self.gpts == other.gpts):
                raise RuntimeError(
                    f"Inconsistent grid gpts ({self.gpts} != {other.gpts})"
                )

    def round_to_power(
        self, powers: Optional[int | list[int]] = None
    ) -> tuple[int, ...]:
        """
        Round the grid gpts up to the nearest value that is a power of n. Fourier
        transforms are faster for arrays of whose size can be factored into small primes
        (2, 3, 5 and 7).

        Parameters
        ----------
        powers : int
            The gpts will be a power of this number.
        """
        if powers is None:
            powers = [2, 3, 5, 7]

        elif not isinstance(powers, Iterable):
            powers = [powers]

        powers = sorted(powers)

        assert self.gpts is not None

        gpts = tuple(
            int(min(power ** np.ceil(np.log(n) / np.log(power)) for power in powers))
            for n in self.gpts
        )

        self.gpts = gpts

        return gpts

    @property
    def _valid_extent(self) -> tuple[float, ...]:
        if self.extent is None:
            raise GridUndefinedError("Grid extent is not defined")
        return self.extent

    @property
    def _valid_gpts(self) -> tuple[int, ...]:
        if self.gpts is None:
            raise GridUndefinedError("Grid gpts is not defined")
        return self.gpts

    @property
    def _valid_sampling(self) -> tuple[float, ...]:
        if self.sampling is None:
            raise GridUndefinedError("Grid sampling is not defined")
        return self.sampling

    def spatial_frequencies(self):
        return spatial_frequencies(self.gpts, self.sampling, False)


# class HasGridMixin:
#     """
#     Mixin class for objects that have a Grid.
#     """

#     _grid: Grid

#     @property
#     def grid(self) -> Grid:
#         """Simulation grid."""
#         return self._grid

#     def match_grid(self, other: HasGridMixin, check_match: bool = False):
#         """Match the grid to another object with a Grid."""
#         self.grid.match(other, check_match=check_match)
#         return self

#     @property
#     def extent(self) -> tuple[float, ...] | None:
#         """Extent of grid for each dimension in Ångstrom."""
#         return self.grid.extent

#     @extent.setter
#     def extent(self, extent: tuple[float, ...] | None):
#         self.grid.extent = extent

#     @property
#     def gpts(self) -> tuple[int, ...] | None:
#         """Number of grid points for each dimension."""
#         return self.grid.gpts

#     @gpts.setter
#     def gpts(self, gpts: tuple[int, ...]):
#         self.grid.gpts = gpts

#     @property
#     def sampling(self) -> tuple[float, ...] | None:
#         """Grid sampling for each dimension in Ångstrom per grid point."""
#         return self.grid.sampling

#     @sampling.setter
#     def sampling(self, sampling: tuple[float, ...]):
#         self.grid.sampling = sampling

#     @property
#     def reciprocal_space_sampling(self) -> tuple[float, ...]:
#         """Reciprocal-space sampling in reciprocal Ångstrom."""
#         return self.grid.reciprocal_space_sampling


class HasGrid2DMixin:
    _grid: Grid

    def match_grid(self, other: HasGrid2DMixin, check_match: bool = False):
        """Match the grid to another object with a Grid."""
        self.grid.match(other, check_match=check_match)
        return self

    @property
    def grid(self) -> Grid:
        """Simulation grid."""
        return self._grid

    @property
    def extent(self) -> tuple[float, float] | None:
        """Extent of grid for each dimension in Ångstrom."""
        extent = self.grid.extent
        if extent is not None:
            assert len(extent) == 2
        return extent

    @extent.setter
    def extent(self, extent: tuple[float, float] | None):
        self.grid.extent = extent

    @property
    def _valid_extent(self) -> tuple[float, float]:
        if self.extent is None:
            raise GridUndefinedError("Grid extent is not defined")
        return self.extent

    @property
    def gpts(self) -> tuple[int, int] | None:
        """Number of grid points for each dimension."""
        gpts = self.grid.gpts
        if gpts is not None:
            assert len(gpts) == 2
        return gpts

    @gpts.setter
    def gpts(self, gpts: tuple[int, int]):
        self.grid.gpts = gpts

    @property
    def _valid_gpts(self) -> tuple[int, int]:
        if self.gpts is None:
            raise GridUndefinedError("Grid gpts is not defined")
        return self.gpts

    @property
    def sampling(self) -> tuple[float, float] | None:
        """Grid sampling for each dimension in Ångstrom per grid point."""
        sampling = self.grid.sampling
        if sampling is not None:
            assert len(sampling) == 2
        return sampling

    @sampling.setter
    def sampling(self, sampling: tuple[float, float]):
        self.grid.sampling = sampling

    @property
    def _valid_sampling(self) -> tuple[float, float]:
        if self.sampling is None:
            raise GridUndefinedError("Grid sampling is not defined")
        return self.sampling

    @property
    def reciprocal_space_sampling(self) -> tuple[float, float]:
        """Reciprocal-space sampling in reciprocal Ångstrom."""
        k = self.grid.reciprocal_space_sampling
        assert len(k) == 2
        return k

    @property
    def cell(self) -> np.ndarray | None:
        """Real-space cell (2x2, rows are the in-plane lattice vectors), or ``None``
        for an orthogonal (axis-aligned) grid."""
        return self.grid.cell


def _integer_frequencies(gpts: tuple[int, int], xp=np):
    """Wrapped integer Fourier indices ``(h1, h2)`` matching ``fft2`` order.

    ``rint(n * fftfreq(n))`` reproduces ``[0, 1, ..., n//2-1, -n//2, ..., -1]`` exactly
    (rounding avoids a 94.999...->94 truncation at high frequencies)."""
    n1, n2 = gpts
    h1 = xp.rint(n1 * xp.fft.fftfreq(n1))
    h2 = xp.rint(n2 * xp.fft.fftfreq(n2))
    return h1, h2


def spatial_frequencies(
    gpts: tuple[int, ...],
    sampling: tuple[float, ...],
    return_grid: bool = False,
    xp: ModuleType | np.ndarray | da.core.Array | str | None = np,
):
    """
    Return the spatial frequencies of a grid.

    Parameters
    ----------
    gpts : tuple of int
        Number of grid points.
    sampling : tuple of float
        Sampling of the grid [Å].
    return_grid : bool
        If True, return the grid as a single meshgrid array.
    xp : module
        Array module to use, options are numpy or cupy. Default is numpy.

    Returns
    -------
    spatial_frequencies : tuple of np.ndarray
        Tuple of spatial frequencies in each dimension.
    spatial_frequencies_grid : np.ndarray
        If return_grid is True, the spatial frequencies as a single meshgrid array.
    """
    dtype = get_dtype(complex=False)

    xp = get_array_module(xp)

    out = tuple(xp.fft.fftfreq(n, d).astype(dtype) for n, d in zip(gpts, sampling))

    if return_grid:
        return xp.meshgrid(*out, indexing="ij")
    else:
        return out


def real_space_grid(gpts, extent, xp=np):
    out = tuple(xp.linspace(0, L, n, endpoint=False) for n, L in zip(gpts, extent))
    return xp.meshgrid(*out, indexing="ij")


def polar_spatial_frequencies(
    gpts: tuple[int, ...],
    sampling: tuple[float, ...],
    xp: ModuleType | np.ndarray | da.core.Array | str | None = np,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the polar spatial frequencies of a grid.

    Parameters
    ----------
    gpts : tuple of int
        Number of grid points.
    sampling : tuple of float
        Sampling of the potential [1 / Å].
    xp : module
        Array module to use, options are numpy or cupy. Default is numpy.

    Returns
    -------
    k_and_phi : tuple of np.ndarray
        Tuple of spatial frequencies in polar coordinates. First element is the radial
        frequency and the second element is the azimuthal angle.
    """
    xp = get_array_module(xp)
    kx, ky = spatial_frequencies(
        gpts, sampling, False, device_name_from_array_module(xp)
    )
    k = xp.sqrt(kx[:, None] ** 2 + ky[None] ** 2)
    phi = xp.arctan2(ky[None], kx[:, None])
    return k, phi


def coordinate_grid(
    extent: tuple[float, ...],
    gpts: tuple[int, ...],
    origin: tuple[float, ...],
    endpoint: bool = True,
) -> tuple[np.ndarray, ...]:
    coordinates = [
        np.linspace(0, r, n, endpoint=endpoint) - o
        for r, n, o in zip(extent, gpts, origin)
    ]
    return np.meshgrid(*coordinates, indexing="ij")


def disk_meshgrid(r: int) -> np.ndarray:
    """
    Return all indices inside a disk with a given radius.

    Parameters
    ----------
    r : int
        Radius of disc in pixels.

    Returns
    -------
    disc_indices : np.ndarray
    """
    cols = np.zeros((2 * r + 1, 2 * r + 1)).astype(np.int32)
    cols[:] = np.linspace(0, 2 * r, 2 * r + 1) - r
    rows = cols.T
    inside = (rows**2 + cols**2) <= r**2
    return np.array((rows[inside], cols[inside])).T
