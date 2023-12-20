from __future__ import annotations
import warnings
from copy import copy
from typing import Union, Sequence, Iterable, Any

import numpy as np

from abtem.core import config
from abtem.core.backend import get_array_module, xp_to_str
from abtem.core.utils import CopyMixin, EqualityMixin


def validate_gpts(gpts):
    if not all(gpts):
        raise ValueError("gpts must be greater than 0")


def adjusted_gpts(
    target_sampling: tuple[float, ...],
    old_sampling: tuple[float, ...],
    old_gpts: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    new_sampling = ()
    new_gpts = ()
    for d_target, d, n in zip(target_sampling, old_sampling, old_gpts):
        scale_factor = d / d_target
        nn = int(np.ceil(n * scale_factor))
        new_sampling += (d * n / nn,)
        new_gpts += (nn,)

    return new_sampling, new_gpts


class GridUndefinedError(Exception):
    pass


def safe_divide(a, b):
    if b == 0.0:
        return 0.0
    else:
        return a / b


class Grid(CopyMixin, EqualityMixin):
    """
    The Grid object represent the simulation grid on which the wave functions and potential are discretized.

    Parameters
    ----------
    extent : two float
        Grid extent in each dimension [Å].
    gpts : two int
        Number of grid points in each dimension.
    sampling : two float
        Grid sampling in each dimension [1 / Å].
    dimensions : int
        Number of dimensions represented by the grid.
    endpoint : bool
        If true include the grid endpoint. Default is False. For periodic grids the endpoint should not be included.
    """

    def __init__(
        self,
        extent: float | Sequence[float] = None,
        gpts: int | Sequence[int] = None,
        sampling: float | Sequence[float] = None,
        dimensions: int = 2,
        endpoint: bool | Sequence[bool] = False,
        lock_extent: bool = False,
        lock_gpts: bool = False,
        lock_sampling: bool = False,
    ):

        self._dimensions = dimensions

        if isinstance(endpoint, bool):
            endpoint = (endpoint,) * dimensions

        self._endpoint = tuple(endpoint)

        extent = self._validate(extent, dtype=float)
        gpts = self._validate(gpts, dtype=int)
        sampling = self._validate(sampling, dtype=float)

        if (
            extent is not None
            and gpts is not None
            and sampling is not None
            and config.get("warnings.overspecified-grid")
            and not np.allclose(np.array(extent) / gpts, sampling)
        ):
            warnings.warn("Overspecified grid, the provided sampling is ignored")

        # if sum([lock_extent, lock_gpts, lock_sampling]) > 1:
        #    raise RuntimeError('At most one of extent, gpts, and sampling may be locked')

        self._lock_extent = lock_extent
        self._lock_gpts = lock_gpts
        self._lock_sampling = lock_sampling

        self._extent = extent
        self._gpts = gpts
        self._sampling = sampling

        if self.extent is None:
            self._adjust_extent(self.gpts, self.sampling)

        if self.gpts is None:
            self._adjust_gpts(self.extent, self.sampling)

        if sampling is None or extent is not None:
            self._adjust_sampling(self.extent, self.gpts)

    def _validate(self, value: Any, dtype):
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) != self.dimensions:
                raise RuntimeError(
                    "Grid value length of {} != {}".format(len(value), self._dimensions)
                )
            return tuple((map(dtype, value)))

        if isinstance(value, (int, float, complex)):
            return (dtype(value),) * self.dimensions

        if value is None:
            return value

        raise RuntimeError("Invalid grid property ({})".format(value))

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
    def extent(self) -> tuple[float, ...]:
        """Grid extent in each dimension [Å]."""
        return self._extent

    @extent.setter
    def extent(self, extent: float | Sequence[float]):
        if self._lock_extent:
            raise RuntimeError("Extent cannot be modified")

        extent = self._validate(extent, dtype=float)

        if self._lock_sampling or (self.gpts is None):
            self._adjust_gpts(extent, self.sampling)
            self._adjust_sampling(extent, self.gpts)
        elif self.gpts is not None:
            self._adjust_sampling(extent, self.gpts)

        self._extent = extent

    @property
    def gpts(self) -> tuple[int, ...]:
        """Number of grid points in each dimension."""
        return self._gpts

    @gpts.setter
    def gpts(self, gpts: int | Sequence[int]):
        if self._lock_gpts:
            raise RuntimeError("Grid gpts cannot be modified")

        gpts = self._validate(gpts, dtype=int)

        if self._lock_sampling:
            self._adjust_extent(gpts, self.sampling)
        elif self.extent is not None:
            self._adjust_sampling(self.extent, gpts)
        else:
            self._adjust_extent(gpts, self.sampling)

        self._gpts = gpts

    @property
    def sampling(self) -> tuple[float, ...]:
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
    def reciprocal_space_sampling(self) -> tuple[float, float]:
        self.check_is_defined()
        return (
            1 / (self.gpts[0] * self.sampling[0]),
            1 / (self.gpts[1] * self.sampling[1]),
        )

    def _adjust_extent(self, gpts: tuple, sampling: tuple):
        if (gpts is not None) & (sampling is not None):
            self._extent = tuple(
                (n - 1) * d if e else n * d
                for n, d, e in zip(gpts, sampling, self._endpoint)
            )
            self._extent = self._validate(self._extent, float)

    def _adjust_gpts(self, extent: tuple, sampling: tuple):
        if (extent is not None) & (sampling is not None):
            self._gpts = tuple(
                int(np.ceil(r / d)) + 1 if e else int(np.ceil(r / d))
                for r, d, e in zip(extent, sampling, self._endpoint)
            )

    def _adjust_sampling(self, extent: tuple, gpts: tuple):
        if (extent is not None) & (gpts is not None):
            self._sampling = tuple(
                safe_divide(r, (n - 1)) if e else safe_divide(r, n)
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

        elif self.gpts is None:
            is_defined = False

        if raise_error and not is_defined:
            raise GridUndefinedError("grid is not defined")

        return is_defined

    def match(self, other: Grid | HasGridMixin, check_match: bool = False):
        """
        Set the parameters of this grid to match another grid.

        Parameters
        ----------
        other : Grid object
            The grid that should be matched.
        check_match : bool
            If true check whether grids can match without overriding already defined grid parameters.
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

    def check_match(self, other: Grid | HasGridMixin):
        """
        Raise error if the grid of another object is different from this object.

        Parameters
        ----------
        other : Grid object
            The grid that should be checked.
        """

        if (self.extent is not None) & (other.extent is not None):
            if not np.all(np.isclose(self.extent, other.extent)):
                raise RuntimeError(
                    "Inconsistent grid extent ({} != {})".format(
                        self.extent, other.extent
                    )
                )

        if (self.gpts is not None) & (other.gpts is not None):
            if not np.all(self.gpts == other.gpts):
                raise RuntimeError(
                    "Inconsistent grid gpts ({} != {})".format(self.gpts, other.gpts)
                )

    def round_to_power(self, powers: int | tuple[int, ...] = (2, 3, 5, 7)):
        """
        Round the grid gpts up to the nearest value that is a power of n. Fourier transforms are faster for arrays of
        whose size can be factored into small primes (2, 3, 5 and 7).

        Parameters
        ----------
        powers : int
            The gpts will be a power of this number.
        """

        if not isinstance(powers, Iterable):
            powers = (powers,)

        powers = sorted(powers)

        gpts = ()
        for n in self.gpts:
            best_n = powers[0] ** np.ceil(np.log(n) / np.log(powers[0]))
            for power in powers[1:]:
                best_n = min(power ** np.ceil(np.log(n) / np.log(power)), best_n)
            gpts += (best_n,)

        self.gpts = gpts


class HasGridMixin:
    _grid: Grid

    @property
    def grid(self) -> Grid:
        """Simulation grid."""
        return self._grid

    @property
    def extent(self) -> tuple[float] | tuple[float, float] | tuple[float, ...]:
        """Extent of grid for each dimension in Ångstrom."""
        return self.grid.extent

    @extent.setter
    def extent(self, extent: tuple[float, ...]):
        self.grid.extent = extent

    @property
    def gpts(self) -> tuple[int] | tuple[int, int] | tuple[int, ...]:
        """Number of grid points for each dimension."""
        return self.grid.gpts

    @gpts.setter
    def gpts(self, gpts: tuple[int, ...]):
        self.grid.gpts = gpts

    @property
    def sampling(self) -> tuple[float] | tuple[float, float] | tuple[float, ...]:
        """Grid sampling for each dimension in Ångstrom per grid point."""
        return self.grid.sampling

    @sampling.setter
    def sampling(self, sampling: tuple[float, ...]):
        self.grid.sampling = sampling

    @property
    def reciprocal_space_sampling(self) -> tuple[float] | tuple[float, float] | tuple[float, ...]:
        """Reciprocal-space sampling in reciprocal Ångstrom."""
        return self.grid.reciprocal_space_sampling

    def match_grid(self, other: "HasGridMixin", check_match: bool = False):
        """Match the grid to another object with a Grid."""
        self.grid.match(other, check_match=check_match)
        return self


def spatial_frequencies(
    gpts: tuple[int, ...], sampling: tuple[float, ...], return_grid: bool = False, xp=np, dtype=np.float32
):
    """
    Calculate spatial frequencies of a grid.

    Parameters
    ----------
    gpts: tuple of int
        Number of grid points.
    sampling: tuple of float
        Sampling of the potential [1 / Å].

    Returns
    -------
    tuple of arrays
    """

    xp = get_array_module(xp)

    out = ()
    for n, d in zip(gpts, sampling):
        out += (xp.fft.fftfreq(n, d).astype(dtype),)

    if return_grid:
        return xp.meshgrid(*out, indexing="ij")
    else:
        return out


def polar_spatial_frequencies(
    gpts: tuple[int, ...], sampling: tuple[float, ...], xp=np, dtype=np.float32
) -> tuple[np.ndarray, np.ndarray]:
    xp = get_array_module(xp)
    kx, ky = spatial_frequencies(gpts, sampling, False, xp_to_str(xp), dtype=dtype)
    k = xp.sqrt(kx[:, None] ** 2 + ky[None] ** 2)
    phi = xp.arctan2(ky[None], kx[:, None])
    return k, phi


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
