from copy import copy
from typing import Union, Sequence, Tuple

import numpy as np

from abtem.core.backend import get_array_module, xp_to_str
from abtem.core.events import Events, HasEventsMixin, watch


class GridUndefinedError(Exception):
    pass


class Grid(HasEventsMixin):
    """
    Grid object.

    The grid object represent the simulation grid on which the wave functions and potential are discretized.

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

    def __init__(self,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 dimensions: int = 2,
                 endpoint: Union[bool, Sequence[bool]] = False,
                 lock_extent: bool = False,
                 lock_gpts: bool = False,
                 lock_sampling: bool = False):

        self._dimensions = dimensions

        if isinstance(endpoint, bool):
            endpoint = (endpoint,) * dimensions

        self._endpoint = tuple(endpoint)

        # if sum([lock_extent, lock_gpts, lock_sampling]) > 1:
        #    raise RuntimeError('At most one of extent, gpts, and sampling may be locked')

        self._lock_extent = lock_extent
        self._lock_gpts = lock_gpts
        self._lock_sampling = lock_sampling

        self._extent = self._validate(extent, dtype=float)
        self._gpts = self._validate(gpts, dtype=int)
        self._sampling = self._validate(sampling, dtype=float)

        if self.extent is None:
            self._adjust_extent(self.gpts, self.sampling)

        if self.gpts is None:
            self._adjust_gpts(self.extent, self.sampling)

        if sampling is None or extent is not None:
            self._adjust_sampling(self.extent, self.gpts)

        self._events = Events()

    def _validate(self, value, dtype):
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) != self.dimensions:
                raise RuntimeError('Grid value length of {} != {}'.format(len(value), self._dimensions))
            return tuple((map(dtype, value)))

        if isinstance(value, (int, float, complex)):
            return (dtype(value),) * self.dimensions

        if value is None:
            return value

        raise RuntimeError('Invalid grid property ({})'.format(value))

    def __len__(self) -> int:
        return self.dimensions

    @property
    def endpoint(self) -> Tuple[bool, ...]:
        """Include the grid endpoint."""
        return self._endpoint

    @property
    def dimensions(self) -> int:
        """Number of dimensions represented by the grid."""
        return self._dimensions

    @property
    def extent(self) -> Tuple[float, ...]:
        """Grid extent in each dimension [Å]."""
        return self._extent

    @extent.setter
    @watch
    def extent(self, extent: Union[float, Sequence[float]]):
        if self._lock_extent:
            raise RuntimeError('Extent cannot be modified')

        extent = self._validate(extent, dtype=float)

        if self._lock_sampling or (self.gpts is None):
            self._adjust_gpts(extent, self.sampling)
            self._adjust_sampling(extent, self.gpts)
        elif self.gpts is not None:
            self._adjust_sampling(extent, self.gpts)

        self._extent = extent

    @property
    def gpts(self) -> Tuple[int, ...]:
        """Number of grid points in each dimension."""
        return self._gpts

    @gpts.setter
    @watch
    def gpts(self, gpts: Union[int, Sequence[int]]):
        if self._lock_gpts:
            raise RuntimeError('Grid gpts cannot be modified')

        gpts = self._validate(gpts, dtype=int)

        if self._lock_sampling:
            self._adjust_extent(gpts, self.sampling)
        elif self.extent is not None:
            self._adjust_sampling(self.extent, gpts)
        else:
            self._adjust_extent(gpts, self.sampling)

        self._gpts = gpts

    @property
    def sampling(self) -> Tuple[float, ...]:
        """Grid sampling in each dimension [1 / Å]."""
        return self._sampling

    @sampling.setter
    @watch
    def sampling(self, sampling):
        if self._lock_sampling:
            raise RuntimeError('Sampling cannot be modified')

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
    def fourier_space_sampling(self) -> Tuple[float, float]:
        self.check_is_defined()
        return 1 / (self.gpts[0] * self.sampling[0]), 1 / (self.gpts[1] * self.sampling[1])

    def _adjust_extent(self, gpts: tuple, sampling: tuple):
        if (gpts is not None) & (sampling is not None):
            self._extent = tuple((n - 1) * d if e else n * d for n, d, e in zip(gpts, sampling, self._endpoint))
            self._extent = self._validate(self._extent, float)

    def _adjust_gpts(self, extent: tuple, sampling: tuple):
        if (extent is not None) & (sampling is not None):
            self._gpts = tuple(int(np.ceil(r / d)) + 1 if e else int(np.ceil(r / d))
                               for r, d, e in zip(extent, sampling, self._endpoint))

    def _adjust_sampling(self, extent: tuple, gpts: tuple):
        if (extent is not None) & (gpts is not None):
            self._sampling = tuple(r / (n - 1) if e else r / n for r, n, e in zip(extent, gpts, self._endpoint))
            self._sampling = self._validate(self._sampling, float)

    def check_is_defined(self):
        """
        Raise error if the grid is not defined.
        """

        if self.extent is None:
            raise GridUndefinedError('grid extent is not defined')

        elif self.gpts is None:
            raise GridUndefinedError('grid gpts are not defined')

        elif self.gpts is None:
            raise GridUndefinedError('grid sampling is not defined')

    def match(self, other: Union['Grid', 'HasGridMixin'], check_match: bool = False):
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
        elif np.any(np.array(self.extent, np.float32) != np.array(other.extent, np.float32)):
            self.extent = other.extent

        # if (self.gpts is None) & (other.gpts is None):
        #    raise RuntimeError('Grid gpts cannot be inferred')

        if other.gpts is None:
            other.gpts = self.gpts
        elif np.any(self.gpts != other.gpts):
            self.gpts = other.gpts

        if other.sampling is None:
            other.sampling = self.sampling
        elif np.any(np.array(self.sampling, np.float32) != np.array(other.sampling, np.float32)):
            self.sampling = other.sampling

    def check_match(self, other):
        """
        Raise error if the grid of another object is different from this object.

        Parameters
        ----------
        other : Grid object
            The grid that should be checked.
        """

        if (self.extent is not None) & (other.extent is not None):
            if not np.all(np.isclose(self.extent, other.extent)):
                raise RuntimeError('Inconsistent grid extent ({} != {})'.format(self.extent, other.extent))

        if (self.gpts is not None) & (other.gpts is not None):
            if not np.all(self.gpts == other.gpts):
                raise RuntimeError('Inconsistent grid gpts ({} != {})'.format(self.gpts, other.gpts))

    def __eq__(self, other):
        self.check_match(other)

    def round_to_power(self, power: int = 2):
        """
        Round the grid gpts up to the nearest value that is a power of n. Fourier transforms are faster for arrays of
        whose size can be factored into small primes (2, 3, 5 and 7).

        Parameters
        ----------
        power : int
            The gpts will be a power of this number.
        """

        self.gpts = tuple(power ** np.ceil(np.log(n) / np.log(power)) for n in self.gpts)


class HasGridMixin:
    _grid: Grid

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def extent(self):
        return self.grid.extent

    @extent.setter
    def extent(self, extent):
        self.grid.extent = extent

    @property
    def gpts(self):
        return self.grid.gpts

    @gpts.setter
    def gpts(self, gpts):
        self.grid.gpts = gpts

    @property
    def sampling(self):
        return self.grid.sampling

    @sampling.setter
    def sampling(self, sampling):
        self.grid.sampling = sampling

    @property
    def fourier_space_sampling(self):
        return self.grid.fourier_space_sampling

    def match_grid(self, other, check_match=False):
        self.grid.match(other, check_match=check_match)
        return self


def spatial_frequencies(gpts: Tuple[int, ...],
                        sampling: Tuple[float, ...],
                        return_grid: bool = False,
                        xp=np):
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
        out += (xp.fft.fftfreq(n, d).astype(np.float32),)

    if return_grid:
        return xp.meshgrid(*out, indexing='ij')
    else:
        return out

def polar_spatial_frequencies(gpts, sampling, xp=np):
    xp = get_array_module(xp)
    kx, ky = spatial_frequencies(gpts, sampling, False, xp_to_str(xp))
    k = xp.sqrt(kx[:, None] ** 2 + ky[None] ** 2)
    phi = xp.arctan2(kx[:, None], ky[None])
    return k, phi


def disc_meshgrid(r):
    """Internal function to return all indices inside a disk with a given radius."""
    cols = np.zeros((2 * r + 1, 2 * r + 1)).astype(np.int32)
    cols[:] = np.linspace(0, 2 * r, 2 * r + 1) - r
    rows = cols.T
    inside = (rows ** 2 + cols ** 2) <= r ** 2
    return np.array((rows[inside], cols[inside])).T
