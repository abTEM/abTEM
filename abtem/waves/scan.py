"""Module for describing different types of scans."""
from abc import ABCMeta, abstractmethod
from copy import copy, deepcopy
from numbers import Number
from typing import Union, Sequence, Tuple

import dask
import dask.array as da
import dask.bag
import numpy as np
from ase import Atom
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from abtem.core.axes import ScanAxis, PositionsAxis
from abtem.core.grid import Grid, HasGridMixin
from abtem.core.utils import subdivide_into_chunks, generate_chunks


def validate_scan(scan, probe=None):
    if scan is None:
        scan = GridScan()

    if not hasattr(scan, 'get_positions'):
        scan = CustomScan(scan)

    if probe is not None:
        scan.match_probe(probe)

    return scan


class AbstractScan(metaclass=ABCMeta):
    """Abstract class to describe scans."""

    def __init__(self):
        pass

    def __len__(self):
        return self.num_positions

    @property
    def num_positions(self):
        return len(self.get_positions())

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """The shape the scan."""
        pass

    @property
    @abstractmethod
    def axes_metadata(self):
        pass

    @abstractmethod
    def get_positions(self, *args, **kwargs):
        """Get the scan positions as numpy array."""
        pass

    @property
    @abstractmethod
    def limits(self):
        pass

    def copy(self):
        """Make a copy."""
        return deepcopy(self)


class CustomScan(AbstractScan):

    def __init__(self, positions=None):
        if positions is None:
            positions = np.zeros((0, 2), dtype=np.float32)
        else:
            positions = np.array(positions, dtype=np.float32)

        if len(positions.shape) == 1:
            positions = positions[None]

        self._positions = positions
        super().__init__()

    def match_probe(self, probe):
        if len(self.positions) == 0:
            self._positions = np.array(probe.extent, dtype=np.float32)[None] / 2.

    @property
    def shape(self):
        return self._positions.shape[:-1]

    @property
    def positions(self):
        return self._positions

    @property
    def limits(self):
        return [(np.min(self.positions[:, 0]), np.min(self.positions[:, 1])),
                (np.max(self.positions[:, 0]), np.max(self.positions[:, 1]))]

    def divide(self, num_chunks):
        return [CustomScan(self._positions)]

    def get_positions(self, chunks: int = None, lazy: bool = False) -> np.ndarray:
        if chunks is None:
            chunks = (len(self._positions), 2)
        else:
            chunks = (min(chunks, len(self._positions)), 2)

        if lazy:
            return da.from_array(self._positions, chunks=chunks)
        else:
            return self._positions

    def generate_positions(self, chunks):
        # if isinstance(chunks, Number):
        #    chunks = (int(np.floor(np.sqrt(chunks))),) * 2

        # positions = self.get_positions(lazy=False)

        if len(self.positions.shape) > 1:
            yield (slice(0, 1),), self._positions
        else:
            yield (), self._positions

    @property
    def axes_metadata(self):
        return [PositionsAxis() for i in range(len(self.positions.shape) - 1)]


def linescan_positions(start, end, gpts, endpoint):
    x = np.linspace(start[0], end[0], gpts, endpoint=endpoint, dtype=np.float32)
    y = np.linspace(start[1], end[1], gpts, endpoint=endpoint, dtype=np.float32)
    return np.stack((np.reshape(x, (-1,)), np.reshape(y, (-1,))), axis=1)


class LineScan(AbstractScan):
    """
    Line scan object.

    Defines a scan along a straight line.

    Parameters
    ----------
    start : two float
        Start point of the scan [Å].
    end : two float
        End point of the scan [Å].
    gpts: int
        Number of scan positions.
    sampling: float
        Sampling rate of scan positions [1 / Å].
    endpoint: bool
        If True, end is the last position. Otherwise, it is not included. Default is True.
    """

    def __init__(self,
                 start: Union[Tuple[float, float], None] = (0., 0.),
                 end: Union[Tuple[float, float], None] = None,
                 gpts: int = None,
                 sampling: float = None,
                 endpoint: bool = True):

        super().__init__()
        self._gpts = gpts
        self._sampling = sampling

        self._start = start if start is None else tuple(start)
        self._end = end if end is None else tuple(end)

        if self.start is not None and self.end is not None:
            if np.allclose(self._start, self._end):
                raise RuntimeError('line scan start and end is identical')

        self._endpoint = endpoint
        self._adjust_gpts()
        self._adjust_sampling()

    @classmethod
    def at_position(cls,
                    position: Union[Tuple[float, float], Atom],
                    extent: float = 1.,
                    angle: float = 0.,
                    gpts: int = None,
                    sampling: float = None,
                    endpoint: bool = True):

        if isinstance(position, Atom):
            position = (position.x, position.y)

        direction = np.array((np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))))

        start = tuple(np.array(position) - extent / 2 * direction)
        end = tuple(np.array(position) + extent / 2 * direction)
        return cls(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)

    def match_probe(self, probe):
        if self.start is None:
            self.start = (0., 0.)

        if self.end is None and probe.extent is not None:
            self.end = (0., probe.extent[1])

        if self.sampling is None:
            self.sampling = .9 * probe.ctf.nyquist_sampling

    @property
    def extent(self) -> Union[float, None]:
        if self._start is None or self._end is None:
            return None

        return np.linalg.norm(np.array(self._end) - np.array(self._start))

    def _adjust_gpts(self):
        if self.extent is None or self.sampling is None:
            return

        self._gpts = int(np.ceil(self.extent / self.sampling))

        self._adjust_sampling()

    def _adjust_sampling(self):

        if self.extent is None or self.gpts is None:
            return

        self._sampling = self.extent / self.gpts

    @property
    def endpoint(self) -> bool:
        return self._endpoint

    @property
    def limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self.start, self.end

    @property
    def gpts(self) -> int:
        return self._gpts

    @gpts.setter
    def gpts(self, gpts: int):
        self._gpts = gpts
        self._adjust_sampling()

    @property
    def sampling(self) -> float:
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: float):
        self._sampling = sampling
        self._adjust_gpts()

    @property
    def shape(self) -> Tuple[int]:
        return (self._gpts,)

    @property
    def axes_metadata(self):
        return [ScanAxis(label='x', sampling=self.sampling, units='Å', start=self.start, end=self.end)]

    @property
    def start(self) -> Union[Tuple[float, float], None]:
        """
        Start point of the scan [Å].
        """
        return self._start

    @start.setter
    def start(self, start: Tuple[float, float]):
        self._start = start
        self._adjust_gpts()

    @property
    def end(self) -> Union[Tuple[float, float], None]:
        """
        End point of the scan [Å].
        """
        return self._end

    @end.setter
    def end(self, end: Tuple[float, float]):
        self._end = end
        self._adjust_gpts()

    def get_positions(self, chunks: int = None, lazy: bool = False) -> np.ndarray:

        if chunks is None:
            return linescan_positions(self.start, self.end, self.gpts, self.endpoint)

        chunks = (chunks,)

        if lazy:
            positions = dask.delayed(linescan_positions)(self.start, self.end, self.gpts, self.endpoint)
            positions = da.from_delayed(positions, shape=(self.gpts, 2), dtype=np.float32)
            positions = positions.rechunk(chunks + (2,))
        else:
            positions = linescan_positions(self.start, self.end, self.gpts, self.endpoint)

        return positions

    def add_to_plot(self, ax: Axes, linestyle: str = '-', color: str = 'r', **kwargs):
        """
        Add a visualization of a scan line to a matplotlib plot.

        Parameters
        ----------
        ax : matplotlib Axes
            The axes of the matplotlib plot the visualization should be added to.
        linestyle : str, optional
            Linestyle of scan line. Default is '-'.
        color : str, optional
            Color of the scan line. Default is 'r'.
        kwargs :
            Additional options for matplotlib.pyplot.plot as keyword arguments.
        """
        ax.plot([self.start[0], self.end[0]], [self.start[1], self.end[1]], linestyle=linestyle, color=color, **kwargs)


def split_array_2d(array, chunks):
    return [np.split(p, np.cumsum(chunks[1][:-1]), axis=1) for p in np.split(array, np.cumsum(chunks[0][:-1]), axis=0)]


def gridscan_positions(start, end, gpts, endpoint):
    x = np.linspace(start[0], end[0], gpts[0], endpoint=endpoint[0], dtype=np.float32)
    y = np.linspace(start[1], end[1], gpts[1], endpoint=endpoint[1], dtype=np.float32)
    x, y = np.meshgrid(x, y, indexing='ij')
    return np.stack((x, y), axis=-1)


class GridScan(HasGridMixin, AbstractScan):
    """
    Grid scan object.

    Defines a scan on a regular grid.

    Parameters
    ----------
    start : two float
        Start corner of the scan [Å].
    end : two float
        End corner of the scan [Å].
    gpts : two int
        Number of scan positions in the x- and y-direction of the scan.
    sampling : two float
        Sampling rate of scan positions [1 / Å].
    endpoint : bool
        If True, end is the last position. Otherwise, it is not included. Default is False.
    """

    def __init__(self,
                 start: Tuple[float, float] = None,
                 end: Tuple[float, float] = None,
                 gpts: Union[int, Tuple[int, int]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 endpoint: Union[bool, Tuple[bool, bool]] = False):

        super().__init__()

        if (start is None) and (end is None):
            self._start = None
            self._end = None
            extent = None
        else:
            try:
                self._start = np.array(start)[:2]
                end = np.array(end)[:2]
                assert (self._start.shape == (2,)) & (end.shape == (2,))
            except AssertionError:
                raise ValueError('Scan start/end has incorrect shape')

            extent = end - start

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling, dimensions=2, endpoint=endpoint)

    @property
    def limits(self):
        return [self.start, self.end]

    @property
    def endpoint(self) -> Tuple[bool, bool]:
        return self.grid.endpoint[0], self.grid.endpoint[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.gpts

    @property
    def start(self) -> Union[np.ndarray, None]:
        """Start corner of the scan [Å]."""
        return self._start

    @start.setter
    def start(self, start: Sequence[float]):
        self._start = np.array(start)
        if self.end is not None:
            self.extent = self.end - self._start

    def match_probe(self, probe):
        if self.start is None:
            self.start = (0., 0.)

        if self.end is None:
            self.end = probe.extent

        if self.sampling is None:
            self.sampling = .9 * probe.ctf.nyquist_sampling

    @property
    def end(self) -> Union[np.ndarray, None]:
        """End corner of the scan [Å]."""
        if self.extent is None:
            return

        return self.start + self.extent

    @end.setter
    def end(self, end: Sequence[float]):
        if self.start is not None:
            self.extent = np.array(end) - self.start

    @property
    def area(self) -> float:
        """Get the area of the scan."""
        return abs(self.start[0] - self.end[0]) * abs(self.start[1] - self.end[1])

    @property
    def axes_metadata(self):
        return [
            ScanAxis(label='x', sampling=self.sampling[0], offset=self.start[0], units='Å', endpoint=self.endpoint[0]),
            ScanAxis(label='y', sampling=self.sampling[1], offset=self.start[1], units='Å', endpoint=self.endpoint[1])]

    def generate_positions(self, chunks):
        if isinstance(chunks, Number):
            chunks = (int(np.floor(np.sqrt(chunks))),) * 2

        positions = self.get_positions(lazy=False)

        for start_x, end_x in generate_chunks(positions.shape[0], chunks=chunks[0]):
            for start_y, end_y in generate_chunks(positions.shape[1], chunks=chunks[1]):
                slice_x = slice(start_x, end_x)
                slice_y = slice(start_y, end_y)
                yield (slice_x, slice_y), positions[slice_x, slice_y]

    def get_positions(self, chunks: Union[int, Tuple[int, int]] = None, lazy: bool = False) -> np.ndarray:

        if isinstance(chunks, Number):
            chunks = (int(np.floor(np.sqrt(chunks))),) * 2

        if chunks is not None:
            chunks = (subdivide_into_chunks(self.gpts[0], chunks=chunks[0]),
                      subdivide_into_chunks(self.gpts[1], chunks=chunks[1]), 2)

            assert len(chunks) == 3

        if lazy:
            positions = dask.delayed(gridscan_positions)(self.start, self.end, self.gpts, self.grid.endpoint)
            positions = da.from_delayed(positions, shape=self.gpts + (2,), dtype=np.float32)
            if chunks:
                positions = positions.rechunk(chunks)
        else:
            positions = gridscan_positions(self.start, self.end, self.gpts, self.grid.endpoint)
            if chunks:
                positions = split_array_2d(positions, chunks)

        return positions

    def divide(self, divisions: Union[int, Tuple[int, int]]):
        """
        Partition the scan into smaller grid scans

        Parameters
        ----------
        divisions : two int
            The number of partitions to create in x and y.

        Returns
        -------
        List of GridScan objects
        """

        if isinstance(divisions, Number):
            divisions = (int(np.round(np.sqrt(divisions))),) * 2

        Nx = subdivide_into_chunks(self.gpts[0], divisions[0])
        Ny = subdivide_into_chunks(self.gpts[1], divisions[1])
        Sx = np.concatenate(([0], np.cumsum(Nx)))
        Sy = np.concatenate(([0], np.cumsum(Ny)))

        scans = []
        for i, nx in enumerate(Nx):
            inner_scans = []
            for j, ny in enumerate(Ny):
                start = (Sx[i] * self.sampling[0], Sy[j] * self.sampling[1])
                end = (start[0] + nx * self.sampling[0], start[1] + ny * self.sampling[1])
                endpoint = False

                if i + 1 == divisions[0]:
                    endpoint = self.grid.endpoint[0]
                    if endpoint:
                        end[0] -= self.sampling[0]

                if j + 1 == divisions[1]:
                    endpoint = self.grid.endpoint[1]
                    if endpoint:
                        end[1] -= self.sampling[1]

                scan = self.__class__(start, end, gpts=(nx, ny), endpoint=endpoint)

                inner_scans.append(scan)
            scans.append(inner_scans)
        return scans

    def add_to_plot(self, ax, alpha: float = .33, facecolor: str = 'r', edgecolor: str = 'r', **kwargs):
        """
        Add a visualization of the scan area to a matplotlib plot.

        Parameters
        ----------
        ax : matplotlib Axes
            The axes of the matplotlib plot the visualization should be added to.
        alpha : float, optional
            Transparency of the scan area visualization. Default is 0.33.
        facecolor : str, optional
            Color of the scan area visualization.
        edgecolor : str, optional
            Color of the edge of the scan area visualization.
        kwargs :
            Additional options for matplotlib.patches.Rectangle used for scan area visualization as keyword arguments.
        """
        rect = Rectangle(tuple(self.start), *self.extent, alpha=alpha, facecolor=facecolor, edgecolor=edgecolor,
                         **kwargs)
        ax.add_patch(rect)
