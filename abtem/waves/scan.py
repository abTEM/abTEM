"""Module for describing different types of scans."""
from abc import ABCMeta, abstractmethod
from copy import copy
from numbers import Number
from typing import Union, Sequence, Tuple

import dask
import dask.array as da
import dask.bag
import numpy as np
from ase import Atom
from matplotlib.patches import Rectangle

from abtem.basic.grid import Grid, HasGridMixin
from abtem.basic.utils import subdivide_into_chunks


class AbstractScan(metaclass=ABCMeta):
    """Abstract class to describe scans."""

    def __init__(self):
        self._batches = None

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

    @abstractmethod
    def __copy__(self):
        pass

    def copy(self):
        """Make a copy."""
        return copy(self)


class LineScan(AbstractScan, HasGridMixin):
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
                 start: Union[Tuple[float, float], Atom],
                 end: Union[Tuple[float, float], Atom] = None,
                 angle: float = 0.,
                 gpts: int = None,
                 sampling: float = None,
                 margin: float = 0.,
                 endpoint: bool = True):

        super().__init__()

        if isinstance(start, Atom):
            start = (start.x, start.y)

        if isinstance(end, Atom):
            end = (end.x, end.y)

        # if (end is not None) & (angle is not None):
        #    raise ValueError('only one of "end" and "angle" may be specified')

        if (gpts is None) & (sampling is None):
            raise RuntimeError('grid gpts or sampling must be set')

        self._grid = Grid(gpts=gpts, sampling=sampling, endpoint=endpoint, dimensions=1)

        self._start = start[:2]
        self._margin = margin

        if end is not None:
            self._set_direction_and_extent(self._start, end[:2])
        else:
            self.angle = angle
            self.extent = 2 * self._margin

    def _set_direction_and_extent(self, start: Tuple[float, float], end: Tuple[float, float]):
        difference = np.array(end) - np.array(start)
        extent = np.linalg.norm(difference, axis=0)
        self._direction = difference / extent
        extent = extent + 2 * self._margin
        if extent == 0.:
            raise RuntimeError('scan has no extent')
        self.extent = extent

    @property
    def shape(self) -> Tuple[int]:
        return self.gpts

    @property
    def axes_metadata(self):
        return [{'label': 'x', 'type': 'linescan', 'sampling': self.sampling[0], 'start_x': self.start[0],
                 'start_y': self.start[1], 'end_x': self.end[0], 'end_y': self.end[1]}]

    @property
    def start(self) -> Tuple[float, float]:
        """
        Start point of the scan [Å].
        """
        return self._start

    @start.setter
    def start(self, start: Tuple[float, float]):
        self._start = start
        self._set_direction_and_extent(self._start, self.end)

    @property
    def end(self) -> Tuple[float, float]:
        """
        End point of the scan [Å].
        """
        return (self.start[0] + self.direction[0] * self.extent[0] - self.direction[0] * 2 * self._margin,
                self.start[1] + self.direction[1] * self.extent[0] - self.direction[1] * 2 * self._margin)

    @end.setter
    def end(self, end: Tuple[float, float]):
        self._set_direction_and_extent(self.start, end)

    @property
    def angle(self) -> float:
        """
        End point of the scan [Å].
        """
        return np.arctan2(self._direction[0], self._direction[1])

    @angle.setter
    def angle(self, angle: float):
        self._direction = (np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)))

    @property
    def direction(self) -> Tuple[float, float]:
        """Direction of the scan line."""
        return self._direction

    @property
    def margin(self) -> float:
        return self._margin

    @property
    def margin_start(self) -> Tuple[float, float]:
        return self.start[0] - self.direction[0] * self.margin, self.start[1] - self.direction[1] * self.margin

    @property
    def margin_end(self) -> Tuple[float, float]:
        return self.end[0] + self.direction[0] * self.margin, self.end[1] + self.direction[1] * self.margin

    def get_positions(self, chunks: int = None, lazy: bool = False) -> np.ndarray:
        def linescan_positions(start, end, gpts, endpoint):
            x = np.linspace(start[0], end[0], gpts[0], endpoint=endpoint[0], dtype=np.float32)
            y = np.linspace(start[1], end[1], gpts[0], endpoint=endpoint[0], dtype=np.float32)

            return np.stack((np.reshape(x, (-1,)), np.reshape(y, (-1,))), axis=1)

        if chunks is None:
            return linescan_positions(self.start, self.end, self.gpts, self.grid.endpoint)

        chunks = (chunks,)

        if lazy:
            positions = dask.delayed(linescan_positions)(self.start, self.end, self.gpts, self.grid.endpoint)
            positions = da.from_delayed(positions, shape=self.gpts + (2,), dtype=np.float32)
            positions = positions.rechunk(chunks + (2,))
        else:
            positions = linescan_positions(self.start, self.end, self.gpts, self.grid.endpoint)

        return positions

    def add_to_plot(self, ax, linestyle: str = '-', color: str = 'r', **kwargs):
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
        start = self.margin_start
        end = self.margin_end
        ax.plot([start[0], end[0]], [start[1], end[1]], linestyle=linestyle, color=color, **kwargs)

    def __copy__(self):
        return self.__class__(start=self.start, end=self.end, gpts=self.gpts, endpoint=self.grid.endpoint[0])


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
    batch_partition : 'squares' or 'lines'
        Specify how to split the scan into batches. If 'squares', the scan position batches are divided into the best
        matching squares for the batch size. If 'lines', the batches are divided into lines of scan positions.
    measurement_shift : two int
        The insertion indices of new measurements will be shifted by this amount in x and y. This is used for
        correctly inserting measurements collected from a partitioned scan.
    """

    def __init__(self,
                 start: Sequence[float] = None,
                 end: Sequence[float] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 endpoint: bool = False):

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
            except:
                raise ValueError('Scan start/end has incorrect shape')

            if (gpts is None) & (sampling is None):
                raise RuntimeError('Grid gpts or sampling must be set')

            extent = end - start

        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling, dimensions=2, endpoint=endpoint)

    @property
    def shape(self):
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

    def match(self, probe):
        if self.extent is None:
            self.start = (0., 0.)
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
        return [{'label': 'x', 'type': 'gridscan', 'sampling': float(self.sampling[0]), 'offset': float(self.start[0]),
                 'units': 'Å'},
                {'label': 'y', 'type': 'gridscan', 'sampling': float(self.sampling[1]), 'offset': float(self.start[1]),
                 'units': 'Å'}]

    def get_positions(self, chunks=None, lazy=False) -> np.ndarray:
        def gridscan_positions(start, end, gpts, endpoint):
            x = np.linspace(start[0], end[0], gpts[0], endpoint=endpoint[0], dtype=np.float32)
            y = np.linspace(start[1], end[1], gpts[1], endpoint=endpoint[1], dtype=np.float32)
            x, y = np.meshgrid(x, y, indexing='ij')
            return np.stack((x, y), axis=-1)

        if chunks is None:
            return gridscan_positions(self.start, self.end, self.gpts, self.grid.endpoint)

        if isinstance(chunks, Number):
            chunks = (int(np.floor(np.sqrt(chunks))),) * 2

        chunks = (subdivide_into_chunks(self.gpts[0], chunks=chunks[0]),
                  subdivide_into_chunks(self.gpts[1], chunks=chunks[1]), 2)

        if lazy:
            positions = dask.delayed(gridscan_positions)(self.start, self.end, self.gpts, self.grid.endpoint)
            positions = da.from_delayed(positions, shape=self.gpts + (2,), dtype=np.float32)
            positions = positions.rechunk(chunks)
        else:
            positions = gridscan_positions(self.start, self.end, self.gpts, self.grid.endpoint)
            positions = [np.split(p, np.cumsum(chunks[1][:-1]), axis=1)
                         for p in np.split(positions, np.cumsum(chunks[0][:-1]), axis=0)]

        return positions

    def divide(self, partitions: Sequence[int]):
        """
        Partition the scan into smaller grid scans

        Parameters
        ----------
        partitions : two int
            The number of partitions to create in x and y.

        Returns
        -------
        List of GridScan objects
        """

        Nx = subdivide_into_chunks(self.gpts[0], partitions[0])
        Ny = subdivide_into_chunks(self.gpts[1], partitions[1])
        Sx = np.concatenate(([0], np.cumsum(Nx)))
        Sy = np.concatenate(([0], np.cumsum(Ny)))

        scans = []
        for i, nx in enumerate(Nx):
            inner_scans = []
            for j, ny in enumerate(Ny):
                start = [Sx[i] * self.sampling[0], Sy[j] * self.sampling[1]]
                end = [start[0] + nx * self.sampling[0], start[1] + ny * self.sampling[1]]
                endpoint = False

                if i + 1 == partitions[0]:
                    endpoint = self.grid.endpoint[0]
                    if endpoint:
                        end[0] -= self.sampling[0]

                if j + 1 == partitions[1]:
                    endpoint = self.grid.endpoint[1]
                    if endpoint:
                        end[1] -= self.sampling[1]

                scan = self.__class__(start,
                                      end,
                                      gpts=(nx, ny),
                                      endpoint=endpoint)

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

    def __copy__(self):
        return self.__class__(start=self.start,
                              end=self.end,
                              gpts=self.gpts,
                              endpoint=self.grid.endpoint)


class HasScanMixin:
    _scan: Union[AbstractScan, None]

    @property
    def scan(self):
        return self._scan

    @property
    def scan_sampling(self):
        if self._scan is None:
            return None
        else:
            return self.scan.sampling
