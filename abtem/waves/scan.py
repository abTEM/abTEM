"""Module for describing different types of scans."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Union, Sequence, Tuple

import h5py
import numpy as np
from ase import Atom
from matplotlib.patches import Rectangle

from abtem.basic.grid import Grid, HasGridMixin


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
    def get_positions(self):
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
        return self.gpts[0],

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

    def get_positions(self) -> np.ndarray:
        start = self.margin_start
        end = self.margin_end
        x = np.linspace(start[0], end[0], self.gpts[0], endpoint=self.grid.endpoint[0])
        y = np.linspace(start[1], end[1], self.gpts[0], endpoint=self.grid.endpoint[0])
        return np.stack((np.reshape(x, (-1,)), np.reshape(y, (-1,))), axis=1)

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
                 start: Sequence[float],
                 end: Sequence[float],
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 endpoint: bool = False):

        super().__init__()

        try:
            self._start = np.array(start)[:2]
            end = np.array(end)[:2]
            assert (self._start.shape == (2,)) & (end.shape == (2,))
        except:
            raise ValueError('Scan start/end has incorrect shape')

        if (gpts is None) & (sampling is None):
            raise RuntimeError('Grid gpts or sampling must be set')

        self._grid = Grid(extent=end - start, gpts=gpts, sampling=sampling, dimensions=2, endpoint=endpoint)

    @property
    def shape(self):
        return self.gpts

    @property
    def start(self) -> np.ndarray:
        """Start corner of the scan [Å]."""
        return self._start

    @start.setter
    def start(self, start: Sequence[float]):
        self._start = np.array(start)
        self.extent = self.end - self._start

    @property
    def end(self) -> np.ndarray:
        """End corner of the scan [Å]."""
        return self.start + self.extent

    @end.setter
    def end(self, end: Sequence[float]):
        self.extent = np.array(end) - self.start

    @property
    def area(self) -> float:
        """Get the area of the scan."""
        return abs(self.start[0] - self.end[0]) * abs(self.start[1] - self.end[1])

    @property
    def axes_metadata(self):
        return [{'label': 'x', 'type': 'gridscan', 'sampling': self.sampling[0], 'offset': self.start[0]},
                {'label': 'y', 'type': 'gridscan', 'sampling': self.sampling[1], 'offset': self.start[1]}]

    def get_positions(self) -> np.ndarray:
        x = np.linspace(self.start[0], self.end[0], self.gpts[0], endpoint=self.grid.endpoint[0], dtype=np.float32)
        y = np.linspace(self.start[1], self.end[1], self.gpts[1], endpoint=self.grid.endpoint[1], dtype=np.float32)

        x, y = np.meshgrid(x, y, indexing='ij')
        return np.stack((x, y), axis=-1)

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
