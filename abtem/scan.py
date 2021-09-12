"""Module for describing different types of scans."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Union, Sequence, Tuple, List

import h5py
import numpy as np
from matplotlib.patches import Rectangle

from abtem.base_classes import Grid, HasGridMixin
from abtem.device import asnumpy
from abtem.measure import Calibration, Measurement
from abtem.utils import subdivide_into_batches, ProgressBar
from ase import Atom


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
    def calibrations(self) -> tuple:
        """The measurement calibrations associated with the scan."""
        pass

    @abstractmethod
    def get_positions(self):
        """Get the scan positions as numpy array."""
        pass

    @abstractmethod
    def insert_new_measurement(self, measurement, indices, new_values):
        """
        Insert new measurement values into a Measurement object or HDF5 file.

        Parameters
        ----------
        measurement : Measurement object
            The measurement to which new values are inserted.
        start : int
            First index of slice.
        end : int
            Last index of slice.
        new_values : ndarray
            New measurement values to be inserted. Length should be (end - start).
        """
        pass

    def generate_positions(self, max_batch, pbar=False):
        positions = self.get_positions()
        self._partition_batches(max_batch)

        if pbar:
            pbar = ProgressBar(total=len(self))

        for i in range(len(self._batches)):
            indices = self.get_next_batch()
            yield indices, positions[indices]

            if pbar:
                pbar.update(len(indices))

        if pbar:
            pbar.close()

    def get_next_batch(self):
        return self._batches.pop(0)

    def _partition_batches(self, max_batch):
        n = len(self)
        n_batches = (n + (-n % max_batch)) // max_batch
        batch_sizes = subdivide_into_batches(len(self), n_batches)

        self._batches = []

        start = 0
        for batch_size in batch_sizes:
            end = start + batch_size
            indices = np.arange(start, end, dtype=np.int)
            start += batch_size
            self._batches.append(indices)

    @abstractmethod
    def __copy__(self):
        pass

    def copy(self):
        """Make a copy."""
        return copy(self)


class PositionScan(AbstractScan):
    """
    Position scan object.

    Defines a scan based on user-provided positions.

    Parameters
    ----------
    positions : list
        A list of xy scan positions [Å].
    """

    def __init__(self, positions: np.ndarray):

        self._positions = np.array(positions)

        if (len(self._positions.shape) != 2) or (self._positions.shape[1] != 2):
            raise RuntimeError('The shape of the sequence of positions must be (n, 2).')

        super().__init__()

    @property
    def shape(self) -> tuple:
        return len(self),

    @property
    def calibrations(self) -> tuple:
        return None,

    def insert_new_measurement(self, measurement, indices, new_measurement):
        if isinstance(measurement, str):
            with h5py.File(measurement, 'a') as f:
                f['array'][indices] = asnumpy(new_measurement)

        else:
            measurement.array[indices] = asnumpy(new_measurement)

    def get_positions(self):
        return self._positions

    def add_to_mpl_plot(self, ax, marker: str = 'o', color: str = 'r', **kwargs):
        """
        Add a visualization of the scan positions to a matplotlib plot.

        Parameters
        ----------
        ax: matplotlib Axes
            The axes of the matplotlib plot the visualization should be added to.
        marker: str, optional
            Style of scan position markers. Default is '-'.
        color: str, optional
            Color of the scan position markers. Default is 'r'.
        kwargs:
            Additional options for matplotlib.pyplot.plot as keyword arguments.
        """
        ax.plot(*self.get_positions().T, marker=marker, linestyle='', color=color, **kwargs)

    def __copy__(self):
        return self.__class__(self._positions.copy())


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

        #if (end is not None) & (angle is not None):
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
    def calibrations(self) -> Tuple[Calibration]:
        return Calibration(offset=0, sampling=self.sampling[0], units='Å', name='x', endpoint=self.grid.endpoint[0]),

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

    def insert_new_measurement(self,
                               measurement: Measurement,
                               indices,
                               new_measurement_values: np.ndarray):

        if isinstance(measurement, str):
            with h5py.File(measurement, 'a') as f:
                f['array'][indices] += asnumpy(new_measurement_values)

        else:
            measurement.array[indices] += asnumpy(new_measurement_values)

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

    def add_to_mpl_plot(self, ax, linestyle: str = '-', color: str = 'r', **kwargs):
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


class GridScan(AbstractScan, HasGridMixin):
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
                 endpoint: bool = False,
                 batch_partition: str = 'squares',
                 measurement_shift: Sequence[int] = None):

        super().__init__()

        try:
            self._start = np.array(start)[:2]
            end = np.array(end)[:2]
            assert (self._start.shape == (2,)) & (end.shape == (2,))
        except:
            raise ValueError('Scan start/end has incorrect shape')

        if (gpts is None) & (sampling is None):
            raise RuntimeError('Grid gpts or sampling must be set')

        if not batch_partition.lower() in ['squares', 'lines']:
            raise ValueError('batch partition must be "squares" or "lines"')

        self._batch_partition = batch_partition
        self._measurement_shift = measurement_shift

        self._grid = Grid(extent=end - start, gpts=gpts, sampling=sampling, dimensions=2, endpoint=endpoint)

    @property
    def shape(self):
        return self.gpts

    @property
    def calibrations(self) -> tuple:
        return (Calibration(offset=self.start[0], sampling=self.sampling[0], units='Å', name='x',
                            endpoint=self.grid.endpoint[0]),
                Calibration(offset=self.start[1], sampling=self.sampling[1], units='Å', name='y',
                            endpoint=self.grid.endpoint[1]))

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

    def get_scan_area(self) -> float:
        """Get the area of the scan."""
        height = abs(self.start[0] - self.end[0])
        width = abs(self.start[1] - self.end[1])
        return height * width

    def get_positions(self) -> np.ndarray:
        x = np.linspace(self.start[0], self.end[0], self.gpts[0], endpoint=self.grid.endpoint[0])
        y = np.linspace(self.start[1], self.end[1], self.gpts[1], endpoint=self.grid.endpoint[1])
        x, y = np.meshgrid(x, y, indexing='ij')
        return np.stack((np.reshape(x, (-1,)),
                         np.reshape(y, (-1,))), axis=1)

    def insert_new_measurement(self, measurement, indices: np.ndarray, new_measurement: np.ndarray):
        x, y = np.unravel_index(indices, self.shape)

        if self._measurement_shift is not None:
            x += self._measurement_shift[0]
            y += self._measurement_shift[1]

        if isinstance(measurement, str):
            with h5py.File(measurement, 'a') as f:
                for unique in np.unique(x):
                    f['array'][unique, y[unique == x]] += asnumpy(new_measurement[unique == x])
        else:
            measurement.array[x, y] += asnumpy(new_measurement)

    def partition_scan(self, partitions: Sequence[int]) -> List['GridScan']:
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
        Nx = subdivide_into_batches(self.gpts[0], partitions[0])
        Ny = subdivide_into_batches(self.gpts[1], partitions[1])
        Sx = np.concatenate(([0], np.cumsum(Nx)))
        Sy = np.concatenate(([0], np.cumsum(Ny)))

        scans = []
        for i, nx in enumerate(Nx):
            for j, ny in enumerate(Ny):
                start = [Sx[i] * self.sampling[0], Sy[j] * self.sampling[1]]
                end = [start[0] + nx * self.sampling[0], start[1] + ny * self.sampling[1]]
                endpoint = [False, False]

                if i + 1 == partitions[0]:
                    endpoint[0] = self.grid.endpoint[0]
                    if endpoint[0]:
                        end[0] -= self.sampling[0]

                if j + 1 == partitions[1]:
                    endpoint[1] = self.grid.endpoint[1]
                    if endpoint[1]:
                        end[1] -= self.sampling[1]

                scan = self.__class__(start,
                                      end,
                                      gpts=(nx, ny),
                                      endpoint=endpoint,
                                      batch_partition='squares',
                                      measurement_shift=(Sx[i], Sy[j]))

                scans.append(scan)
        return scans

    def _partition_batches(self, max_batch: int):
        if self._batch_partition == 'lines':
            super()._partition_batches(max_batch)
            return

        if max_batch == 1:
            self._batches = [[i] for i in range(len(self))]
            return

        max_batch_x = int(np.floor(np.sqrt(max_batch)))
        max_batch_y = int(np.floor(np.sqrt(max_batch)))

        Nx = subdivide_into_batches(self.gpts[0], (self.gpts[0] + (-self.gpts[0] % max_batch_x)) // max_batch_x)
        Ny = subdivide_into_batches(self.gpts[1], (self.gpts[1] + (-self.gpts[1] % max_batch_y)) // max_batch_y)

        self._batches = []
        Sx = np.concatenate(([0], np.cumsum(Nx)))
        Sy = np.concatenate(([0], np.cumsum(Ny)))

        for i, nx in enumerate(Nx):
            for j, ny in enumerate(Ny):
                x = np.arange(Sx[i], Sx[i] + nx, dtype=np.int)
                y = np.arange(Sy[j], Sy[j] + ny, dtype=np.int)
                self._batches.append((y[None] + x[:, None] * self.gpts[1]).ravel())

    def add_to_mpl_plot(self, ax, alpha: float = .33, facecolor: str = 'r', edgecolor: str = 'r', **kwargs):
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
                              endpoint=self.grid.endpoint,
                              batch_partition=self._batch_partition,
                              measurement_shift=self._measurement_shift)
