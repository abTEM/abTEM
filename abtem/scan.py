from abc import ABCMeta, abstractmethod
from typing import Union, Sequence

import cupy as cp
import h5py
import numpy as np
from matplotlib.patches import Rectangle

from abtem.bases import Grid, HasGridMixin
from abtem.measure import Calibration
from abtem.utils import split_integer


class AbstractScan(metaclass=ABCMeta):

    def __init__(self, fractional=False):
        self._batches = None
        self._fractional = fractional

    def check_is_fractional(self):
        raise RuntimeError()

    @abstractmethod
    def to_absolute(self, grid):
        pass

    @property
    def __len__(self):
        return self.num_positions

    @property
    @abstractmethod
    def num_positions(self):
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def calibrations(self) -> tuple:
        pass

    @abstractmethod
    def get_positions(self):
        pass

    @abstractmethod
    def insert_new_measurement(self, measurement_key, start, end, new_values):
        pass

    def generate_positions(self, max_batch=1):
        positions = self.get_positions()
        self._partition_batches(max_batch)

        while len(self._batches) > 0:
            start, end = self.get_next_batch()
            yield start, end, positions[start:end]

    @property
    def batches(self):
        return self._batches

    def get_next_batch(self):
        return self._batches.pop(0)

    def _partition_batches(self, max_batch):
        n = len(self)
        n_batches = (n + (-n % max_batch)) // max_batch
        batch_sizes = split_integer(len(self), n_batches)
        self._batches = [(0, batch_sizes[0])]
        for batch_size in batch_sizes[1:]:
            self._batches.append((self._batches[-1][-1], self._batches[-1][-1] + batch_size))

    @abstractmethod
    def __copy__(self):
        pass


class CustomScan(AbstractScan):
    def __init__(self, positions):
        self._positions = positions
        super().__init__()

    @property
    def num_measurements(self):
        return len(self._positions)

    def get_positions(self):
        return self._positions


class LineScan(AbstractScan):

    def __init__(self, start: Sequence[float], end: Sequence[float],
                 num_steps: Union[int, Sequence[int]] = None,
                 step_size: Union[float, Sequence[float]] = None, endpoint: bool = True):

        super().__init__()

        start = np.array(start)
        end = np.array(end)

        if (start.shape != (2,)) | (end.shape != (2,)):
            raise ValueError('scan start/end has wrong shape')

        self._grid = Grid(gpts=num_steps, sampling=step_size, endpoint=endpoint, dimensions=1)
        self._start = start
        self._direction, self.extent = self._direction_and_extent(start, end)

    def _direction_and_extent(self, start, end):
        extent = np.linalg.norm((end - start), axis=0)
        direction = (end - start) / extent
        return direction, extent

    @property
    def num_steps(self):
        return self._grid.gpts[0]

    @property
    def shape(self):
        return self._grid.gpts

    @property
    def calibrations(self) -> tuple:
        return (Calibration(offset=0, sampling=self._grid.sampling, units='Å', name='x'),)

    @property
    def start(self) -> np.ndarray:
        return self._start

    @start.setter
    def start(self, start: np.ndarray):
        self._start = np.array(start)
        self._direction, self.extent = self._direction_and_extent(self._start, self.end)

    @property
    def end(self) -> np.ndarray:
        return self.start + self.direction * self.extent

    @end.setter
    def end(self, end: np.ndarray):
        self._direction, self.extent = self._direction_and_extent(self.start, end)

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    def insert_new_measurement(self, measurement, start, end, new_measurement):
        if isinstance(measurement, str):
            with h5py.File(measurement, 'a') as f:
                f['array'][start:end] = cp.asnumpy(new_measurement)

        else:
            measurement.array[start:end] = cp.asnumpy(new_measurement)

    def get_positions(self) -> np.ndarray:
        x = np.linspace(self.start[0], self.start[0] + np.array(self.extent) * self.direction[0], self._grid.gpts[0],
                        endpoint=self._grid.endpoint)
        y = np.linspace(self.start[1], self.start[1] + np.array(self.extent) * self.direction[1], self._grid.gpts[0],
                        endpoint=self._grid.endpoint)
        return np.stack((np.reshape(x, (-1,)), np.reshape(y, (-1,))), axis=1)

    def add_to_mpl_plot(self, ax, linestyle='-', color='r', **kwargs):
        ax.plot([self.start[0], self.end[0]], [self.start[1], self.end[1]], linestyle=linestyle, color=color, **kwargs)

    def __copy__(self):
        return self.__class__(start=self.start, end=self.end, gpts=self._grid.gpts, endpoint=self._grid.endpoint)


def unravel_slice_2d(start, end, shape):
    slices = []
    rows = []
    slices_1d = []
    n = 0
    n_accum = 0
    for index in range(start, end):
        index_in_row = index % shape[-1]
        n += 1
        if index_in_row == shape[-1] - 1:
            slices_1d.append(slice(n_accum, n_accum + n))
            slices.append(slice(index_in_row - n + 1, index_in_row + 1))
            rows.append(index // shape[-1])
            n_accum += n
            n = 0
    if n > 0:
        slices_1d.append(slice(n_accum, n_accum + n))
        slices.append(slice(index_in_row - n + 1, index_in_row + 1))
        rows.append(index // shape[-1])
    return rows, slices, slices_1d


class GridScan(AbstractScan):

    def __init__(self, start, end, num_steps=None, step_size=None, endpoint=False, fractional=False):
        super().__init__()

        self._start = np.array(start)
        self._end = np.array(end)
        self._num_steps = num_steps
        self._step_size = step_size

        if (self._start.shape != (2,)) | (end.shape != (2,)):
            raise ValueError('scan start/end has wrong shape')

        super().__init__(fractional=fractional)

    @property
    def extent(self):
        return self._

    @property
    def num_steps(self):
        return Grid()

    @property
    def step_size(self):
        return self._step_size

    @property
    def calibrations(self) -> tuple:
        return (Calibration(offset=0, sampling=self._grid.sampling[0], units='Å', name='x'),
                Calibration(offset=0, sampling=self._grid.sampling[1], units='Å', name='y'))

    @property
    def start(self) -> np.ndarray:
        return self._start

    @start.setter
    def start(self, start: Sequence[float]):
        self._start = np.array(start)
        self.extent = self.end - self._start

    @property
    def end(self) -> np.ndarray:
        return self.start + self.extent

    @end.setter
    def end(self, end: Sequence[float]):
        self.extent = np.array(end) - self.start

    def get_scan_area(self):
        height = abs(self.start[0] - self.end[0])
        width = abs(self.start[1] - self.end[1])
        return height * width

    def get_positions(self, grid=None) -> np.ndarray:

        x = np.linspace(self.start[0], self.end[0], self._grid.gpts[0], endpoint=self._grid.endpoint)
        y = np.linspace(self.start[1], self.end[1], self._grid.gpts[1], endpoint=self._grid.endpoint)
        x, y = np.meshgrid(x, y, indexing='ij')
        return np.stack((np.reshape(x, (-1,)),
                         np.reshape(y, (-1,))), axis=1)

    def insert_new_measurement(self, measurement, start, end, new_measurement):
        for row, slic, slic_1d in zip(*unravel_slice_2d(start, end, self.shape)):
            if isinstance(measurement, str):
                with h5py.File(measurement, 'a') as f:
                    f['array'][row, slic] = cp.asnumpy(new_measurement[slic_1d])
            else:
                measurement.array[row, slic] = cp.asnumpy(new_measurement[slic_1d])

    def add_to_mpl_plot(self, ax, alpha=.33, facecolor='r', edgecolor='r', **kwargs):
        rect = Rectangle(tuple(self.start), *self.extent, alpha=alpha, facecolor=facecolor, edgecolor=edgecolor,
                         **kwargs)
        ax.add_patch(rect)

    def __copy__(self):
        return self.__class__(start=self.start, end=self.end, num_steps=self.num_steps, endpoint=self._grid.endpoint)
