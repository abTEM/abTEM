from contextlib import ExitStack
from typing import Union, Sequence

import h5py
import numpy as np
from ase import Atoms
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

from abtem.bases import Grid, ArrayWithGrid1D, ArrayWithGrid2D
from abtem.potentials import Potential
from abtem.utils import split_integer
from collections.abc import Iterable
import cupy as cp


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


class ScanBase:

    def __init__(self, **kwargs):
        self._data = None
        self._partitions = None
        super().__init__(**kwargs)

    def __len__(self):
        return self.num_measurements

    @property
    def num_measurements(self):
        raise NotImplementedError()

    @property
    def shape(self):
        raise NotImplementedError()

    def get_all_positions(self):
        raise NotImplementedError()

    def get_next_partition(self):
        return self._partitions.pop(0)

    def generate_positions(self, max_batch):
        positions = self.get_all_positions()
        self._partition_positions(max_batch)

        while len(self._partitions) > 0:
            start, end = self.get_next_partition()
            yield start, end, positions[start:end]

    def allocate_measurements(self, detectors):
        raise NotImplementedError()

    def _partition_positions(self, max_batch):
        n = len(self)
        n_batches = (n + (-n % max_batch)) // max_batch
        batch_sizes = split_integer(len(self), n_batches)
        self._partitions = [(0, batch_sizes[0])]
        for batch_size in batch_sizes[1:]:
            self._partitions.append((self._partitions[-1][-1], self._partitions[-1][-1] + batch_size))

    def insert_new_measurement(self, measurement, start, end, new_measurement):
        raise NotImplementedError()

    def go(self, max_batch, show_progress=False):
        if not isinstance(detectors, Iterable):
            detectors = [detectors]

        self._detectors = detectors

        measurements = self.allocate()
        jobs = self._create_jobs(max_batch)
        positions = self.get_positions()

        with tqdm(total=len(self)) if show_progress else ExitStack() as pbar:
            while jobs:
                start, end = jobs.pop(0)
                exit_probes = self._probe.multislice_at(positions[start:end], self._potential)

                for detector, measurement in measurements.items():
                    self.insert_new_measurement(measurement, start, end, detector.detect(exit_probes))

                if show_progress:
                    pbar.update(end - start)

        return measurements

    def finalise_measurements(self):
        raise NotImplementedError()

    @property
    def measurements(self):
        return self._measurements


class CustomScan(ScanBase):
    def __init__(self, probe, potential, positions, detectors=None):
        ScanBase.__init__(self, probe=probe, potential=potential, detectors=detectors)
        self._positions = positions

    @property
    def num_measurements(self):
        return len(self._positions)

    def get_positions(self):
        return self._positions


class LineScan(Grid, ScanBase):

    def __init__(self, start: Sequence[float], end: Sequence[float],
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None, endpoint: bool = True):

        start = np.array(start)
        end = np.array(end)

        if (start.shape != (2,)) | (end.shape != (2,)):
            raise ValueError('scan start/end has wrong shape')

        self._start = start
        self._direction, extent = self._direction_and_extent(start, end)

        super().__init__(extent=extent, gpts=gpts, sampling=sampling, dimensions=1,
                         endpoint=endpoint)

    def _direction_and_extent(self, start, end):
        extent = np.linalg.norm((end - start), axis=0)
        direction = (end - start) / extent
        return direction, extent

    @property
    def num_measurements(self):
        return self.gpts[0]

    @property
    def shape(self):
        return (self.gpts[0].item(),)

    @property
    def start(self) -> np.ndarray:
        return self._start

    @start.setter
    def start(self, start: np.ndarray):
        self._start = np.array(start)
        self._direction, extent = self._direction_and_extent(self._start, self.end)

    @property
    def end(self) -> np.ndarray:
        return self.start + self.direction * self.extent

    @end.setter
    def end(self, end: np.ndarray):
        self._direction, extent = self._direction_and_extent(self.start, np.array(end))

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    def allocate_measurements(self, detectors):
        extent = self.extent * self.gpts / (self.gpts - 1) if self.endpoint else self.extent
        measurements = {}
        for detector in self._detectors:
            array = np.zeros(self.shape + detector.output_shape)
            measurement = ArrayWithGrid1D(array, extent=extent)

            if isinstance(detector.output, str):
                measurement = measurement.write(detector.output)
            measurements[detector] = measurement

        return measurements

    def insert_new_measurement(self, measurement, start, end, new_measurement):
        if isinstance(measurement, str):
            with h5py.File(measurement, 'a') as f:
                f['array'][start:end] = cp.asnumpy(new_measurement)

        else:
            measurement.array[start:end] = cp.asnumpy(new_measurement)

    def get_all_positions(self) -> np.ndarray:
        x = np.linspace(self.start[0], self.start[0] + self.extent * self.direction[0], self.gpts[0],
                        endpoint=self._endpoint)
        y = np.linspace(self.start[1], self.start[1] + self.extent * self.direction[1], self.gpts[0],
                        endpoint=self._endpoint)
        return np.stack((np.reshape(x, (-1,)), np.reshape(y, (-1,))), axis=1)

    def add_to_mpl_plot(self, ax, linestyle='-', color='r', **kwargs):
        ax.plot([self.start[0], self.end[0]], [self.start[1], self.end[1]], linestyle=linestyle, color=color, **kwargs)


class GridScan(Grid, ScanBase):

    def __init__(self, probe, potential, start=None, end=None, gpts=None, sampling=None, endpoint=False,
                 detectors=None):

        if start is None:
            start = (0, 0)

        if (end is None) & (potential is not None):
            end = potential.extent

        self._start = np.array(start)
        end = np.array(end)

        if (self._start.shape != (2,)) | (end.shape != (2,)):
            raise ValueError('scan start/end has wrong shape')

        super().__init__(probe=probe, potential=potential, extent=end - start, gpts=gpts, sampling=sampling,
                         dimensions=2, endpoint=endpoint, detectors=detectors)

    @property
    def num_measurements(self):
        return np.prod(self.gpts)

    @property
    def shape(self):
        return tuple(self.gpts)

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

    def get_x_positions(self) -> np.ndarray:
        return np.linspace(self.start[0], self.end[0], self.gpts[0], endpoint=self._endpoint)

    def get_y_positions(self) -> np.ndarray:
        return np.linspace(self.start[1], self.end[1], self.gpts[1], endpoint=self._endpoint)

    def get_positions(self) -> np.ndarray:
        x, y = np.meshgrid(self.get_x_positions(), self.get_y_positions(), indexing='ij')
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

    def allocate(self):
        self._measurements = {}
        extent = self.extent * (self.gpts) / (self.gpts - 1) if self.endpoint else self.extent
        for detector in self._detectors:
            array = np.zeros(self.shape + detector.output_shape)
            measurement = ArrayWithGrid2D(array, extent=extent)

            if isinstance(detector.output, str):
                measurement = measurement.write(detector.output)
            self._measurements[detector] = measurement

        return self._measurements
