import numpy as np

from abtem.bases import Grid
from abtem.utils import BatchGenerator, split_integer
from typing import Union, Sequence, Tuple


class ScanBase:

    def __init__(self, **kwargs):
        self._measurements = {}
        super().__init__(**kwargs)

    @property
    def measurements(self):
        return self._measurements

    def get_positions(self):
        raise NotImplementedError()

    @property
    def gpts(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def extent(self) -> Union[np.ndarray, None]:
        return None

    @property
    def endpoint(self) -> bool:
        return True

    def generate_positions(self, max_batch):
        positions = self.get_positions()

        batch_generator = BatchGenerator(len(positions), max_batch)

        for start, stop in batch_generator.generate():
            yield start, stop, positions[start:start + stop]


class CustomScan(ScanBase):
    def __init__(self, positions):
        ScanBase.__init__(self)
        self._positions = positions

    @property
    def gpts(self):
        return np.array([len(self._positions)])

    def get_positions(self):
        return self._positions


class LineScan(Grid, ScanBase):

    def __init__(self, start: Sequence[float], end: Sequence[float], gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None, endpoint: bool = True):
        ScanBase.__init__(self)

        start = np.array(start)
        end = np.array(end)

        if (start.shape != (2,)) | (end.shape != (2,)):
            raise ValueError('scan start/end has wrong shape')

        self._start = start
        self._direction, extent = self._direction_and_extent(start, end)

        super().__init__(extent=extent, gpts=gpts, sampling=sampling, dimensions=1, endpoint=endpoint)

    def _direction_and_extent(self, start, end):
        extent = np.linalg.norm((end - start), axis=0)
        direction = (end - start) / extent

        return direction, extent

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

    def partition(self, partitions: int):
        scans = {}

        start = self.start
        for i, n in enumerate(split_integer(self.gpts[0], partitions)):

            if self._endpoint:
                extent = self.sampling * (n - 1)
            else:
                extent = self.sampling * n

            end = start + self.direction * extent
            scans[i] = LineScan(start, end, np.array([n]), endpoint=self._endpoint)
            start = start + self.direction * self.sampling * n

        return scans

    def get_positions(self) -> np.ndarray:
        x = np.linspace(self.start[0], self.start[0] + self.extent * self.direction[0], self.gpts[0],
                        endpoint=self._endpoint)
        y = np.linspace(self.start[1], self.start[1] + self.extent * self.direction[1], self.gpts[0],
                        endpoint=self._endpoint)
        return np.stack((np.reshape(x, (-1,)), np.reshape(y, (-1,))), axis=1)


class GridScan(Grid, ScanBase):

    def __init__(self, start, end, gpts=None, sampling=None, endpoint=False):
        self._start = np.array(start)
        end = np.array(end)

        if (self._start.shape != (2,)) | (end.shape != (2,)):
            raise ValueError('scan start/end has wrong shape')

        super().__init__(extent=end - start, gpts=gpts, sampling=sampling, dimensions=2, endpoint=endpoint)

    def __len__(self):
        return np.prod(self.gpts)

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

    def partition(self, partitions: Tuple[int, int]):
        scans = {}
        start_n = self.start[0]
        for i, n in enumerate(split_integer(self.gpts[0], partitions[0])):
            start_m = self.start[1]
            for j, m in enumerate(split_integer(self.gpts[1], partitions[1])):
                start = np.array([start_n, start_m])

                if self._endpoint:
                    extent = self.sampling * (n - 1, m - 1)
                else:
                    extent = self.sampling * (n, m)

                end = start + extent
                scans[(i, j)] = GridScan(start, end, np.array([n, m]), endpoint=self._endpoint)
                start_m = start_m + self.sampling[1] * m

            start_n = start_n + self.sampling[0] * n
        return scans

    def get_x_positions(self) -> np.ndarray:
        return np.linspace(self.start[0], self.end[0], self.gpts[0], endpoint=self._endpoint)

    def get_y_positions(self) -> np.ndarray:
        return np.linspace(self.start[1], self.end[1], self.gpts[1], endpoint=self._endpoint)

    def get_positions(self) -> np.ndarray:
        x, y = np.meshgrid(self.get_x_positions(), self.get_y_positions(), indexing='ij')
        return np.stack((np.reshape(x, (-1,)),
                         np.reshape(y, (-1,))), axis=1)


def assemble_partitions_2d(partitions):
    n = max([key[0] for key in partitions.keys()]) + 1
    m = max([key[1] for key in partitions.keys()]) + 1

    N = sum([partition.shape[0] for key, partition in partitions.items() if key[1] == 0])
    M = sum([partition.shape[1] for key, partition in partitions.items() if key[0] == 0])

    data = np.zeros((N, M) + partitions[(0, 0)].shape[2:])

    l = 0
    for i in range(n):
        k = 0
        for j in range(m):
            partition = partitions[(i, j)]
            data[l:l + partition.shape[0], k:k + partition.shape[1]] = partition
            k += partition.shape[1]
        l += partition.shape[0]
    return data
