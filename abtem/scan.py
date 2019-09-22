import numpy as np

from abtem.bases import Grid
from abtem.utils import BatchGenerator


def split_integer(n, m):
    if n < m:
        raise RuntimeError()

    elif n % m == 0:
        return [n // m] * m
    else:
        v = []
        zp = m - (n % m)
        pp = n // m
        for i in range(m):
            if i >= zp:
                v.append(pp + 1)
            else:
                v.append(pp)

        return v


class Scan(object):

    def __init__(self, **kwargs):
        self._measurements = {}
        super().__init__(**kwargs)

    @property
    def measurements(self):
        return self._measurements

    def get_positions(self):
        raise NotImplementedError('')

    def generate_positions(self, max_batch, show_progress=True):
        positions = self.get_positions()

        batch_generator = BatchGenerator(len(positions), max_batch)

        for start, stop in batch_generator.generate(show_progress=show_progress):
            yield start, stop, positions[start:start + stop]


class CustomScan(Scan):
    def __init__(self, positions):
        Scan.__init__(self)
        self._positions = positions


class LineScan(Scan, Grid):

    def __init__(self, start, end=None, gpts=None, sampling=None, endpoint=False):
        Scan.__init__(self)

        start = np.array(start)
        end = np.array(end)

        self._start = start
        self._direction, extent = self._direction_and_extent(start, end)

        super().__init__(extent=extent, gpts=gpts, sampling=sampling, dimensions=1, endpoint=endpoint)

    def _direction_and_extent(self, start, end):
        extent = np.linalg.norm((end - start), axis=0)
        direction = (end - start) / extent
        return direction, extent

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        end = self.end
        self._start = np.array(start)
        self._direction, extent = self._direction_and_extent(self._start, end)

    @property
    def end(self):
        return self.start + self.direction * self.extent

    @end.setter
    def end(self, end):
        end = np.array(end)
        self._direction, extent = self._direction_and_extent(self.start, end)

    @property
    def total(self):
        return self.gpts[0]

    @property
    def direction(self):
        return self._direction

    def partition(self, partitions):
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

    # def get_positions(self):
    #     return np.ascontiguousarray(
    #         np.linspace(0., self.extent, self.gpts, endpoint=self._endpoint) *
    #         np.expand_dims(self.direction, axis=0) + self.start)

    def get_positions(self):
        x = np.linspace(self.start[0], self.start[0] + self.extent * self.direction[0], self.gpts,
                        endpoint=self._endpoint)
        y = np.linspace(self.start[1], self.start[1] + self.extent * self.direction[1], self.gpts,
                        endpoint=self._endpoint)
        return np.stack((np.reshape(x, (-1,)),
                         np.reshape(y, (-1,))), axis=1)


class GridScan(Scan, Grid):

    def __init__(self, start, end=None, gpts=None, sampling=None, endpoint=False):
        Scan.__init__(self)

        self._start = np.array(start)
        super().__init__(extent=np.array(end) - start, gpts=gpts, sampling=sampling, dimensions=2, endpoint=endpoint)

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        start = np.array(start)
        self._start = start
        self.extent = self.end - self._start

    @property
    def end(self):
        return self.start + self.extent

    @end.setter
    def end(self, end):
        self.extent = np.array(end) - self.start

    def partition(self, partitions):
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

    def get_x_positions(self):
        return np.linspace(self.start[0], self.end[0], self.gpts[0], endpoint=self._endpoint)

    def get_y_positions(self):
        return np.linspace(self.start[1], self.end[1], self.gpts[1], endpoint=self._endpoint)

    def get_positions(self):
        x, y = np.meshgrid(self.get_x_positions(), self.get_y_positions(), indexing='ij')
        return np.stack((np.reshape(x, (-1,)),
                         np.reshape(y, (-1,))), axis=1)
