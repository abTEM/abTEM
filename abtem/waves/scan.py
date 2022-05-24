"""Module for describing different types of scans."""
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Union, Tuple, TYPE_CHECKING

import dask
import dask.array as da
import dask.bag
import numpy as np
from ase import Atom
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from abtem.core.axes import ScanAxis, PositionsAxis
from abtem.core.backend import get_array_module, validate_device
from abtem.core.blockwise import Ensemble
from abtem.core.dask import validate_chunks
from abtem.core.distributions import MultidimensionalAxisAlignedDistribution
from abtem.core.fft import fft_shift_kernel, ifft2
from abtem.core.grid import Grid, HasGridMixin

if TYPE_CHECKING:
    from abtem.waves import Waves


def validate_scan(scan, probe=None):
    if scan is None:
        scan = GridScan()

    if not hasattr(scan, 'get_positions'):
        scan = CustomScan(scan)

    if probe is not None:
        scan.match_probe(probe)

    return scan


class AbstractScan(Ensemble, metaclass=ABCMeta):
    """Abstract class to describe scans."""

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
    def ensemble_shape(self):
        return self.shape

    @property
    def default_ensemble_chunks(self):
        return ('auto',) * len(self.ensemble_shape)

    @abstractmethod
    def get_positions(self, *args, **kwargs):
        """Get the scan positions as numpy array."""
        pass

    @property
    @abstractmethod
    def limits(self):
        pass

    def apply_fft_shift(self, waves, out_space: 'str' = 'in_space'):
        kernel = self.fft_shift(extent=waves.extent, gpts=waves.gpts, device=waves.device)

        if out_space == 'in_space':
            fourier_space_out = waves.fourier_space
        else:
            fourier_space_out = out_space == 'fourier_space'

        waves = waves.ensure_fourier_space()

        kernel = kernel[(slice(None),) * len(self.shape) + (None,) * len(waves.ensemble_shape)]

        array = waves.array[(None,) * len(self.shape)] * kernel

        if not fourier_space_out:
            array = ifft2(array, overwrite_x=True)

        d = waves._copy_as_dict(copy_array=False)
        d['array'] = array
        d['ensemble_axes_metadata'] = self.ensemble_axes_metadata + d['ensemble_axes_metadata']
        return waves.__class__(**d)

    def fft_shift(self, extent=None, gpts=None, sampling=None, device=None):
        device = validate_device(device)
        xp = get_array_module(device)

        grid = Grid(extent=extent, gpts=gpts, sampling=sampling)
        grid.check_is_defined()

        positions = xp.asarray(self.get_positions()) / xp.asarray(grid.sampling).astype(np.float32)
        return fft_shift_kernel(positions, shape=grid.gpts)

    def copy(self):
        """Make a copy."""
        return deepcopy(self)


class SourceOffset(AbstractScan):

    def __init__(self, distribution):
        self._distribution = distribution

    @property
    def shape(self):
        return self._distribution.shape

    def get_positions(self):
        xi = [factor.values for factor in self._distribution.factors]
        return np.stack(np.meshgrid(*xi, indexing='ij'), axis=-1)

    @property
    def ensemble_axes_metadata(self):
        return [PositionsAxis()] * len(self.shape)

    def ensemble_blocks(self, chunks=None):
        if chunks is None:
            chunks = self.default_ensemble_chunks

        chunks = validate_chunks(self.ensemble_shape, chunks, limit=None)

        blocks = ()
        for parameter, n in zip(self._distribution.factors, chunks):
            blocks += (parameter.divide(n, lazy=True),)

        return blocks

    def ensemble_partial(self):
        def distribution(*args):
            factors = [arg.item() for arg in args]
            dist = MultidimensionalAxisAlignedDistribution(factors)
            arr = np.empty((1,) * len(args), dtype=object)
            arr.itemset(dist)
            return arr

        return distribution

    @property
    def limits(self):
        pass


class CompoundScan(AbstractScan):

    def __init__(self, scans):
        self._scans = scans

    @property
    def shape(self):
        shape = ()
        for scan in self._scans:
            shape += scan.shape
        return shape

    def get_positions(self):

        positions = self._scans.get_positions()
        for scan in self._scans[1:]:
            positions = np.add.outer(positions, scan.get_positions())

        return positions

    @property
    def ensemble_axes_metadata(self):
        return [PositionsAxis()] * len(self.shape)

    def ensemble_blocks(self, chunks=None):
        if chunks is None:
            chunks = self.default_ensemble_chunks

        chunks = validate_chunks(self.ensemble_shape, chunks, limit=None)

        blocks = ()
        for parameter, n in zip(self._distribution.factors, chunks):
            blocks += (parameter.divide(n, lazy=True),)

        return blocks

    def ensemble_partial(self):
        def distribution(*args):
            factors = [arg.item() for arg in args]
            dist = MultidimensionalAxisAlignedDistribution(factors)
            arr = np.empty((1,) * len(args), dtype=object)
            arr.itemset(dist)
            return arr

        return distribution

    @property
    def limits(self):
        pass



class CustomScan(AbstractScan):

    def __init__(self, positions: np.ndarray = None):
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
    def ensemble_axes_metadata(self):
        return [PositionsAxis()]

    def ensemble_blocks(self, chunks):
        chunks = validate_chunks(self.ensemble_shape, chunks)
        cumchunks = tuple(np.cumsum(chunks[0]))
        positions = np.empty(len(chunks[0]), dtype=object)
        for i, (start_chunk, chunk) in enumerate(zip((0,) + cumchunks, chunks[0])):
            positions[i] = CustomScan(self._positions[start_chunk:start_chunk + chunk])
        return da.from_array(positions, chunks=1),

    def ensemble_partial(self):
        return lambda x: x

    @property
    def shape(self):
        return self.positions.shape[:-1]

    @property
    def positions(self):
        return self._positions

    @property
    def limits(self):
        return [(np.min(self.positions[:, 0]), np.min(self.positions[:, 1])),
                (np.max(self.positions[:, 0]), np.max(self.positions[:, 1]))]

    def get_positions(self) -> np.ndarray:
        return self._positions


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
    _num_scan_axes = 1

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
            self.sampling = .9 * probe.aperture.nyquist_sampling

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

        if self.endpoint:
            self._sampling = self.extent / (self.gpts - 1)
        else:
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

        return [ScanAxis(label='x', sampling=float(self.sampling), units='Å', start=start, end=self.end)]

    @property
    def start(self) -> Union[Tuple[float, float], None]:
        """
        Start point of the scan [Å].
        """
        return self._start

    @start.setter
    def start(self, start: Tuple[float, float]):
        if start is not None:
            start = (float(start[0]), float(start[1]))

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
        if end is not None:
            end = (float(end[0]), float(end[1]))
        self._end = end
        self._adjust_gpts()

    @property
    def ensemble_axes_metadata(self):
        return [ScanAxis(label='x', sampling=self.sampling, offset=0., units='Å', endpoint=self.endpoint)]

    @property
    def ensemble_shape(self):
        return self.shape

    @property
    def default_ensemble_chunks(self):
        return 'auto', 'auto'

    def ensemble_partial(self):
        return lambda x: x

    def ensemble_blocks(self, chunks=None):

        # self.grid.check_is_defined()

        if chunks is None:
            chunks = self.default_ensemble_chunks

        chunks = validate_chunks(self.ensemble_shape, chunks)

        direction = np.array(self.end) - np.array(self.start)
        direction = direction / np.linalg.norm(direction, axis=0)

        cumchunks = tuple(np.cumsum(chunks[0]))

        block = np.empty(len(chunks[0]), dtype=object)
        for i, (start_chunk, chunk) in enumerate(zip((0,) + cumchunks, chunks[0])):
            start = np.array(self.start) + start_chunk * self.sampling * direction

            end = start + self.sampling * chunk * direction
            block[i] = LineScan(start=start, end=end, gpts=chunk, endpoint=False)

        block = da.from_array(block, chunks=1)

        return block,

    def get_positions(self, chunks: int = None, lazy: bool = False) -> np.ndarray:
        x = np.linspace(self.start[0], self.end[0], self.gpts, endpoint=self.endpoint, dtype=np.float32)
        y = np.linspace(self.start[1], self.end[1], self.gpts, endpoint=self.endpoint, dtype=np.float32)
        return np.stack((np.reshape(x, (-1,)), np.reshape(y, (-1,))), axis=1)

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

        if start is not None:
            if np.isscalar(start):
                start = (start,) * 2

            start = tuple(map(float, start))
            assert len(start) == 2

        if end is not None:
            if np.isscalar(end):
                end = (end,) * 2

            end = tuple(map(float, end))

            assert len(end) == 2

        if start is not None and end is not None:
            extent = np.array(end, dtype=float) - start
        else:
            extent = None

        self._start = start
        self._end = end
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling, dimensions=2, endpoint=endpoint)

    @property
    def dimensions(self):
        return self.grid.dimensions

    @property
    def limits(self):
        return [self.start, self.end]

    @property
    def endpoint(self) -> Tuple[bool, ...]:
        return self.grid.endpoint

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.gpts

    @property
    def start(self) -> Union[Tuple[float, ...], None]:
        """Start corner of the scan [Å]."""
        return self._start

    @start.setter
    def start(self, start: Tuple[float, ...]):
        self._start = start
        self._adjust_extent()

    @property
    def end(self) -> Union[Tuple[float, ...], None]:
        """End corner of the scan [Å]."""
        return self._end

    @end.setter
    def end(self, end: Tuple[float, ...]):
        self._end = end
        self._adjust_extent()

    def _adjust_extent(self):
        if self.start is None or self.end is None:
            return

        self.extent = np.array(self.end) - self.start

    def match_probe(self, probe):
        if self.start is None:
            self.start = (0., 0.)

        if self.end is None:
            self.end = probe.extent

        if self.sampling is None:
            self.sampling = .9 * probe.aperture.nyquist_sampling

    def get_positions(self) -> np.ndarray:
        xi = []
        for start, end, gpts, endpoint in zip(self.start, self.end, self.gpts, self.endpoint):
            xi.append(np.linspace(start, end, gpts, endpoint=endpoint, dtype=np.float32))

        if len(xi) == 1:
            return xi[0]

        return np.stack(np.meshgrid(*xi, indexing='ij'), axis=-1)

    @property
    def ensemble_axes_metadata(self):
        axes_metadata = []
        labels = ('x', 'y', 'z')
        for label, sampling, offset, endpoint in zip(labels, self.sampling, self.start, self.endpoint):
            axes_metadata.append(ScanAxis(label=label, sampling=sampling, offset=offset, units='Å', endpoint=endpoint))
        return axes_metadata

    @property
    def ensemble_shape(self):
        return self.shape

    @property
    def default_ensemble_chunks(self):
        return 'auto', 'auto'

    def ensemble_partial(self):

        def scan(x_scan, y_scan):
            x_scan, y_scan = x_scan.item(), y_scan.item()
            start = (x_scan['start'], y_scan['start'])
            end = (x_scan['end'], y_scan['end'])
            gpts = (x_scan['gpts'], y_scan['gpts'])
            endpoint = (x_scan['endpoint'], y_scan['endpoint'])
            arr = np.empty((1, 1), dtype=object)
            arr[0] = GridScan(start=start, end=end, gpts=gpts, endpoint=endpoint)
            return arr

        return scan

    def ensemble_blocks(self, chunks=None):

        self.grid.check_is_defined()

        if chunks is None:
            chunks = self.default_ensemble_chunks

        chunks = validate_chunks(self.ensemble_shape, chunks)

        blocks = ()
        for i in range(2):
            cumchunks = tuple(np.cumsum(chunks[i]))

            block = np.empty(len(chunks[i]), dtype=object)
            for j, (start_chunk, chunk) in enumerate(zip((0,) + cumchunks, chunks[i])):
                start = self.start[i] + start_chunk * self.sampling[i]
                end = start + self.sampling[i] * chunk
                block[j] = {'start': start, 'end': end, 'gpts': chunk, 'endpoint': False}

            blocks += da.from_array(block, chunks=1),

        return blocks

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