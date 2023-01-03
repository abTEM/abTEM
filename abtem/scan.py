"""Module for describing different types of scans."""
import itertools
from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Union, Tuple, TYPE_CHECKING

import dask.array as da
import numpy as np
from ase import Atom
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from abtem.core.axes import ScanAxis, PositionsAxis
from abtem.core.backend import get_array_module, validate_device
from abtem.core.chunks import validate_chunks
from abtem.core.fft import fft_shift_kernel
from abtem.core.grid import Grid, HasGridMixin
from abtem.core.transform import WaveTransform
from abtem.core.utils import safe_floor_int
from abtem.distributions import _AxisAlignedDistributionND, BaseDistribution
from abtem.potentials import BasePotential, _validate_potential
from abtem.transfer import nyquist_sampling

if TYPE_CHECKING:
    from abtem.waves import Waves


def _validate_scan(scan, probe=None):
    if scan is None:
        scan = GridScan()

    if not hasattr(scan, "get_positions"):
        scan = CustomScan(scan)

    if probe is not None:
        scan = scan.copy()
        scan.match_probe(probe)

    return scan


def _validate_scan_sampling(scan, probe):
    if scan.sampling is None:
        if not hasattr(probe, "semiangle_cutoff"):
            raise ValueError()

        scan.sampling = 0.99 * nyquist_sampling(probe.semiangle_cutoff, probe.energy)


class BaseScan(WaveTransform, metaclass=ABCMeta):
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
    def _default_ensemble_chunks(self):
        return ("auto",) * len(self.ensemble_shape)

    @abstractmethod
    def get_positions(self, *args, **kwargs):
        """Get the scan positions as numpy array."""
        pass

    def get_weights(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def limits(self):
        pass

    @abstractmethod
    def sort_into_extents(self, extents):
        pass

    @property
    def metadata(self):
        return {}

    def generate_blocks_by_extents(self, extents):
        scan, chunks = self.sort_into_extents(extents)

        for sub_scan in self.generate_blocks(chunks):
            yield sub_scan

    def evaluate(self, waves):
        device = validate_device(waves.device)
        xp = get_array_module(device)

        waves.grid.check_is_defined()

        positions = xp.asarray(self.get_positions()) / xp.asarray(
            waves.sampling
        ).astype(np.float32)

        kernel = fft_shift_kernel(positions, shape=waves.gpts)

        try:
            kernel *= self.get_weights()[..., None, None]
        except NotImplementedError:
            pass

        return kernel

    def apply(self, waves: "Waves", overwrite_x: bool = False) -> "Waves":
        array = self.evaluate(waves)
        axes_metadata = self.ensemble_axes_metadata
        return waves.convolve(array, axes_metadata, overwrite_x=overwrite_x)


class SourceOffset(BaseScan):
    """
    Distribution of electron source offsets.

    Parameters
    ----------
    distribution : 2D :class:`.BaseDistribution`
        Distribution describing the positions and weights of the source offsets.
    """

    def __init__(self, distribution: BaseDistribution):
        self._distribution = distribution

    @property
    def shape(self):
        return self._distribution.shape

    def get_positions(self):
        xi = [factor.values for factor in self._distribution.factors]
        return np.stack(np.meshgrid(*xi, indexing="ij"), axis=-1)

    def get_weights(self):
        return self._distribution.weights

    @property
    def ensemble_axes_metadata(self):
        return [PositionsAxis()] * len(self.shape)

    def ensemble_blocks(self, chunks=None):
        if chunks is None:
            chunks = self._default_ensemble_chunks

        chunks = validate_chunks(self.ensemble_shape, chunks, limit=None)

        blocks = ()
        for parameter, n in zip(self._distribution.factors, chunks):
            blocks += (parameter.divide(n, lazy=True),)

        return blocks

    def ensemble_partial(self):
        def distribution(*args):
            factors = [arg.item() for arg in args]
            dist = SourceOffset(_AxisAlignedDistributionND(factors))
            arr = np.empty((1,) * len(args), dtype=object)
            arr.itemset(dist)
            return arr

        return distribution

    @property
    def limits(self):
        pass


class CustomScan(BaseScan):
    """
    Custom scan based on explicit 2D probe positions.

    Parameters
    ----------
    positions : np.ndarray, optional
        Probe positions. Anything that can be converted to an ndarray of shape (n, 3) is accepted. Default is
        (0., 0.).
    """

    def __init__(self, positions: np.ndarray = (0.0, 0.0)):
        positions = np.array(positions, dtype=np.float32)

        if len(positions.shape) == 1:
            positions = positions[None]

        self._positions = positions

        super().__init__()

    def match_probe(self, probe):
        if len(self.positions) == 0:
            self._positions = np.array(probe.extent, dtype=np.float32)[None] / 2.0

    @property
    def ensemble_axes_metadata(self):
        a = [
            PositionsAxis(
                values=tuple(
                    (float(position[0]), float(position[1]))
                    for position in self.positions
                )
            )
        ]
        return a

    @staticmethod
    def _from_partitioned_args_func(*args, **kwargs):
        return CustomScan(args[0])

    def _from_partitioned_args(self):
        return self._from_partitioned_args_func

    def _partition_args(self, chunks=None, lazy: bool = True):
        chunks = self._validate_chunks(chunks)
        cumchunks = tuple(np.cumsum(chunks[0]))
        positions = np.empty(len(chunks[0]), dtype=object)
        for i, (start_chunk, chunk) in enumerate(zip((0,) + cumchunks, chunks[0])):
            positions.itemset(i, self._positions[start_chunk : start_chunk + chunk])

        if lazy:
            positions = da.from_array(positions, chunks=1)

        return (positions,)

    def sort_into_extents(self, extents):
        new_positions = np.zeros_like(self.positions)
        chunks = ()
        start = 0
        for x_extents, y_extents in itertools.product(*extents):
            mask = (
                (self.positions[:, 0] >= x_extents[0])
                * (self.positions[:, 0] < x_extents[1])
                * (self.positions[:, 1] >= y_extents[0])
                * (self.positions[:, 1] < y_extents[1])
            )

            n = np.sum(mask)
            chunks += (n,)
            stop = start + n
            new_positions[start:stop] = self.positions[mask]
            start = stop

        assert sum(chunks) == len(self)
        return CustomScan(new_positions), (chunks,)

    @property
    def shape(self):
        return self.positions.shape[:-1]

    @property
    def positions(self):
        return self._positions

    @property
    def limits(self):
        return [
            (np.min(self.positions[:, 0]), np.min(self.positions[:, 1])),
            (np.max(self.positions[:, 0]), np.max(self.positions[:, 1])),
        ]

    def get_positions(self) -> np.ndarray:
        return self._positions


def _validate_coordinate(coordinate, potential=None, fractional: bool = False):
    if isinstance(coordinate, Atom):
        if fractional:
            raise ValueError()

        coordinate = coordinate.x, coordinate.y

    if fractional:
        if potential is None:
            raise ValueError("provide potential for fractional coordinates")

        potential = _validate_potential(potential)

        coordinate = (
            potential.extent[0] * coordinate[0],
            potential.extent[1] * coordinate[1],
        )

    coordinate = coordinate if coordinate is None else tuple(coordinate)

    return coordinate


def _validate_coordinates(start, end, potential, fractional):
    start = _validate_coordinate(start, potential, fractional)
    end = _validate_coordinate(end, potential, fractional)

    if start is not None and end is not None:
        if np.allclose(start, end):
            raise RuntimeError("scan start and end is identical")

    return start, end


class LineScan(BaseScan):
    """
    A scan along a straight line.

    Parameters
    ----------
    start : two float
        Start point of the scan [Å]. Default is (0., 0.).
    end : two float
        End point of the scan [Å]. Default is None, the scan end point will match the extent of the potential.
    gpts: int
        Number of scan positions. Default is None. Provide one of gpts or sampling.
    sampling: float
        Sampling rate of scan positions [1 / Å]. Provide one of gpts or sampling. If not provided the sampling will
        match the Nyquist sampling of the Probe in a multislice simulation.
    endpoint: bool
        If True, end is the last position. Otherwise, it is not included. Default is True.
    """

    def __init__(
        self,
        start: Union[Tuple[float, float], None] = (0.0, 0.0),
        end: Union[Tuple[float, float], None] = None,
        gpts: int = None,
        sampling: float = None,
        endpoint: bool = True,
        fractional: bool = False,
        potential: BasePotential = None,
    ):

        super().__init__()
        self._gpts = gpts
        self._sampling = sampling

        self._start, self._end = _validate_coordinates(
            start, end, potential, fractional
        )

        self._endpoint = endpoint
        self._adjust_gpts()
        self._adjust_sampling()

    @classmethod
    def from_fractional_coordinates(
        cls,
        potential: BasePotential,
        start: Tuple[float, float] = (0.0, 0.0),
        end: Tuple[float, float] = (1.0, 1.0),
        sampling: Union[float, Tuple[float, float]] = None,
        endpoint: Union[bool, Tuple[bool, bool]] = False,
    ) -> "LineScan":
        """
        Create grid scan using fractional coordinates.

        Parameters
        ----------
        potential : BasePotential
            Potential defining the grid with respect to which the fractional coordinates should be given.
        start : two float, optional
            Scan start as a fraction of the extent of the potential.
        end : two float, optional
            Scan end as a fraction of the extent of the potential.
        sampling : float or two float
            Sampling rate of scan positions [1 / Å]. Provide one of gpts or sampling. If not provided the sampling will
            match the Nyquist sampling of the probe in a multislice simulation.
        endpoint : bool or two bool
            If True, end is the last positions. Otherwise, it those positions are not included. Default is False.
        Returns
        -------
        grid_scan : GridScan
        """

        return cls(
            start=start,
            end=end,
            sampling=sampling,
            endpoint=endpoint,
            potential=potential,
            fractional=True,
        )

    def add_margin(self, margin: Union[float, Tuple[float, float]]):
        if isinstance(margin, Number):
            margin = (margin,) * 2

        direction = np.array(self.end) - np.array(self.start)
        direction = direction / np.linalg.norm(direction)

        self.start = tuple(np.array(self.start) - direction * margin[0])
        self.end = tuple(np.array(self.end) + direction * margin[1])

    @classmethod
    def at_position(
        cls,
        position: Union[Tuple[float, float], Atom],
        extent: float = 1.0,
        angle: float = 0.0,
        gpts: int = None,
        sampling: float = None,
        endpoint: bool = True,
    ):

        if isinstance(position, Atom):
            position = (position.x, position.y)

        direction = np.array((np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))))

        start = tuple(np.array(position) - extent / 2 * direction)
        end = tuple(np.array(position) + extent / 2 * direction)
        return cls(
            start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint
        )

    def match_probe(self, probe):
        if self.start is None:
            self.start = (0.0, 0.0)

        if self.end is None and probe.extent is not None:
            self.end = (0.0, probe.extent[1])

        _validate_scan_sampling(self, probe)

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

        if self.endpoint and self.gpts > 1:
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
    def metadata(self):
        return {"start": self.start, "end": self.end}

    @property
    def axes_metadata(self):

        return [
            ScanAxis(
                label="x",
                sampling=float(self.sampling),
                units="Å",
                start=self.start,
                end=self.end,
            )
        ]

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
        return [
            ScanAxis(
                label="x",
                sampling=self.sampling,
                offset=0.0,
                units="Å",
                endpoint=self.endpoint,
            )
        ]

    @property
    def ensemble_shape(self):
        return self.shape

    @property
    def _default_ensemble_chunks(self):
        return ("auto",)

    def sort_into_extents(self, extents):
        raise NotImplementedError

    @staticmethod
    def _from_partitioned_args(*args, **kwargs):
        return lambda x: x

    def _partition_args(self, chunks=None, lazy: bool = True):
        if chunks is None:
            chunks = self._default_ensemble_chunks

        chunks = validate_chunks(self.ensemble_shape, chunks)

        direction = np.array(self.end) - np.array(self.start)
        direction = direction / np.linalg.norm(direction, axis=0)

        cumchunks = tuple(np.cumsum(chunks[0]))

        block = np.empty(len(chunks[0]), dtype=object)
        for i, (start_chunk, chunk) in enumerate(zip((0,) + cumchunks, chunks[0])):
            start = np.array(self.start) + start_chunk * self.sampling * direction

            end = start + self.sampling * chunk * direction
            block[i] = LineScan(start=start, end=end, gpts=chunk, endpoint=False)

        if lazy:
            block = da.from_array(block, chunks=1)

        return (block,)

    def get_positions(self, chunks: int = None, lazy: bool = False) -> np.ndarray:
        x = np.linspace(
            self.start[0],
            self.end[0],
            self.gpts,
            endpoint=self.endpoint,
            dtype=np.float32,
        )
        y = np.linspace(
            self.start[1],
            self.end[1],
            self.gpts,
            endpoint=self.endpoint,
            dtype=np.float32,
        )
        return np.stack((np.reshape(x, (-1,)), np.reshape(y, (-1,))), axis=1)

    def add_to_plot(self, ax: Axes, linestyle: str = "-", color: str = "r", **kwargs):
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
        ax.plot(
            [self.start[0], self.end[0]],
            [self.start[1], self.end[1]],
            linestyle=linestyle,
            color=color,
            **kwargs
        )


class GridScan(HasGridMixin, BaseScan):
    """
    A scan over a regular grid for calculating scanning transmission electron microscopy.

    Parameters
    ----------
    start : two float
        Start corner of the scan [Å]. Default is (0., 0.).
    end : two float
        End corner of the scan [Å]. Default is None, the scan end point will match the extent of the potential.
    gpts : two int
        Number of scan positions in the `x`- and `y`-direction of the scan. Provide one of gpts or sampling.
    sampling : two float
        Sampling rate of scan positions [1 / Å]. Provide one of gpts or sampling. If not provided the sampling will
        match the Nyquist sampling of the  Probe in a multislice simulation.
    endpoint : bool
        If True, end is the last position. Otherwise, it is not included. Default is False.
    """

    def __init__(
        self,
        start: Tuple[float, float] = (0.0, 0.0),
        end: Tuple[float, float] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        endpoint: Union[bool, Tuple[bool, bool]] = False,
        fractional: bool = False,
        potential: BasePotential = None,
    ):

        super().__init__()

        start, end = _validate_coordinates(start, end, potential, fractional)

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
        self._grid = Grid(
            extent=extent, gpts=gpts, sampling=sampling, dimensions=2, endpoint=endpoint
        )

    def __len__(self):
        return self.gpts[0] * self.gpts[1]

    @classmethod
    def from_fractional_coordinates(
        cls,
        potential: BasePotential,
        start: Tuple[float, float] = (0.0, 0.0),
        end: Tuple[float, float] = (1.0, 1.0),
        sampling: Union[float, Tuple[float, float]] = None,
        endpoint: Union[bool, Tuple[bool, bool]] = False,
    ) -> "GridScan":
        """
        Create grid scan using fractional coordinates.

        Parameters
        ----------
        potential : BasePotential
            Potential defining the grid with respect to which the fractional coordinates should be given.
        start : two float, optional
            Scan start as a fraction of the extent of the potential.
        end : two float, optional
            Scan end as a fraction of the extent of the potential.
        sampling : float or two float

        endpoint : bool or two bool

        Returns
        -------
        grid_scan : GridScan
        """

        return cls(
            start=start,
            end=end,
            sampling=sampling,
            endpoint=endpoint,
            potential=potential,
            fractional=True,
        )

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
            self.start = (0.0, 0.0)

        if self.end is None:
            self.end = probe.extent

        _validate_scan_sampling(self, probe)

    def _x_coordinates(self):
        return np.linspace(
            self.start[0],
            self.end[0],
            self.gpts[0],
            endpoint=self.endpoint[0],
            dtype=np.float32,
        )

    def _y_coordinates(self):
        return np.linspace(
            self.start[1],
            self.end[1],
            self.gpts[1],
            endpoint=self.endpoint[1],
            dtype=np.float32,
        )

    def get_positions(self) -> np.ndarray:
        xi = []
        for start, end, gpts, endpoint in zip(
            self.start, self.end, self.gpts, self.endpoint
        ):
            xi.append(
                np.linspace(start, end, gpts, endpoint=endpoint, dtype=np.float32)
            )

        if len(xi) == 1:
            return xi[0]

        return np.stack(np.meshgrid(*xi, indexing="ij"), axis=-1)

    def sort_into_extents(self, extents):

        x = np.linspace(
            self.start[0], self.end[0], self.gpts[0], endpoint=self.endpoint[0]
        )


        _, x_chunks = np.unique(np.digitize(x, [l for _, l in extents[0]]), return_counts=True)

        y = np.linspace(
            self.start[1], self.end[1], self.gpts[1], endpoint=self.endpoint[1]
        )

        _, y_chunks = np.unique(np.digitize(y, [l for _, l in extents[1]]), return_counts=True)

        return self, (tuple(x_chunks), tuple(y_chunks))

        #print(bins, counts)
        #sss

        # x_chunks = ()
        # for start, end in extents[0]:
        #     end_with_endpoint = self.end[0]
        #
        #     if self.endpoint:
        #         end_with_endpoint = end_with_endpoint + self.sampling[0]
        #
        #     start_gpt = safe_floor_int(max(start, self.start[0]) / self.sampling[0])
        #     end_gpt = safe_floor_int(min(end, end_with_endpoint) / self.sampling[0])
        #
        #     start_gpt = max(0, start_gpt)
        #     end_gpt = min(self.gpts[0], end_gpt)
        #
        #     x_chunks += (max(end_gpt - start_gpt, 0),)
        #
        # assert sum(x_chunks) == self.gpts[0]

        # y_chunks = ()
        # for start, end in extents[1]:
        #     end_with_endpoint = self.end[1]
        #     if self.endpoint:
        #         end_with_endpoint = end_with_endpoint + self.sampling[1]
        #     start_gpt = safe_floor_int(max(start, self.start[1]) / self.sampling[1])
        #     end_gpt = safe_floor_int(min(end, end_with_endpoint) / self.sampling[1])
        #
        #     start_gpt = max(0, start_gpt)
        #     end_gpt = min(self.gpts[1], end_gpt)
        #
        #     y_chunks += (max(end_gpt - start_gpt, 0),)
        #
        # assert sum(y_chunks) == self.gpts[1]
        # return self, (x_chunks, y_chunks)

    @property
    def ensemble_axes_metadata(self):
        axes_metadata = []
        labels = ("x", "y", "z")
        for label, sampling, offset, endpoint in zip(
            labels, self.sampling, self.start, self.endpoint
        ):
            axes_metadata.append(
                ScanAxis(
                    label=label,
                    sampling=sampling,
                    offset=offset,
                    units="Å",
                    endpoint=endpoint,
                )
            )
        return axes_metadata

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        x_scan, y_scan = args
        start = (x_scan["start"], y_scan["start"])
        end = (x_scan["end"], y_scan["end"])
        gpts = (x_scan["gpts"], y_scan["gpts"])
        endpoint = (x_scan["endpoint"], y_scan["endpoint"])
        return cls(start=start, end=end, gpts=gpts, endpoint=endpoint)

    def _from_partitioned_args(self):
        return self._from_partitioned_args_func

    @property
    def ensemble_shape(self):
        return self.shape

    @property
    def _default_ensemble_chunks(self):
        return "auto", "auto"

    def _partition_args(self, chunks=None, lazy=True):
        self.grid.check_is_defined()
        chunks = self._validate_chunks(chunks)
        blocks = ()
        for i in range(2):
            cumchunks = tuple(np.cumsum(chunks[i]))

            block = np.empty(len(chunks[i]), dtype=object)
            for j, (start_chunk, chunk) in enumerate(zip((0,) + cumchunks, chunks[i])):
                start = self.start[i] + start_chunk * self.sampling[i]
                end = start + self.sampling[i] * chunk
                block[j] = {
                    "start": start,
                    "end": end,
                    "gpts": chunk,
                    "endpoint": False,
                }

            if lazy:
                blocks += (da.from_array(block, chunks=1),)
            else:
                blocks += (block,)

        return blocks

    def add_to_plot(
        self,
        ax,
        alpha: float = 0.33,
        facecolor: str = "r",
        edgecolor: str = "r",
        **kwargs
    ):
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
        rect = Rectangle(
            tuple(self.start),
            *self.extent,
            alpha=alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
            **kwargs
        )
        ax.add_patch(rect)
