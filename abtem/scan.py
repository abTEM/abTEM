"""Module for describing different types of scans."""
from __future__ import annotations

import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
from ase import Atom, Atoms
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from abtem.array import ArrayObject, T
from abtem.core.axes import ScanAxis, PositionsAxis, AxisMetadata
from abtem.core.backend import get_array_module, validate_device
from abtem.core.chunks import validate_chunks
from abtem.core.ensemble import _wrap_with_array, unpack_blockwise_args
from abtem.core.fft import fft_shift_kernel
from abtem.core.grid import Grid, HasGridMixin
from abtem.potentials.iam import BasePotential, _validate_potential
from abtem.transfer import nyquist_sampling
from abtem.transform import ReciprocalSpaceMultiplication

if TYPE_CHECKING:
    from abtem.waves import Waves, Probe
    from abtem.prism.s_matrix import BaseSMatrix


def _validate_scan(scan:np.ndarray | BaseScan, probe:Probe=None):
    if scan is None and probe is None:
        scan = CustomScan(np.zeros((1,2)), squeeze=True)
    elif scan is None:
        scan = CustomScan(np.zeros((0, 2)), squeeze=True)

    if not isinstance(scan, BaseScan):
        scan = CustomScan(scan)

    if probe is not None:
        scan = scan.copy()
        scan.match_probe(probe)

    return scan


def _validate_scan_sampling(scan, probe):
    if scan.sampling is None:
        if not hasattr(probe, "semiangle_cutoff"):
            raise ValueError()

        if hasattr(probe, "dummy_probes"):
            probe = probe.dummy_probes()

        semiangle_cutoff = probe.aperture._max_semiangle_cutoff

        scan.sampling = 0.99 * nyquist_sampling(semiangle_cutoff, probe.energy)


class BaseScan(ReciprocalSpaceMultiplication):
    """Abstract class to describe scans."""

    def __len__(self) -> int:
        return self.num_positions

    @property
    def num_positions(self) -> int:
        """Number of probe positions in the scan."""
        return len(self.get_positions())

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape the scan."""
        pass

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return self.shape

    @property
    def _default_ensemble_chunks(self):
        return ("auto",) * len(self.ensemble_shape)

    @abstractmethod
    def get_positions(self, *args, **kwargs) -> np.ndarray:
        """Get the scan positions as numpy array."""
        pass

    def _get_weights(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def limits(self):
        """Lower left and upper right corner of the bounding box containing all positions in the scan."""
        pass

    @abstractmethod
    def _sort_into_extents(
        self,
        extents: tuple[
            tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]
        ],
    ):
        pass

    def _evaluate_kernel(self, waves: Waves) -> np.ndarray:
        """
        Evaluate the array to be multiplied with the waves in reciprocal space.

        Parameters
        ----------
        waves : Waves, optional
            If given, the array will be evaluated to match the provided waves.

        Returns
        -------
        kernel : np.ndarray or dask.array.Array
        """
        device = validate_device(waves.device)
        xp = get_array_module(device)

        waves.grid.check_is_defined()

        positions = self.get_positions()
        if len(positions) == 0:
            return xp.ones(waves.gpts, dtype=xp.complex64)

        positions = xp.asarray(positions) / xp.asarray(
            waves.sampling
        ).astype(np.float32)

        kernel = fft_shift_kernel(positions, shape=waves.gpts)

        try:
            kernel *= self._get_weights()[..., None, None]
        except NotImplementedError:
            pass

        return kernel


#
# class SourceDistribution(BaseScan):
#     """
#     Distribution of electron source offsets.
#
#     Parameters
#     ----------
#     distribution : 2D :class:`.BaseDistribution`
#         Distribution describing the positions and weights of the source offsets.
#     """
#
#     def __init__(self, distribution: BaseDistribution):
#         self._distribution = distribution
#
#     @property
#     def shape(self):
#         return self._distribution.shape
#
#     def get_positions(self):
#         xi = [factor.values for factor in self._distribution.factors]
#         return np.stack(np.meshgrid(*xi, indexing="ij"), axis=-1)
#
#     def _get_weights(self):
#         return self._distribution.weights
#
#     @property
#     def ensemble_axes_metadata(self):
#         return [PositionsAxis()] * len(self.shape)
#
#     def ensemble_blocks(self, chunks=None):
#         if chunks is None:
#             chunks = self._default_ensemble_chunks
#
#         chunks = validate_chunks(self.ensemble_shape, chunks, limit=None)
#
#         blocks = ()
#         for parameter, n in zip(self._distribution.factors, chunks):
#             blocks += (parameter.divide(n, lazy=True),)
#
#         return blocks
#
#     def ensemble_partial(self):
#         def distribution(*args):
#             factors = [arg.item() for arg in args]
#             dist = SourceDistribution(AxisAlignedDistributionND(factors))
#             arr = np.empty((1,) * len(args), dtype=object)
#             arr.itemset(dist)
#             return arr
#
#         return distribution
#
#     @property
#     def limits(self):
#         pass


class CustomScan(BaseScan):
    """
    Custom scan based on explicit 2D probe positions.

    Parameters
    ----------
    positions : np.ndarray, optional
        Scan positions [Å]. Anything that can be converted to an ndarray of shape (n, 3) is accepted. Default is
        (0., 0.).
    """

    def __init__(self, positions: np.ndarray = (0.0, 0.0), squeeze: bool = False):

        positions = np.array(positions, dtype=np.float32)

        if len(positions.shape) == 1:
            positions = positions[None]

        self._positions = positions
        self._squeeze = squeeze

        super().__init__()

    def match_probe(self, probe: Probe | BaseSMatrix):
        """
        Sets the positions to a single position in the center of the probe extent.

        Parameters
        ----------
        probe : Probe or BaseSMatrix
            The matched probe or s-matrix.
        """
        if len(self.positions) == 0:
            probe.grid.check_is_defined()
            self._positions = np.array(probe.extent, dtype=np.float32)[None] / 2.0

    @property
    def ensemble_axes_metadata(self):
        if len(self.positions) == 0:
            return []

        return [
            PositionsAxis(
                values=tuple(
                    (float(position[0]), float(position[1]))
                    for position in self.positions
                ),
                _squeeze=self._squeeze,
            )
        ]

    @staticmethod
    def _from_partitioned_args_func(*args, **kwargs):
        scan = unpack_blockwise_args(args)

        positions = scan[0]["positions"]

        new_scan = CustomScan(positions)

        new_scan = _wrap_with_array(new_scan, 1)
        return new_scan

    def _from_partitioned_args(self):
        if len(self.positions) == 0:
            return lambda *args, **kwargs : _wrap_with_array(self)
        return self._from_partitioned_args_func

    def _partition_args(self, chunks=None, lazy: bool = True):
        if len(self.positions) == 0:
            return ()

        chunks = self._validate_ensemble_chunks(chunks)
        cumchunks = tuple(np.cumsum(chunks[0]))
        positions = np.empty(len(chunks[0]), dtype=object)
        for i, (start_chunk, chunk) in enumerate(zip((0,) + cumchunks, chunks[0])):
            positions.itemset(
                i, {"positions": self._positions[start_chunk : start_chunk + chunk]}
            )

        if lazy:
            positions = da.from_array(positions, chunks=1)

        return (positions,)

    def _sort_into_extents(self, extents):
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
        if len(self.positions) == 0:
            return ()
        return self.positions.shape[:-1]

    @property
    def positions(self):
        """Scan positions [Å]."""
        return self._positions

    @property
    def limits(self):
        return [
            (np.min(self.positions[:, 0]), np.min(self.positions[:, 1])),
            (np.max(self.positions[:, 0]), np.max(self.positions[:, 1])),
        ]

    def get_positions(self) -> np.ndarray:
        return self._positions


def _validate_coordinate(
    coordinate: tuple[float, float] | Atom,
    potential: BasePotential | Atoms = None,
    fractional: bool = False,
) -> tuple[float, float]:
    if isinstance(coordinate, Atom):
        if fractional:
            raise ValueError()

        coordinate = coordinate.x, coordinate.y

    if fractional:
        potential = _validate_potential(potential)

        if isinstance(potential, BasePotential):
            if potential is None:
                raise ValueError("provide potential for fractional coordinates")

            potential = _validate_potential(potential)
            extent = potential.extent
        else:
            extent = potential

        coordinate = (
            extent[0] * coordinate[0],
            extent[1] * coordinate[1],
        )

    coordinate = coordinate if coordinate is None else tuple(coordinate)

    return coordinate


def _validate_coordinates(
    start: tuple[float, float] | Atom,
    end: tuple[float, float] | Atom,
    potential: BasePotential | Atoms,
    fractional: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:

    if fractional:
        potential = _validate_potential(potential)

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
    start : two float or Atom, optional
        Start point of the scan [Å]. May be given as fractional coordinate if `fractional=True`. Default is (0., 0.).
    end : two float or Atom, optional
        End point of the scan [Å]. May be given as fractional coordinate if `fractional=True`.
        Default is None, the scan end point will match the extent of the potential.
    gpts : int, optional
        Number of scan positions. Default is None. Provide one of gpts or sampling.
    sampling : float, optional
        Sampling rate of scan positions [1 / Å]. Provide one of gpts or sampling. If not provided the sampling will
        match the Nyquist sampling of the Probe in a multislice simulation.
    endpoint : bool, optional
        If True, end is the last position. Otherwise, it is not included. Default is True.
    fractional : bool, optional
        If True, use fractional coordinates with respect to the given potential for `start` and `end`.
    potential : BasePotential or Atoms, optional
        Potential defining the grid with respect to which the fractional coordinates should be given.
    """

    def __init__(
        self,
        start: tuple[float, float] | Atom = (0.0, 0.0),
        end: tuple[float, float] | Atom = None,
        gpts: int = None,
        sampling: float = None,
        endpoint: bool = True,
        fractional: bool = False,
        potential: BasePotential | Atoms = None,
    ):

        self._gpts = gpts
        self._sampling = sampling

        self._start, self._end = _validate_coordinates(
            start, end, potential, fractional
        )

        self._endpoint = endpoint
        self._adjust_gpts()
        self._adjust_sampling()

        super().__init__()

    @property
    def direction(self):
        """Normal vector pointing from `start` to `end`."""
        direction = np.array(self.end) - np.array(self.start)
        return direction / np.linalg.norm(direction)

    @property
    def angle(self):
        """Angle of the line from `start` to `end` and the x-axis [deg.]."""
        direction = self.direction
        return np.arctan2(direction[1], direction[0])

    def add_margin(self, margin: float | tuple[float, float]):
        """
        Extend the line scan by adding a margin to the start and end of the line scan.

        Parameters
        ----------
        margin : float or tuple of float
            The margin added to the start and end of the linescan [Å]. If float the same margin is added.
        """
        if not np.isscalar(margin):
            margin = (margin,) * 2

        direction = self.direction

        self.start = tuple(np.array(self.start) - direction * margin[0])
        self.end = tuple(np.array(self.end) + direction * margin[1])
        return self

    @classmethod
    def at_position(
        cls,
        center: tuple[float, float] | Atom,
        extent: float = 1.0,
        angle: float = 0.0,
        gpts: int = None,
        sampling: float = None,
        endpoint: bool = True,
    ):
        """
        Make a line scan centered at a given position.

        Parameters
        ----------
        center : two float
            Center position of the line [Å]. May be given as an Atom.
        angle : float
            Angle of the line [deg.].
        extent : float
            Extent of the line [Å].
        gpts : int
            Number of grid points along the line.
        sampling : float
            Sampling of grid points along the line [Å].
        endpoint : bool
            Sets whether the ending position is included or not.

        Returns
        -------
        line_scan : LineScan
        """

        if isinstance(center, Atom):
            position = (center.x, center.y)

        direction = np.array((np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))))

        start = tuple(np.array(center) - extent / 2 * direction)
        end = tuple(np.array(center) + extent / 2 * direction)

        return cls(
            start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint
        )

    def match_probe(self, probe: Probe | BaseSMatrix):
        """
        Sets sampling to the Nyquist frequency. If the start and end point of the scan is not given, set them to the
        lower and upper left corners of the probe extent.

        Parameters
        ----------
        probe : Probe or BaseSMatrix
            The matched probe or s-matrix.
        """
        if self.start is None:
            self.start = (0.0, 0.0)

        if self.end is None and probe.extent is not None:
            self.end = (0.0, probe.extent[1])

        _validate_scan_sampling(self, probe)

    @property
    def extent(self) -> float | None:
        """Grid extent [Å]."""
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
        """True if the scan endpoint is the last position. Otherwise, the endpoint is not included."""
        return self._endpoint

    @property
    def limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.start, self.end

    @property
    def gpts(self) -> int:
        """Number of grid points."""
        return self._gpts

    @gpts.setter
    def gpts(self, gpts: int):
        self._gpts = gpts
        self._adjust_sampling()

    @property
    def sampling(self) -> float:
        """Grid sampling [Å]."""
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: float):
        self._sampling = sampling
        self._adjust_gpts()

    @property
    def shape(self) -> tuple[int]:
        return (self._gpts,)

    @property
    def metadata(self):
        return {"start": self.start, "end": self.end}

    @property
    def start(self) -> tuple[float, float] | None:
        """
        Start point of the scan [Å].
        """
        return self._start

    @start.setter
    def start(self, start: tuple[float, float]):
        if start is not None:
            start = (float(start[0]), float(start[1]))

        self._start = start
        self._adjust_gpts()

    @property
    def end(self) -> tuple[float, float] | None:
        """
        End point of the scan [Å].
        """
        return self._end

    @end.setter
    def end(self, end: tuple[float, float]):
        if end is not None:
            end = (float(end[0]), float(end[1]))
        self._end = end
        self._adjust_gpts()

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
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

    def _out_ensemble_shape(self, array_object) -> tuple[int, ...]:
        return self.ensemble_shape + array_object.ensemble_shape

    def _out_ensemble_axes_metadata(
        self, array_object: ArrayObject | T
    ) -> list[AxisMetadata] | tuple[list[AxisMetadata], ...]:
        return [*self.ensemble_axes_metadata, *array_object.ensemble_axes_metadata]

    @property
    def _default_ensemble_chunks(self):
        return ("auto",)

    def _sort_into_extents(self, extents):
        raise NotImplementedError

    @staticmethod
    def _from_partitioned_args_func(*args, **kwargs):
        args = unpack_blockwise_args(args)
        line_scan = args[0]
        return _wrap_with_array(line_scan)

    def _from_partitioned_args(self):
        return self._from_partitioned_args_func

    def _partition_args(self, chunks=None, lazy: bool = True):
        if chunks is None:
            chunks = self._default_ensemble_chunks

        chunks = validate_chunks(self.ensemble_shape, chunks)

        direction = np.array(self.end) - np.array(self.start)
        direction = direction / np.linalg.norm(direction, axis=0)

        cumchunks = tuple(np.cumsum(chunks[0]))

        blocks = []
        for i, (start_chunk, chunk) in enumerate(zip((0,) + cumchunks, chunks[0])):
            start = np.array(self.start) + start_chunk * self.sampling * direction

            end = start + self.sampling * chunk * direction

            block = _wrap_with_array(LineScan(start=start, end=end, gpts=chunk, endpoint=False))

            if lazy:
                block = da.from_array(block, chunks=1)

            blocks.append(block)

        if lazy:
            blocks = da.concatenate(blocks)
        else:
            blocks = np.concatenate(blocks)

        return (blocks,)

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

    def add_to_axes(self, ax: Axes, width: float = 0.0, **kwargs):
        """
        Add a visualization of a scan line to a matplotlib plot.

        Parameters
        ----------
        ax : matplotlib Axes
            The axes of the matplotlib plot the visualization should be added to.
        width : float, optional
            Width of line [Å].
        kwargs :
            Additional options for matplotlib.pyplot.plot as keyword arguments.
        """

        if width:
            rect = Rectangle(
                tuple(self.start), self.extent, width, angle=self.angle, **kwargs
            )
            ax.add_patch(rect)
        else:
            ax.plot(
                [self.start[0], self.end[0]], [self.start[1], self.end[1]], **kwargs
            )


class GridScan(HasGridMixin, BaseScan):
    """
    A scan over a regular grid for calculating scanning transmission electron microscopy.

    Parameters
    ----------
    start : two float or Atom, optional
        Start corner of the scan [Å]. May be given as fractional coordinate if `fractional=True`. Default is (0., 0.).
    end : two float or Atom, optional
        End corner of the scan [Å]. May be given as fractional coordinate if `fractional=True`.
        Default is None, the scan end point will match the extent of the potential.
    gpts : two int, optional
        Number of scan positions in the `x`- and `y`-direction of the scan. Provide one of gpts or sampling.
    sampling : two float, optional
        Sampling rate of scan positions [1 / Å]. Provide one of gpts or sampling. If not provided the sampling will
        match the Nyquist sampling of the  Probe in a multislice simulation.
    endpoint : bool, optional
        If True, end is the last position. Otherwise, it is not included. Default is False.
    fractional : bool, optional
        If True, use fractional coordinates with respect to the given potential for `start` and `end`.
    potential : BasePotential or Atoms, optional
        Potential defining the grid with respect to which the fractional coordinates should be given.
    """

    def __init__(
        self,
        start: tuple[float, float] | Atom = (0.0, 0.0),
        end: tuple[float, float] | Atom = None,
        gpts: int | tuple[int, int] = None,
        sampling: float | tuple[float, float] = None,
        endpoint: bool | tuple[bool, bool] = False,
        fractional: bool = False,
        potential: BasePotential | Atoms = None,
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

    @property
    def limits(self):
        return [self.start, self.end]

    @property
    def endpoint(self) -> tuple[bool, bool]:
        """True if the scan endpoint is the last position. Otherwise, the endpoint is not included."""
        return self.grid.endpoint

    @property
    def shape(self) -> tuple[int, int]:
        return self.gpts

    @property
    def start(self) -> tuple[float, float] | None:
        """Start corner of the scan [Å]."""
        return self._start

    @start.setter
    def start(self, start: tuple[float, float]):
        self._start = start
        self._adjust_extent()

    @property
    def end(self) -> tuple[float, float] | None:
        """End corner of the scan [Å]."""
        return self._end

    @end.setter
    def end(self, end: tuple[float, float]):
        self._end = end
        self._adjust_extent()

    def _adjust_extent(self):
        if self.start is None or self.end is None:
            return

        self.extent = np.array(self.end) - self.start

    def match_probe(self, probe: Probe | BaseSMatrix):
        """
        Sets sampling to the Nyquist frequency. If the start and end point of the scan is not given, set them to the
        lower left and upper right corners of the probe extent.

        Parameters
        ----------
        probe : Probe or BaseSMatrix
            The matched probe or s-matrix.
        """
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

    def _sort_into_extents(self, extents):

        x = np.linspace(
            self.start[0], self.end[0], self.gpts[0], endpoint=self.endpoint[0]
        )

        separators = [l for _, l in extents[0]]

        unique, x_chunks = np.unique(np.digitize(x, separators), return_counts=True)
        unique = list(unique)

        x_chunks_new = []
        for i in range(len(separators)):
            if i in unique:
                x_chunks_new.append(x_chunks[unique.index(i)])
            else:
                x_chunks_new.append(0)

        y = np.linspace(
            self.start[1], self.end[1], self.gpts[1], endpoint=self.endpoint[1]
        )

        separators = [l for _, l in extents[1]]
        unique, y_chunks = np.unique(
            np.digitize(y, [l for _, l in extents[1]]), return_counts=True
        )
        unique = list(unique)

        y_chunks_new = []
        for i in range(len(separators)):
            if i in unique:
                y_chunks_new.append(y_chunks[unique.index(i)])
            else:
                y_chunks_new.append(0)

        return self, (tuple(x_chunks_new), tuple(y_chunks_new))

    @property
    def ensemble_axes_metadata(self):
        axes_metadata = []
        for label, sampling, offset, endpoint in zip(
            ("x", "y"), self.sampling, self.start, self.endpoint
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
        x_scan, y_scan = unpack_blockwise_args(args)
        start = (x_scan["start"], y_scan["start"])
        end = (x_scan["end"], y_scan["end"])
        gpts = (x_scan["gpts"], y_scan["gpts"])
        endpoint = (x_scan["endpoint"], y_scan["endpoint"])
        new_scan = cls(start=start, end=end, gpts=gpts, endpoint=endpoint)
        new_scan = _wrap_with_array(new_scan, 2)
        return new_scan

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
        chunks = self._validate_ensemble_chunks(chunks)
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
        **kwargs,
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
            **kwargs,
        )
        ax.add_patch(rect)
