"""Module for describing different types of scans."""

from __future__ import annotations

import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
from ase import Atom, Atoms
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from abtem.array import ArrayObject
from abtem.core.axes import AxisMetadata, PositionsAxis, ScanAxis
from abtem.core.backend import get_array_module, validate_device
from abtem.core.chunks import validate_chunks
from abtem.core.ensemble import _wrap_with_array, unpack_blockwise_args
from abtem.core.fft import fft_shift_kernel
from abtem.core.grid import Grid, HasGrid2DMixin
from abtem.core.utils import get_dtype, itemset
from abtem.potentials.iam import BasePotential, validate_potential
from abtem.transfer import nyquist_sampling
from abtem.transform import ReciprocalSpaceMultiplication
from abtem.visualize.visualizations import Visualization

if TYPE_CHECKING:
    from abtem.prism.s_matrix import BaseSMatrix
    from abtem.waves import Probe, Waves


ScanWithSampling = Union["LineScan", "GridScan"]


def validate_scan(
    scan: Optional[Sequence | np.ndarray | BaseScan], probe: Probe | None = None
) -> BaseScan:
    """
    Validate that the input is a valid scan or a sequence of valid scan positions.

    Parameters
    ----------
    scan : Sequence or np.ndarray or BaseScan
        The scan or scan positions to validate. If None, a scan with a single position
        at (0, 0) is returned.
    probe : Probe or None
        If given the scan is matched to the extent of the probe.

    Returns
    -------
    validated_scan : BaseScan
        The validated scan object.
    """
    if not isinstance(scan, BaseScan) and scan is not None:
        scan = np.array(scan)

        if (
            (len(scan.shape) == 1 and not len(scan) == 2)
            or (len(scan.shape) == 2 and not scan.shape[1] == 2)
            or len(scan.shape) > 2
        ):
            raise ValueError(
                "scan must be a 1D sequence of length 2, a Nx2 array or a BaseScan"
            )

    validated_scan: BaseScan
    if scan is None and probe is None:
        validated_scan = CustomScan(np.zeros((1, 2)), squeeze=True)
    elif scan is None:
        validated_scan = CustomScan(np.zeros((0, 2)), squeeze=True)
    elif not isinstance(scan, BaseScan):
        validated_scan = CustomScan(scan, squeeze=True)
    else:
        validated_scan = scan

    if probe is not None:
        validated_scan = validated_scan.copy()
        validated_scan.match_probe(probe)

    return validated_scan


def _validate_scan_sampling(scan: ScanWithSampling, probe: Probe | BaseSMatrix):
    if scan.sampling is None:
        if not hasattr(probe, "semiangle_cutoff"):
            raise ValueError()

        if hasattr(probe, "dummy_probes"):
            probe = probe.dummy_probes()

        semiangle_cutoff = probe.aperture._max_semiangle_cutoff

        scan.sampling = 0.99 * nyquist_sampling(semiangle_cutoff, probe._valid_energy)


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

    def _get_weights(self):
        raise NotImplementedError

    @abstractmethod
    def match_probe(self, probe: Probe | BaseSMatrix):
        """Match the scan to a probe or s-matrix."""

    @property
    @abstractmethod
    def limits(self):
        """Lower left and upper right corner of the bounding box containing all
        positions in the scan."""

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
            return xp.ones(waves.gpts, dtype=get_dtype(complex=False))

        positions = xp.asarray(positions) / xp.asarray(waves.sampling)
        positions = positions.astype(get_dtype(complex=False))

        kernel = fft_shift_kernel(positions, shape=waves._valid_gpts)

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
        Scan positions [Å]. Anything that can be converted to a ndarray of shape (n, 3)
        is accepted. Default is (0., 0.).
    """

    def __init__(
        self, positions: np.ndarray | Sequence = (0.0, 0.0), squeeze: bool = False
    ):
        positions = np.array(positions, dtype=get_dtype(complex=False))

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
            self._positions = (
                np.array(probe.extent, dtype=get_dtype(complex=False))[None] / 2.0
            )

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
        new_scan = CustomScan(positions, **kwargs)
        new_scan = _wrap_with_array(new_scan, 1)
        return new_scan

    def _from_partitioned_args(self):
        if len(self.positions) == 0:
            return lambda *args, **kwargs: _wrap_with_array(self)
        return self._from_partitioned_args_func

    def _partition_args(self, chunks=None, lazy: bool = True):
        if len(self.positions) == 0:
            return ()

        chunks = self._validate_ensemble_chunks(chunks)
        cumchunks = tuple(np.cumsum(chunks[0]))
        positions = np.empty(len(chunks[0]), dtype=object)
        for i, (start_chunk, chunk) in enumerate(zip((0,) + cumchunks, chunks[0])):
            itemset(
                positions,
                i,
                {"positions": self._positions[start_chunk : start_chunk + chunk]},
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

            n = int(np.sum(mask))
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


def validate_coordinate(
    coordinate: float | tuple[float, float] | Atom | None,
    potential: BasePotential | Atoms | None = None,
    fractional: bool = False,
) -> tuple[float, float] | None:
    if isinstance(coordinate, Atom):
        if fractional:
            raise ValueError()
        coordinate = coordinate.x, coordinate.y
    elif isinstance(coordinate, (int, float)):
        coordinate = (float(coordinate), float(coordinate))
    elif isinstance(coordinate, (tuple, list, np.ndarray)):
        assert len(coordinate) == 2
        coordinate = (float(coordinate[0]), float(coordinate[1]))
    elif coordinate is None:
        return None
    else:
        raise ValueError(
            f"coordinate must be a float or a tuple of two floats, got {coordinate}"
        )

    if fractional and potential is None:
        raise ValueError("provide potential for fractional coordinates")
    elif fractional and potential is not None:
        potential = validate_potential(potential)

        if isinstance(potential, BasePotential):
            extent = potential._valid_extent
        else:
            assert potential is not None
            extent = potential

        coordinate = (
            extent[0] * coordinate[0],
            extent[1] * coordinate[1],
        )

    return coordinate


class LineScan(BaseScan):
    """
    A scan along a straight line.

    Parameters
    ----------
    start : two float or Atom, optional
        Start point of the scan [Å]. May be given as fractional coordinate if
        `fractional=True`. Default is (0., 0.).
    end : two float or Atom, optional
        End point of the scan [Å]. May be given as fractional coordinate if
        `fractional=True`.
        Default is None, the scan end point will match the extent of the potential.
    gpts : int, optional
        Number of scan positions. Default is None. Provide one of gpts or sampling.
    sampling : float, optional
        Sampling rate of scan positions [1 / Å]. Provide one of gpts or sampling.
        If not provided the sampling will match the Nyquist sampling of the Probe
        in a multislice simulation.
    endpoint : bool, optional
        If True, end is the last position. Otherwise, it is not included.
        Default is True.
    fractional : bool, optional
        If True, use fractional coordinates with respect to the given potential for
        `start` and `end`.
    potential : BasePotential or Atoms, optional
        Potential defining the grid with respect to which the fractional coordinates
        should be given.
    """

    def __init__(
        self,
        start: tuple[float, float] | Atom = (0.0, 0.0),
        end: tuple[float, float] | Atom | None = None,
        gpts: int | None = None,
        sampling: float | None = None,
        endpoint: bool = True,
        fractional: bool = False,
        potential: BasePotential | Atoms | None = None,
    ):
        self._gpts = gpts
        self._sampling = sampling

        self._start = validate_coordinate(start, potential, fractional)
        self._end = validate_coordinate(end, potential, fractional)

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
        """Angle of the line from `start` to `end` and the `x`-axis [deg.]."""
        direction = self.direction
        return np.arctan2(direction[1], direction[0])

    def add_margin(self, margin: float | tuple[float, float]):
        """
        Extend the line scan by adding a margin to the start and end of the line scan.

        Parameters
        ----------
        margin : float or tuple of float
            The margin added to the start and end of the linescan [Å]. If float the same
            margin is added.
        """
        if np.isscalar(margin):
            validated_margin = (margin, margin)
        elif isinstance(margin, tuple):
            assert len(margin) == 2
            validated_margin = margin
        else:
            raise ValueError("margin must be a float or a tuple of two floats")

        direction = self.direction

        self.start = tuple(np.array(self.start) - direction * validated_margin[0])
        self.end = tuple(np.array(self.end) + direction * validated_margin[1])
        return self

    @classmethod
    def at_position(
        cls,
        center: tuple[float, float] | Atom,
        extent: float = 1.0,
        angle: float = 0.0,
        gpts: int | None = None,
        sampling: float | None = None,
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
            center = (center.x, center.y)

        direction = np.array((np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))))

        start = (
            center[0] - extent / 2 * direction[0],
            center[1] - extent / 2 * direction[1],
        )
        end = (
            center[0] + extent / 2 * direction[0],
            center[1] + extent / 2 * direction[1],
        )

        return cls(
            start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint
        )

    def match_probe(self, probe: Probe | BaseSMatrix):
        """
        Sets sampling to the Nyquist frequency. If the start and end point of the scan
        is not given, set them to the lower and upper left corners of the probe extent.

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

        difference = np.array(self._end) - np.array(self._start)
        extent = np.linalg.norm(difference)
        assert isinstance(extent, (float | type(None)))
        return extent

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
        """True if the scan endpoint is the last position. Otherwise, the endpoint is
        not included."""
        return self._endpoint

    @property
    def limits(
        self,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        return self.start, self.end

    @property
    def gpts(self) -> int | None:
        """Number of grid points."""
        return self._gpts

    @gpts.setter
    def gpts(self, gpts: int):
        self._gpts = gpts
        self._adjust_sampling()

    @property
    def sampling(self) -> float | None:
        """Grid sampling [Å]."""
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: float):
        self._sampling = sampling
        self._adjust_gpts()

    @property
    def shape(self) -> tuple[int]:
        if self.gpts is None:
            raise RuntimeError("gpts is not defined")
        return (self.gpts,)

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
        assert self.sampling is not None
        return [
            ScanAxis(
                label="r",
                sampling=self.sampling,
                offset=0.0,
                units="Å",
                endpoint=self.endpoint,
            )
        ]

    @property
    def ensemble_shape(self):
        return self.shape

    def _out_ensemble_shape(self, array_object: ArrayObject) -> tuple[tuple[int, ...]]:
        return (self.ensemble_shape + array_object.ensemble_shape,)

    # def _out_ensemble_axes_metadata(
    #    self, array_object: ArrayObject | ArrayObjectSubclass, index: int = 0
    # ) -> list[AxisMetadata] | tuple[list[AxisMetadata], ...]:
    #    ensemble_axes_metadata = self.ensemble_axes_metadata
    #    #
    #    if len(_scan_axes(array_object)) > 0:
    #        for axis in ensemble_axes_metadata:
    #            axis._main = False
    #
    #    return [*ensemble_axes_metadata, *array_object.ensemble_axes_metadata]

    @property
    def _default_ensemble_chunks(self):
        return ("auto",)

    def _sort_into_extents(self, extents):
        raise NotImplementedError

    @staticmethod
    def _from_partitioned_args_func(*args):
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

            block = _wrap_with_array(
                LineScan(start=start, end=end, gpts=chunk, endpoint=False)
            )

            if lazy:
                block = da.from_array(block, chunks=1)

            blocks.append(block)

        if lazy:
            blocks = da.concatenate(blocks)
        else:
            blocks = np.concatenate(blocks)

        return (blocks,)

    def get_positions(
        self, chunks: int | None = None, lazy: bool = False
    ) -> np.ndarray:
        if self.gpts is None:
            raise RuntimeError("gpts is not defined")

        if self.start is None or self.end is None:
            raise RuntimeError("start and end is not defined")

        x = np.linspace(
            self.start[0],
            self.end[0],
            self.gpts,
            endpoint=self.endpoint,
            dtype=get_dtype(complex=False),
        )
        y = np.linspace(
            self.start[1],
            self.end[1],
            self.gpts,
            endpoint=self.endpoint,
            dtype=get_dtype(complex=False),
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
        assert isinstance(self.start, tuple)
        assert isinstance(self.end, tuple)
        assert isinstance(self.extent, float)

        if width:
            rect = Rectangle(self.start, self.extent, width, angle=self.angle, **kwargs)
            ax.add_patch(rect)
        else:
            ax.plot(
                [self.start[0], self.end[0]], [self.start[1], self.end[1]], **kwargs
            )


class GridScan(HasGrid2DMixin, BaseScan):
    """
    A scan over a regular grid for calculating scanning transmission electron
    microscopy.

    Parameters
    ----------
    start : two float or Atom, optional
        Start corner of the scan [Å]. May be given as fractional coordinate if
        `fractional=True`. Default is (0., 0.).
    end : two float or Atom, optional
        End corner of the scan [Å]. May be given as fractional coordinate
        if `fractional=True`.
        Default is None, the scan end point will match the extent of the potential.
    gpts : two int, optional
        Number of scan positions in the `x`- and `y`-direction of the scan. Provide one
        of gpts or sampling.
    sampling : two float, optional
        Sampling rate of scan positions [1 / Å]. Provide one of gpts or sampling.
        If not provided the sampling will match the Nyquist sampling of the  Probe
        in a multislice simulation.
    endpoint : bool, optional
        If True, end is the last position. Otherwise, it is not included.
        Default is False.
    fractional : bool, optional
        If True, use fractional coordinates with respect to the given potential for
        `start` and `end`.
    potential : BasePotential or Atoms, optional
        Potential defining the grid with respect to which the fractional coordinates
        should be given.
    """

    def __init__(
        self,
        start: Union[float, Tuple[float, float], Atom] = (0.0, 0.0),
        end: tuple[float, float] | Atom | None = None,
        gpts: int | tuple[int, int] | None = None,
        sampling: float | tuple[float, float] | None = None,
        endpoint: bool | tuple[bool, bool] = False,
        fractional: bool = False,
        potential: BasePotential | Atoms | None = None,
    ):
        super().__init__()

        self._start = validate_coordinate(
            coordinate=start, potential=potential, fractional=fractional
        )
        self._end = validate_coordinate(
            coordinate=end, potential=potential, fractional=fractional
        )

        if self._start is not None and self._end is not None:
            extent = (self._end[0] - self._start[0], self._end[1] - self._start[1])
            if all(d <= 0.0 for d in extent):
                raise ValueError(f"scan extent must be positive, got {extent}")
        else:
            extent = None

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
        """True if the scan endpoint is the last position. Otherwise, the endpoint is
        not included."""
        assert len(self.grid.endpoint) == 2
        return self.grid.endpoint

    @property
    def shape(self) -> tuple[int, int]:
        return self._valid_gpts

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
        Sets sampling to the Nyquist frequency. If the start and end point of the scan
        is not given, set them to the lower left and upper right corners of the probe
        extent.

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
            dtype=get_dtype(complex=False),
        )

    def _y_coordinates(self):
        return np.linspace(
            self.start[1],
            self.end[1],
            self.gpts[1],
            endpoint=self.endpoint[1],
            dtype=get_dtype(complex=False),
        )

    def get_positions(self) -> np.ndarray:
        if self.start is None or self.end is None or self.gpts is None:
            raise RuntimeError("start, end, or gpts is not defined")

        xi = [
            np.linspace(
                start, end, gpts, endpoint=endpoint, dtype=get_dtype(complex=False)
            )
            for start, end, gpts, endpoint in zip(
                self.start, self.end, self.gpts, self.endpoint
            )
        ]

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
                x_chunks_new.append(int(x_chunks[unique.index(i)]))
            else:
                x_chunks_new.append(0)

        y = np.linspace(
            self.start[1], self.end[1], self.gpts[1], endpoint=self.endpoint[1]
        )

        separators = [l for _, l in extents[1]]
        unique, y_chunks = np.unique(np.digitize(y, separators), return_counts=True)
        unique = list(unique)

        y_chunks_new = []
        for i in range(len(separators)):
            if i in unique:
                y_chunks_new.append(int(y_chunks[unique.index(i)]))
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

    # def _out_ensemble_axes_metadata(
    #    self, array_object: ArrayObject | ArrayObjectSubclass, index: int = 0
    # ) -> list[AxisMetadata] | tuple[list[AxisMetadata], ...]:
    #    ensemble_axes_metadata = self.ensemble_axes_metadata
    #
    #    if len(_scan_axes(array_object)) > 0:
    #        for axis in ensemble_axes_metadata:
    #            axis._main = False
    #
    #    return [*ensemble_axes_metadata, *array_object.ensemble_axes_metadata]

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        x_scan, y_scan = unpack_blockwise_args(args)
        start = (x_scan["start"], y_scan["start"])
        end = (x_scan["end"], y_scan["end"])
        gpts = (x_scan["gpts"], y_scan["gpts"])
        endpoint = (x_scan["endpoint"], y_scan["endpoint"])
        new_scan = cls(start=start, end=end, gpts=gpts, endpoint=endpoint, **kwargs)
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
            Additional options for matplotlib.patches.Rectangle used for scan area
            visualization as keyword arguments.
        """

        if isinstance(ax, Visualization):
            axes = np.array(ax.axes).ravel()
            for ax in axes:
                self.add_to_plot(
                    ax, alpha=alpha, facecolor=facecolor, edgecolor=edgecolor, **kwargs
                )

        if self.start is None or self.extent is None:
            raise RuntimeError("start or extent is not defined")

        rect = Rectangle(
            xy=self.start,
            width=self.extent[0],
            height=self.extent[1],
            alpha=alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
            **kwargs,
        )
        ax.add_patch(rect)
