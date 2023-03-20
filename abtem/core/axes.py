import dataclasses
from copy import copy
from dataclasses import dataclass
from numbers import Number
from typing import List, Tuple, Union
from tabulate import tabulate
import numpy as np
import dask.array as da
from abtem.core.chunks import validate_chunks, iterate_chunk_ranges
from abtem.core import config
from abtem.core.utils import safe_equality

from abtem.core.units import _get_conversion_factor, _format_units, _validate_units


@dataclass(eq=False, repr=False, unsafe_hash=True)
class AxisMetadata:
    _concatenate: bool = True
    _events: bool = False
    label: str = "unknown"
    _tex_label = None

    def _tabular_repr_data(self, n):
        return [self.format_type(), self.format_label(), self.format_coordinates(n)]

    def format_coordinates(self, n: int = None):
        return "-"

    def __eq__(self, other):
        return safe_equality(self, other)

    def coordinates(self, n):
        return np.arange(n)

    def format_type(self):
        return self.__class__.__name__

    def format_label(self):
        return f"{self.label}"

    def format_title(self, *args, **kwargs):
        return f"{self.label}"

    def item_metadata(self, item):
        return {}

    def __getitem__(self, item):
        return self

    def _to_blocks(self, chunks):
        arr = np.empty((len(chunks[0]),), dtype=object)
        for i, slic in iterate_chunk_ranges(chunks):
            arr[i] = copy(self)
        arr = da.from_array(arr, chunks=1)
        return arr

    def copy(self):
        return copy(self)

    def plot_extent(self):
        raise NotImplementedError


@dataclass(eq=False, repr=False, unsafe_hash=True)
class UnknownAxis(AxisMetadata):
    label: str = "unknown"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class SampleAxis(AxisMetadata):
    pass


def format_label_with_units(label, units=None, old_units=None) -> str:
    units = _validate_units(units, old_units)

    if config.get("visualize.use_tex", False):
        return f"${label} \ [{_format_units(units)}]$"
    else:
        return f"{label} [{units}]"


def latex_float(f, formatting):
    float_str = f"{f:>{formatting}}"
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return f"{base} \\times 10^{{{int(exponent)}}}"
    else:
        return float_str


def format_value(value, formatting):
    if isinstance(value, float):
        if config.get("visualize.use_tex", False):
            return latex_float(value, formatting)
        else:
            return f"{value:>{formatting}}"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class LinearAxis(AxisMetadata):
    sampling: float = 1.0
    units: str = ""
    offset: float = 0.0
    _ensemble_mean: bool = False

    def format_coordinates(self, n: int = None) -> str:
        coordinates = self.coordinates(n)
        if n > 3:
            coordinates = [f"{coord:.2f}" for coord in coordinates[[0, 1, -1]]]
            return f"{coordinates[0]} {coordinates[1]} ... {coordinates[2]}"
        else:
            return " ".join([f"{coord:.2f}" for coord in coordinates])

    def coordinates(self, n: int) -> np.ndarray:
        return np.linspace(
            self.offset, self.offset + self.sampling * n, n, endpoint=False
        )

    def format_label(self, units=None) -> str:
        return format_label_with_units(self.label, units, self.units)

    def concatenate(self, other: AxisMetadata) -> "LinearAxis":
        if not self._concatenate:
            raise RuntimeError()

        if not self.__eq__(other):
            raise RuntimeError()

        return self


@dataclass(eq=False, repr=False, unsafe_hash=True)
class RealSpaceAxis(LinearAxis):
    sampling: float = 1.0
    units: str = "pixels"
    endpoint: bool = True


@dataclass(eq=False, repr=False, unsafe_hash=True)
class FourierSpaceAxis(LinearAxis):
    sampling: float = 1.0
    units: str = "pixels"
    fftshift: bool = True
    _concatenate: bool = False


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ScanAxis(RealSpaceAxis):
    start: Tuple[float, float] = None
    end: Tuple[float, float] = None


@dataclass(eq=False, repr=False, unsafe_hash=True)
class OrdinalAxis(AxisMetadata):
    values: tuple = ()

    def format_title(self, formatting, **kwargs):
        return f"{self.values[0]}"

    def concatenate(self, other):
        if not safe_equality(self, other, ("values",)):
            raise RuntimeError()

        kwargs = dataclasses.asdict(self)
        kwargs["values"] = kwargs["values"] + other.values

        return self.__class__(**kwargs)  # noqa

    def compatible(self, other):
        return

    def __len__(self):
        return len(self.values)

    def __post_init__(self):
        if not isinstance(self.values, tuple):
            try:
                self.values = tuple(self.values)
            except TypeError:
                raise ValueError()

    def item_metadata(self, item):
        return {self.label: self.values[item]}

    def __getitem__(self, item):
        kwargs = dataclasses.asdict(self)

        if isinstance(item, Number):
            kwargs["values"] = (kwargs["values"][item],)
        else:
            array = np.empty(len(kwargs["values"]), dtype=object)
            array[:] = kwargs["values"]
            kwargs["values"] = tuple(array[item])

        return self.__class__(**kwargs)  # noqa

    def coordinates(self, n):
        return self.values

    def _to_blocks(self, chunks):
        chunks = validate_chunks(shape=(len(self),), chunks=chunks)

        arr = np.empty((len(chunks[0]),), dtype=object)
        for i, slic in iterate_chunk_ranges(chunks):
            arr[i] = self[slic]

        arr = da.from_array(arr, chunks=1)

        return arr


@dataclass(eq=False, repr=False, unsafe_hash=True)
class NonLinearAxis(OrdinalAxis):
    units: str = "unknown"

    def format_label(self, units=None):
        return format_label_with_units(self.label, units, self.units)

    def format_coordinates(self, n: int = None):
        if len(self.values) > 3:
            values = [f"{self.values[i]:.2f}" for i in [0, 1, -1]]
            return f"{values[0]} {values[1]} ... {values[-1]}"
        else:
            return " ".join([f"{value:.2f}" for value in self.values])

    def format_title(self, formatting, units=None, include_label=True):
        value = self.values[0] #* _get_conversion_factor(units, self.units)

        units = _validate_units(units,self.units)

        if include_label:
            label = f"{self.label} = "
        else:
            label = ""

        if config.get("visualize.use_tex", False):
            return f"$\mathrm{{{label}}}{format_value(value, formatting)} \ {_format_units(units)}$"
        else:
            return f"{label}{value:>{formatting}} {units}"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class AxisAlignedTiltAxis(NonLinearAxis):
    units: str = "mrad"
    direction: str = "x"

    @property
    def tilt(self):
        if self.direction == "x":
            values = tuple((value, 0.0) for value in self.values)
        elif self.direction == "y":
            values = tuple((0.0, value) for value in self.values)
        else:
            raise RuntimeError()

        return values

    _ensemble_mean: bool = False


@dataclass(eq=False, repr=False, unsafe_hash=True)
class TiltAxis(OrdinalAxis):
    units: str = "mrad"
    _ensemble_mean: bool = False

    @property
    def tilt(self):
        return self.values


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ThicknessAxis(NonLinearAxis):
    label: str = "thickness"
    units: str = "Å"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ParameterAxis(NonLinearAxis):
    label: str = ""
    _ensemble_mean: bool = False


@dataclass(eq=False, repr=False, unsafe_hash=True)
class PositionsAxis(OrdinalAxis):
    label: str = "x, y"
    units: str = "Å"

    def format_title(self, formatting, units=None,  include_label=True):
        formatted = ", ".join(
            tuple(f"{value:>{formatting}}" for value in self.values[0])
        )
        if include_label:
            return f"{self.label} = {formatted} {self.units}"
        else:
            return f"{formatted} {self.units}"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class FrozenPhononsAxis(AxisMetadata):
    label: str = "Frozen phonons"
    _ensemble_mean: bool = False


@dataclass(eq=False, repr=False, unsafe_hash=True)
class PrismPlaneWavesAxis(AxisMetadata):
    pass


def axis_to_dict(axis: AxisMetadata):
    d = dataclasses.asdict(axis)
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = tuple(value.tolist())

    d["type"] = axis.__class__.__name__
    return d


def axis_from_dict(d):
    cls = globals()[d["type"]]
    return cls(**{key: value for key, value in d.items() if key != "type"})


def format_axes_metadata(axes_metadata, shape):
    data = []
    for axis, n in zip(axes_metadata, shape):
        data += [axis._tabular_repr_data(n)]

    return tabulate(data, headers=["type", "label", "coordinates"], tablefmt="simple")


def _iterate_axes_type(has_axes, axis_type):
    for i, axis_metadata in enumerate(has_axes.axes_metadata):
        if isinstance(axis_metadata, axis_type):
            yield axis_metadata


def _find_axes_type(has_axes, axis_type):
    indices = ()
    for i, _ in enumerate(_iterate_axes_type(has_axes, axis_type)):
        indices += (i,)

    return indices


class AxesMetadataList(list):
    def __init__(self, l, shape):
        self._shape = shape
        super().__init__(l)

    def __repr__(self):
        return format_axes_metadata(self, self._shape)


class HasAxes:
    base_shape: Tuple[int, ...]
    ensemble_shape: Tuple[int, ...]
    base_axes_metadata: List[AxisMetadata]
    ensemble_axes_metadata: List[AxisMetadata]

    @property
    def axes_metadata(self) -> AxesMetadataList:
        """
        List of AxisMetadata.
        """
        return AxesMetadataList(
            self.ensemble_axes_metadata + self.base_axes_metadata, self.shape
        )

    @property
    def num_base_axes(self) -> int:
        """
        Number of base axes.
        """
        return len(self.base_axes_metadata)

    @property
    def num_ensemble_axes(self) -> int:
        """
        Number of ensemble axes.
        """
        return len(self.ensemble_axes_metadata)

    @property
    def num_axes(self) -> int:
        """
        Number of axes.
        """
        return self.num_ensemble_axes + self.num_base_axes

    @property
    def base_axes(self) -> Tuple[int, ...]:
        """
        Axis indices of base axes.
        """
        return tuple(
            range(self.num_ensemble_axes, self.num_ensemble_axes + self.num_base_axes)
        )

    @property
    def ensemble_axes(self) -> Tuple[int, ...]:
        """
        Axis indices of ensemble axes.
        """
        return tuple(range(self.num_ensemble_axes))

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The size of each axis.
        """
        return self.ensemble_shape + self.base_shape

    def _check_axes_metadata(self):
        if len(self.shape) != self.num_axes:
            raise RuntimeError(
                f"number of dimensions ({len(self.shape)}) does not match number of axis metadata items "
                f"({self.num_axes})"
            )

        for n, axis in zip(self.shape, self.axes_metadata):
            if isinstance(axis, OrdinalAxis) and len(axis) != n:
                raise RuntimeError(
                    f"number of values for ordinal axis ({len(axis)}), does not match size of dimension "
                    f"({n})"
                )

    def _is_base_axis(self, axis: Union[int, Tuple[int, ...]]) -> bool:
        if isinstance(axis, Number):
            axis = (axis,)
        return len(set(axis).intersection(self.base_axes)) > 0
