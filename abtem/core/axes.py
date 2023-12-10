from __future__ import annotations

import dataclasses
from copy import copy
from dataclasses import dataclass
from numbers import Number
from typing import Union, Sequence

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike
from tabulate import tabulate

from abtem.core import config
from abtem.core.chunks import validate_chunks, iterate_chunk_ranges
from abtem.core.units import _get_conversion_factor, _format_units, _validate_units
from abtem.core.utils import safe_equality


def format_label(axes, units=None):
    if axes._tex_label is not None and config.get("visualize.use_tex", False):
        label = axes._tex_label
    else:
        label = axes.label

    if len(label) == 0:
        return ""

    if units is None and axes.units is not None:
        units = axes.units

    units = _format_units(units)

    if units is None or len(units) == 0:
        return f"{label}"
    else:
        return f"{label} [{units}]"


def latex_float(f, formatting):
    float_str = f"{f:>{formatting}}"
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return f"{base} \\times 10^{{{int(exponent)}}}"
    else:
        return float_str


def format_value(value: Union[tuple, float], formatting: str, tolerance: float = 1e-14):
    if isinstance(value, tuple):
        return ", ".join(str(format_value(v, formatting=formatting)) for v in value)

    if isinstance(value, float):
        if np.abs(value) < tolerance:
            value = 0.0

        if config.get("visualize.use_tex", False):
            return latex_float(value, formatting)
        else:
            return f"{value:>{formatting}}"

    return value


def format_title(
    axes, formatting: str = ".3f", units: str = None, include_label: bool = True
):
    try:
        value = axes.values[0] * _get_conversion_factor(units, axes.units)
    except KeyError:
        value = axes.values[0]

    units = _validate_units(units, axes.units)

    use_tex = config.get("visualize.use_tex", False)

    if include_label and use_tex and (axes._tex_label is not None):
        label = f"{axes._tex_label} = "
    elif include_label and (axes.label is not None) and len(axes.label):
        label = f"{axes.label} = "
    else:
        label = ""

    if use_tex and (units is not None):
        units = f" {_format_units(units)}"
    elif units is not None:
        units = f" {units}"
    else:
        units = ""

    if use_tex:
        value = format_value(value, formatting)
        if isinstance(value, Number):
            value = f"${value}$"
        else:
            value = f"{value}"

        return f"{label}{value}{units}"
    else:
        return f"{label}{value:>{formatting}}{units}"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class AxisMetadata:
    label: str = ""
    units: str = None
    _tex_label: str = None
    _default_type: str = None
    _concatenate: bool = True
    _ensemble_mean: bool = False
    _squeeze: bool = False

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

    def format_label(self, units: str = None):
        return format_label(self, units=units)

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

    def to_dict(self):
        d = dataclasses.asdict(self)
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                d[key] = tuple(value.tolist())

        d["type"] = self.__class__.__name__
        return d

    def concatenate(self, other: AxisMetadata) -> AxisMetadata:
        if not self._concatenate:
            raise RuntimeError()

        if not self.__eq__(other):
            raise RuntimeError()

        return self

    @staticmethod
    def from_dict(d):
        cls = globals()[d["type"]]
        return cls(**{key: value for key, value in d.items() if key != "type"})


@dataclass(eq=False, repr=False, unsafe_hash=True)
class UnknownAxis(AxisMetadata):
    label: str = "unknown"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class SampleAxis(AxisMetadata):
    pass


@dataclass(eq=False, repr=False, unsafe_hash=True)
class LinearAxis(AxisMetadata):
    sampling: float = 1.0
    units: str = ""
    offset: float = 0.0

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

    def to_nonlinear_axis(self, n):
        values = tuple(self.coordinates(n))
        return NonLinearAxis(
            label=self.label,
            _tex_label=self._tex_label,
            units=self.units,
            values=values,
            _concatenate=self._concatenate,
        )


@dataclass(eq=False, repr=False, unsafe_hash=True)
class RealSpaceAxis(LinearAxis):
    sampling: float = 1.0
    units: str = "pixels"
    endpoint: bool = True


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ReciprocalSpaceAxis(LinearAxis):
    sampling: float = 1.0
    units: str = "pixels"
    fftshift: bool = True
    _concatenate: bool = False


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ScanAxis(RealSpaceAxis):
    pass


@dataclass(eq=False, repr=False, unsafe_hash=True)
class OrdinalAxis(AxisMetadata):
    values: Union[Sequence, ArrayLike] = ()

    def format_title(self, formatting, include_label: bool = True, **kwargs):
        return format_title(
            self, formatting=formatting, include_label=include_label, **kwargs
        )

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
            values = self.values
            if isinstance(values, Number):
                values = (values,)

            try:
                self.values = tuple(values)
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

    def format_coordinates(self, n: int = None):
        if len(self.values) > 3:
            values = [f"{self.values[i]:.2f}" for i in [0, 1, -1]]
            return f"{values[0]} {values[1]} ... {values[-1]}"
        else:
            return " ".join([f"{value:.2f}" for value in self.values])

    def format_title(self, formatting, **kwargs):
        return format_title(self, formatting=formatting, **kwargs)


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
class WaveVectorAxis(OrdinalAxis):
    units: str = "1/Å"

    def format_title(self, formatting, include_label: bool = True, **kwargs):
        return format_title(self, formatting, units=None, include_label=include_label)


@dataclass(eq=False, repr=False, unsafe_hash=True)
class TiltAxis(OrdinalAxis):
    units: str = "mrad"

    @property
    def tilt(self):
        return self.values

    def format_title(self, formatting, include_label: bool = True, **kwargs):
        return format_title(self, formatting, units=None, include_label=include_label)


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ThicknessAxis(NonLinearAxis):
    label: str = "thickness"
    units: str = "Å"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ParameterAxis(NonLinearAxis):
    label: str = ""


@dataclass(eq=False, repr=False, unsafe_hash=True)
class PositionsAxis(OrdinalAxis):
    label: str = "x, y"
    units: str = "Å"

    def format_title(self, formatting, units=None, include_label=True):
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


@dataclass(eq=False, repr=False, unsafe_hash=True)
class PrismPlaneWavesAxis(AxisMetadata):
    pass


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ScaleAxis:
    label: str = ""
    units: str = None
    _tex_label: str | None = None

    def format_label(self):
        return format_label(self)


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
    with config.set({"visualize.use_tex": False}):
        data = []
        for axis, n in zip(axes_metadata, shape):
            data += [axis._tabular_repr_data(n)]

        return tabulate(
            data, headers=["type", "label", "coordinates"], tablefmt="simple"
        )


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
