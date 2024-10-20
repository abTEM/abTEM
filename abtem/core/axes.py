"""Module for handling axes metadata."""

from __future__ import annotations

import dataclasses
from copy import copy
from dataclasses import dataclass
from numbers import Number
from typing import Any, Optional

import dask.array as da
import numpy as np
from tabulate import tabulate  # type: ignore

from abtem.core import config
from abtem.core.chunks import iterate_chunk_ranges, validate_chunks
from abtem.core.units import format_units, get_conversion_factor, validate_units
from abtem.core.utils import safe_equality


def format_label(axes: AxisMetadata, units: Optional[str] = None) -> str:
    if axes.tex_label is not None and config.get("visualize.use_tex", False):
        label = axes.tex_label
    else:
        label = axes.label

    if len(label) == 0:
        return ""

    if units is None and axes.units is not None:
        units = axes.units

    units = format_units(units)

    if units is None or len(units) == 0:
        return f"{label}"
    else:
        return f"{label} [{units}]"


def latex_float(number: float, formatting: str) -> str:
    float_str = f"{number:>{formatting}}"
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return f"{base} \\times 10^{{{int(exponent)}}}"
    else:
        return float_str


def format_value(
    value: Number | tuple, formatting: str, tolerance: float = 1e-14
) -> str:
    if isinstance(value, (tuple, list, np.ndarray)):
        return ", ".join(str(format_value(v, formatting=formatting)) for v in value)
    elif isinstance(value, float):
        if np.abs(value) < tolerance:
            float_value = 0.0
        else:
            float_value = value

        if config.get("visualize.use_tex", False):
            return f"${latex_float(float_value, formatting)}$"
        else:
            return f"{float_value:>{formatting}}"
    elif isinstance(value, (int, str, np.number)):
        return str(value)
    else:
        raise ValueError(f"Cannot format value of type {type(value)}")


def format_title(
    axes: OrdinalAxis,
    formatting: Optional[str] = None,
    units: Optional[str] = None,
    include_label: bool = True,
) -> str:
    if formatting is None:
        formatting = ".3f"

    if units:
        value = axes.values[0] * get_conversion_factor(units, axes.units)
    else:
        value = axes.values[0]

    units = validate_units(units, axes.units)

    use_tex = config.get("visualize.use_tex", False)

    if include_label and use_tex and (axes.tex_label is not None):
        label = f"{axes.tex_label} = "
    elif include_label and (axes.label is not None) and len(axes.label):
        label = f"{axes.label} = "
    else:
        label = ""

    if use_tex and (units is not None):
        if axes.tex_units is not None:
            units = f" {axes.tex_units}"
        else:
            units = f" {format_units(units)}"
    elif units is not None:
        units = f" {units}"
    else:
        units = ""

    if use_tex:
        value = format_value(value, formatting)
        return f"{label}{value}{units}"
    else:
        return f"{label}{value:>{formatting}}{units}"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class AxisMetadata:
    label: str = ""
    units: Optional[str] = None
    tex_label: Optional[str] = None
    tex_units: Optional[str] = None
    _default_type: str = "index"
    _concatenate: bool = True
    _ensemble_mean: bool = False
    _squeeze: bool = False

    def _tabular_repr_data(self, n):
        return [self.format_type(), self.format_label(), self.format_coordinates(n)]

    def format_coordinates(self, n: Optional[int] = None):
        return "-"

    def __eq__(self, other: object) -> bool:
        return safe_equality(self, other)

    def coordinates(self, n: int) -> tuple:
        return tuple(np.arange(n))

    def format_type(self):
        return self.__class__.__name__

    def format_label(self, units: Optional[str] = None):
        return format_label(self, units=units)

    def format_title(self, *args: Any, **kwargs: Any) -> str:
        return f"{self.label}"

    def item_metadata(self, item, metadata=None):
        return {}

    def to_ordinal_axis(self, n):
        values = tuple(range(n))
        return OrdinalAxis(
            label=self.label,
            tex_label=self.tex_label,
            units=self.units,
            values=values,
            _concatenate=self._concatenate,
        )

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

    def limits(self, n=None):
        coordinates = self.coordinates(n)
        min_limit = coordinates[0]
        max_limit = coordinates[-1]
        return min_limit, max_limit


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

    def format_coordinates(self, n: Optional[int] = None) -> str:
        if n is None:
            raise ValueError("n must be provided")

        coordinates = self.coordinates(n)
        if n > 3:
            coordinates_str = [f"{coordinates[i]:.2f}" for i in (0, 1, -1)]
            return f"{coordinates_str[0]} {coordinates_str[1]} ... {coordinates_str[2]}"
        else:
            return " ".join([f"{coord:.2f}" for coord in coordinates])

    def coordinates(self, n: int) -> tuple[float, ...]:
        return tuple(
            np.linspace(self.offset, self.offset + self.sampling * n, n, endpoint=False)
        )

    def to_ordinal_axis(self, n):
        values = tuple(self.coordinates(n))
        return OrdinalAxis(
            label=self.label,
            tex_label=self.tex_label,
            units=self.units,
            values=values,
            _concatenate=self._concatenate,
        )

    def convert_units(self, units: str, **kwargs):
        new_copy = self.copy()
        new_copy.units = units
        conversion = get_conversion_factor(units, old_units=self.units, **kwargs)
        new_copy.sampling = new_copy.sampling * conversion
        new_copy.offset = new_copy.offset * conversion
        return new_copy


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
    _main: bool = True


@dataclass(eq=False, repr=False, unsafe_hash=True)
class OrdinalAxis(AxisMetadata):
    values: tuple = ()

    def format_title(
        self, formatting: Optional[str] = None, include_label: bool = True, **kwargs
    ) -> str:
        return format_title(
            self, formatting=formatting, include_label=include_label, **kwargs
        )

    def format_all_titles(self) -> list[str]:
        return [
            f"{self.label} = {value} [{self.units}]"
            if i == 0
            else f"{self.label} [{self.units}]"
            for i, value in enumerate(self.values)
        ]

    def to_ordinal_axis(self, n) -> OrdinalAxis:
        assert n == len(self)
        return self

    def concatenate(self, other: AxisMetadata) -> OrdinalAxis:
        if not safe_equality(self, other, ("values",)):
            raise RuntimeError()

        assert isinstance(other, OrdinalAxis)

        kwargs = dataclasses.asdict(self)
        kwargs["values"] = kwargs["values"] + other.values

        return self.__class__(**kwargs)

    def __len__(self) -> int:
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

    def item_metadata(self, item, metadata=None):
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

    def coordinates(self, n: int) -> tuple:
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

    def format_coordinates(self, n: Optional[int] = None):
        if len(self.values) > 3:
            values = [f"{self.values[i]:.2f}" for i in [0, 1, -1]]
            return f"{values[0]} {values[1]} ... {values[-1]}"
        else:
            try:
                return " ".join([f"{value:.2f}" for value in self.values])
            except TypeError:
                return self.values

    def format_title(
        self, formatting: Optional[str] = None, include_label: bool = True, **kwargs
    ) -> str:
        return format_title(
            self, formatting=formatting, include_label=include_label, **kwargs
        )


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
            raise RuntimeError(f"Invalid tilt direction {self.direction}")

        return values

    def item_metadata(self, item, metadata=None):
        key = f"base_tilt_{self.direction}"
        new_metadata = {key: self.values[item]}
        if metadata is not None and key in metadata:
            new_metadata[key] += metadata[key]

        return new_metadata

    _ensemble_mean: bool = False


@dataclass(eq=False, repr=False, unsafe_hash=True)
class WaveVectorAxis(OrdinalAxis):
    units: str = "1/Å"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class TiltAxis(OrdinalAxis):
    units: str = "mrad"

    @property
    def tilt(self) -> tuple:
        return self.values

    def item_metadata(self, item, metadata=None):
        return {
            "base_tilt_x": self.values[item][0],
            "base_tilt_y": self.values[item][1],
        }


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

    def format_title(
        self, formatting: Optional[str] = None, include_label: bool = True, **kwargs
    ) -> str:
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
    units: Optional[str] = None
    tex_label: str | None = None

    def format_label(self):
        return format_label(self)


categories = {
    "phase [rad]": ("phase", "angle"),
    "amplitude": ("amplitude", "abs"),
    "intensity [arb. unit]": ("intensity", "abs2"),
    "real": ("real",),
    "imaginary": ("imaginary", "imag"),
}

complex_labels = {
    unit: category for category, units in categories.items() for unit in units
}

# labels = {"phase"}
#
# class ComplexScaleAxis:
#     label: str = ""
#     units: str = None
#     _tex_label: str | None = None
#
#     def format_label(self):
#         return format_label(self)


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
    def __init__(self, lst, shape):
        self._shape = shape
        super().__init__(lst)

    def __repr__(self):
        return format_axes_metadata(self, self._shape)
