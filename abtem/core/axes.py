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

    def format_type(self) -> str:
        """Return the axis type name (the class name)."""
        return self.__class__.__name__

    def format_label(self, units: Optional[str] = None) -> str:
        """Return a formatted label string, optionally with units."""
        return format_label(self, units=units)

    def format_title(self, *args: Any, **kwargs: Any) -> str:
        """Return a formatted title string for display."""
        return f"{self.label}"

    def item_metadata(self, item, metadata=None) -> dict:
        """Return metadata associated with a specific item along this axis."""
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
        # Mirrors `ArrayObject._partition_ensemble_axes_metadata`'s
        # `axis[slic] if hasattr(axis, "__getitem__") else axis.copy()`
        # fallback. Any axis whose metadata depends on WHICH range of the
        # array it covers (e.g. LinearAxis's offset) must define
        # __getitem__ to shift that dependent state per block -- without it,
        # every block silently gets an identical copy of the GLOBAL axis,
        # including state that is only valid for the first block.
        # `OrdinalAxis` overrides this method directly rather than relying
        # on the fallback, since its `values` tuple needs no per-block
        # adjustment beyond slicing, which `__getitem__` already does.
        arr = np.empty((len(chunks[0]),), dtype=object)
        has_getitem = hasattr(self, "__getitem__")
        for i, slic in iterate_chunk_ranges(chunks):
            arr[i] = self[slic] if has_getitem else copy(self)
        arr = da.from_array(arr, chunks=1)
        return arr

    def copy(self) -> AxisMetadata:
        """Return a deep copy of this axis metadata."""
        return copy(self)

    def to_dict(self) -> dict:
        """Serialize this axis metadata to a dictionary."""
        d = dataclasses.asdict(self)
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                d[key] = tuple(value.tolist())

        d["type"] = self.__class__.__name__
        return d

    def concatenate(self, other: AxisMetadata) -> AxisMetadata:
        """Concatenate this axis metadata with another compatible axis metadata."""
        if not self._concatenate:
            raise RuntimeError()

        if not self.__eq__(other):
            raise RuntimeError()

        return self

    @staticmethod
    def from_dict(d) -> AxisMetadata:
        """Reconstruct an AxisMetadata instance from a dictionary."""
        cls = globals()[d["type"]]
        return cls(**{key: value for key, value in d.items() if key != "type"})

    def limits(self, n=None) -> tuple:
        """Return the (min, max) coordinate limits for this axis."""
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

    def __getitem__(self, item):
        """Return the axis restricted to a contiguous sub-range, with `offset`
        shifted to match.

        Without this, `_partition_ensemble_axes_metadata` (dask ensemble
        partitioning) falls back to `hasattr(axis, "__getitem__")` being
        False and uses `axis.copy()` instead: every chunk gets an identical
        copy of the GLOBAL axis, including its global `offset`, regardless
        of which chunk it actually is. A `LinearAxis` (offset + sampling)
        does not store its own length, so unlike `OrdinalAxis.__getitem__`
        (which slices an explicit `values` tuple), only `offset` needs to
        change here -- the number of points continues to come from the
        array's own shape wherever this axis is used.

        Only simple, positive-step, non-empty slices/indices -- which is all
        `iterate_chunk_ranges` chunk partitioning ever produces -- are
        represented exactly; anything else raises TypeError rather than
        silently returning an axis with the wrong offset. TypeError
        specifically: `ArrayObject._get_ensemble_axes_metadata_items`
        (general user-facing indexing, e.g. `waves[::-1]`) already wraps
        `axis[item]` in `try/except TypeError` and falls back to a plain
        copy for whatever an axis can't represent exactly -- the same
        fallback this axis relied on before it had a `__getitem__` at all.
        Chunk partitioning is unaffected either way, since it never
        constructs a slice this doesn't support.
        """
        kwargs = dataclasses.asdict(self)

        # `iterate_chunk_ranges` always yields a tuple of slices, one per
        # axis, even when partitioning a single axis on its own (as
        # `_to_blocks` does) -- numpy's indexing unwraps a length-1 tuple to
        # its element automatically, which is what makes this transparent
        # for `OrdinalAxis` (a plain array index); do the same here.
        if isinstance(item, tuple):
            if len(item) != 1:
                raise TypeError(
                    f"{type(self).__name__} indices must be a single "
                    f"int/slice or a length-1 tuple of one, got {item!r}"
                )
            item = item[0]

        if isinstance(item, Number):
            start = item
        elif isinstance(item, slice):
            if item.step not in (None, 1):
                raise TypeError(
                    f"{type(self).__name__} does not support a strided "
                    f"slice, got step={item.step}"
                )
            start = item.start or 0
            if start < 0 or (item.stop is not None and item.stop <= start):
                raise TypeError(
                    f"{type(self).__name__} does not support a negative "
                    f"or empty slice, got {item}"
                )
        else:
            raise TypeError(
                f"{type(self).__name__} indices must be an int or a simple "
                f"positive-step slice, got {type(item).__name__}"
            )

        kwargs["offset"] = self.offset + start * self.sampling
        return self.__class__(**kwargs)

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

    def coordinates(self, n: int) -> tuple[float, ...]:
        """Spatial frequencies in storage order.

        `offset` is the lowest frequency of the centred grid, taken as a whole multiple
        of the sampling: the coordinates are ``(round(offset / sampling) + arange(n))
        * sampling``. With ``fftshift=False`` the array is
        stored in unshifted (``np.fft.fftfreq``) order, zero frequency first,
        so the centred grid is ``ifftshift``-ed into that order (``ifftshift``,
        not ``fftshift``: the two differ by one element for odd `n`).
        """
        # every coordinate is a whole multiple of the sampling, so zero frequency is
        # exactly 0 and opposite frequencies are exact negatives
        coordinates = tuple(
            (round(self.offset / self.sampling) + np.arange(n)) * self.sampling
        )
        if self.fftshift:
            return coordinates
        return tuple(np.fft.ifftshift(np.array(coordinates)))


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ScanAxis(RealSpaceAxis):
    _main: bool = True


@dataclass(eq=False, repr=False, unsafe_hash=True)
class OrdinalAxis(AxisMetadata):
    """Axis described by an explicit tuple of values.

    Parameters
    ----------
    values : tuple
        The coordinate (parameter value) of each element along the axis.
    weights : tuple of float, optional
        Probability weight of each element along the axis, aligned with
        ``values``. An ensemble axis produced by a weighted distribution (e.g.
        :func:`abtem.distributions.gaussian`) carries the distribution's weights
        here, and :meth:`BaseMeasurements.reduce_ensemble` then computes the
        probability-weighted mean ``Σ p_i I_i / Σ p_i`` over the axis. ``None``
        (default) means equal weights, i.e. a plain mean. The weights need not
        be normalized.
    """

    values: tuple = ()
    weights: Optional[tuple] = None

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
        if not safe_equality(self, other, ("values", "weights")):
            raise RuntimeError()

        assert isinstance(other, OrdinalAxis)

        kwargs = dataclasses.asdict(self)
        kwargs["values"] = kwargs["values"] + other.values

        if self.weights is not None and other.weights is not None:
            kwargs["weights"] = self.weights + other.weights
        elif _has_unequal_weights(self.weights) or _has_unequal_weights(
            other.weights
        ):
            # The weights need not be normalized, so an unweighted axis has no
            # defined scale relative to a weighted one: any choice (e.g. unit
            # weights) would silently set the relative weight of the two parts.
            raise ValueError(
                f"cannot concatenate an ensemble axis '{self.label}' carrying "
                "probability weights with one that has none; their relative "
                "weighting is undefined"
            )
        else:
            # Equal weights on both sides: a plain mean is exact.
            kwargs["weights"] = None

        return self.__class__(**kwargs)

    @classmethod
    def from_distribution(
        cls, distribution: Any, values: Optional[tuple] = None, **kwargs: Any
    ) -> OrdinalAxis:
        """Ensemble axis described by a one-dimensional distribution.

        Sets the values, the probability weights and the ``_ensemble_mean`` flag
        from the distribution, so that no axis built from a distribution can
        lose its weights.

        Parameters
        ----------
        distribution : BaseDistribution
            The distribution defining the ensemble axis.
        values : tuple, optional
            The axis values, if they must be converted from the distribution
            values (default is ``tuple(distribution.values)``).
        **kwargs
            Further fields of the axis metadata (label, units, ...).
        """
        if values is None:
            values = tuple(distribution.values)

        return cls(
            values=values,
            weights=_distribution_axis_weights(distribution),
            _ensemble_mean=distribution.ensemble_mean,
            **kwargs,
        )

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

        if self.weights is not None:
            weights = self.weights
            if isinstance(weights, Number):
                weights = (weights,)
            weights = tuple(float(weight) for weight in np.ravel(np.asarray(weights)))

            if len(weights) != len(self.values):
                raise ValueError(
                    f"{type(self).__name__} has {len(self.values)} values but "
                    f"{len(weights)} weights"
                )
            if any(weight < 0.0 for weight in weights):
                raise ValueError("axis weights must be non-negative")

            self.weights = weights

    def item_metadata(self, item, metadata=None):
        return {self.label: self.values[item]}

    def __getitem__(self, item):
        kwargs = dataclasses.asdict(self)

        if isinstance(item, Number):
            kwargs["values"] = (kwargs["values"][item],)
            if self.weights is not None:
                kwargs["weights"] = (self.weights[item],)
        else:
            array = np.empty(len(kwargs["values"]), dtype=object)
            array[:] = kwargs["values"]
            kwargs["values"] = tuple(array[item])
            if self.weights is not None:
                # Index the weights with exactly the same item as the values
                # so they stay aligned under any slice, fancy index or mask.
                kwargs["weights"] = tuple(np.asarray(self.weights, dtype=float)[item])

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
class SpinAxis(OrdinalAxis):
    """
    Labeled axis holding the two components of a spinor wave function.

    The two components are mixed by the Pauli multislice solver, unlike
    ordinary ensemble axes whose members evolve independently, so this
    axis must never be split across dask chunks. Locate it by isinstance,
    never by position.
    """

    label: str = "spin"
    values: tuple = ("up", "down")


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ThicknessAxis(NonLinearAxis):
    label: str = "thickness"
    units: str = "Å"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ParameterAxis(NonLinearAxis):
    label: str = ""


@dataclass(eq=False, repr=False, unsafe_hash=True)
class EnergyAxis(NonLinearAxis):
    label: str = "Energy"
    units: str = "eV"

    def item_metadata(self, item, metadata=None):
        # Use the lowercase "energy" key to match the metadata convention used
        # everywhere else in the codebase (metadata["energy"]).  The inherited
        # OrdinalAxis implementation would use self.label ("Energy") which would
        # silently miss all lookups that check for "energy".
        return {"energy": self.values[item]}

    def format_title(
        self, formatting: Optional[str] = None, include_label: bool = True, **kwargs
    ) -> str:
        """Format title displaying energy in keV regardless of stored eV units."""
        if formatting is None:
            formatting = ".3g"
        value_kev = self.values[0] / 1000.0
        formatted = f"{value_kev:>{formatting}}"
        if include_label:
            return f"{self.label} = {formatted} keV"
        else:
            return f"{formatted} keV"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class EnergyLossAxis(NonLinearAxis):
    """Ensemble axis for energy loss (e.g. phonon/TDS or EELS), distinct from
    the accelerating beam :class:`.EnergyAxis` so the two never collide in
    metadata or in code that pattern-matches on ``isinstance(ax, EnergyAxis)``.
    """

    label: str = "energy loss"
    units: str = "eV"

    def item_metadata(self, item, metadata=None):
        return {"energy_loss": self.values[item]}

    def format_title(
        self, formatting: Optional[str] = None, include_label: bool = True, **kwargs
    ) -> str:
        """Format title displaying energy loss in meV."""
        if formatting is None:
            formatting = ".3g"
        value_meV = self.values[0] * 1000.0
        formatted = f"{value_meV:>{formatting}}"
        if include_label:
            return f"{self.label} = {formatted} meV"
        else:
            return f"{formatted} meV"


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


def _normalized_axis_weights(axis: AxisMetadata, n: int) -> Optional[np.ndarray]:
    """Return the probability weights of an ensemble axis normalized to sum to one,
    or None if the axis has equal weights (the plain mean is then exact).

    Equal weights include all-zero weights, e.g. on a slice holding only
    zero-weight members: their weighted mean is undefined (0/0), and the plain
    mean is used as the fallback, so such a slice can still be reduced or shown.

    Parameters
    ----------
    axis : AxisMetadata
        The ensemble axis. Only axes with a ``weights`` attribute (OrdinalAxis and
        subclasses) can be non-uniformly weighted.
    n : int
        The length of the array along the axis, used to check that the weights are
        aligned with the array.
    """
    weights = getattr(axis, "weights", None)
    if weights is None:
        return None

    # Host-side metadata arithmetic; cast to the array dtype where applied.
    weights = np.asarray(weights, dtype=float)

    if weights.shape != (n,):
        raise RuntimeError(
            f"ensemble axis '{axis.label}' has {weights.size} weights, but the "
            f"array has length {n} along the axis"
        )

    if not _has_unequal_weights(weights):
        return None

    # Unequal non-negative weights always have a positive sum.
    return weights / weights.sum()


def _has_unequal_weights(weights: Optional[Any]) -> bool:
    """Whether the weights are given and not all equal (equal or absent weights
    make the plain mean exact)."""
    if weights is None or len(weights) == 0:
        return False
    weights = np.asarray(weights, dtype=float)
    return not np.all(weights == weights[0])


def _distribution_axis_weights(distribution: Any) -> Optional[tuple[float, ...]]:
    """The probability weights of a one-dimensional distribution as a tuple, for
    ensemble axis metadata, or None if the weights are all equal (plain mean)."""
    weights = np.asarray(distribution.weights)
    if weights.ndim != 1 or len(weights) != len(distribution.values):
        raise NotImplementedError(
            "only one-dimensional distributions can define an ensemble axis"
        )
    if not _has_unequal_weights(weights):
        return None
    return tuple(float(weight) for weight in weights)


def axis_to_dict(axis: AxisMetadata):
    d = dataclasses.asdict(axis)
    for key, value in d.items():
        if isinstance(value, np.ndarray) or hasattr(value, "__cuda_array_interface__"):
            d[key] = tuple(value.tolist())

    # Unweighted axes are written without the key, so files that carry no
    # weighted axis stay readable by abTEM versions predating ``weights``.
    if "weights" in d and d["weights"] is None:
        del d["weights"]

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
