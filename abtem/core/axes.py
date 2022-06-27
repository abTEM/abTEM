import dataclasses
from dataclasses import dataclass
from numbers import Number
from typing import List, Tuple

import numpy as np

from abtem.core.utils import safe_equality


@dataclass(eq=False)
class AxisMetadata:
    _concatenate: bool = True
    label: str = 'unknown'

    def __eq__(self, other):
        return safe_equality(self, other)


@dataclass(eq=False)
class UnknownAxis(AxisMetadata):
    label: str = 'unknown'


@dataclass(eq=False)
class SampleAxis(AxisMetadata):
    pass


@dataclass(eq=False)
class LinearAxis(AxisMetadata):
    sampling: float = 1.
    offset: float = 0.
    units: str = 'pixels'
    _ensemble_mean: bool = False

    def concatenate(self, other):
        if not self._concatenate:
            raise RuntimeError()

        if not self.__eq__(other):
            raise RuntimeError()

        return self


@dataclass(eq=False)
class RealSpaceAxis(LinearAxis):
    sampling: float = 1.
    units: str = 'pixels'
    offset: float = 0.
    endpoint: bool = True


@dataclass(eq=False)
class FourierSpaceAxis(LinearAxis):
    sampling: float = 1.
    units: str = 'pixels'
    fftshift: bool = True
    _concatenate: bool = False


@dataclass(eq=False)
class ScanAxis(RealSpaceAxis):
    start: Tuple[float, float] = None
    end: Tuple[float, float] = None


@dataclass(eq=False)
class OrdinalAxis(AxisMetadata):
    values: tuple = ()

    def concatenate(self, other):
        if not safe_equality(self, other, ('values',)):
            raise RuntimeError()

        kwargs = dataclasses.asdict(self)
        kwargs['values'] = kwargs['values'] + other.values

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

    def __getitem__(self, item):
        kwargs = dataclasses.asdict(self)

        if isinstance(item, Number):
            kwargs['values'] = (kwargs['values'][item],)
        else:
            kwargs['values'] = tuple(np.array(kwargs['values'])[item])

        return self.__class__(**kwargs)  # noqa


@dataclass(eq=False)
class NonLinearAxis(OrdinalAxis):
    units: str = ''


@dataclass(eq=False)
class ThicknessAxis(NonLinearAxis):
    label: str = 'thickness'
    units: str = 'Å'


@dataclass(eq=False)
class ParameterSeriesAxis(NonLinearAxis):
    label: str = ''
    _ensemble_mean: bool = False


@dataclass(eq=False)
class PositionsAxis(NonLinearAxis):
    label: str = 'Positions'
    units: str = 'Å'


@dataclass(eq=False)
class FrozenPhononsAxis(OrdinalAxis):
    label: str = 'Frozen phonons'
    _ensemble_mean: bool = False


@dataclass(eq=False)
class PrismPlaneWavesAxis(OrdinalAxis):
    pass


def axis_to_dict(axis: AxisMetadata):
    d = dataclasses.asdict(axis)
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = tuple(value.tolist())

    d['type'] = axis.__class__.__name__
    return d


def axis_from_dict(d):
    cls = globals()[d['type']]
    return cls(**{key: value for key, value in d.items() if key != 'type'})


class HasAxes:
    base_shape: Tuple[int, ...]
    ensemble_shape: Tuple[int, ...]
    base_axes_metadata: List[AxisMetadata]
    ensemble_axes_metadata: List[AxisMetadata]

    @property
    def axes_metadata(self):
        return self.ensemble_axes_metadata + self.base_axes_metadata

    @property
    def num_base_axes(self):
        return len(self.base_axes_metadata)

    @property
    def num_ensemble_axes(self):
        return len(self.ensemble_axes_metadata)

    @property
    def num_axes(self):
        return self.num_ensemble_axes + self.num_base_axes

    @property
    def base_axes(self):
        return tuple(range(self.num_ensemble_axes, self.num_ensemble_axes + self.num_base_axes))

    @property
    def ensemble_axes(self):
        return tuple(range(self.num_ensemble_axes))

    @property
    def shape(self):
        return self.ensemble_shape + self.base_shape

    def check_axes_metadata(self):
        if len(self.shape) != self.num_axes:
            raise RuntimeError(f'number of dimensions ({len(self.shape)}) does not match number of axis metadata items '
                               f'({self.num_axes})')

        for n, axis in zip(self.shape, self.axes_metadata):
            #print(n, self.axes_metadata)
            if isinstance(axis, OrdinalAxis) and len(axis) != n:
                raise RuntimeError(f'number of values for ordinal axis ({len(axis)}), does not match size of dimension '
                                   f'({n})')

    def _is_base_axis(self, axis) -> bool:
        if isinstance(axis, Number):
            axis = (axis,)
        return len(set(axis).intersection(self.base_axes)) > 0

    def find_axes_type(self, cls):
        indices = ()
        for i, axis_metadata in enumerate(self.axes_metadata):
            if isinstance(axis_metadata, cls):
                indices += (i,)

        return indices

    @property
    def num_scan_axes(self):
        return len(self.scan_axes)

    @property
    def scan_axes(self):
        num_trailing_scan_axes = 0
        for axis in reversed(self.ensemble_axes_metadata):
            if not isinstance(axis, ScanAxis) or num_trailing_scan_axes == 2:
                break

            num_trailing_scan_axes += 1

        return tuple(range(len(self.ensemble_shape) - num_trailing_scan_axes, len(self.ensemble_shape)))

    @property
    def scan_axes_metadata(self):
        return [self.axes_metadata[i] for i in self.scan_axes]

    @property
    def scan_shape(self):
        return tuple(self.shape[i] for i in self.scan_axes)

    @property
    def scan_sampling(self):
        return tuple(self.axes_metadata[i].sampling for i in self.scan_axes)

    def _ensemble_axes_to_reduce(self):
        reduce = ()
        for i, axis in enumerate(self.axes_metadata):
            if hasattr(axis, '_ensemble_mean') and axis._ensemble_mean:
                reduce += (i,)
        return reduce
