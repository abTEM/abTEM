import dataclasses
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AxisMetadata:
    label: str = 'unknown'


@dataclass
class LinearAxis(AxisMetadata):
    sampling: float = 1.
    offset: float = 0.
    units: str = 'pixels'


@dataclass
class RealSpaceAxis(LinearAxis):
    sampling: float = 1.
    units: str = 'pixels'
    offset: float = 0.
    endpoint: bool = True


@dataclass
class FourierSpaceAxis(LinearAxis):
    sampling: float = 1.
    units: str = 'pixels'


@dataclass
class ScanAxis(RealSpaceAxis):
    start: Tuple[float, float] = None
    end: Tuple[float, float] = None


@dataclass
class NonLinearAxis(AxisMetadata):
    values: tuple = ()
    units: str = ''


@dataclass
class ThicknessSeriesAxis(NonLinearAxis):
    label: str = 'thickness'
    units: str = 'Å'


@dataclass
class ParameterSeriesAxis(NonLinearAxis):
    label: str = ''
    ensemble_mean: bool = False


@dataclass
class OrdinalAxis(AxisMetadata):
    domain: tuple = None


@dataclass
class PositionsAxis(OrdinalAxis):
    label: str = 'Positions'
    units: str = 'Å'


@dataclass
class FrozenPhononsAxis(OrdinalAxis):
    label = 'Frozen phonons'
    ensemble_mean: bool = False


@dataclass
class PrismPlaneWavesAxis(OrdinalAxis):
    pass


def axis_to_dict(axis: AxisMetadata):
    d = dataclasses.asdict(axis)
    d['type'] = axis.__class__.__name__
    return d


def axis_from_dict(d):
    cls = globals()[d['type']]
    return cls(**{key: value for key, value in d.items() if key != 'type'})


class HasAxes:
    shape: Tuple[int, ...]
    _extra_axes_metadata: List[AxisMetadata]
    base_axes_metadata: List[AxisMetadata]

    @property
    def base_axes_shape(self):
        return tuple(self.shape[i] for i in self.base_axes)

    @property
    def extra_axes_shape(self):
        return tuple(self.shape[i] for i in self.extra_axes)

    @property
    def num_axes(self):
        return len(self.shape)

    @property
    def num_base_axes(self):
        return len(self.base_axes_metadata)

    @property
    def num_extra_axes(self):
        return self.num_axes - self.num_base_axes

    @property
    def base_axes(self):
        return tuple(range(self.num_axes - self.num_base_axes, self.num_axes))

    @property
    def extra_axes(self):
        return tuple(range(self.num_extra_axes))

    @property
    def base_axes_metadata(self):
        raise NotImplementedError

    @property
    def extra_axes_metadata(self):
        return self._extra_axes_metadata

    @property
    def axes_metadata(self):
        return self.extra_axes_metadata + self.base_axes_metadata

    def find_axes(self, cls):
        indices = ()
        for i, axis_metadata in enumerate(self.axes_metadata):
            if isinstance(axis_metadata, cls):
                indices += (i,)

        return indices

    def _check_axes_metadata(self):
        if len(self.axes_metadata) != self.num_axes:
            raise RuntimeError(f'{len(self.axes_metadata)} != {self.num_axes}')

    @property
    def num_scan_axes(self):
        return len(self.scan_axes)

    @property
    def scan_axes(self):
        return self.find_axes(ScanAxis)

    @property
    def scan_axes_metadata(self):
        return [self.axes_metadata[i] for i in self.scan_axes]

    @property
    def scan_shape(self):
        return tuple(self.shape[i] for i in self.scan_axes)

    @property
    def scan_sampling(self):
        return tuple(self.axes_metadata[i].sampling for i in self.scan_axes)

    @property
    def frozen_phonon_axes(self):
        return self.find_axes(FrozenPhononsAxis)

    def _axes_to_reduce(self):
        reduce = ()
        for i, axis in enumerate(self.axes_metadata):
            if hasattr(axis, 'ensemble_mean') and axis.ensemble_mean:
                reduce += (i,)
        return reduce

    @property
    def num_frozen_phonon_axes(self):
        return len(self.frozen_phonon_axes)
