import dataclasses
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AxisMetadata:
    label: str = 'unknown'


@dataclass
class UnknownAxis(AxisMetadata):
    label: str = 'unknown'


@dataclass
class LinearAxis(AxisMetadata):
    sampling: float = 1.
    offset: float = 0.
    units: str = 'pixels'
    _ensemble_mean: bool = False


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
    fftshift: bool = True


@dataclass
class ScanAxis(RealSpaceAxis):
    start: Tuple[float, float] = None
    end: Tuple[float, float] = None


@dataclass
class NonLinearAxis(AxisMetadata):
    values: tuple = ()
    units: str = ''


@dataclass
class ThicknessAxis(NonLinearAxis):
    label: str = 'thickness'
    units: str = 'Å'


@dataclass
class ParameterSeriesAxis(NonLinearAxis):
    label: str = ''
    _ensemble_mean: bool = False


@dataclass
class OrdinalAxis(AxisMetadata):
    domain: tuple = None


@dataclass
class PositionsAxis(OrdinalAxis):
    label: str = 'Positions'
    units: str = 'Å'


@dataclass
class FrozenPhononsAxis(OrdinalAxis):
    label: str = 'Frozen phonons'
    _ensemble_mean: bool = False


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

    # @property
    # def base_shape(self):
    #     return tuple(self.shape[i] for i in self.base_axes)
    #
    # @property
    # def ensemble_shape(self):
    #     return tuple(self.shape[i] for i in self.ensemble_axes)

    def find_axes_type(self, cls):
        indices = ()
        for i, axis_metadata in enumerate(self.axes_metadata):
            if isinstance(axis_metadata, cls):
                indices += (i,)

        return indices

    def _check_axes_metadata(self):
        # ssss
        pass
        # if hasattr(self, 'array'):
        #     if len(self.array.shape) != self.num_axes:
        #         raise RuntimeError(f'{len(self.axes_metadata)} != {self.num_axes}')

    @property
    def num_scan_axes(self):
        return len(self.scan_axes)

    @property
    def scan_axes(self):
        return self.find_axes_type(ScanAxis)

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
