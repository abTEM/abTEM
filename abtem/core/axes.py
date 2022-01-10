import dataclasses
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class AxisMetadata:
    pass


@dataclass
class LinearAxis(AxisMetadata):
    sampling: float = 1.
    label: str = 'unknown'
    units: str = 'pixels'


@dataclass
class RealSpaceAxis(LinearAxis):
    sampling: float = 1.
    label: str = 'unknown'
    units: str = 'pixels'
    offset: float = 0.
    endpoint: bool = True


@dataclass
class FourierSpaceAxis(LinearAxis):
    sampling: float = 1.
    label: str = 'unknown'
    units: str = 'pixels'


@dataclass
class ScanAxis(RealSpaceAxis):
    start: Tuple[float, float] = None
    end: Tuple[float, float] = None


@dataclass
class OrdinalAxis(AxisMetadata):
    pass


@dataclass
class PositionsAxis(OrdinalAxis):
    pass


@dataclass
class FrozenPhononsAxis(OrdinalAxis):
    pass


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
    array: np.ndarray
    _extra_axes_metadata: List[AxisMetadata]
    base_axes_metadata: List[AxisMetadata]

    @property
    def num_axes(self):
        return len(self.array.shape)

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

    def find_axes(self, cls, keys=None):
        indices = ()
        for i, axis_metadata in enumerate(self.axes_metadata):
            if isinstance(axis_metadata, cls):
                indices += (i,)

        return indices

    def _check_axes_metadata(self):
        # if extra_axes_metadata is None:
        #     extra_axes_metadata = []

        # missing_extra_axes_metadata = self.num_axes - len(extra_axes_metadata) - self.num_base_axes
        # extra_axes_metadata = [{'type': 'unknown'} for _ in range(missing_extra_axes_metadata)] + extra_axes_metadata

        if len(self.axes_metadata) != self.num_axes:
            raise RuntimeError()

    @property
    def scan_axes(self):
        return self.find_axes(ScanAxis)

    @property
    def scan_axes_metadata(self):
        return [self.axes_metadata[i] for i in self.scan_axes]

    @property
    def num_scan_axes(self):
        return len(self.scan_axes)

    @property
    def scan_shape(self):
        return tuple(self.array.shape[i] for i in self.scan_axes)

    @property
    def scan_sampling(self):
        return tuple(self.array.shape[i] for i in self.scan_axes)

    @property
    def base_axes_shape(self):
        return tuple(self.array.shape[i] for i in self.base_axes)

    @property
    def extra_axes_shape(self):
        return tuple(self.array.shape[i] for i in self.extra_axes)

    @property
    def ensemble_axes(self):
        return self._type_indices(('ensemble',))

    @property
    def num_ensemble_axes(self):
        return len(self.ensemble_axes)
