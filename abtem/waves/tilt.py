from typing import Tuple, Union

import dask.array as da

from abtem.core.axes import TiltAxis
from abtem.core.backend import get_array_module
from abtem.core.distributions import Distribution, AxisAlignedDistributionND
from abtem.waves.transfer import WaveTransform, HasParameters


def validate_tilt(tilt):
    if isinstance(tilt, Distribution):
        if not isinstance(tilt, AxisAlignedDistributionND) and tilt.dimensions == 2:
            raise ValueError()

        tilt = tilt.distributions

    tilt = tuple(tilt)
    assert len(tilt) == 2

    return tilt


def tilt_from_metadata(ensemble_axes_metadata, metadata):
    tilt_x = 0.
    tilt_y = 0.
    for axis in ensemble_axes_metadata:
        if isinstance(axis, TiltAxis) and axis.label == 'tilt_x':
            tilt_x = axis.values

        elif isinstance(axis, TiltAxis) and axis.label == 'tilt_y':
            tilt_y = axis.values

    tilt_x = metadata['tilt_x'] if 'tilt_x' in metadata else tilt_x
    tilt_y = metadata['tilt_y'] if 'tilt_y' in metadata else tilt_y
    return tilt_x, tilt_y


class BeamTilt(HasParameters, WaveTransform):

    def __init__(self,
                 tilt: Union[Distribution, Tuple[Union[float, Distribution], Union[float, Distribution]]] = (0., 0.)
                 ):
        self._tilt = validate_tilt(tilt)

    @property
    def tilt(self) -> Tuple[float, float]:
        """Beam tilt [mrad]."""
        return self._tilt

    @property
    def parameters(self):
        return {'tilt_x': self.tilt[0], 'tilt_y': self.tilt[1]}

    @property
    def ensemble_axes_metadata(self):
        axes_metadata = []
        for parameter_name, parameter in self.ensemble_parameters.items():
            direction = 'x' if parameter_name == 'tilt_x' else 'y'
            axes_metadata += [TiltAxis(label=parameter_name,
                                       values=tuple(parameter.values),
                                       direction=direction,
                                       _ensemble_mean=parameter.ensemble_mean)]
        return axes_metadata

    @tilt.setter
    def tilt(self, value: Union[Distribution, Tuple[Union[float, Distribution], Union[float, Distribution]]]):
        self._tilt = value

    def apply(self, waves):
        xp = get_array_module(waves.device)

        array = xp.expand_dims(waves.array, tuple(range(self.ensemble_dims)))

        if waves.is_lazy:
            array = da.tile(array, self.ensemble_shape + (1,) * len(waves.shape))
        else:
            array = xp.tile(array, self.ensemble_shape + (1,) * len(waves.shape))

        metadata = {name: parameter for name, parameter in self.parameters.items()
                    if name not in self.ensemble_parameters}

        kwargs = waves.copy_kwargs(exclude=('array',))
        kwargs['array'] = array
        kwargs['metadata'] = {**kwargs['metadata'], **metadata}
        kwargs['ensemble_axes_metadata'] = self.ensemble_axes_metadata + kwargs['ensemble_axes_metadata']
        return waves.__class__(**kwargs)
