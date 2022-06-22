from typing import Tuple
from abtem.core.events import HasEventsMixin, Events, watch
from abtem.core.utils import EqualityMixin


class BeamTilt(HasEventsMixin, EqualityMixin):

    def __init__(self, tilt: Tuple[float, float] = (0., 0.)):
        self._tilt = tilt
        self._events = Events()

    @property
    def tilt(self) -> Tuple[float, float]:
        """Beam tilt [mrad]."""
        return self._tilt

    @tilt.setter
    @watch
    def tilt(self, value: Tuple[float, float]):
        self._tilt = value

    def match(self, other):
        if other.tilt is None:
            other.tilt = self.tilt

        else:
            self.tilt = other.tilt


class HasBeamTiltMixin:
    _beam_tilt: BeamTilt

    @property
    def beam_tilt(self):
        return self._beam_tilt

    @property
    def tilt(self) -> Tuple[float, float]:
        return self._beam_tilt.tilt

    @tilt.setter
    def tilt(self, value: Tuple[float, float]):
        self._beam_tilt.tilt = value
