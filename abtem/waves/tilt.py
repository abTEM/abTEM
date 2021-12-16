from typing import Tuple

class BeamTilt:

    def __init__(self, tilt: Tuple[float, float] = (0., 0.)):
        self._tilt = tilt

    @property
    def tilt(self) -> Tuple[float, float]:
        """Beam tilt [mrad]."""
        return self._tilt

    @tilt.setter
    def tilt(self, value: Tuple[float, float]):
        self._tilt = value


class HasBeamTiltMixin:
    _beam_tilt: BeamTilt

    @property
    def tilt(self) -> Tuple[float, float]:
        return self._beam_tilt.tilt

    @tilt.setter
    def tilt(self, value: Tuple[float, float]):
        self.tilt = value