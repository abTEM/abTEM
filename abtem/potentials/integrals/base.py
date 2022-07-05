from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np


class ProjectionIntegrator:

    def integrate_on_grid(self,
                          positions: np.ndarray,
                          a: np.ndarray,
                          b: np.ndarray,
                          gpts: Tuple[int, int],
                          sampling: Tuple[float, float],
                          device: str = 'cpu',
                          ):
        pass


class ProjectionIntegratorPlan(metaclass=ABCMeta):

    def __init__(self, periodic, finite):
        self._periodic = periodic
        self._finite = finite

    @property
    def periodic(self):
        return self._periodic

    @property
    def finite(self):
        return self._finite

    @abstractmethod
    def build(self, symbol: str, gpts: Tuple[int, int], sampling: Tuple[float, float], device: str):
        pass

    @abstractmethod
    def cutoff(self, symbol: str):
        pass
