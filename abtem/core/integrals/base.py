from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np

from abtem.core.utils import EqualityMixin, CopyMixin


class ProjectionIntegrator:

    def integrate_on_grid(self,
                          positions: np.ndarray,
                          a: np.ndarray,
                          b: np.ndarray,
                          gpts: tuple[int, int],
                          sampling: tuple[float, float],
                          device: str = 'cpu',
                          ):
        pass


class ProjectionIntegratorPlan(EqualityMixin, CopyMixin, metaclass=ABCMeta):

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
    def build(self, symbol: str, gpts: tuple[int, int], sampling: tuple[float, float], device: str):
        pass

    @abstractmethod
    def cutoff(self, symbol: str):
        pass
