from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np

from abtem.core.utils import EqualityMixin, CopyMixin


class ProjectionIntegrator:
    """Base class for projection integrators used for integrating projected potentials."""
    def integrate_on_grid(
        self,
        positions: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Integrate radial potential between two limits at the given 2D positions on a grid.

        Parameters
        ----------
        positions : np.ndarray
            2D array of xy-positions of the centers of each radial function [Å].
        a : np.ndarray
            Lower integration limit of the projection integrals along z for each position [Å]. The limit is given
            relative to the center of the radial function.
        b : np.ndarray
            Upper integration limit of the projection integrals along z for each position [Å]. The limit is given
            relative to the center of the radial function.
        gpts : two int
            Number of grid points in `x` and `y` describing each slice of the potential.
        sampling : two float
            Sampling of the potential in `x` and `y` [1 / Å].
        device : str, optional
            The device used for calculating the potential, 'cpu' or 'gpu'. The default is determined by the user
            configuration file.
        """
        pass


class ProjectionIntegratorPlan(EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """
    The projection integrator plan facilitates the creation of projection integrator objects.

    Parameters
    ----------
    periodic : bool
        True indicates that the projection integrals are periodic perpendicular to the projection direction.
    finite : bool
        True indicates that the projection integrals are finite along the projection direction.
    """
    def __init__(self, periodic: bool, finite: bool):
        self._periodic = periodic
        self._finite = finite

    @property
    def periodic(self) -> bool:
        """True indicates that the created projection integrators are implemented only for periodic potentials."""
        return self._periodic

    @property
    def finite(self) -> bool:
        """True indicates that the created projection integrators are implemented only for infinite potential
        projections."""
        return self._finite

    @abstractmethod
    def build(
        self,
        symbol: str,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str,
    ) -> ProjectionIntegrator:
        """
        Build projection integrator for given chemical symbol, grid and device.

        Parameters
        ----------
        symbol : str
            Chemical symbol to build the projection integrator for.
        gpts : two int
            Number of grid points in `x` and `y` describing each slice of the potential.
        sampling : two float
            Sampling of the potential in `x` and `y` [1 / Å].
        device : str, optional
            The device used for calculating the potential, 'cpu' or 'gpu'. The default is determined by the user
            configuration file.

        Returns
        -------
        projection_integrator : ProjectionIntegrator
            The projection integrator for the specified chemical symbol.
        """

        pass

    @abstractmethod
    def cutoff(self, symbol: str) -> float:
        """Radial cutoff of the potential for the given chemical symbol."""
        pass
