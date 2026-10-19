"""Projections between directions on the northern hemisphere and a square image.

Both projections map the hemisphere ``z >= 0`` onto :math:`[-1, 1]^2`.
:meth:`~HemisphereProjection.project` takes directions to image coordinates,
and :meth:`~HemisphereProjection.unproject` takes an even grid of image
coordinates to the directions to calculate.

``stereographic``
    Projection from the south pole onto the equatorial plane. Conformal, but
    it fills only the inscribed disk and stretches areas towards the equator.

``lambert``
    The equal-area azimuthal projection composed with the Shirley-Chiu
    concentric map to the square. Fills the whole square with equal areas;
    EMsoft stores master patterns this way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "HemisphereProjection",
    "pixel_centers",
    "StereographicProjection",
    "SquareLambertProjection",
    "validate_projection",
    "bin_directions",
]


class HemisphereProjection(ABC):
    """Base class for maps between hemisphere directions and square images."""

    #: Name accepted by :func:`validate_projection`.
    name: str

    @abstractmethod
    def project(self, directions: np.ndarray) -> np.ndarray:
        """Map unit directions to square image coordinates.

        Parameters
        ----------
        directions : np.ndarray
            Unit vectors of shape ``(N, 3)`` on the northern hemisphere.

        Returns
        -------
        xy : np.ndarray
            Coordinates of shape ``(N, 2)`` in :math:`[-1, 1]^2`.
        """

    @abstractmethod
    def unproject(self, xy: np.ndarray) -> np.ndarray:
        """Map square image coordinates back to unit directions.

        Parameters
        ----------
        xy : np.ndarray
            Coordinates of shape ``(N, 2)``, inside :meth:`domain_mask`.

        Returns
        -------
        directions : np.ndarray
            Unit vectors of shape ``(N, 3)`` with ``z >= 0``.
        """

    @abstractmethod
    def domain_mask(self, xy: np.ndarray) -> np.ndarray:
        """Boolean mask selecting the coordinates the projection covers.

        The stereographic projection only fills the inscribed disk of the
        square; the square Lambert projection fills all of it.
        """

    def grid(self, gpts: int) -> np.ndarray:
        """Directions from an evenly spaced grid of image coordinates.

        Parameters
        ----------
        gpts : int
            Number of grid points along each axis of the square.

        Returns
        -------
        directions : np.ndarray
            Unit vectors of shape ``(N, 3)``, where ``N <= gpts ** 2`` is the
            number of grid points inside the projection's domain.
        """
        points = np.linspace(-1.0, 1.0, gpts)
        x, y = np.meshgrid(points, points)
        xy = np.stack([x.ravel(), y.ravel()], axis=1)
        return self.unproject(xy[self.domain_mask(xy)])


class StereographicProjection(HemisphereProjection):
    """Stereographic projection of the northern hemisphere from the south pole."""

    name = "stereographic"

    def project(self, directions: np.ndarray) -> np.ndarray:
        directions = np.asarray(directions, dtype=float)
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        return np.stack([x / (1.0 + z), y / (1.0 + z)], axis=1)

    def unproject(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        X, Y = xy[:, 0], xy[:, 1]
        r2 = X**2 + Y**2
        denominator = 1.0 + r2
        return np.stack(
            [2.0 * X / denominator, 2.0 * Y / denominator, (1.0 - r2) / denominator],
            axis=1,
        )

    def domain_mask(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return xy[:, 0] ** 2 + xy[:, 1] ** 2 <= 1.0


class SquareLambertProjection(HemisphereProjection):
    """Equal-area Lambert projection composed with the Shirley-Chiu square map."""

    name = "lambert"

    def project(self, directions: np.ndarray) -> np.ndarray:
        directions = np.asarray(directions, dtype=float)
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]

        # hemisphere -> disk (equal area): z = 1 - r**2
        r = np.sqrt(np.maximum(0.0, 1.0 - z))
        scale = np.sqrt(np.maximum(1e-300, 1.0 + z))
        x_disk = x / scale
        y_disk = y / scale

        # disk -> square (inverse Shirley-Chiu), by wedge
        theta = np.arctan2(y_disk, x_disk)
        quarter_pi = np.pi / 4.0

        x_square = np.zeros_like(r)
        y_square = np.zeros_like(r)

        right = np.abs(theta) <= quarter_pi
        top = (theta > quarter_pi) & (theta <= 3.0 * quarter_pi)
        bottom = (theta < -quarter_pi) & (theta >= -3.0 * quarter_pi)
        left = np.abs(theta) > 3.0 * quarter_pi

        x_square[right] = r[right]
        y_square[right] = r[right] * theta[right] / quarter_pi

        y_square[top] = r[top]
        x_square[top] = r[top] * (np.pi / 2.0 - theta[top]) / quarter_pi

        y_square[bottom] = -r[bottom]
        x_square[bottom] = r[bottom] * (theta[bottom] + np.pi / 2.0) / quarter_pi

        x_square[left] = -r[left]
        y_square[left] = (
            r[left] * (np.sign(theta[left]) * np.pi - theta[left]) / quarter_pi
        )

        return np.stack([x_square, y_square], axis=1)

    def unproject(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        x_square, y_square = xy[:, 0], xy[:, 1]

        r = np.zeros_like(x_square)
        theta = np.zeros_like(x_square)

        abs_x = np.abs(x_square)
        abs_y = np.abs(y_square)

        # The centre (0, 0) falls through every wedge and keeps r = theta = 0.
        right = (abs_x >= abs_y) & (x_square > 0)
        left = (abs_x >= abs_y) & (x_square < 0)
        top = (abs_y > abs_x) & (y_square > 0)
        bottom = (abs_y > abs_x) & (y_square < 0)

        quarter_pi = np.pi / 4.0

        r[right] = x_square[right]
        theta[right] = quarter_pi * (y_square[right] / x_square[right])

        r[left] = -x_square[left]
        y_sign = np.where(y_square[left] >= 0, 1.0, -1.0)
        theta[left] = y_sign * np.pi - quarter_pi * (y_square[left] / -x_square[left])

        r[top] = y_square[top]
        theta[top] = (np.pi / 2.0) - quarter_pi * (x_square[top] / y_square[top])

        r[bottom] = -y_square[bottom]
        theta[bottom] = -(np.pi / 2.0) + quarter_pi * (
            x_square[bottom] / -y_square[bottom]
        )

        x_disk = r * np.cos(theta)
        y_disk = r * np.sin(theta)

        # disk -> hemisphere (equal area)
        scale = np.sqrt(np.maximum(0.0, 2.0 - r**2))
        return np.stack([x_disk * scale, y_disk * scale, 1.0 - r**2], axis=1)

    def domain_mask(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.ones(len(xy), dtype=bool)

    def grid(self, gpts: int) -> np.ndarray:
        points = np.linspace(-1.0, 1.0, gpts)
        x, y = np.meshgrid(points, points)
        return self.unproject(np.stack([x.ravel(), y.ravel()], axis=1))


_PROJECTIONS: dict[str, type[HemisphereProjection]] = {
    StereographicProjection.name: StereographicProjection,
    SquareLambertProjection.name: SquareLambertProjection,
}


def validate_projection(
    projection: str | HemisphereProjection,
) -> HemisphereProjection:
    """Resolve a projection given by name.

    Parameters
    ----------
    projection : str or HemisphereProjection
        Either ``'stereographic'``, ``'lambert'``, or an instance to pass
        through unchanged.

    Returns
    -------
    projection : HemisphereProjection
    """
    if isinstance(projection, HemisphereProjection):
        return projection

    try:
        return _PROJECTIONS[projection]()
    except KeyError:
        raise ValueError(
            f"projection must be one of {sorted(_PROJECTIONS)}, got {projection!r}"
        ) from None


def pixel_centers(gpts: int) -> np.ndarray:
    """Centers of the pixels a square image of `gpts` covers ``[-1, 1]`` with.

    Parameters
    ----------
    gpts : int
        Number of pixels along each axis.

    Returns
    -------
    xy : np.ndarray
        Coordinates of shape ``(gpts ** 2, 2)``, ordered to match the raveled
        image.
    """
    edges = np.linspace(-1.0, 1.0, gpts + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    x, y = np.meshgrid(centers, centers, indexing="ij")
    return np.stack([x.ravel(), y.ravel()], axis=1)


def bin_directions(
    directions: np.ndarray,
    values: np.ndarray,
    gpts: int,
    projection: str | HemisphereProjection = "stereographic",
    return_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Bin values sampled at directions into a square image.

    Directions in the southern hemisphere are dropped. Pixels that no direction
    falls into are set to zero.

    Parameters
    ----------
    directions : np.ndarray
        Unit vectors of shape ``(N, 3)``.
    values : np.ndarray
        Values of shape ``(N,)`` sampled at those directions.
    gpts : int
        Number of pixels along each axis of the output image.
    projection : str or HemisphereProjection, optional
        Projection used to place each direction in the image.
    return_counts : bool, optional
        If True, also return how many directions fell into each pixel.

    Returns
    -------
    image : np.ndarray
        Array of shape ``(gpts, gpts)`` holding the mean of the values falling
        in each pixel.
    counts : np.ndarray
        Directions per pixel, same shape. Only if `return_counts` is True.
    """
    projection = validate_projection(projection)

    directions = np.asarray(directions, dtype=float)
    values = np.asarray(values, dtype=float)

    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(f"directions must have shape (N, 3), got {directions.shape}")

    if len(values) != len(directions):
        raise ValueError(
            f"values has length {len(values)} but there are "
            f"{len(directions)} directions"
        )

    northern = directions[:, 2] >= 0.0
    xy = projection.project(directions[northern])

    limits = [[-1.0, 1.0], [-1.0, 1.0]]
    total, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=gpts, range=limits, weights=values[northern]
    )
    counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=gpts, range=limits)

    image = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)

    if return_counts:
        return image, counts

    return image
