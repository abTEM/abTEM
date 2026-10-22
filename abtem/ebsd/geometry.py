"""The geometry of an EBSD detector, in EMsoft's conventions.

Each pixel of the flat detector looks along the ray from the illuminated point
through it, a gnomonic projection. The sample and camera tilts rotate these
rays into the sample frame, and the crystal orientation rotates them into the
crystal frame, where a reference pattern is defined.

The parameters are EMsoft's, so a geometry calibrated for EMsoft or kikuchipy
can be used unchanged. Lengths are in micrometres and angles in degrees. The
arithmetic follows kikuchipy's ``_get_direction_cosines_for_fixed_pc``, which
works in the sample frame throughout.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from abtem.core.utils import CopyMixin, EqualityMixin

__all__ = ["EBSDGeometry", "bunge_rotation"]


def bunge_rotation(
    euler: np.ndarray | Sequence[float], degrees: bool = True
) -> np.ndarray:
    """Orientation matrix from Bunge Euler angles.

    The matrix takes a direction in the sample frame to the crystal frame, as
    in EMsoft, TSL and HKL. It equals ``euler_to_rotation(phi1, Phi, phi2,
    axes="zxz", convention="extrinsic").T`` from :mod:`abtem.atoms`.

    Parameters
    ----------
    euler : np.ndarray
        Bunge angles ``(phi1, Phi, phi2)``, of shape ``(3,)`` or ``(N, 3)``.
    degrees : bool, optional
        If True (default), `euler` is in degrees.

    Returns
    -------
    rotation : np.ndarray
        Matrix of shape ``(3, 3)``, or ``(N, 3, 3)`` for a stack of angles.
    """
    angles = np.asarray(euler, dtype=float)

    squeeze = angles.ndim == 1
    angles = np.atleast_2d(angles)

    if angles.ndim != 2 or angles.shape[-1] != 3:
        raise ValueError(f"euler must have shape (3,) or (N, 3), got {angles.shape}")

    if degrees:
        angles = np.radians(angles)

    phi1, phi, phi2 = angles[:, 0], angles[:, 1], angles[:, 2]
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(phi), np.sin(phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)

    rotation = np.empty(angles.shape[:1] + (3, 3))
    rotation[:, 0, 0] = c1 * c2 - s1 * s2 * c
    rotation[:, 0, 1] = s1 * c2 + c1 * s2 * c
    rotation[:, 0, 2] = s2 * s
    rotation[:, 1, 0] = -c1 * s2 - s1 * c2 * c
    rotation[:, 1, 1] = -s1 * s2 + c1 * c2 * c
    rotation[:, 1, 2] = c2 * s
    rotation[:, 2, 0] = s1 * s
    rotation[:, 2, 1] = -c1 * s
    rotation[:, 2, 2] = c

    return rotation[0] if squeeze else rotation


class EBSDGeometry(CopyMixin, EqualityMixin):
    """Where an EBSD detector sits relative to the sample.

    Parameters
    ----------
    shape : two int, optional
        Detector pixels as ``(rows, columns)``, ie. ``(numsy, numsx)`` in
        EMsoft (default ``(120, 120)``).
    detector_distance : float, optional
        Distance from the illuminated point to the scintillator [µm], EMsoft's
        ``L`` (default 15000.0).
    pattern_center : two float, optional
        Pattern centre in pixels, EMsoft's ``(xpc, ypc)``, measured from the
        centre of the detector (default ``(0.0, 0.0)``).
    pixel_size : float, optional
        Scintillator pixel size [µm], EMsoft's ``delta`` (default 50.0).
    sample_tilt : float, optional
        Tilt of the sample [degrees], EMsoft's ``sig`` (default 70.0).
    camera_tilt : float, optional
        Tilt of the camera below the horizontal [degrees], EMsoft's ``thetac``
        (default 0.0).
    azimuthal_angle : float, optional
        Angle between the sample normal and the detector [degrees], EMsoft's
        ``omega`` (default 0.0).
    """

    def __init__(
        self,
        shape: tuple[int, int] = (120, 120),
        detector_distance: float = 15000.0,
        pattern_center: tuple[float, float] = (0.0, 0.0),
        pixel_size: float = 50.0,
        sample_tilt: float = 70.0,
        camera_tilt: float = 0.0,
        azimuthal_angle: float = 0.0,
    ):
        rows, columns = int(shape[0]), int(shape[1])
        if rows < 1 or columns < 1:
            raise ValueError(f"shape must be positive, got {shape}")

        if detector_distance <= 0.0:
            raise ValueError(
                f"detector_distance must be positive, got {detector_distance}"
            )

        if pixel_size <= 0.0:
            raise ValueError(f"pixel_size must be positive, got {pixel_size}")

        self._shape = (rows, columns)
        self._detector_distance = float(detector_distance)
        self._pattern_center = (float(pattern_center[0]), float(pattern_center[1]))
        self._pixel_size = float(pixel_size)
        self._sample_tilt = float(sample_tilt)
        self._camera_tilt = float(camera_tilt)
        self._azimuthal_angle = float(azimuthal_angle)

    @property
    def shape(self) -> tuple[int, int]:
        """Detector pixels, as ``(rows, columns)``."""
        return self._shape

    @property
    def detector_distance(self) -> float:
        """Distance to the scintillator [µm]."""
        return self._detector_distance

    @property
    def pattern_center(self) -> tuple[float, float]:
        """Pattern centre in pixels from the detector centre."""
        return self._pattern_center

    @property
    def pixel_size(self) -> float:
        """Scintillator pixel size [µm]."""
        return self._pixel_size

    @property
    def sample_tilt(self) -> float:
        """Sample tilt [degrees]."""
        return self._sample_tilt

    @property
    def camera_tilt(self) -> float:
        """Camera tilt [degrees]."""
        return self._camera_tilt

    @property
    def azimuthal_angle(self) -> float:
        """Sample-to-detector azimuth [degrees]."""
        return self._azimuthal_angle

    @property
    def scintillator_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel centres on the scintillator [µm], relative to the pattern centre.

        Returns
        -------
        x, y : np.ndarray
            Of shape ``(columns,)`` and ``(rows,)``. ``x`` runs left to right
            across the columns and ``y`` bottom to top up the rows.
        """
        rows, columns = self._shape
        xpc, ypc = self._pattern_center
        delta = self._pixel_size

        x = delta * (np.arange(columns) + 0.5 - columns / 2.0 - xpc)
        y = delta * (rows / 2.0 - 0.5 - np.arange(rows) - ypc)

        return x, y

    @property
    def detector_to_sample(self) -> np.ndarray:
        """Rotation taking a direction in the detector frame to the sample frame.

        Returns
        -------
        rotation : np.ndarray
            Matrix of shape ``(3, 3)``.
        """
        angle = np.radians(self._sample_tilt - self._camera_tilt)
        ca, sa = np.cos(angle), np.sin(angle)
        tilt = np.array([[0.0, -sa, ca], [1.0, 0.0, 0.0], [0.0, ca, sa]])

        omega = np.radians(-self._azimuthal_angle)
        cw, sw = np.cos(omega), np.sin(omega)
        azimuth = np.array([[cw, 0.0, sw], [0.0, 1.0, 0.0], [-sw, 0.0, cw]])

        return tilt @ azimuth

    @property
    def directions(self) -> np.ndarray:
        """Unit vector each pixel looks along, in the sample frame.

        Returns
        -------
        directions : np.ndarray
            Of shape ``(rows, columns, 3)``.
        """
        x, y = self.scintillator_coordinates
        rows, columns = self._shape

        rays = np.stack(
            np.broadcast_arrays(
                x[None, :],
                y[:, None],
                np.full((rows, columns), self._detector_distance),
            ),
            axis=-1,
        )

        directions = rays @ self.detector_to_sample.T

        return directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    def rotated_directions(self, euler: np.ndarray, degrees: bool = True) -> np.ndarray:
        """The pixel directions expressed in the crystal frame.

        Parameters
        ----------
        euler : np.ndarray
            Bunge Euler angles of shape ``(3,)`` or ``(N, 3)``.
        degrees : bool, optional
            If True (default), `euler` is in degrees.

        Returns
        -------
        directions : np.ndarray
            Of shape ``(rows, columns, 3)``, or ``(N, rows, columns, 3)``.
        """
        rotation = bunge_rotation(euler, degrees=degrees)
        directions = self.directions

        if rotation.ndim == 2:
            return directions @ rotation.T

        return np.einsum("nij,rcj->nrci", rotation, directions)
