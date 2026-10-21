"""Backscattered intensity sampled on the unit hemisphere."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Sequence, cast

import dask.array as da
import numpy as np
import zarr

from abtem.core.axes import AxisMetadata, OrdinalAxis, RealSpaceAxis
from abtem.core.utils import CopyMixin, EqualityMixin
from abtem.ebsd.projections import (
    HemisphereProjection,
    bin_directions,
    pixel_centers,
    validate_projection,
)
from abtem.measurements import Images

if TYPE_CHECKING:
    from ase import Atoms

    from abtem.ebsd.geometry import EBSDGeometry

__all__ = [
    "ReferencePatternImages",
    "EBSDPatternImages",
    "SphericalPattern",
    "SparseProjectionWarning",
]


class SparseProjectionWarning(UserWarning):
    """The projected image has pixels no sampled direction reached."""


class ReferencePatternImages(Images):
    """An EBSD reference pattern projected onto a square image.

    Like :class:`~abtem.measurements.Images`, with the dimensionless projection
    coordinates in ``[-1, 1]`` as base axes.
    """

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        return [
            RealSpaceAxis(
                label="x", sampling=self.sampling[0], units="", tex_label="$x$"
            ),
            RealSpaceAxis(
                label="y", sampling=self.sampling[1], units="", tex_label="$y$"
            ),
        ]


class EBSDPatternImages(Images):
    """A pattern projected onto an EBSD detector.

    Like :class:`~abtem.measurements.Images`, with the detector pixels in
    micrometres as base axes.
    """

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        return [
            RealSpaceAxis(
                label="x", sampling=self.sampling[0], units="µm", tex_label="$x$"
            ),
            RealSpaceAxis(
                label="y", sampling=self.sampling[1], units="µm", tex_label="$y$"
            ),
        ]


def _lambert_grid(xy: np.ndarray, tolerance: float = 1e-6):
    """The Lambert grid a set of points was drawn from, if they were.

    Parameters
    ----------
    xy : np.ndarray
        Points in the square Lambert plane, of shape ``(N, 2)``.

    Returns
    -------
    grid : tuple of (int, np.ndarray) or None
        ``(gpts, nodes)``, with each point's integer node position of shape
        ``(N, 2)`` on ``linspace(-1, 1, gpts)``, or None unless the points
        cover every node of such a grid. Repeated nodes are allowed.
    """
    if len(xy) < 4:
        return None

    gpts = len(np.unique(np.round(xy[:, 0], 8)))
    if gpts < 2:
        return None

    position = (xy + 1.0) / 2.0 * (gpts - 1)
    nodes = np.rint(position)
    if np.max(np.abs(position - nodes)) > tolerance:
        return None

    nodes = nodes.astype(int)
    if len(np.unique(nodes[:, 0] * gpts + nodes[:, 1])) != gpts * gpts:
        return None

    return gpts, nodes


def _bilinear(
    values: np.ndarray, gpts: int, nodes: np.ndarray, xy: np.ndarray
) -> np.ndarray:
    """Bilinear interpolation on a Lambert grid, as EMsoft does it.

    Follows EMsoft's ``LambertgetInterpolation``, including its clamping at the
    edges of the square.

    Parameters
    ----------
    values : np.ndarray
        Sampled values, of shape ``(ensemble, N)``, at the points `nodes`
        indexes. Values sharing a node are averaged.
    gpts : int
        Nodes along each axis of the grid.
    nodes : np.ndarray
        Integer node position of each sample, of shape ``(N, 2)``.
    xy : np.ndarray
        Targets in the square Lambert plane, of shape ``(M, 2)``.

    Returns
    -------
    interpolated : np.ndarray
        Array of shape ``(ensemble, M)``.
    """
    index = nodes[:, 0] * gpts + nodes[:, 1]
    totals = np.zeros((len(values), gpts * gpts))
    counts = np.zeros(gpts * gpts)
    np.add.at(totals, (slice(None), index), values)
    np.add.at(counts, index, 1.0)
    image = (totals / counts).reshape(len(values), gpts, gpts)

    position = np.clip((xy + 1.0) / 2.0 * (gpts - 1), 0.0, gpts - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, gpts - 1)
    d = position - lower

    ix, iy, jx, jy = lower[:, 0], lower[:, 1], upper[:, 0], upper[:, 1]
    dx, dy = d[:, 0], d[:, 1]

    return (
        image[:, ix, iy] * (1 - dx) * (1 - dy)
        + image[:, jx, iy] * dx * (1 - dy)
        + image[:, ix, jy] * (1 - dx) * dy
        + image[:, jx, jy] * dx * dy
    )


class SphericalPattern(CopyMixin, EqualityMixin):
    """Backscattered intensity sampled at directions on the unit sphere.

    Use :meth:`interpolate` or :meth:`bin` for a square image of it, and
    :meth:`project` for a detector.

    A pattern may cover the northern hemisphere, the southern one, or both
    (:attr:`hemisphere`). For the hemisphere it does not cover, it uses the
    inversion image of the one it does, ``I(-k) = I(k)``, and warns if its
    metadata records that the crystal is not centrosymmetric.

    It is not an :class:`~abtem.array.ArrayObject`, since its base axis is a
    list of direction vectors.

    Parameters
    ----------
    array : np.ndarray
        Intensities of shape ``(..., N)``, where the last axis runs over the
        directions and any leading axes are ensemble axes.
    directions : np.ndarray
        Unit vectors of shape ``(N, 3)`` in the crystal frame.
    ensemble_axes_metadata : list of AxisMetadata, optional
        Metadata for the leading axes of `array`.
    metadata : dict, optional
        Measurement metadata.
    """

    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        directions: np.ndarray,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        metadata: Optional[dict] = None,
    ):
        # np.asarray would compute a dask array
        if not isinstance(array, da.core.Array):
            array = np.asarray(array)
        directions = np.asarray(directions, dtype=float)

        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError(
                f"directions must have shape (N, 3), got {directions.shape}"
            )

        if array.shape[-1] != len(directions):
            raise ValueError(
                f"the last axis of array has length {array.shape[-1]} but there "
                f"are {len(directions)} directions"
            )

        self._array = array
        self._directions = directions
        self._ensemble_axes_metadata = list(ensemble_axes_metadata or [])
        self._metadata = dict(metadata or {})

        if len(self._ensemble_axes_metadata) != array.ndim - 1:
            raise ValueError(
                f"got {len(self._ensemble_axes_metadata)} ensemble axes for an "
                f"array with {array.ndim - 1} leading axes"
            )

    @property
    def array(self) -> np.ndarray | da.core.Array:
        """The intensities, of shape ``(..., N)``."""
        return self._array

    @property
    def is_lazy(self) -> bool:
        """Whether the intensities are a dask array awaiting computation."""
        return isinstance(self._array, da.core.Array)

    def compute(self, **kwargs) -> "SphericalPattern":
        """Compute a lazy pattern, returning one holding a plain array.

        Parameters
        ----------
        kwargs :
            Passed to :meth:`dask.array.Array.compute`.

        Returns
        -------
        pattern : SphericalPattern
            Self, if the pattern was not lazy.
        """
        if not isinstance(self._array, da.core.Array):
            return self

        return self.__class__(
            self._array.compute(**kwargs),
            self._directions,
            ensemble_axes_metadata=self._ensemble_axes_metadata,
            metadata=self._metadata,
        )

    @property
    def directions(self) -> np.ndarray:
        """The sampled directions, of shape ``(N, 3)``."""
        return self._directions

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        """Metadata for the leading axes of :attr:`array`."""
        return self._ensemble_axes_metadata

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        """Shape of the leading axes of :attr:`array`."""
        return self._array.shape[:-1]

    @property
    def metadata(self) -> dict:
        """Measurement metadata."""
        return self._metadata

    def __len__(self) -> int:
        return len(self._directions)

    @property
    def hemisphere(self) -> str:
        """Which hemispheres the pattern covers: 'north', 'south' or 'both'.

        A direction is southern when its ``z`` component is negative, including
        -0.0.
        """
        south = np.signbit(self._directions[:, 2])
        if south.all():
            return "south"
        return "both" if south.any() else "north"

    @property
    def centrosymmetric(self) -> Optional[bool]:
        """Whether the crystal has an inversion centre, if the pattern records it."""
        value = self._metadata.get("centrosymmetric")
        return None if value is None else bool(value)

    def _hemisphere_pattern(self, south: bool) -> "SphericalPattern":
        """The pattern over one hemisphere: its own, or the inversion image."""
        inside = np.signbit(self._directions[:, 2]) == south
        if inside.any():
            if inside.all():
                return self
            return self.__class__(
                self._array[..., inside],
                self._directions[inside],
                ensemble_axes_metadata=self._ensemble_axes_metadata,
                metadata=self._metadata,
            )

        if self.centrosymmetric is False:
            missing, computed = (
                ("southern", "north") if south else ("northern", "south")
            )
            warnings.warn(
                f"the {missing} hemisphere was not calculated, and the crystal "
                f"is not centrosymmetric, so the inversion image of the "
                f"{computed}ern one standing in for it is not the right answer; "
                f"build the reference pattern with hemisphere='both'"
            )
        return self.__class__(
            self._array,
            -self._directions,
            ensemble_axes_metadata=self._ensemble_axes_metadata,
            metadata=self._metadata,
        )

    @property
    def northern(self) -> "SphericalPattern":
        """The pattern over the northern hemisphere.

        The calculated one, or else the inversion image of the southern one.
        """
        return self._hemisphere_pattern(south=False)

    @property
    def southern(self) -> "SphericalPattern":
        """The pattern over the southern hemisphere.

        The calculated one, or else the inversion image of the northern one.
        """
        return self._hemisphere_pattern(south=True)

    def interpolate_directions(self, directions: np.ndarray) -> np.ndarray:
        """Evaluate the pattern at arbitrary directions.

        Interpolates in the square Lambert plane. A pattern sampled on a Lambert
        grid, the default of
        :class:`~abtem.ebsd.reference.EBSDReferencePattern`, is interpolated
        bilinearly as EMsoft interpolates a master pattern; any other sampling
        is triangulated and interpolated linearly.

        Parameters
        ----------
        directions : np.ndarray
            Unit vectors of shape ``(M, 3)``, anywhere on the sphere. Each is
            evaluated in its own hemisphere (see :attr:`southern`).

        Returns
        -------
        values : np.ndarray
            Array of shape ``self.ensemble_shape + (M,)``.
        """

        if self.is_lazy:
            raise RuntimeError(
                "compute() the pattern before interpolating it; the "
                "interpolation is not built as a dask graph"
            )

        directions = np.asarray(directions, dtype=float)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError(
                f"directions must have shape (M, 3), got {directions.shape}"
            )

        values = np.empty(self.ensemble_shape + (len(directions),))
        south = np.signbit(directions[:, 2])
        for in_south in (False, True):
            wanted = south == in_south
            if wanted.any():
                pattern = self.southern if in_south else self.northern
                values[..., wanted] = pattern._interpolate_one_hemisphere(
                    directions[wanted], mirror=in_south
                )
        return values

    def _interpolate_one_hemisphere(
        self, directions: np.ndarray, mirror: bool
    ) -> np.ndarray:
        """Interpolate a pattern covering one hemisphere, at directions in it.

        A southern hemisphere is mirrored onto the northern Lambert square,
        ``(x, y, -z)``, as in EMsoft.
        """
        from scipy.interpolate import griddata  # type: ignore[import-untyped]

        lambert = validate_projection("lambert")
        flip = np.array([1.0, 1.0, -1.0 if mirror else 1.0])

        source = lambert.project(self._directions * flip)
        target = lambert.project(directions * flip)
        flat = self._array.reshape(-1, len(self._directions))

        grid = _lambert_grid(source)
        if grid is not None:
            values = _bilinear(flat, *grid, target)
            return values.reshape(self.ensemble_shape + (len(directions),))

        values = np.empty((len(flat), len(directions)))
        fraction_filled = 0.0
        for i, sampled in enumerate(flat):
            interpolated = griddata(source, sampled, target, method="linear")
            missing = np.isnan(interpolated)
            if missing.any():
                fraction_filled = max(fraction_filled, float(missing.mean()))
                interpolated[missing] = griddata(
                    source, sampled, target[missing], method="nearest"
                )
            values[i] = interpolated

        if fraction_filled > 0.001:
            warnings.warn(
                f"{fraction_filled:.1%} of the requested directions lie outside "
                f"the sampled ones and were filled from the nearest, which "
                f"shows up as flat patches. Sample the pattern in the lambert "
                f"projection, which covers the hemisphere completely.",
                SparseProjectionWarning,
            )

        return values.reshape(self.ensemble_shape + (len(directions),))

    def interpolate(
        self,
        gpts: int,
        projection: str | HemisphereProjection = "lambert",
        hemisphere: str = "north",
    ) -> ReferencePatternImages:
        """Evaluate the intensities at the nodes of a square grid.

        Unlike :meth:`bin`, it is defined for any grid size. If the pattern was
        calculated on the same grid, the values are exact.

        Parameters
        ----------
        gpts : int
            Number of grid nodes along each axis, spanning ``[-1, 1]``.
        projection : str or HemisphereProjection, optional
            ``'lambert'`` (default) or ``'stereographic'``.
        hemisphere : {'north', 'south'}, optional
            Which hemisphere to lay out (default 'north'). As in EMsoft, the
            southern node at ``(X, Y)`` is the direction ``(x, y, -z)`` of the
            northern one.

        Returns
        -------
        images : ReferencePatternImages
            Nodes outside the projection's domain are zero.
        """
        projection = validate_projection(projection)
        return ReferencePatternImages(
            self._interpolate_nodes(gpts, projection, hemisphere),
            sampling=2.0 / (gpts - 1),
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata={
                **self._metadata,
                "projection": projection.name,
                "hemisphere": hemisphere,
            },
        )

    def _interpolate_nodes(
        self,
        gpts: int,
        projection: HemisphereProjection,
        hemisphere: str,
        rotation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """The values at the nodes of :meth:`interpolate`'s grid.

        With a `rotation`, the grid is laid out in the frame it rotates the
        pattern's directions into: a node's direction ``g`` is evaluated at
        ``rotation.T @ g``.
        """
        if hemisphere not in ("north", "south"):
            raise ValueError(
                f"hemisphere must be 'north' or 'south', got {hemisphere!r}"
            )

        nodes = np.linspace(-1.0, 1.0, gpts)
        x, y = np.meshgrid(nodes, nodes, indexing="ij")
        square = np.stack([x.ravel(), y.ravel()], axis=1)
        inside = projection.domain_mask(square)

        directions = projection.unproject(square[inside])
        if hemisphere == "south":
            # -0.0 on the equator, so that it counts as southern
            directions = directions * np.array([1.0, 1.0, -1.0])
        if rotation is not None:
            directions = directions @ np.asarray(rotation, dtype=float)
        values = self.interpolate_directions(directions)

        images = np.zeros(self.ensemble_shape + (gpts * gpts,))
        images[..., inside] = values
        return images.reshape(self.ensemble_shape + (gpts, gpts))

    def bin(
        self,
        gpts: int,
        projection: str | HemisphereProjection = "stereographic",
    ) -> ReferencePatternImages:
        """Bin the intensities into a square image.

        Each pixel is the mean of the directions falling inside it, so the
        directions have to outnumber the pixels. :meth:`interpolate` works at
        any size.

        Parameters
        ----------
        gpts : int
            Number of pixels along each axis of the output image.
        projection : str or HemisphereProjection, optional
            ``'stereographic'`` (default) or ``'lambert'``.

        Returns
        -------
        images : ReferencePatternImages
            Pixels no direction falls into are zero.
        """
        projection = validate_projection(projection)

        if self.is_lazy:
            raise RuntimeError(
                "compute() the pattern before projecting it; the binning is "
                "not built as a dask graph"
            )

        flat = self._array.reshape(-1, len(self._directions))
        binned = [
            bin_directions(
                self._directions, values, gpts, projection, return_counts=True
            )
            for values in flat
        ]
        images = np.stack([image for image, _ in binned])
        images = images.reshape(self.ensemble_shape + (gpts, gpts))

        self._warn_if_sparse(binned[0][1], gpts, projection)

        return ReferencePatternImages(
            images,
            sampling=2.0 / gpts,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata={**self._metadata, "projection": projection.name},
        )

    @staticmethod
    def _warn_if_sparse(counts, gpts, projection, tolerance=0.01):
        """Warn if pixels inside the projection's domain are empty."""
        inside = projection.domain_mask(pixel_centers(gpts)).reshape(gpts, gpts)
        if not inside.any():
            return

        empty = float((counts[inside] == 0).mean())
        if empty > tolerance:
            warnings.warn(
                f"{empty:.1%} of the pixels inside the {projection.name} "
                f"projection have no sampled direction in them. Sample the "
                f"directions in the projection you mean to view, or lower gpts.",
                SparseProjectionWarning,
            )

    def project(
        self,
        geometry: "EBSDGeometry",
        euler: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
        degrees: bool = True,
    ) -> "EBSDPatternImages":
        """Project the pattern onto an EBSD detector.

        Evaluates the pattern at the direction each pixel looks along, rotated
        into the crystal frame. The values are a modulation about one, without
        the Monte Carlo background. With
        :meth:`~abtem.measurements.Images.poisson_noise`, ``total_dose`` is
        then the counts per pixel of a featureless specimen.

        Parameters
        ----------
        geometry : EBSDGeometry
            The detector geometry.
        euler : np.ndarray, optional
            Bunge Euler angles of the crystal, of shape ``(3,)`` or ``(N, 3)``
            for several orientations at once (default no rotation).
        degrees : bool, optional
            If True (default), `euler` is in degrees.

        Returns
        -------
        images : EBSDPatternImages
            Of shape ``geometry.shape``, with leading axes for any orientations
            and for this pattern's own ensemble.
        """

        directions = geometry.rotated_directions(
            np.asarray(euler, dtype=float), degrees=degrees
        )

        orientation_shape = directions.shape[:-3]
        rows, columns = geometry.shape

        values = self.interpolate_directions(directions.reshape(-1, 3))
        values = values.reshape(
            self.ensemble_shape + orientation_shape + (rows, columns)
        )

        axes: list[AxisMetadata] = list(self.ensemble_axes_metadata)
        if orientation_shape:
            euler_values = np.atleast_2d(np.asarray(euler, dtype=float))
            axes = axes + [
                OrdinalAxis(
                    label="orientation",
                    values=tuple(tuple(float(a) for a in e) for e in euler_values),
                )
            ]

        return EBSDPatternImages(
            values,
            sampling=geometry.pixel_size,
            ensemble_axes_metadata=axes,
            metadata={**self._metadata, "label": "backscattered intensity"},
        )

    def show(
        self,
        gpts: int = 256,
        projection: str | HemisphereProjection = "stereographic",
        **kwargs,
    ):
        """Interpolate the pattern onto a square image and show it.

        Uses :meth:`interpolate`, so the image has no holes in either
        projection.

        Parameters
        ----------
        gpts : int, optional
            Number of pixels along each axis of the image shown (default 256).
        projection : str or HemisphereProjection, optional
            ``'stereographic'`` (default) or ``'lambert'``.
        kwargs :
            Passed to :meth:`abtem.measurements.Images.show`.
        """
        return self.interpolate(gpts, projection).show(**kwargs)

    def to_zarr(self, url: str, overwrite: bool = False) -> None:
        """Write the pattern to a zarr store.

        Parameters
        ----------
        url : str
            Location of the zarr store.
        overwrite : bool, optional
            If True, replace an existing store (default False).
        """
        from abtem.core.axes import axis_to_dict

        if self.is_lazy:
            raise RuntimeError("compute() the pattern before writing it")

        root = zarr.open_group(url, mode="w" if overwrite else "w-")
        root.create_array("array", shape=self._array.shape, dtype=self._array.dtype)[
            :
        ] = self._array
        root.create_array(
            "directions", shape=self._directions.shape, dtype=self._directions.dtype
        )[:] = self._directions
        root.attrs["metadata"] = self._metadata
        root.attrs["ensemble_axes_metadata"] = [
            axis_to_dict(axis) for axis in self._ensemble_axes_metadata
        ]

    def to_emsoft(
        self,
        path: str,
        atoms: Atoms,
        npx: Optional[int] = None,
        energy: Optional[float] = None,
        space_group: Optional[int] = None,
        sample_tilt: float = 70.0,
        debye_waller: float | Sequence[float] = 0.005,
        name: str = "abTEM reciprocity multislice",
        overwrite: bool = False,
    ) -> None:
        """Write the pattern as an EMsoft EBSD master pattern file.

        EMsoft and kikuchipy read the file. Writing it requires h5py, and
        spglib unless `space_group` is given.

        Parameters
        ----------
        path : str
            Destination ``.h5`` file.
        atoms : ase.Atoms
            The unit cell the pattern was calculated from, for the crystal data
            a reader needs to build a phase.
        npx : int, optional
            Half-width of the square Lambert grid; the stored arrays are
            ``2 * npx + 1`` on a side. Defaults to the pattern's own Lambert
            grid, whose values are then written as calculated when it has an
            odd number of points per side, as a reference pattern's has by
            default.
        energy : float, optional
            Beam energy [eV]. Defaults to the pattern's metadata.
        space_group : int, optional
            Space group number. Determined with spglib if not given.
        sample_tilt : float, optional
            Sample tilt [degrees] recorded in the file (default 70.0).
        debye_waller : float or sequence of float, optional
            Debye-Waller factor per atom [nm²] (default 0.005, i.e. 0.5 Å²),
            recorded in the file. It does not change the pattern.
        name : str, optional
            Phase name recorded in the file.
        overwrite : bool, optional
            If True, replace an existing file.

        Notes
        -----
        For a pattern of the northern hemisphere alone, the southern one is its
        inversion image, which is only right for a centrosymmetric crystal;
        build the pattern with ``hemisphere='both'`` otherwise.
        """
        from abtem.ebsd.emsoft import write_emsoft_master_pattern

        write_emsoft_master_pattern(
            path,
            self,
            atoms,
            npx=npx,
            energy=energy,
            space_group=space_group,
            sample_tilt=sample_tilt,
            debye_waller=debye_waller,
            name=name,
            overwrite=overwrite,
        )

    @classmethod
    def from_zarr(cls, url: str) -> "SphericalPattern":
        """Read a pattern written by :meth:`to_zarr`.

        Parameters
        ----------
        url : str
            Location of the zarr store.

        Returns
        -------
        pattern : SphericalPattern
        """
        from abtem.core.axes import axis_from_dict

        root = zarr.open_group(url, mode="r")
        # written by to_zarr as a list and a mapping
        axes = cast(list, root.attrs["ensemble_axes_metadata"])
        metadata = cast(dict, root.attrs["metadata"])

        return cls(
            np.asarray(root["array"]),
            np.asarray(root["directions"]),
            ensemble_axes_metadata=[axis_from_dict(d) for d in axes],
            metadata=dict(metadata),
        )

    @classmethod
    def concatenate(cls, patterns: Sequence["SphericalPattern"]) -> "SphericalPattern":
        """Join patterns sampled at different directions.

        Parameters
        ----------
        patterns : sequence of SphericalPattern
            Patterns to join. All must share the same ensemble shape.

        Returns
        -------
        pattern : SphericalPattern
        """
        patterns = list(patterns)

        if not patterns:
            raise ValueError("cannot concatenate an empty sequence of patterns")

        shapes = {pattern.ensemble_shape for pattern in patterns}
        if len(shapes) != 1:
            raise ValueError(f"patterns have mismatched ensemble shapes: {shapes}")

        return cls(
            np.concatenate([pattern.array for pattern in patterns], axis=-1),
            np.concatenate([pattern.directions for pattern in patterns], axis=0),
            ensemble_axes_metadata=patterns[0].ensemble_axes_metadata,
            metadata=patterns[0].metadata,
        )
