"""Module for describing electrostatic potentials using the independent atom model."""

from __future__ import annotations

import copy
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial, reduce
from numbers import Number
from operator import mul
from typing import TYPE_CHECKING, Optional, Sequence, Type

import dask
import dask.array as da
import numpy as np
from ase import Atom, Atoms
from ase.cell import Cell
from ase.data import chemical_symbols

from abtem.array import ArrayObject, validate_lazy
from abtem.atoms import (
    _box_strain_warning_silenced,
    _box_strain_warning_suppressed,
    _cell_in_plane_frame,
    _rotate_atoms_to_plane,
    _stacklevel_outside_package,
    _warn_if_box_is_strained,
    _wrap_far_atoms,
    wrap_and_snap_atoms,
    best_orthogonal_cell,
    cut_cell,
    is_cell_ab_in_plane,
    is_cell_orthogonal,
    orthogonalize_cell,
    pad_atoms,
)
from abtem.core.axes import (
    AxisMetadata,
    FrozenPhononsAxis,
    RealSpaceAxis,
    ThicknessAxis,
    _find_axes_type,
)
from abtem.core.backend import get_array_module, validate_device
from abtem.core.chunks import Chunks, chunk_ranges, generate_chunks, validate_chunks
from abtem.core.complex import complex_exponential, complex_exponential_scaled
from abtem.core.energy import Accelerator, HasAcceleratorMixin, energy2sigma
from abtem.core.ensemble import Ensemble, _wrap_with_array, unpack_blockwise_args
from abtem.core.grid import Grid, HasGrid2DMixin, round_auto_derived_gpts
from abtem.core.utils import (
    CopyMixin,
    EqualityMixin,
    get_dtype,
    itemset,
    safe_ceiling_int,
)
from abtem.inelastic.phonons import (
    SOURCE_INDEX,
    AtomsEnsemble,
    BaseFrozenPhonons,
    DummyFrozenPhonons,
    EnergyResolvedAtomsEnsemble,
    FrozenPhonons,
    validate_seeds,
)
from abtem.integrals import (
    QuadratureProjectionIntegrals,
    ScatteringFactorProjectionIntegrals,
)
from abtem.measurements import Images, RealSpaceLineProfiles, _interpolate_stack
from abtem.slicing import (
    FACE_SLICE_WARNING_DISTANCE,
    BaseSlicedAtoms,
    SlicedAtoms,
    SliceIndexedAtoms,
    _far_outside_faces,
    _validate_slice_thickness,
    commensurate_gpts,
    commensurate_slice_thickness,
    slice_limits,
)

if TYPE_CHECKING:
    from abtem.integrals import FieldIntegrator
    from abtem.parametrizations import Parametrization
    from abtem.waves import BaseWaves, Waves


class BaseField(Ensemble, HasGrid2DMixin, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    # @property
    # @abstractmethod
    # def device(self) -> str:
    #    pass

    @property
    def base_shape(self):
        """Shape of the base axes of the potential."""
        return (self.num_slices,) + self.gpts

    @property
    @abstractmethod
    def num_configurations(self):
        """Number of frozen phonons in the ensemble of potentials."""
        pass

    @property
    @abstractmethod
    def base_axes_metadata(self):
        pass

    def _get_exit_planes_axes_metadata(self):
        return ThicknessAxis(label="z", values=tuple(self.exit_thicknesses))

    @property
    @abstractmethod
    def exit_planes(self) -> tuple[int, ...]:
        """The "exit planes" of the potential. The indices of slices where a measurement
        is returned."""
        pass

    @property
    def _exit_plane_after(self):
        exit_plane_index = 0
        exit_planes = self.exit_planes

        if len(exit_planes) == 0:
            return np.zeros(len(self), dtype=bool)

        if exit_planes[0] == -1:
            exit_plane_index += 1

        is_exit_plane = np.zeros(len(self), dtype=bool)
        for i in range(len(is_exit_plane)):
            if exit_plane_index < len(exit_planes) and i == exit_planes[exit_plane_index]:
                is_exit_plane[i] = True
                exit_plane_index += 1

        return is_exit_plane

    @property
    def exit_thicknesses(self) -> tuple[float, ...]:
        """The "exit thicknesses" of the potential. The thicknesses in the potential
        where a measurement is returned."""
        thicknesses = np.cumsum(self.slice_thickness)
        exit_indices = np.array(self.exit_planes, dtype=int)
        exit_thicknesses = tuple(thicknesses[i] for i in exit_indices)
        if self.exit_planes[0] == -1:
            return (0.0,) + exit_thicknesses[1:]
        else:
            return exit_thicknesses

    @property
    def num_exit_planes(self) -> int:
        """Number of exit planes."""
        return len(self.exit_planes)

    @abstractmethod
    def generate_slices(self, first_slice: int = 0, last_slice: Optional[int] = None):
        pass

    def generate_chunked_slices(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        chunk_size: int | str = "auto",
    ):
        """
        Generate the slices of ``generate_slices``, each as a chunk of its own.

        A class that groups slices into memory-budgeted chunks overrides this;
        ``chunk_size`` is then the number of slices per chunk.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice.
        last_slice : int, optional
            Index of the last slice.
        chunk_size : int or str, optional
            Not used: every chunk is a single slice.

        Yields
        ------
        FieldArray
            A single slice, as ``generate_slices`` yields it.
        """
        yield from self.generate_slices(first_slice, last_slice)

    @abstractmethod
    def build(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        chunks: int = 1,
        lazy: Optional[bool] = None,
    ):
        pass

    def __len__(self) -> int:
        return self.num_slices

    @property
    def num_slices(self) -> int:
        """Number of projected potential slices."""
        return len(self.slice_thickness)

    @property
    @abstractmethod
    def slice_thickness(self) -> tuple[float, ...]:
        """Slice thicknesses for each slice."""
        pass

    @property
    def slice_limits(self) -> list[tuple[float, float]]:
        """The entrance and exit thicknesses of each slice [Å]."""
        return slice_limits(self.slice_thickness)

    @property
    def thickness(self) -> float:
        """Thickness of the potential [Å]."""
        return sum(self.slice_thickness)

    def __iter__(self):
        for slic in self.generate_slices():
            yield slic

    def project(self) -> Images:
        """
        Sum of the potential slices as an image.

        Returns
        -------
        projected : Images
            The projected potential.
        """
        return self.build().project()

    @property
    def _default_ensemble_chunks(self) -> tuple:
        return validate_chunks(self.ensemble_shape, (1,) * len(self.ensemble_shape))

    def to_images(self):
        """
        Converts the potential to an ensemble of images.

        Returns
        -------
        image_ensemble : Images
            The potential slices as images.
        """
        return self.build().to_images()

    def show(self, project: bool = True, **kwargs):
        """
        Show the potential projection. This requires building all potential slices.

        Parameters
        ----------
        project : bool, optional
            Show the projected potential (True, default) or show all potential slices.
            It is recommended to index a subset of the potential slices when this
            keyword set to False.
        kwargs :
            Additional keyword arguments for the show method of :class:`.Images`.
        """
        kwargs.setdefault("interpolation", "antialiased")
        if project:
            return self.project().show(**kwargs)
        else:
            if "explode" not in kwargs.keys():
                kwargs["explode"] = True

            return self.to_images().show(**kwargs)

    def depth_profile(
        self,
        projection_axis: str = "y",
        depth: Optional[float] = None,
    ) -> Images:
        """Create a depth profile by projecting the potential along a spatial axis.

        Parameters
        ----------
        projection_axis : str
            Spatial axis to project (sum) along. ``"y"`` (default) produces an
            x–z cross-section; ``"x"`` produces a y–z cross-section.
        depth : float, optional
            If given, project only over a finite slab of this thickness [Å],
            centered on the midpoint of the projected axis. The number of grid
            points is rounded to the nearest integer. If ``None``, the full
            extent is projected.

        Returns
        -------
        depth_profile : Images
            2D image(s) with the remaining spatial axis horizontal and depth
            (z) vertical. Any ensemble axes (e.g. frozen phonons) are preserved.
        """
        return self.build().depth_profile(
            projection_axis=projection_axis,
            depth=depth,
        )

    def show_depth_profile(
        self,
        projection_axis: str = "y",
        depth: Optional[float] = None,
        z_scale: float = 1.0,
        slice_lines: bool = True,
        ax=None,
        cbar: bool = False,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        power: float = 1.0,
        common_color_scale: bool = False,
        explode: bool | Sequence[int] = (),
        figsize: Optional[tuple[int, int]] = None,
        title: bool | str = True,
        **kwargs,
    ):
        """Show a depth cross-section of the potential.

        Parameters
        ----------
        projection_axis : str
            Spatial axis to project (sum) along. ``"y"`` (default) produces an
            x–z cross-section; ``"x"`` produces a y–z cross-section.
        depth : float, optional
            If given, project only over a finite slab of this thickness [Å],
            centered on the midpoint of the projected axis. The number of grid
            points is rounded to the nearest integer. If ``None``, the full
            extent is projected.
        z_scale : float
            Scaling factor for the z-axis relative to the spatial axis.
            Values less than 1 compress the z-axis, making panels of thick
            specimens more compact. Default is 1.0 (equal scaling).
        slice_lines : bool
            If True (default), draw horizontal lines at slice boundaries.
        ax : matplotlib.axes.Axes, optional
            If given the plot is added to the axis.
        cbar : bool, optional
            Add a colorbar to the plot. Default is False.
        cmap : str, optional
            Matplotlib colormap name.
        vmin : float, optional
            Minimum of the color scale.
        vmax : float, optional
            Maximum of the color scale.
        power : float
            Show image on a power scale.
        common_color_scale : bool, optional
            If True, all images in a grid share the same color scale.
        explode : bool or sequence of int, optional
            If True, create a grid of images for ensemble items.
        figsize : two int, optional
            Figure size as (width, height) in inches.
        title : bool or str, optional
            Column title for the images.
        **kwargs
            Additional keyword arguments passed to the show method.

        Returns
        -------
        visualization : Visualization
        """
        from abtem.visualize import Visualization

        profile = self.depth_profile(
            projection_axis=projection_axis,
            depth=depth,
        )

        if figsize is None and ax is None:
            spatial_extent = profile.extent[0]
            z_extent = profile.extent[1]

            if explode is True or (isinstance(explode, Sequence) and explode):
                n_panels = (
                    profile.ensemble_shape[0] if profile.ensemble_shape else 1
                )
            else:
                n_panels = 1

            visual_ratio = (z_extent * z_scale) / spatial_extent
            panel_width = 3.0
            panel_height = panel_width * visual_ratio
            if panel_height < 1.0:
                panel_width = min(5.0, 1.0 / visual_ratio)
                panel_height = panel_width * visual_ratio
            elif panel_height > 8.0:
                panel_height = 8.0
                panel_width = panel_height / visual_ratio

            figsize = (
                panel_width * n_panels + 1.0 * n_panels + 0.5,
                max(2.5, panel_height + 1.5),
            )

        visualization = Visualization(
            measurement=profile,
            ax=ax,
            common_scale=common_color_scale,
            figsize=figsize,
            title=title,
            aspect=False,
            share_x=True,
            share_y=True,
            explode=explode,
            overlay=(),
            interactive=True,
            value_limits=(vmin, vmax),
            power=power,
            cmap=cmap,
            cbar=cbar,
            **kwargs,
        )

        spatial_label = "x" if projection_axis == "y" else "y"
        visualization.set_xlabel(f"{spatial_label} [Å]")
        visualization.set_ylabel("z [Å]")

        z_sampling = profile.sampling[1]
        for idx in np.ndindex(visualization.axes.shape):
            artist = visualization.artists[idx]
            xlim = artist.get_xlim()
            ylim = artist.get_ylim()
            artist.set_extent(
                (xlim[0], xlim[1], ylim[0] + z_sampling / 2, ylim[1] + z_sampling / 2)
            )

        visualization.adjust_coordinate_limits_to_artists()

        for idx in np.ndindex(visualization.axes.shape):
            visualization.axes[idx].set_aspect(z_scale)

        if slice_lines:
            limits = self.slice_limits
            z_boundaries = sorted({z for lo, hi in limits for z in (lo, hi)})
            for idx in np.ndindex(visualization.axes.shape):
                for z in z_boundaries:
                    visualization.axes[idx].axhline(
                        z, color="white", linewidth=0.5, alpha=0.5
                    )

        return visualization

    def interpolate_line(
        self,
        start: tuple[float, float] | tuple[float, float, float] | Atom | None = None,
        end: tuple[float, float] | tuple[float, float, float] | Atom | None = None,
        sampling: Optional[float] = None,
        gpts: Optional[int] = None,
        width: float = 0.0,
        margin: float = 0.0,
        order: int = 3,
        endpoint: bool = False,
        fractional: bool = False,
        projected: bool = True,
    ) -> RealSpaceLineProfiles:
        """
        Interpolate the potential along a line. Either `sampling` or `gpts` must be
        provided. This requires building the potential (done automatically).

        Parameters
        ----------
        start : two or three float, Atom, optional
            Starting position of the line [Å] (alternatively taken from a selected
            atom). Given as lateral `(x, y)` coordinates when `projected` is True.
            When `projected` is False, a depth coordinate may be added, `(x, y, z)`
            (default `z` is 0.).
        end : two or three float, Atom, optional
            Ending position of the line [Å]. See `start`.
        sampling : float, optional
            Sampling of grid points along the line [Å].
        gpts : int, optional
            Number of grid points along the line.
        width : float, optional
            The interpolation will be averaged across a perpendicular distance equal to
            this width. Only supported when `projected` is True.
        margin : float, optional
            Add margin [Å] to the start and end of the interpolated line. Only
            supported when `projected` is True.
        order : int, optional
            The spline interpolation order.
        endpoint : bool
            Sets whether the ending position is included or not.
        fractional : bool
            If True, use fractional coordinates with respect to the lateral extent
            (and, if `projected` is False, the thickness) of the potential.
        projected : bool, optional
            If True (default), interpolate the projected (thickness-summed)
            potential; `start` and `end` are lateral positions. If False, interpolate
            the full 3D potential along a (possibly tilted) line in 3D; `start` and
            `end` may include a depth coordinate. Meaningful results in this case
            typically require the potential to be sliced with a fine, close-to-uniform
            slice thickness.

        Returns
        -------
        line_profiles : RealSpaceLineProfiles
            The interpolated line(s).
        """
        return self.build().interpolate_line(
            start=start,
            end=end,
            sampling=sampling,
            gpts=gpts,
            width=width,
            margin=margin,
            order=order,
            endpoint=endpoint,
            fractional=fractional,
            projected=projected,
        )

    def interpolate_line_at_position(
        self,
        center: tuple[float, float] | tuple[float, float, float] | Atom,
        angle: float = 0.0,
        extent: float = 1.0,
        gpts: Optional[int] = None,
        sampling: Optional[float] = None,
        width: float = 0.0,
        order: int = 3,
        endpoint: bool = True,
        projected: bool = True,
    ) -> RealSpaceLineProfiles:
        """
        Interpolate the potential along a line centered at a given position. This
        requires building the potential (done automatically).

        Parameters
        ----------
        center : two or three float, Atom
            Center position of the line [Å]. May be given as an Atom. Given as a
            lateral `(x, y)` position when `projected` is True. When `projected` is
            False, a depth coordinate may be added, `(x, y, z)` (default `z` is 0.,
            or the atom's `z` coordinate for an `Atom`); the line then runs at
            constant depth.
        angle : float
            Angle of the line in the lateral `xy`-plane [deg.].
        extent : float
            Extent of the line [Å].
        gpts : int, optional
            Number of grid points along the line.
        sampling : float, optional
            Sampling of grid points along the line [Å].
        width : float, optional
            The interpolation will be averaged across a perpendicular distance equal to
            this width. Only supported when `projected` is True.
        order : int, optional
            The spline interpolation order.
        endpoint : bool
            Sets whether the ending position is included or not.
        projected : bool, optional
            See :meth:`.interpolate_line`.

        Returns
        -------
        line_profiles : RealSpaceLineProfiles
            The interpolated line(s).
        """
        return self.build().interpolate_line_at_position(
            center=center,
            angle=angle,
            extent=extent,
            gpts=gpts,
            sampling=sampling,
            width=width,
            order=order,
            endpoint=endpoint,
            projected=projected,
        )


class BasePotential(BaseField, metaclass=ABCMeta):
    """Base class of all potentials. Documented in the subclasses."""

    @property
    def base_axes_metadata(self):
        """List of AxisMetadata for the base axes."""
        return [
            ThicknessAxis(
                label="z", values=tuple(np.cumsum(self.slice_thickness)), units="Å"
            ),
            RealSpaceAxis(
                label="x", sampling=self.sampling[0], units="Å", endpoint=False
            ),
            RealSpaceAxis(
                label="y", sampling=self.sampling[1], units="Å", endpoint=False
            ),
        ]


def validate_potential(
    potential: Atoms | BasePotential, waves: Optional[BaseWaves] = None
) -> BasePotential:
    if isinstance(potential, (Atoms, BaseFrozenPhonons)):
        device = None
        if waves is not None:
            device = waves.device

        potential = Potential(potential, device=device)
    # elif not isinstance(potential, BasePotential):
    #    raise ValueError()

    if waves is not None and potential is not None:
        potential.match_grid(waves)

    return potential


def _validate_exit_planes(exit_planes, num_slices):
    if isinstance(exit_planes, int):
        if exit_planes > num_slices:
            return (num_slices - 1,)

        exit_planes = list(range(exit_planes - 1, num_slices, exit_planes))
        if exit_planes[-1] != (num_slices - 1):
            exit_planes.append(num_slices - 1)
        exit_planes = (-1,) + tuple(exit_planes)
    elif exit_planes is None:
        exit_planes = (num_slices - 1,)

    return exit_planes


def _require_cell_transform(cell, box, plane, origin):
    """
    Whether atoms with the given cell must be transformed (rotated, translated,
    repeated or strained) to fill a potential with the given `box`, `plane` and
    `origin`. A `box` of None stands for the default box.
    """
    if not is_cell_orthogonal(cell):
        return True

    if plane != "xy":
        return True

    if tuple(origin) != (0.0, 0.0, 0.0):
        return True

    if box is not None and tuple(box) != tuple(np.diag(cell)):
        return True

    return False


def _three_floats(values, name) -> tuple[float, float, float]:
    """`values` as three finite floats; strings are not numbers."""
    try:
        if isinstance(values, (str, bytes)) or any(
            isinstance(value, (str, bytes)) for value in values
        ):
            raise TypeError
        floats = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be three numbers [Å], got {values!r}") from None

    if len(floats) != 3:
        raise ValueError(
            f"{name} must have three elements, got {len(floats)}: {values!r}"
        )

    if not np.all(np.isfinite(floats)):
        raise ValueError(f"{name} must be finite, got {values!r}")

    return floats


def _validate_box(box) -> tuple[float, float, float]:
    """The given box as three positive, finite lengths [Å]."""
    lengths = _three_floats(box, "box")

    if min(lengths) <= 0.0:
        raise ValueError(f"box must be three positive lengths [Å], got {box!r}")

    return lengths


def _validate_origin(origin) -> tuple[float, float, float]:
    """The given origin as three finite floats [Å]; None is the zero origin."""
    if origin is None:
        return (0.0, 0.0, 0.0)

    return _three_floats(origin, "origin")


def _default_box(cell, plane) -> tuple[float, float, float]:
    """
    The box of a potential that needs a cell transform and was given no box: the
    best orthogonal cell of the atoms' cell rotated to `plane`, as
    `orthogonalize_cell` rotates it (axes permuted, then the rotation about the new
    z that `standardize_cell` applies). Raises a RuntimeError for a cell that
    cannot be rotated to `plane`, which every build would otherwise raise later.
    """
    if not isinstance(plane, str):
        raise NotImplementedError
    if plane == "xy":
        return tuple(best_orthogonal_cell(np.array(cell)))
    try:
        cell = _cell_in_plane_frame(cell, plane)
    except RuntimeError as error:
        raise RuntimeError(
            f"The cell cannot be rotated to plane={plane!r} ({error}): a plane "
            "other than 'xy' needs a cell that is orthogonal once rotated to it. "
            "Make the cell orthogonal first, for example with "
            "abtem.orthogonalize_cell(atoms)."
        ) from error
    return tuple(best_orthogonal_cell(cell))


class _FieldBuilder(BaseField):
    # False for builders whose slices are interpolated from a calculator's grid of
    # the atoms' own cell rather than computed from transformed atoms: they place
    # their field in the default box at the default origin only, and reject any
    # other box or origin instead of ignoring it.
    _supports_box_and_origin: bool = True

    def __init__(
        self,
        array_object: Type[FieldArray],
        slice_thickness: float | tuple[float, ...],
        cell: np.ndarray | Cell,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        box: Optional[tuple[float, float, float]] = None,
        plane: (
            str | tuple[tuple[float, float, float], tuple[float, float, float]]
        ) = "xy",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        periodic: bool = True,
        device: Optional[str] = None,
        non_orthogonal: Optional[bool] = None,
    ):
        self._array_object = array_object

        origin = _validate_origin(origin)

        box_given = box is not None

        if not self._supports_box_and_origin:
            self._check_default_box_and_origin(cell, box, plane, origin)
            box = None

        if box is not None:
            box = _validate_box(box)

        cell_array = np.array(cell, dtype=float)
        # Resolve whether to build directly on the crystal's in-plane (a, b) grid instead
        # of orthogonalising. This requires a and b to lie in the xy-plane so slicing
        # along the beam stays valid; the in-plane cell may be non-orthogonal (a skewed
        # grid) and/or the c-axis may be tilted out of z (the atoms then simply drift
        # laterally with depth, captured by their true coordinates -- no shear or tilted
        # propagator needed).
        #   None  -> auto: use it iff the cell is non-orthogonal but a, b are in-plane,
        #            plane is "xy" and no explicit box was requested.
        #   True  -> force it (raise if a, b not in-plane / plane != "xy").
        #   False -> force the orthogonalising path (legacy behaviour).
        if non_orthogonal is None:
            use_skew = (
                not is_cell_orthogonal(cell_array)
                and is_cell_ab_in_plane(cell_array)
                and isinstance(plane, str)
                and plane == "xy"
                # A builder that supports only its default box drops a box it
                # is given once it is checked; it was still asked for.
                and not box_given
            )
        elif non_orthogonal:
            if not is_cell_ab_in_plane(cell_array):
                raise NotImplementedError(
                    "non_orthogonal potentials require the a- and b-axes in the "
                    "xy-plane (the c-axis may be tilted out of z)"
                )
            if not (isinstance(plane, str) and plane == "xy"):
                raise NotImplementedError(
                    "non_orthogonal potentials currently support only plane='xy'"
                )
            use_skew = True
        else:
            use_skew = False
        self._non_orthogonal = use_skew

        if use_skew:
            cell_2d = cell_array[:2, :2]
            extent = tuple(np.linalg.norm(cell_2d, axis=1))
            box = (extent[0], extent[1], float(cell_array[2, 2]))
            # Only carry a skewed in-plane metric when a and b are genuinely
            # non-orthogonal; a tilted c-axis with an orthogonal (a, b) keeps a plain
            # orthogonal grid (no spurious cell in the metadata / detectors).
            in_plane_orthogonal = abs(float(cell_2d[0] @ cell_2d[1])) <= 1e-9 * (
                extent[0] * extent[1]
            )
            self._grid = Grid(
                extent=extent,
                gpts=gpts,
                sampling=sampling,
                lock_extent=True,
                cell=None if in_plane_orthogonal else cell_2d,
            )
        else:
            if _require_cell_transform(cell, box=box, plane=plane, origin=origin):
                if not isinstance(plane, str):
                    raise NotImplementedError
                default_box = _default_box(cell, plane)
                if box is None:
                    box = default_box
                    if periodic and not box_given:
                        _warn_if_box_is_strained(cell, box, plane, default=True)
                elif periodic and box != default_box:
                    _warn_if_box_is_strained(cell, box, plane)

            elif box is None:
                box = tuple(np.diag(cell))

            self._grid = Grid(
                extent=box[:2], gpts=gpts, sampling=sampling, lock_extent=True
            )
        self._device = validate_device(device)

        self._box = box
        self._plane = plane
        self._origin = origin
        self._periodic = periodic

        self._slice_thickness = _validate_slice_thickness(
            slice_thickness, thickness=box[2]
        )
        self._exit_planes = _validate_exit_planes(
            exit_planes, len(self._slice_thickness)
        )

    def _check_default_box_and_origin(self, cell, box, plane, origin):
        """Raise unless `box` is the default box and `origin` is zero."""
        if _require_cell_transform(cell, box=None, plane=plane, origin=(0.0,) * 3):
            default_box = _default_box(cell, plane)
        else:
            default_box = tuple(np.diag(cell))

        given_box = None if box is None else _validate_box(box)

        if origin != (0.0, 0.0, 0.0) or (
            given_box is not None
            and not np.allclose(given_box, default_box, rtol=1e-9, atol=0.0)
        ):
            raise NotImplementedError(
                f"{type(self).__name__} supports only its default box "
                f"{tuple(float(b) for b in default_box)} and origin (0, 0, 0), "
                f"got box={given_box} and origin={origin}."
            )

    @property
    def slice_thickness(self) -> tuple[float, ...]:
        return self._slice_thickness

    @property
    def exit_planes(self) -> tuple[int]:
        return self._exit_planes

    @property
    def device(self) -> str:
        """The device where the potential is created."""
        return self._device

    @property
    def periodic(self) -> bool:
        """Specifies whether the potential is periodic."""
        return self._periodic

    @property
    def plane(
        self,
    ) -> str | tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The plane relative to the atoms mapped to `xy` plane of the potential,
        i.e. the plane is perpendicular to the propagation direction."""
        return self._plane

    @property
    def box(self) -> tuple[float, float, float]:
        """The extent of the potential in `x`, `y` and `z`."""
        return self._box

    @property
    def origin(self) -> tuple[float, float, float]:
        """The origin relative to the provided atoms mapped to the origin of the
        potential."""
        # Validated on every read as well as on construction, because a
        # potential restored from a pickle skips `__init__` and may carry the
        # origin exactly as it was passed (None, a list).
        return _validate_origin(self._origin)

    @property
    def non_orthogonal(self) -> bool:
        """Whether the potential is built on a non-orthogonal (skewed) in-plane grid."""
        return self._non_orthogonal

    def __getitem__(self, item) -> PotentialArray:
        return self.build(lazy=False)[item]

    @staticmethod
    def _wrap_build_potential(potential, first_slice, last_slice):
        potential = potential.item()
        array = potential.build(first_slice, last_slice, lazy=False).array
        return array

    def build(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        max_batch: int | str = 1,
        lazy: Optional[bool] = None,
    ) -> FieldArray:
        """
        Build the potential.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the generated potential.
        last_slice : int, optional
            Index of the last slice of the generated potential
        max_batch : int or str, optional
            Maximum number of slices to calculate in task. Default is 1.
        lazy : bool, optional
            If True, create the wave functions lazily, otherwise, calculate instantly.
            If None, this defaults to the value set in the configuration file.

        Returns
        -------
        potential_array : PotentialArray
            The built potential as an array.
        """
        lazy = validate_lazy(lazy)

        self.grid.check_is_defined()

        if last_slice is None:
            last_slice = len(self)

        if lazy:
            blocks = self.ensemble_blocks(self._default_ensemble_chunks)

            xp = get_array_module(self.device)
            chunks = validate_chunks(self.ensemble_shape, self._default_ensemble_chunks)
            chunks = chunks + (last_slice - first_slice,) + self.base_shape[1:]

            if self.ensemble_shape:
                new_axis = tuple(
                    range(
                        len(self.ensemble_shape),
                        len(self.ensemble_shape) + len(self.base_shape),
                    )
                )
            else:
                new_axis = tuple(range(0, len(self.base_shape)))

            # new_axis = (0, 1, 2) # This was causing problems with FrozenPhonons

            array = da.map_blocks(
                self._wrap_build_potential,
                blocks,
                new_axis=new_axis,
                first_slice=first_slice,
                last_slice=last_slice,
                chunks=chunks,
                meta=xp.array((), dtype=get_dtype(complex=False)),
            )

        else:
            xp = get_array_module(self.device)

            array = xp.zeros(
                self.ensemble_shape + (last_slice - first_slice,) + self.base_shape[1:],
                dtype=get_dtype(complex=False),
            )

            if self.ensemble_shape:
                for i, _, potential_wrapped in self.generate_blocks(1):
                    potential = potential_wrapped.item()

                    for j, slic in enumerate(
                        potential.generate_slices(first_slice, last_slice)
                    ):
                        array[i + (j,)] = slic.array[0]

            else:
                for j, slic in enumerate(self.generate_slices(first_slice, last_slice)):
                    array[j] = slic.array[0]

        # The exit planes index the slices of the whole potential; the range keeps
        # those inside it, counted from its first slice. With none left, the exit
        # plane is the last slice of the range. A range starting at the first slice
        # also keeps the entrance plane (-1, the incident wave) of the potential.
        exit_planes = tuple(
            plane - first_slice
            for plane in self.exit_planes
            if first_slice <= plane < last_slice
        ) or (last_slice - first_slice - 1,)
        if first_slice == 0 and -1 in self.exit_planes:
            exit_planes = (-1,) + exit_planes

        output_potential = self._array_object(
            array,
            sampling=self._valid_sampling,
            slice_thickness=self.slice_thickness[first_slice:last_slice],
            exit_planes=exit_planes,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            cell=self.grid.cell,
        )
        return output_potential

    def generate_chunked_slices(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        chunk_size: int | str = "auto",
    ):
        """
        Generate the slices in memory-budgeted chunks.

        Each chunk is built eagerly with ``build(chunk_start, chunk_end,
        lazy=False)``, so it has the sampling and ensemble axes of the built
        field. The caller discards a chunk after use, which bounds the memory
        (on a GPU, the VRAM) held at once to one chunk instead of the whole
        slice axis of ``build()``. The GPAW magnetic builders compute their
        field once, on the first chunk, and keep its slices for the others.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice.
        last_slice : int, optional
            Index of the last slice.
        chunk_size : int or str, optional
            Number of slices per chunk. ``"auto"`` selects based on the
            configured memory budget (``dask.chunk-size`` on CPU,
            ``dask.chunk-size-gpu`` on GPU), priced for a chunk that holds
            every member of the ensemble. Can also be set globally via the
            ``potential.slice-chunk-size`` configuration key.

        Yields
        ------
        FieldArray
            An eagerly computed chunk of contiguous slices, of the class this
            builder builds (``PotentialArray`` for a potential), with exit planes
            counted from its first slice.
        """
        from abtem.core.chunks import (
            estimate_potential_chunk_size,
            generate_chunks,
        )

        if last_slice is None:
            last_slice = len(self)

        if chunk_size == "auto":
            # A chunk holds every member of the ensemble.
            chunk_size = estimate_potential_chunk_size(
                self.ensemble_shape + self.base_shape[1:], self.device
            )

        # Cap so the whole range is one chunk when it fits in the budget,
        # then distribute evenly so the last chunk is never smaller than
        # necessary (equal_sized_chunks inside generate_chunks handles this).
        chunk_size = min(chunk_size, last_slice - first_slice)

        exit_plane_after = self._exit_plane_after

        for chunk_start, chunk_end in generate_chunks(
            last_slice - first_slice, chunks=chunk_size, start=first_slice
        ):
            chunk = self.build(
                first_slice=chunk_start, last_slice=chunk_end, lazy=False
            )

            # A chunk keeps only the exit planes inside its range, counted from
            # its first slice: build() would add the entrance plane to the first
            # range and, with none in range, the range's last slice.
            chunk._exit_planes = tuple(
                np.where(exit_plane_after[chunk_start:chunk_end])[0]
            )

            yield chunk


def _plane_frame(atoms: Atoms, plane, small_cell_components: float = 0.0) -> np.ndarray:
    """The linear map `rotate_atoms_to_plane` applies to the positions of
    `atoms`, acting on row vectors. Cell components smaller than
    `small_cell_components` are zeroed first, as `orthogonalize_cell` does
    before it rotates the atoms."""
    if plane == "xy":
        return np.eye(3)
    atoms = atoms.copy()
    if small_cell_components:
        cell = np.array(atoms.cell)
        cell[np.abs(cell) < small_cell_components] = 0.0
        atoms.set_cell(cell)
    return _rotate_atoms_to_plane(atoms, plane)[1]


def _pad_atoms_marking_images(
    atoms: Atoms, margins: tuple[float, float, float]
) -> tuple[Atoms, np.ndarray]:
    """`pad_atoms`, and a mask that is True for the atoms of the result that are
    the given atoms rather than images `pad_atoms` added."""
    tagged = atoms.copy()
    tagged.set_array("abtem_pad_index", np.arange(len(atoms)))
    padded = pad_atoms(tagged, margins=margins)
    index = padded.arrays.pop("abtem_pad_index")
    # An image is the given atom shifted by a nonzero multiple of the cell
    # lengths along a padded axis; the given atom itself is not shifted.
    # Measured in fractional coordinates, so that it also holds on a skewed
    # cell, whose lattice vectors are not along the Cartesian axes.
    inverse_cell = np.linalg.inv(np.array(atoms.cell, dtype=float))
    shifts = np.round((padded.positions - atoms.positions[index]) @ inverse_cell)
    return padded, np.all(shifts == 0.0, axis=1)


def _in_cell(atoms: Atoms) -> np.ndarray:
    """A mask of the atoms in the half-open cell, by the interval `atoms_in_cell`
    uses without a margin."""
    scaled_positions = atoms.positions / np.diag(np.array(atoms.cell))
    return np.all(
        (scaled_positions >= -1e-12) & (scaled_positions < 1.0 - 1e-12), axis=1
    )


def _fold_far_outside_faces(atoms: Atoms) -> None:
    """Wrap along z, in place, the atoms more than `FACE_SLICE_WARNING_DISTANCE`
    outside the orthogonal cell along z, onto their image in the cell."""
    height = float(atoms.cell[2, 2])
    z = atoms.positions[:, 2]
    far = _far_outside_faces(z, height)
    z[far] = np.mod(z[far], height)


def _given_configurations(frozen_phonons: BaseFrozenPhonons) -> list[Atoms]:
    """The atoms an ensemble is given, as far as they are known without
    computing anything: every member of an AtomsEnsemble whose trajectory is in
    memory, none of a lazily read one, otherwise the atoms the ensemble
    displaces."""
    trajectory = getattr(frozen_phonons, "trajectory", None)
    if trajectory is None:
        return [frozen_phonons.atoms]
    if isinstance(trajectory, np.ndarray):
        return list(trajectory.ravel())
    return []


def _heights_along_plane_normal(atoms: Atoms, plane) -> tuple[np.ndarray, float]:
    """The heights of the atoms of an orthogonal cell along the axis that
    `rotate_atoms_to_plane` turns along z, and the cell's height along it, as
    they are given: the rotation wraps every atom into the cell. Reversing the
    axis (z to height - z) does not change how far outside the cell an atom is."""
    if plane == "xy":
        return atoms.positions[:, 2], float(np.array(atoms.cell)[2, 2])
    frame = _rotate_atoms_to_plane(Atoms(cell=atoms.cell), plane)[1]
    axis = int(np.argmax(np.abs(frame[:, 2])))
    return atoms.positions[:, axis], float(np.array(atoms.cell)[axis, axis])


def _warn_if_given_atoms_far_outside_along_z(
    configurations: Sequence[Atoms], plane="xy"
) -> None:
    """Warn about the atoms given more than `FACE_SLICE_WARNING_DISTANCE`
    outside the cell along z of a non-periodic potential that does not
    transform its cell, or only rotates it to `plane`. They are folded into the
    cell as images of the repeated structure, which is unlikely to be meant if
    the cell should enclose them."""
    counts = []
    for atoms in configurations:
        heights, height = _heights_along_plane_normal(atoms, plane)
        counts.append(int(_far_outside_faces(heights, height).sum()))
    if not any(counts):
        return

    warnings.warn(
        f"{max(counts)} atom(s) are given more than {FACE_SLICE_WARNING_DISTANCE} "
        f"Å outside the cell along z (0 to {height:.4g} Å). A non-periodic "
        "potential is cut out of the repeated structure, so they are taken as "
        "their images in the cell. If the cell should enclose them, make it "
        "taller.",
        stacklevel=_stacklevel_outside_package(),
    )


class _FieldBuilderFromAtoms(_FieldBuilder):
    # _sliced_atoms is derived state: get_sliced_atoms() builds it lazily from
    # the atoms, the slicing and the cell, all of which are compared already.
    # Declared here, where the attribute is created, so that Potential,
    # MagneticField and VectorPotential all inherit it rather than one of them
    # carrying it for the others. Without it a built field stopped comparing
    # equal to an identical unbuilt one, i.e. equality depended on whether a
    # result had been computed.
    #
    # The exclusion is blunt, and deliberately so for now. get_sliced_atoms()
    # returns this object rather than a copy, so a caller who mutates what it
    # returned changes what the field builds while `==` still reports equal.
    # What that costs depends on the projection, because the two cache classes
    # differ. For projection="infinite" the cache is a SliceIndexedAtoms,
    # whose _slice_index is a list of arrays that safe_equality cannot compare
    # -- `==` on it raises ValueError, which becomes "unequal" -- so the
    # comparison is constant-False and excluding it gives up nothing. For
    # projection="finite" the cache is a SlicedAtoms, which has no
    # _slice_index and compares correctly; there the exclusion does give up a
    # real check, one that the np.all fix above would otherwise have made
    # catch the mutation. Neither path regresses against the old behaviour,
    # which missed the mutation on both. Comparing derived state only when
    # both operands have it is the better rule, and belongs with the same
    # question for lazy arrays -- both are abTEM issue #413 -- rather than
    # here.
    _eq_exclude = ("_sliced_atoms",)

    def __init__(
        self,
        atoms: Atoms | BaseFrozenPhonons,
        array_object: Type[FieldArray],
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: (
            str | tuple[tuple[float, float, float], tuple[float, float, float]]
        ) = "xy",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        integrator=None,
        device: Optional[str] = None,
        non_orthogonal: Optional[bool] = None,
    ):
        self._frozen_phonons = _validate_frozen_phonons(atoms)
        self._integrator = integrator
        self._sliced_atoms: Optional[BaseSlicedAtoms] = None
        self._array_object = array_object

        super().__init__(
            array_object=array_object,
            gpts=gpts,
            sampling=sampling,
            cell=self._frozen_phonons.cell,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            origin=origin,
            box=box,
            periodic=periodic,
            non_orthogonal=non_orthogonal,
        )

        # Every build folds atoms given far outside along z into the cell; this
        # says so where the potential is made, which names the caller's line for
        # a lazy build too. A builder rebuilt from this one (a lazy block) does
        # not repeat it. The members of a lazily read trajectory are not known
        # here; each block checks its own as it is made (_from_partitioned_args).
        if not _box_strain_warning_suppressed.get() and self._reports_given_atoms():
            _warn_if_given_atoms_far_outside_along_z(
                _given_configurations(self._frozen_phonons), self.plane
            )

    def _reports_given_atoms(self) -> bool:
        """Whether atoms given far outside the cell along z are reported: for a
        non-periodic potential that uses its cell as it is or only rotates it to
        the plane. A cut cell (`cut_cell`) folds them silently."""
        if self.periodic:
            return False
        if not self._requires_cell_transform():
            return True
        # The rotate-only path of _transform_atoms.
        cell = self._frozen_phonons.cell
        if not (
            is_cell_orthogonal(cell)
            and self.plane != "xy"
            and tuple(self.origin) == (0.0, 0.0, 0.0)
        ):
            return False
        rotated = _rotate_atoms_to_plane(Atoms(cell=cell), self.plane)[0]
        return tuple(np.diag(rotated.cell)) == self.box

    @property
    def frozen_phonons(self) -> BaseFrozenPhonons:
        """Ensemble of atomic configurations representing frozen phonons."""
        return self._frozen_phonons

    @property
    def num_configurations(self) -> int:
        """Size of the ensemble of atomic configurations representing frozen phonons."""
        return len(self.frozen_phonons)

    @property
    def integrator(self):
        """The integrator determining how the projection integrals for each slice is
        calculated."""
        return self._integrator

    def _requires_cell_transform(self) -> bool:
        """Whether the atoms are transformed (rotated, translated, repeated or
        strained) to fill the potential."""
        return _require_cell_transform(
            self._frozen_phonons.cell,
            box=self.box,
            plane=self.plane,
            origin=self.origin,
        )

    def _folds_atoms_far_outside_along_z(self) -> bool:
        """Whether atoms given in the potential's own cell, more than
        `FACE_SLICE_WARNING_DISTANCE` outside it along z, are folded into the cell
        (`_fold_far_outside_faces`) when the slices are built: for a non-periodic
        potential that is not cut from a transformed cell
        (`_reports_given_atoms`) and whose integrator is infinite. A finite one
        wraps them by its margins."""
        return self._reports_given_atoms() and not self.integrator.finite

    def _cutoffs(self):
        atoms = self.frozen_phonons.atoms
        unique_numbers = np.unique(atoms.numbers)
        return tuple(
            self._integrator.cutoff(chemical_symbols[number])
            for number in unique_numbers
        )

    def get_transformed_atoms(self):
        """
        The atoms used in the multislice algorithm, transformed to the given plane,
        origin and box.

        Returns
        -------
        transformed_atoms : Atoms
            Transformed atoms.
        """
        atoms = self._transform_atoms()[0]
        atoms.arrays.pop(SOURCE_INDEX, None)
        return atoms

    def _margins(self) -> tuple[float, float, float]:
        """The margin the integrator needs along each axis beyond the cell.

        A real-space finite integrator needs every atom within its cutoff of
        the cell in all three directions. One built by FFT in-plane already
        sees the in-plane neighbours through the periodic build, so it needs
        them along z only, and an infinite projection needs none at all.
        """
        if not self.integrator.finite:
            return (0.0, 0.0, 0.0)

        cutoffs = self._cutoffs()
        margin = max(cutoffs) if len(cutoffs) else 0.0

        if self.integrator.periodic:
            return (0.0, 0.0, margin)

        return (margin, margin, margin)

    def _transform_atoms(self) -> tuple[Atoms, bool, np.ndarray]:
        """The transformed atoms; whether they were cut out of a larger
        repeated structure, in which case they already carry the integrator's
        margin and must not be padded again; and the linear map from the
        Cartesian axes of the frozen phonons' atoms to those of the transformed
        atoms, acting on row vectors.

        Each transformed atom carries, in the ``SOURCE_INDEX`` array, the index
        of the atom of the frozen phonons' atoms it is a copy of.
        """
        atoms = self.frozen_phonons.atoms.copy()
        atoms.set_array(SOURCE_INDEX, np.arange(len(atoms)))

        if getattr(self, "_non_orthogonal", False):
            # keep the non-orthogonal cell; the grid carries the skewed metric
            return atoms, False, np.eye(3)

        # An orthogonal cell that only changes plane is rotated, as long as the
        # rotated cell is the box and there is no origin to translate it by.
        if (
            is_cell_orthogonal(atoms.cell)
            and self.plane != "xy"
            and tuple(self.origin) == (0.0, 0.0, 0.0)
        ):
            rotated, frame = _rotate_atoms_to_plane(atoms, self.plane)
            if tuple(np.diag(rotated.cell)) == self.box:
                return rotated, False, frame

        # `diag(atoms.cell) == self.box` is not by itself proof the cell is
        # orthogonal: for a near-orthorhombic cell with off-diagonal noise
        # below ~2e-8 relative, best_orthogonal_cell's box norms round to
        # the exact diagonal entries in float64 (see the matching guard in
        # atoms.py's orthogonalize_cell). _require_cell_transform checks
        # is_cell_orthogonal first, so such noisy cells still reach
        # orthogonalize_cell below instead of being silently used as-is.
        if _require_cell_transform(
            atoms.cell, box=self.box, plane=self.plane, origin=self.origin
        ):
            if self.periodic:
                plane_frame = _plane_frame(
                    atoms, self.plane, small_cell_components=1e-6
                )
                atoms, affine = orthogonalize_cell(
                    atoms,
                    box=self.box,
                    plane=self.plane,
                    origin=self.origin,
                    return_transform_matrix=True,
                    allow_transform=True,
                )
                return atoms, False, plane_frame @ affine
            else:
                # The margin comes from the larger repeated structure, so it
                # is the true neighbourhood of a cell that need not be
                # commensurate with it. Periodic padding on top of it added
                # images of the margin atoms -- in-plane, where an FFT build
                # also wraps them, and in depth -- and inflated the potential
                # of a transformed non-periodic cell many times over. An
                # infinite projection's cutoff is infinite, which cut every
                # atom; it needs no margin.
                plane_frame = _plane_frame(atoms, self.plane)
                atoms = cut_cell(
                    atoms,
                    cell=self.box,
                    plane=self.plane,
                    origin=self.origin,
                    margin=self._margins(),
                )
                return atoms, True, plane_frame

        return atoms, False, np.eye(3)

    def _displace(self, atoms: Atoms, frame: np.ndarray) -> Atoms:
        """The frozen phonons' displacement of atoms this potential transformed,
        as an object this potential owns."""
        displaced = self.frozen_phonons._randomize_transformed(atoms, frame)
        if displaced is not atoms:
            # Only `atoms` is known to be this potential's own copy; an
            # ensemble may return an object it keeps, such as a stored
            # configuration, which the wrap below must not change.
            displaced = displaced.copy()
        return displaced

    def _configuration(self) -> tuple[Atoms, Atoms, tuple[float, float, float]]:
        """This configuration of the atoms; the atoms as they are sliced; and
        the margin the integrator needs beyond the cell along each axis.

        The configuration is the transformed atoms, displaced, and wrapped
        into the cell when the potential is periodic. When it is non-periodic,
        atoms given outside the cell are images of the repeated structure it is
        cut from, and are folded into the cell before they are displaced: the
        atoms that the padding does not keep with all their images
        (`_wrap_far_atoms`) are wrapped along the axes the integrator pads, and
        for the infinite projection, which does not pad, the atoms given more
        than `FACE_SLICE_WARNING_DISTANCE` outside the cell along z are wrapped
        along z. An atom closer to a face than that keeps its position, and the
        infinite projection puts it in the face slice, as it does an atom the
        frozen phonons displace out of the cell. The atoms to slice add the
        atoms within the margin outside the cell: images of the configuration
        for a periodic potential, and for a non-periodic one the surrounding
        atoms, displaced independently of those in the cell.
        """
        atoms, is_cut, frame = self._transform_atoms()
        margins = self._margins()

        if self.periodic:
            atoms = self._displace(atoms, frame)
            # Shared with SliceIndexedAtoms, which applies the same wrap to the
            # atoms it is handed directly -- e.g. explicit core-loss ``sites``
            # and CrystalPotential's tiled atoms, which do not come through
            # here. In place, because _displace returns atoms this potential
            # owns: neither the atoms the potential stores nor an object the
            # frozen phonons keep can be reached from here.
            if not getattr(self, "_non_orthogonal", False):
                configuration = wrap_and_snap_atoms(atoms, copy=False)
            else:
                # Not wrap_and_snap_atoms: it snaps against `np.diag(cell)`,
                # which is not the right boundary for a non-orthogonal cell.
                # In place, for the same reason as above.
                atoms.wrap(eps=0.0)
                # wrap(eps=0.0) uses strict modulo: z positions that are tiny-negative
                # (floating-point artifact from ASE surface builders) become z ≈ cell_z
                # instead of z = 0.  The SliceIndexedAtoms bin edges are nudged down by
                # 1e-12 to fix cumsum drift, so any atom in (cell_z-1e-12, cell_z) falls
                # outside all bins and is silently dropped.  Snap those back to 0.
                cell_z = atoms.cell[2, 2]
                atoms.positions[atoms.positions[:, 2] > cell_z - 1e-10, 2] = 0.0

                # The same wrap(eps=0.0) issue affects the in-plane axes of a skewed
                # cell: an atom at fractional 0 can evaluate to a tiny negative
                # coordinate (floating point in the non-orthogonal inv(cell)) and wrap
                # to ~1, where it falls outside the grid and is dropped -- breaking the
                # primitive periodicity. Snap those boundary atoms back to 0.
                scaled = atoms.get_scaled_positions(wrap=False)
                snap = scaled > 1.0 - 1e-6
                snap[:, 2] = False  # z handled above
                if snap.any():
                    atoms.positions = atoms.positions - snap.astype(float) @ np.asarray(
                        atoms.cell
                    )

                # When the c-axis is tilted (has in-plane components) the 3D `wrap()`
                # leaves atoms inside the supercell parallelepiped, but their (x, y)
                # Cartesian coordinates can extend beyond the in-plane (a, b) box of the
                # multislice grid -- so they would fall outside the grid and be silently
                # skipped. Wrap each atom's (x, y) independently into the in-plane (a, b)
                # parallelogram at its current z (the physically equivalent in-plane
                # periodic image; the atoms are then in the same "z-separable" reference
                # frame that the rest of the slicing/integration machinery assumes).
                in_plane = np.asarray(atoms.cell[:2, :2], dtype=float)
                c_tilted = (
                    abs(atoms.cell[2, 0]) > 1e-12 or abs(atoms.cell[2, 1]) > 1e-12
                )
                if c_tilted:
                    inv = np.linalg.inv(in_plane.T)
                    uv = atoms.positions[:, :2] @ inv.T  # fractional (a, b) coords
                    uv = uv - np.floor(uv)
                    atoms.positions[:, :2] = uv @ in_plane
                    # After the in-plane wrap the atoms are positioned as if c were
                    # vertical; reflect that in the cell so downstream scaled-coordinate
                    # checks (e.g. atoms_in_cell in pad_atoms) use the right reference.
                    new_cell = np.array(atoms.cell, dtype=float)
                    new_cell[2, 0] = 0.0
                    new_cell[2, 1] = 0.0
                    atoms.set_cell(new_cell, scale_atoms=False)

                configuration = atoms
            # Repeats exactly the axes with a nonzero margin. A periodic
            # potential's atoms are never cut.
            return configuration, pad_atoms(configuration, margins=margins), margins

        # A non-periodic potential is displaced after padding, so that the
        # atoms in the margin are displaced independently of those in the
        # cell. A cut cell already holds the margin cut_cell supplied.
        if is_cut:
            in_cell = _in_cell(atoms)
        elif getattr(self, "_non_orthogonal", False):
            # The far-atom wraps below are along the Cartesian axes of an
            # orthogonal cell; a skewed cell is only padded, along its lattice
            # vectors.
            atoms, in_cell = _pad_atoms_marking_images(atoms, margins)
        else:
            # The images of an atom given outside the cell come from pad_atoms,
            # which does not reach an atom given far outside: in-plane for a
            # real-space integrator (the build is periodic in-plane), and along
            # z for every finite one.
            directions = ("" if self.integrator.periodic else "xy") + (
                "z" if self.integrator.finite else ""
            )
            _wrap_far_atoms(atoms, margins, directions)
            if not self.integrator.finite:
                _fold_far_outside_faces(atoms)
            atoms, in_cell = _pad_atoms_marking_images(atoms, margins)

        atoms = self._displace(atoms, frame)

        if len(atoms) == len(in_cell):
            configuration = atoms[in_cell]
        else:
            # The frozen phonons returned a different number of atoms.
            configuration = atoms.copy()

        return configuration, atoms, margins

    def _prepare_atoms(self):
        _, atoms, margins = self._configuration()
        atoms.arrays.pop(SOURCE_INDEX, None)

        if self.integrator.finite:
            sliced_atoms = SlicedAtoms(
                atoms=atoms,
                slice_thickness=self.slice_thickness,
                z_padding=margins[2],
            )
        else:
            sliced_atoms = SliceIndexedAtoms(
                atoms=atoms,
                slice_thickness=self.slice_thickness,
                # Non-periodic potentials are randomised after padding and are
                # deliberately never wrapped; see the note in
                # SliceIndexedAtoms.__init__.
                wrap=self.periodic,
            )

        return sliced_atoms

    def to_atoms_ensemble(self) -> AtomsEnsemble | EnergyResolvedAtomsEnsemble:
        """
        The atomic configurations this potential simulates, one for each frozen
        phonon configuration.

        Each configuration is the atoms in the potential's box: transformed to
        its plane, origin and box, displaced by the frozen phonons, and wrapped
        into the box when the potential is periodic. A potential built from a
        configuration alone, with this potential's `gpts` (as
        ``gpts=potential.gpts``; a `sampling` that derives the grid from the atoms,
        such as ``"auto"``, may give the displaced atoms another grid)
        and the same slicing, projection and `periodic`, and the default plane,
        origin and box, reproduces that configuration's member of this
        potential's ensemble. A non-periodic potential whose integrator is
        real-space in-plane (quadrature) first wraps in-plane the given atoms
        that the padding does not keep with all their images.

        The exception is a non-periodic potential with a finite projection.
        It also integrates the atoms within its cutoff outside the box, each
        displaced independently, which are not part of a configuration; a
        potential built from a configuration uses images of the configuration's
        own atoms there instead.

        Iterating the frozen phonons gives configurations of the untransformed
        atoms instead, which are different ones whenever the potential
        transforms the cell.

        Returns
        -------
        atoms_ensemble : AtomsEnsemble or EnergyResolvedAtomsEnsemble
            An :class:`~abtem.inelastic.phonons.EnergyResolvedAtomsEnsemble` for
            frozen phonons of that type, with its energies and axes, otherwise an
            :class:`~abtem.inelastic.phonons.AtomsEnsemble` with the axes of the
            frozen phonons.

        Raises
        ------
        NotImplementedError
            For frozen phonons with more than one ensemble axis, other than an
            :class:`~abtem.inelastic.phonons.EnergyResolvedAtomsEnsemble`.
        """
        frozen_phonons = self.frozen_phonons

        def configuration(potential):
            atoms = potential._configuration()[0]
            atoms.arrays.pop(SOURCE_INDEX, None)
            return atoms

        if not self.ensemble_shape:
            return AtomsEnsemble([configuration(self)])

        if len(self.ensemble_shape) > 1 and not isinstance(
            frozen_phonons, EnergyResolvedAtomsEnsemble
        ):
            raise NotImplementedError(
                f"The frozen phonons have {len(self.ensemble_shape)} ensemble "
                "axes; configurations can be returned for one axis, or for an "
                "EnergyResolvedAtomsEnsemble."
            )

        configurations = np.empty(self.ensemble_shape, dtype=object)
        for index, _, wrapped in self.generate_blocks(1):
            itemset(configurations, index, configuration(wrapped.item()))

        axes_metadata = [copy.deepcopy(axis) for axis in self.ensemble_axes_metadata]

        if isinstance(frozen_phonons, EnergyResolvedAtomsEnsemble):
            return EnergyResolvedAtomsEnsemble(
                configurations,
                energies=frozen_phonons.energies,
                ensemble_mean=frozen_phonons.ensemble_mean,
                ensemble_axes_metadata=axes_metadata,
            )

        return AtomsEnsemble(
            list(configurations),
            ensemble_mean=frozen_phonons.ensemble_mean,
            ensemble_axes_metadata=axes_metadata,
        )

    def get_sliced_atoms(self) -> BaseSlicedAtoms:
        """
        The atoms grouped into the slices given by the slice thicknesses.

        Returns
        -------
        sliced_atoms : BaseSlicedAtoms
        """
        if self._sliced_atoms is not None:
            return self._sliced_atoms

        self._sliced_atoms = self._prepare_atoms()

        return self._sliced_atoms

    def generate_slices(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        return_depth: float = False,
    ):
        """
        Generate the slices for the potential.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the generated potential.
        last_slice : int, optional
            Index of the last slice of the generated potential.
        return_depth : bool
            If True, return the depth of each generated slice.

        Yields
        ------
        slices : generator of numpy.ndarray
            Generator for the array of slices.
        """
        if last_slice is None:
            last_slice = len(self)

        xp = get_array_module(self.device)

        sliced_atoms = self.get_sliced_atoms()

        numbers = np.unique(sliced_atoms.atoms.numbers)

        exit_plane_after = self._exit_plane_after

        cumulative_thickness = np.cumsum(self.slice_thickness)

        for start, stop in generate_chunks(
            last_slice - first_slice, chunks=1, start=first_slice
        ):
            if len(numbers) > 1 or stop - start > 1:
                array = xp.zeros(
                    (stop - start,) + self.base_shape[1:],
                    dtype=get_dtype(complex=False),
                )
            else:
                array = None

            for i, slice_idx in enumerate(range(start, stop)):
                atoms = sliced_atoms.get_atoms_in_slices(slice_idx)

                new_array = self._integrator.integrate_on_grid(
                    atoms,
                    a=sliced_atoms.slice_limits[slice_idx][0],
                    b=sliced_atoms.slice_limits[slice_idx][1],
                    gpts=self.gpts,
                    sampling=self.sampling,
                    device=self.device,
                    cell=self.cell,
                )

                if array is not None:
                    array[i] += new_array
                else:
                    array = new_array[None]

            if array is None:
                array = xp.zeros(
                    (stop - start,) + self.base_shape[1:],
                    dtype=get_dtype(complex=False),
                )

            # array -= array.min()

            exit_planes = tuple(np.where(exit_plane_after[start:stop])[0])

            potential_array = self._array_object(
                array,
                slice_thickness=self.slice_thickness[start:stop],
                exit_planes=exit_planes,
                extent=self.extent,
                cell=self.grid.cell,
            )

            if return_depth:
                depth = cumulative_thickness[stop - 1]
                yield depth, potential_array
            else:
                yield potential_array

    @property
    def ensemble_axes_metadata(self):
        return self.frozen_phonons.ensemble_axes_metadata

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return self.frozen_phonons.ensemble_shape

    @classmethod
    def _from_partitioned_args_func(
        cls, *args, frozen_phonons_partial, check_given_atoms=False, **kwargs
    ):
        args = unpack_blockwise_args(args)

        frozen_phonons = frozen_phonons_partial(*args)
        frozen_phonons = frozen_phonons.item()

        # The box was reported when the potential was constructed.
        with _box_strain_warning_silenced():
            new_potential = cls(frozen_phonons, **kwargs)

        # The members of a lazily read trajectory, which the construction could
        # not check, are known now: once per block, not per slice.
        if check_given_atoms:
            _warn_if_given_atoms_far_outside_along_z(
                _given_configurations(frozen_phonons), new_potential.plane
            )

        ndims = max(len(new_potential.ensemble_shape), 1)
        new_potential = _wrap_with_array(new_potential, ndims)
        return new_potential

    def _from_partitioned_args(self, *args, **kwargs):
        frozen_phonons_partial = self.frozen_phonons._from_partitioned_args()
        # integrator is excluded from the deep-copied kwargs and passed through
        # by reference: generate_blocks()/ensemble_blocks() reconstruct a fresh
        # Potential per ensemble member, and integrators (e.g.
        # QuadratureProjectionIntegrals) lazily cache per-symbol projection
        # tables and, on GPU, device-resident arrays specifically so repeated
        # calls across many slices don't re-upload them (see the PR #309
        # discussion in integrals.py). Deep-copying the integrator per member
        # silently defeats that cache for every single ensemble member -- on
        # GPU this means a real device-to-device copy of the cached arrays for
        # every member, for no benefit, since the copy is used once and
        # discarded.
        kwargs = self._copy_kwargs(exclude=("atoms", "sampling", "integrator"))
        kwargs["integrator"] = self.integrator

        return partial(
            self._from_partitioned_args_func,
            frozen_phonons_partial=frozen_phonons_partial,
            check_given_atoms=isinstance(
                getattr(self.frozen_phonons, "trajectory", None), da.Array
            )
            and self._reports_given_atoms(),
            **kwargs,
        )

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True):
        if chunks is None:
            chunks = (1,)

        return self.frozen_phonons._partition_args(chunks, lazy=lazy)


class _PotentialBuilder(_FieldBuilder, BasePotential):
    pass


def _validate_frozen_phonons(atoms):
    if isinstance(atoms, Atoms):
        atoms = atoms.copy()
        atoms.calc = None

    if not hasattr(atoms, "randomize"):
        if isinstance(atoms, (list, tuple)):
            frozen_phonons = AtomsEnsemble(atoms)
        elif isinstance(atoms, Atoms):
            frozen_phonons = DummyFrozenPhonons(atoms)
        else:
            raise ValueError(
                "Frozen phonons should be of types `FrozenPhonons`, `Atoms` or"
                f"`AtomsEnsemble`, not {atoms}"
            )
    else:
        frozen_phonons = atoms

    return frozen_phonons


class Potential(_FieldBuilderFromAtoms, BasePotential):
    """
    Calculate the electrostatic potential of a set of atoms or frozen phonon
    configurations. The potential is calculated with the Independent Atom Model (IAM)
    using a user-defined parametrization of the atomic potentials.

    Parameters
    ----------
    atoms : ase.Atoms or abtem.FrozenPhonons
        Atoms or FrozenPhonons defining the atomic configuration(s) used in the
        independent atom model for calculating the electrostatic potential(s).
    gpts : one or two int, optional
        Number of grid points in `x` and `y` describing each slice of the potential.
        Provide either "sampling" (spacing between consecutive grid points) or "gpts"
        (total number of grid points).
    sampling : one or two float or 'auto', optional
        Sampling of the potential in `x` and `y` [Å].
        Provide either "sampling" or "gpts". If 'auto', the grid points are chosen
        to be commensurate with the atom positions (closest to a default of 0.05 Å)
        and, whenever compatible with commensurability, a fast FFT size (all prime
        factors in {2, 3, 5, 7}); the commensurate grid nearest the target is kept
        when it is already such a size. Set the configuration option
        'grid.round-to-fast-fft' to False for the plain commensurate grid. The
        commensurability search follows the lattice vectors, so it is correct for
        a non-orthogonal (skewed) cell too.
        For an `AtomsEnsemble` with more than one configuration (e.g. an MD
        trajectory), each configuration is an independent, generally
        non-commensurate snapshot, so commensurability is not attempted and the
        target sampling is used directly (rounded up to a fast FFT size).
    slice_thickness : float or sequence of float or 'auto', optional
        Thickness of the potential slices in the propagation direction in [Å]
        (default is 1 Å).
        If given as a float, the number of slices is calculated by dividing the slice
        thickness into the `z`-height of supercell. The slice thickness may be given as
        a sequence of values for each slice, in which case an error will be thrown if
        the sum of slice thicknesses is not equal to the height of the atoms.
        If 'auto', slice boundaries are aligned with the crystal planes, with slices
        merged to stay close to a default of 1.0 Å. As with `sampling`, this
        commensurability search is skipped for an `AtomsEnsemble` with more than
        one configuration, which uses a uniform 1.0 Å target thickness instead.
    parametrization : 'lobato' or 'kirkland', optional
        The potential parametrization describes the radial dependence of the potential
        for each element. Two of the most accurate parametrizations are available
        (by Lobato et al. and Kirkland; default is 'lobato').
        See the citation guide for references.
    projection : 'finite' or 'infinite', optional
        If 'finite' the 3D potential is numerically integrated between the slice
        boundaries. If 'infinite' (default), the infinite potential projection of each
        atom will be assigned to a single slice.

        On GPU, ``projection='finite'`` is not bit-reproducible: its radial
        interpolation kernel accumulates overlapping atoms' contributions with
        an atomic add, whose order (and therefore float32 rounding) varies
        from run to run (deviation ~1e-7 relative). ``projection='infinite'``
        has no such accumulation and is deterministic on both CPU and GPU, as
        is ``'finite'`` on CPU. Compare finite-projection GPU results with a
        relative tolerance of at least 1e-6 rather than exact equality, and
        build on CPU (or use ``'infinite'``) where bit-for-bit reproducibility
        matters.
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the
        slice indices after which an exit plane is desired, and hence during a
        multislice simulation a measurement is created. If `exit_planes` is an integer
        `n`, a measurement is collected every `n` slices and after the last slice. The
        first measurement is then taken at the entrance surface (zero thickness),
        before any scattering, so the thickness series has ``1 + ceil(num_slices / n)``
        planes; index it with ``[-1]`` for the exit surface. If `n` exceeds the number
        of slices, only the exit surface is returned and no thickness axis is added.
    plane : str or two tuples of three float, optional
        The plane relative to the provided atoms mapped to `xy` plane of the potential,
        i.e. provided plane is perpendicular to the propagation direction. If string,
        it must be a concatenation of two of 'x', 'y' and 'z'; the default value 'xy'
        indicates that potential slices are cuts along the `xy`-plane of the atoms.
        The atoms are rotated to the plane, never mirrored, so for 'xz', 'yx' and
        'zy' the second axis is reversed (see
        `abtem.atoms.rotate_atoms_to_plane`). The plane may also be specified with two
        arbitrary 3D vectors, which are mapped to the `x` and `y` directions of the
        potential, respectively. The length of the vectors has no influence. If the
        vectors are not perpendicular, the second vector is rotated in the plane to
        become perpendicular to the first. Providing a value of
        ((1., 0., 0.), (0., 1., 0.)) is equivalent to providing 'xy'.
    origin : three float, optional
        The origin relative to the provided atoms mapped to the origin of the potential.
        This is equivalent to translating the atoms. The default is (0., 0., 0.).
    box : three float, optional
        The extent of the potential in `x`, `y` and `z`. If not given this is determined
        from the atoms' cell. If the box size does not match an integer number of the
        atoms' supercell, an affine transformation may be necessary to preserve
        periodicity, determined by the `periodic` keyword. A periodic box must hold at
        least one period of the atoms' cell along each direction (about half the cell
        or more), and is otherwise rejected with a `ValueError`; the atoms are strained
        to fit the box, however large the strain, and a `UserWarning` quotes the stretch
        of each axis and the shear when either exceeds 0.1 %. The same warning is given
        for the box chosen for a non-orthogonal cell when none is given (a box given
        and equal to it is not checked). The box is filled with repetitions of the
        atoms' cell whatever their `pbc`; it never adds vacuum.
    periodic : bool, True
        If a transformation of the atomic structure is required, `periodic` determines
        how the atomic structure is transformed. If True, the periodicity of the Atoms
        is preserved, which may require applying a small affine transformation to the
        atoms. If False, the transformed potential is effectively cut out of a larger
        repeated potential, which may not preserve periodicity. Atoms given outside
        the cell are then images of that repeated structure, folded into the cell,
        except along `z` within 2 Å of a face: such an atom, as frozen phonons
        displace them, is kept where it is, and the infinite projection puts it in
        the face slice. The slice of an atom given outside along `z` therefore jumps
        at 2 Å from the face slice to that of its image; the finite integrators
        spread its potential over the atom and its images at any distance. Atoms
        given more than 2 Å outside along `z` are reported when the potential is
        made.
    integrator : ProjectionIntegrator, optional
        Provide a custom integrator for the projection integrals of the potential
        slicing.
    device : str, optional
        The device used for calculating the potential, 'cpu' or 'gpu'. The default is
        determined by the user configuration file.
    non_orthogonal : bool, optional
        Whether to build the potential on a non-orthogonal (skewed) in-plane grid rather
        than orthogonalising the cell into a supercell. The `z`-axis must be
        perpendicular to the `xy`-plane (monoclinic-along-beam). If ``None`` (default)
        this is detected automatically from the cell: a non-orthogonal but z-separable
        cell (with ``plane='xy'`` and no explicit ``box``) builds a skewed grid, while an
        orthogonal cell is unaffected. Pass ``False`` to force the legacy orthogonalising
        behaviour, or ``True`` to require a skewed grid.
    """

    _exclude_from_copy = ("parametrization", "projection")

    def __init__(
        self,
        atoms: Atoms | BaseFrozenPhonons,
        gpts: int | tuple[int, int] | None = None,
        sampling: float | tuple[float, float] | str | None = None,
        slice_thickness: float | tuple[float, ...] | str = 1,
        parametrization: str | Parametrization = "lobato",
        projection: str = "infinite",
        exit_planes: int | tuple[int, ...] | None = None,
        plane: (
            str | tuple[tuple[float, float, float], tuple[float, float, float]]
        ) = "xy",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: tuple[float, float, float] | None = None,
        periodic: bool = True,
        integrator: FieldIntegrator | None = None,
        device: str | None = None,
        non_orthogonal: bool | None = None,
    ):
        origin = _validate_origin(origin)
        if box is not None:
            box = _validate_box(box)

        frozen_phonons = _validate_frozen_phonons(atoms)
        atoms_obj = frozen_phonons.atoms
        # A multi-configuration `AtomsEnsemble` (e.g. an MD trajectory) has no
        # shared reference lattice: each configuration is an independent,
        # generally non-commensurate snapshot, and `atoms_obj` here is only the
        # first frame. A single-configuration `AtomsEnsemble` has no such
        # ambiguity -- that one frame *is* the configuration, just as it would be
        # if passed as plain `Atoms` -- so only skip commensurability search when
        # there is genuinely more than one configuration to be ambiguous about.
        has_multiple_configs = (
            isinstance(frozen_phonons, AtomsEnsemble) and frozen_phonons.num_configs > 1
        )

        def _placed_atoms(placed_box):
            # The atoms as the potential places them in its box: strained into a
            # periodic box, or cut out of the repeated structure for a
            # non-periodic potential.
            if periodic:
                return orthogonalize_cell(
                    atoms_obj,
                    box=placed_box,
                    plane=plane,
                    origin=origin,
                    return_transform=False,
                    allow_transform=True,
                )
            return cut_cell(atoms_obj, cell=placed_box, plane=plane, origin=origin)

        # Decide skew-native vs. legacy orthogonalising using the same
        # criteria as _FieldBuilder.__init__'s own `use_skew` detection,
        # so the grid this chooses is commensurate with whichever
        # geometry the potential actually ends up built on. Deciding
        # this here (rather than always calling _require_cell_transform,
        # which predates skew-native support and knows nothing about
        # `non_orthogonal`) is what makes 'auto' sampling correct for a
        # skewed cell: without it, the target grid would be computed for
        # an orthogonalised auxiliary supercell and then silently reused
        # as the gpts of an entirely different (skewed) grid.
        # the cell of the ensemble, as the builder reads it: the atoms of a lazily
        # read trajectory are not known until its blocks are computed
        cell = np.array(frozen_phonons.cell, dtype=float)
        if non_orthogonal is None:
            use_skew = (
                not is_cell_orthogonal(cell)
                and is_cell_ab_in_plane(cell)
                and isinstance(plane, str)
                and plane == "xy"
                and box is None
            )
        else:
            use_skew = bool(non_orthogonal)

        if sampling == "auto":
            if gpts is not None:
                raise ValueError("Cannot specify both gpts and sampling='auto'")
            if not atoms_obj.pbc[:2].all() or has_multiple_configs:
                # Non-periodic in xy (e.g. a nanoparticle): atom positions are not
                # translationally repeated, so commensurability has no meaning and
                # the period-search algorithm may produce spurious results for
                # arbitrary rotations. A multi-configuration `AtomsEnsemble` has
                # the same problem: the chosen grid is applied to every frame
                # regardless, and searching for commensurate planes in one
                # arbitrary frame is meaningless. Just use the target sampling
                # directly, rounded up to a fast FFT size (no commensurability
                # constrains the grid here, so rounding is free).
                from abtem.core.fft import next_fast_fft_size

                # The extent the builder gives the potential: the box, else the
                # cell rotated to the plane and made orthogonal when it needs a
                # transform, else the cell itself.
                if box is not None:
                    extent = box[:2]
                elif use_skew:
                    extent = tuple(np.linalg.norm(cell[:2, :2], axis=1))
                elif _require_cell_transform(
                    cell, box=None, plane=plane, origin=origin
                ):
                    extent = _default_box(cell, plane)[:2]
                else:
                    extent = (float(cell[0, 0]), float(cell[1, 1]))
                gpts = tuple(safe_ceiling_int(extent[i] / 0.05) for i in range(2))
                if round_auto_derived_gpts():
                    gpts = tuple(next_fast_fft_size(n) for n in gpts)
            else:
                if use_skew:
                    cell_2d = cell[:2, :2]
                    extent = tuple(np.linalg.norm(cell_2d, axis=1))
                    gpts = commensurate_gpts(
                        extent,
                        atoms_obj.positions,
                        target_sampling=0.05,
                        round_to_fast_fft=round_auto_derived_gpts(),
                        cell=cell_2d,
                    )
                elif _require_cell_transform(cell, box=box, plane=plane, origin=origin):
                    if not isinstance(plane, str):
                        raise NotImplementedError
                    if box is not None:
                        auto_box = box
                    else:
                        auto_box = _default_box(cell, plane)
                    extent = auto_box[:2]
                    # Place the atoms in the box so positions match the extent
                    _auto_atoms = _placed_atoms(auto_box)
                    gpts = commensurate_gpts(
                        extent,
                        _auto_atoms.positions,
                        target_sampling=0.05,
                        round_to_fast_fft=round_auto_derived_gpts(),
                    )
                else:
                    if box is not None:
                        extent = box[:2]
                        _auto_atoms = atoms_obj
                    else:
                        extent = tuple(np.linalg.norm(cell[:2, :2], axis=1))
                        _auto_atoms = atoms_obj
                    gpts = commensurate_gpts(
                        extent,
                        _auto_atoms.positions,
                        target_sampling=0.05,
                        round_to_fast_fft=round_auto_derived_gpts(),
                    )
            sampling = None

        if slice_thickness == "auto":
            if atoms_obj.pbc[2] and not has_multiple_configs:
                # Periodic in z: align slice boundaries with the crystal planes
                # of the atoms as they are placed in the potential's box.
                plane_atoms = atoms_obj
                # A skew-native potential keeps the cell, so its atoms are not
                # placed in an orthogonal box.
                if not use_skew and _require_cell_transform(
                    atoms_obj.cell, box=box, plane=plane, origin=origin
                ):
                    if box is not None:
                        st_box = box
                    else:
                        st_box = _default_box(atoms_obj.cell, plane)
                    plane_atoms = _placed_atoms(st_box)
                slice_thickness = commensurate_slice_thickness(
                    plane_atoms, target_thickness=1.0
                )
            else:
                # Non-periodic in z (e.g. a nanoparticle or slab in vacuum), or a
                # multi-configuration `AtomsEnsemble` (independent snapshots with
                # no shared commensurate lattice, see the sampling branch above):
                # crystal-plane commensurability is not applicable; use a uniform
                # target thickness and let _validate_slice_thickness divide evenly.
                slice_thickness = 1.0

        if integrator is None:
            if projection == "finite":
                integrator = QuadratureProjectionIntegrals(
                    parametrization=parametrization
                )
            elif projection == "infinite":
                integrator = ScatteringFactorProjectionIntegrals(
                    parametrization=parametrization
                )
            else:
                raise NotImplementedError

        super().__init__(
            atoms=atoms,
            array_object=PotentialArray,
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            origin=origin,
            box=box,
            periodic=periodic,
            integrator=integrator,
            non_orthogonal=non_orthogonal,
        )


def _validate_3d_position(
    position: tuple[float, float] | tuple[float, float, float] | Atom,
) -> tuple[float, float, float]:
    if isinstance(position, Atom):
        return (position.x, position.y, position.z)

    position = tuple(float(c) for c in position)

    if len(position) == 3:
        return position
    elif len(position) == 2:
        return position + (0.0,)
    else:
        raise ValueError(
            "position must be given as two or three floats, or an Atom, got "
            f"{position}"
        )


class FieldArray(BaseField, ArrayObject):
    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        slice_thickness: float | Sequence[float],
        extent: Optional[float | tuple[float, float]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        metadata: Optional[dict] = None,
        cell: Optional[np.ndarray] = None,
    ):
        # assert len(array.shape) == self._base_dims

        self._slice_thickness = _validate_slice_thickness(
            slice_thickness, num_slices=array.shape[-self._base_dims]
        )

        self._exit_planes = _validate_exit_planes(
            exit_planes, len(self._slice_thickness)
        )
        self._grid = Grid(
            extent=extent,
            gpts=array.shape[-2:],
            sampling=sampling,
            lock_gpts=True,
            cell=cell,
        )

        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    def metadata(self) -> dict:
        cell = self.grid.cell
        if cell is not None:
            # the grid cell takes precedence so a non-orthogonal cell stays consistent
            # with the grid (and overrides any stale cell carried in the metadata)
            self._metadata["cell"] = tuple(map(tuple, cell.tolist()))
        else:
            self._metadata.pop("cell", None)
        return self._metadata

    @property
    def num_configurations(self):
        indices = _find_axes_type(self, FrozenPhononsAxis)
        if indices:
            return reduce(mul, tuple(self.array.shape[i] for i in indices))
        else:
            return 1

    @property
    def slice_thickness(self) -> tuple[float, ...]:
        return self._slice_thickness

    @property
    def exit_planes(self) -> tuple[int, ...]:
        return self._exit_planes

    @property
    def _slice_kwargs(self) -> dict:
        """The constructor arguments, besides the array and thicknesses, that a
        slice or chunk of this array shares with it."""
        return {"extent": self.extent, "metadata": self.metadata}

    def _warn_if_ensemble(self):
        if np.prod(self.ensemble_shape) > 1:
            warnings.warn(
                f"{type(self).__name__} has an ensemble axis of shape "
                f"{self.ensemble_shape}; only the slices of its first member are "
                "generated. Index the array to choose a member.",
                stacklevel=_stacklevel_outside_package(),
            )

    def build(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        chunks: int = 1,
        lazy: Optional[bool] = None,
    ):
        raise RuntimeError("potential is already built")

    def generate_slices(self, first_slice: int = 0, last_slice: Optional[int] = None):
        """
        Generate the slices for the potential.

        An array with an ensemble axis (frozen phonons) of more than one member
        generates the slices of its first member, with a warning.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the generated potential.
        last_slice : int, optional
            Index of the last slice of the generated potential.

        Yields
        ------
        FieldArray
            A single slice, of the same class as this array.
        """
        self._warn_if_ensemble()

        if last_slice is None:
            last_slice = len(self)

        exit_plane_after = self._exit_plane_after
        # cum_thickness = np.cumsum(self.slice_thickness)
        start = first_slice
        stop = first_slice + 1

        for i in range(first_slice, last_slice):
            s = (0,) * (len(self.array.shape) - self._base_dims) + (i,)
            array = self.array[s][None]

            slic = self.__class__(
                array, self.slice_thickness[i : i + 1], **self._slice_kwargs
            )

            exit_planes = tuple(np.where(exit_plane_after[start:stop])[0])

            slic._exit_planes = exit_planes

            start += 1
            stop += 1

            yield slic

    def generate_chunked_slices(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        chunk_size: int | str = "auto",
    ):
        """
        Generate potential slices in memory-budgeted chunks.

        For a pre-built ``PotentialArray`` the data is already in memory
        (or backed by a dask array whose single chunk spans all slices).
        This method yields views into the existing array without any new
        allocation or copy, so chunking only controls iteration grouping.

        Note: if the array is dask-backed, the full potential is still
        materialized as a single chunk when computed (dask never chunks
        along the slice axis). To benefit from true memory-bounded
        slice chunking, pass an unbuilt :class:`.Potential` to the
        multislice algorithm instead.

        An array with an ensemble axis (frozen phonons) of more than one member
        generates the chunks of its first member, with a warning.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice.
        last_slice : int, optional
            Index of the last slice.
        chunk_size : int or str, optional
            Number of slices per chunk. ``"auto"`` selects based on the
            configured memory budget.

        Yields
        ------
        FieldArray
            A view into the existing array covering a chunk of slices, of the
            same class as this array.
        """
        self._warn_if_ensemble()

        from abtem.core.chunks import (
            estimate_potential_chunk_size,
            generate_chunks,
        )

        if last_slice is None:
            last_slice = len(self)

        if chunk_size == "auto":
            chunk_size = estimate_potential_chunk_size(
                self.base_shape[1:], self.device
            )

        # Cap so the whole range is one chunk when it fits in the budget,
        # then distribute evenly so the last chunk is never smaller than
        # necessary (equal_sized_chunks inside generate_chunks handles this).
        chunk_size = min(chunk_size, last_slice - first_slice)

        exit_plane_after = self._exit_plane_after

        for chunk_start, chunk_end in generate_chunks(
            last_slice - first_slice, chunks=chunk_size, start=first_slice
        ):
            s = (0,) * (len(self.array.shape) - self._base_dims) + (
                slice(chunk_start, chunk_end),
            )
            chunk_array = self.array[s]

            exit_planes = tuple(
                np.where(exit_plane_after[chunk_start:chunk_end])[0]
            )

            chunk = self.__class__(
                chunk_array,
                slice_thickness=self.slice_thickness[chunk_start:chunk_end],
                **self._slice_kwargs,
            )
            chunk._exit_planes = exit_planes
            yield chunk

    def __getitem__(self, items):
        if isinstance(items, (Number, slice)):
            items = (items,)

        if not len(items) <= len(self.ensemble_shape) + 1:
            raise IndexError(
                f"Too many indices for potential array with {len(self.ensemble_shape)}"
                "ensemble axes. Only slice indices and ensemble indices are allowed."
            )

        ensemble_items = items[: len(self.ensemble_shape)]
        slic_items = items[len(self.ensemble_shape) :]

        if len(ensemble_items):
            potential_array = super().__getitem__(ensemble_items)
        else:
            potential_array = self

        if len(slic_items) == 0:
            return potential_array

        padded_items = (slice(None),) * len(potential_array.ensemble_shape) + slic_items

        array = potential_array._array[padded_items]

        slice_thickness = np.array(potential_array.slice_thickness)[slic_items]

        if len(array.shape) < len(potential_array.shape):
            array = array[
                (slice(None),) * len(potential_array.ensemble_shape) + (None,)
            ]
            slice_thickness = slice_thickness[None]

        kwargs = potential_array._copy_kwargs(exclude=("array", "slice_thickness"))
        kwargs["array"] = array
        kwargs["slice_thickness"] = slice_thickness
        kwargs["sampling"] = None

        # the exit planes index the slices, hence they have to be mapped into the
        # sliced potential; those falling outside it are dropped, and if none
        # remain the exit plane defaults to the last slice of the new potential
        selected = np.atleast_1d(
            np.arange(potential_array.num_slices)[slic_items[0]]
        )
        exit_planes = tuple(
            int(np.flatnonzero(selected == plane)[0])
            for plane in potential_array.exit_planes
            if plane in selected
        )
        kwargs["exit_planes"] = exit_planes if exit_planes else None

        return potential_array.__class__(**kwargs)

    def tile(self, repetitions: tuple[int, int] | tuple[int, int, int]):
        """
        Tile the potential.

        Parameters
        ----------
        repetitions: two or three int
            The number of repetitions of the potential along each axis. NOTE: if three
            integers are given, the last represents the number of repetitions along the
            `z`-axis.

        Returns
        -------
        PotentialArray object
            The tiled potential.
        """
        if len(repetitions) == 2:
            repetitions = (repetitions[0], repetitions[1], 1)

        assert len(repetitions) == 3

        tile_reps = [1] * len(self.array.shape)
        tile_reps[-self._base_dims] = repetitions[2]
        tile_reps[-2] = repetitions[0]
        tile_reps[-1] = repetitions[1]

        new_array = np.tile(self.array, tuple(tile_reps))

        if self.extent is not None:
            new_extent = (
                self.extent[0] * repetitions[0],
                self.extent[1] * repetitions[1],
            )
        else:
            new_extent = None

        if self.grid.cell is not None:
            new_cell = self.grid.cell.copy()
            new_cell[0] *= repetitions[0]
            new_cell[1] *= repetitions[1]
        else:
            new_cell = None

        new_slice_thickness = tuple(np.tile(self.slice_thickness, repetitions[2]))

        return self.__class__(
            array=new_array,
            slice_thickness=new_slice_thickness,
            extent=new_extent,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            cell=new_cell,
        )

    def to_hyperspy(self, transpose: bool = True):
        return self.to_images().to_hyperspy(transpose=transpose)

    def to_images(self):
        """Convert slices of the potential to a stack of images."""
        return Images(
            array=self._array,
            sampling=(self.sampling[0], self.sampling[1]),
            metadata=self.metadata,
            ensemble_axes_metadata=self.axes_metadata[:-2],
        )

    def depth_profile(
        self,
        projection_axis: str = "y",
        depth: Optional[float] = None,
    ) -> Images:
        """Create a depth profile by projecting the potential along a spatial axis.

        Parameters
        ----------
        projection_axis : str
            Spatial axis to project (sum) along. ``"y"`` (default) produces an
            x–z cross-section; ``"x"`` produces a y–z cross-section.
        depth : float, optional
            If given, project only over a finite slab of this thickness [Å],
            centered on the midpoint of the projected axis. The number of grid
            points is rounded to the nearest integer. If ``None``, the full
            extent is projected.

        Returns
        -------
        depth_profile : Images
            2D image(s) with the remaining spatial axis horizontal and depth
            (z) vertical.
        """
        from copy import copy

        if projection_axis not in ("x", "y"):
            raise ValueError("projection_axis must be 'x' or 'y'.")

        array = self.array

        if projection_axis == "y":
            sum_axis = -1
            spatial_sampling = self.sampling[0]
        else:
            sum_axis = -2
            spatial_sampling = self.sampling[1]

        if depth is not None:
            proj_sampling = (
                self.sampling[1] if projection_axis == "y" else self.sampling[0]
            )
            proj_gpts = self.gpts[1] if projection_axis == "y" else self.gpts[0]
            n = max(1, min(proj_gpts, round(depth / proj_sampling)))
            start = (proj_gpts - n) // 2
            slices = [slice(None)] * len(array.shape)
            slices[sum_axis] = slice(start, start + n)
            array = array[tuple(slices)]

        array = array.sum(axis=sum_axis)

        xp = get_array_module(array)
        if hasattr(array, "rechunk"):
            array = da.moveaxis(array, -2, -1)
        else:
            array = xp.moveaxis(array, -2, -1)

        n_z = self.num_slices
        z_extent = self.thickness
        z_sampling = z_extent / n_z if n_z > 0 else 1.0

        metadata = copy(self.metadata)

        return Images(
            array,
            sampling=(spatial_sampling, z_sampling),
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=metadata,
        )

    def project(self) -> Images:
        """
        Create a 2D array representing a projected image of the potential(s).

        Returns
        -------
        images : Images
            One or more images of the projected potential(s).
        """
        array = self.array.sum(-self._base_dims)
        # array -= array.min((-2, -1), keepdims=True)

        ensemble_axes_metadata = (
            self.ensemble_axes_metadata + self.base_axes_metadata[1:-2]
        )

        return Images(
            array=array,
            sampling=self._valid_sampling,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=self.metadata,
        )

    def interpolate_line(
        self,
        start: tuple[float, float] | tuple[float, float, float] | Atom | None = None,
        end: tuple[float, float] | tuple[float, float, float] | Atom | None = None,
        sampling: Optional[float] = None,
        gpts: Optional[int] = None,
        width: float = 0.0,
        margin: float = 0.0,
        order: int = 3,
        endpoint: bool = False,
        fractional: bool = False,
        projected: bool = True,
    ) -> RealSpaceLineProfiles:
        """See :meth:`.BaseField.interpolate_line`."""
        if projected:
            return self.project().interpolate_line(
                start=start,
                end=end,
                sampling=sampling,
                gpts=gpts,
                width=width,
                margin=margin,
                order=order,
                endpoint=endpoint,
                fractional=fractional,
            )

        if width:
            raise NotImplementedError(
                "width > 0 is only supported for projected=True."
            )

        if margin:
            raise NotImplementedError("margin is only supported for projected=True.")

        if start is None:
            start = (0.0, 0.0, 0.0)
        start = _validate_3d_position(start)

        if end is None:
            end = (0.0, 1.0, 0.0) if fractional else (0.0, self.extent[0], 0.0)
        end = _validate_3d_position(end)

        # The in-plane grid may be non-orthogonal (skewed); `effective_cell` is the
        # 2x2 matrix whose rows are the two in-plane lattice vectors (falling back to
        # a diagonal cell built from `extent` for an orthogonal grid -- the same
        # fallback `Grid._effective_cell()` uses for reciprocal_metric/k_components).
        # It is used both ways: fractional (u, v) -> Cartesian (x, y) via `@ cell`, and
        # Cartesian (x, y) -> fractional pixel index via `@ inv(cell)`.
        effective_cell = self.grid._effective_cell()

        if fractional:
            start_xy = np.array(start[:2]) @ effective_cell
            end_xy = np.array(end[:2]) @ effective_cell
            start = (float(start_xy[0]), float(start_xy[1]), start[2] * self.thickness)
            end = (float(end_xy[0]), float(end_xy[1]), end[2] * self.thickness)

        length = float(np.linalg.norm(np.array(end) - np.array(start)))

        if (sampling is None) and (gpts is None):
            sampling = min(min(self.sampling), min(self.slice_thickness))

        if gpts is None:
            gpts = max(int(np.ceil(length / sampling)), 1)

        if endpoint and gpts > 1:
            line_sampling = length / (gpts - 1)
        elif gpts > 0:
            line_sampling = length / gpts
        else:
            line_sampling = 0.0

        dtype = get_dtype(complex=False)
        x = np.linspace(start[0], end[0], gpts, endpoint=endpoint, dtype=dtype)
        y = np.linspace(start[1], end[1], gpts, endpoint=endpoint, dtype=dtype)
        z = np.linspace(start[2], end[2], gpts, endpoint=endpoint, dtype=dtype)

        slice_thickness = np.array(self.slice_thickness)
        z_centers = np.cumsum(slice_thickness) - slice_thickness / 2
        z_px = np.interp(z, z_centers, np.arange(len(z_centers)))

        inv_cell = np.linalg.inv(effective_cell)
        frac_xy = np.stack((x, y), axis=1) @ inv_cell
        x_px = frac_xy[:, 0] * self.gpts[0]
        y_px = frac_xy[:, 1] * self.gpts[1]

        xp = get_array_module(self.array)
        positions = xp.asarray(np.stack((z_px, x_px, y_px), axis=1))

        if self.is_lazy:
            n_base = len(self.base_shape)
            base_axes = tuple(range(self.array.ndim - n_base, self.array.ndim))
            chunks = self.array.chunks[:-n_base] + (gpts,)
            new_axis = (base_axes[0],)

            array = da.map_blocks(
                _interpolate_stack,
                self.array,
                positions=positions,
                mode="wrap",
                order=order,
                drop_axis=base_axes,
                new_axis=new_axis,
                chunks=chunks,
                meta=xp.array((), dtype=dtype),
            )
        else:
            array = _interpolate_stack(
                self.array, positions, mode="wrap", order=order
            )

        from copy import copy

        metadata = copy(self.metadata)
        metadata["start"] = start
        metadata["end"] = end

        return RealSpaceLineProfiles(
            array=array,
            sampling=line_sampling,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=metadata,
        )

    def interpolate_line_at_position(
        self,
        center: tuple[float, float] | tuple[float, float, float] | Atom,
        angle: float = 0.0,
        extent: float = 1.0,
        gpts: Optional[int] = None,
        sampling: Optional[float] = None,
        width: float = 0.0,
        order: int = 3,
        endpoint: bool = True,
        projected: bool = True,
    ) -> RealSpaceLineProfiles:
        """See :meth:`.BaseField.interpolate_line_at_position`."""
        if projected:
            if not isinstance(center, Atom) and len(center) == 3:
                raise ValueError(
                    "center has a depth (z) coordinate, which is only meaningful "
                    "for projected=False; pass a two-float (x, y) center, or set "
                    "projected=False."
                )

            return self.project().interpolate_line_at_position(
                center=center,
                angle=angle,
                extent=extent,
                gpts=gpts,
                sampling=sampling,
                width=width,
                order=order,
                endpoint=endpoint,
            )

        center = _validate_3d_position(center)

        direction = np.array(
            (np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.0)
        )
        start = tuple(np.array(center) - extent / 2 * direction)
        end = tuple(np.array(center) + extent / 2 * direction)

        return self.interpolate_line(
            start,
            end,
            gpts=gpts,
            sampling=sampling,
            width=width,
            order=order,
            endpoint=endpoint,
            projected=False,
        )


class PotentialArray(BasePotential, FieldArray):
    """
    The potential array represents slices of the electrostatic potential as an array.
    All other potentials build potential arrays.

    Parameters
    ----------
    array: 3D numpy.ndarray
        The array representing the potential slices. The first dimension is the slice
        index and the last two are the spatial dimensions.
    slice_thickness: float
        The thicknesses of potential slices [Å]. If a float, the thickness is the same
        for all slices.
        If a sequence, the length must equal the length of the potential array.
    extent: one or two float, optional
        Lateral extent of the potential [Å].
    sampling: one or two float, optional
        Lateral sampling of the potential [1 / Å].
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the
        slice indices after which an exit plane is desired, and hence during a
        multislice simulation a measurement is created. If `exit_planes` is an integer
        `n`, a measurement is collected every `n` slices and after the last slice. The
        first measurement is then taken at the entrance surface (zero thickness),
        before any scattering, so the thickness series has ``1 + ceil(num_slices / n)``
        planes; index it with ``[-1]`` for the exit surface. If `n` exceeds the number
        of slices, only the exit surface is returned and no thickness axis is added.
    ensemble_axes_metadata : list of AxesMetadata
        Axis metadata for each ensemble axis. The axis metadata must be compatible with
        the shape of the array.
    metadata : dict
        A dictionary defining wave function metadata. All items will be added to the
        metadata of measurements derived from the waves.
    """

    _base_dims = 3

    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        slice_thickness: float | Sequence[float],
        extent: Optional[float | tuple[float, float]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        metadata: Optional[dict] = None,
        cell: Optional[np.ndarray] = None,
    ):
        if metadata is None:
            metadata = {}
        metadata = {"label": "potential", "units": "eV / e", **metadata}

        super().__init__(
            array=array,
            slice_thickness=slice_thickness,
            extent=extent,
            sampling=sampling,
            exit_planes=exit_planes,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
            cell=cell,
        )

    @property
    def _slice_kwargs(self) -> dict:
        # The cell is the grid's, which a slice must keep: the "cell" its
        # metadata carries is dropped by a grid without one.
        return {**super()._slice_kwargs, "cell": self.cell}

    @staticmethod
    def _transmission_function(array, energy):
        # complex_exponential_scaled fuses the sigma multiplication into the
        # GPU sin/cos kernel, avoiding one slice-sized real temporary.
        array = complex_exponential_scaled(array, energy2sigma(energy))
        return array

    @classmethod
    def from_array_and_metadata(
        cls: type[PotentialArray],
        array: np.ndarray | da.core.Array,
        axes_metadata: list[AxisMetadata],
        metadata: dict,
    ) -> PotentialArray:
        raise NotImplementedError

    def transmission_function(self, energy: float) -> TransmissionFunction:
        """
        Calculate the transmission functions for each slice for a specific energy.

        Parameters
        ----------
        energy: float
            Electron energy [eV].

        Returns
        -------
        transmissionfunction : TransmissionFunction
            Transmission functions for each slice.
        """
        xp = get_array_module(self.array)

        if self.is_lazy:
            array = da.map_blocks(
                self._transmission_function,
                self.array,
                energy=energy,
                meta=xp.array(
                    (), dtype=np.result_type(self.array.dtype, np.complex64)
                ),
            )
        else:
            array = self._transmission_function(self.array, energy=energy)

        t = TransmissionFunction(
            array,
            slice_thickness=self.slice_thickness,
            extent=self.extent,
            energy=energy,
            exit_planes=self.exit_planes,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=self.metadata,
            cell=self.cell,
        )
        return t

    def transmit(self, waves: Waves, conjugate: bool = False) -> Waves:
        """
        Transmit a wave function through a potential slice.

        Parameters
        ----------
        waves: Waves
            Waves object to transmit.
        conjugate : bool, optional
            If True, use the conjugate of the transmission function. Default is False.

        Returns
        -------
        transmission_function : TransmissionFunction
            Transmission function for the wave function through the potential slice.
        """

        transmission_function = self.transmission_function(waves._valid_energy)

        return transmission_function.transmit(waves, conjugate=conjugate)


class TransmissionFunction(PotentialArray, HasAcceleratorMixin):
    """Class to describe transmission functions.

    Parameters
    ----------
    array : 3D numpy.ndarray
        The array representing the potential slices. The first dimension is the slice
        index and the last two are the spatial dimensions.
    slice_thickness : float
        The thicknesses of potential slices [Å]. If a float, the thickness is the same
        for all slices. If a sequence, the length must equal the length of the potential
        array.
    extent : one or two float, optional
        Lateral extent of the potential [Å].
    sampling : one or two float, optional
        Lateral sampling of the potential [1 / Å].
    energy : float
        Electron energy [eV].
    exit_planes : int or tuple of int, optional
        The slice indices after which an exit plane is desired, as for
        :class:`.PotentialArray`.
    ensemble_axes_metadata : list of AxisMetadata, optional
        Axis metadata for each ensemble axis.
    metadata : dict, optional
        A dictionary defining the metadata of the transmission functions.
    """

    def __init__(
        self,
        array: np.ndarray,
        slice_thickness: float | Sequence[float],
        extent: Optional[float | tuple[float, float]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        energy: Optional[float] = None,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        metadata: Optional[dict] = None,
        cell: Optional[np.ndarray] = None,
    ):
        self._accelerator = Accelerator(energy=energy)
        super().__init__(
            array,
            slice_thickness,
            extent,
            sampling,
            exit_planes=exit_planes,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
            cell=cell,
        )

    @property
    def _slice_kwargs(self) -> dict:
        # The constructor takes no metadata: a transmission function always has
        # the default metadata of a potential array.
        return {"extent": self.extent, "energy": self.energy, "cell": self.cell}

    def get_chunk(self, first_slice, last_slice) -> TransmissionFunction:
        ensemble = (slice(None),) * len(self.ensemble_shape)
        return self[ensemble + (slice(first_slice, last_slice),)]

    def transmission_function(self, energy) -> TransmissionFunction:
        """
        Calculate the transmission functions for each slice for a specific energy.

        Parameters
        ----------
        energy: float
            Electron energy [eV].

        Returns
        -------
        transmissionfunction : TransmissionFunction
            Transmission functions for each slice.
        """
        if energy != self.energy:
            raise RuntimeError()
        return self

    def transmit(self, waves: Waves, conjugate: bool = False) -> Waves:
        """
        Transmit a wave function through a potential slice.

        Parameters
        ----------
        waves: Waves
            Waves object to transmit.
        conjugate : bool, optional
            If True, use the conjugate of the transmission function. Default is False.

        Returns
        -------
        transmission_function : Waves
            Transmission function for the wave function through the potential slice.
        """
        self.accelerator.check_match(waves)
        self.grid.check_match(waves)

        transmission = self.array[0]
        if conjugate:
            # the method, not xp.conjugate: a CuPy function rejects a dask array
            transmission = transmission.conj()

        lazy_waves = isinstance(waves._array, da.Array)
        if isinstance(transmission, da.Array) and not lazy_waves:
            # The waves are transmitted in place, and an in-place NumPy or CuPy
            # product cannot take a dask operand. Keep eager waves eager: compute
            # the (single-slice) transmission function, synchronously so CuPy
            # tasks never run concurrently.
            transmission = transmission.compute(scheduler="synchronous")

        waves._array *= transmission

        return waves


class CrystalPotential(_PotentialBuilder):
    """
    The crystal potential may be used to represent a potential consisting of a repeating
    unit. This may allow calculations to be performed with lower computational cost by
    calculating the potential unit once and repeating it.

    If the repeating unit is a potential with frozen phonons, it is treated as a
    pool of displaced configurations: every repetition of the unit (each lateral
    tile of every `z`-repetition) draws a configuration from the pool. Draws are
    balanced over the whole crystal, so reuse of a configuration is the minimum
    the pool size allows -- no two tiles within a layer are identical whenever
    the pool permits, and a pool of at least
    ``repetitions[0] * repetitions[1] * repetitions[2]`` configurations gives
    every repeated unit a distinct configuration (statistically equivalent to
    tiling the displaced atoms directly). If `num_frozen_phonons` is set, an
    ensemble of crystal potentials is created; each member independently
    rebuilds its own pool of atomic displacement snapshots (reseeded from
    that member's own seed) rather than sharing one fixed pool across the
    ensemble, so members are genuinely independent thermal realisations --
    there is no need to size the pool for the ensemble, only for a single
    crystal (see above).

    Parameters
    ----------
    potential_unit : BasePotential
        The potential unit to assemble the crystal potential from.
    repetitions : three int
        The repetitions of the potential in `x`, `y` and `z`.
    num_frozen_phonons : int, optional
        Number of crystal realisations in the frozen-phonon ensemble; each
        realisation independently rebuilds its own pool of atomic
        displacement snapshots. Without `seeds`, the member seeds are spawned
        from the crystal's root seed (see `seeds`).
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the
        slice indices after which an exit plane is desired, and hence during a
        multislice simulation a measurement is created. If `exit_planes` is an integer
        `n`, a measurement is collected every `n` slices and after the last slice. The
        first measurement is then taken at the entrance surface (zero thickness),
        before any scattering, so the thickness series has ``1 + ceil(num_slices / n)``
        planes; index it with ``[-1]`` for the exit surface. If `n` exceeds the number
        of slices, only the exit surface is returned and no thickness axis is added.
    seeds: int or sequence of int, optional
        One seed for each member of the frozen-phonon ensemble. An int is the seed
        of a one-member ensemble, or, with `num_frozen_phonons`, the seed the
        member seeds are drawn from. Without `seeds`, the crystal is drawn from a
        root seed fixed when it is created, so that every simulation of it sees
        the same crystal: the seeds of the unit's `FrozenPhonons` determine it,
        and a unit with several configurations but no `FrozenPhonons` (a built
        potential ensemble, a list of GPAW calculators) gets a random one. Two
        crystals made from one `FrozenPhonons` unit are therefore the same
        crystal, with the same mosaic; pass different `seeds` for independent
        crystals.
    ensemble_mean : bool, optional
        If True (default), the mean over the frozen-phonon ensemble is calculated.
        If False, the individual configurations are returned.
    """

    # Derived state: the sliced atoms, as on _FieldBuilderFromAtoms (built by
    # this class's own get_sliced_atoms(); CrystalPotential descends from
    # _PotentialBuilder, so it does not inherit that declaration), and the pools
    # that _partition_args builds once, for all members or for each member.
    _eq_exclude = ("_sliced_atoms", "_shared_pool", "_member_pools")

    # Appended to a FrozenPhonons unit's seeds to make the root seed of a
    # crystal (see _root_seed_of): "CRYS" in ASCII.
    _root_seed_tag = 0x43525953

    # A crystal pickled without a shared pool or member pools loads with none;
    # `__setstate__` gives one pickled without a root seed that of a fresh crystal
    # of its unit, or none when it has member seeds.
    _shared_pool = None
    _member_pools = None

    def __init__(
        self,
        potential_unit: BasePotential,
        repetitions: tuple[int, int, int],
        num_frozen_phonons: int | None = None,
        exit_planes: int | None = None,
        seeds: int | tuple[int, ...] | None = None,
        ensemble_mean: bool = True,
        _root_seed: int | tuple[int, ...] | None = None,
    ):
        # _root_seed is given when a crystal is rebuilt, e.g. for a lazy block
        root_seed = None
        if num_frozen_phonons is None and seeds is None:
            root_seed = _root_seed
            if root_seed is None:
                root_seed = self._root_seed_of(potential_unit)
            self._seeds = None
        else:
            if seeds is None:
                root = np.random.SeedSequence(self._root_seed_of(potential_unit))
                # one distinct seed per member: a repeat takes the next child
                members = {}
                while len(members) < num_frozen_phonons:
                    for child in root.spawn(num_frozen_phonons - len(members)):
                        members.setdefault(int(child.generate_state(1)[0]))
                seeds = tuple(members)
            elif num_frozen_phonons is None:
                seeds = validate_seeds(seeds)
                num_frozen_phonons = len(seeds)

            self._seeds = validate_seeds(seeds, num_frozen_phonons)

        if (
            (potential_unit.num_configurations == 1)
            and (num_frozen_phonons is not None)
            and (num_frozen_phonons > 1)
        ):
            warnings.warn(
                "'num_frozen_phonons' is greater than one, but the potential unit does"
                " not have frozen phonons"
            )

        gpts = (
            potential_unit._valid_gpts[0] * repetitions[0],
            potential_unit._valid_gpts[1] * repetitions[1],
        )
        extent = (
            potential_unit._valid_extent[0] * repetitions[0],
            potential_unit._valid_extent[1] * repetitions[1],
        )

        box = extent + (potential_unit.thickness * repetitions[2],)
        slice_thickness = potential_unit.slice_thickness * repetitions[2]

        assert hasattr(potential_unit, "device")

        # Propagate the in-plane cell geometry: if the unit potential has a
        # non-orthogonal grid, tile the full 3×3 cell so _FieldBuilder can
        # detect and preserve the skew metric.  Do not pass ``box`` in that
        # case -- _FieldBuilder derives it from the cell and ``box is None``
        # is required for the auto-detection of the skew path.
        unit_cell_2d = potential_unit.grid.cell
        if unit_cell_2d is not None:
            cell_3d = np.zeros((3, 3))
            cell_3d[:2, :2] = np.asarray(unit_cell_2d)
            cell_3d[2, 2] = potential_unit.thickness
            cell_3d[0] *= repetitions[0]
            cell_3d[1] *= repetitions[1]
            cell_3d[2] *= repetitions[2]
            cell = Cell(cell_3d)
            init_box = None
        else:
            cell = Cell(np.diag(box))
            init_box = box

        super().__init__(
            array_object=PotentialArray,
            gpts=gpts,
            cell=cell,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=potential_unit.device,
            plane="xy",
            origin=(0.0, 0.0, 0.0),
            box=init_box,
            periodic=True,
        )

        self._potential_unit = potential_unit
        self._repetitions = repetitions
        self._ensemble_mean = ensemble_mean
        self._sliced_atoms: Optional[BaseSlicedAtoms] = None
        self._root_seed = root_seed
        self._shared_pool: Optional[PotentialArray] = None
        self._member_pools: Optional[tuple[PotentialArray, ...]] = None

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_root_seed" not in state:
            self._root_seed = (
                None
                if self._seeds is not None
                else self._root_seed_of(self._potential_unit)
            )

    def __eq__(self, other):
        # The root seed is an identity, compared exactly: the generic equality
        # compares numbers to a relative tolerance, under which two different
        # 128-bit roots can compare equal.
        return super().__eq__(other) and self._root_seed == other._root_seed

    @property
    def ensemble_mean(self) -> bool:
        return self._ensemble_mean

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        if self._seeds is None:
            return ()
        else:
            return (self.num_configurations,)

    @property
    def num_configurations(self):
        if self._seeds is None:
            return 1
        else:
            return len(self._seeds)

    @property
    def seeds(self):
        return self._seeds

    @property
    def potential_unit(self) -> BasePotential:
        return self._potential_unit

    @property
    def gpts(self) -> tuple[int, int] | None:
        return super().gpts

    @gpts.setter
    def gpts(self, gpts: int | tuple[int, int]):
        gpts = self.grid._validate(gpts, dtype=int)
        if not (
            (gpts[0] % self.repetitions[0] == 0)
            and (gpts[1] % self.repetitions[1] == 0)
        ):
            raise ValueError(
                "Number of grid points must be divisible by the number of potential"
                " repetitions."
            )
        unit_gpts = (
            gpts[0] // self._repetitions[0],
            gpts[1] // self._repetitions[1],
        )
        if isinstance(self._potential_unit, PotentialArray):
            self._require_the_grid_of_a_built_unit(unit_gpts, self._potential_unit.gpts)
            return
        # The unit first: a unit that rejects its gpts leaves the crystal unchanged.
        self._potential_unit.gpts = unit_gpts
        self.grid.gpts = gpts

    def match_grid(self, other, check_match: bool = False):
        """Match the grid to another object with a Grid, keeping the unit in step."""
        # The gpts go through the setter, which keeps the unit in step or raises
        # before anything changes; the extent is checked first so that a mismatch
        # leaves the gpts alone as well.
        if check_match:
            self.grid.check_match(other)
        self.grid.check_match(Grid(extent=other.extent))
        if other.extent is None:
            # An object given a sampling works out its gpts from the extent.
            other.extent = self.extent
        if other.gpts is not None and tuple(other.gpts) != self.gpts:
            self.gpts = other.gpts
        self.grid.match(other, check_match=check_match)
        return self

    @staticmethod
    def _require_the_grid_of_a_built_unit(requested, current):
        """
        A built `PotentialArray` has the grid of its data. Setting the grid it has
        changes nothing; any other grid raises, with the unit left untouched.
        """
        if not np.allclose(requested, current):
            raise RuntimeError(
                "The grid of a built PotentialArray unit is that of its data and "
                "cannot be changed; set the gpts or sampling of the unit before "
                "building it."
            )

    @property
    def sampling(self) -> tuple[float, float] | None:
        return super().sampling

    @sampling.setter
    def sampling(self, sampling: tuple[float, float]):
        validated = self.grid._validate(sampling, dtype=float)
        if validated is None or not np.all(
            np.isfinite(validated) & (np.asarray(validated) > 0)
        ):
            raise ValueError(
                f"The sampling must be positive and finite, got {sampling}."
            )
        if isinstance(self._potential_unit, PotentialArray):
            self._require_the_grid_of_a_built_unit(
                validated, self._potential_unit.sampling
            )
            return
        # The unit rounds its own gpts up for the requested sampling, and the
        # crystal takes whole units, so its gpts follow the unit's.
        self._potential_unit.sampling = sampling
        unit_gpts = self._potential_unit._valid_gpts
        self.grid.gpts = (
            unit_gpts[0] * self._repetitions[0],
            unit_gpts[1] * self._repetitions[1],
        )

    @property
    def repetitions(self) -> tuple[int, int, int]:
        return self._repetitions

    @property
    def num_slices(self) -> int:
        return self._potential_unit.num_slices * self.repetitions[2]

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        if self.seeds is None:
            return []
        else:
            return [FrozenPhononsAxis(_ensemble_mean=self._ensemble_mean)]

    def get_sliced_atoms(self) -> BaseSlicedAtoms:
        """
        The atoms of the full crystal grouped into the slices given by the slice
        thicknesses.

        The atoms are those of the crystal that `generate_slices` builds. A unit
        without frozen phonons is tiled: its transformed (orthogonalised) atoms
        are repeated by the crystal repetitions. A unit with frozen phonons is a
        pool of displaced configurations, and every repeated unit (each lateral
        tile of every `z`-repetition) takes the displaced atoms of the pool
        configuration its slices are built from, at the offset of that unit.
        This makes ``CrystalPotential`` work with any code path that derives
        atomic sites from a potential via ``get_sliced_atoms`` -- e.g. the
        core-loss EELS drivers' automatic site extraction, whose sites then sit
        on the displaced atoms the elastic slices hold, as for ``Potential``.

        Notes
        -----
        - The atoms are those of one crystal: of the first member of a
          frozen-phonon ensemble (`seeds`), whose slices `generate_slices`
          yields, as ``Potential.get_sliced_atoms`` returns those of its first
          configuration. Each member of an ensemble is a crystal of its own,
          and the core-loss drivers take the sites of each member from it.
        - The result is cached; the tile is non-trivial for large supercells.

        Returns
        -------
        sliced_atoms : BaseSlicedAtoms
        """
        if self._sliced_atoms is not None:
            return self._sliced_atoms

        if not hasattr(self._potential_unit, "get_transformed_atoms"):
            raise RuntimeError(
                "Cannot derive atoms from a CrystalPotential whose "
                f"potential_unit ({type(self._potential_unit).__name__}) does "
                "not expose 'get_transformed_atoms' (e.g. a precomputed "
                "PotentialArray). Pass the scattering sites explicitly instead."
            )

        if self._has_displaced_atoms:
            tiled_atoms = self._displaced_atoms()
        else:
            unit_atoms = self._potential_unit.get_transformed_atoms()
            tiled_atoms = unit_atoms * self._repetitions

        self._sliced_atoms = SliceIndexedAtoms(
            tiled_atoms, slice_thickness=self.slice_thickness
        )

        return self._sliced_atoms

    @property
    def _has_displaced_atoms(self) -> bool:
        """Whether the unit is a pool of displaced configurations of atoms it
        can return (see `_displaced_atoms`): the crystals of the members of an
        ensemble then differ in their atoms."""
        frozen_phonons = getattr(self._potential_unit, "frozen_phonons", None)
        return (
            hasattr(self._potential_unit, "to_atoms_ensemble")
            and isinstance(frozen_phonons, BaseFrozenPhonons)
            and not isinstance(frozen_phonons, DummyFrozenPhonons)
        )

    def _displaced_atoms(self) -> Atoms:
        """The displaced atoms of the crystal `generate_slices` builds: for
        every repeated unit, the configuration of the pool that `_draw_mosaic`
        draws for it, translated to the offset of the unit."""
        member_seed = None if self.seeds is None else int(self.seeds[0])
        pool_unit = self._pool_unit_for_member(member_seed)
        # The configurations of the pool, in the order its build stacks them:
        # transformed to the unit's box, displaced, and wrapped into it (see
        # Potential.to_atoms_ensemble), exactly as the pool's slices hold them.
        configurations = list(
            np.asarray(pool_unit.to_atoms_ensemble().trajectory, dtype=object).ravel()
        )
        drawn_tiles = self._draw_mosaic(len(configurations))
        if drawn_tiles is None:
            drawn_tiles = np.zeros(
                (self.repetitions[2],) + tuple(self.repetitions[:2]), dtype=int
            )

        # The unit's box, as the mosaic places it: lateral tile (a, b) of
        # z-repetition k is offset by (a, b, k) times the box.
        box = np.asarray(pool_unit.box, dtype=float)
        arrays: dict[str, list[np.ndarray]] = {
            key: [] for key in configurations[0].arrays
        }
        for (k, a, b), config_idx in np.ndenumerate(drawn_tiles):
            configuration = configurations[config_idx]
            offset = box * (a, b, k)
            for key in arrays:
                value = configuration.arrays[key]
                arrays[key].append(value + offset if key == "positions" else value)

        displaced = Atoms(cell=np.diag(box * np.array(self.repetitions)), pbc=True)
        for key, values in arrays.items():
            displaced.arrays[key] = np.concatenate(values)
        displaced.arrays.pop(SOURCE_INDEX, None)
        return displaced

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        args = unpack_blockwise_args(args)
        potential, seed, shared_pool, member_pools = args[0]
        if hasattr(potential, "item"):
            potential = potential.item()

        if seed is not None:
            num_frozen_phonons = len(seed)
        else:
            num_frozen_phonons = None

        new = cls(
            potential_unit=potential,
            seeds=seed,
            num_frozen_phonons=num_frozen_phonons,
            **kwargs,
        )
        new._shared_pool = shared_pool
        new._member_pools = member_pools
        return _wrap_with_array(new)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(
            exclude=("potential_unit", "seeds", "num_frozen_phonons")
        )
        output = partial(self._from_partitioned_args_func, **kwargs)
        return output

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True):
        if chunks is None:
            chunks = 1

        chunks = validate_chunks(self.ensemble_shape, chunks)
        # print(self.ensemble_shape)

        if chunks == ():
            old_chunks = ()
            chunks = ((1,),)
        else:
            old_chunks = chunks

        potential_unit = self.potential_unit
        # A lazily built PotentialArray unit is computed once, here, for all
        # ensemble members. Left lazy, every member's generate_slices would
        # compute it again, as a nested compute inside that member's task.
        unit_is_lazy = (
            isinstance(potential_unit, PotentialArray) and potential_unit.is_lazy
        )
        # Unless every member reseeds its own frozen-phonon pool (see
        # _pool_unit_for_member), the members of an unbuilt unit draw from one
        # pool. It is built once, here, for all members and blocks, rather
        # than by every member's generate_slices. Members that reseed their
        # pools are given them in a lazy graph (see below).
        shared_pool = self._shared_pool
        member_pools = self._member_pools
        fp = getattr(potential_unit, "frozen_phonons", None)
        pool_per_member = (
            self.seeds is not None
            and isinstance(fp, FrozenPhonons)
            and fp.num_configs > 1
        )
        build_pool = (
            shared_pool is None
            and not pool_per_member
            and not isinstance(potential_unit, PotentialArray)
        )

        if lazy:
            if unit_is_lazy:
                # one task of this graph, which every block depends on
                lazy_unit = dask.delayed(
                    partial(
                        potential_unit.__class__,
                        **potential_unit._copy_kwargs(exclude=("array",)),
                    )
                )(potential_unit.array)
            else:
                lazy_unit = dask.delayed(potential_unit)

            if build_pool:
                # one task of this graph, which every block depends on; it
                # derives the pool from the graph's one copy of the unit
                shared_pool = dask.delayed(self._build_shared_pool)(
                    lazy_unit, self._n_lateral_tiles
                )
            elif shared_pool is not None:
                shared_pool = dask.delayed(shared_pool)

            if pool_per_member and member_pools is None:
                # One task per member, which every block of the member depends
                # on, so that each member's pool is built once per graph rather
                # than once per block. Like the shared pool, it is derived from
                # the graph's one copy of the unit. A pool is held until the last
                # block of its member has run.
                member_pools = tuple(
                    dask.delayed(self._build_member_pool)(
                        lazy_unit, self._n_lateral_tiles, int(seed)
                    )
                    for seed in self.seeds
                )

            arrays = []

            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                if self.seeds is not None:
                    seeds = self.seeds[start:stop]
                else:
                    seeds = None

                lazy_args = dask.delayed(_wrap_with_array)(
                    (
                        lazy_unit,
                        seeds,
                        shared_pool,
                        None if member_pools is None else member_pools[start:stop],
                    ),
                    ndims=1,
                )
                lazy_array = da.from_delayed(lazy_args, shape=(1,), dtype=object)
                arrays.append(lazy_array)

            array = da.concatenate(arrays)

            if old_chunks == ():
                array = array[0]

        else:
            if unit_is_lazy:
                potential_unit = potential_unit.ensure_computed(progress_bar=False)
            if build_pool:
                shared_pool = self._build_shared_pool(
                    potential_unit, self._n_lateral_tiles
                )

            array = np.zeros((len(chunks[0]),), dtype=object)
            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                if self.seeds is not None:
                    seeds = self.seeds[start:stop]
                else:
                    seeds = None

                # An eager ensemble runs its members one after another, and each
                # member that reseeds its pool builds it once, unless it was given
                # its pool as a member of a lazy block.
                itemset(
                    array,
                    i,
                    (
                        potential_unit,
                        seeds,
                        shared_pool,
                        None if member_pools is None else member_pools[start:stop],
                    ),
                )

            if old_chunks == ():
                array = _wrap_with_array(array[0], ndims=0)

        return (array,)

    @property
    def _n_lateral_tiles(self) -> int:
        return self.repetitions[0] * self.repetitions[1]

    @classmethod
    def _root_seed_of(
        cls, potential_unit: BasePotential
    ) -> int | tuple[int, ...] | None:
        """The entropy of the ``np.random.SeedSequence`` that draws the mosaic
        of a crystal without ``seeds``, and from which the member seeds of one
        with only ``num_frozen_phonons`` are spawned.

        It is fixed when the crystal is created, so that every
        ``generate_slices`` call of one crystal (every lazy block, every pass
        of a simulation) draws the same crystal. For a ``FrozenPhonons`` unit
        it is all of the unit's seeds followed by a tag, so a seeded
        ``FrozenPhonons`` makes the crystal reproducible, and its streams are
        independent of the ``default_rng(seed)`` streams ``FrozenPhonons``
        draws its configurations from. Any other unit with more than one
        configuration gets fresh entropy. A unit with one configuration draws
        no mosaic and gets none.
        """
        fp = getattr(potential_unit, "frozen_phonons", None)
        if isinstance(fp, FrozenPhonons):
            return (*(int(seed) for seed in fp.seed), cls._root_seed_tag)
        if potential_unit.num_configurations > 1:
            return np.random.SeedSequence().entropy
        return None

    def _pool_unit_for_member(self, member_seed: Optional[int]) -> BasePotential:
        """The unit to draw pool configurations from for one ensemble member, or
        for the single default builder (see `_pool_unit`)."""
        return self._pool_unit(self.potential_unit, self._n_lateral_tiles, member_seed)

    @classmethod
    def _build_shared_pool(cls, unit: BasePotential, n_tiles: int) -> PotentialArray:
        """The pool every member draws from, unless each reseeds its own."""
        return cls._pool_unit(unit, n_tiles, None).build(lazy=False)

    @classmethod
    def _build_member_pool(
        cls, unit: BasePotential, n_tiles: int, member_seed: int
    ) -> PotentialArray:
        """The pool of the member with seed `member_seed`, which reseeds its own."""
        return cls._pool_unit(unit, n_tiles, member_seed).build(lazy=False)

    @staticmethod
    def _pool_unit(
        unit: BasePotential, n_tiles: int, member_seed: Optional[int]
    ) -> BasePotential:
        """Return the unit potential to draw pool configurations from for one
        ensemble member (``member_seed`` is that member's seed), or for the
        single default builder (``member_seed`` is None), of a crystal of
        ``n_tiles`` lateral tiles of ``unit``.

        Two independent adjustments are made when the unit carries frozen
        phonons; a precomputed ``PotentialArray`` unit has a fixed pool and is
        always returned unchanged.

        1. **Enlarge to the tile count.** A frozen-phonon ``CrystalPotential``
           assembles each slice as a mosaic: every lateral tile draws an
           independent pool configuration. If the pool holds fewer
           configurations than there are lateral tiles
           (``repetitions[0] * repetitions[1]``), some tiles must reuse a
           configuration -- reintroducing the artificial in-plane periodicity
           the mosaic is meant to remove. The pool is transparently enlarged
           to the tile count (warning).

        2. **Reseed per ensemble member.** Every ensemble member is built from
           the *same* ``potential_unit`` object, so without reseeding every
           member would draw from an identical, fixed pool of configurations
           -- differing only in how those same snapshots are arranged across
           the crystal, not in which atomic displacements exist. That is a
           much weaker form of independence than a frozen-phonon ensemble is
           supposed to provide, and sizing the pool cannot fix it (drawing
           from a bigger *shared* pool still shares it). Instead, when this
           call belongs to an ensemble (``member_seed`` is not None), the pool
           is quietly rebuilt with ``member_seed`` as its root seed, so each
           member gets its own independent set of atomic snapshots. This adds
           no cost: the pool was already rebuilt once per member.
        """
        fp = getattr(unit, "frozen_phonons", None)
        if not isinstance(fp, FrozenPhonons) or fp.num_configs <= 1:
            return unit

        enlarge = fp.num_configs < n_tiles
        reseed = member_seed is not None
        if not enlarge and not reseed:
            return unit

        if enlarge:
            warnings.warn(
                f"frozen-phonon pool ({fp.num_configs}) is smaller than the "
                f"number of lateral tiles ({n_tiles}); enlarging the pool to "
                f"{n_tiles} so each tile draws a distinct configuration and no "
                "lateral duplication occurs. Pass a unit with "
                f"num_configs >= {n_tiles} to silence this."
            )

        new_fp = FrozenPhonons(
            fp.atoms,
            num_configs=n_tiles if enlarge else fp.num_configs,
            sigmas=fp.sigmas,
            directions=fp.directions,
            ensemble_mean=fp.ensemble_mean,
            seed=int(member_seed) if reseed else int(fp.seed[0]),
        )
        if "frozen_phonons" in unit._arg_keys(type(unit)):
            # ``GPAWPotential`` takes its frozen phonons next to its calculators,
            # which may be unread ``.gpw`` paths that its constructor cannot take
            # back. Only the frozen phonons differ, so the calculators are shared.
            pool_unit = copy.copy(unit)
            pool_unit._frozen_phonons = new_fp
            return pool_unit

        kwargs = unit._copy_kwargs(exclude=("atoms",))
        # The unit's box was reported when the potential was constructed.
        with _box_strain_warning_silenced():
            return type(unit)(new_fp, **kwargs)

    def _built_pool(self) -> PotentialArray:
        """The configurations the slices are drawn from, in memory, with a leading
        configuration axis."""
        if self._shared_pool is not None:
            pool_unit = self._shared_pool
        elif self._member_pools is not None:
            pool_unit = self._member_pools[0]
        else:
            member_seed = None if self.seeds is None else int(self.seeds[0])
            pool_unit = self._pool_unit_for_member(member_seed)
        if not isinstance(pool_unit, PotentialArray):
            potentials = pool_unit.build(lazy=False)
        else:
            potentials = pool_unit

        assert isinstance(potentials, PotentialArray)

        if len(potentials.shape) == 3:
            potentials = potentials.expand_dims(axis=0)

        # A lazily-built PotentialArray unit (the default of Potential.build())
        # carries a dask array, and tiling it would yield dask-backed slices:
        # eager consumers such as build(lazy=False) cannot place those into a
        # CuPy array, and on CPU compute them one slice at a time. The unit cell
        # is small; materialise it once, into a new object, so that the caller's
        # unit stays lazy. Blocks from _partition_args already receive it in
        # memory; this covers direct calls. The synchronous scheduler keeps a
        # compute that runs inside a task from starting a nested thread pool,
        # and CuPy kernels from running concurrently.
        return potentials.ensure_computed(scheduler="synchronous", progress_bar=False)

    def _draw_mosaic(self, n_configs: int) -> Optional[np.ndarray]:
        """The pool configuration of every repeated unit of the crystal, drawn
        from a pool of `n_configs` configurations: an index into the pool for
        every lateral tile of every z-repetition, with axes z-repetition,
        ``repetitions[0]`` and ``repetitions[1]``. None for a pool of one
        configuration, which is tiled instead.

        The draws come from the seed of the first member, as the pool does (see
        `_built_pool`), or, without seeds, from the crystal's root seed, so
        every call draws the same mosaic. `generate_slices` assembles the slices
        from it and `get_sliced_atoms` places the displaced atoms on it; both
        must take it from here, or the core-loss sites stop following the
        configurations the elastic slices are built from.
        """
        if n_configs <= 1:
            return None

        if self.seeds is None:
            rng = np.random.default_rng(np.random.SeedSequence(self._root_seed))
        else:
            rng = np.random.default_rng(int(self.seeds[0]))

        tile_xy = self.repetitions[:2]
        n_tiles = tile_xy[0] * tile_xy[1]

        # Balanced global drawing: every pool configuration receives a total
        # usage budget of floor/ceil(total_slots / n_configs) over the whole
        # crystal (all lateral tiles x all z-repetitions), and each z-layer
        # draws the ``n_tiles`` configurations with the most budget remaining
        # (random tie-breaking keeps assignments uniform). Drawing each layer
        # independently instead (i.e. with replacement across z) lets the
        # same configuration recur in many layers even when the pool is large
        # enough to avoid it, correlating slices along z and measurably
        # inflating thermal-diffuse statistics above the tiled-atoms ground
        # truth. With budgets, reuse is the minimum the pool size allows and
        # is spread evenly: once ``n_configs >= n_tiles * repetitions[2]``
        # every unit cell in the crystal receives a distinct configuration --
        # statistically identical to tiling the displaced atoms directly.
        # Within a layer draws remain distinct whenever the pool allows (no
        # in-plane duplication), as before.
        total_slots = n_tiles * self.repetitions[2]
        base, extra = divmod(total_slots, n_configs)
        budgets = np.full(n_configs, base, dtype=np.int64)
        if extra:
            budgets[rng.permutation(n_configs)[:extra]] += 1

        def _draw_config_tiles() -> np.ndarray:
            if n_configs >= n_tiles:
                # The ``n_tiles`` most-underused configurations, in random
                # order (permute first; the stable sort then orders by budget
                # only, keeping ties shuffled).
                order = rng.permutation(n_configs)
                chosen = order[np.argsort(-budgets[order], kind="stable")[:n_tiles]]
            else:
                # Pool smaller than a single layer: in-plane repeats are
                # unavoidable; cycle freshly shuffled permutations to spread
                # them as evenly as possible.
                n_perms = -(-n_tiles // n_configs)  # ceil
                chosen = np.concatenate(
                    [rng.permutation(n_configs) for _ in range(n_perms)]
                )[:n_tiles]
            np.subtract.at(budgets, chosen, 1)
            return chosen.reshape(tile_xy)

        # One layer after another, in z order: the draws of a layer depend on
        # the budgets the layers before it left.
        return np.stack([_draw_config_tiles() for _ in range(self.repetitions[2])])

    def generate_slices(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        return_depth: bool = False,
    ):
        """
        Generate the slices for the potential.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the generated potential.
        last_slice : int, optional
            Index of the last slice of the generated potential.
        return_depth : bool
            If True, return the depth of each generated slice.

        Yields
        ------
        slices : generator of numpy.ndarray
            Generator for the array of slices.
        """
        potentials = self._built_pool()

        if last_slice is None:
            last_slice = len(self)

        exit_plane_after = self._exit_plane_after
        cum_thickness = np.cumsum(self.slice_thickness)
        unit_slices = len(self.potential_unit)
        global_idx = 0  # global slice counter across all z-repetitions

        # Lazy cache of tiled unit slices, keyed by (config_idx, slice_idx).
        # Without it each (z-rep, unit-slice) pair re-tiles the same array via
        # ``.tile(self.repetitions[:2])`` — for the no-frozen-phonon case
        # (n_configs == 1) every z-rep produces an identical result so the
        # cost scales linearly with ``repetitions[2]``. The cache turns this
        # into ``n_configs * len(self.potential_unit)`` unique tile calls.
        # For the SrTiO3 tutorial (reps=(4,4,25), 2 unit slices, no FP) this
        # is 2 tiles instead of 50; the cache footprint is bounded by the
        # tiled-unit byte size and released with the generator.
        tiled_cache: dict[tuple[int, int], PotentialArray] = {}
        unit_generators: dict[int, object] = {}
        tile_xy = self.repetitions[:2]

        n_configs = potentials.shape[0]

        # The mosaic path (frozen-phonon pools, n_configs > 1) needs random
        # per-tile access into the pool; the pool array is eager (see above).
        xp = get_array_module(self.device)
        _pool_array = potentials.array

        def _tiled_slice(config_idx: int, j: int) -> PotentialArray:
            key = (config_idx, j)
            cached = tiled_cache.get(key)
            if cached is not None:
                return cached
            gen = unit_generators.get(config_idx)
            if gen is None:
                gen = potentials[config_idx].generate_slices()
                unit_generators[config_idx] = gen
            slic = next(gen).tile(tile_xy)
            tiled_cache[key] = slic
            return slic

        def _mosaic_slice(config_tiles: np.ndarray, j: int) -> PotentialArray:
            # Assemble sub-slice ``j`` of the lateral supercell by placing an
            # *independently drawn* pool configuration at every lateral
            # repetition (a mosaic), rather than replicating a single displaced
            # unit across all tiles. This is what reproduces genuine lateral
            # (in-plane) thermal disorder: with plain ``.tile()`` every one of
            # the ``repetitions[0] * repetitions[1]`` tiles is a bit-identical
            # copy, so there is no in-plane disorder at all and no diffuse
            # (Kikuchi) scattering can form. ``config_tiles`` holds one pool
            # index per lateral tile, shaped ``repetitions[:2]``.
            sub = _pool_array[:, j]  # (n_configs, uy, ux)
            uy, ux = sub.shape[-2], sub.shape[-1]
            mosaic = sub[xp.asarray(config_tiles)]  # (rep0, rep1, uy, ux)
            # Interleave to match ``PotentialArray.tile`` block layout, which
            # tiles the row axis by repetitions[0] and the col axis by
            # repetitions[1] (np.tile(array, (rep2, rep0, rep1))).
            mosaic = mosaic.transpose(0, 2, 1, 3).reshape(
                tile_xy[0] * uy, tile_xy[1] * ux
            )
            return potentials.__class__(
                mosaic[None],
                potentials.slice_thickness[j : j + 1],
                extent=self.extent,
            )

        # The whole mosaic is drawn up front, whatever the requested window, so
        # that different chunks of the same crystal see the same per-layer
        # configuration draws (see _draw_mosaic).
        drawn_tiles = self._draw_mosaic(n_configs)

        for i in range(self.repetitions[2]):
            # An independent displaced realisation per unit cell in the x, y
            # and z supercell directions. For a single-config pool this
            # collapses to the cheap cached ``.tile()`` path below.
            config_tiles = None if drawn_tiles is None else drawn_tiles[i]

            if global_idx + unit_slices <= first_slice:
                # This entire z-repetition is before the requested window;
                # advance the counter and skip.
                global_idx += unit_slices
                continue

            if global_idx >= last_slice:
                # Past the requested window; nothing more to yield.
                return

            for j in range(unit_slices):
                # Iterate j from 0 even for slices before first_slice in a
                # partially-overlapping rep, so the unit generator advances in
                # order (j=0, j=1, ...). The tiling cache ensures each
                # (config, j) pair is tiled at most once.
                if config_tiles is None:
                    slic = _tiled_slice(0, j)
                else:
                    slic = _mosaic_slice(config_tiles, j)

                if global_idx >= first_slice:
                    exit_planes = tuple(
                        np.where(exit_plane_after[global_idx : global_idx + 1])[0]
                    )
                    # Mutating the cached slice's exit_planes is safe: consumer
                    # reads exit_planes immediately on each yield and holds no
                    # back-reference across iterations.
                    slic._exit_planes = exit_planes

                    if return_depth:
                        yield cum_thickness[global_idx], slic
                    else:
                        yield slic

                global_idx += 1

                if global_idx >= last_slice:
                    return

    def generate_chunked_slices(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        chunk_size: int | str = "auto",
    ):
        """
        Generate potential slices in memory-budgeted chunks.

        The chunks hold exactly the slices of ``generate_slices``, each written
        into one preallocated array rather than stacked from a list of slices.
        A unit with one configuration is built once and each slice is tiled
        straight into its chunk. A frozen-phonon unit takes its slices from one
        ``generate_slices`` call (the pool is built and the mosaic drawn once
        for all chunks), and its pool, enlarged to the number of lateral tiles
        when smaller, is held on top of the chunk budget for the whole call.

        The dtype of the output follows the unit potential's array dtype,
        which is set by the abtem ``precision`` config key (float32 / float64).
        """
        from abtem.core.chunks import estimate_potential_chunk_size

        if last_slice is None:
            last_slice = len(self)

        if chunk_size == "auto":
            chunk_size = estimate_potential_chunk_size(self.gpts, self.device)
        chunk_size = min(chunk_size, last_slice - first_slice)

        xp = get_array_module(self.device)
        exit_plane_after = self._exit_plane_after
        if self.potential_unit.num_configurations == 1:
            unit = self._built_pool().array[0]
            unit_thickness = self.potential_unit.slice_thickness
            slices = (
                (
                    xp.tile(unit[i % len(unit)], self.repetitions[:2]),
                    unit_thickness[i % len(unit)],
                )
                for i in range(first_slice, last_slice)
            )
        else:
            slices = (
                (slic.array[0], slic.slice_thickness[0])
                for slic in self.generate_slices(first_slice, last_slice)
            )

        for chunk_start, chunk_end in generate_chunks(
            last_slice - first_slice, chunks=chunk_size, start=first_slice
        ):
            out = None
            slice_thicknesses = []

            for k in range(chunk_end - chunk_start):
                array, thickness = next(slices)
                if out is None:
                    out = xp.empty(
                        (chunk_end - chunk_start,) + array.shape, dtype=array.dtype
                    )
                out[k] = array
                slice_thicknesses.append(thickness)

            exit_planes = tuple(
                np.where(exit_plane_after[chunk_start:chunk_end])[0]
            )
            chunk = PotentialArray(
                out,
                slice_thickness=tuple(slice_thicknesses),
                extent=self.extent,
            )
            chunk._exit_planes = exit_planes
            yield chunk
