"""Module for describing electrostatic potentials using the independent atom model."""

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from functools import partial, reduce
from numbers import Number
from operator import mul
from typing import TYPE_CHECKING, Optional, Sequence, Type

import dask
import dask.array as da
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.data import chemical_symbols

from abtem.array import ArrayObject, validate_lazy
from abtem.atoms import (
    best_orthogonal_cell,
    cut_cell,
    is_cell_ab_in_plane,
    is_cell_orthogonal,
    orthogonalize_cell,
    pad_atoms,
    plane_to_axes,
    rotate_atoms_to_plane,
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
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype, itemset
from abtem.inelastic.phonons import (
    AtomsEnsemble,
    BaseFrozenPhonons,
    DummyFrozenPhonons,
    FrozenPhonons,
    validate_seeds,
)
from abtem.integrals import (
    QuadratureProjectionIntegrals,
    ScatteringFactorProjectionIntegrals,
)
from abtem.measurements import Images
from abtem.slicing import (
    BaseSlicedAtoms,
    SlicedAtoms,
    SliceIndexedAtoms,
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
        Generate potential slices in memory-budgeted chunks.

        Previously, ``build()`` always placed the entire slice dimension into a
        single dask chunk — meaning the full ``(num_slices, gpts_y, gpts_x)``
        array had to fit in memory (or VRAM) at once. There was no slice-level
        chunking. This method introduces that missing middle ground: it eagerly
        builds a group of contiguous slices that fits within a configurable
        memory budget, yields it as a ``PotentialArray``, and the caller can
        discard it after propagation before the next chunk is built. This
        bounds peak memory and enables simulations of systems whose full
        potential would not fit in memory.

        On GPU this is especially important: dask uses a synchronous scheduler,
        so the full potential chunk would be materialized at once in VRAM.
        Chunking over slices keeps VRAM usage bounded while still feeding the
        GPU enough data per chunk for efficient computation.

        This default implementation collects slices from ``generate_slices()``
        and stacks them. Subclasses may override for more efficient
        implementations (e.g. ``_FieldBuilderFromAtoms`` uses
        ``build(first_slice, last_slice)`` to avoid intermediate single-slice
        allocations).

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice.
        last_slice : int, optional
            Index of the last slice.
        chunk_size : int or str, optional
            Number of slices per chunk. ``"auto"`` selects based on the
            configured memory budget (``dask.chunk-size`` on CPU,
            ``dask.chunk-size-gpu`` on GPU). Can also be set globally via the
            ``potential.slice-chunk-size`` configuration key.

        Yields
        ------
        PotentialArray
            A chunk of contiguous potential slices with correctly assigned
            exit planes.
        """
        from abtem.core.chunks import (
            estimate_potential_chunk_size,
            generate_chunks,
        )

        if last_slice is None:
            last_slice = len(self)

        if chunk_size == "auto":
            chunk_size = estimate_potential_chunk_size(
                self.gpts, self.device
            )

        # Cap so the whole range is one chunk when it fits in the budget,
        # then distribute evenly so the last chunk is never smaller than
        # necessary (equal_sized_chunks inside generate_chunks handles this).
        chunk_size = min(chunk_size, last_slice - first_slice)

        xp = get_array_module(self.device)
        exit_plane_after = self._exit_plane_after

        for chunk_start, chunk_end in generate_chunks(
            last_slice - first_slice, chunks=chunk_size, start=first_slice
        ):
            arrays = []
            slice_thicknesses = []
            for slic in self.generate_slices(chunk_start, chunk_end):
                arrays.append(slic.array)
                slice_thicknesses.extend(slic.slice_thickness)

            array = xp.concatenate(arrays, axis=0)
            exit_planes = tuple(
                np.where(exit_plane_after[chunk_start:chunk_end])[0]
            )

            chunk = PotentialArray(
                array,
                slice_thickness=tuple(slice_thicknesses),
                extent=self.extent,
            )
            chunk._exit_planes = exit_planes
            yield chunk

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
        potential.grid.match(waves)

    return potential


def _validate_exit_planes(exit_planes, num_slices):
    if isinstance(exit_planes, int):
        if exit_planes >= num_slices:
            return (num_slices - 1,)

        exit_planes = list(range(exit_planes - 1, num_slices, exit_planes))
        if exit_planes[-1] != (num_slices - 1):
            exit_planes.append(num_slices - 1)
        exit_planes = (-1,) + tuple(exit_planes)
    elif exit_planes is None:
        exit_planes = (num_slices - 1,)

    return exit_planes


def _require_cell_transform(cell, box, plane, origin):
    if box == tuple(np.diag(cell)):
        return False

    if not is_cell_orthogonal(cell):
        return True

    if box is not None:
        return True

    if plane != "xy":
        return True

    if origin != (0.0, 0.0, 0.0):
        return True

    return False


class _FieldBuilder(BaseField):
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
                and box is None
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
                axes = plane_to_axes(plane)
                cell = np.array(cell)[:, list(axes)]
                box = tuple(best_orthogonal_cell(cell))

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
        return self._origin

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
            chunks = chunks + self.base_shape

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

        output_potential = self._array_object(
            array,
            sampling=self._valid_sampling,
            slice_thickness=self.slice_thickness[first_slice:last_slice],
            exit_planes=self.exit_planes,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            cell=self.grid.cell,
        )
        return output_potential


class _FieldBuilderFromAtoms(_FieldBuilder):
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
        atoms = self.frozen_phonons.atoms

        if getattr(self, "_non_orthogonal", False):
            # keep the non-orthogonal cell; the grid carries the skewed metric
            return atoms

        if is_cell_orthogonal(atoms.cell) and self.plane != "xy":
            atoms = rotate_atoms_to_plane(atoms, self.plane)

        elif tuple(np.diag(atoms.cell)) != self.box:
            if self.periodic:
                atoms = orthogonalize_cell(
                    atoms,
                    box=self.box,
                    plane=self.plane,
                    origin=self.origin,
                    return_transform=False,
                    allow_transform=True,
                )
                return atoms
            else:
                cutoffs = self._cutoffs()
                atoms = cut_cell(
                    atoms,
                    cell=self.box,
                    plane=self.plane,
                    origin=self.origin,
                    margin=max(cutoffs) if cutoffs else 0.0,
                )

        return atoms

    def _prepare_atoms(self):
        atoms = self.get_transformed_atoms()

        if self.integrator.finite:
            cutoffs = self._cutoffs()
            margins = max(cutoffs) if len(cutoffs) else 0.0
        else:
            margins = 0.0

        if self.periodic:
            atoms = self.frozen_phonons.randomize(atoms)
            atoms.wrap(eps=0.0)
            # wrap(eps=0.0) uses strict modulo: z positions that are tiny-negative
            # (floating-point artifact from ASE surface builders) become z ≈ cell_z
            # instead of z = 0.  The SliceIndexedAtoms bin edges are nudged down by
            # 1e-12 to fix cumsum drift, so any atom in (cell_z-1e-12, cell_z) falls
            # outside all bins and is silently dropped.  Snap those back to 0.
            cell_z = atoms.cell[2, 2]
            atoms.positions[atoms.positions[:, 2] > cell_z - 1e-10, 2] = 0.0

            if not getattr(self, "_non_orthogonal", False):
                # Same issue for x and y on the orthogonal (or legacy-orthogonalised)
                # path: orthogonalize_cell can produce -0.0 or tiny-negative values
                # from matrix multiplication. wrap(eps=0.0) maps -ε to L-ε rather than
                # 0, placing the atom's FFT peak at the wrong position. (The
                # skew-native path below has its own equivalent fix, since a diagonal
                # atoms.cell[ax, ax] is not the right boundary for a skewed cell.)
                for ax in (0, 1):
                    L = atoms.cell[ax, ax]
                    atoms.positions[atoms.positions[:, ax] > L - 1e-10, ax] = 0.0
                    atoms.positions[np.abs(atoms.positions[:, ax]) < 1e-10, ax] = 0.0

            if getattr(self, "_non_orthogonal", False):
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

        if not self.integrator.periodic and self.integrator.finite:
            atoms = pad_atoms(atoms, margins=margins)
        elif self.integrator.periodic:
            atoms = pad_atoms(atoms, margins=margins, directions="z")

        if not self.periodic:
            atoms = self.frozen_phonons.randomize(atoms)

        if self.integrator.finite:
            sliced_atoms = SlicedAtoms(
                atoms=atoms, slice_thickness=self.slice_thickness, z_padding=margins
            )
        else:
            sliced_atoms = SliceIndexedAtoms(
                atoms=atoms, slice_thickness=self.slice_thickness
            )

        return sliced_atoms

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
        slices : generator of np.ndarray
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

    def generate_chunked_slices(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        chunk_size: int | str = "auto",
    ):
        """
        Generate potential slices in memory-budgeted chunks.

        Overrides the base class to use ``build(first_slice, last_slice,
        lazy=False)`` for each chunk range. This eagerly computes a contiguous
        block of slices directly into a single allocation, avoiding the
        overhead of building and stacking individual slices. Each chunk is
        discarded by the caller after propagation, so only one chunk needs
        to reside in memory (or VRAM) at a time.

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
        PotentialArray
            An eagerly computed chunk of contiguous potential slices.
        """
        from abtem.core.chunks import (
            estimate_potential_chunk_size,
            generate_chunks,
        )

        if last_slice is None:
            last_slice = len(self)

        if chunk_size == "auto":
            chunk_size = estimate_potential_chunk_size(
                self.gpts, self.device
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

            # Remap exit planes to chunk-local indices (build() sets the
            # full potential's exit_planes which are global indices).
            chunk._exit_planes = tuple(
                np.where(exit_plane_after[chunk_start:chunk_end])[0]
            )

            yield chunk

    @property
    def ensemble_axes_metadata(self):
        return self.frozen_phonons.ensemble_axes_metadata

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return self.frozen_phonons.ensemble_shape

    @classmethod
    def _from_partitioned_args_func(cls, *args, frozen_phonons_partial, **kwargs):
        args = unpack_blockwise_args(args)

        frozen_phonons = frozen_phonons_partial(*args)
        frozen_phonons = frozen_phonons.item()

        new_potential = cls(frozen_phonons, **kwargs)

        ndims = max(len(new_potential.ensemble_shape), 1)
        new_potential = _wrap_with_array(new_potential, ndims)
        return new_potential

    def _from_partitioned_args(self, *args, **kwargs):
        frozen_phonons_partial = self.frozen_phonons._from_partitioned_args()
        kwargs = self._copy_kwargs(exclude=("atoms", "sampling"))

        return partial(
            self._from_partitioned_args_func,
            frozen_phonons_partial=frozen_phonons_partial,
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
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the
        slice indices after which an exit plane is desired, and hence during a
        multislice simulation a measurement is created. If `exit_planes` is an integer
        a measurement will be collected every `exit_planes` number of slices.
    plane : str or two tuples of three float, optional
        The plane relative to the provided atoms mapped to `xy` plane of the potential,
        i.e. provided plane is perpendicular to the propagation direction. If string,
        it must be a concatenation of two of 'x', 'y' and 'z'; the default value 'xy'
        indicates that potential slices are cuts along the `xy`-plane of the atoms.
        The plane may also be specified with two arbitrary 3D vectors, which are mapped
        to the `x` and `y` directions of the potential, respectively. The length of the
        vectors has no influence. If the vectors are not perpendicular, the second
        vector is rotated in the plane to become perpendicular to the first.
        Providing a value of ((1., 0., 0.), (0., 1., 0.)) is equivalent to providing
        'xy'.
    origin : three float, optional
        The origin relative to the provided atoms mapped to the origin of the potential.
        This is equivalent to translating the atoms. The default is (0., 0., 0.).
    box : three float, optional
        The extent of the potential in `x`, `y` and `z`. If not given this is determined
        from the atoms' cell. If the box size does not match an integer number of the
        atoms' supercell, an affine transformation may be necessary to preserve
        periodicity, determined by the `periodic` keyword.
    periodic : bool, True
        If a transformation of the atomic structure is required, `periodic` determines
        how the atomic structure is transformed. If True, the periodicity of the Atoms
        is preserved, which may require applying a small affine transformation to the
        atoms. If False, the transformed potential is effectively cut out of a larger
        repeated potential, which may not preserve periodicity.
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

        if sampling == "auto":
            if gpts is not None:
                raise ValueError("Cannot specify both gpts and sampling='auto'")
            cell = np.array(atoms_obj.cell, dtype=float)
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

                if box is not None:
                    extent = box[:2]
                else:
                    extent = tuple(np.linalg.norm(cell[:2, :2], axis=1))
                gpts = tuple(int(np.ceil(extent[i] / 0.05)) for i in range(2))
                if round_auto_derived_gpts():
                    gpts = tuple(next_fast_fft_size(n) for n in gpts)
            else:
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
                    axes = plane_to_axes(plane)
                    cell_2d = cell[:, list(axes)]
                    auto_box = tuple(best_orthogonal_cell(cell_2d))
                    extent = auto_box[:2]
                    # Transform atoms to orthogonal cell so positions match the extent
                    _auto_atoms = orthogonalize_cell(
                        atoms_obj,
                        box=auto_box,
                        plane=plane,
                        origin=origin,
                        return_transform=False,
                        allow_transform=True,
                    )
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
                # Periodic in z: align slice boundaries with crystal planes.
                slice_thickness = commensurate_slice_thickness(
                    atoms_obj, target_thickness=1.0
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
            extent=extent, gpts=array.shape[-2:], sampling=sampling, cell=cell
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

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the generated potential.
        last_slice : int, optional
            Index of the last slice of the generated potential.

        Yields
        ------
        slices : generator of np.ndarray
            Generator for the array of slices.
        """
        if last_slice is None:
            last_slice = len(self)

        exit_plane_after = self._exit_plane_after
        # cum_thickness = np.cumsum(self.slice_thickness)
        start = first_slice
        stop = first_slice + 1

        for i in range(first_slice, last_slice):
            s = (0,) * (len(self.array.shape) - 3) + (i,)
            array = self.array[s][None]

            slic = self.__class__(
                array, self.slice_thickness[i : i + 1], extent=self.extent
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
        PotentialArray
            A view into the existing array covering a chunk of slices.
        """
        from abtem.core.chunks import (
            estimate_potential_chunk_size,
            generate_chunks,
        )

        if last_slice is None:
            last_slice = len(self)

        if chunk_size == "auto":
            chunk_size = estimate_potential_chunk_size(
                self.gpts, self.device
            )

        # Cap so the whole range is one chunk when it fits in the budget,
        # then distribute evenly so the last chunk is never smaller than
        # necessary (equal_sized_chunks inside generate_chunks handles this).
        chunk_size = min(chunk_size, last_slice - first_slice)

        exit_plane_after = self._exit_plane_after

        for chunk_start, chunk_end in generate_chunks(
            last_slice - first_slice, chunks=chunk_size, start=first_slice
        ):
            s = (0,) * (len(self.array.shape) - 3) + (
                slice(chunk_start, chunk_end),
            )
            chunk_array = self.array[s]

            exit_planes = tuple(
                np.where(exit_plane_after[chunk_start:chunk_end])[0]
            )

            chunk = self.__class__(
                chunk_array,
                slice_thickness=self.slice_thickness[chunk_start:chunk_end],
                extent=self.extent,
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


class PotentialArray(BasePotential, FieldArray):
    """
    The potential array represents slices of the electrostatic potential as an array.
    All other potentials build potential arrays.

    Parameters
    ----------
    array: 3D np.ndarray
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
        multislice simulation a measurement is created. If `exit_planes` is an integer a
        measurement will be collected every `exit_planes` number of slices.
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
                meta=xp.array((), dtype=get_dtype(complex=True)),
            )
        else:
            array = self._transmission_function(self.array, energy=energy)

        t = TransmissionFunction(
            array,
            slice_thickness=self.slice_thickness,
            extent=self.extent,
            energy=energy,
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
    array : 3D np.ndarray
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
    """

    def __init__(
        self,
        array: np.ndarray,
        slice_thickness: float | Sequence[float],
        extent: Optional[float | tuple[float, float]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        energy: Optional[float] = None,
        cell: Optional[np.ndarray] = None,
    ):
        self._accelerator = Accelerator(energy=energy)
        super().__init__(array, slice_thickness, extent, sampling, cell=cell)

    def get_chunk(self, first_slice, last_slice) -> TransmissionFunction:
        array = self.array[first_slice:last_slice]
        if len(array.shape) == 2:
            array = array[None]
        return self.__class__(
            array,
            self.slice_thickness[first_slice:last_slice],
            extent=self.extent,
            energy=self.energy,
            cell=self.cell,
        )

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

        xp = get_array_module(self.array[0])

        if conjugate:
            waves._array *= xp.conjugate(self.array[0])
        else:
            waves._array *= self.array[0]

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
        displacement snapshots.
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the
        slice indices after which an exit plane is desired, and hence during a
        multislice simulation a measurement is created. If `exit_planes` is an integer
        a measurement will be collected every `exit_planes` number of slices.
    seeds: int or sequence of int
        Seed for the random number generator (RNG), or one seed for each RNG in the
        frozen phonon ensemble.
    ensemble_mean : bool, optional
        If True (default), the mean over the frozen-phonon ensemble is calculated.
        If False, the individual configurations are returned.
    """

    def __init__(
        self,
        potential_unit: BasePotential,
        repetitions: tuple[int, int, int],
        num_frozen_phonons: int | None = None,
        exit_planes: int | None = None,
        seeds: int | tuple[int, ...] | None = None,
        ensemble_mean: bool = True,
    ):
        if num_frozen_phonons is None and seeds is None:
            self._seeds = None
        else:
            if num_frozen_phonons is None and seeds:
                assert isinstance(seeds, tuple)
                num_frozen_phonons = len(seeds)
            elif num_frozen_phonons is None and seeds is None:
                num_frozen_phonons = 1

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
    def gpts(self, gpts: tuple[int, int]):
        if not (
            (gpts[0] % self.repetitions[0] == 0)
            and (gpts[1] % self.repetitions[0] == 0)
        ):
            raise ValueError(
                "Number of grid points must be divisible by the number of potential"
                "repetitions."
            )
        self.grid.gpts = gpts
        self._potential_unit.gpts = (
            gpts[0] // self._repetitions[0],
            gpts[1] // self._repetitions[1],
        )

    @property
    def sampling(self) -> tuple[float, float] | None:
        return super().sampling

    @sampling.setter
    def sampling(self, sampling: tuple[float, float]):
        self.sampling = sampling
        self._potential_unit.sampling = sampling

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

        The atoms are reconstructed by tiling the unit potential's transformed
        (orthogonalised) atoms by the crystal repetitions. This makes
        ``CrystalPotential`` work with any code path that derives atomic sites
        from a potential via ``get_sliced_atoms`` -- e.g. the core-loss EELS
        driver's automatic site extraction -- without special-casing the
        repeating-unit structure.

        Notes
        -----
        - **Frozen phonons are not displaced.** ``get_transformed_atoms``
          returns the equilibrium (mean) positions, so the returned sites are
          the un-displaced atomic columns. This is deliberate: a
          ``CrystalPotential`` ensemble draws an independent random unit
          configuration per z-repetition, so there is no single displaced
          realisation to return, and atomic-column site identification (the
          main consumer) wants the equilibrium column positions anyway. This
          differs from ``Potential.get_sliced_atoms``, which applies the
          frozen-phonon displacement of its single configuration.
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

        unit_atoms = self._potential_unit.get_transformed_atoms()
        tiled_atoms = unit_atoms * self._repetitions

        self._sliced_atoms = SliceIndexedAtoms(
            tiled_atoms, slice_thickness=self.slice_thickness
        )

        return self._sliced_atoms

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        args = unpack_blockwise_args(args)
        potential, seed = args[0]
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

        if lazy:
            arrays = []

            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                if self.seeds is not None:
                    seeds = self.seeds[start:stop]
                else:
                    seeds = None

                lazy_atoms = dask.delayed(self.potential_unit)
                lazy_args = dask.delayed(_wrap_with_array)((lazy_atoms, seeds), ndims=1)
                lazy_array = da.from_delayed(lazy_args, shape=(1,), dtype=object)
                arrays.append(lazy_array)

            array = da.concatenate(arrays)

            if old_chunks == ():
                array = array[0]

        else:
            potential_unit = self.potential_unit

            array = np.zeros((len(chunks[0]),), dtype=object)
            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                if self.seeds is not None:
                    seeds = self.seeds[start:stop]
                else:
                    seeds = None

                itemset(array, i, (potential_unit, seeds))

            if old_chunks == ():
                array = _wrap_with_array(array[0], ndims=0)

        return (array,)

    @property
    def _n_lateral_tiles(self) -> int:
        return self.repetitions[0] * self.repetitions[1]

    def _pool_unit_for_member(self, member_seed: Optional[int]) -> BasePotential:
        """Return the unit potential to draw pool configurations from for one
        ensemble member (``member_seed`` is that member's seed), or for the
        single default builder (``member_seed`` is None).

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
        unit = self.potential_unit
        n_tiles = self._n_lateral_tiles
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
        kwargs = unit._copy_kwargs(exclude=("atoms",))
        return type(unit)(new_fp, **kwargs)

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
        slices : generator of np.ndarray
            Generator for the array of slices.
        """
        # if hasattr(self.potential_unit, "array")
        #    potentials = self.potential_unit
        member_seed = None if self.seeds is None else int(self.seeds[0])
        pool_unit = self._pool_unit_for_member(member_seed)
        if not isinstance(pool_unit, PotentialArray):
            potentials = pool_unit.build(lazy=False)
        else:
            potentials = pool_unit

        assert isinstance(potentials, PotentialArray)

        if len(potentials.shape) == 3:
            potentials = potentials.expand_dims(axis=0)

        rng = np.random.default_rng(member_seed)

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
        # tiled-unit byte size and freed when the generator is exhausted.
        tiled_cache: dict[tuple[int, int], PotentialArray] = {}
        unit_generators: dict[int, object] = {}
        tile_xy = self.repetitions[:2]

        n_configs = potentials.shape[0]

        # The mosaic path (frozen-phonon pools, n_configs > 1) needs random
        # per-tile access into the pool, so materialise the (small unit-cell)
        # pool array once. A lazily-built PotentialArray unit carries a dask
        # array here; compute it so per-tile fancy indexing works and stays on
        # the target device.
        xp = get_array_module(self.device)
        _pool_array = potentials.array
        if n_configs > 1 and hasattr(_pool_array, "compute"):
            _pool_array = _pool_array.compute()

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
        if n_configs > 1:
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

        for i in range(self.repetitions[2]):
            # Draw an independent displaced realisation per unit cell in the x,
            # y and z supercell directions. For a single-config pool this
            # collapses to the cheap cached ``.tile()`` path below.
            # Always draw (even for z-repetitions skipped below) so the
            # frozen-phonon sequence and the pool budget accounting stay
            # consistent regardless of first_slice -- different chunks of the
            # same crystal must see the same per-layer configuration draws.
            if n_configs > 1:
                config_tiles = _draw_config_tiles()
            else:
                config_tiles = None

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

        Unlike the base-class implementation, this override builds the unit
        potential **once** (not once per chunk) and fills each output chunk
        array in-place, slice by slice, using ``xp.tile``.  This avoids the
        ~2× peak-memory spike that the base class incurs from accumulating
        per-slice tiled arrays into a list before concatenating them.

        The dtype of the output follows the unit potential's array dtype,
        which is set by the abtem ``precision`` config key (float32 / float64).
        """
        from abtem.core.chunks import estimate_potential_chunk_size, generate_chunks

        if last_slice is None:
            last_slice = len(self)

        if chunk_size == "auto":
            chunk_size = estimate_potential_chunk_size(self.gpts, self.device)
        chunk_size = min(chunk_size, last_slice - first_slice)

        xp = get_array_module(self.device)
        exit_plane_after = self._exit_plane_after

        # Build the unit potential once; the base class would re-build it on
        # every chunk (one generate_slices() call per chunk).
        if not isinstance(self.potential_unit, PotentialArray):
            unit_built = self.potential_unit.build(lazy=False)
        else:
            unit_built = self.potential_unit

        unit_arr = unit_built.array  # (n_unit_slices, h, w) or (n_configs, n_unit_slices, h, w)
        if unit_arr.ndim == 3:
            unit_arr = unit_arr[np.newaxis]  # → (1, n_unit_slices, h, w)

        rng = np.random.default_rng(self.seeds[0] if self.seeds is not None else None)
        unit_slices = len(self.potential_unit)
        n_configs = unit_arr.shape[0]

        # Pre-draw frozen-phonon config indices — one per z-repetition —
        # to match the sequence that generate_slices() would produce.
        config_indices = rng.integers(0, n_configs, size=self.repetitions[2])

        unit_st = self.potential_unit.slice_thickness

        for chunk_start, chunk_end in generate_chunks(
            last_slice - first_slice, chunks=chunk_size, start=first_slice
        ):
            n = chunk_end - chunk_start
            out = None
            slice_thicknesses = []

            for k, global_idx in enumerate(range(chunk_start, chunk_end)):
                rep_i, unit_j = divmod(global_idx, unit_slices)
                slc = unit_arr[config_indices[rep_i], unit_j]   # (h, w)
                tiled = xp.tile(slc, self.repetitions[:2])       # (full_h, full_w)

                if out is None:
                    out = xp.empty((n,) + tiled.shape, dtype=tiled.dtype)
                out[k] = tiled
                slice_thicknesses.append(unit_st[unit_j])

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
