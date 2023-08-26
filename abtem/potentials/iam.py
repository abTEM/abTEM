"""Module for describing electrostatic potentials using the independent atom model."""
from __future__ import annotations

import warnings
from functools import partial
from typing import Sequence, TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
from ase import Atoms
from ase.cell import Cell

from abtem.core.axes import (
    RealSpaceAxis,
)
from abtem.core.axes import ThicknessAxis, FrozenPhononsAxis, AxisMetadata
from abtem.core.backend import get_array_module
from abtem.core.chunks import chunk_ranges
from abtem.core.chunks import validate_chunks
from abtem.core.complex import complex_exponential
from abtem.core.energy import HasAcceleratorMixin, Accelerator, energy2sigma
from abtem.core.ensemble import _wrap_with_array, unpack_blockwise_args
from abtem.core.grid import HasGridMixin
from abtem.fields import BaseField, _FieldBuilderFromAtoms, FieldArray
from abtem.inelastic.phonons import (
    BaseFrozenPhonons,
    _validate_seeds,
)
from abtem.integrals import ScatteringFactorProjectionIntegrals, QuadratureProjectionIntegrals

if TYPE_CHECKING:
    from abtem.waves import Waves, BaseWaves
    from abtem.parametrizations import Parametrization
    from abtem.integrals import FieldIntegrator

class BasePotential(BaseField):
    """Base class of all potentials. Documented in the subclasses."""

    @property
    def base_shape(self):
        """Shape of the base axes of the potential."""
        return (self.num_slices,) + self.gpts

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


def _validate_potential(
    potential: Atoms | BasePotential, waves: BaseWaves = None
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


class _PotentialBuilder(BasePotential):
    pass


class Potential(_FieldBuilderFromAtoms, BasePotential):
    """
    Calculate the electrostatic potential of a set of atoms or frozen phonon configurations. The potential is calculated
    with the Independent Atom Model (IAM) using a user-defined parametrization of the atomic potentials.

    Parameters
    ----------
    atoms : ase.Atoms or abtem.FrozenPhonons
        Atoms or FrozenPhonons defining the atomic configuration(s) used in the independent atom model for calculating
        the electrostatic potential(s).
    gpts : one or two int, optional
        Number of grid points in `x` and `y` describing each slice of the potential. Provide either "sampling" (spacing
        between consecutive grid points) or "gpts" (total number of grid points).
    sampling : one or two float, optional
        Sampling of the potential in `x` and `y` [1 / Å]. Provide either "sampling" or "gpts".
    slice_thickness : float or sequence of float, optional
        Thickness of the potential slices in the propagation direction in [Å] (default is 0.5 Å).
        If given as a float, the number of slices is calculated by dividing the slice thickness into the `z`-height of
        supercell. The slice thickness may be given as a sequence of values for each slice, in which case an error will
        be thrown if the sum of slice thicknesses is not equal to the height of the atoms.
    parametrization : 'lobato' or 'kirkland', optional
        The potential parametrization describes the radial dependence of the potential for each element. Two of the
        most accurate parametrizations are available (by Lobato et al. and Kirkland; default is 'lobato').
        See the citation guide for references.
    projection : 'finite' or 'infinite', optional
        If 'finite' the 3D potential is numerically integrated between the slice boundaries. If 'infinite' (default),
        the infinite potential projection of each atom will be assigned to a single slice.
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the slice indices after which an
        exit plane is desired, and hence during a multislice simulation a measurement is created. If `exit_planes` is
        an integer a measurement will be collected every `exit_planes` number of slices.
    plane : str or two tuples of three float, optional
        The plane relative to the provided atoms mapped to `xy` plane of the potential, i.e. provided plane is
        perpendicular to the propagation direction. If string, it must be a concatenation of two of 'x', 'y' and 'z';
        the default value 'xy' indicates that potential slices are cuts along the `xy`-plane of the atoms.
        The plane may also be specified with two arbitrary 3D vectors, which are mapped to the `x` and `y` directions of
        the potential, respectively. The length of the vectors has no influence. If the vectors are not perpendicular,
        the second vector is rotated in the plane to become perpendicular to the first. Providing a value of
        ((1., 0., 0.), (0., 1., 0.)) is equivalent to providing 'xy'.
    origin : three float, optional
        The origin relative to the provided atoms mapped to the origin of the potential. This is equivalent to
        translating the atoms. The default is (0., 0., 0.).
    box : three float, optional
        The extent of the potential in `x`, `y` and `z`. If not given this is determined from the atoms' cell.
        If the box size does not match an integer number of the atoms' supercell, an affine transformation may be
        necessary to preserve periodicity, determined by the `periodic` keyword.
    periodic : bool, True
        If a transformation of the atomic structure is required, `periodic` determines how the atomic structure is
        transformed. If True, the periodicity of the Atoms is preserved, which may require applying a small affine
        transformation to the atoms. If False, the transformed potential is effectively cut out of a larger repeated
        potential, which may not preserve periodicity.
    integrator : ProjectionIntegrator, optional
        Provide a custom integrator for the projection integrals of the potential slicing.
    device : str, optional
        The device used for calculating the potential, 'cpu' or 'gpu'. The default is determined by the user
        configuration file.
    """

    _exclude_from_copy = ("parametrization", "projection")

    def __init__(
        self,
        atoms: Atoms | BaseFrozenPhonons = None,
        gpts: int | tuple[int, int] = None,
        sampling: float | tuple[float, float] = None,
        slice_thickness: float | tuple[float, ...] = 1,
        parametrization: str | Parametrization = "lobato",
        projection: str = "infinite",
        exit_planes: int | tuple[int, ...] = None,
        plane: str
        | tuple[tuple[float, float, float], tuple[float, float, float]] = "xy",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: tuple[float, float, float] = None,
        periodic: bool = True,
        integrator: FieldIntegrator = None,
        device: str = None,
    ):

        if integrator is None:
            if projection == "finite":
                integrator = QuadratureProjectionIntegrals(parametrization=parametrization)
            elif projection == "infinite":
                integrator = ScatteringFactorProjectionIntegrals(parametrization=parametrization)
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
        )


class PotentialArray(BasePotential, FieldArray):
    """
    The potential array represents slices of the electrostatic potential as an array. All other potentials build
    potential arrays.

    Parameters
    ----------
    array: 3D np.ndarray
        The array representing the potential slices. The first dimension is the slice index and the last two are the
        spatial dimensions.
    slice_thickness: float
        The thicknesses of potential slices [Å]. If a float, the thickness is the same for all slices.
        If a sequence, the length must equal the length of the potential array.
    extent: one or two float, optional
        Lateral extent of the potential [Å].
    sampling: one or two float, optional
        Lateral sampling of the potential [1 / Å].
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the slice indices after which an
        exit plane is desired, and hence during a multislice simulation a measurement is created. If `exit_planes` is
        an integer a measurement will be collected every `exit_planes` number of slices.
    ensemble_axes_metadata : list of AxesMetadata
        Axis metadata for each ensemble axis. The axis metadata must be compatible with the shape of the array.
    metadata : dict
        A dictionary defining wave function metadata. All items will be added to the metadata of measurements derived
        from the waves.
    """

    _base_dims = 3

    def __init__(
        self,
        array: np.ndarray | da.core.Array,
        slice_thickness: float | tuple[float, ...] = None,
        extent: float | tuple[float, float] = None,
        sampling: float | tuple[float, float] = None,
        exit_planes: int | tuple[int, ...] = None,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        metadata: dict = None,
    ):
        super().__init__(
            array=array,
            slice_thickness=slice_thickness,
            extent=extent,
            sampling=sampling,
            exit_planes=exit_planes,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

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

        def _transmission_function(array, energy):
            array = complex_exponential(xp.float32(energy2sigma(energy)) * array)
            return array

        if self.is_lazy:
            array = self._array.map_blocks(
                _transmission_function,
                energy=energy,
                meta=xp.array((), dtype=xp.complex64),
            )
        else:
            array = _transmission_function(self._array, energy=energy)

        t = TransmissionFunction(
            array,
            slice_thickness=self.slice_thickness,
            extent=self.extent,
            energy=energy,
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

        transmission_function = self.transmission_function(waves.energy)

        return transmission_function.transmit(waves, conjugate=conjugate)


class TransmissionFunction(PotentialArray, HasAcceleratorMixin):
    """Class to describe transmission functions.

    Parameters
    ----------
    array : 3D np.ndarray
        The array representing the potential slices. The first dimension is the slice index and the last two are the
        spatial dimensions.
    slice_thickness : float
        The thicknesses of potential slices [Å]. If a float, the thickness is the same for all slices.
        If a sequence, the length must equal the length of the potential array.
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
        extent: float | tuple[float, float] = None,
        sampling: float | tuple[float, float] = None,
        energy: float = None,
    ):

        self._accelerator = Accelerator(energy=energy)
        super().__init__(array, slice_thickness, extent, sampling)

    def get_chunk(self, first_slice, last_slice) -> TransmissionFunction:
        array = self.array[first_slice:last_slice]
        if len(array.shape) == 2:
            array = array[None]
        return self.__class__(
            array,
            self.slice_thickness[first_slice:last_slice],
            extent=self.extent,
            energy=self.energy,
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
    The crystal potential may be used to represent a potential consisting of a repeating unit. This may allow
    calculations to be performed with lower computational cost by calculating the potential unit once and repeating it.

    If the repeating unit is a potential with frozen phonons it is treated as an ensemble from which each repeating
    unit along the `z`-direction is randomly drawn. If `num_frozen_phonons` an ensemble of crystal potentials are created
    each with a random seed for choosing potential units.

    Parameters
    ----------
    potential_unit : BasePotential
        The potential unit to assemble the crystal potential from.
    repetitions : three int
        The repetitions of the potential in `x`, `y` and `z`.
    num_frozen_phonons : int, optional
        Number of frozen phonon configurations assembled from the potential units.
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the slice indices after which an
        exit plane is desired, and hence during a multislice simulation a measurement is created. If `exit_planes` is
        an integer a measurement will be collected every `exit_planes` number of slices.
    seeds: int or sequence of int
        Seed for the random number generator (RNG), or one seed for each RNG in the frozen phonon ensemble.
    """

    def __init__(
        self,
        potential_unit: BasePotential,
        repetitions: tuple[int, int, int],
        num_frozen_phonons: int = None,
        exit_planes: int = None,
        seeds: int | tuple[int, ...] = None,
    ):

        if num_frozen_phonons is None and seeds is None:
            self._seeds = None
        else:
            if num_frozen_phonons is None and seeds:
                num_frozen_phonons = len(seeds)
            elif num_frozen_phonons is None and seeds is None:
                num_frozen_phonons = 1

            self._seeds = _validate_seeds(seeds, num_frozen_phonons)

        if (
            (potential_unit.num_configurations == 1)
            and (num_frozen_phonons is not None)
            and (num_frozen_phonons > 1)
        ):
            warnings.warn(
                "'num_frozen_phonons' is greater than one, but the potential unit does not have frozen phonons"
            )

        # if (potential_unit.num_frozen_phonons > 1) and (num_frozen_phonons is not None):
        #     warnings.warn(
        #         "the potential unit has frozen phonons, but 'num_frozen_phonons' is not set"
        #     )

        gpts = (
            potential_unit.gpts[0] * repetitions[0],
            potential_unit.gpts[1] * repetitions[1],
        )
        extent = (
            potential_unit.extent[0] * repetitions[0],
            potential_unit.extent[1] * repetitions[1],
        )

        box = extent + (potential_unit.thickness * repetitions[2],)
        slice_thickness = potential_unit.slice_thickness * repetitions[2]
        super().__init__(
            gpts=gpts,
            cell=Cell(np.diag(box)),
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=potential_unit.device,
            plane="xy",
            origin=(0.0, 0.0, 0.0),
            box=box,
            periodic=True,
        )

        self._potential_unit = potential_unit
        self._repetitions = repetitions

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

    @HasGridMixin.gpts.setter
    def gpts(self, gpts):
        if not (
            (gpts[0] % self.repetitions[0] == 0)
            and (gpts[1] % self.repetitions[0] == 0)
        ):
            raise ValueError(
                "Number of grid points must be divisible by the number of potential repetitions."
            )
        self.grid.gpts = gpts
        self._potential_unit.gpts = (
            gpts[0] // self._repetitions[0],
            gpts[1] // self._repetitions[1],
        )

    @HasGridMixin.sampling.setter
    def sampling(self, sampling):
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
            return [FrozenPhononsAxis(_ensemble_mean=True)]

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

    def _partition_args(self, chunks: int = 1, lazy: bool = True):

        chunks = validate_chunks(self.ensemble_shape, chunks)

        if chunks == ():
            chunks = ((1,),)

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
        else:

            potential_unit = self.potential_unit
            # if self.potential_unit.array:
            #    atoms = atoms.compute()
            array = np.zeros((len(chunks[0]),), dtype=object)
            for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
                if self.seeds is not None:
                    seeds = self.seeds[start:stop]
                else:
                    seeds = None

                array.itemset(i, (potential_unit, self.seeds))

        return (array,)

        # chunks = validate_chunks(self.ensemble_shape, chunks)
        #
        # if not len(self.ensemble_shape):
        #     chunks = ((1,),)
        #
        # if lazy:
        #     arrays = []
        #     for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
        #         if self.seeds is not None:
        #             seeds = self.seeds[start:stop]
        #         else:
        #             seeds = self.seeds
        #         lazy_potential = self.potential_unit.ensemble_blocks(-1)
        #         lazy_args = dask.delayed(_wrap_with_array)((lazy_potential, seeds), ndims=1)
        #         lazy_array = da.from_delayed(lazy_args, shape=(1,), dtype=object)
        #         arrays.append(lazy_array)
        #
        #     array = da.concatenate(arrays)
        # else:
        #
        #     array = np.zeros((chunks[0],), dtype=object)
        #     for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
        #         if self.seeds is not None:
        #             seeds = self.seeds[start:stop]
        #         else:
        #             seeds = self.seeds
        #
        #         array.itemset(i, (self.potential_unit, seeds))
        # return (array,)

    def generate_slices(
        self, first_slice: int = 0, last_slice: int = None, return_depth: bool = False
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
        if hasattr(self.potential_unit, "array"):
            potentials = self.potential_unit
        else:
            potentials = self.potential_unit.build(lazy=False)

        if len(potentials.shape) == 3:
            potentials = potentials.expand_dims(axis=0)

        if self.seeds is None:
            rng = np.random.default_rng(self.seeds)
        else:
            rng = np.random.default_rng(self.seeds[0])

        exit_plane_after = self._exit_plane_after
        cum_thickness = np.cumsum(self.slice_thickness)
        start = first_slice
        stop = first_slice + 1

        for i in range(self.repetitions[2]):
            generator = potentials[
                rng.integers(0, potentials.shape[0])
            ].generate_slices()

            for i in range(len(self.potential_unit)):
                slic = next(generator).tile(self.repetitions[:2])

                exit_planes = tuple(np.where(exit_plane_after[start:stop])[0])

                slic._exit_planes = exit_planes

                start += 1
                stop += 1

                if return_depth:
                    yield cum_thickness[stop - 1], slic
                else:
                    yield slic
