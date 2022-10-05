from functools import partial
from typing import Union, Tuple, List, TYPE_CHECKING

import dask
import numpy as np
from matplotlib.axes import Axes

from abtem.core.axes import AxisMetadata, PlasmonAxis
from abtem.core.backend import get_array_module
from abtem.core.chunks import validate_chunks, chunk_ranges
import matplotlib.pyplot as plt
import dask.array as da

from abtem.transfer import WaveTransform

if TYPE_CHECKING:
    from abtem.waves import Waves
    from abtem.potentials import BasePotential


def scattering_depths(mean_free_path: float, max_depth: float):
    depths = []
    total_depth = 0.0

    while total_depth <= max_depth:
        depth = -mean_free_path * np.log(np.random.rand())
        total_depth += depth

        if total_depth <= max_depth:
            depths.append(depth)
        else:
            break

    return tuple(np.cumsum(depths))


def draw_scattering_depths(
    mean_free_path: float,
    filter_excitations: Tuple[int, ...],
    max_depth: float,
    max_attempts: int = 1_000_000_000,
) -> Tuple[float, ...]:

    for i in range(max_attempts):
        candidate_excitation_depths = scattering_depths(mean_free_path, max_depth)

        if len(candidate_excitation_depths) in filter_excitations:
            break
    else:
        raise RuntimeError()

    return candidate_excitation_depths


def draw_radial_scattering_angle(
    critical_angle: float, characteristic_angle: float
) -> float:
    return np.sqrt(
        characteristic_angle ** 2
        * (
            (critical_angle ** 2 + characteristic_angle ** 2)
            / characteristic_angle ** 2
        )
        ** np.random.rand()
        - characteristic_angle ** 2
    )


def draw_azimuthal_angle() -> float:
    return 2 * np.pi * np.random.rand()


class PlasmonScatteringEvents(WaveTransform):
    def __init__(
        self,
        depths: Tuple[Tuple[float, ...]],
        radial_angles: Tuple[Tuple[float, ...]],
        azimuthal_angles: Tuple[Tuple[float, ...]],
        ensemble_mean: bool,
    ):
        self._depths = depths
        self._radial_angles = radial_angles
        self._azimuthal_angles = azimuthal_angles
        self._ensemble_mean = ensemble_mean

    @property
    def ensemble_shape(self):
        return (len(self.depths),)

    @property
    def _default_ensemble_chunks(self):
        return ("auto",)

    @property
    def ensemble_mean(self):
        return self._ensemble_mean

    @property
    def depths(self):
        return self._depths

    @property
    def radial_angles(self):
        return self._radial_angles

    @property
    def azimuthal_angles(self):
        return self._azimuthal_angles

    @property
    def num_excitations(self):
        return tuple(len(depths_element) for depths_element in self.depths)

    @property
    def max_excitations(self):
        return max(self.num_excitations)

    def show_excitations_histogram(self, ax: Axes = None):
        bins = range(0, self.max_excitations + 2)
        if ax is None:
            ax = plt.subplot()
        ax.hist(self.num_excitations, bins=bins)
        ax.set_xticks(np.array(bins) + 0.5)
        ax.set_xticklabels(bins)
        ax.set_xlabel("Number of excitations")
        ax.set_ylabel("Number of events")

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return [
            PlasmonAxis(
                values=tuple(
                    (depths, radial_angles, azimuthal_angles)
                    for depths, radial_angles, azimuthal_angles in zip(
                        self.depths, self.radial_angles, self.azimuthal_angles
                    )
                ),
                _ensemble_mean=self.ensemble_mean,
            )
        ]

    @classmethod
    def _from_partitioned_args_func(cls, *args, **kwargs):
        args = args[0]
        if hasattr(args, "item"):
            args = args.item()

        kwargs["depths"] = args["depths"]
        kwargs["radial_angles"] = args["radial_angles"]
        kwargs["azimuthal_angles"] = args["azimuthal_angles"]
        return cls(**kwargs)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(
            exclude=("depths", "radial_angles", "azimuthal_angles")
        )
        return partial(self._from_partitioned_args_func, **kwargs)

    @staticmethod
    def _plasmon_scattering_events(depths, radial_angles, azimuthal_angles):
        arr = np.zeros((1,), dtype=object)
        arr.itemset(
            0,
            {
                "depths": depths,
                "radial_angles": radial_angles,
                "azimuthal_angles": azimuthal_angles,
            },
        )
        return arr

    def _partition_args(self, chunks: int = 1, lazy: bool = True):
        chunks = validate_chunks(self.ensemble_shape, chunks)

        array = np.zeros((len(chunks[0]),), dtype=object)
        for i, (start, stop) in enumerate(chunk_ranges(chunks)[0]):
            depths = self.depths[start:stop]
            radial_angles = self.radial_angles[start:stop]
            azimuthal_angles = self.azimuthal_angles[start:stop]

            if lazy:
                lazy_frozen_phonon = dask.delayed(self._plasmon_scattering_events)(
                    depths=depths,
                    radial_angles=radial_angles,
                    azimuthal_angles=azimuthal_angles,
                )
                array.itemset(
                    i, da.from_delayed(lazy_frozen_phonon, shape=(1,), dtype=object)
                )
            else:
                array.itemset(
                    i,
                    self._plasmon_scattering_events(
                        depths=depths,
                        radial_angles=radial_angles,
                        azimuthal_angles=azimuthal_angles,
                    ),
                )

        if lazy:
            array = da.concatenate(list(array))

        return (array,)

    def apply(self, waves: "Waves") -> "Waves":
        xp = get_array_module(waves.device)

        array = waves.array[(None,) * len(self.ensemble_shape)]

        if waves.is_lazy:
            array = da.tile(array, self.ensemble_shape + (1,) * len(waves.shape))
        else:
            array = xp.tile(array, self.ensemble_shape + (1,) * len(waves.shape))

        kwargs = waves._copy_kwargs(exclude=("array",))
        kwargs["array"] = array
        kwargs["ensemble_axes_metadata"] = (
            self.ensemble_axes_metadata + kwargs["ensemble_axes_metadata"]
        )
        return waves.__class__(**kwargs)


class MonteCarloPlasmons:
    def __init__(
        self,
        mean_free_path: float,
        excitation_energy: float,
        critical_angle: float,
        num_samples: int = None,
        ensemble_mean: bool = False,
        filter_excitations: Union[int, Tuple[int, ...]] = None,
        seed: Union[int, Tuple[int, ...]] = None,
    ):
        self._mean_free_path = mean_free_path
        self._excitation_energy = excitation_energy
        self._critical_angle = critical_angle
        self._ensemble_mean = ensemble_mean
        self._num_samples = num_samples
        self._seed = seed

        if isinstance(filter_excitations, int):
            filter_excitations = tuple(range(filter_excitations))

        if filter_excitations is None:
            raise ValueError()

        self._filter_excitations = filter_excitations

    @property
    def ensemble_mean(self) -> bool:
        return self._ensemble_mean

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def mean_free_path(self) -> float:
        return self._mean_free_path

    @property
    def seed(self) -> int:
        return self._seed

    def __len__(self) -> int:
        return self.num_samples

    def characteristic_angle(self, energy: float) -> float:
        return self._excitation_energy / (2 * energy)

    def draw_events(
        self, waves: "Waves", potential: "BasePotential"
    ) -> PlasmonScatteringEvents:
        depth = potential.thickness
        energy = waves.energy

        depths = tuple(
            draw_scattering_depths(
                mean_free_path=self._mean_free_path,
                filter_excitations=self._filter_excitations,
                max_depth=depth,
            )
            for _ in range(self.num_samples)
        )

        radial_angles = tuple(
            tuple(
                draw_radial_scattering_angle(
                    self._critical_angle / 1e-3, self.characteristic_angle(energy)
                )
                for _ in range(len(depths[i]))
            )
            for i in range(self.num_samples)
        )

        azimuthal_angles = tuple(
            tuple(draw_azimuthal_angle() for _ in range(len(depths[i])))
            for i in range(self.num_samples)
        )

        return PlasmonScatteringEvents(
            depths, radial_angles, azimuthal_angles, ensemble_mean=self.ensemble_mean
        )
