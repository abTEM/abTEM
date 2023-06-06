import itertools
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Union, Tuple, List, TYPE_CHECKING

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from abtem.core.axes import (
    AxisMetadata,
    OrdinalAxis,
    _iterate_axes_type,
)
from abtem.core.backend import get_array_module
from abtem.core.chunks import validate_chunks, chunk_ranges
from abtem.transform import ArrayObjectTransform

if TYPE_CHECKING:
    from abtem.waves import Waves
    from abtem.potentials import BasePotential


nth = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
ntuples = {
    0: "Zero loss",
    1: "Single plasmon",
    2: "Double plasmon",
    3: "Triple plasmon",
    4: "Quadruple plasmon",
    5: "Quintuple plasmon",
    6: "Sextuple plasmon",
    7: "Septuple plasmon",
    8: "Octuple plasmon",
    9: "Nonuble plasmon",
}


def draw_scattering_depths(
    num_depths: int,
    num_samples: int,
    mean_free_path: float,
    max_depth: float,
    max_batch: int = 10_000,
    max_attempts: int = 50_000_000,
    rng=None,
) -> Tuple[Tuple]:

    if rng is None:
        rng = np.random.default_rng()

    if num_depths == 0:
        return ((),) * num_samples  # noqa

    max_num_batches = max_attempts // max_batch

    depths = np.zeros((num_samples, num_depths))
    k = 0
    for i in range(max_num_batches):
        new_depths = np.cumsum(
            -mean_free_path * np.log(rng.random((max_batch, num_depths + 1))), axis=-1
        )
        new_depths = new_depths[
            (new_depths[:, -1] > max_depth) * (new_depths[:, -2] < max_depth)
        ]
        new_k = min(num_samples, k + len(new_depths))
        depths[k:new_k] = new_depths[: new_k - k, :num_depths]

        k = new_k
        if k == num_samples:
            break

    if k != num_samples:
        raise ValueError(
            f"requested scattering events did not occur in {max_attempts} attempts"
        )

    return tuple(tuple(d) for d in depths)


def draw_radial_scattering_angle(
    critical_angle: float,
    characteristic_angle: float,
    num_samples,
    num_depths,
    rng=None,
) -> Tuple[Tuple[float]]:

    if rng is None:
        rng = np.random.default_rng()

    radial_scattering_angles = []
    for _ in range(num_samples):
        radial_scattering_angles.append(
            tuple(
                np.sqrt(
                    characteristic_angle**2
                    * (
                        (critical_angle**2 + characteristic_angle**2)
                        / characteristic_angle**2
                    )
                    ** rng.random()
                    - characteristic_angle**2
                )
                for _ in range(num_depths)
            )
        )

    return tuple(radial_scattering_angles)


def draw_azimuthal_angle(num_samples, num_depths, rng=None) -> Tuple[float]:

    if rng is None:
        rng = np.random.default_rng()

    azimuthal_angles = []
    for _ in range(num_samples):
        azimuthal_angles.append(
            tuple(2 * np.pi * rng.random() for _ in range(num_depths))
        )

    return tuple(azimuthal_angles)


def excitations_weights(n: int, thickness: float, mean_free_path: float) -> float:
    return (
        1
        / np.math.factorial(n)
        * (thickness / mean_free_path) ** n
        * np.exp(-thickness / mean_free_path)
    )


@dataclass(eq=False, repr=False, unsafe_hash=True)
class PlasmonAxis(OrdinalAxis):
    units: str = ""
    label: str = "Plasmons excitations"
    _ensemble_mean: bool = False

    @property
    def excitations(self):
        return tuple(value[3] for value in self.values)

    @property
    def azimuthal_angles(self):
        return tuple(value[2] for value in self.values)

    @property
    def radial_angles(self):
        return tuple(value[1] for value in self.values)

    @property
    def depths(self):
        return tuple(value[0] for value in self.values)

    @property
    def tilt(self):
        tilt = ()
        for radial_angles, azimuthal_angles, excitations in zip(
            self.radial_angles, self.azimuthal_angles, self.excitations
        ):
            radial_angle = sum(radial_angles[:excitations])
            azimuthal_angle = sum(radial_angles[:excitations])

            tilt += (
                (
                    radial_angle * np.cos(azimuthal_angle),
                    radial_angle * np.sin(azimuthal_angle),
                ),
            )

        return tilt

    def update(self, depth):

        values = ()
        for excitation_depths, value in zip(self.depths, self.values):

            for i, excitation_depth in enumerate(excitation_depths):
                if excitation_depth > depth:
                    break
            else:
                i = len(excitation_depths)

            values += (value[:-1] + (i,),)

        self.values = values


def _update_plasmon_axes(waves, depth):
    for axis in _iterate_axes_type(waves, PlasmonAxis):
        axis.update(depth)


def reduce_plasmon_axes(measurement):
    plasmon_axes = [
        (i, axes_metadata)
        for i, axes_metadata in enumerate(measurement.axes_metadata)
        if isinstance(axes_metadata, PlasmonAxis)
    ]

    if len(plasmon_axes) == 0:
        return measurement

    plasmon_axis_index, plasmon_axis = plasmon_axes[0]

    num_excitations = [len(value[0]) for value in plasmon_axis.values]

    uniques, inverse = np.unique(num_excitations, return_inverse=True)

    axis_values = []
    new_array = []
    for i, unique in enumerate(uniques):
        axis_values.append(f"{ntuples[unique]}")
        indices = np.where(i == inverse)[0]
        new_array.append(measurement.array[indices].mean(0, keepdims=True))

    array = da.concatenate(new_array, axis=plasmon_axis_index)

    kwargs = measurement._copy_kwargs(exclude=("array",))
    kwargs["ensemble_axes_metadata"][plasmon_axis_index] = OrdinalAxis(
        label="", values=axis_values
    )

    return measurement.__class__(array, **kwargs)


class PlasmonScatteringEvents(ArrayObjectTransform):
    def __init__(
        self,
        depths: Tuple[Tuple[float, ...]],
        radial_angles: Tuple[Tuple[float, ...]],
        azimuthal_angles: Tuple[Tuple[float, ...]],
        weights: Tuple[float],
        ensemble_mean: bool,
    ):
        if not (
            len(depths) == len(radial_angles) == len(azimuthal_angles) == len(weights)
        ):
            raise ValueError()

        if not all(
            len(d) == len(r) == len(a)
            for d, r, a in zip(depths, radial_angles, azimuthal_angles)
        ):
            raise ValueError()

        self._depths = depths
        self._radial_angles = radial_angles
        self._azimuthal_angles = azimuthal_angles
        self._weights = weights
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
    def depths(self) -> Tuple[Tuple[float, ...]]:
        return self._depths

    @property
    def radial_angles(self) -> Tuple[Tuple[float, ...]]:
        return self._radial_angles

    @property
    def azimuthal_angles(self) -> Tuple[Tuple[float, ...]]:
        return self._azimuthal_angles

    @property
    def weights(self) -> Tuple[float]:
        return self._weights

    @property
    def num_events(self):
        return len(self._depths)

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

    def get_scattering_event_depths(self, num_excitations: int = 1):

        event_depths = defaultdict(list)
        for depths in self.depths:

            n = len(depths)
            if n >= num_excitations:
                event_depths[ntuples[n]].append(depths[num_excitations - 1])

        return event_depths

    def show_cumulative_scattering_events(
        self, ax=None, num_excitations: Union[int, List[int]] = 1, **kwargs
    ):
        if isinstance(num_excitations, int):
            num_excitations = [1]

        if ax is None:
            fig, axes = plt.subplots(1, len(num_excitations), sharey=True)
        else:
            axes = [ax]

        print(axes)

        if isinstance(axes, Axes):
            axes = [axes]

        if not "bins" in kwargs:
            kwargs["bins"] = 20

        for i, (n, ax) in enumerate(zip(num_excitations, axes)):
            scattering_depths = self.get_scattering_event_depths(n)
            ax.hist(
                scattering_depths.values(),
                cumulative=True,
                density=True,
                histtype="step",
                label=list(scattering_depths.keys()),
                **kwargs,
            )
            ax.set_xlabel("Depth [Å]")
            if i == 0:
                ax.set_ylabel("Cumulative distribution")
            ax.set_title(f"{nth[n]} scattering event")
            ax.legend(loc=2)
        return ax

    def show_scattering_angle_distribution(self, ax=None, **kwargs):
        scattering_angles = list(itertools.chain(*self.radial_angles))

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.hist(scattering_angles, **kwargs)
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Scattering angle [mrad]")

    def show_weights(self):

        uniques, indices = np.unique(
            [len(depths) for depths in self.depths], return_index=True
        )

        weights = [self.weights[index] for index in indices]

        x = [ntuples[unique] for unique in uniques]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x, weights)
        ax.set_ylabel("Weight")

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return [
            PlasmonAxis(
                values=tuple(
                    (depths, radial_angles, azimuthal_angles, 0)
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
        kwargs["weights"] = args["weights"]
        return cls(**kwargs)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(
            exclude=("depths", "radial_angles", "azimuthal_angles")
        )
        return partial(self._from_partitioned_args_func, **kwargs)

    @staticmethod
    def _plasmon_scattering_events(depths, radial_angles, azimuthal_angles, weights):
        arr = np.zeros((1,), dtype=object)
        arr.itemset(
            0,
            {
                "depths": depths,
                "radial_angles": radial_angles,
                "azimuthal_angles": azimuthal_angles,
                "weights": weights,
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
            weights = self.weights[start:stop]

            if lazy:
                lazy_frozen_phonon = dask.delayed(self._plasmon_scattering_events)(
                    depths=depths,
                    radial_angles=radial_angles,
                    azimuthal_angles=azimuthal_angles,
                    weights=weights,
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
                        weights=weights,
                    ),
                )

        if lazy:
            array = da.concatenate(list(array))

        return (array,)

    def apply(self, waves: "Waves", in_place: bool = False) -> "Waves":
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
        num_excitations: Union[int, Tuple[int, ...]] = None,
        num_samples: int = None,
        weights: Union[bool] = True,
        ensemble_mean: bool = False,
        seed: Union[int, Tuple[int, ...]] = None,
    ):
        self._mean_free_path = mean_free_path
        self._excitation_energy = excitation_energy
        self._critical_angle = critical_angle
        self._ensemble_mean = ensemble_mean
        self._num_samples = num_samples
        self._seed = seed

        if isinstance(num_excitations, int):
            num_excitations = tuple(range(num_excitations + 1))

        self._num_excitations = num_excitations

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
        return self._excitation_energy / (2 * energy) * 1e3

    def draw_events(
        self, waves: "Waves", potential: "BasePotential"
    ) -> PlasmonScatteringEvents:
        depth = potential.thickness
        energy = waves.energy

        rng = np.random.default_rng(self.seed)

        depths = []
        radial_angles = []
        azimuthal_angles = []
        weights = []

        for n in self._num_excitations:
            if n == 0:
                num_samples = 1
            else:
                num_samples = self.num_samples

            depths.append(
                draw_scattering_depths(
                    mean_free_path=self._mean_free_path,
                    num_depths=n,
                    max_depth=depth,
                    num_samples=num_samples,
                    rng=rng,
                )
            )

            radial_angles.append(
                draw_radial_scattering_angle(
                    self._critical_angle,
                    self.characteristic_angle(energy),
                    rng=rng,
                    num_samples=num_samples,
                    num_depths=n,
                )
            )

            azimuthal_angles.append(
                draw_azimuthal_angle(num_samples=num_samples, num_depths=n, rng=rng)
            )

            weights.append(
                (excitations_weights(n, depth, self._mean_free_path),) * num_samples
            )

        depths = list(itertools.chain(*depths))
        radial_angles = list(itertools.chain(*radial_angles))
        azimuthal_angles = list(itertools.chain(*azimuthal_angles))
        weights = list(itertools.chain(*weights))

        # depths = tuple(
        #     depths
        #     for n in self._num_excitations
        #     for depths in draw_scattering_depths(
        #         mean_free_path=self._mean_free_path,
        #         num_depths=n,
        #         max_depth=depth,
        #         num_samples=self.num_samples,
        #         rng=rng,
        #     )
        # )
        #
        # radial_angles = tuple(
        #     tuple(
        #         draw_radial_scattering_angle(
        #             self._critical_angle, self.characteristic_angle(energy), rng=rng
        #         )
        #         for _ in range(n)
        #     )
        #     for n in self._num_excitations
        #     for _ in range(self.num_samples)
        # )
        #
        # azimuthal_angles = tuple(
        #     tuple(draw_azimuthal_angle(rng=rng) for _ in range(n))
        #     for n in self._num_excitations
        #     for _ in range(self.num_samples)
        # )
        #
        # weights = tuple(
        #     excitations_weights(n, depth, self._mean_free_path)
        #     for n in self._num_excitations
        #     for _ in range(self.num_samples)
        # )

        return PlasmonScatteringEvents(
            depths,
            radial_angles,
            azimuthal_angles,
            weights,
            ensemble_mean=self.ensemble_mean,
        )
