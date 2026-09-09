import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, List, Tuple, Union

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from abtem.core.axes import (
    AxisMetadata,
    OrdinalAxis,
    PlasmonOrderAxis,
    _iterate_axes_type,
)
from abtem.core.backend import get_array_module
from abtem.core.chunks import chunk_ranges, validate_chunks
from abtem.core.complex import abs2
from abtem.core.ensemble import _wrap_with_array
from abtem.core.utils import get_dtype, itemset
from abtem.transform import ArrayObjectTransform

if TYPE_CHECKING:
    from abtem.potentials import BasePotential
    from abtem.waves import Waves


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
        / math.factorial(n)
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
            # Successive scattering events add as vectors in the small-angle
            # limit; sum the x and y tilt components rather than the angles.
            tilt_x = sum(
                r * np.cos(a)
                for r, a in zip(radial_angles[:excitations], azimuthal_angles)
            )
            tilt_y = sum(
                r * np.sin(a)
                for r, a in zip(radial_angles[:excitations], azimuthal_angles)
            )
            tilt += ((float(tilt_x), float(tilt_y)),)

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

        if "bins" not in kwargs:
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
        return _wrap_with_array(cls(**kwargs), 0)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(
            exclude=("depths", "radial_angles", "azimuthal_angles")
        )
        return partial(self._from_partitioned_args_func, **kwargs)

    @staticmethod
    def _plasmon_scattering_events(depths, radial_angles, azimuthal_angles, weights):
        arr = np.zeros((1,), dtype=object)
        itemset(
            arr,
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
                itemset(
                    array,
                    i,
                    da.from_delayed(lazy_frozen_phonon, shape=(1,), dtype=object),
                )
            else:
                itemset(
                    array,
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

    def _calculate_new_array(self, waves: "Waves") -> np.ndarray:
        xp = get_array_module(waves.device)
        array = waves.array[(None,) * len(self.ensemble_shape)]
        return xp.tile(array, self.ensemble_shape + (1,) * len(waves.shape))

    def apply(self, waves: "Waves", max_batch: int | str = "auto") -> "Waves":
        return waves.apply_transform(self, max_batch=max_batch)


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
        energy = waves._valid_energy

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

        return PlasmonScatteringEvents(
            depths,
            radial_angles,
            azimuthal_angles,
            weights,
            ensemble_mean=self.ensemble_mean,
        )


# ---------------------------------------------------------------------------
# Deterministic (quadrature) plasmon scattering
# ---------------------------------------------------------------------------


def _find_plasmon_order_axis(waves) -> Union[Tuple[int, PlasmonOrderAxis], None]:
    for i, axis in enumerate(waves.ensemble_axes_metadata):
        if isinstance(axis, PlasmonOrderAxis):
            return i, axis
    return None


class QuadraturePlasmons(ArrayObjectTransform):
    """
    Plasmon energy-loss scattering integrated by deterministic quadrature.

    The plasmon model is that of B.G. Mendis, Ultramicroscopy 206 (2019) 112816: a
    plasmon excitation is a delocalized event that transfers a transverse momentum to
    the fast electron, tilting it by an angle distributed as a Lorentzian
    :math:`P(\\theta) \\propto \\theta / (\\theta^2 + \\theta_E^2)` up to the critical
    angle :math:`\\theta_c`, at a depth that is uniformly distributed through the
    specimen given the number of excitations (Poisson statistics). Different
    scattering angles and depths are mutually incoherent.

    Instead of sampling scattering events at random (:class:`MonteCarloPlasmons`), the
    incoherent integral over scattering angle and depth is evaluated by quadrature:

    * The angular distribution is split at ``min_angle``. Tilts below ``min_angle``
      (by default one reciprocal-space pixel) are too small to change the channeling
      of the electron and are treated exactly as a momentum transfer, i.e. as a
      convolution of the diffraction pattern with the Lorentzian core. Larger tilts
      are represented by ``num_angles`` rings of ``num_azimuthal`` tilt nodes, spaced
      uniformly in the cumulative probability of the Lorentzian. Each node carries
      the exact momentum-transfer distribution of its cell of the Lorentzian, again
      applied as a convolution of its diffraction pattern.
    * The scattering depth of each channeling-changing event is represented by
      ``num_depths`` uniformly spaced depth nodes. Every node spawns a copy of the
      wave function that continues through the specimen as a tilted beam.

    A single pass of the multislice algorithm therefore propagates the elastic wave
    function together with ``num_depths * num_angles * num_azimuthal`` tilted copies,
    and every plasmon-loss order up to ``max_loss_order`` is assembled from the same
    set of copies. Paths with more than ``max_tilt_events`` channeling-changing
    events are approximated by treating the additional events as pure momentum
    transfers.

    Parameters
    ----------
    mean_free_path : float
        Plasmon mean free path :math:`\\lambda_p` [Å]. Only used for the Poisson
        weights of the loss orders, see :meth:`excitation_weights`.
    excitation_energy : float
        Plasmon excitation energy :math:`E_p` [eV], setting the characteristic angle
        :math:`\\theta_E = E_p / (2 E_0)`.
    critical_angle : float
        Critical angle :math:`\\theta_c` [mrad] above which plasmon scattering is
        neglected.
    max_loss_order : int, optional
        Highest number of plasmon excitations to compute (default is 1). The output
        has a leading axis with the zero-loss and every loss order up to this value,
        each normalized to the intensity of the incident wave function.
    num_angles : int, optional
        Number of radial tilt nodes between ``min_angle`` and the critical angle
        (default is 6).
    num_azimuthal : int, optional
        Number of azimuthal tilt nodes per ring (default is 8).
    num_depths : int, optional
        Number of depth nodes for each channeling-changing scattering event
        (default is 4).
    min_angle : float, optional
        Scattering angle [mrad] below which the tilt does not change the channeling
        of the electron. If not given, one reciprocal-space pixel of the wave function
        grid is used.
    max_tilt_events : int, optional
        Maximum number of channeling-changing scattering events per path, 0, 1 or 2
        (default is 1). With 2, pairs of events are represented by
        ``pair_num_angles`` rings of ``pair_num_azimuthal`` nodes for the second event
        of each pair. With 0, plasmon scattering is treated as pure momentum transfer
        (a convolution of the elastic diffraction pattern) without any change to the
        propagation, which is fast but neglects the change of channeling.
    pair_num_angles : int, optional
        Number of radial tilt nodes for the second event of a pair (default is 2).
    pair_num_azimuthal : int, optional
        Number of azimuthal tilt nodes for the second event of a pair (default is 4).
    lab_frame : bool, optional
        If True (default), the diffraction patterns include the momentum transferred
        to the electron, i.e. they are shifted by the plasmon scattering angle. If
        False, every scattered wave function is detected in its own tilted frame of
        reference, which is the convention of :class:`MonteCarloPlasmons`.
    """

    def __init__(
        self,
        mean_free_path: float,
        excitation_energy: float,
        critical_angle: float,
        max_loss_order: int = 1,
        num_angles: int = 6,
        num_azimuthal: int = 8,
        num_depths: int = 4,
        min_angle: float = None,
        max_tilt_events: int = 1,
        pair_num_angles: int = 2,
        pair_num_azimuthal: int = 4,
        lab_frame: bool = True,
    ):
        if max_tilt_events not in (0, 1, 2):
            raise ValueError("`max_tilt_events` must be 0, 1 or 2")
        if max_loss_order < 0:
            raise ValueError("`max_loss_order` must be non-negative")

        self._mean_free_path = float(mean_free_path)
        self._excitation_energy = float(excitation_energy)
        self._critical_angle = float(critical_angle)
        self._max_loss_order = int(max_loss_order)
        self._num_angles = int(num_angles)
        self._num_azimuthal = int(num_azimuthal)
        self._num_depths = int(num_depths)
        self._min_angle = None if min_angle is None else float(min_angle)
        self._max_tilt_events = int(max_tilt_events)
        self._pair_num_angles = int(pair_num_angles)
        self._pair_num_azimuthal = int(pair_num_azimuthal)
        self._lab_frame = bool(lab_frame)

    @property
    def mean_free_path(self) -> float:
        """Plasmon mean free path [Å]."""
        return self._mean_free_path

    @property
    def excitation_energy(self) -> float:
        """Plasmon excitation energy [eV]."""
        return self._excitation_energy

    @property
    def critical_angle(self) -> float:
        """Critical scattering angle [mrad]."""
        return self._critical_angle

    @property
    def max_loss_order(self) -> int:
        """Highest number of plasmon excitations computed."""
        return self._max_loss_order

    @property
    def num_angles(self) -> int:
        return self._num_angles

    @property
    def num_azimuthal(self) -> int:
        return self._num_azimuthal

    @property
    def num_depths(self) -> int:
        return self._num_depths

    @property
    def min_angle(self) -> Union[float, None]:
        return self._min_angle

    @property
    def max_tilt_events(self) -> int:
        return self._max_tilt_events

    @property
    def pair_num_angles(self) -> int:
        return self._pair_num_angles

    @property
    def pair_num_azimuthal(self) -> int:
        return self._pair_num_azimuthal

    @property
    def lab_frame(self) -> bool:
        return self._lab_frame

    @property
    def num_orders(self) -> int:
        """Number of loss channels including the zero loss."""
        return self._max_loss_order + 1

    def characteristic_angle(self, energy: float) -> float:
        """Characteristic plasmon scattering angle [mrad] at the given energy [eV]."""
        return self._excitation_energy / (2 * energy) * 1e3

    def excitation_weights(self, thickness: float) -> Tuple[float, ...]:
        """Poisson probability of each loss order for the given thickness [Å]."""
        return tuple(
            excitations_weights(n, thickness, self._mean_free_path)
            for n in range(self.num_orders)
        )

    def show_weights(self, thickness: float, ax: Axes = None):
        """Bar chart of the Poisson weights of the loss orders."""
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))
        ax.bar(self.order_labels, self.excitation_weights(thickness))
        ax.set_ylabel("Weight")
        return ax

    @property
    def order_labels(self) -> Tuple[str, ...]:
        return tuple(ntuples[n] for n in range(self.num_orders))

    # -- angular quadrature --------------------------------------------------

    def _angular_nodes(self, energy: float, min_angle: float):
        """Tilt nodes [mrad] and the quadrature of the Lorentzian.

        Returns a dict with the class probabilities and the ring/sector partition
        used for the nodes and for the second event of a pair.
        """
        theta_e = self.characteristic_angle(energy)
        theta_c = self._critical_angle
        if min_angle >= theta_c:
            raise ValueError("`min_angle` must be smaller than the critical angle")

        def u(theta):
            return np.log(1 + (theta / theta_e) ** 2)

        def theta_of_u(value):
            return theta_e * np.sqrt(np.expm1(value))

        u_min, u_max = u(min_angle), u(theta_c)
        p_small = u_min / u_max
        p_large = 1 - p_small

        def partition(num_angles, num_azimuthal):
            edges = np.linspace(u_min, u_max, num_angles + 1)
            centers = 0.5 * (edges[1:] + edges[:-1])
            tilts = []
            for a, uc in enumerate(centers):
                theta = theta_of_u(uc)
                offset = 0.5 * (a % 2)
                for b in range(num_azimuthal):
                    phi = 2 * np.pi * (b + offset) / num_azimuthal
                    tilts.append((theta * np.cos(phi), theta * np.sin(phi)))
            return {
                "u_edges": edges,
                "num_angles": num_angles,
                "num_azimuthal": num_azimuthal,
                "tilts": np.array(tilts, dtype=float).reshape(-1, 2),
            }

        return {
            "theta_e": theta_e,
            "theta_c": theta_c,
            "u": u,
            "u_min": u_min,
            "u_max": u_max,
            "p_small": p_small,
            "p_large": p_large,
            "single": partition(self._num_angles, self._num_azimuthal),
            "pair": partition(self._pair_num_angles, self._pair_num_azimuthal),
        }

    # -- ensemble/transform protocol -------------------------------------------

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        return (self.num_orders,)

    @property
    def _default_ensemble_chunks(self):
        return (self.num_orders,)

    @property
    def parameters(self) -> dict:
        """The constructor arguments of the model."""
        return {
            "mean_free_path": self._mean_free_path,
            "excitation_energy": self._excitation_energy,
            "critical_angle": self._critical_angle,
            "max_loss_order": self._max_loss_order,
            "num_angles": self._num_angles,
            "num_azimuthal": self._num_azimuthal,
            "num_depths": self._num_depths,
            "min_angle": self._min_angle,
            "max_tilt_events": self._max_tilt_events,
            "pair_num_angles": self._pair_num_angles,
            "pair_num_azimuthal": self._pair_num_azimuthal,
            "lab_frame": self._lab_frame,
        }

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return [PlasmonOrderAxis(values=self.order_labels, parameters=self.parameters)]

    def _partition_args(self, chunks: int = 1, lazy: bool = True):
        chunks = validate_chunks(self.ensemble_shape, chunks)
        if len(chunks[0]) != 1:
            raise RuntimeError(
                "the plasmon excitation axis must be kept in a single chunk"
            )
        array = np.zeros((1,), dtype=object)
        itemset(array, 0, self)
        if lazy:
            array = da.from_array(array, chunks=1)
        return (array,)

    @staticmethod
    def _from_partitioned_args_func(*args, **kwargs):
        args = args[0]
        if hasattr(args, "item"):
            args = args.item()
        return _wrap_with_array(args, 0)

    def _from_partitioned_args(self):
        return partial(self._from_partitioned_args_func)

    def _calculate_new_array(self, waves: "Waves") -> np.ndarray:
        xp = get_array_module(waves.device)
        array = waves.array[(None,) * len(self.ensemble_shape)]
        return xp.tile(array, self.ensemble_shape + (1,) * len(waves.shape))

    def apply(self, waves: "Waves", max_batch: int | str = "auto") -> "Waves":
        """
        Attach the plasmon-loss channels to the wave functions.

        The returned wave functions have a leading ensemble axis with one entry per
        loss order. Running the multislice algorithm on them computes every loss
        channel in a single pass.
        """
        return waves.apply_transform(self, max_batch=max_batch)


def _lorentzian_kernels(
    gpts: Tuple[int, int],
    angular_sampling: Tuple[float, float],
    nodes: dict,
    supersampling: int = 4,
    xp=np,
):
    """Fourier transforms of the momentum-transfer kernels on the diffraction grid.

    Every kernel is the (normalized) shape of the Lorentzian scattering distribution
    restricted to one cell of the angular quadrature, sampled on the unshifted
    reciprocal-space grid of the wave functions. Returns the Fourier transforms of the
    small-angle kernel, the whole large-angle kernel, the kernels of the tilt nodes
    and the kernels of the second-event nodes.
    """
    theta_e = nodes["theta_e"]
    theta_c = nodes["theta_c"]
    u_min = nodes["u_min"]
    dx, dy = angular_sampling
    radius_x = int(np.ceil(theta_c / dx)) + 1
    radius_y = int(np.ceil(theta_c / dy)) + 1
    ix = np.arange(-radius_x, radius_x + 1)
    iy = np.arange(-radius_y, radius_y + 1)
    sub = (np.arange(supersampling) + 0.5) / supersampling - 0.5
    tx = (ix[:, None] + sub[None]) * dx  # (nx, s)
    ty = (iy[:, None] + sub[None]) * dy
    theta_x = tx[:, None, :, None]
    theta_y = ty[None, :, None, :]
    theta2 = theta_x**2 + theta_y**2
    theta = np.sqrt(theta2)
    phi = np.arctan2(theta_y, theta_x)
    weight = 1 / (theta2 + theta_e**2)
    weight[theta >= theta_c] = 0.0
    u = np.log(1 + theta2 / theta_e**2)

    def to_kernel(w):
        kernel = w.sum((-2, -1))
        total = kernel.sum()
        if total <= 0:
            raise RuntimeError("empty momentum-transfer kernel")
        kernel = kernel / total
        full = np.zeros(gpts, dtype=np.float32)
        # wrap into the unshifted grid
        gx = np.mod(ix, gpts[0])
        gy = np.mod(iy, gpts[1])
        np.add.at(full, (gx[:, None], gy[None, :]), kernel.astype(np.float32))
        return full

    small = to_kernel(np.where(u < u_min, weight, 0.0))
    large = to_kernel(np.where(u >= u_min, weight, 0.0))

    def partition_kernels(part):
        edges = part["u_edges"]
        n_az = part["num_azimuthal"]
        kernels = []
        for a in range(part["num_angles"]):
            in_ring = (u >= edges[a]) & (u < edges[a + 1])
            if a == part["num_angles"] - 1:
                in_ring = (u >= edges[a]) & (theta < theta_c)
            offset = 0.5 * (a % 2)
            for b in range(n_az):
                phi_b = 2 * np.pi * (b + offset) / n_az
                dphi = np.angle(np.exp(1j * (phi - phi_b)))
                in_sector = np.abs(dphi) <= np.pi / n_az
                kernels.append(to_kernel(np.where(in_ring & in_sector, weight, 0.0)))
        return np.stack(kernels)

    single = partition_kernels(nodes["single"])
    pair = partition_kernels(nodes["pair"])

    from abtem.core.fft import fft2

    def transform(kernel):
        return fft2(xp.asarray(kernel.astype(get_dtype(complex=True))))

    return {
        "small": transform(small),
        "large": transform(large),
        "single": transform(single),
        "pair": transform(pair),
    }


def quadrature_plasmon_multislice_and_detect(
    waves: "Waves",
    potential: "BasePotential",
    detectors: list = None,
    algorithm=None,
    pbar: bool = False,
    potential_chunk_size: int | str = "auto",
    **kwargs,
):
    """
    Multislice algorithm with plasmon scattering evaluated by quadrature.

    The wave functions must carry a :class:`PlasmonOrderAxis` as their leading
    ensemble axis, see :meth:`QuadraturePlasmons.apply`. One measurement per detector
    is returned with that axis resolving the number of plasmon excitations.
    """
    from math import comb

    from abtem.antialias import AntialiasAperture
    from abtem.core.axes import TiltAxis
    from abtem.core.diagnostics import TqdmWrapper
    from abtem.core.energy import energy2wavelength
    from abtem.core.fft import fft2, ifft2
    from abtem.detectors import validate_detectors
    from abtem.multislice import (
        FourierMultislice,
        FresnelPropagator,
        _generate_potential_configurations,
        _potential_ensemble_shape_and_metadata,
        _validate_potential_ensemble_indices,
        allocate_multislice_measurements,
        conventional_multislice_step,
    )

    if algorithm is None:
        algorithm = FourierMultislice()
    if not isinstance(algorithm, FourierMultislice):
        raise NotImplementedError(
            "quadrature plasmon scattering requires the Fourier multislice algorithm"
        )

    found = _find_plasmon_order_axis(waves)
    if found is None:
        raise RuntimeError("wave functions do not carry a plasmon excitation axis")
    order_axis_index, order_axis = found
    if order_axis_index != 0:
        raise RuntimeError("the plasmon excitation axis must be the leading axis")

    plasmons: QuadraturePlasmons = order_axis.plasmons
    n_orders = plasmons.num_orders
    if waves.shape[0] != n_orders:
        raise RuntimeError(
            "the plasmon excitation axis must be kept in a single chunk "
            f"(got {waves.shape[0]} of {n_orders} channels)"
        )

    waves = waves.ensure_real_space()
    detectors = validate_detectors(detectors)
    xp = get_array_module(waves.device)
    energy = waves._valid_energy
    wavelength = energy2wavelength(energy)
    gpts = waves._valid_gpts
    extent = waves.extent
    angular_sampling = tuple(wavelength / e * 1e3 for e in extent)

    min_angle = plasmons.min_angle
    if min_angle is None:
        min_angle = max(angular_sampling)
    nodes = plasmons._angular_nodes(energy, min_angle)
    p_small, p_large = nodes["p_small"], nodes["p_large"]
    max_tilt_events = plasmons.max_tilt_events
    num_depths = plasmons.num_depths

    if plasmons.lab_frame:
        kernels = _lorentzian_kernels(gpts, angular_sampling, nodes, xp=xp)
    else:
        kernels = None

    (
        extra_ensemble_axes_shape,
        extra_ensemble_axes_metadata,
    ) = _potential_ensemble_shape_and_metadata(potential)

    measurements = allocate_multislice_measurements(
        waves, detectors, extra_ensemble_axes_shape, extra_ensemble_axes_metadata
    )

    # The elastic wave functions, without the excitation axis.
    base_kwargs = waves._copy_kwargs(exclude=("array", "ensemble_axes_metadata"))
    base_axes = waves.ensemble_axes_metadata[1:]
    n_base = len(base_axes)

    def make_waves(array, extra_axes):
        return waves.__class__(
            array=array,
            ensemble_axes_metadata=list(extra_axes) + base_axes,
            **base_kwargs,
        )

    def tilt_axis(tilts):
        return TiltAxis(
            label="tilt",
            values=tuple(tuple(map(float, t)) for t in tilts),
            units="mrad",
        )

    single_tilts = nodes["single"]["tilts"]
    pair_tilts = nodes["pair"]["tilts"]
    n_single = len(single_tilts)
    n_pair = len(pair_tilts)
    single_axis = tilt_axis(single_tilts)
    combined = (single_tilts[:, None, :] + pair_tilts[None, :, :]).reshape(-1, 2)
    combined_axis = tilt_axis(combined)

    thickness = potential.thickness
    depth_nodes = (np.arange(num_depths) + 0.5) * thickness / num_depths

    antialias_aperture = AntialiasAperture()
    propagators = {
        "elastic": FresnelPropagator(),
        "single": FresnelPropagator(),
        "pair": FresnelPropagator(),
    }

    def step(w, transmission_function, key):
        return conventional_multislice_step(
            w,
            potential_slice=transmission_function,
            antialias_aperture=antialias_aperture,
            propagator=propagators[key],
            conjugate=algorithm.conjugate,
            transpose=algorithm.transpose,
            order=algorithm.order,
        )

    def kernel_broadcast(k):
        # (n, gx, gy) -> (n, 1..., gx, gy) for the base ensemble axes
        return k[(slice(None),) + (None,) * n_base]

    def detect(elastic, groups, measurement_index):
        """Assemble the loss channels and accumulate the detector signals."""
        ens = elastic.array.shape[:-2]
        i0 = abs2(fft2(elastic.array, overwrite_x=False))
        f0 = fft2(i0.astype(get_dtype(complex=True)))

        if kernels is None:
            ks = xp.ones((), dtype=get_dtype(complex=True))
            kl = ks
        else:
            ks = kernels["small"]
            kl = kernels["large"]

        accumulators = [None] * n_orders
        accumulators[0] = f0

        def add(n, value):
            if accumulators[n] is None:
                accumulators[n] = value
            else:
                accumulators[n] = accumulators[n] + value

        # m: number of channeling-changing events on the path
        def path_weight(n, m):
            return comb(n, m) * p_large**m * p_small ** (n - m)

        def contribute(f_pattern, m_group, depth_weight):
            # contributes to m = m_group if m_group < max_tilt_events,
            # otherwise to every m >= max_tilt_events (extra events as momentum
            # transfer)
            for n in range(1, n_orders):
                if m_group < max_tilt_events:
                    ms = [m_group] if m_group <= n else []
                else:
                    ms = range(m_group, n + 1)
                for m in ms:
                    factor = path_weight(n, m) * depth_weight
                    if factor == 0:
                        continue
                    term = f_pattern * factor
                    if n - m > 0:
                        term = term * ks ** (n - m)
                    if m - m_group > 0:
                        term = term * kl ** (m - m_group)
                    add(n, term)

        contribute(f0, 0, 1.0)

        for group in groups:
            arr = group["waves"].array
            intensity = abs2(fft2(arr, overwrite_x=False))
            f = fft2(intensity.astype(get_dtype(complex=True)))
            if kernels is not None:
                f = f * kernel_broadcast(group["kernel"])
            f = f.sum(0) / arr.shape[0]
            contribute(f, group["m"], group["depth_weight"])

        patterns = []
        for n in range(n_orders):
            if accumulators[n] is None:
                patterns.append(
                    xp.zeros(ens + tuple(gpts), dtype=get_dtype(complex=False))
                )
            else:
                patterns.append(xp.clip(ifft2(accumulators[n]).real, 0, None))
        intensity = xp.stack(patterns)
        # A wave function whose diffraction pattern is the accumulated intensity.
        carrier = ifft2(xp.sqrt(intensity).astype(get_dtype(complex=True)))
        carrier_waves = make_waves(carrier, [order_axis])
        for i, detector in enumerate(detectors):
            new_measurement = detector.detect(carrier_waves)
            measurements[i].array[measurement_index] += new_measurement.array

    n_waves = np.prod(waves.shape[1:-2]) if len(waves.shape) > 3 else 1
    n_slices = int(n_waves * potential.num_slices * potential.num_configurations)
    tqdm_pbar = TqdmWrapper(
        enabled=pbar, total=n_slices, leave=False, desc="multislice"
    )

    elastic_input = make_waves(waves.array[0].copy(), [])

    for potential_index, potential_configuration in _generate_potential_configurations(
        potential
    ):
        elastic = elastic_input.copy()
        groups = []
        next_node = 0
        depth = 0.0
        exit_plane_index = 0

        if potential.exit_planes[0] == -1:
            measurement_index = _validate_potential_ensemble_indices(
                potential_index, exit_plane_index, potential
            )
            detect(elastic, groups, measurement_index)
            exit_plane_index += 1

        for potential_chunk in potential_configuration.generate_chunked_slices(
            chunk_size=potential_chunk_size
        ):
            for potential_slice in potential_chunk.generate_slices():
                if potential_slice.device != elastic.device:
                    potential_slice = potential_slice.copy_to_device(elastic.device)
                transmission_function = potential_slice.transmission_function(
                    energy=energy
                )
                transmission_function = antialias_aperture.bandlimit(
                    transmission_function, in_place=True
                )

                elastic = step(elastic, transmission_function, "elastic")
                for group in groups:
                    group["waves"] = step(
                        group["waves"], transmission_function, group["key"]
                    )

                depth += potential_slice.axes_metadata[0].values[0]
                tqdm_pbar.update_if_exists(int(n_waves))

                # spawn tilted copies at the depth nodes
                while next_node < num_depths and depth >= depth_nodes[next_node]:
                    next_node += 1
                    if max_tilt_events == 0:
                        continue
                    if max_tilt_events == 2:
                        # pairs: second event at node j, first at an earlier node
                        for group in list(groups):
                            if group["m"] != 1:
                                continue
                            arr = group["waves"].array
                            tiled = xp.repeat(arr[:, None], n_pair, axis=1)
                            tiled = tiled.reshape((n_single * n_pair,) + arr.shape[1:])
                            groups.append(
                                {
                                    "waves": make_waves(tiled, [combined_axis]),
                                    "key": "pair",
                                    "m": 2,
                                    "depth_weight": 2.0 / num_depths**2,
                                    "kernel": None
                                    if kernels is None
                                    else (
                                        kernels["single"][:, None]
                                        * kernels["pair"][None]
                                    ).reshape((n_single * n_pair,) + tuple(gpts)),
                                }
                            )
                        # both events at node j
                        tiled = xp.tile(
                            elastic.array[None],
                            (n_single * n_pair,) + (1,) * len(elastic.shape),
                        )
                        groups.append(
                            {
                                "waves": make_waves(tiled, [combined_axis]),
                                "key": "pair",
                                "m": 2,
                                "depth_weight": 1.0 / num_depths**2,
                                "kernel": None
                                if kernels is None
                                else (
                                    kernels["single"][:, None] * kernels["pair"][None]
                                ).reshape((n_single * n_pair,) + tuple(gpts)),
                            }
                        )
                    tiled = xp.tile(
                        elastic.array[None], (n_single,) + (1,) * len(elastic.shape)
                    )
                    groups.append(
                        {
                            "waves": make_waves(tiled, [single_axis]),
                            "key": "single",
                            "m": 1,
                            "depth_weight": 1.0 / num_depths,
                            "kernel": None if kernels is None else kernels["single"],
                        }
                    )

                if potential_slice.exit_planes:
                    measurement_index = _validate_potential_ensemble_indices(
                        potential_index, exit_plane_index, potential
                    )
                    detect(elastic, groups, measurement_index)
                    exit_plane_index += 1

    tqdm_pbar.close_if_exists()
    return measurements
