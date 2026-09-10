"""Plasmon (bulk valence) energy-loss scattering in the multislice algorithm.

Three models share one convention: excitations form a Poisson process in depth with
mean free path ``mean_free_path``; each excitation deflects the electron with the
Lorentzian angular distribution ``P(theta) ~ theta / (theta^2 + theta_E^2)`` up to the
critical angle. :class:`MonteCarloPlasmons` samples events (Mendis, Ultramicroscopy
2019), :class:`PhaseScramblePlasmons` applies random kicks with random phases every
slice (Mendis, Ultramicroscopy 2023) and :class:`QuadraturePlasmons` integrates the
same distributions on deterministic nodes.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, List, Tuple, Union

import dask
import dask.array as da
import numpy as np

from abtem.core.axes import (
    AxisMetadata,
    PlasmonAxis,
    PlasmonOrderAxis,
    ThicknessAxis,
    _iterate_axes_type,
)
from abtem.core.backend import get_array_module
from abtem.core.chunks import chunk_ranges, validate_chunks
from abtem.core.complex import abs2
from abtem.core.energy import energy2wavelength, relativistic_mass_correction
from abtem.core.ensemble import _wrap_with_array
from abtem.core.utils import get_dtype, itemset
from abtem.transform import ArrayObjectTransform

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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
    9: "Nonuple plasmon",
}


def characteristic_angle(excitation_energy: float, energy: float) -> float:
    """Characteristic plasmon scattering angle [mrad].

    ``theta_E = E_p / (gamma m v^2) = E_p / (2 E_0) * 2 gamma / (gamma + 1)``, the
    relativistic form (Egerton, Electron Energy-Loss Spectroscopy in the Electron
    Microscope, 3rd ed., Eq. 3.28); ``gamma`` is the relativistic mass correction. At
    200 keV this is 16 percent wider than the non-relativistic ``E_p / (2 E_0)``.

    Parameters
    ----------
    excitation_energy : float
        Plasmon energy [eV].
    energy : float
        Electron energy [eV].
    """
    gamma = relativistic_mass_correction(energy)
    return excitation_energy / (2 * energy) * (2 * gamma / (gamma + 1)) * 1e3


def _loss_order_factors(
    num_orders: int,
    num_events: int,
    max_tilt_events: int,
    p_small: float,
    p_large: float,
):
    """Weights of a chain of ``num_events`` direction-changing events in every loss
    order.

    Returns ``(n, m, factor)`` triples: the chain contributes ``factor`` times its
    pattern to loss order ``n``, with ``m`` of the ``n`` excitations being large-angle
    (``m - num_events`` of them beyond the chain, treated as momentum transfers
    only) and ``n - m`` small-angle momentum transfers.
    """
    from math import comb

    factors = []
    for n in range(0 if num_events == 0 else 1, num_orders):
        if num_events < max_tilt_events:
            ms = [num_events] if num_events <= n else []
        else:
            ms = range(num_events, n + 1)
        for m in ms:
            factor = comb(n, m) * p_large**m * p_small ** (n - m)
            if factor:
                factors.append((n, m, factor))
    return factors


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


def _update_plasmon_axes(waves, depth):
    for axis in _iterate_axes_type(waves, PlasmonAxis):
        axis.update(depth)


def _event_tilts(plasmon_axis: PlasmonAxis) -> np.ndarray:
    """Total tilt [mrad] of every sampled event, summing all of its excitations."""
    tilts = []
    for value in plasmon_axis.values:
        radial, azimuthal = value[1], value[2]
        tilts.append(
            (
                sum(r * np.cos(a) for r, a in zip(radial, azimuthal)),
                sum(r * np.sin(a) for r, a in zip(radial, azimuthal)),
            )
        )
    return np.array(tilts, dtype=float).reshape(-1, 2)


def reduce_plasmon_axes(measurement, lab_frame: bool = False):
    """
    Average sampled plasmon scattering events into loss-order channels.

    Parameters
    ----------
    measurement : BaseMeasurements
        Measurement with a :class:`PlasmonAxis` of sampled events.
    lab_frame : bool, optional
        If True, shift the diffraction pattern of every event by the momentum
        transferred to the electron (the sum of its scattering angles) before
        averaging, so the patterns are in the laboratory frame of reference. Only
        possible for diffraction patterns. If False (default), every event is kept
        in its own tilted frame of reference, the convention of the original
        implementation.

    Returns
    -------
    reduced : BaseMeasurements
        Measurement with a :class:`PlasmonOrderAxis` in place of the event axis, one
        channel per number of excitations present in the events, each normalized to
        the intensity of the incident wave function.
    """
    from abtem.measurements import DiffractionPatterns

    plasmon_axes = [
        (i, axes_metadata)
        for i, axes_metadata in enumerate(measurement.axes_metadata)
        if isinstance(axes_metadata, PlasmonAxis)
    ]

    if len(plasmon_axes) == 0:
        return measurement

    plasmon_axis_index, plasmon_axis = plasmon_axes[0]
    if lab_frame and not isinstance(measurement, DiffractionPatterns):
        raise NotImplementedError(
            "the laboratory frame requires diffraction patterns (PixelatedDetector)"
        )
    if lab_frame and any(
        isinstance(axis, ThicknessAxis) for axis in measurement.axes_metadata
    ):
        raise NotImplementedError(
            "the laboratory frame shifts every event by the momentum of all of its "
            "excitations, which is only right at the final exit plane; use a single "
            "exit plane or the tilted frame"
        )

    num_excitations = [len(value[0]) for value in plasmon_axis.values]
    uniques, inverse = np.unique(num_excitations, return_inverse=True)

    array = measurement.array
    lazy = isinstance(array, da.core.Array)
    xp = da if lazy else get_array_module(array)
    if lab_frame:
        tilts = _event_tilts(plasmon_axis)
        sampling = measurement.angular_sampling
        shifts = np.round(tilts / np.array(sampling)).astype(int)
        shape = measurement.shape[-2:]

        def shifted(pattern, shift):
            # intensity moved past the edge of the (cropped) pattern is lost, not
            # wrapped to the other side
            pattern = xp.roll(pattern, tuple(shift), axis=(-2, -1))
            for axis, (n, s) in enumerate(zip(shape, shift)):
                mask = np.ones(n, dtype=bool)
                if s > 0:
                    mask[:s] = False
                elif s < 0:
                    mask[n + s :] = False
                pattern = pattern * mask.reshape((-1, 1) if axis == 0 else (1, -1))
            return pattern

    axis_values = []
    new_array = []
    for i, unique in enumerate(uniques):
        axis_values.append(f"{ntuples[unique]}")
        indices = np.where(i == inverse)[0]
        if lab_frame:
            index = [slice(None)] * len(measurement.shape)
            members = []
            for j in indices:
                index[plasmon_axis_index] = j
                members.append(shifted(array[tuple(index)], shifts[j]))
            channel = xp.stack(members, axis=plasmon_axis_index).mean(
                plasmon_axis_index, keepdims=True
            )
        else:
            index = [slice(None)] * len(measurement.shape)
            index[plasmon_axis_index] = indices
            channel = array[tuple(index)].mean(plasmon_axis_index, keepdims=True)
        new_array.append(channel)

    array = xp.concatenate(new_array, axis=plasmon_axis_index)

    kwargs = measurement._copy_kwargs(exclude=("array",))
    kwargs["ensemble_axes_metadata"][plasmon_axis_index] = PlasmonOrderAxis(
        values=tuple(axis_values), model="monte_carlo"
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
        import matplotlib.pyplot as plt

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
            num_excitations = [num_excitations]

        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes

        if ax is None:
            fig, axes = plt.subplots(1, len(num_excitations), sharey=True)
        else:
            axes = [ax]

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

        import matplotlib.pyplot as plt

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

        import matplotlib.pyplot as plt

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
        num_excitations: Union[int, Tuple[int, ...]],
        num_samples: int = None,
        ensemble_mean: bool = False,
        seed: Union[int, Tuple[int, ...]] = None,
        lab_frame: bool = True,
    ):
        self._mean_free_path = mean_free_path
        self._excitation_energy = excitation_energy
        self._critical_angle = critical_angle
        self._ensemble_mean = ensemble_mean
        self._num_samples = num_samples
        self._seed = seed
        self._lab_frame = bool(lab_frame)

        if isinstance(num_excitations, int):
            num_excitations = tuple(range(num_excitations + 1))

        self._num_excitations = tuple(num_excitations)

    @property
    def lab_frame(self) -> bool:
        """Whether the loss channels are detected in the laboratory frame."""
        return self._lab_frame

    @property
    def num_excitations(self) -> Tuple[int, ...]:
        """The numbers of excitations that are sampled."""
        return self._num_excitations

    @property
    def max_loss_order(self) -> int:
        """Highest number of excitations sampled."""
        return max(self._num_excitations)

    @property
    def parameters(self) -> dict:
        """The constructor arguments of the model."""
        return {
            "mean_free_path": self._mean_free_path,
            "excitation_energy": self._excitation_energy,
            "critical_angle": self._critical_angle,
            "num_excitations": self._num_excitations,
            "num_samples": self._num_samples,
            "seed": self._seed,
            "lab_frame": self._lab_frame,
        }

    @property
    def order_axis(self) -> PlasmonOrderAxis:
        """The loss-order axis of the reduced measurements."""
        return PlasmonOrderAxis(
            values=tuple(ntuples[n] for n in sorted(set(self._num_excitations))),
            model="monte_carlo",
            parameters=self.parameters,
        )

    @property
    def num_orders(self) -> int:
        """Number of loss channels including the zero loss."""
        return len(self.order_labels)

    @property
    def order_labels(self) -> Tuple[str, ...]:
        """Label of every loss channel, in the order of the loss-order axis."""
        return tuple(ntuples[n] for n in sorted(set(self._num_excitations)))

    def excitation_weights(self, thickness: float) -> Tuple[float, ...]:
        """Poisson probability of each sampled loss order at a thickness [Å].

        The loss channels of this model are each normalized to the incident electron
        count, so these are the weights with which they add to the unfiltered signal.
        """
        return tuple(
            excitations_weights(n, thickness, self._mean_free_path)
            for n in sorted(set(self._num_excitations))
        )

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
        """Characteristic plasmon scattering angle [mrad] at the given energy [eV]."""
        return characteristic_angle(self._excitation_energy, energy)

    def draw_events(
        self, waves: "Waves", potential: "BasePotential"
    ) -> PlasmonScatteringEvents:
        return self._draw_events(
            thickness=potential.thickness, energy=waves._valid_energy
        )

    def _draw_events(self, thickness: float, energy: float) -> PlasmonScatteringEvents:
        """Draw Monte Carlo plasmon scattering events for a specimen thickness and
        electron energy.

        The object-agnostic core of :meth:`draw_events`: it needs neither a ``Waves``
        nor a ``BasePotential`` object and is used by the Bloch-wave driver.

        Parameters
        ----------
        thickness : float
            The specimen thickness [Å].
        energy : float
            The electron energy [eV].

        Returns
        -------
        events : PlasmonScatteringEvents
            The sampled scattering events.
        """
        depth = thickness

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


class QuadraturePlasmons:
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
      ``num_depths`` uniformly spaced depth nodes. Every node spawns copies of the
      wave function that continue through the specimen as tilted beams.
    * Paths with several channeling-changing events, up to ``max_tilt_events``, are
      represented by chains of tilted copies: the first event uses the
      ``num_angles`` x ``num_azimuthal`` nodes, every later event the coarser
      ``event_num_angles`` x ``event_num_azimuthal`` nodes, with the cumulative tilt
      applied from the depth node of each event onward. Paths with more events than
      ``max_tilt_events`` treat the additional events as pure momentum transfers.

    All loss orders up to ``max_loss_order`` are assembled from the same set of
    copies. The copies propagate alongside the elastic wave function in a single pass
    of the multislice algorithm, or in several passes over subsets of the first-event
    nodes when ``max_copies`` bounds the number of copies held in memory. The model is
    passed as ``plasmons=`` to the multislice methods, e.g.
    ``probe.multislice(potential, detectors=detector, plasmons=model)``; the
    measurements gain a leading :class:`PlasmonOrderAxis`.

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
        Number of radial tilt nodes between ``min_angle`` and the critical angle for
        the first channeling-changing event (default is 6).
    num_azimuthal : int, optional
        Number of azimuthal tilt nodes per ring for the first event (default is 8).
    num_depths : int, optional
        Number of depth nodes for each channeling-changing scattering event
        (default is 4).
    min_angle : float, optional
        Scattering angle [mrad] below which the tilt does not change the channeling
        of the electron. If not given, one reciprocal-space pixel of the wave function
        grid is used.
    max_tilt_events : int, optional
        Maximum number of channeling-changing scattering events per path (default is
        1). Paths with more events treat the additional events as pure momentum
        transfer. With 0, plasmon scattering is treated as pure momentum transfer
        (a convolution of the elastic diffraction pattern) without any change to the
        propagation. The number of copies grows as
        ``num_angles * num_azimuthal`` times
        ``(event_num_angles * event_num_azimuthal) ** (m - 1)``
        times the number of depth-node combinations for ``m`` events, so use
        ``max_copies`` for values above 1.
    event_num_angles : int, optional
        Number of radial tilt nodes for the second and later events of a path
        (default is 2).
    event_num_azimuthal : int, optional
        Number of azimuthal tilt nodes for the second and later events of a path
        (default is 4).
    max_angular_step : float, optional
        Largest angular width of a quadrature cell [mrad], radially and along the arc.
        Rings that a uniform-probability spacing would leave coarser than this are
        subdivided, which matters when the scattered intensity varies on an angular
        scale rather than a probability scale. If not given, the rings are spaced by
        probability alone.
    event_max_angular_step : float, optional
        As ``max_angular_step``, for the second and later events of a path. Defaults to
        ``max_angular_step``.
    max_copies : int, optional
        Maximum number of tilted copies of the wave function held in memory at once.
        If exceeded, the first-event nodes are split into subsets that are propagated
        in separate passes of the multislice algorithm. If not given, a single pass is
        used.
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
        event_num_angles: int = 2,
        event_num_azimuthal: int = 4,
        max_angular_step: float = None,
        event_max_angular_step: float = None,
        max_copies: int = None,
        lab_frame: bool = True,
    ):
        if max_tilt_events < 0:
            raise ValueError("`max_tilt_events` must be non-negative")
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
        self._event_num_angles = int(event_num_angles)
        self._event_num_azimuthal = int(event_num_azimuthal)
        self._max_angular_step = (
            None if max_angular_step is None else float(max_angular_step)
        )
        self._event_max_angular_step = (
            None if event_max_angular_step is None else float(event_max_angular_step)
        )
        self._max_copies = None if max_copies is None else int(max_copies)
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
    def event_num_angles(self) -> int:
        return self._event_num_angles

    @property
    def event_num_azimuthal(self) -> int:
        return self._event_num_azimuthal

    @property
    def max_angular_step(self) -> Union[float, None]:
        return self._max_angular_step

    @property
    def event_max_angular_step(self) -> Union[float, None]:
        return self._event_max_angular_step

    @property
    def max_copies(self) -> Union[int, None]:
        return self._max_copies

    @property
    def lab_frame(self) -> bool:
        return self._lab_frame

    @property
    def num_orders(self) -> int:
        """Number of loss channels including the zero loss."""
        return self._max_loss_order + 1

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
            "event_num_angles": self._event_num_angles,
            "event_num_azimuthal": self._event_num_azimuthal,
            "max_angular_step": self._max_angular_step,
            "event_max_angular_step": self._event_max_angular_step,
            "max_copies": self._max_copies,
            "lab_frame": self._lab_frame,
        }

    def characteristic_angle(self, energy: float) -> float:
        """Characteristic plasmon scattering angle [mrad] at the given energy [eV]."""
        return characteristic_angle(self._excitation_energy, energy)

    def excitation_weights(self, thickness: float) -> Tuple[float, ...]:
        """Poisson probability of each loss order for the given thickness [Å]."""
        return tuple(
            excitations_weights(n, thickness, self._mean_free_path)
            for n in range(self.num_orders)
        )

    def show_weights(self, thickness: float, ax: Axes = None):
        """Bar chart of the Poisson weights of the loss orders."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))
        ax.bar(self.order_labels, self.excitation_weights(thickness))
        ax.set_ylabel("Weight")
        return ax

    @property
    def order_labels(self) -> Tuple[str, ...]:
        return tuple(ntuples[n] for n in range(self.num_orders))

    def num_copies(self) -> int:
        """Number of tilted copies of the wave function for a single pass."""
        n_single = self._num_angles * self._num_azimuthal
        n_extra = self._event_num_angles * self._event_num_azimuthal
        return n_single * self._copies_per_first_node(n_extra)

    def _copies_per_first_node(self, n_extra: int) -> int:
        from math import comb

        return sum(
            comb(self._num_depths + m - 1, m) * n_extra ** (m - 1)
            for m in range(1, self._max_tilt_events + 1)
        )

    # -- angular quadrature --------------------------------------------------

    def _angular_nodes(
        self,
        energy: float,
        min_angle: float,
        max_angular_step: float = None,
        event_max_angular_step: float = None,
    ):
        """Tilt nodes [mrad] and the quadrature of the Lorentzian.

        Returns a dict with the class probabilities and the ring/sector partitions
        used for the first event (``single``) and for later events (``extra``).

        ``max_angular_step`` [mrad], if given, caps the angular width of a cell, both
        radially and along the arc, refining the rings that a uniform-probability
        spacing would leave too coarse; ``event_max_angular_step`` overrides it for the
        later events.
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

        def partition(num_angles, num_azimuthal, max_step):
            """Cells of the angular quadrature and their probabilities.

            The rings are spaced uniformly in the cumulative probability ``u``. A ring
            wider than ``max_step`` in angle is subdivided until every cell is at most
            that wide, and the azimuthal sampling of a ring is refined so that its cells
            are at most ``max_step`` wide along the arc as well. This resolves
            integrands
            that vary on an angular scale rather than a probability scale, such as the
            rocking curves of Bloch waves.
            """
            edges = np.linspace(u_min, u_max, num_angles + 1)
            if max_step is not None:
                refined = [edges[0]]
                for a in range(num_angles):
                    theta_lo, theta_hi = theta_of_u(edges[a]), theta_of_u(edges[a + 1])
                    parts = int(np.ceil((theta_hi - theta_lo) / max_step))
                    for i in range(1, parts + 1):
                        theta_edge = theta_lo + (theta_hi - theta_lo) * i / parts
                        refined.append(u(theta_edge))
                edges = np.array(refined)
            tilts, weights = [], []
            for a in range(len(edges) - 1):
                u_lo, u_hi = edges[a], edges[a + 1]
                theta = theta_of_u(0.5 * (u_lo + u_hi))
                num_sectors = num_azimuthal
                if max_step is not None:
                    num_sectors = max(
                        num_azimuthal, int(np.ceil(2 * np.pi * theta / max_step))
                    )
                offset = 0.5 * (a % 2)
                weight = (u_hi - u_lo) / (u_max - u_min) / num_sectors
                for b in range(num_sectors):
                    phi = 2 * np.pi * (b + offset) / num_sectors
                    tilts.append((theta * np.cos(phi), theta * np.sin(phi)))
                    weights.append(weight)
            return {
                "u_edges": edges,
                "num_angles": len(edges) - 1,
                "num_azimuthal": num_azimuthal,
                "max_step": max_step,
                "tilts": np.array(tilts, dtype=float).reshape(-1, 2),
                "weights": np.array(weights, dtype=float),
            }

        return {
            "theta_e": theta_e,
            "theta_c": theta_c,
            "u": u,
            "u_min": u_min,
            "u_max": u_max,
            "p_small": p_small,
            "p_large": p_large,
            "single": partition(
                self._num_angles, self._num_azimuthal, max_angular_step
            ),
            "extra": partition(
                self._event_num_angles,
                self._event_num_azimuthal,
                max_angular_step
                if event_max_angular_step is None
                else event_max_angular_step,
            ),
        }

    @property
    def order_axis(self) -> PlasmonOrderAxis:
        """The loss-order axis of the measurements."""
        return PlasmonOrderAxis(
            values=self.order_labels, model="quadrature", parameters=self.parameters
        )


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
    small-angle kernel, the whole large-angle kernel, the kernels of the first-event
    tilt nodes and the kernels of the later-event nodes.
    """
    theta_e = nodes["theta_e"]
    theta_c = nodes["theta_c"]
    u_min = nodes["u_min"]
    dx, dy = angular_sampling
    radius_x = int(np.ceil(theta_c / dx)) + 1
    radius_y = int(np.ceil(theta_c / dy)) + 1
    if 2 * radius_x + 1 > gpts[0] or 2 * radius_y + 1 > gpts[1]:
        warnings.warn(
            f"the critical angle ({theta_c:.1f} mrad) exceeds half the extent of the "
            "diffraction grid; the momentum-transfer kernels wrap around it",
            stacklevel=3,
        )
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
        full = np.zeros(gpts, dtype=get_dtype(complex=False))
        # wrap into the unshifted grid
        gx = np.mod(ix, gpts[0])
        gy = np.mod(iy, gpts[1])
        np.add.at(full, (gx[:, None], gy[None, :]), kernel.astype(full.dtype))
        return full

    small = to_kernel(np.where(u < u_min, weight, 0.0))
    large = to_kernel(np.where(u >= u_min, weight, 0.0))

    def partition_kernels(part):
        edges = part["u_edges"]
        max_step = part["max_step"]
        theta_of_u = lambda value: theta_e * np.sqrt(np.expm1(value))  # noqa: E731
        kernels = []
        for a in range(part["num_angles"]):
            in_ring = (u >= edges[a]) & (u < edges[a + 1])
            if a == part["num_angles"] - 1:
                in_ring = (u >= edges[a]) & (theta < theta_c)
            n_az = part["num_azimuthal"]
            if max_step is not None:
                theta_ring = theta_of_u(0.5 * (edges[a] + edges[a + 1]))
                n_az = max(n_az, int(np.ceil(2 * np.pi * theta_ring / max_step)))
            offset = 0.5 * (a % 2)
            for b in range(n_az):
                phi_b = 2 * np.pi * (b + offset) / n_az
                dphi = np.angle(np.exp(1j * (phi - phi_b)))
                in_sector = np.abs(dphi) <= np.pi / n_az
                kernels.append(to_kernel(np.where(in_ring & in_sector, weight, 0.0)))
        return np.stack(kernels)

    single = partition_kernels(nodes["single"])
    extra = partition_kernels(nodes["extra"])

    from abtem.core.fft import fft2

    def transform(kernel):
        return fft2(xp.asarray(kernel.astype(get_dtype(complex=True))))

    return {
        "small": transform(small),
        "large": transform(large),
        "single": transform(single),
        "extra": transform(extra),
    }


def quadrature_plasmon_multislice_and_detect(
    waves: "Waves",
    potential: "BasePotential",
    plasmons: QuadraturePlasmons,
    detectors: list = None,
    algorithm=None,
    pbar: bool = False,
    potential_chunk_size: int | str = "auto",
):
    """
    Multislice algorithm with plasmon scattering evaluated by quadrature.

    One measurement per detector is returned with a leading
    :class:`PlasmonOrderAxis` resolving the number of plasmon excitations. The loss
    channels are formed from the diffraction patterns at the exit plane, so only
    detectors that reduce diffraction patterns are supported, and only the final
    exit plane.
    """
    from itertools import product
    from math import comb

    from abtem.antialias import AntialiasAperture
    from abtem.core.complex import complex_exponential
    from abtem.core.diagnostics import TqdmWrapper
    from abtem.core.energy import energy2wavelength
    from abtem.core.fft import fft2, ifft2
    from abtem.core.grid import spatial_frequencies
    from abtem.detectors import WavesDetector, validate_detectors
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
    if algorithm.conjugate or algorithm.transpose:
        raise NotImplementedError(
            "quadrature plasmon scattering does not support conjugate or transposed "
            "multislice"
        )

    order_axis = plasmons.order_axis
    n_orders = plasmons.num_orders

    waves = waves.ensure_real_space()
    detectors = validate_detectors(detectors)
    if any(isinstance(detector, WavesDetector) for detector in detectors):
        raise NotImplementedError(
            "quadrature plasmon scattering returns loss-order diffraction patterns, "
            "not wave functions; use a detector that reduces diffraction patterns"
        )
    if tuple(potential.exit_planes) != (potential.num_slices - 1,):
        raise NotImplementedError(
            "quadrature plasmon scattering resolves the loss orders at the final exit "
            "plane only; 'exit_planes' must be the default"
        )
    xp = get_array_module(waves.device)
    energy = waves._valid_energy
    wavelength = energy2wavelength(energy)
    gpts = tuple(waves._valid_gpts)
    sampling = tuple(waves._valid_sampling)
    extent = waves.extent
    angular_sampling = tuple(wavelength / e * 1e3 for e in extent)
    complex_dtype = get_dtype(complex=True)

    min_angle = plasmons.min_angle
    if min_angle is None:
        min_angle = max(angular_sampling)
    nodes = plasmons._angular_nodes(
        energy,
        min_angle,
        max_angular_step=plasmons.max_angular_step,
        event_max_angular_step=plasmons.event_max_angular_step,
    )
    p_small, p_large = nodes["p_small"], nodes["p_large"]
    max_tilt_events = plasmons.max_tilt_events
    num_depths = plasmons.num_depths
    single_tilts = nodes["single"]["tilts"]
    extra_tilts = nodes["extra"]["tilts"]
    single_weights = nodes["single"]["weights"]
    extra_weights = nodes["extra"]["weights"]
    n_single = len(single_tilts)
    n_extra = len(extra_tilts)

    if plasmons.lab_frame:
        kernels = _lorentzian_kernels(gpts, angular_sampling, nodes, xp=xp)
    else:
        kernels = None

    # first-event nodes per pass, from the memory budget
    if max_tilt_events == 0:
        first_node_chunks = [np.arange(0)]
    else:
        per_first = plasmons._copies_per_first_node(n_extra)
        if plasmons.max_copies is None:
            per_pass = n_single
        else:
            per_pass = max(1, plasmons.max_copies // per_first)
        first_node_chunks = [
            np.arange(start, min(start + per_pass, n_single))
            for start in range(0, n_single, per_pass)
        ]

    (
        extra_ensemble_axes_shape,
        extra_ensemble_axes_metadata,
    ) = _potential_ensemble_shape_and_metadata(potential)

    measurements = allocate_multislice_measurements(
        waves,
        detectors,
        (n_orders,) + extra_ensemble_axes_shape,
        [order_axis] + extra_ensemble_axes_metadata,
    )

    base_kwargs = waves._copy_kwargs(exclude=("array", "ensemble_axes_metadata"))
    base_axes = list(waves.ensemble_axes_metadata)
    n_base = len(base_axes)
    copy_axis = AxisMetadata(label="plasmon tilt copies")

    def make_waves(array, extra_axes):
        return waves.__class__(
            array=array,
            ensemble_axes_metadata=list(extra_axes) + base_axes,
            **base_kwargs,
        )

    thickness = potential.thickness
    depth_nodes = (np.arange(num_depths) + 0.5) * thickness / num_depths

    antialias_aperture = AntialiasAperture()
    propagator = FresnelPropagator()
    kx, ky = spatial_frequencies(gpts, sampling, xp=xp)
    kx = kx.astype(get_dtype(complex=False))
    ky = ky.astype(get_dtype(complex=False))

    def tilt_phases(tilts, slice_thickness):
        """Separable reciprocal-space phase factors of the tilted propagator."""
        tilts = xp.asarray(tilts, dtype=get_dtype(complex=False))
        tan_x = xp.tan(tilts[:, 0] / 1e3)[:, None]
        tan_y = xp.tan(tilts[:, 1] / 1e3)[:, None]
        ux = complex_exponential(-2 * np.pi * kx[None] * tan_x * slice_thickness)
        uy = complex_exponential(-2 * np.pi * ky[None] * tan_y * slice_thickness)
        return ux, uy

    member_slice = (slice(None),) + (None,) * n_base

    def step_group(group, transmission_function, slice_thickness):
        group_waves = transmission_function.transmit(group["waves"])
        kernel = propagator.get_array(
            group_waves, slice_thickness, order=algorithm.order
        )
        if group.get("phase_thickness") != slice_thickness:
            group["phases"] = tilt_phases(group["tilts"], slice_thickness)
            group["phase_thickness"] = slice_thickness
        ux, uy = group["phases"]
        array = fft2(group_waves.array, overwrite_x=True)
        array *= kernel
        array *= ux[member_slice + (slice(None), None)]
        array *= uy[member_slice + (None, slice(None))]
        group_waves._array = ifft2(array, overwrite_x=True)
        group["waves"] = group_waves

    def make_group(
        source_array, source_tilts, source_index, k, first_nodes, m_parent, weight
    ):
        """Chain ``k`` new events onto a parent (the elastic wave if m_parent == 0)."""
        if m_parent == 0:
            new_index = np.array(
                [
                    (i,) + e
                    for i in first_nodes
                    for e in product(range(n_extra), repeat=k - 1)
                ],
                dtype=int,
            ).reshape(-1, k)
            n_new = len(new_index)
            array = xp.tile(
                source_array[None], (n_new,) + (1,) * len(source_array.shape)
            )
            tilts = single_tilts[new_index[:, 0]] + sum(
                (extra_tilts[new_index[:, j]] for j in range(1, k)), np.zeros(2)
            )
        else:
            combos = np.array(
                list(product(range(n_extra), repeat=k)), dtype=int
            ).reshape(-1, k)
            n_parent = len(source_index)
            new_index = np.concatenate(
                [
                    np.repeat(source_index, len(combos), axis=0),
                    np.tile(combos, (n_parent, 1)),
                ],
                axis=1,
            )
            array = xp.repeat(source_array, len(combos), axis=0)
            tilts = np.repeat(source_tilts, len(combos), axis=0) + np.tile(
                extra_tilts[combos].sum(1), (n_parent, 1)
            )
        return {
            "waves": make_waves(array, [copy_axis]),
            "tilts": np.asarray(tilts, dtype=float).reshape(-1, 2),
            "index": new_index,
            "m": m_parent + k,
            "weight": weight,
        }

    def member_kernels(index):
        """Fourier kernels of the chain members (product of their node kernels)."""
        k = kernels["single"][index[:, 0]]
        for j in range(1, index.shape[1]):
            k = k * kernels["extra"][index[:, j]]
        return k

    accumulators = {}

    def add(exit_index, n, value):
        acc = accumulators.setdefault(exit_index, [None] * n_orders)
        acc[n] = value if acc[n] is None else acc[n] + value

    # powers of the momentum-transfer kernels for the excitations that are not
    # direction-changing events of a chain, computed once
    if kernels is None:
        one = xp.ones((), dtype=complex_dtype)
        ks_pow = kl_pow = [one] * n_orders
    else:
        ks_pow = [kernels["small"] ** j for j in range(n_orders)]
        kl_pow = [kernels["large"] ** j for j in range(n_orders)]

    def contribute(exit_index, f_pattern, m_group, weight):
        for n, m, factor in _loss_order_factors(
            n_orders, m_group, max_tilt_events, p_small, p_large
        ):
            term = f_pattern * (factor * weight)
            if n - m > 0:
                term = term * ks_pow[n - m]
            if m - m_group > 0:
                term = term * kl_pow[m - m_group]
            add(exit_index, n, term)

    def accumulate_elastic(exit_index, elastic):
        i0 = abs2(fft2(elastic.array, overwrite_x=False))
        contribute(exit_index, fft2(i0.astype(complex_dtype)), 0, 1.0)

    def accumulate_groups(exit_index, groups):
        for group in groups:
            arr = group["waves"].array
            intensity = abs2(fft2(arr, overwrite_x=False))
            f = fft2(intensity.astype(complex_dtype))
            if kernels is not None:
                f = f * member_kernels(group["index"])[member_slice]
            m = group["m"]
            index = group["index"]
            member_weights = single_weights[index[:, 0]]
            for level in range(1, index.shape[1]):
                member_weights = member_weights * extra_weights[index[:, level]]
            member_weights = xp.asarray(member_weights.astype(get_dtype(complex=False)))
            weight_slice = (slice(None),) + (None,) * (n_base + 2)
            f = (f * member_weights[weight_slice]).sum(0)
            contribute(exit_index, f, m, group["weight"])

    def finalize(exit_index, measurement_index, ens):
        acc = accumulators.pop(exit_index)
        patterns = []
        for n in range(n_orders):
            if acc[n] is None:
                patterns.append(xp.zeros(ens + gpts, dtype=get_dtype(complex=False)))
            else:
                patterns.append(xp.clip(ifft2(acc[n]).real, 0, None))
        intensity = xp.stack(patterns)
        # A wave function whose diffraction pattern is the accumulated intensity.
        carrier = ifft2(xp.sqrt(intensity).astype(complex_dtype))
        carrier_waves = make_waves(carrier, [order_axis])
        for i, detector in enumerate(detectors):
            new_measurement = detector.detect(carrier_waves)
            measurements[i].array[(slice(None),) + tuple(measurement_index)] += (
                new_measurement.array
            )

    n_waves = int(np.prod(waves.shape[:-2])) if len(waves.shape) > 2 else 1
    n_slices = int(
        n_waves
        * potential.num_slices
        * potential.num_configurations
        * max(1, len(first_node_chunks))
    )
    tqdm_pbar = TqdmWrapper(
        enabled=pbar, total=n_slices, leave=False, desc="multislice"
    )

    elastic_input = make_waves(waves.array, [])
    ens = tuple(elastic_input.array.shape[:-2])

    for potential_index, potential_configuration in _generate_potential_configurations(
        potential
    ):
        exit_indices = []
        passes = first_node_chunks if first_node_chunks else [np.arange(0)]
        for pass_index, first_nodes in enumerate(passes):
            first_pass = pass_index == 0
            elastic = elastic_input.copy()
            groups = []
            next_node = 0
            depth = 0.0
            exit_plane_index = 0

            if potential.exit_planes[0] == -1:
                if first_pass:
                    accumulate_elastic(exit_plane_index, elastic)
                    exit_indices.append(exit_plane_index)
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
                    slice_thickness = transmission_function.slice_thickness[0]

                    elastic = conventional_multislice_step(
                        elastic,
                        potential_slice=transmission_function,
                        antialias_aperture=antialias_aperture,
                        propagator=propagator,
                        order=algorithm.order,
                    )
                    for group in groups:
                        step_group(group, transmission_function, slice_thickness)

                    depth += slice_thickness
                    tqdm_pbar.update_if_exists(int(n_waves))

                    # spawn chains of tilted copies at the depth nodes
                    while next_node < num_depths and depth >= depth_nodes[next_node]:
                        next_node += 1
                        if max_tilt_events == 0 or len(first_nodes) == 0:
                            continue
                        new_groups = []
                        for group in groups:
                            for k in range(1, max_tilt_events - group["m"] + 1):
                                weight = (
                                    group["weight"]
                                    * comb(group["m"] + k, k)
                                    / num_depths**k
                                )
                                new_groups.append(
                                    make_group(
                                        group["waves"].array,
                                        group["tilts"],
                                        group["index"],
                                        k,
                                        first_nodes,
                                        group["m"],
                                        weight,
                                    )
                                )
                        for k in range(1, max_tilt_events + 1):
                            new_groups.append(
                                make_group(
                                    elastic.array,
                                    None,
                                    None,
                                    k,
                                    first_nodes,
                                    0,
                                    1.0 / num_depths**k,
                                )
                            )
                        groups.extend(new_groups)

                    if potential_slice.exit_planes:
                        if first_pass:
                            accumulate_elastic(exit_plane_index, elastic)
                            exit_indices.append(exit_plane_index)
                        accumulate_groups(exit_plane_index, groups)
                        exit_plane_index += 1

        for exit_plane_index in exit_indices:
            measurement_index = _validate_potential_ensemble_indices(
                potential_index, exit_plane_index, potential
            )
            finalize(exit_plane_index, measurement_index, ens)

    tqdm_pbar.close_if_exists()
    return measurements


def _config_rng(seed, potential_index, config_seed=None) -> np.random.Generator:
    """Deterministic per-configuration random generator for phase scrambling.

    With ``seed=None`` the streams are simply independent (fresh entropy per
    configuration), matching the reference implementation's per-repetition reshuffling.

    With an explicit ``seed`` the stream is reproducible and made unique per
    configuration. The preferred discriminator is ``config_seed`` -- the per-config
    frozen-phonon seed, which is globally unique and survives Dask partitioning (so
    lazy and eager runs agree). If it is unavailable (e.g. a potential without
    frozen-phonon seeds) the local ``potential_index`` is used as a best-effort
    fallback, which is only globally unique under eager execution.
    """
    if seed is None:
        return np.random.default_rng()

    entropy = [int(seed)]
    if config_seed is not None:
        entropy.append(int(config_seed))
    elif isinstance(potential_index, tuple):
        # Elements may be bare ints or (possibly array-wrapped) numpy scalars
        # from ``np.unravel_index`` -- ``np.asarray(i).item()`` handles both
        # without triggering NumPy's ndim>0-to-scalar deprecation warning.
        entropy.extend(int(np.asarray(i).item()) for i in potential_index)
    else:
        entropy.append(int(potential_index))

    return np.random.default_rng(np.random.SeedSequence(entropy))


class _PlasmonSliceOperator:
    """Inline per-slice plasmon scattering operator (single configuration).

    At the bottom of every slice one plasmon excitation is sampled: with probability
    ``slice_thickness / mean_free_path`` the electron is kicked by a transverse
    momentum drawn from the Lorentzian angular distribution, applied as a plane-wave
    factor with a random phase. Averaged over configurations, the intensity of the
    scrambled wave function has the same expectation value as the incoherent sum over
    scattering events; see :class:`PhaseScramblePlasmons`.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        extent: Tuple[float, float],
        wavelength: float,
        theta_e: float,
        theta_c: float,
        min_angle: float,
        mean_free_path: float,
        rng: np.random.Generator,
        xp,
    ):
        # x, y: real-space coordinates of the grid [Å] (1D, on the wave backend)
        self._x = x
        self._y = y
        self._extent = extent
        self._wavelength = wavelength
        self._u_max = np.log(1.0 + (theta_c / theta_e) ** 2)
        self._u_min = np.log(1.0 + (min_angle / theta_e) ** 2)
        self._theta_e = theta_e
        self._mean_free_path = mean_free_path
        self._rng = rng
        self._xp = xp

    def _draw_kick(self):
        """Draw one scattering angle from the Lorentzian and round it to the grid.

        Returns the separable plane-wave factors (or None below one pixel) and the
        random phase.
        """
        u = self._rng.random() * self._u_max
        phase = np.exp(2j * np.pi * self._rng.random())
        if u < self._u_min:
            return None, phase
        theta = self._theta_e * np.sqrt(np.expm1(u))
        phi = 2.0 * np.pi * self._rng.random()
        q = np.sin(theta) / self._wavelength  # [1/Å]
        i = int(round(q * np.cos(phi) * self._extent[0]))
        j = int(round(q * np.sin(phi) * self._extent[1]))
        if i == 0 and j == 0:
            return None, phase
        xp = self._xp
        complex_dtype = get_dtype(complex=True)
        ramp_x = xp.exp(2j * np.pi * (i / self._extent[0]) * self._x).astype(
            complex_dtype
        )
        ramp_y = xp.exp(2j * np.pi * (j / self._extent[1]) * self._y).astype(
            complex_dtype
        )
        return (ramp_x, ramp_y), phase

    def _kicked(self, psi, kick, phase):
        out = psi * psi.dtype.type(phase)
        if kick is not None:
            ramp_x, ramp_y = kick
            out = out * ramp_x[(None,) * (psi.ndim - 2) + (slice(None), None)]
            out = out * ramp_y[(None,) * (psi.ndim - 2) + (None, slice(None))]
        return out

    def _scatter_params(self, slice_thickness: float):
        scatter_prob = float(slice_thickness / self._mean_free_path)
        kick, phase = self._draw_kick()
        return scatter_prob, kick, phase

    def scatter(self, waves: "Waves", depth: float, slice_thickness: float) -> None:
        """Apply one slice of plasmon scattering to ``waves`` in place."""
        xp = self._xp
        scatter_prob, kick, phase = self._scatter_params(slice_thickness)
        psi = waves._array
        real_dtype = psi.real.dtype.type
        out = real_dtype(np.sqrt(1.0 - scatter_prob)) * psi
        out += real_dtype(np.sqrt(scatter_prob)) * self._kicked(psi, kick, phase)
        waves._array = xp.asarray(out, dtype=psi.dtype)

    def scatter_by_order(
        self,
        order_waves: list,
        depth: float,
        slice_thickness: float,
    ) -> None:
        """Apply order-resolved plasmon scattering in place.

        Maintains separate wave functions for each plasmon-loss order. At each
        slice the update rule is::

            ψ_0' = √(1-P) ψ_0
            ψ_n' = √(1-P) ψ_n  +  √P e^{iφ} e^{2πi q·r} ψ_{n-1}   for n ≥ 1

        with the same random kick ``q`` and phase ``φ`` for every order. Every channel
        holds exactly its order: intensity scattered beyond ``max_order`` leaves the
        set, so the sum over the channels reproduces the single-wave result minus the
        weight of the higher orders.
        """
        xp = self._xp
        scatter_prob, kick, phase = self._scatter_params(slice_thickness)
        max_order = len(order_waves) - 1
        real_dtype = order_waves[0]._array.real.dtype.type
        sqrt_one_minus_p = real_dtype(np.sqrt(1.0 - scatter_prob))
        sqrt_p = real_dtype(np.sqrt(scatter_prob))
        for n in range(max_order, -1, -1):
            arr = order_waves[n]._array
            if n > 0:
                prev_arr = order_waves[n - 1]._array
                new = sqrt_one_minus_p * arr + sqrt_p * self._kicked(
                    prev_arr, kick, phase
                )
            else:
                new = sqrt_one_minus_p * arr
            order_waves[n]._array = xp.asarray(new, dtype=arr.dtype)


def _valence_electrons_from_atoms(atoms) -> int:
    """Total number of valence electrons in ``atoms``, via the ``mendeleev`` package.

    ``mendeleev`` is an optional dependency; if it is not installed the caller
    should pass ``valence_electrons`` explicitly instead.
    """
    try:
        from mendeleev import element
    except ImportError as exc:  # pragma: no cover - exercised only without mendeleev
        raise ImportError(
            "Automatic valence-electron lookup requires the 'mendeleev' package "
            "(`pip install mendeleev`). Alternatively pass 'valence_electrons' "
            "explicitly (an int per atom, or a {symbol: count} mapping)."
        ) from exc

    symbols = atoms.get_chemical_symbols()
    per_species = {sym: element(sym).nvalence() for sym in set(symbols)}
    return int(sum(per_species[sym] for sym in symbols))


def estimate_plasmon_parameters(
    atoms,
    energy: float,
    valence_electrons: "int | dict | None" = None,
    method: str = "egerton",
) -> tuple[float, float, float]:
    """Estimate free-electron plasmon parameters for a material.

    Uses the free-electron (jellium) model to estimate the three inputs of
    :class:`PhaseScramblePlasmons` from the atomic structure and beam energy.

    - **Plasmon energy** :math:`E_p = \\hbar\\sqrt{n_e e^2 / (\\varepsilon_0 m_e)}`
      with the valence-electron density :math:`n_e` taken from the cell volume.
      This is accurate (≈1 %) for free-electron-like materials.
    - **Critical angle** from the Landau cut-off wavevector
      :math:`q_c = \\omega_p / v_F`, as :math:`\\theta_c = q_c / k_0`.
    - **Mean free path** — two methods are available (selected by *method*):

      ``"egerton"`` (default)
          Egerton's free-electron expression
          :math:`\\lambda_p = 2 a_0 / [\\gamma\\,\\theta_E \\ln(1 + \\theta_c^2/\\theta_E^2)]`.
          This is a pure plasmon MFP derived from the Kramers-Kronig sum rule.

      ``"malis"``
          The semi-empirical parameterization of Malis *et al.* (1988), Eq. 7:
          :math:`\\lambda = 106\\,F\\,E_0 / [E_m \\ln(2\\,\\beta\\,E_0 / E_m)]`
          with :math:`F = (1+E_0/1022)/(1+E_0/511)^2`,
          :math:`E_m = 7.6\\,Z_{\\mathrm{eff}}^{0.36}` eV, and
          :math:`\\beta = \\theta_c` (the critical angle returned by this
          function, in mrad). This is a *total* inelastic MFP fitted to
          measurements on 11 materials; for free-electron-like metals where
          plasmons dominate it gives values closer to experiment (~112 nm vs
          105 nm for Si at 200 kV) than the Egerton formula (~171 nm).

    .. warning::

        Only :math:`E_p` is reliable. The returned :math:`\\theta_c` is the
        *physical* plasmon dispersion cut-off (the Landau angle
        :math:`q_c = \\omega_p / v_F`, :math:`q_c \\approx 1.2`
        :math:`\\mathrm{\\AA^{-1}}` for Si, consistent with tabulated values),
        which is ~5 mrad at 200 kV. In practice :math:`\\theta_c` is **not
        computed from a formula**: in the Lorentzian model it is the upper
        truncation of :math:`P(\\theta) \\propto \\theta/(\\theta^2+\\theta_E^2)`
        and is **fitted to experiment**. Mendis (*Acta Cryst.* **A80**, 2024)
        uses :math:`\\theta_c = 19.1` mrad for Si at 200 kV (fitted by Barthel
        *et al.*, 2019, at 300 kV and scaled via :math:`q_c = K\\theta_c =`
        const — see :func:`scale_critical_angle`), ~4x the free-electron value.
        Likewise :math:`\\lambda_p` here is an order-of-magnitude estimate;
        the value used in the literature, 105 nm, is **experimentally measured**
        by EELS (Mendis, 2019), not computed. The ``"malis"`` method is closer
        for light/medium-Z materials where plasmons dominate, but it is a
        *total* inelastic MFP and will overestimate the plasmon scattering rate
        for heavy elements with strong core-loss contributions. For
        quantitative work, supply EELS-measured values via the override
        arguments of :meth:`PhaseScramblePlasmons.from_atoms`.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure; the (intensive) valence-electron density is taken from
        its cell volume, so a unit cell or a supercell give the same result.
    energy : float
        Electron beam energy [eV].
    valence_electrons : int or dict, optional
        Valence electrons per atom (int, applied to every atom) or a
        ``{chemical_symbol: count}`` mapping. If ``None`` (default), looked up
        per species via the optional ``mendeleev`` package.
    method : str, optional
        ``"egerton"`` (default) for the free-electron plasmon MFP, or
        ``"malis"`` for the Malis *et al.* (1988) semi-empirical total
        inelastic MFP.

    Returns
    -------
    excitation_energy : float
        Plasmon energy :math:`E_p` [eV].
    critical_angle : float
        Critical angle :math:`\\theta_c` [mrad].
    mean_free_path : float
        Plasmon mean free path :math:`\\lambda_p` [Å].
    """
    if method not in ("egerton", "malis"):
        raise ValueError(f"method must be 'egerton' or 'malis', got {method!r}")

    # Physical constants (SI).
    hbar = 1.054571817e-34
    m_e = 9.1093837015e-31
    e = 1.602176634e-19
    eps0 = 8.8541878128e-12
    c = 299792458.0
    a0 = 5.29177210903e-11

    symbols = atoms.get_chemical_symbols()
    if valence_electrons is None:
        total_valence = _valence_electrons_from_atoms(atoms)
    elif isinstance(valence_electrons, dict):
        total_valence = sum(valence_electrons[sym] for sym in symbols)
    else:
        total_valence = float(valence_electrons) * len(symbols)

    volume = atoms.get_volume() * 1e-30  # m^3
    n_e = total_valence / volume  # valence electrons per m^3

    omega_p = np.sqrt(n_e * e**2 / (eps0 * m_e))
    excitation_energy = hbar * omega_p / e  # eV

    k_F = (3.0 * np.pi**2 * n_e) ** (1.0 / 3.0)
    v_F = hbar * k_F / m_e

    gamma = 1.0 + energy * e / (m_e * c**2)
    v = c * np.sqrt(1.0 - 1.0 / gamma**2)
    k0 = gamma * m_e * v / hbar  # = 2*pi / lambda

    theta_c = (omega_p / v_F) / k0  # rad
    theta_E = excitation_energy * e / (gamma * m_e * v**2)  # rad

    if method == "egerton":
        mean_free_path = (
            2.0 * a0 / (gamma * theta_E * np.log(1.0 + (theta_c / theta_E) ** 2))
        )  # m
    else:
        # Malis et al. (1988) Eq. 7 — semi-empirical total inelastic MFP.
        # Z_eff from Eq. 4: Σ f_i Z_i^(1+r) / Σ f_i Z_i^r, r ≈ 0.3.
        E0_keV = energy / 1e3
        F_rel = (1.0 + E0_keV / 1022.0) / (1.0 + E0_keV / 511.0) ** 2
        Z = np.array(atoms.get_atomic_numbers(), dtype=float)
        f = np.ones(len(Z)) / len(Z)
        z_eff = float(np.sum(f * Z**1.3) / np.sum(f * Z**0.3))
        E_m = 7.6 * z_eff**0.36  # eV
        beta = theta_c * 1e3  # mrad (use the computed critical angle)
        mean_free_path = (
            106.0 * F_rel * E0_keV / (E_m * np.log(2.0 * beta * E0_keV / E_m))
        ) * 1e-9  # nm -> m

    return (
        float(excitation_energy),
        float(theta_c * 1e3),  # mrad
        float(mean_free_path * 1e10),  # Å
    )


def scale_critical_angle(
    critical_angle: float, energy_ref: float, energy: float
) -> float:
    """Scale a plasmon critical angle to a different beam energy.

    The plasmon cut-off is a property of the material — a fixed scattering vector
    :math:`q_c` — so the scattering vector :math:`q_c \\simeq K \\theta_c` is
    constant with beam energy and the critical *angle* scales with the electron
    wavelength,

    .. math::

        \\theta_c(E) = \\theta_c(E_\\mathrm{ref})\\,
        \\frac{\\lambda(E)}{\\lambda(E_\\mathrm{ref})}.

    This is exactly how Mendis (*Acta Cryst.* **A80**, 2024) transfers the Si
    critical angle fitted by Barthel *et al.* (2019) at 300 kV to 200 kV. Use it
    to bring a published/calibrated ``critical_angle`` to your own beam energy.

    Parameters
    ----------
    critical_angle : float
        Known critical angle :math:`\\theta_c` [mrad] at ``energy_ref``.
    energy_ref : float
        Beam energy [eV] at which ``critical_angle`` was determined.
    energy : float
        Target beam energy [eV].

    Returns
    -------
    critical_angle : float
        Critical angle [mrad] scaled to ``energy``.

    Examples
    --------
    >>> # Si fit of Barthel et al. (2019): theta_c = 19.1 mrad at 200 kV.
    >>> scale_critical_angle(19.1, 200e3, 300e3)  # to 300 kV  # doctest: +SKIP
    15.0
    """
    return float(
        critical_angle * energy2wavelength(energy) / energy2wavelength(energy_ref)
    )


class PhaseScramblePlasmons:
    """Stochastic single-pass plasmon energy-loss model (phase scrambling).

    Implements the plasmon-scattering model of B.G. Mendis, *Ultramicroscopy*
    **206** (2019) 112816 in the single-pass form of B.G. Mendis, *Microsc.
    Microanal.* **29** (2023) 1111: instead of running a separate multislice for
    every sampled scattering event (:class:`MonteCarloPlasmons`), one plasmon
    excitation is sampled at the bottom of every slice and added to the wave
    function with a random phase. At each slice the wave function is updated as

    .. math::
        \\psi \\rightarrow \\sqrt{1 - P}\\,\\psi
            + \\sqrt{P}\\, e^{i\\varphi}\\, e^{2\\pi i \\mathbf{q}\\cdot\\mathbf{r}}\\, \\psi,

    where :math:`P = \\Delta z / \\lambda_p` is the plasmon scattering probability of
    the slice, :math:`\\mathbf{q}` is a transverse momentum transfer drawn from
    the Lorentzian angular distribution :math:`P(\\theta) \\propto \\theta /
    (\\theta^2 + \\theta_E^2)` up to the critical angle and rounded to the
    reciprocal-space grid,
    and :math:`\\varphi` is a random phase. Transfers below ``min_angle`` (by default
    one reciprocal-space pixel) only carry the random phase. Averaged over
    repetitions, the intensity of the scrambled wave function has the same
    expectation value as the incoherent sum over scattering events, i.e. the Monte
    Carlo result; the interference between scattering paths that survives in a single
    repetition averages away as one over the square root of the number of
    repetitions. The repetitions are realised by the frozen-phonon configurations of
    the potential (``num_configs``), or by ``num_repetitions`` for a static structure.

    Compared with the original phase-scrambling algorithm, which superposes all
    azimuths of each of a few scattering angles with a random-order Bessel function,
    sampling one kick per slice keeps the full Lorentzian up to the critical angle,
    conserves the electron count in expectation (no renormalization), and gives the
    Poisson distribution of loss orders.

    Parameters
    ----------
    mean_free_path : float
        Plasmon mean free path :math:`\\lambda_p` [Å].
    excitation_energy : float
        Plasmon excitation (peak) energy :math:`E_p` [eV]. Sets the characteristic
        scattering angle :math:`\\theta_E = E_p / (2 E_0)`.
    critical_angle : float
        Critical (cut-off) scattering angle :math:`\\theta_c` [mrad], above which single
        electron excitations dominate.
    min_angle : float, optional
        Scattering angle [mrad] below which the momentum transfer is neglected (the
        excitation still counts as a loss event). If not given, one reciprocal-space
        pixel of the wave function grid is used.
    seed : int, optional
        Base random seed. Combined with the frozen-phonon configuration seed to give a
        reproducible, independent stream per configuration. If ``None`` (default),
        each configuration draws fresh entropy.
    max_loss_order : int, optional
        If set, the multislice loop maintains separate wave functions for each
        plasmon-loss order from 0 (zero loss) up to ``max_loss_order``, returning
        order-resolved diffraction patterns (every channel holds exactly its order;
        higher orders are dropped). If ``None`` (default), a single wave function
        accumulating all orders is propagated (faster, but only the total unfiltered
        signal is available).
    num_repetitions : int, optional
        Number of phase-scramble repetitions to incoherently average when the
        potential has **no** frozen phonons (a static structure). Each repetition
        reuses the same static potential with an independent phase scramble; the
        repetitions are realised as a zero-displacement frozen-phonon ensemble.
        Ignored when the potential already has frozen phonons, in which case its
        ``num_configs`` configurations serve as the repetitions. If ``None``
        (default) and the structure is static, a single repetition is run (no
        statistical averaging).
    """

    def __init__(
        self,
        mean_free_path: float,
        excitation_energy: float,
        critical_angle: float,
        min_angle: float = None,
        seed: int = None,
        max_loss_order: int = None,
        num_repetitions: int = None,
    ):
        self._mean_free_path = mean_free_path
        self._excitation_energy = excitation_energy
        self._critical_angle = critical_angle
        self._min_angle = None if min_angle is None else float(min_angle)
        self._seed = seed
        self._max_loss_order = max_loss_order
        self._num_repetitions = num_repetitions

    @property
    def parameters(self) -> dict:
        """The constructor arguments of the model."""
        return {
            "mean_free_path": self._mean_free_path,
            "excitation_energy": self._excitation_energy,
            "critical_angle": self._critical_angle,
            "min_angle": self._min_angle,
            "seed": self._seed,
            "max_loss_order": self._max_loss_order,
            "num_repetitions": self._num_repetitions,
        }

    @property
    def order_axis(self) -> Union[PlasmonOrderAxis, None]:
        """The loss-order axis of the measurements (None when unresolved)."""
        if self._max_loss_order is None:
            return None
        return PlasmonOrderAxis(
            values=tuple(ntuples[n] for n in range(self._max_loss_order + 1)),
            model="phase_scramble",
            parameters=self.parameters,
        )

    @classmethod
    def from_atoms(
        cls,
        atoms,
        energy: float,
        valence_electrons: "int | dict | None" = None,
        excitation_energy: float = None,
        critical_angle: float = None,
        mean_free_path: float = None,
        method: str = "egerton",
        **kwargs,
    ) -> "PhaseScramblePlasmons":
        """Construct a model with parameters estimated from a free-electron model.

        Convenience constructor that fills in ``excitation_energy``,
        ``critical_angle`` and ``mean_free_path`` from the atomic structure and
        beam energy via :func:`estimate_plasmon_parameters`. Any of the three may
        be overridden by passing it explicitly — recommended for ``critical_angle``
        and ``mean_free_path``, whose free-electron estimates are only
        order-of-magnitude (see the warning in
        :func:`estimate_plasmon_parameters`). Only ``excitation_energy`` is
        reliably estimated.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure used to estimate the valence-electron density.
        energy : float
            Electron beam energy [eV].
        valence_electrons : int or dict, optional
            Valence electrons per atom or a ``{symbol: count}`` mapping. If
            ``None`` (default), looked up via the optional ``mendeleev`` package.
        excitation_energy, critical_angle, mean_free_path : float, optional
            Explicit overrides ([eV], [mrad], [Å]). Any left as ``None`` is taken
            from the free-electron estimate.
        method : str, optional
            ``"egerton"`` (default) or ``"malis"`` — forwarded to
            :func:`estimate_plasmon_parameters` to select the MFP formula.
        kwargs
            Forwarded to :class:`PhaseScramblePlasmons` (``min_angle``, ``seed``,
            ``max_loss_order``, ``num_repetitions``).
        """
        est_energy, est_angle, est_mfp = estimate_plasmon_parameters(
            atoms, energy, valence_electrons, method=method
        )
        return cls(
            mean_free_path=est_mfp if mean_free_path is None else mean_free_path,
            excitation_energy=(
                est_energy if excitation_energy is None else excitation_energy
            ),
            critical_angle=est_angle if critical_angle is None else critical_angle,
            **kwargs,
        )

    @property
    def mean_free_path(self) -> float:
        return self._mean_free_path

    @property
    def excitation_energy(self) -> float:
        return self._excitation_energy

    @property
    def critical_angle(self) -> float:
        return self._critical_angle

    @property
    def seed(self):
        return self._seed

    @property
    def max_loss_order(self):
        return self._max_loss_order

    @property
    def num_repetitions(self):
        return self._num_repetitions

    @property
    def num_orders(self) -> int:
        """Number of loss channels including the zero loss."""
        if self._max_loss_order is None:
            raise ValueError(
                "the loss orders are not resolved by this model; set 'max_loss_order'"
            )
        return self._max_loss_order + 1

    @property
    def order_labels(self) -> Tuple[str, ...]:
        """Label of every loss channel, in the order of the loss-order axis."""
        return tuple(ntuples[n] for n in range(self.num_orders))

    def excitation_weights(self, thickness: float) -> Tuple[float, ...]:
        """Poisson probability of each loss order at a thickness [Å].

        Unlike :class:`MonteCarloPlasmons` and :class:`QuadraturePlasmons`, whose
        channels are each normalized to the incident electron count, the channels of
        the phase-scramble model carry these weights already: the electron splits
        between the orders as it propagates. Divide channel ``n`` by weight ``n`` to
        put this model on the same footing as the other two.
        """
        return tuple(
            excitations_weights(n, thickness, self._mean_free_path)
            for n in range(self.num_orders)
        )

    def expand_static_potential(self, potential):
        """Return a potential providing the phase-scramble repetitions.

        The phase-scramble method draws statistical convergence from an ensemble
        of independent scrambles, normally realised by the potential's
        frozen-phonon configurations. When the structure is static (no frozen
        phonons) and ``num_repetitions`` is set, the same static potential is
        reused for every repetition: it is represented as a zero-displacement
        frozen-phonon ensemble of ``num_repetitions`` configurations, each
        seeded independently so it receives its own phase scramble. Potentials
        that already carry frozen phonons (or do not expose their atoms) are
        returned unchanged.
        """
        if self._num_repetitions is None:
            return potential

        if potential.num_configurations > 1:
            warnings.warn(
                "'num_repetitions' is ignored because the potential already has "
                "frozen phonons; its configurations serve as the phase-scramble "
                "repetitions."
            )
            return potential

        frozen_phonons = getattr(potential, "frozen_phonons", None)
        atoms = getattr(frozen_phonons, "atoms", None)
        if atoms is None:
            raise ValueError(
                "'num_repetitions' requires a potential built from atoms (so the "
                "static structure can be repeated). Pass scattering atoms or a "
                "frozen-phonon potential instead."
            )

        from abtem.inelastic.phonons import FrozenPhonons

        repeated = FrozenPhonons(
            atoms,
            num_configs=self._num_repetitions,
            sigmas=0.0,
            seed=self._seed,
        )
        kwargs = potential._copy_kwargs(exclude=("atoms",))
        return potential.__class__(repeated, **kwargs)

    def _build_operator(
        self, waves: "Waves", potential_index=0, config_seed=None
    ) -> _PlasmonSliceOperator:
        """Build the per-configuration slice operator for the given wave functions."""
        extent = tuple(waves.extent)
        gpts = tuple(waves.gpts)
        wavelength = energy2wavelength(waves._valid_energy)  # [Å]
        rng = _config_rng(self._seed, potential_index, config_seed)

        theta_e = characteristic_angle(self._excitation_energy, waves._valid_energy)
        theta_e = theta_e * 1e-3  # [rad]
        theta_c = self._critical_angle * 1e-3
        if self._min_angle is None:
            min_angle = wavelength * max(1.0 / extent[0], 1.0 / extent[1])
        else:
            min_angle = self._min_angle * 1e-3

        xp = get_array_module(waves.device)
        real_dtype = get_dtype(complex=False)
        x = xp.asarray(np.arange(gpts[0]) * (extent[0] / gpts[0]), dtype=real_dtype)
        y = xp.asarray(np.arange(gpts[1]) * (extent[1] / gpts[1]), dtype=real_dtype)
        return _PlasmonSliceOperator(
            x=x,
            y=y,
            extent=extent,
            wavelength=wavelength,
            theta_e=theta_e,
            theta_c=theta_c,
            min_angle=min_angle,
            mean_free_path=self._mean_free_path,
            rng=rng,
            xp=xp,
        )


def _tds_differential_cross_section(
    theta: np.ndarray,
    scattering_factor_func,
    debye_waller_factor: float,
    energy: float,
) -> np.ndarray:
    """Evaluate the uncorrelated phonon (TDS) differential scattering cross section
    ``dσ/dΩ = f(q)² [1 − exp(−2Bq²)]`` [Mendis Eq. 8, Pennycook & Jesson 1991].

    Parameters
    ----------
    theta : np.ndarray
        Polar scattering angles [rad].
    scattering_factor_func : callable
        Electron scattering factor ``f(g²)`` as a function of the squared scattering
        vector magnitude ``g² = q²`` [1/Å²].
    debye_waller_factor : float
        The isotropic Debye-Waller factor ``B = 8π²⟨u²⟩`` [Å²].
    energy : float
        The electron energy [eV].

    Returns
    -------
    np.ndarray
        The differential cross section (unnormalised), same shape as ``theta``.
    """
    from abtem.core.energy import energy2wavelength

    wavelength = energy2wavelength(energy)
    K = 1.0 / wavelength
    q = 2 * K * np.sin(theta / 2.0)
    q2 = q**2
    f = scattering_factor_func(q2)
    return f**2 * (1.0 - np.exp(-2.0 * debye_waller_factor * q2))


def _compute_tds_cdf(
    scattering_factor_func,
    debye_waller_factor: float,
    energy: float,
    theta_max: float,
    num_points: int = 2000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Numerically compute the CDF of the phonon polar scattering angle distribution
    [Mendis Eq. 11].

    Returns ``(theta_grid, cdf_values, sigma_total)`` where ``cdf_values`` goes from
    0 to 1 and ``sigma_total`` is the total TDS cross section.
    """
    theta = np.linspace(0, theta_max, num_points)
    dsigma = _tds_differential_cross_section(
        theta,
        scattering_factor_func,
        debye_waller_factor,
        energy,
    )
    integrand = dsigma * np.sin(theta) * 2 * np.pi
    dtheta = theta[1] - theta[0]
    sigma_total = float(np.trapezoid(integrand, dx=dtheta))
    cdf = np.cumsum(integrand)
    cdf[0] = 0.0
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    return theta, cdf, sigma_total


def _draw_phonon_radial_angle(
    theta_grid: np.ndarray,
    cdf: np.ndarray,
    num_samples: int,
    num_depths: int,
    rng,
) -> Tuple[Tuple[float]]:
    """Draw phonon polar scattering angles by inverse-CDF sampling [Mendis Eq. 11]."""
    if num_depths == 0:
        return tuple(() for _ in range(num_samples))
    rands = rng.random((num_samples, num_depths))
    thetas_flat = np.interp(rands.ravel(), cdf, theta_grid)
    # ``theta_grid`` is in radians; the scattering events carry milliradians, as the
    # plasmon events do.
    thetas_2d = thetas_flat.reshape(num_samples, num_depths) * 1e3
    return tuple(tuple(row) for row in thetas_2d)


class MonteCarloPhonons:
    """Monte-Carlo phonon (thermal diffuse) scattering for Bloch waves.

    Uses the uncorrelated phonon model of Mendis (Acta Cryst. A80, 2024), Eq. 8–11 and
    16a–16c. The TDS differential cross section is ``dσ/dΩ = f(q)²[1 − exp(−2Bq²)]``
    (Pennycook & Jesson, 1991). The mean free path is ``λ_ph = 1/(Nᵥ σ_TDS^T)``
    [Eq. 9]. The polar angle is drawn by numerical inversion of the CDF [Eq. 11].

    The returned :class:`PlasmonScatteringEvents` object has the same format as
    plasmon events and can be consumed by the Bloch-wave inelastic driver directly.

    Parameters
    ----------
    atoms : Atoms
        The atoms object describing the structure (used for scattering factors and
        number density).
    thermal_sigma : float
        The isotropic r.m.s. thermal vibration amplitude ``σ = √⟨u²⟩`` [Å].
    parametrization : str
        The scattering-factor parametrization (``'lobato'``, ``'kirkland'``, etc.).
    theta_max : float
        The maximum polar scattering angle [rad] for the cross-section integration.
        Should cover the range where ``dσ/dΩ`` is significant.
    num_excitations : int or tuple of int
        The excitation orders to sample.
    num_samples : int
        The number of Monte-Carlo configurations per order.
    ensemble_mean : bool
        Whether to average over configurations when reducing.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        atoms,
        thermal_sigma: float,
        parametrization: str = "kirkland",
        theta_max: float = 0.1,
        num_excitations: Union[int, Tuple[int, ...]] = None,
        num_samples: int = None,
        ensemble_mean: bool = False,
        seed: Union[int, Tuple[int, ...]] = None,
    ):
        from ase import Atoms as AseAtoms

        if not isinstance(atoms, AseAtoms):
            raise TypeError("atoms must be an ASE Atoms object")

        self._atoms = atoms
        self._thermal_sigma = thermal_sigma
        self._parametrization_name = parametrization
        self._theta_max = theta_max
        self._ensemble_mean = ensemble_mean
        self._num_samples = num_samples
        self._seed = seed

        if isinstance(num_excitations, int):
            num_excitations = tuple(range(num_excitations + 1))
        self._num_excitations = num_excitations

        self._debye_waller_factor = 8.0 * np.pi**2 * thermal_sigma**2

    @property
    def debye_waller_factor(self) -> float:
        return self._debye_waller_factor

    @property
    def ensemble_mean(self) -> bool:
        return self._ensemble_mean

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def seed(self):
        return self._seed

    def _get_scattering_factor_func(self):
        """Return a callable ``f(g²)`` that sums the scattering factors of all atom
        species weighted by their fractional composition."""
        from abtem.parametrizations import validate_parametrization

        param = validate_parametrization(self._parametrization_name)

        symbols = self._atoms.get_chemical_symbols()
        unique_symbols = list(dict.fromkeys(symbols))
        counts = {s: symbols.count(s) for s in unique_symbols}
        total = len(symbols)

        funcs = {s: param.scattering_factor(s) for s in unique_symbols}

        def weighted_f(g2):
            result = np.zeros_like(g2, dtype=float)
            for s in unique_symbols:
                result += (counts[s] / total) * funcs[s](g2)
            return result

        return weighted_f

    def mean_free_path(self, energy: float) -> float:
        """Compute the phonon mean free path ``λ_ph = 1/(Nᵥ σ_TDS^T)`` [Eq. 9]."""

        f_func = self._get_scattering_factor_func()
        theta_grid = np.linspace(0, self._theta_max, 2000)

        dsigma = _tds_differential_cross_section(
            theta_grid,
            f_func,
            self._debye_waller_factor,
            energy,
        )
        integrand = dsigma * np.sin(theta_grid) * 2 * np.pi
        dtheta = theta_grid[1] - theta_grid[0]
        sigma_total = np.trapezoid(integrand, dx=dtheta)

        cell_volume = self._atoms.get_volume()
        num_atoms = len(self._atoms)
        number_density = num_atoms / cell_volume

        if sigma_total <= 0:
            return np.inf

        return 1.0 / (number_density * sigma_total)

    def _draw_events(self, thickness: float, energy: float) -> PlasmonScatteringEvents:
        """Draw Monte-Carlo phonon scattering events."""
        f_func = self._get_scattering_factor_func()
        theta_grid, cdf, sigma_total = _compute_tds_cdf(
            f_func,
            self._debye_waller_factor,
            energy,
            self._theta_max,
        )

        number_density = len(self._atoms) / self._atoms.get_volume()
        mfp = 1.0 / (number_density * sigma_total) if sigma_total > 0 else np.inf

        rng = np.random.default_rng(self.seed)

        depths = []
        radial_angles = []
        azimuthal_angles = []
        weights = []

        for n in self._num_excitations:
            if n == 0:
                ns = 1
            else:
                ns = self.num_samples

            depths.append(
                draw_scattering_depths(
                    mean_free_path=mfp,
                    num_depths=n,
                    max_depth=thickness,
                    num_samples=ns,
                    rng=rng,
                )
            )

            radial_angles.append(
                _draw_phonon_radial_angle(
                    theta_grid,
                    cdf,
                    num_samples=ns,
                    num_depths=n,
                    rng=rng,
                )
            )

            azimuthal_angles.append(
                draw_azimuthal_angle(num_samples=ns, num_depths=n, rng=rng)
            )

            weights.append((excitations_weights(n, thickness, mfp),) * ns)

        depths = list(itertools.chain(*depths))
        radial_angles = list(itertools.chain(*radial_angles))
        azimuthal_angles = list(itertools.chain(*azimuthal_angles))
        weights = list(itertools.chain(*weights))

        return PlasmonScatteringEvents(
            depths,
            radial_angles,
            azimuthal_angles,
            weights,
            ensemble_mean=self._ensemble_mean,
        )
