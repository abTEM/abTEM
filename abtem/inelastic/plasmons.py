import itertools
import warnings
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
    _iterate_axes_type,
)
from abtem.core.backend import get_array_module
from abtem.core.chunks import chunk_ranges, validate_chunks
from abtem.core.energy import energy2wavelength
from abtem.core.grid import coordinate_grid
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
        return cls(**kwargs)

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

        return PlasmonScatteringEvents(
            depths,
            radial_angles,
            azimuthal_angles,
            weights,
            ensemble_mean=self.ensemble_mean,
        )


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

    Holds the precomputed, phase-scrambled tilted-beam basis (random-order Bessel
    functions ``J_n(2*pi*k_t*R)``) and the per-configuration random generator. The
    operator is applied to the real-space wave function at the bottom of every slice
    during the multislice loop; see :class:`PhaseScramblePlasmons`.
    """

    def __init__(
        self,
        bessel_stack: np.ndarray,
        angle_weights: np.ndarray,
        azimuthal_norm: float,
        mean_free_path: float,
        rng: np.random.Generator,
    ):
        # bessel_stack: (num_angles, num_copies, gpts_x, gpts_y), real valued
        self._bessel_stack = bessel_stack
        self._angle_weights = angle_weights  # P(theta) per angle bin
        self._azimuthal_norm = azimuthal_norm  # sqrt(2*pi/phi_min)
        self._mean_free_path = mean_free_path
        self._rng = rng

    def _scatter_params(self, depth: float, slice_thickness: float):
        """Compute per-slice scatter probability and draw random Bessel copies."""
        lp = self._mean_free_path
        scatter_prob = float(np.exp(-depth / lp) * (slice_thickness / lp))
        num_angles, num_copies = self._bessel_stack.shape[:2]
        chosen = [int(self._rng.integers(num_copies)) for _ in range(num_angles)]
        return scatter_prob, chosen

    def scatter(self, waves: "Waves", depth: float, slice_thickness: float) -> None:
        """Apply one slice of plasmon scattering to ``waves`` in place (real space)."""
        xp = get_array_module(waves.device)
        scatter_prob, chosen = self._scatter_params(depth, slice_thickness)
        num_angles = self._bessel_stack.shape[0]

        psi = waves._array
        # Keep scalar coefficients at the wave's real precision; a Python/NumPy
        # float64 scalar times a complex64 array would upcast the (large)
        # intermediates to complex128, defeating ``config['precision']``.
        real_dtype = psi.real.dtype.type

        # ``psi`` is only read below — the scattered wave is accumulated in a
        # fresh array ``out`` — so no defensive copy of ``psi`` is needed.
        out = real_dtype(np.sqrt(1.0 - scatter_prob)) * psi
        for a in range(num_angles):
            bessel = self._bessel_stack[a, chosen[a]]
            amplitude = np.sqrt(scatter_prob * self._angle_weights[a])
            coeff = real_dtype(amplitude * self._azimuthal_norm)
            out += coeff * (bessel * psi)

        waves._array = xp.asarray(out, dtype=psi.dtype)

    def scatter_by_order(
        self,
        order_waves: list,
        depth: float,
        slice_thickness: float,
    ) -> None:
        """Apply order-resolved plasmon scattering in place.

        Maintains separate wave functions for each plasmon-loss order.  At each
        slice the update rule is::

            ψ_0' = √(1-P) ψ_0
            ψ_n' = √(1-P) ψ_n  +  S(ψ_{n-1})   for n ≥ 1

        where S is the phase-scramble scattering operator (sum over angle bins).
        The same random Bessel copy draw is shared across all orders so that the
        sum  Σ_n ψ_n  reproduces the single-pass result (up to truncation at
        ``max_order``).

        Parameters
        ----------
        order_waves : list of Waves
            ``[ψ_0, ψ_1, …, ψ_N]`` — one Waves object per loss order.
            Modified in place.
        depth : float
            Cumulative depth at the bottom of the current slice [Å].
        slice_thickness : float
            Thickness of the current slice [Å].
        """
        xp = get_array_module(order_waves[0].device)
        scatter_prob, chosen = self._scatter_params(depth, slice_thickness)
        num_angles = self._bessel_stack.shape[0]
        max_order = len(order_waves) - 1

        # Cast scalars to the wave's real precision so a float64 scalar does not
        # upcast the complex64 channels/kernel to complex128 (see ``scatter``).
        real_dtype = order_waves[0]._array.real.dtype.type

        scatter_kernel = xp.zeros(
            self._bessel_stack.shape[2:], dtype=self._bessel_stack.dtype
        )
        for a in range(num_angles):
            amplitude = np.sqrt(scatter_prob * self._angle_weights[a])
            scatter_kernel += (
                real_dtype(amplitude * self._azimuthal_norm)
                * self._bessel_stack[a, chosen[a]]
            )

        sqrt_one_minus_p = real_dtype(np.sqrt(1.0 - scatter_prob))

        for n in range(max_order, -1, -1):
            arr = order_waves[n]._array
            if n > 0:
                prev_arr = order_waves[n - 1]._array
                order_waves[n]._array = xp.asarray(
                    sqrt_one_minus_p * arr + scatter_kernel * prev_arr,
                    dtype=arr.dtype,
                )
            else:
                order_waves[n]._array = xp.asarray(
                    sqrt_one_minus_p * arr, dtype=arr.dtype
                )


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
        raise ValueError(
            f"method must be 'egerton' or 'malis', got {method!r}"
        )

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
        mean_free_path = 2.0 * a0 / (
            gamma * theta_E * np.log(1.0 + (theta_c / theta_E) ** 2)
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
    """Fast single-pass plasmon energy-loss model (phase-scramble method).

    Implements the plasmon-scattering model of B.G. Mendis, *Ultramicroscopy*
    **206** (2019) 112816 (and its 2020 corrigendum), which B.G. Mendis,
    *Microsc. Microanal.* **29** (2023) 1111 cites and reuses directly rather
    than presenting an independent plasmon method. In contrast to
    :class:`MonteCarloPlasmons` — the 2019 paper's own original implementation,
    which runs a separate full multislice for every sampled scattering event —
    this model applies the same plasmon-scattering physics *inline* at the
    bottom of every slice within a single multislice pass, so all plasmon
    orders accumulate simultaneously. Statistical convergence is obtained by
    *incoherently* averaging over repetitions, realised by reusing the
    frozen-phonon configuration ensemble of the potential (``num_configs`` plays
    the role of the number of repetitions) — ordinary ensemble averaging, not
    the coherent multi-configuration combination (eq. 7c) that the 2023 paper
    introduces for phonon disorder; see :func:`abtem.CrystalPotential`'s
    ``mixing="phase_scramble"``.

    At the bottom of each slice the real-space wave function ``psi`` is updated as

    .. math::

        \\psi \\rightarrow \\sqrt{1 - P_s}\\,\\psi
            + \\sqrt{\\tfrac{2\\pi}{\\phi_{min}}}
              \\sum_a \\sqrt{P_s\\,P(\\theta_a)}\\, J_{n}(2\\pi k_{t,a} R)\\, \\psi,

    where :math:`P_s = e^{-s/\\lambda_p}\\,\\Delta z/\\lambda_p` is the plasmon
    scattering probability for the slice at depth :math:`s`, :math:`P(\\theta_a)` is the
    (Lorentzian) angular scattering probability, and the random-order Bessel functions
    represent azimuthally-scrambled tilted beams with transverse wavenumber
    :math:`k_{t,a} = k\\sin\\theta_a`.

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
    num_angles : int, optional
        Number of discrete scattering-angle bins (default 5).
    num_copies : int, optional
        Number of independent random-order Bessel realisations per angle bin to draw
        from during phase scrambling (default 5).
    max_bessel_order : float, optional
        Maximum (non-integer) Bessel-function order used for phase scrambling
        (default 30).
    seed : int, optional
        Base random seed. Combined with the frozen-phonon configuration index to give a
        reproducible, per-configuration scramble (eager execution). If ``None``
        (default), each configuration draws fresh entropy, giving independent scramble
        streams in both eager and lazy execution (matching the reference
        implementation's per-repetition reshuffling).
    max_loss_order : int, optional
        If set, the multislice loop maintains separate wave functions for each
        plasmon-loss order from 0 (zero loss) up to ``max_loss_order``, returning
        order-resolved diffraction patterns.  If ``None`` (default), a single wave
        function accumulating all orders is propagated (faster, but only the total
        unfiltered signal is available).
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
        num_angles: int = 5,
        num_copies: int = 5,
        max_bessel_order: float = 30.0,
        seed: int = None,
        max_loss_order: int = None,
        num_repetitions: int = None,
    ):
        self._mean_free_path = mean_free_path
        self._excitation_energy = excitation_energy
        self._critical_angle = critical_angle
        self._num_angles = num_angles
        self._num_copies = num_copies
        self._max_bessel_order = max_bessel_order
        self._seed = seed
        self._max_loss_order = max_loss_order
        self._num_repetitions = num_repetitions
        # Single-entry cache of the unique-radii decomposition, reused across
        # configurations sharing the same grid (geometry is config-independent).
        self._radial_cache = None

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
            Forwarded to :class:`PhaseScramblePlasmons` (``num_angles``,
            ``num_copies``, ``max_bessel_order``, ``seed``, ``max_loss_order``).
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
        """Build the per-configuration slice operator for the given wave functions.

        Precomputes the radial-distance image, the discrete scattering angles, the
        Lorentzian angular weights and the random-order Bessel-function basis on the
        host (SciPy provides non-integer-order Bessel functions), transferring the basis
        to the wave backend once per configuration.
        """
        from scipy.special import jv

        extent = waves.extent
        gpts = waves.gpts
        wavelength = energy2wavelength(waves.energy)  # [Å]
        k = 1.0 / wavelength  # [1/Å]

        rng = _config_rng(self._seed, potential_index, config_seed)

        # Characteristic and critical angles [rad].
        theta_E = self._excitation_energy / (2.0 * waves.energy)
        theta_c = self._critical_angle * 1e-3

        # Pixel-limited angular step and discrete scattering angles (Matlab reference).
        reciprocal_step = wavelength * min(1.0 / extent[0], 1.0 / extent[1])
        theta = (np.arange(self._num_angles) + 0.5) * reciprocal_step
        phi_min = reciprocal_step / theta[-1]
        azimuthal_norm = float(np.sqrt(2.0 * np.pi / phi_min))

        # Normalised Lorentzian angular scattering probability P(theta_a) (Eq. 3).
        lorentz_norm = np.log(1.0 + (theta_c / theta_E) ** 2)
        angle_weights = (
            2.0 * theta * reciprocal_step / (theta**2 + theta_E**2) / lorentz_norm
        )

        # ``J_n(2*pi*k_t*R)`` is radially symmetric, so it takes only as many
        # distinct values as there are distinct radii — far fewer than the
        # ``gpts**2`` pixels. Evaluate each (expensive, non-integer-order) Bessel
        # function on the unique radii and scatter the result back to the grid;
        # this is the dominant cost of building the operator. The decomposition
        # depends only on the grid, so it is cached across configurations.
        cache_key = (tuple(gpts), tuple(extent))
        if self._radial_cache is not None and self._radial_cache[0] == cache_key:
            unique_radial, inverse = self._radial_cache[1]
        else:
            origin = (extent[0] / 2.0, extent[1] / 2.0)
            x, y = coordinate_grid(extent, gpts, origin=origin, endpoint=False)
            radial = np.sqrt(x**2 + y**2)
            unique_radial, inverse = np.unique(radial.ravel(), return_inverse=True)
            # int32 indices halve the cached map and the per-beam gather traffic
            # (grids never approach the 2**31-pixel limit).
            inverse = inverse.reshape(-1).astype(np.int32)
            self._radial_cache = (cache_key, (unique_radial, inverse))

        real_dtype = get_dtype(complex=False)
        bessel_stack = np.empty(
            (self._num_angles, self._num_copies) + tuple(gpts), dtype=real_dtype
        )
        for a in range(self._num_angles):
            kt = k * np.sin(theta[a])
            argument = 2.0 * np.pi * kt * unique_radial
            for c in range(self._num_copies):
                order = self._max_bessel_order * rng.random()
                values = jv(order, argument).astype(real_dtype)
                bessel_stack[a, c] = values[inverse].reshape(tuple(gpts))

        xp = get_array_module(waves.device)
        bessel_stack = xp.asarray(bessel_stack)

        return _PlasmonSliceOperator(
            bessel_stack=bessel_stack,
            angle_weights=angle_weights,
            azimuthal_norm=azimuthal_norm,
            mean_free_path=self._mean_free_path,
            rng=rng,
        )
