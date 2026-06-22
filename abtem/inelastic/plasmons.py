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
    _iterate_axes_type,
)
from abtem.core.backend import get_array_module
from abtem.core.chunks import chunk_ranges, validate_chunks
from abtem.core.utils import itemset
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

    def _calculate_new_array(self, array_object):
        # ``PlasmonScatteringEvents`` produces its output through the overridden
        # ``apply`` (which tiles the incident waves over the ensemble) rather than the
        # generic ``_calculate_new_array`` path. The method is implemented only so the
        # class is concrete and can be instantiated.
        raise NotImplementedError(
            "PlasmonScatteringEvents uses 'apply' directly; '_calculate_new_array' is "
            "not used."
        )

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
        return self._draw_events(thickness=potential.thickness, energy=waves.energy)

    def _draw_events(
        self, thickness: float, energy: float
    ) -> PlasmonScatteringEvents:
        """Draw Monte-Carlo plasmon scattering events for a given specimen thickness and
        electron energy.

        This is the object-agnostic core of :meth:`draw_events`; it does not require a
        ``Waves`` or ``BasePotential`` object and is used by the Bloch-wave inelastic
        driver.

        Parameters
        ----------
        thickness : float
            The specimen thickness [Å].
        energy : float
            The electron energy [eV].

        Returns
        -------
        PlasmonScatteringEvents
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
        theta, scattering_factor_func, debye_waller_factor, energy,
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
    thetas_2d = thetas_flat.reshape(num_samples, num_depths)
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
        from abtem.core.energy import energy2wavelength

        f_func = self._get_scattering_factor_func()
        theta_grid = np.linspace(0, self._theta_max, 2000)

        dsigma = _tds_differential_cross_section(
            theta_grid, f_func, self._debye_waller_factor, energy,
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

    def _draw_events(
        self, thickness: float, energy: float
    ) -> PlasmonScatteringEvents:
        """Draw Monte-Carlo phonon scattering events."""
        f_func = self._get_scattering_factor_func()
        theta_grid, cdf, sigma_total = _compute_tds_cdf(
            f_func, self._debye_waller_factor, energy, self._theta_max,
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
                    theta_grid, cdf, num_samples=ns, num_depths=n, rng=rng,
                )
            )

            azimuthal_angles.append(
                draw_azimuthal_angle(num_samples=ns, num_depths=n, rng=rng)
            )

            weights.append(
                (excitations_weights(n, thickness, mfp),) * ns
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
            ensemble_mean=self._ensemble_mean,
        )
