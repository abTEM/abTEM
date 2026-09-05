"""X-ray detection for EDX simulations.

This module provides :class:`XrayDetector`, which carries the radiometric chain
converting an ionisation probability into detected X-ray counts:

.. math::
    I_L = P_\\mathrm{ion} \\times \\omega \\times b_L
          \\times \\frac{\\Omega}{4\\pi} \\times \\varepsilon(E_L)

where :math:`\\omega` is the fluorescence yield of the ionised subshell,
:math:`b_L` the radiative branching ratio into line :math:`L`, :math:`\\Omega`
the collected solid angle and :math:`\\varepsilon` the detection efficiency.
Emission is assumed isotropic and specimen self-absorption is neglected.

.. warning::
    The multislice path that computes :math:`P_\\mathrm{ion}` integrated over all
    scattering angles and over the whole ionisation edge is not yet implemented.
    :meth:`XrayDetector.to_counts` will convert any ionisation measurement it is
    given, but a measurement produced at a single continuum energy
    ``epsilon`` is differential in energy loss and is therefore meaningful only
    up to an overall scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, cast

import numpy as np

from abtem.core.axes import OrdinalAxis
from abtem.detectors import IonizationDetector
from abtem.core.utils import CopyMixin, get_dtype
from abtem.inelastic.xray_data import (
    EmissionLine,
    emission_lines,
    line_families,
)

if TYPE_CHECKING:
    from abtem.measurements import BaseMeasurements

__all__ = [
    "XrayDetector",
    "TabulatedEfficiency",
    "SDDEfficiency",
    "SpecimenAbsorption",
]


_INSTALL_HINT = (
    "Modelling detector efficiency from absorber thicknesses requires xraydb. "
    "Install it with `pip install abtem[gpaw]` or `pip install xraydb`. "
    "Alternatively pass a measured efficiency curve to XrayDetector."
)


def _validate_layer(layer, name: str) -> Optional[tuple[str, float]]:
    if layer is None:
        return None

    try:
        formula, thickness = layer
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"'{name}' must be a (formula, thickness) pair with the thickness "
            f"in micrometres, got {layer!r}"
        ) from e

    thickness = float(thickness)
    if thickness < 0.0:
        raise ValueError(f"'{name}' thickness must be non-negative, got {thickness}")

    return str(formula), thickness


@dataclass(frozen=True)
class TabulatedEfficiency:
    """
    Detection efficiency interpolated from a measured or tabulated curve.

    Parameters
    ----------
    energies : array of float
        Photon energies [eV], in increasing order.
    values : array of float
        Efficiency at each energy, between 0 and 1.

    Notes
    -----
    Energies outside the tabulated range are clamped to the end values.
    """

    energies: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        energies = np.asarray(self.energies, dtype=get_dtype())
        values = np.asarray(self.values, dtype=get_dtype())

        if energies.ndim != 1 or values.ndim != 1:
            raise ValueError("'energies' and 'values' must be one-dimensional")

        if energies.shape != values.shape:
            raise ValueError(
                f"'energies' and 'values' must have the same shape, got "
                f"{energies.shape} and {values.shape}"
            )

        if not np.all(np.diff(energies) > 0):
            raise ValueError("'energies' must be strictly increasing")

        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "values", values)

    def __call__(self, energies) -> np.ndarray:
        return np.interp(
            np.asarray(energies, dtype=get_dtype()), self.energies, self.values
        )


@dataclass(frozen=True)
class SDDEfficiency:
    """
    Detection efficiency of a layered solid-state detector.

    Photons are attenuated by the window, contact and dead layer, and then
    absorbed in the active layer. Attenuating layers use the total cross-section
    while the active layer uses the photo-absorption cross-section, since only a
    photo-absorption event contributes to the full-energy peak.

    All thicknesses are in **micrometres**.

    Parameters
    ----------
    window : (str, float), optional
        Entrance window as a ``(chemical formula, thickness)`` pair, e.g.
        ``("Be", 8.0)``. Default is None, i.e. a windowless detector.
    contact : (str, float), optional
        Front contact. Default is ``("Al", 0.03)``.
    dead_layer : (str, float), optional
        Inactive layer at the front of the sensor. Default is ``("Si", 0.05)``.
    active_layer : (str, float), optional
        Sensitive volume. Default is ``("Si", 450.0)``.
    densities : dict, optional
        Mass densities [g/cm^3] keyed by chemical formula, overriding the
        tabulated values. Required for compounds xraydb does not know.

    Examples
    --------
    >>> efficiency = SDDEfficiency(window=("Be", 8.0))  # doctest: +SKIP
    >>> efficiency(8040.0)  # Cu Ka  # doctest: +SKIP
    0.9958...
    """

    window: Optional[tuple[str, float]] = None
    contact: Optional[tuple[str, float]] = ("Al", 0.03)
    dead_layer: Optional[tuple[str, float]] = ("Si", 0.05)
    active_layer: Optional[tuple[str, float]] = ("Si", 450.0)
    densities: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("window", "contact", "dead_layer", "active_layer"):
            object.__setattr__(
                self, name, _validate_layer(getattr(self, name), name)
            )

    @property
    def _attenuating_layers(self) -> tuple[tuple[str, float], ...]:
        layers = (self.window, self.contact, self.dead_layer)
        return tuple(layer for layer in layers if layer is not None)

    def _mu(self, formula: str, energies: np.ndarray, kind: str) -> np.ndarray:
        try:
            import xraydb
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e

        # material_mu returns the linear attenuation coefficient in 1/cm. Note
        # that mu_elam returns the *mass* attenuation coefficient in cm^2/g and
        # must not be used here without multiplying by the density.
        return np.asarray(
            xraydb.material_mu(
                formula,
                np.asarray(energies, dtype=get_dtype()),
                density=self.densities.get(formula),
                kind=kind,
            ),
            dtype=get_dtype(),
        )

    def __call__(self, energies) -> np.ndarray:
        energies = np.atleast_1d(np.asarray(energies, dtype=get_dtype()))

        # Micrometres to centimetres, the unit of the tabulated coefficients.
        micron = 1e-4

        transmission = np.ones_like(energies)
        for formula, thickness in self._attenuating_layers:
            mu = self._mu(formula, energies, kind="total")
            transmission = transmission * np.exp(-mu * thickness * micron)

        if self.active_layer is None:
            absorbed = np.ones_like(energies)
        else:
            formula, thickness = self.active_layer
            mu = self._mu(formula, energies, kind="photo")
            absorbed = 1.0 - np.exp(-mu * thickness * micron)

        return transmission * absorbed


@dataclass(frozen=True)
class SpecimenAbsorption:
    """
    Attenuation of the emitted X-rays on their way out of the specimen.

    A photon generated at depth ``z`` below the entrance surface travels
    ``z / sin(takeoff_angle)`` through the specimen to reach a detector mounted
    on the entrance side, which is the usual STEM-EDX geometry. The
    transmission is ``exp(-mu(E) z / sin(alpha))``, with ``mu`` the linear
    attenuation coefficient of the specimen at the line energy.

    This is a single-scattering correction: photons removed from the beam are
    lost, and secondary fluorescence excited by them is not modelled.

    Parameters
    ----------
    formula : str
        Chemical formula of the specimen, e.g. ``"SrTiO3"``.
    density : float, optional
        Mass density [g/cm^3]. Taken from the tabulation when the formula is
        recognised, which works for elements but not for most compounds.
    takeoff_angle : float, optional
        Elevation of the detector above the specimen plane [degrees]. Default
        is 18.0, typical of a STEM-EDX geometry.

    Examples
    --------
    >>> absorption = SpecimenAbsorption("Si", takeoff_angle=22.0)  # doctest: +SKIP
    >>> detector = XrayDetector(0.7, absorption=absorption)  # doctest: +SKIP
    """

    formula: str
    density: Optional[float] = None
    takeoff_angle: float = 18.0

    def __post_init__(self):
        if not 0.0 < self.takeoff_angle <= 90.0:
            raise ValueError(
                f"'takeoff_angle' must be in (0, 90] degrees, got "
                f"{self.takeoff_angle}"
            )
        if self.density is not None and self.density <= 0.0:
            raise ValueError(f"'density' must be positive, got {self.density}")

    def transmission(self, energies, depth: float) -> np.ndarray:
        """
        Fraction of photons escaping from a given depth.

        Parameters
        ----------
        energies : float or array of float
            Photon energies [eV].
        depth : float
            Depth below the entrance surface at which the photons are
            generated [Angstrom].

        Returns
        -------
        transmission : array of float
        """
        energies = np.atleast_1d(np.asarray(energies, dtype=get_dtype()))

        if depth <= 0.0:
            return np.ones_like(energies)

        try:
            import xraydb  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e

        mu = np.asarray(
            xraydb.material_mu(self.formula, energies, density=self.density),
            dtype=get_dtype(),
        )

        # mu is per cm; the path is the depth divided by the sine of the
        # take-off angle, converted from Angstrom.
        path = depth * 1e-8 / np.sin(np.deg2rad(self.takeoff_angle))
        return np.exp(-mu * path)


class XrayDetector(IonizationDetector):
    """
    Detector for characteristic X-ray emission.

    Converts an ionisation probability into detected photons per incident
    electron, assuming isotropic emission and no self-absorption in the
    specimen.

    Parameters
    ----------
    solid_angle : float
        Solid angle subtended by the detector [sr]. Must be greater than zero
        and at most ``4 * pi``.
    efficiency : float or callable, optional
        Detection efficiency. Either a constant between 0 and 1, or a callable
        mapping photon energies [eV] to efficiencies, such as
        :class:`SDDEfficiency` or :class:`TabulatedEfficiency`. Default is 1.0.
    lines : str or sequence of str, optional
        Emission lines to detect, given as Siegbahn names (``"Ka1"``) or
        families (``"Ka"``). Default is ``"all"``, which detects every
        tabulated line of the ionised subshell.
    coster_kronig : bool, optional
        Apply Coster-Kronig redistribution of the vacancy within the ionised
        subshell. Default is True.
    absorption : SpecimenAbsorption, optional
        Attenuation of the emitted photons inside the specimen. Default is
        None, i.e. a transparent specimen. Needs the emission depth, which the
        simulation drivers supply; :meth:`to_counts` cannot apply it, because a
        finished ionisation measurement has already been summed over depth.
    to_cpu : bool, optional
        If True, copy the measurement data to CPU memory. Default is True.

    Notes
    -----
    This is the object that models the experiment, and it can be passed
    straight to a simulation::

        maps = probe.transition_potential_scan(
            potential, transitions, scan=scan,
            detectors=[abtem.AnnularDetector(0.0, 30.0),          # EELS
                       abtem.XrayDetector(solid_angle=0.7)],      # EDX
        )

    Doing so needs the waves to carry the identity of the ionised edge, which
    the transition potential stamps on them. Applied to waves that carry no such
    metadata it raises; use :meth:`apply` on an already-detected ionisation
    measurement instead.

    Examples
    --------
    >>> detector = XrayDetector(solid_angle=0.7)  # doctest: +SKIP
    >>> detector.total_yield("Cu", 1, 0)  # photons per K ionisation  # doctest: +SKIP
    0.02456...

    See Also
    --------
    XrayDetector.from_sdd : construct with a layered detector efficiency model.
    """

    def __init__(
        self,
        solid_angle: float,
        efficiency: float | Callable[[np.ndarray], np.ndarray] = 1.0,
        lines: str | Sequence[str] = "all",
        coster_kronig: bool = True,
        absorption: Optional[SpecimenAbsorption] = None,
        to_cpu: bool = True,
    ):
        solid_angle = float(solid_angle)
        if not 0.0 < solid_angle <= 4 * np.pi:
            raise ValueError(
                f"'solid_angle' must be in (0, 4*pi] steradians, got {solid_angle}"
            )

        if not callable(efficiency):
            efficiency = float(efficiency)
            if not 0.0 <= efficiency <= 1.0:
                raise ValueError(
                    f"a constant 'efficiency' must be between 0 and 1, got "
                    f"{efficiency}"
                )

        if isinstance(lines, str):
            lines = (lines,)
        lines = tuple(lines)

        if not lines:
            raise ValueError("'lines' must select at least one line")

        self._solid_angle = solid_angle
        self._efficiency = efficiency
        self._lines = lines
        self._coster_kronig = coster_kronig
        self._absorption = absorption

        super().__init__(mu=None, to_cpu=to_cpu)

    @classmethod
    def from_sdd(
        cls,
        solid_angle: float,
        window: Optional[tuple[str, float]] = None,
        contact: Optional[tuple[str, float]] = ("Al", 0.03),
        dead_layer: Optional[tuple[str, float]] = ("Si", 0.05),
        active_layer: Optional[tuple[str, float]] = ("Si", 450.0),
        densities: Optional[dict[str, float]] = None,
        **kwargs,
    ) -> XrayDetector:
        """
        Create a detector with a layered efficiency model.

        Thicknesses are in **micrometres**. See :class:`SDDEfficiency`.

        Parameters
        ----------
        solid_angle : float
            Solid angle subtended by the detector [sr].
        window, contact, dead_layer, active_layer : (str, float), optional
            Absorbing layers as ``(chemical formula, thickness)`` pairs.
        densities : dict, optional
            Mass densities [g/cm^3] overriding the tabulated values.
        kwargs :
            Passed to :class:`XrayDetector`.

        Returns
        -------
        detector : XrayDetector

        Examples
        --------
        >>> XrayDetector.from_sdd(0.7, window=("Be", 8.0))  # doctest: +SKIP
        """
        efficiency = SDDEfficiency(
            window=window,
            contact=contact,
            dead_layer=dead_layer,
            active_layer=active_layer,
            densities={} if densities is None else dict(densities),
        )
        return cls(solid_angle=solid_angle, efficiency=efficiency, **kwargs)

    @property
    def solid_angle(self) -> float:
        """Solid angle subtended by the detector [sr]."""
        return self._solid_angle

    @property
    def efficiency(self) -> float | Callable[[np.ndarray], np.ndarray]:
        """Detection efficiency, as a constant or a callable of photon energy."""
        return self._efficiency

    @property
    def lines(self) -> tuple[str, ...]:
        """Selected emission lines or line families."""
        return self._lines

    @property
    def coster_kronig(self) -> bool:
        """Coster-Kronig redistribution is applied within the ionised subshell."""
        return self._coster_kronig

    @property
    def absorption(self) -> Optional[SpecimenAbsorption]:
        """Self-absorption model, or None if the specimen is treated as transparent."""
        return self._absorption

    @property
    def collection_fraction(self) -> float:
        """Fraction of isotropically emitted photons reaching the detector."""
        return self._solid_angle / (4 * np.pi)

    def efficiency_at(self, energies) -> np.ndarray:
        """
        Detection efficiency at the given photon energies.

        Parameters
        ----------
        energies : float or array of float
            Photon energies [eV].

        Returns
        -------
        efficiency : array of float
        """
        energies = np.atleast_1d(np.asarray(energies, dtype=get_dtype()))

        if callable(self._efficiency):
            values = np.atleast_1d(
                np.asarray(self._efficiency(energies), dtype=get_dtype())
            )
            if values.shape != energies.shape:
                raise ValueError(
                    f"the efficiency callable returned shape {values.shape} for "
                    f"{energies.shape} energies"
                )
        else:
            values = np.full(energies.shape, self._efficiency, dtype=get_dtype())

        return values

    def _selects(self, line: EmissionLine) -> bool:
        if "all" in self._lines:
            return True
        return line.name in self._lines or line.family in self._lines

    def detected_lines(
        self, element: int | str, n: int, l: int
    ) -> dict[str, EmissionLine]:
        """
        Selected emission lines, with intensities scaled to detected photons.

        Parameters
        ----------
        element : int or str
            Atomic number or chemical symbol of the emitting element.
        n, l : int
            Quantum numbers of the ionised subshell, matching the
            :class:`~abtem.inelastic.core_loss.SubshellTransitions` used to
            compute the ionisation probability.

        Returns
        -------
        lines : dict
            Mapping of Siegbahn name to :class:`.EmissionLine` whose
            ``intensity`` is the number of photons detected per ionisation of
            the subshell, sorted by decreasing intensity.
        """
        lines = emission_lines(element, n, l, coster_kronig=self._coster_kronig)

        selected = {
            name: line for name, line in lines.items() if self._selects(line)
        }

        if not selected:
            available = sorted({line.family for line in lines.values()})
            raise ValueError(
                f"none of the requested lines {self._lines} are emitted by "
                f"{element} (n={n}, l={l}); available families are {available}"
            )

        energies = np.array(
            [line.energy for line in selected.values()], dtype=get_dtype()
        )
        efficiencies = self.efficiency_at(energies)

        fraction = self.collection_fraction

        detected = {
            name: EmissionLine(
                name=line.name,
                energy=line.energy,
                intensity=line.intensity * fraction * float(efficiency),
                initial_level=line.initial_level,
                final_level=line.final_level,
            )
            for (name, line), efficiency in zip(selected.items(), efficiencies)
        }

        return dict(sorted(detected.items(), key=lambda kv: -kv[1].intensity))

    def detected_families(
        self, element: int | str, n: int, l: int
    ) -> dict[str, list[EmissionLine]]:
        """
        Selected emission lines grouped into families, e.g. ``Ka1`` into ``Ka``.

        Parameters
        ----------
        element : int or str
            Atomic number or chemical symbol of the emitting element.
        n, l : int
            Quantum numbers of the ionised subshell.

        Returns
        -------
        families : dict
            Mapping of family name to its detected lines.
        """
        return line_families(self.detected_lines(element, n, l))

    def total_yield(self, element: int | str, n: int, l: int) -> float:
        """
        Detected photons per ionisation of the subshell, summed over lines.

        Parameters
        ----------
        element : int or str
            Atomic number or chemical symbol of the emitting element.
        n, l : int
            Quantum numbers of the ionised subshell.

        Returns
        -------
        yield : float
            Photons detected per ionisation.
        """
        lines = self.detected_lines(element, n, l)
        return float(sum(line.intensity for line in lines.values()))

    def _photon_yield(self, metadata: dict) -> float:
        """Detected photons per ionisation of the edge named in the metadata."""
        try:
            element = metadata["Z"]
            n = metadata["n"]
            l = metadata["l"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(
                "XrayDetector needs to know which edge was ionised, but these "
                "waves carry no transition-potential metadata. Detect the waves "
                "of a transition potential simulation, or call "
                "XrayDetector.to_counts on an ionisation measurement instead."
            ) from e

        if self._absorption is None:
            return self.total_yield(element, n, l)

        depth = metadata.get("depth")
        if depth is None:
            raise RuntimeError(
                "modelling self-absorption needs the depth at which the photon "
                "was generated, but these waves carry none. The simulation "
                "drivers supply it; a finished ionisation measurement has "
                "already been summed over depth, so absorption cannot be "
                "applied to it after the fact."
            )

        lines = self.detected_lines(element, n, l)
        energies = np.array(
            [line.energy for line in lines.values()], dtype=get_dtype()
        )
        transmission = self._absorption.transmission(energies, float(depth))

        return float(
            sum(
                line.intensity * float(t)
                for line, t in zip(lines.values(), transmission)
            )
        )

    def to_counts_from_subshells(
        self,
        ionizations: dict[tuple[int, int], BaseMeasurements],
        element: int | str,
        per_family: bool = False,
    ) -> BaseMeasurements:
        """
        Combine the ionisation of several subshells of one element.

        An L-family measurement needs the 2s and 2p edges together: a vacancy
        created in L1 largely migrates to L2 and L3 by Coster-Kronig
        transitions and radiates from there, so the edges cannot simply be
        added after converting each on its own -- each one's yield already
        accounts for its own cascade, but they must be weighted by their own
        ionisation probabilities.

        Parameters
        ----------
        ionizations : dict
            Mapping of ``(n, l)`` to the ionisation probability measured for
            that subshell. All entries must share the same shape.
        element : int or str
            Atomic number or chemical symbol of the emitting element.
        per_family : bool, optional
            Return one measurement per line family instead of the total.
            Default is False.

        Returns
        -------
        counts : BaseMeasurements
            Detected photons per incident electron.

        Examples
        --------
        >>> l1 = probe.ionization_scan(potential, transitions_2s)  # doctest: +SKIP
        >>> l23 = probe.ionization_scan(potential, transitions_2p)  # doctest: +SKIP
        >>> counts = detector.to_counts_from_subshells(  # doctest: +SKIP
        ...     {(2, 0): l1, (2, 1): l23}, "Ag"
        ... )
        """
        if not ionizations:
            raise ValueError("'ionizations' must contain at least one subshell")

        shells = {n for (n, _) in ionizations}
        if len(shells) > 1:
            raise ValueError(
                "Coster-Kronig transitions only connect levels of one shell, "
                f"but subshells from shells {sorted(shells)} were given; "
                "combine each shell separately"
            )

        if per_family:
            raise NotImplementedError(
                "per_family is not yet supported when combining subshells"
            )

        total = None
        for (n, l), measurement in ionizations.items():
            scaled = cast(
                "BaseMeasurements",
                cast(Any, measurement) * self.total_yield(element, n, l),
            )
            total = scaled if total is None else cast(
                "BaseMeasurements", cast(Any, total) + cast(Any, scaled)
            )

        assert total is not None
        total.metadata["label"] = "detected X-rays"
        total.metadata["units"] = "photons / electron"
        return total

    def to_counts(
        self,
        ionization: BaseMeasurements,
        element: int | str,
        n: int,
        l: int,
        per_family: bool = False,
    ) -> BaseMeasurements:
        """
        Convert an already-detected ionisation measurement into X-ray counts.

        The post-hoc counterpart of using this object as a detector: use it when
        the ionisation probability has been computed separately, for instance by
        :meth:`~abtem.Probe.ionization_scan` or
        :meth:`~abtem.SMatrix.ionization_scan`, which know the edge themselves
        and so return a bare probability.

        Parameters
        ----------
        ionization : BaseMeasurements
            Ionisation probability per incident electron, as produced by a
            transition-potential simulation integrated over all scattering
            angles.
        element : int or str
            Atomic number or chemical symbol of the emitting element.
        n, l : int
            Quantum numbers of the ionised subshell.
        per_family : bool, optional
            Return one measurement per line family, stacked along a new leading
            ordinal axis, instead of the sum over all selected lines. Default is
            False.

        Returns
        -------
        counts : BaseMeasurements
            Detected photons per incident electron.

        Notes
        -----
        The result is only on an absolute scale if ``ionization`` is itself
        absolute, which requires integrating over all scattering angles and over
        the whole ionisation edge. See the module docstring.
        """
        # ArrayObject.__mul__ is annotated as taking another array object, but
        # scaling by a scalar is supported; cast to keep the type checker quiet.
        def _scale(measurement: BaseMeasurements, factor: float) -> BaseMeasurements:
            scaled = cast("BaseMeasurements", cast(Any, measurement) * factor)
            scaled.metadata["label"] = "detected X-rays"
            scaled.metadata["units"] = "photons / electron"
            return scaled

        if not per_family:
            return _scale(ionization, self.total_yield(element, n, l))

        from abtem.array import stack

        families = self.detected_families(element, n, l)

        measurements = []
        for members in families.values():
            intensity = float(sum(line.intensity for line in members))
            measurements.append(_scale(ionization, intensity))

        if len(measurements) == 1:
            return measurements[0]

        return stack(
            measurements,
            OrdinalAxis(
                label="X-ray line",
                values=tuple(families.keys()),
                units="",
            ),
        )
