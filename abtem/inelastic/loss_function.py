"""Momentum-resolved energy-loss functions for the plasmon models.

:class:`LossFunction` holds :math:`\\mathrm{Im}[-1/\\varepsilon(q, E)]` on a grid of
momentum transfers and energy losses, for example from a first-principles linear
response calculation read with :meth:`LossFunction.from_gpaw`. From it follow the
inputs of the plasmon models (:meth:`LossFunction.plasmon_parameters`) and the energy
distribution of a single loss (:meth:`LossFunction.angle_integrated`), consistently
with each other.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Optional, Sequence

import numpy as np

from abtem.core.energy import energy2wavelength, relativistic_mass_correction

_BOHR = 0.529177210903  # Bohr radius [Å]
_ELECTRON_REST_ENERGY = 510998.95  # m0 c^2 [eV]
_HBAR2_2M = 3.80998212  # hbar^2 / 2 m0 [eV Å^2]


def _characteristic_angle(energy_losses, energy):
    """Characteristic angle :math:`\\theta_E = E / (\\gamma m_0 v^2)` [rad]."""
    gamma = relativistic_mass_correction(energy)
    return np.asarray(energy_losses) / (2 * energy) * (2 * gamma / (gamma + 1))


def _m0v2(energy):
    """:math:`m_0 v^2` [eV] of an electron of kinetic energy ``energy`` [eV]."""
    gamma = relativistic_mass_correction(energy)
    return _ELECTRON_REST_ENERGY * (1 - 1 / gamma**2)


class LossFunction:
    """Energy-loss function :math:`\\mathrm{Im}[-1/\\varepsilon(q, E)]`.

    In the first Born approximation, the probability per unit path length that a fast
    electron loses energy :math:`E` into solid angle :math:`\\Omega` is (Egerton,
    Electron Energy-Loss Spectroscopy in the Electron Microscope, 3rd ed., Eq. 3.44)

    .. math::
        \\frac{d^2 P}{dE\\, d\\Omega}
            = \\frac{\\mathrm{Im}[-1/\\varepsilon(q, E)]}
                   {\\pi^2 a_0 m_0 v^2 (\\theta^2 + \\theta_E^2)},
        \\qquad q = \\frac{\\sqrt{\\theta^2 + \\theta_E^2}}{\\lambda},

    with :math:`\\theta_E` the characteristic angle and :math:`\\lambda` the electron
    wavelength. Everything the plasmon models need follows from this one function:
    the mean free path, the effective critical angle, the plasmon energy and the
    energy distribution of a single loss.

    The loss function is taken as isotropic: it depends on the magnitude of
    :math:`q` only. Below the smallest tabulated momentum transfer it is the loss
    function at that momentum transfer (the optical limit, if it is tabulated at
    :math:`q = 0`); above the largest it is zero. With a single momentum transfer
    the loss function does not depend on :math:`q`, as for a measured optical
    dielectric function, and angular integrals then need an explicit upper angle.

    Parameters
    ----------
    energy_losses : np.ndarray
        Increasing energy losses [eV], shape ``(n_E,)``.
    momentum_transfers : np.ndarray
        Increasing, distinct momentum transfers :math:`|q|` [1/Å], without a factor
        :math:`2\\pi` (the abTEM convention for spatial frequencies), shape
        ``(n_q,)``.
    values : np.ndarray
        :math:`\\mathrm{Im}[-1/\\varepsilon(q, E)]`, shape ``(n_q, n_E)``.
    metadata : dict, optional
        Provenance of the data, for example the settings of the response
        calculation. Saved with :meth:`write`.
    """

    def __init__(
        self,
        energy_losses: Sequence[float],
        momentum_transfers: Sequence[float],
        values: np.ndarray,
        metadata: Optional[dict] = None,
    ):
        energy_losses = np.asarray(energy_losses, dtype=float)
        momentum_transfers = np.atleast_1d(np.asarray(momentum_transfers, dtype=float))
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[None]

        if energy_losses.ndim != 1 or len(energy_losses) < 2:
            raise ValueError("energy_losses must be 1D with at least two values")
        if np.any(np.diff(energy_losses) <= 0):
            raise ValueError("energy_losses must be increasing")
        if momentum_transfers.ndim != 1:
            raise ValueError("momentum_transfers must be 1D")
        if np.any(momentum_transfers < 0) or np.any(np.diff(momentum_transfers) <= 0):
            raise ValueError(
                "momentum_transfers must be non-negative, increasing and distinct"
            )
        if values.shape != (len(momentum_transfers), len(energy_losses)):
            raise ValueError(
                f"values must have shape (n_q, n_E) = "
                f"{(len(momentum_transfers), len(energy_losses))}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("the loss function must be finite")

        # broadened response functions can dip slightly below zero; anything more is
        # an error in the input
        scale = np.abs(values).max()
        if np.any(values < -1e-3 * scale):
            raise ValueError("the loss function must be non-negative")
        self._energy_losses = energy_losses
        self._momentum_transfers = momentum_transfers
        self._values = np.clip(values, 0.0, None)
        self._metadata = {} if metadata is None else dict(metadata)

    @property
    def energy_losses(self) -> np.ndarray:
        """Energy losses [eV]."""
        return self._energy_losses

    @property
    def momentum_transfers(self) -> np.ndarray:
        """Momentum transfers :math:`|q|` [1/Å]."""
        return self._momentum_transfers

    @property
    def values(self) -> np.ndarray:
        """:math:`\\mathrm{Im}[-1/\\varepsilon(q, E)]`, shape ``(n_q, n_E)``."""
        return self._values

    @property
    def metadata(self) -> dict:
        """Provenance of the data."""
        return self._metadata

    @property
    def is_momentum_resolved(self) -> bool:
        """Whether the loss function is tabulated at more than one momentum
        transfer."""
        return len(self._momentum_transfers) > 1

    @property
    def optical_limit(self) -> np.ndarray:
        """The loss function at the smallest tabulated momentum transfer."""
        return self._values[0]

    @property
    def peak_energy(self) -> float:
        """Energy loss [eV] of the maximum of :attr:`optical_limit`."""
        return float(self._energy_losses[np.argmax(self.optical_limit)])

    @property
    def excitation_energy(self) -> float:
        """Effective plasmon energy :math:`E_p` [eV] of the Lorentzian models.

        The small-angle cross section is a sum of Lorentzians
        :math:`1/(\\theta^2 + \\theta_E(E)^2)` weighted by the optical-limit loss
        function; integrated over angle, it equals a single Lorentzian at the
        energy :math:`\\ln E_p = \\langle \\ln E \\rangle` averaged with that weight.
        For a single narrow peak this is the peak energy.
        """
        e, weight = self._positive_energies(self.optical_limit)
        total = np.trapezoid(weight, e)
        if total <= 0:
            raise ValueError("the optical-limit loss function vanishes")
        return float(np.exp(np.trapezoid(weight * np.log(e), e) / total))

    def _positive_energies(self, values):
        keep = self._energy_losses > 0
        return self._energy_losses[keep], values[..., keep]

    def __call__(self, energy_losses, momentum_transfers=None) -> np.ndarray:
        """Interpolate the loss function, linearly in :math:`q` and :math:`E`.

        Parameters
        ----------
        energy_losses : array_like
            Energy losses [eV].
        momentum_transfers : array_like, optional
            Momentum transfers [1/Å], broadcast against ``energy_losses``. Default
            is the smallest tabulated momentum transfer.

        Returns
        -------
        values : np.ndarray
            The loss function; zero outside the tabulated energy range and above the
            largest tabulated momentum transfer.
        """
        e = np.asarray(energy_losses, dtype=float)
        if momentum_transfers is None:
            q = np.full_like(e, self._momentum_transfers[0])
        else:
            q = np.asarray(momentum_transfers, dtype=float)
        e, q = np.broadcast_arrays(e, q)

        qs, es, values = self._momentum_transfers, self._energy_losses, self._values
        inside = (e >= es[0]) & (e <= es[-1]) & (q <= qs[-1])
        if len(qs) == 1:
            out = np.interp(e, es, values[0])
        else:
            qc = np.clip(q, qs[0], qs[-1])
            i = np.clip(np.searchsorted(qs, qc, side="right") - 1, 0, len(qs) - 2)
            t = (qc - qs[i]) / (qs[i + 1] - qs[i])
            j = np.clip(np.searchsorted(es, e, side="right") - 1, 0, len(es) - 2)
            s = np.clip((e - es[j]) / (es[j + 1] - es[j]), 0.0, 1.0)
            out = (
                (1 - t) * (1 - s) * values[i, j]
                + (1 - t) * s * values[i, j + 1]
                + t * (1 - s) * values[i + 1, j]
                + t * s * values[i + 1, j + 1]
            )
        return np.where(inside, out, 0.0)

    def _max_angle(self, energy, max_angle):
        """Upper scattering angle [rad] of the angular integrals."""
        if max_angle is not None:
            return max_angle * 1e-3
        if not self.is_momentum_resolved:
            raise ValueError(
                "the loss function does not depend on the momentum transfer, so the "
                "angular integral needs an upper angle; give max_angle"
            )
        return np.inf

    def angle_integrated(
        self,
        energy_losses,
        energy: float,
        max_angle: float = None,
        num_nodes: int = 256,
    ) -> np.ndarray:
        """Single-scattering energy distribution :math:`dP/dE` per unit path length.

        The cross section integrated over the scattering angles up to ``max_angle``,

        .. math::
            \\frac{dP}{dE} = \\frac{2}{\\pi a_0 m_0 v^2}
                \\int_{q_E}^{q_\\mathrm{max}} \\mathrm{Im}
                \\left[\\frac{-1}{\\varepsilon(q, E)}\\right] \\frac{dq}{q},

        with :math:`q_E = \\theta_E / \\lambda`; for a loss function that does not
        depend on :math:`q` this is
        :math:`\\mathrm{Im}[-1/\\varepsilon(E)]\\, \\ln(1 + \\theta^2_\\mathrm{max} /
        \\theta_E^2) / (\\pi a_0 m_0 v^2)`.

        Parameters
        ----------
        energy_losses : array_like
            Energy losses [eV].
        energy : float
            Electron energy [eV].
        max_angle : float, optional
            Upper scattering angle [mrad]. Default is the angle at which the
            momentum transfer reaches the largest tabulated value; required when the
            loss function does not depend on the momentum transfer.
        num_nodes : int, optional
            Nodes of the integral over :math:`\\ln q` per energy loss.

        Returns
        -------
        distribution : np.ndarray
            :math:`dP/dE` [1/(Å eV)].
        """
        e = np.asarray(energy_losses, dtype=float)
        out = np.zeros_like(e)
        positive = e > 0
        e = e[positive]

        wavelength = energy2wavelength(energy)
        theta_e = _characteristic_angle(e, energy)
        max_angle = self._max_angle(energy, max_angle)
        prefactor = 2 / (np.pi * _BOHR * _m0v2(energy))

        if not self.is_momentum_resolved:
            log_term = np.log1p(max_angle**2 / theta_e**2)
            out[positive] = prefactor / 2 * self(e) * log_term
            return out

        q_low = theta_e / wavelength
        q_high = np.minimum(
            np.sqrt(max_angle**2 + theta_e**2) / wavelength,
            self._momentum_transfers[-1],
        )
        valid = q_high > q_low
        # nodes uniform in ln q, refined between the tabulated momentum transfers
        u = np.linspace(0.0, 1.0, num_nodes)
        log_low = np.log(q_low[valid])
        log_high = np.log(q_high[valid])
        log_q = log_low[:, None] + (log_high - log_low)[:, None] * u[None]
        integrand = self(e[valid, None], np.exp(log_q))
        integral = np.zeros_like(e)
        integral[valid] = np.trapezoid(integrand, log_q, axis=1)
        out[positive] = prefactor * integral
        return out

    def mean_free_path(self, energy: float, max_angle: float = None) -> float:
        """Inelastic mean free path [Å] for scattering up to ``max_angle``.

        The inverse of the integral of :meth:`angle_integrated` over the tabulated
        energy losses.

        Parameters
        ----------
        energy : float
            Electron energy [eV].
        max_angle : float, optional
            Upper scattering angle [mrad]; see :meth:`angle_integrated`.
        """
        e, _ = self._positive_energies(self._values)
        rate = np.trapezoid(self.angle_integrated(e, energy, max_angle), e)
        if rate <= 0:
            raise ValueError("the loss function vanishes on the tabulated grid")
        if max_angle is None and self.is_momentum_resolved:
            self._warn_if_not_decayed()
        return float(1 / rate)

    def _warn_if_not_decayed(self):
        e, values = self._positive_energies(self._values)
        strength = np.trapezoid(values, e, axis=1)
        if strength[0] > 0 and strength[-1] > 0.05 * strength[0]:
            warnings.warn(
                "the loss function at the largest momentum transfer still has "
                f"{strength[-1] / strength[0]:.0%} of the optical-limit strength; the "
                "mean free path misses the scattering beyond it and is overestimated",
                stacklevel=3,
            )

    def critical_angle(
        self,
        energy: float,
        mean_free_path: float = None,
        excitation_energy: float = None,
    ) -> float:
        """Effective critical angle :math:`\\theta_c` [mrad] of the Lorentzian models.

        The plasmon models scatter with the Lorentzian distribution
        :math:`\\theta / (\\theta^2 + \\theta_E^2)` cut at :math:`\\theta_c`. The
        effective critical angle is the cut at which that distribution, with the
        small-angle amplitude of the optical-limit loss function, carries the total
        scattering rate :math:`1/\\lambda`:

        .. math::
            \\frac{1}{\\lambda} = \\frac{1}{\\pi a_0 m_0 v^2}
                \\int \\mathrm{Im}\\left[\\frac{-1}{\\varepsilon(0, E)}\\right] dE\\;
                \\ln\\left(1 + \\frac{\\theta_c^2}{\\theta_E^2}\\right),

        with :math:`\\theta_E` at :attr:`excitation_energy`. The small-angle
        distribution and the total rate of the model then both agree with the loss
        function.

        Parameters
        ----------
        energy : float
            Electron energy [eV].
        mean_free_path : float, optional
            Mean free path [Å] to match. Default is :meth:`mean_free_path`.
        excitation_energy : float, optional
            Plasmon energy [eV] that sets :math:`\\theta_E`. Default is
            :attr:`excitation_energy`.
        """
        if mean_free_path is None:
            mean_free_path = self.mean_free_path(energy)
        if excitation_energy is None:
            excitation_energy = self.excitation_energy
        e, weight = self._positive_energies(self.optical_limit)
        strength = np.trapezoid(weight, e)
        small_angle_rate = strength / (np.pi * _BOHR * _m0v2(energy))
        theta_e = _characteristic_angle(excitation_energy, energy)
        exponent = 1 / (mean_free_path * small_angle_rate)
        if exponent > 700:
            raise ValueError(
                "the mean free path is too long for the optical-limit strength of the "
                "loss function to give a finite critical angle"
            )
        return float(theta_e * np.sqrt(np.expm1(exponent)) * 1e3)

    def plasmon_parameters(
        self,
        energy: float,
        excitation_energy: float = None,
        critical_angle: float = None,
        mean_free_path: float = None,
        max_angle: float = None,
    ) -> tuple[float, float, float]:
        """Inputs of the plasmon models from the loss function.

        Any parameter given is used as it is; the others follow from the loss
        function and from the given ones. The critical angle is fitted to the mean
        free path and the plasmon energy in use (see :meth:`critical_angle`), so a
        measured mean free path can be combined with the calculated loss function.

        Parameters
        ----------
        energy : float
            Electron energy [eV].
        excitation_energy : float, optional
            Plasmon energy [eV]. Default is :attr:`excitation_energy`.
        critical_angle : float, optional
            Critical angle [mrad]. Default is :meth:`critical_angle`.
        mean_free_path : float, optional
            Mean free path [Å]. Default is :meth:`mean_free_path`.
        max_angle : float, optional
            Upper scattering angle [mrad] of the mean free path; see
            :meth:`angle_integrated`.

        Returns
        -------
        excitation_energy : float
            Plasmon energy :math:`E_p` [eV].
        critical_angle : float
            Critical angle :math:`\\theta_c` [mrad].
        mean_free_path : float
            Mean free path :math:`\\lambda` [Å].

        The order is that of
        :func:`~abtem.inelastic.plasmons.estimate_plasmon_parameters`.
        """
        if excitation_energy is None:
            excitation_energy = self.excitation_energy
        if mean_free_path is None:
            mean_free_path = self.mean_free_path(energy, max_angle)
        if critical_angle is None:
            critical_angle = self.critical_angle(
                energy, mean_free_path, excitation_energy
            )
        return float(excitation_energy), float(critical_angle), float(mean_free_path)

    def write(self, path: str):
        """Save to a NumPy ``.npz`` file.

        Parameters
        ----------
        path : str
            File name.
        """
        np.savez(
            path,
            energy_losses=self._energy_losses,
            momentum_transfers=self._momentum_transfers,
            values=self._values,
            metadata=json.dumps(self._metadata),
        )

    @classmethod
    def read(cls, path: str) -> "LossFunction":
        """Load a loss function saved with :meth:`write`.

        Parameters
        ----------
        path : str
            File name.
        """
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["energy_losses"],
                data["momentum_transfers"],
                data["values"],
                metadata=json.loads(str(data["metadata"])),
            )

    @classmethod
    def from_dielectric_function(
        cls,
        energy_losses: Sequence[float],
        dielectric_function: np.ndarray,
        momentum_transfers: Sequence[float] = (0.0,),
        metadata: Optional[dict] = None,
    ) -> "LossFunction":
        """Loss function :math:`\\mathrm{Im}[-1/\\varepsilon]` of a complex
        dielectric function.

        Parameters
        ----------
        energy_losses : array_like
            Energy losses [eV].
        dielectric_function : np.ndarray
            Complex :math:`\\varepsilon(q, E)`, shape ``(n_q, n_E)`` or ``(n_E,)``.
        momentum_transfers : array_like, optional
            Momentum transfers [1/Å]. Default is the optical limit only.
        metadata : dict, optional
            Provenance of the data.
        """
        eps = np.asarray(dielectric_function, dtype=complex)
        return cls(energy_losses, momentum_transfers, (-1 / eps).imag, metadata)

    @classmethod
    def from_gpaw(
        cls,
        files: Sequence[str],
        log: str = None,
        cell=None,
        q_points: Sequence[Sequence[float]] = None,
        momentum_transfers: Sequence[float] = None,
        ecut: float = None,
        local_field_effects: bool = True,
    ) -> "LossFunction":
        """Read the loss function from the output files of a GPAW response calculation.

        GPAW's ``DielectricFunction`` writes one file per momentum transfer:
        ``get_eels_spectrum(q_c, filename=...)`` the columns :math:`\\omega`,
        :math:`-\\mathrm{Im}\\,1/\\varepsilon` without and with local-field effects,
        and ``get_dielectric_function(q_c, filename=...)`` :math:`\\omega` and the
        real and imaginary parts of :math:`\\varepsilon` without and with local-field
        effects. Either kind can be read.

        The files do not record the momentum transfer. GPAW's log (the ``txt`` of
        ``DielectricFunction``) does: ``q_c`` and the plane-wave cutoff of every
        calculation, in the order they ran. Give the ``log`` and the ``cell`` of the
        ground state (for example the ``.gpw`` file), with the files in the order
        they were calculated, one calculation each. Otherwise give ``q_points`` and
        the ``cell``, or ``momentum_transfers``.

        Some GPAW output is not the loss function at the momentum transfer
        :math:`q`, and is rejected:

        - **Momentum transfers beyond the response cutoff,**
          :math:`\\hbar^2 (2\\pi q)^2 / 2m > E_\\mathrm{cut}`. GPAW's plane-wave basis
          is the sphere :math:`|q + G|` within the cutoff, which then excludes
          :math:`G = 0`, and the head it returns is that of another :math:`G`. The
          cutoff is read from the log or given as ``ecut``; raise it to reach larger
          momentum transfers.
        - **q_c equal to a reciprocal lattice vector,** where the spectrum with
          local-field effects is NaN.

        Include the optical limit, ``q_c = [0, 0, 0]``: the plasmon cross section is
        dominated by momentum transfers far below the k-point spacing. The frequency
        grid must reach the energy losses of the plasmon and of the continuum at the
        largest momentum transfer, and every file must share it.

        Parameters
        ----------
        files : sequence of str
            GPAW output files, one per momentum transfer, all along one direction;
            the loss function is taken as a function of :math:`|q|`.
        log : str, optional
            GPAW's response log, for ``q_c`` and the cutoff of every file.
        cell : ase.Atoms or str or os.PathLike or np.ndarray, optional
            Unit cell of the ground state for ``log`` or ``q_points``: an Atoms
            object, a (3, 3) array [Å] or any structure file ASE reads with the
            lattice in it, such as the ground-state ``.gpw``, a POSCAR, a CIF or an
            extended XYZ (a plain ``.xyz`` has no cell).
        q_points : array_like, optional
            The ``q_c`` of each file, in units of the reciprocal lattice vectors,
            shape ``(n_files, 3)``, in place of the log.
        momentum_transfers : array_like, optional
            :math:`|q|` [1/Å] of each file, without a factor :math:`2\\pi`, in place of
            the log.
        ecut : float, optional
            Plane-wave cutoff [eV] of the response, to check the momentum transfers
            when there is no log.
        local_field_effects : bool, optional
            Whether to take the spectrum with local-field effects (default) or in the
            independent-particle approximation.
        """
        if isinstance(files, str):
            files = [files]
        files = [str(path) for path in files]
        given = [x is not None for x in (log, q_points, momentum_transfers)]
        if sum(given) != 1:
            raise ValueError("give one of log, q_points and momentum_transfers")

        if log is not None:
            q_points, cutoffs = _read_gpaw_log(log)
            if len(q_points) != len(files):
                raise ValueError(
                    f"{log} has {len(q_points)} response calculations for "
                    f"{len(files)} files; give one file per calculation, in the "
                    "order they ran, or give q_points"
                )
            if ecut is None:
                ecut = cutoffs
        if q_points is not None:
            q_points = np.atleast_2d(np.asarray(q_points, dtype=float))
            if q_points.shape != (len(files), 3):
                raise ValueError("q_points must have shape (number of files, 3)")
            if cell is None:
                raise ValueError("q_c needs the cell of the ground state")
            q = np.linalg.norm(q_points @ np.linalg.inv(_read_cell(cell)).T, axis=1)
        else:
            q = np.atleast_1d(np.asarray(momentum_transfers, dtype=float))
            if q.shape != (len(files),):
                raise ValueError("give one momentum transfer per file")

        if ecut is not None:
            ecut = np.broadcast_to(np.asarray(ecut, dtype=float), q.shape)
            beyond = _HBAR2_2M * (2 * np.pi * q) ** 2 >= ecut
            if np.any(beyond):
                listed = ", ".join(
                    f"{files[i]} (|q| = {q[i]:.3g} 1/Å)" for i in np.flatnonzero(beyond)
                )
                raise ValueError(
                    "the momentum transfer of "
                    f"{listed} is beyond the response cutoff sphere "
                    f"(|q| < {np.sqrt(ecut.min() / _HBAR2_2M) / (2 * np.pi):.3g} 1/Å "
                    "for the smallest cutoff); GPAW's spectrum there is not that of "
                    "q. Leave these files out or raise ecut"
                )

        omega = None
        values = []
        for path in files:
            w, loss = _read_gpaw_response(path, local_field_effects)
            if not np.all(np.isfinite(loss)):
                raise ValueError(
                    f"{path} has non-finite values; GPAW returns NaN with local-field "
                    "effects when q_c is a reciprocal lattice vector, so leave such "
                    "momentum transfers out"
                )
            if omega is None:
                omega = w
            elif len(w) != len(omega) or not np.allclose(w, omega, atol=1e-5):
                raise ValueError(f"{path} has a different frequency grid")
            values.append(loss)

        order = np.argsort(q)
        if q[order][0] > 0:
            warnings.warn(
                "no file is at the optical limit q = 0; the small-angle cross "
                "section, which dominates the plasmon scattering, is taken at the "
                f"smallest momentum transfer {q[order][0]:.3g} 1/Å",
                stacklevel=2,
            )
        metadata = {
            "source": "gpaw.response.df.DielectricFunction",
            "files": [files[i] for i in order],
            "local_field_effects": bool(local_field_effects),
        }
        if q_points is not None:
            metadata["q_points"] = q_points[order].tolist()
        if ecut is not None:
            metadata["ecut"] = ecut[order].tolist()
        return cls(omega, q[order], np.array(values)[order], metadata)


def _read_gpaw_log(path: str):
    """``q_c`` and plane-wave cutoff [eV] of every response calculation in a GPAW
    ``DielectricFunction`` log, in the order they ran."""
    blocks = []
    block = None
    with open(path) as fd:
        for line in fd:
            text = line.strip()
            if text.startswith("Calculating chi0 body"):
                block = {}
                blocks.append(block)
            elif block is not None and text.startswith("q_c:"):
                block["q_c"] = json.loads(text[len("q_c:") :])
            elif block is not None and text.startswith("Planewave cutoff:"):
                block["ecut"] = float(text.split(":")[1])
    if not blocks:
        raise ValueError(f"{path} has no GPAW response calculations")
    if any(set(block) != {"q_c", "ecut"} for block in blocks):
        raise ValueError(f"{path} has a response calculation without q_c or cutoff")
    return [b["q_c"] for b in blocks], [b["ecut"] for b in blocks]


def _read_cell(cell) -> np.ndarray:
    """Unit cell [Å] of an Atoms object, a structure file or an array."""
    source = None
    if isinstance(cell, (str, os.PathLike)):
        from ase.io import read

        source = os.fspath(cell)
        cell = read(source)
    if hasattr(cell, "get_cell"):
        cell = cell.get_cell()
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("cell must be a (3, 3) array")
    if abs(np.linalg.det(cell)) < 1e-6:
        where = f"{source} has" if source is not None else "the cell is"
        raise ValueError(
            f"{where} no unit cell (a plain .xyz file stores none); use a file with "
            "the lattice, such as the ground-state .gpw, a POSCAR, a CIF or an "
            "extended XYZ with Lattice=, or give the Atoms or the (3, 3) array"
        )
    return cell


def _read_gpaw_response(path: str, local_field_effects: bool):
    """Frequencies [eV] and loss function from a GPAW ``eels.csv`` or ``df.csv``."""
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] == 3:
        # omega, EELS without and with local-field effects
        loss = data[:, 2] if local_field_effects else data[:, 1]
    elif data.shape[1] == 5:
        # omega, Re and Im of the dielectric function without and with local fields
        column = 3 if local_field_effects else 1
        eps = data[:, column] + 1j * data[:, column + 1]
        loss = (-1 / eps).imag
    else:
        raise ValueError(
            f"{path} has {data.shape[1]} columns; expected a GPAW eels.csv (3) or "
            "df.csv (5)"
        )
    return data[:, 0], loss
