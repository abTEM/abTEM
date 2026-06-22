"""Bloch-wave + Monte-Carlo inelastic (plasmon/phonon) scattering.

This implements the combined Bloch wave--Monte Carlo method of B. G. Mendis,
"Modelling dynamical 3D electron diffraction intensities. II. The role of inelastic
scattering", Acta Cryst. A80 (2024).

The elastic dynamical scattering is carried by the Bloch waves (diagonalize the
structure matrix once and propagate analytically). Inelastic scattering is a chain of
discrete Monte-Carlo events: between events the electron propagates elastically, and at
each event the incident wavevector is deflected by a sampled (polar, azimuthal) angle.
After a deflection only the diagonal of the structure matrix (the excitation errors)
changes [Mendis Eq. 12], so the off-diagonal structure-factor block is reused and the
matrix is cheaply re-diagonalized. The diffracted intensities ``|phi_g|^2`` are
incoherently averaged over many sampled configurations [Mendis Eq. 15].
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Optional

import numpy as np

from abtem.bloch.dynamical import (
    plane_wave_coefficients,
    set_structure_matrix_diagonal,
)
from abtem.bloch.utils import calculate_g_vec
from abtem.core.backend import get_array_module
from abtem.core.complex import abs2
from abtem.core.energy import energy2wavelength

if TYPE_CHECKING:
    from abtem.bloch.dynamical import BlochWaves
    from abtem.inelastic.plasmons import MonteCarloPlasmons, PlasmonScatteringEvents


def _rotation_y(theta: float) -> np.ndarray:
    """Rotation tilting the z axis towards x by the polar angle ``theta`` [rad]."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(phi: float) -> np.ndarray:
    """Rotation about the z axis by the azimuthal angle ``phi`` [rad]."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _deflect(rotation: np.ndarray, theta: float, phi: float) -> np.ndarray:
    """Compose a deflection by (polar ``theta``, azimuthal ``phi``) onto the cumulative
    rotation, applied in the current (post-deflection) electron frame.

    The polar and azimuthal angles are defined with respect to the electron trajectory
    prior to the scattering event, so successive deflections compose as intrinsic
    rotations (Mendis 2024, Sec. 2; Euler-rotation tracking).
    """
    return rotation @ _rotation_z(phi) @ _rotation_y(theta)


def _chain_one_configuration(
    A_base: np.ndarray,
    g: np.ndarray,
    Mii: np.ndarray,
    hkl: np.ndarray,
    energy: float,
    use_wave_eq: bool,
    thickness: float,
    depths: tuple[float, ...],
    radial_angles: tuple[float, ...],
    azimuthal_angles: tuple[float, ...],
    _untilted_eig: tuple | None = None,
) -> np.ndarray:
    """Propagate a single Monte-Carlo scattering configuration through the crystal and
    return the exit Bloch amplitudes ``phi_g``.

    The electron starts as a plane wave (delta at ``000``) and propagates elastically
    between events [Mendis Eq. 13]. At each event the incident-beam direction is
    deflected, the structure-matrix diagonal is re-formed for the new direction
    [Eq. 12], and propagation continues to the next event or the exit surface [Eq. 14].

    Parameters
    ----------
    _untilted_eig : tuple, optional
        Pre-computed ``(eigenvalues, eigenvectors)`` of the untilted structure matrix.
        Avoids a redundant ``eigh`` for the first (pre-deflection) segment.
    """
    xp = get_array_module(A_base)
    wavelength = energy2wavelength(energy)

    phi = plane_wave_coefficients(hkl, xp)

    rotation = np.eye(3)
    beam_direction: Optional[np.ndarray] = None  # None == along z, exact legacy values
    z_prev = 0.0
    _z_hat = np.array([0.0, 0.0, 1.0])

    def _propagate_eig(dz, v, C, amplitudes):
        """Propagate using pre-computed eigendecomposition."""
        phase = xp.exp(1.0j * xp.pi * dz * wavelength * v)
        psi = amplitudes / Mii
        psi = C @ (phase * (C.conj().T @ psi))
        return Mii * psi

    def propagate(dz: float, direction: Optional[np.ndarray], amplitudes):
        if dz <= 0.0:
            return amplitudes
        A = A_base.copy()
        set_structure_matrix_diagonal(
            A, g, Mii, energy, beam_direction=direction, use_wave_eq=use_wave_eq
        )
        v, C = xp.linalg.eigh(A)
        return _propagate_eig(dz, v, C, amplitudes)

    if len(depths) == 0:
        # No events — propagate full thickness with the untilted matrix.
        if _untilted_eig is not None:
            phi = _propagate_eig(thickness, *_untilted_eig, phi)
        else:
            phi = propagate(thickness, None, phi)
        return phi, beam_direction

    # First segment: use pre-computed eigendecomposition if available.
    first_dz = depths[0]
    if first_dz > 0.0:
        if _untilted_eig is not None:
            phi = _propagate_eig(first_dz, *_untilted_eig, phi)
        else:
            phi = propagate(first_dz, None, phi)

    rotation = _deflect(rotation, radial_angles[0], azimuthal_angles[0])
    beam_direction = rotation @ _z_hat
    z_prev = depths[0]

    for depth, theta, azimuth in zip(
        depths[1:], radial_angles[1:], azimuthal_angles[1:]
    ):
        phi = propagate(depth - z_prev, beam_direction, phi)
        rotation = _deflect(rotation, theta, azimuth)
        beam_direction = rotation @ _z_hat
        z_prev = depth

    phi = propagate(thickness - z_prev, beam_direction, phi)
    return phi, beam_direction


def _prepare_bloch_matrices(bloch_waves: "BlochWaves"):
    """Build the beam-independent matrices once and return them for reuse across
    thicknesses."""
    from abtem.bloch.dynamical import calculate_M_matrix

    xp = get_array_module(bloch_waves.device)

    hkl = bloch_waves.hkl
    cell = bloch_waves.cell
    energy = bloch_waves.energy

    A_base = xp.asarray(bloch_waves.calculate_structure_matrix(lazy=False))
    g = xp.asarray(calculate_g_vec(hkl, cell))
    Mii = xp.asarray(calculate_M_matrix(hkl, cell, energy))

    return A_base, g, Mii


def _precompute_untilted_eig(A_base, g, Mii, energy, use_wave_eq, xp):
    """Eigendecompose the untilted structure matrix once for reuse across all configs."""
    A = A_base.copy()
    set_structure_matrix_diagonal(
        A, g, Mii, energy, beam_direction=None, use_wave_eq=use_wave_eq
    )
    v, C = xp.linalg.eigh(A)
    return (v, C)


def calculate_bloch_plasmon_intensities(
    bloch_waves: "BlochWaves",
    events: "PlasmonScatteringEvents",
    thickness: float,
    _precomputed: tuple | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Incoherently average the Bloch-wave diffracted intensities over Monte-Carlo
    inelastic scattering configurations, resolved by excitation order (energy loss)
    [Mendis Eq. 15].

    Within each excitation order ``n`` the randomly sampled configurations are averaged
    without further weighting (the random sampling already follows the scattering
    probability distributions; Mendis 2024, "alternative" to Eq. 15). The Poisson
    probability ``P(n)`` of exactly ``n`` excitations is returned separately so that the
    energy-filtered patterns can be combined into the total (unfiltered) pattern as
    ``sum_n P(n) I^(n)``.

    Parameters
    ----------
    bloch_waves : BlochWaves
        The Bloch-wave object providing the (beam-independent) structure-factor block,
        the beam set, the unit cell and the electron energy.
    events : PlasmonScatteringEvents
        The sampled scattering configurations (depths, polar/azimuthal angles, weights).
    thickness : float
        The specimen thickness [Å].
    _precomputed : tuple, optional
        ``(A_base, g, Mii)`` from :func:`_prepare_bloch_matrices`; avoids rebuilding
        the structure matrix on every call when looping over thicknesses.

    Returns
    -------
    orders : list of int
        The excitation orders present, in increasing order.
    intensities : np.ndarray
        The energy-filtered diffracted intensities ``I^(n)_g`` averaged within each
        order, shape ``(len(orders), num_beams)``.
    weights : np.ndarray
        The Poisson weight ``P(n)`` of each order, shape ``(len(orders),)``.
    """
    xp = get_array_module(bloch_waves.device)

    hkl = bloch_waves.hkl
    energy = bloch_waves.energy
    use_wave_eq = bloch_waves.use_wave_eq

    if _precomputed is not None:
        A_base, g, Mii = _precomputed
    else:
        A_base, g, Mii = _prepare_bloch_matrices(bloch_waves)

    # Pre-compute eigendecomposition of the untilted structure matrix — reused for
    # every order-0 config and for the first (pre-deflection) segment of every
    # higher-order config, saving one eigh per configuration.
    untilted_eig = _precompute_untilted_eig(A_base, g, Mii, energy, use_wave_eq, xp)

    num_excitations = events.num_excitations
    order_counts = Counter(num_excitations)
    orders = sorted(order_counts)

    real_dtype = A_base.real.dtype
    order_index = {n: i for i, n in enumerate(orders)}
    accumulated = xp.zeros((len(orders), len(hkl)), dtype=real_dtype)
    weights = np.zeros(len(orders), dtype=real_dtype)

    iterator = zip(
        events.depths,
        events.radial_angles,
        events.azimuthal_angles,
        events.weights,
        num_excitations,
    )
    for depths, radial, azimuthal, weight, order in iterator:
        phi, _beam_dir = _chain_one_configuration(
            A_base=A_base,
            g=g,
            Mii=Mii,
            hkl=hkl,
            energy=energy,
            use_wave_eq=use_wave_eq,
            thickness=thickness,
            depths=depths,
            radial_angles=radial,
            azimuthal_angles=azimuthal,
            _untilted_eig=untilted_eig,
        )
        i = order_index[order]
        accumulated[i] += abs2(phi)
        weights[i] = weight  # P(n); identical for all configs of a given order

    for n, i in order_index.items():
        accumulated[i] /= order_counts[n]

    return orders, accumulated, weights


def calculate_bloch_diffuse_pattern(
    bloch_waves: "BlochWaves",
    events: "PlasmonScatteringEvents",
    thickness: float,
    gpts: tuple[int, int],
    extent: tuple[float, float] | None = None,
    _precomputed: tuple | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Render a 2D diffuse-background diffraction pattern using the rigid-shift model.

    For each Monte-Carlo configuration the exit Bloch amplitudes are computed
    at the discrete Bragg positions. The rigid-shift model then places each
    spot's intensity at its reciprocal-space position **shifted** by the
    transverse component of the accumulated beam tilt after all inelastic
    events in that configuration [Mendis (2024), Sec. 2]. Incoherently
    averaging over many configurations broadens each Bragg spot into a
    diffuse halo whose width grows with excitation order.

    Parameters
    ----------
    bloch_waves : BlochWaves
        The Bloch-wave object.
    events : PlasmonScatteringEvents
        The sampled scattering configurations.
    thickness : float
        The specimen thickness [Å].
    gpts : tuple of int
        Grid dimensions ``(ny, nx)`` for the output image.
    extent : tuple of float, optional
        Reciprocal-space extent ``(ky_max, kx_max)`` [1/Å] so the output spans
        ``[-ky_max, ky_max] × [-kx_max, kx_max]``. Defaults to 1.2 × the
        maximum g-vector length.
    _precomputed : tuple, optional
        ``(A_base, g, Mii)`` from :func:`_prepare_bloch_matrices`.

    Returns
    -------
    orders : list of int
        The excitation orders present, in increasing order.
    images : np.ndarray
        The rendered 2D images, shape ``(len(orders), gpts[0], gpts[1])``.
    weights : np.ndarray
        The Poisson weight ``P(n)`` of each order, shape ``(len(orders),)``.
    """
    xp = get_array_module(bloch_waves.device)

    hkl = bloch_waves.hkl
    energy = bloch_waves.energy
    use_wave_eq = bloch_waves.use_wave_eq

    if _precomputed is not None:
        A_base, g, Mii = _precomputed
    else:
        A_base, g, Mii = _prepare_bloch_matrices(bloch_waves)

    untilted_eig = _precompute_untilted_eig(A_base, g, Mii, energy, use_wave_eq, xp)

    g_np = np.asarray(g)
    g_xy = g_np[:, :2]

    if extent is None:
        g_max = float(np.max(np.linalg.norm(g_xy, axis=1)))
        extent = (g_max * 1.2, g_max * 1.2)

    ny, nx = gpts
    ky_max, kx_max = extent

    num_excitations = events.num_excitations
    order_counts = Counter(num_excitations)
    orders = sorted(order_counts)

    real_dtype = A_base.real.dtype
    order_index = {n: i for i, n in enumerate(orders)}
    images = np.zeros((len(orders), ny, nx), dtype=real_dtype)
    weights = np.zeros(len(orders), dtype=real_dtype)

    wavelength = energy2wavelength(energy)
    K = 1.0 / wavelength

    iterator = zip(
        events.depths,
        events.radial_angles,
        events.azimuthal_angles,
        events.weights,
        num_excitations,
    )
    for depths, radial, azimuthal, weight, order in iterator:
        phi, beam_dir = _chain_one_configuration(
            A_base=A_base,
            g=g,
            Mii=Mii,
            hkl=hkl,
            energy=energy,
            use_wave_eq=use_wave_eq,
            thickness=thickness,
            depths=depths,
            radial_angles=radial,
            azimuthal_angles=azimuthal,
            _untilted_eig=untilted_eig,
        )
        intensities = np.asarray(abs2(phi))

        if beam_dir is None:
            dk_xy = np.array([0.0, 0.0])
        else:
            beam_dir_np = np.asarray(beam_dir)
            dk_xy = K * beam_dir_np[:2]

        shifted_xy = g_xy + dk_xy[None, :]

        ix = np.round((shifted_xy[:, 0] + kx_max) / (2 * kx_max) * (nx - 1)).astype(int)
        iy = np.round((shifted_xy[:, 1] + ky_max) / (2 * ky_max) * (ny - 1)).astype(int)

        mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        i = order_index[order]
        np.add.at(images[i], (iy[mask], ix[mask]), intensities[mask])
        weights[i] = weight

    for n, i in order_index.items():
        images[i] /= order_counts[n]

    return orders, images, weights, extent
