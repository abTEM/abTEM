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

import warnings
from collections import Counter
from typing import TYPE_CHECKING, Optional

import numpy as np

from abtem.bloch.dynamical import (
    plane_wave_coefficients,
    set_structure_matrix_diagonal,
)
from abtem.bloch.utils import calculate_g_vec
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.complex import abs2
from abtem.core.energy import energy2wavelength

if TYPE_CHECKING:
    from abtem.bloch.dynamical import BlochWaves
    from abtem.inelastic.plasmons import (
        PlasmonScatteringEvents,
        QuadraturePlasmons,
    )
    from abtem.measurements import DiffractionPatterns
    from abtem.waves import Probe


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

    # The sampled polar angles are milliradians (the abTEM convention); the rotations
    # are built in radians.
    radial_angles = tuple(theta * 1e-3 for theta in radial_angles)

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
    """Eigendecompose the untilted structure matrix once, for reuse."""
    A = A_base.copy()
    set_structure_matrix_diagonal(
        A, g, Mii, energy, beam_direction=None, use_wave_eq=use_wave_eq
    )
    v, C = xp.linalg.eigh(A)
    return (v, C)


def _batched_excitation_errors(g, beam_dirs, wavelength, use_wave_eq, xp):
    """Vectorized excitation errors for B beam directions at once.

    Parameters
    ----------
    g : array, shape (N, 3)
    beam_dirs : array, shape (B, 3) — need not be normalized.
    wavelength : float
    use_wave_eq : bool

    Returns
    -------
    sg : array, shape (B, N)
    """
    norms = xp.linalg.norm(beam_dirs, axis=-1, keepdims=True)
    d = beam_dirs / norms
    gn = xp.einsum("ni,bi->bn", g, d)
    g2 = xp.sum(g * g, axis=-1)
    if use_wave_eq:
        sg = -gn - wavelength * (g2[None, :] - gn**2) / 2.0
    else:
        sg = -gn - wavelength * g2[None, :] / 2.0
    return sg


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
        weights[i] = weight

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
) -> tuple[list[int], np.ndarray, np.ndarray, tuple[float, float]]:
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
        Grid dimensions ``(nx, ny)`` for the output image.
    extent : tuple of float, optional
        Reciprocal-space extent ``(kx_max, ky_max)`` [1/Å] so the output spans
        ``[-kx_max, kx_max] × [-ky_max, ky_max]``. Defaults to 1.2 × the
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
    extent : tuple of float
        The actual reciprocal-space half-extent ``(ky_max, kx_max)`` [1/Å].
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

    g_np = asnumpy(g)
    g_xy = g_np[:, :2]

    if extent is None:
        g_max = float(np.max(np.linalg.norm(g_xy, axis=1)))
        extent = (g_max * 1.2, g_max * 1.2)

    nx, ny = gpts
    kx_max, ky_max = extent

    num_excitations = events.num_excitations
    order_counts = Counter(num_excitations)
    orders = sorted(order_counts)

    real_dtype = A_base.real.dtype
    order_index = {n: i for i, n in enumerate(orders)}
    images = np.zeros((len(orders), nx, ny), dtype=real_dtype)
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
        intensities = asnumpy(abs2(phi))

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
        np.add.at(images[i], (ix[mask], iy[mask]), intensities[mask])
        weights[i] = weight

    for n, i in order_index.items():
        images[i] /= order_counts[n]

    return orders, images, weights, extent


# ---------------------------------------------------------------------------
# Deterministic depth-slice integration (Mendis Matlab approach)
# ---------------------------------------------------------------------------


def _plasmon_probability_grid(dp_range_rad, dp_step_rad, theta_E_rad, theta_c_rad):
    """Build 2D plasmon Lorentzian scattering probability grid.

    Follows the angular discretisation in ``Bloch_plasmon_DP.m``.

    Parameters
    ----------
    dp_range_rad : np.ndarray
        1D array of angular pixel positions [rad], e.g. ``np.arange(-0.15, 0.1505, 0.0005)``.
    dp_step_rad : float
        Angular pixel size [rad].
    theta_E_rad : float
        Characteristic plasmon scattering angle [rad].
    theta_c_rad : float
        Critical (cutoff) plasmon scattering angle [rad].

    Returns
    -------
    P : np.ndarray, shape (nDP, nDP)
        Scattering probability per pixel (sums to ~1 over the active region).
    theta : np.ndarray, shape (nDP, nDP)
        Polar scattering angle [rad] for each pixel.
    phi : np.ndarray, shape (nDP, nDP)
        Azimuthal scattering angle [rad] for each pixel.
    """
    dtheta = dp_step_rad  # polar cell of area dtheta * (kt * dphi) = one pixel
    theta_ratio_sq = (theta_c_rad / theta_E_rad) ** 2
    log_norm = np.log(1 + theta_ratio_sq)

    angle_x = dp_range_rad[:, None]
    angle_y = dp_range_rad[None, :]
    kt = np.sqrt(angle_x**2 + angle_y**2)

    mask = (kt > 0) & (kt < theta_c_rad)
    dphi = np.where(kt > 0, np.minimum(dtheta / kt, 2 * np.pi), 0.0)
    P_theta = 2 * kt * dtheta / (kt**2 + theta_E_rad**2) / log_norm
    P_phi = dphi / (2 * np.pi)
    P = np.where(mask, P_theta * P_phi, 0.0)

    phi = np.arctan2(angle_y, angle_x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    return P, kt, phi


def _phonon_probability_grid(
    dp_range_rad,
    dp_step_rad,
    dp_max_rad,
    scattering_factor_func,
    debye_waller_factor,
    energy,
):
    """Build 2D phonon (TDS) scattering probability grid.

    Follows the angular discretisation in ``Bloch_phonon_DP.m``.

    Parameters
    ----------
    dp_range_rad : np.ndarray
        1D angular pixel positions [rad].
    dp_step_rad : float
        Angular pixel size [rad].
    dp_max_rad : float
        Maximum scattering angle [rad].
    scattering_factor_func : callable
        Electron scattering factor ``f(g²)`` [1/Å].
    debye_waller_factor : float
        Isotropic Debye–Waller factor ``B = 8π²⟨u²⟩`` [Å²].
    energy : float
        Electron energy [eV].

    Returns
    -------
    P : np.ndarray, shape (nDP, nDP)
    theta : np.ndarray, shape (nDP, nDP)
    phi : np.ndarray, shape (nDP, nDP)
    sigma_total : float
        Total TDS cross-section (for MFP calculation).
    """
    from scipy.interpolate import interp1d

    from abtem.inelastic.plasmons import _tds_differential_cross_section

    qc_mrad = dp_max_rad * 1000
    theta_1d_mrad = np.arange(0.5, qc_mrad + 1.0, 1.0)
    theta_1d_rad = theta_1d_mrad / 1000

    dsigma = _tds_differential_cross_section(
        theta_1d_rad,
        scattering_factor_func,
        debye_waller_factor,
        energy,
    )
    sigma_p = 2 * np.pi * dsigma * np.sin(theta_1d_rad)
    sigma_total = float(np.sum(sigma_p) * 1e-3)

    dtheta = dp_step_rad  # polar cell of area dtheta * (kt * dphi) = one pixel

    angle_x = dp_range_rad[:, None]
    angle_y = dp_range_rad[None, :]
    kt = np.sqrt(angle_x**2 + angle_y**2)

    mask = (kt > 0) & (kt < dp_max_rad)
    dphi = np.where(kt > 0, np.minimum(dtheta / kt, 2 * np.pi), 0.0)

    sigma_interp = interp1d(
        theta_1d_mrad,
        sigma_p,
        bounds_error=False,
        fill_value=0.0,
    )
    P_theta = np.where(mask, sigma_interp(kt * 1000) * dtheta / sigma_total, 0.0)
    P_phi = dphi / (2 * np.pi)
    P = P_theta * P_phi

    phi = np.arctan2(angle_y, angle_x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    return P, kt, phi, sigma_total


def calculate_deterministic_diffuse_dp(
    bloch_waves: "BlochWaves",
    thickness: float,
    mfp: float,
    P_grid: np.ndarray,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    dp_range_rad: np.ndarray,
    dp_step_rad: float,
    num_slices: int = 19,
    batch_size: int = 256,
    _precomputed: tuple | None = None,
) -> np.ndarray:
    """Deterministic depth-slice integration for a diffuse diffraction pattern.

    Implements the algorithm of ``Bloch_plasmon_DP.m`` / ``Bloch_phonon_DP.m``
    (Mendis, Acta Cryst. A80, 2024).  The specimen is divided into *num_slices*
    depth slices.  At each slice the electron is propagated elastically to that
    depth, then for every scattering-angle pixel with non-zero probability the
    beam is deflected, the structure-matrix diagonal is updated for the new beam
    direction, and the electron is propagated to the exit surface.  The diffracted
    intensities are placed on the DP grid shifted by the transverse momentum
    transfer, weighted by ``exp(−z/λ) (Δz/λ) P(θ,φ)``.

    The eigendecompositions are batched: on GPU (CuPy) the batched ``eigh`` runs
    in parallel via cuSOLVER; on CPU (NumPy) it avoids Python-loop overhead.

    Parameters
    ----------
    bloch_waves : BlochWaves
        Provides the structure matrix, beam set, cell and energy.
    thickness : float
        Specimen thickness [Å].
    mfp : float
        Mean free path [Å] for the scattering process.
    P_grid, theta_grid, phi_grid : np.ndarray
        2D grids from :func:`_plasmon_probability_grid` or
        :func:`_phonon_probability_grid`.
    dp_range_rad : np.ndarray
        1D angular axis [rad] (same as used to build the grids).
    dp_step_rad : float
        Angular pixel size [rad].
    num_slices : int
        Number of depth slices (default 19 for 1990 Å / 100 Å).
    batch_size : int
        Number of tilted matrices to eigendecompose per batch (controls peak
        memory).  Default 256.
    _precomputed : tuple, optional
        ``(A_base, g, Mii)`` from :func:`_prepare_bloch_matrices`.

    Returns
    -------
    dp_total : np.ndarray, shape (nDP, nDP)
        The accumulated diffuse diffraction pattern.
    """
    from abtem.core.diagnostics import TqdmWrapper

    xp = get_array_module(bloch_waves.device)

    hkl = bloch_waves.hkl
    energy = bloch_waves.energy
    use_wave_eq = bloch_waves.use_wave_eq
    wavelength = energy2wavelength(energy)
    K = 1.0 / wavelength

    if _precomputed is not None:
        A_base, g, Mii = _precomputed
    else:
        A_base, g, Mii = _prepare_bloch_matrices(bloch_waves)

    v_elastic, C_elastic = _precompute_untilted_eig(
        A_base,
        g,
        Mii,
        energy,
        use_wave_eq,
        xp,
    )
    phi_inc = plane_wave_coefficients(hkl, xp)

    g_np = asnumpy(g)
    g_xy = g_np[:, :2]

    nDP = len(dp_range_rad)
    dp_total = np.zeros((nDP, nDP))

    pixel_size = K * dp_step_rad
    center_k = K * dp_range_rad[-1]

    beam_px = np.round((g_xy[:, 0] + center_k) / pixel_size).astype(int)
    beam_py = np.round((g_xy[:, 1] + center_k) / pixel_size).astype(int)

    active_idx = np.nonzero(P_grid.ravel() > 0)[0]
    n_active = len(active_idx)
    active_ix = active_idx // nDP
    active_iy = active_idx % nDP

    slice_thickness = thickness / num_slices

    active_theta = theta_grid.ravel()[active_idx]
    active_phi_angle = phi_grid.ravel()[active_idx]
    active_P = P_grid.ravel()[active_idx]

    sin_theta = np.sin(active_theta)
    beam_dirs = xp.asarray(
        np.stack(
            [
                sin_theta * np.cos(active_phi_angle),
                sin_theta * np.sin(active_phi_angle),
                np.cos(active_theta),
            ],
            axis=-1,
        )
    )

    shift_x = np.round(dp_range_rad[active_ix] / dp_step_rad).astype(int)
    shift_y = np.round(dp_range_rad[active_iy] / dp_step_rad).astype(int)

    def _propagate(dz, v, C, amplitudes):
        phase = xp.exp(1.0j * xp.pi * dz * wavelength * v)
        psi = amplitudes / Mii
        psi = C @ (phase * (C.conj().T @ psi))
        return Mii * psi

    # The eigendecomposition depends on the scattering direction but not on the depth,
    # so the direction batches are the outer loop: one decomposition per direction
    # serves every depth slice, instead of one per (direction, depth) pair.
    depths = (np.arange(num_slices) + 0.5) * slice_thickness
    entry_beams = xp.stack(
        [_propagate(depth, v_elastic, C_elastic, phi_inc) for depth in depths]
    )
    depth_weights = np.exp(-depths / mfp) * (slice_thickness / mfp)
    remaining = thickness - depths

    pbar = TqdmWrapper(
        enabled=None, total=n_active, desc="Scattering angles", leave=False
    )

    for start in range(0, n_active, batch_size):
        stop = min(start + batch_size, n_active)
        vals, vecs = _batched_eig(
            A_base, g, Mii, beam_dirs[start:stop], wavelength, use_wave_eq, xp
        )
        scaled = entry_beams / Mii[None, :]
        # (slices, batch, beams): every depth slice through every direction of the batch
        coefficients = xp.einsum("bji,sj->sbi", vecs.conj(), scaled)
        phase = xp.exp(
            1.0j
            * np.pi
            * xp.asarray(remaining)[:, None, None]
            * wavelength
            * vals[None]
        )
        beams_f = Mii[None, None, :] * xp.einsum(
            "bij,sbj->sbi", vecs, phase * coefficients
        )
        intensities_np = asnumpy(abs2(beams_f))

        for b in range(start, stop):
            px = beam_px + shift_x[b]
            py = beam_py + shift_y[b]
            valid = (px >= 0) & (px < nDP) & (py >= 0) & (py < nDP)
            contribution = (
                depth_weights[:, None] * active_P[b] * intensities_np[:, b - start]
            )
            np.add.at(dp_total, (px[valid], py[valid]), contribution[:, valid].sum(0))

        pbar.update_if_exists(stop - start)

    pbar.close_if_exists()
    return dp_total


# ---------------------------------------------------------------------------
# Interleaved phonon + plasmon Monte-Carlo (Mendis Matlab approach)
# ---------------------------------------------------------------------------


def draw_mixed_scattering_events(
    mfp_plasmon: float,
    mfp_phonon: float,
    thickness: float,
    n_plasmon_target: int,
    theta_E_rad: float,
    theta_c_rad: float,
    phonon_theta_grid: np.ndarray,
    phonon_cdf: np.ndarray,
    num_configs: int = 50000,
    max_events: int = 10,
    seed: int | None = None,
) -> tuple[list, list, list]:
    """Draw interleaved phonon + plasmon MC configurations.

    Follows ``Bloch_single_plasmon_phonon.m`` / ``Bloch_double_plasmon_phonon.m``.
    At each scattering site a coin flip (weighted by the relative MFP) decides
    whether the event is a plasmon or phonon excitation.  The path length between
    events is drawn from the corresponding MFP distribution.  Only configurations
    with exactly *n_plasmon_target* plasmon events within the specimen are kept.

    Returns
    -------
    config_depths : list of tuple
        Scattering depths within the specimen for each accepted configuration.
    config_radials : list of tuple
        Polar scattering angles [rad] for each event.
    config_azimuths : list of tuple
        Azimuthal scattering angles [rad] for each event.
    """
    rng = np.random.default_rng(seed)
    plasmon_ratio = (1 / mfp_plasmon) / (1 / mfp_plasmon + 1 / mfp_phonon)

    config_depths = []
    config_radials = []
    config_azimuths = []

    configs_found = 0
    while configs_found < num_configs:
        event_depths = np.zeros(max_events)
        event_types = np.zeros(max_events, dtype=int)
        depth = 0.0
        n_plasmons = 0

        for e in range(max_events):
            if rng.random() <= plasmon_ratio:
                sp = -mfp_plasmon * np.log(rng.random())
                event_types[e] = 1
                depth += sp
                if depth <= thickness:
                    n_plasmons += 1
            else:
                sp = -mfp_phonon * np.log(rng.random())
                depth += sp
            event_depths[e] = depth

        if n_plasmons == n_plasmon_target and depth > thickness:
            depths = []
            radials = []
            azimuths = []
            for e in range(max_events):
                if event_depths[e] > thickness:
                    break
                depths.append(event_depths[e])
                if event_types[e] == 1:
                    theta = theta_E_rad * np.sqrt(
                        ((theta_c_rad / theta_E_rad) ** 2 + 1) ** rng.random() - 1
                    )
                else:
                    theta = float(
                        np.interp(rng.random(), phonon_cdf, phonon_theta_grid)
                    )
                phi = 2 * np.pi * rng.random()
                radials.append(theta * 1e3)  # [mrad], as every event producer
                azimuths.append(phi)

            config_depths.append(tuple(depths))
            config_radials.append(tuple(radials))
            config_azimuths.append(tuple(azimuths))
            configs_found += 1

    return config_depths, config_radials, config_azimuths


def calculate_mixed_mc_intensities(
    bloch_waves: "BlochWaves",
    thickness: float,
    config_depths: list,
    config_radials: list,
    config_azimuths: list,
    _precomputed: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average Bloch-wave beam intensities over mixed phonon+plasmon MC configs.

    Parameters
    ----------
    bloch_waves : BlochWaves
    thickness : float
    config_depths, config_radials, config_azimuths : list of tuple
        From :func:`draw_mixed_scattering_events`.
    _precomputed : tuple, optional

    Returns
    -------
    intensities : np.ndarray, shape (num_beams,)
        Average beam intensities.
    transverse_k : np.ndarray, shape (num_configs,)
        Transverse momentum [1/Å] for each configuration (for radial profiles).
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

    wavelength = energy2wavelength(energy)
    K = 1.0 / wavelength
    n_configs = len(config_depths)

    real_dtype = A_base.real.dtype
    accumulated = xp.zeros(len(hkl), dtype=real_dtype)
    transverse_k = np.zeros(n_configs)

    for i, (depths, radial, azimuthal) in enumerate(
        zip(config_depths, config_radials, config_azimuths)
    ):
        phi, beam_dir = _chain_one_configuration(
            A_base,
            g,
            Mii,
            hkl,
            energy,
            use_wave_eq,
            thickness,
            depths,
            radial,
            azimuthal,
            _untilted_eig=untilted_eig,
        )
        accumulated += abs2(phi)

        if beam_dir is not None:
            beam_dir_np = np.asarray(beam_dir)
            transverse_k[i] = K * np.sqrt(
                float(beam_dir_np[0]) ** 2 + float(beam_dir_np[1]) ** 2
            )

    accumulated /= n_configs
    return asnumpy(accumulated), transverse_k


# ---------------------------------------------------------------------------
# Deterministic (quadrature) plasmon scattering
# ---------------------------------------------------------------------------


def _tilt_directions(tilts: np.ndarray) -> np.ndarray:
    """Unit beam directions of transverse tilt vectors [mrad], shape (..., 2)."""
    tilts = np.asarray(tilts, dtype=float) * 1e-3
    theta = np.sqrt(tilts[..., 0] ** 2 + tilts[..., 1] ** 2)
    phi = np.arctan2(tilts[..., 1], tilts[..., 0])
    sin_theta = np.sin(theta)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)], axis=-1
    )


def _batched_eig(A_base, g, Mii, beam_dirs, wavelength, use_wave_eq, xp):
    """Eigendecompose the structure matrix for a batch of beam directions.

    Only the diagonal depends on the direction, so the off-diagonal structure-factor
    block is reused (Mendis, Acta Cryst. A80, 2024, Eq. 12).

    Returns
    -------
    vals : array, shape (B, N)
    vecs : array, shape (B, N, N)
    """
    sg = _batched_excitation_errors(g, beam_dirs, wavelength, use_wave_eq, xp)
    diags = (2.0 / wavelength) * sg * Mii[None, :]
    B, N = diags.shape
    A = xp.broadcast_to(A_base[None], (B, N, N)).copy()
    idx = xp.arange(N)
    A[:, idx, idx] = diags.astype(A.dtype)
    return xp.linalg.eigh(A)


def _propagate_batch(vals, vecs, Mii, wavelength, dz, rows, xp):
    """Propagate Bloch amplitudes over per-row distances for a batch of directions.

    Parameters
    ----------
    vals, vecs : arrays, shapes (B, N) and (B, N, N)
        Eigenvalues and eigenvectors of the structure matrix of every direction.
    dz : array, shape (D,)
        Propagation distance of every row [Å], shared by the batch.
    rows : array, shape (B, D, N)
        Bloch amplitudes.

    Returns
    -------
    array, shape (B, D, N)
    """
    scaled = rows / Mii
    # (scaled^* V)^* = scaled V^*: conjugates the small operand instead of ``vecs``
    coefficients = (scaled.conj() @ vecs).conj()
    dz = xp.asarray(dz, dtype=vals.dtype)
    phase = xp.exp(1.0j * np.pi * wavelength * dz[None, :, None] * vals[:, None, :])
    phase = phase.astype(rows.dtype)
    return Mii * ((coefficients * phase) @ xp.swapaxes(vecs, -1, -2))


def _check_beam_set(bloch_waves, g, max_tilt: float):
    """Warn when tilting the beam by ``max_tilt`` [mrad] excites beams that the beam
    set cannot hold: a tilt shifts the excitation errors by about ``g . theta``."""
    g_xy = float(np.max(np.linalg.norm(asnumpy(g)[:, :2], axis=1)))
    shift = g_xy * max_tilt * 1e-3
    if shift > bloch_waves.sg_max:
        warnings.warn(
            f"a tilt of {max_tilt:.1f} mrad shifts the excitation errors of the outer "
            f"beams by up to {shift:.3f} 1/Å, beyond 'sg_max' = {bloch_waves.sg_max}; "
            "beams that become strongly excited after a deflection are missing from "
            "the beam set. Increase 'sg_max' (or reduce 'g_max')",
            stacklevel=3,
        )


def _walk_quadrature_chains(
    bloch_waves: "BlochWaves",
    thickness: float,
    incident_tilts: np.ndarray,
    emit,
    nodes: dict | None = None,
    max_events: int = 0,
    num_depths: int = 1,
    _precomputed: tuple | None = None,
):
    """Visit every chain of direction-changing plasmon events of the quadrature.

    The incident plane waves of a batch travel in the directions ``incident_tilts``
    (shape ``(B, 2)``, [mrad]); a convergent beam is such a batch. Between events the
    electrons propagate analytically with the structure matrix of their current
    direction. Only the direction enters the structure matrix, so the
    eigendecomposition of a node of the chain tree is shared by every depth-node
    combination of its events.

    ``emit(index, weight, pattern)`` is called once per node: ``index`` holds the
    angular-node indices along the chain (empty for the elastic wave), ``weight`` the
    angular weight of the chain, and ``pattern`` the exit intensities of the beams
    averaged over the event depths, shape ``(B, N)``, per incident electron.
    """
    from abtem.core.diagnostics import TqdmWrapper

    xp = get_array_module(bloch_waves.device)
    if _precomputed is not None:
        A_base, g, Mii = _precomputed
    else:
        A_base, g, Mii = _prepare_bloch_matrices(bloch_waves)

    energy = bloch_waves.energy
    wavelength = energy2wavelength(energy)
    use_wave_eq = bloch_waves.use_wave_eq
    hkl = bloch_waves.hkl
    num_beams = len(hkl)
    real_dtype = A_base.real.dtype
    incident_tilts = np.asarray(incident_tilts, dtype=float).reshape(-1, 2)
    num_incident = len(incident_tilts)

    def eig(tilt):
        directions = xp.asarray(_tilt_directions(incident_tilts + tilt[None]))
        return _batched_eig(A_base, g, Mii, directions, wavelength, use_wave_eq, xp)

    phi_incident = plane_wave_coefficients(hkl, xp).astype(A_base.dtype)
    incident = xp.broadcast_to(phi_incident, (num_incident, 1, num_beams))

    vals, vecs = eig(np.zeros(2))
    exit_elastic = _propagate_batch(
        vals, vecs, Mii, wavelength, np.array([thickness]), incident, xp
    )
    emit((), 1.0, abs2(exit_elastic)[:, 0])

    if max_events == 0:
        return

    single_tilts = nodes["single"]["tilts"]
    extra_tilts = nodes["extra"]["tilts"]
    single_weights = nodes["single"]["weights"]
    extra_weights = nodes["extra"]["weights"]
    n_single, n_extra = len(single_tilts), len(extra_tilts)
    max_tilt = float(np.max(np.linalg.norm(incident_tilts, axis=1))) + float(
        np.max(np.linalg.norm(single_tilts, axis=1))
    )
    if max_events > 1:
        max_tilt += (max_events - 1) * float(
            np.max(np.linalg.norm(extra_tilts, axis=1))
        )
    _check_beam_set(bloch_waves, g, max_tilt)

    depth_nodes = (np.arange(num_depths) + 0.5) * thickness / num_depths

    # elastic amplitudes at every depth node, the entry points of the first event
    entry = _propagate_batch(
        vals,
        vecs,
        Mii,
        wavelength,
        depth_nodes,
        xp.broadcast_to(phi_incident, (num_incident, num_depths, num_beams)),
        xp,
    )
    entry_depths = np.arange(num_depths)
    entry_runs = np.ones(num_depths, dtype=int)
    entry_weights = np.full(num_depths, 1.0 / num_depths)

    tree_size = n_single * sum(n_extra**k for k in range(max_events))
    pbar = TqdmWrapper(
        enabled=None, total=tree_size, desc="Plasmon chains", leave=False
    )

    def visit(index, tilt, vals, vecs, amplitudes, depths, runs, weights, weight):
        """Exit-detect a chain and extend it with one more scattering event."""
        pbar.update_if_exists(1)

        exit_rows = _propagate_batch(
            vals, vecs, Mii, wavelength, thickness - depth_nodes[depths], amplitudes, xp
        )
        row_weights = xp.asarray(weights, dtype=real_dtype)
        emit(index, weight, xp.tensordot(abs2(exit_rows), row_weights, axes=([1], [0])))

        if len(index) == max_events:
            return

        # Advance to every depth node at or beyond the last event, still travelling in
        # the current direction; shared by all child nodes. A chain whose events fall
        # in the same depth bin ``run`` times in a row carries the multinomial weight
        # of that ordered sequence of independent uniform depths.
        next_order = len(index) + 1
        advanced, next_depths, next_runs, next_weights = [], [], [], []
        for node in range(num_depths):
            selected = np.where(depths <= node)[0]
            if len(selected) == 0:
                continue
            distance = depth_nodes[node] - depth_nodes[depths[selected]]
            advanced.append(
                _propagate_batch(
                    vals, vecs, Mii, wavelength, distance, amplitudes[:, selected], xp
                )
            )
            next_depths.append(np.full(len(selected), node))
            run = np.where(depths[selected] == node, runs[selected] + 1, 1)
            next_runs.append(run)
            next_weights.append(weights[selected] * (next_order / run) / num_depths)
        advanced = xp.concatenate(advanced, axis=1)
        next_depths = np.concatenate(next_depths)
        next_runs = np.concatenate(next_runs)
        next_weights = np.concatenate(next_weights)

        for child in range(n_extra):
            child_tilt = tilt + extra_tilts[child]
            child_vals, child_vecs = eig(child_tilt)
            visit(
                index + (child,),
                child_tilt,
                child_vals,
                child_vecs,
                advanced,
                next_depths,
                next_runs,
                next_weights,
                weight * extra_weights[child],
            )

    for node in range(n_single):
        node_vals, node_vecs = eig(single_tilts[node])
        visit(
            (node,),
            single_tilts[node],
            node_vals,
            node_vecs,
            entry,
            entry_depths,
            entry_runs,
            entry_weights,
            single_weights[node],
        )

    pbar.close_if_exists()


def _quadrature_nodes(bloch_waves, plasmons, min_angle):
    """Angular nodes of the model and the check that every order is reachable."""
    nodes = plasmons._angular_nodes(
        bloch_waves.energy,
        min_angle,
        max_angular_step=plasmons.max_angular_step,
        event_max_angular_step=plasmons.event_max_angular_step,
    )
    if nodes["p_small"] == 0.0 and plasmons.max_tilt_events < plasmons.num_orders - 1:
        raise ValueError(
            "with 'min_angle' zero every excitation changes the beam direction, so "
            f"'max_tilt_events' ({plasmons.max_tilt_events}) must be at least "
            f"'max_loss_order' ({plasmons.num_orders - 1}); raise it, or set a "
            "non-zero 'min_angle'"
        )
    return nodes


def calculate_quadrature_plasmon_intensities(
    bloch_waves: "BlochWaves",
    thickness: float,
    plasmons: "QuadraturePlasmons",
    _precomputed: tuple | None = None,
) -> np.ndarray:
    """Loss-order resolved beam intensities from deterministic plasmon quadrature.

    The Bloch-wave counterpart of
    :func:`abtem.inelastic.plasmons.quadrature_plasmon_multislice_and_detect`. The
    incoherent integral over plasmon scattering angle and depth is evaluated on the
    quadrature nodes of :class:`~abtem.inelastic.plasmons.QuadraturePlasmons` instead
    of being sampled at random: every chain of scattering events is a sequence of beam
    deflections, and between events the electron propagates analytically with the
    structure matrix of its current direction.

    Only the *direction* enters the structure matrix, so the eigendecomposition depends
    on the cumulative tilt of a chain but not on the depths at which its events
    occurred. One eigendecomposition per node of the chain tree therefore serves every
    depth-node combination, which makes the depth quadrature almost free -- the reason
    this is much cheaper than the multislice quadrature, where every copy of the wave
    function must be propagated through the remaining specimen.

    The beams are detected in their own tilted frame of reference (the momentum
    transferred to the electron would move them off the reciprocal lattice); see
    :func:`calculate_quadrature_plasmon_diffraction_patterns` for the laboratory
    frame.

    Parameters
    ----------
    bloch_waves : BlochWaves
        Provides the structure matrix, beam set, cell and energy.
    thickness : float
        Specimen thickness [Å].
    plasmons : QuadraturePlasmons
        The quadrature model. ``min_angle`` defaults to zero here: unlike the
        multislice grid there is no pixel below which a tilt cannot change the
        channeling, so every node changes the beam direction.

        Setting ``max_angular_step`` is worthwhile for Bloch waves. Spacing the rings
        by probability alone leaves the outermost ring spanning many milliradians,
        which the rocking curves resolve poorly. For 500 Å of silicon with a 2 Å-1 beam
        set, the single-loss error against a 800-sample Monte Carlo run (whose own noise
        is 0.008) falls from 0.029 with 48 nodes to 0.007 with 200 nodes at a step of
        4 mrad, and does not improve with finer steps.

        The beam set must be generous enough for tilted beams: a tilt moves the Ewald
        sphere, so beams with larger excitation errors are excited than at the zone
        axis. On the same silicon case the elastic pattern of a 5 mrad tilted plane
        wave agrees with multislice to 2.8 percent with ``sg_max=0.3, g_max=5`` but only
        to 7.3 percent with ``sg_max=0.05, g_max=2.5``. A warning is issued when the
        largest tilt shifts the excitation errors of the outer beams beyond ``sg_max``.
    _precomputed : tuple, optional
        ``(A_base, g, Mii)`` from :func:`_prepare_bloch_matrices`.

    Returns
    -------
    intensities : np.ndarray, shape (num_orders, num_beams)
        The beam intensities of every loss order, each normalized to the incident
        electron count. Multiply by
        :meth:`~abtem.inelastic.plasmons.QuadraturePlasmons.excitation_weights` for the
        Poisson-weighted signal.
    """
    from abtem.inelastic.plasmons import _loss_order_factors

    xp = get_array_module(bloch_waves.device)
    if _precomputed is None:
        _precomputed = _prepare_bloch_matrices(bloch_waves)

    min_angle = 0.0 if plasmons.min_angle is None else plasmons.min_angle
    nodes = _quadrature_nodes(bloch_waves, plasmons, min_angle)
    num_orders = plasmons.num_orders
    max_events = plasmons.max_tilt_events
    real_dtype = _precomputed[0].real.dtype
    intensities = xp.zeros((num_orders, len(bloch_waves.hkl)), dtype=real_dtype)

    def emit(index, weight, pattern):
        for n, m, factor in _loss_order_factors(
            num_orders, len(index), max_events, nodes["p_small"], nodes["p_large"]
        ):
            intensities[n] += real_dtype.type(factor * weight) * pattern[0]

    _walk_quadrature_chains(
        bloch_waves,
        float(thickness),
        np.zeros((1, 2)),
        emit,
        nodes=nodes,
        max_events=max_events,
        num_depths=plasmons.num_depths,
        _precomputed=_precomputed,
    )
    return asnumpy(intensities)


def calculate_quadrature_plasmon_diffraction_patterns(
    bloch_waves: "BlochWaves",
    thickness: float,
    probe: "Probe",
    plasmons: "QuadraturePlasmons | None" = None,
    max_angle: str | float = "valid",
    _precomputed: tuple | None = None,
) -> "DiffractionPatterns":
    """Convergent-beam diffraction patterns from Bloch waves, with plasmon scattering
    by quadrature in the laboratory frame.

    Every pixel of the probe aperture is an incident plane wave whose Bragg beams land
    at the incident direction plus the reciprocal lattice vector; the discs of a
    convergent-beam pattern are the incoherent sum over the aperture (exact when the
    discs do not overlap, which is also when the pattern is independent of the probe
    position and aberrations). With ``plasmons`` the beams of every chain of the
    quadrature are placed at the deflected direction, and the sub-cell momentum
    transfer is applied with the same Lorentzian kernels as the multislice quadrature,
    so the result is the Bloch-wave counterpart of
    ``probe.multislice(potential, PixelatedDetector(max_angle), plasmons=plasmons)``
    on the probe's grid: the same nodes, weights, kernels and cropping, with the
    elastic dynamical scattering carried by the beam set of ``bloch_waves`` instead of
    the multislice potential (no thermal diffuse scattering).

    Parameters
    ----------
    bloch_waves : BlochWaves
        The beam set, oriented so that its ``x`` and ``y`` axes are those of the
        probe grid (``orientation_matrix``); the reciprocal lattice vectors must fall
        on the reciprocal grid of the probe, i.e. the probe extent must be a multiple
        of the projected cell.
    thickness : float
        Specimen thickness [Å].
    probe : Probe
        Defines the aperture, the grid (``extent`` and ``gpts``, matched to the
        multislice potential to compare with) and the energy.
    plasmons : QuadraturePlasmons, optional
        The quadrature model. ``min_angle`` defaults to one reciprocal pixel as in the
        multislice, ``lab_frame=False`` keeps every chain in its tilted frame.
    max_angle : float or {'cutoff', 'valid', 'full'}
        Cropping of the patterns, as for :class:`~abtem.detectors.PixelatedDetector`.
    _precomputed : tuple, optional
        ``(A_base, g, Mii)`` from :func:`_prepare_bloch_matrices`.

    Returns
    -------
    DiffractionPatterns
        With a leading :class:`~abtem.core.axes.PlasmonOrderAxis` when ``plasmons`` is
        given, every channel normalized to the incident electron count (the Poisson
        weights are in the metadata under ``"plasmon_weights"``).
    """
    from abtem.core.fft import fft2, fft_crop, ifft2
    from abtem.inelastic.plasmons import _lorentzian_kernels, _loss_order_factors
    from abtem.measurements import DiffractionPatterns

    xp = get_array_module(bloch_waves.device)
    if _precomputed is None:
        _precomputed = _prepare_bloch_matrices(bloch_waves)
    A_base, g, Mii = _precomputed
    real_dtype = A_base.real.dtype
    complex_dtype = A_base.dtype

    if not np.isclose(probe.energy, bloch_waves.energy):
        raise ValueError("the probe and the Bloch waves must have the same energy")

    probe_waves = probe.build(lazy=False)
    gpts = tuple(probe_waves.gpts)
    extent = tuple(probe_waves.extent)
    wavelength = probe_waves.wavelength
    angular_sampling = tuple(probe_waves.angular_sampling)
    new_gpts = probe_waves._gpts_within_angle(max_angle)

    # the aperture: weight and direction [mrad] of every incident plane wave
    aperture = asnumpy(abs2(fft2(probe_waves.array)))
    aperture = aperture / aperture.sum()
    pixels = np.argwhere(aperture > 1e-8 * aperture.max())
    weights = xp.asarray(aperture[pixels[:, 0], pixels[:, 1]], dtype=real_dtype)
    frequencies = [np.fft.fftfreq(n, d=e / n) for n, e in zip(gpts, extent)]
    incident_tilts = np.stack(
        [frequencies[0][pixels[:, 0]], frequencies[1][pixels[:, 1]]], axis=1
    ) * (wavelength * 1e3)

    # pixel offsets of the beams; the reciprocal lattice must sit on the grid
    g_pixels = asnumpy(g)[:, :2] * np.array(extent)
    deviation = float(np.max(np.abs(g_pixels - np.round(g_pixels))))
    if deviation > 0.05:
        warnings.warn(
            f"reciprocal lattice vectors are up to {deviation:.2f} pixels off the "
            "reciprocal grid of the probe; orient the Bloch waves along the probe grid "
            "and make the probe extent a multiple of the projected cell",
            stacklevel=2,
        )
    g_pixels = np.round(g_pixels).astype(int)
    signed = [np.fft.fftfreq(n, d=1 / n).astype(int) for n in gpts]
    beam_x = signed[0][pixels[:, 0]][:, None] + g_pixels[None, :, 0]
    beam_y = signed[1][pixels[:, 1]][:, None] + g_pixels[None, :, 1]
    inside = (np.abs(beam_x) <= (gpts[0] - 1) // 2) & (
        np.abs(beam_y) <= (gpts[1] - 1) // 2
    )
    flat = xp.asarray(
        (np.mod(beam_x, gpts[0]) * gpts[1] + np.mod(beam_y, gpts[1]))[inside]
    )
    inside = xp.asarray(inside)

    def rasterize(pattern):
        """Unshifted diffraction pattern of the beams of every incident direction."""
        values = (weights[:, None] * pattern)[inside]
        image = xp.bincount(flat, weights=values, minlength=gpts[0] * gpts[1])
        return image.reshape(gpts).astype(real_dtype)

    if plasmons is None:
        nodes, max_events, num_depths, num_orders = None, 0, 1, 1
        kernels = None
    else:
        min_angle = plasmons.min_angle
        if min_angle is None:
            min_angle = max(angular_sampling)
        nodes = _quadrature_nodes(bloch_waves, plasmons, min_angle)
        max_events = plasmons.max_tilt_events
        num_depths = plasmons.num_depths
        num_orders = plasmons.num_orders
        if plasmons.lab_frame:
            kernels = _lorentzian_kernels(gpts, angular_sampling, nodes, xp=xp)
        else:
            kernels = None

    if kernels is None:
        one = xp.ones((), dtype=complex_dtype)
        ks_pow = kl_pow = [one] * num_orders
    else:
        ks_pow = [kernels["small"] ** j for j in range(num_orders)]
        kl_pow = [kernels["large"] ** j for j in range(num_orders)]

    accumulators = [None] * num_orders

    def emit(index, weight, pattern):
        f = fft2(rasterize(pattern).astype(complex_dtype))
        if kernels is not None and len(index) > 0:
            f = f * kernels["single"][index[0]]
            for node in index[1:]:
                f = f * kernels["extra"][node]
        if plasmons is None:
            factors = [(0, 0, 1.0)]
        else:
            factors = _loss_order_factors(
                num_orders, len(index), max_events, nodes["p_small"], nodes["p_large"]
            )
        for n, m, factor in factors:
            term = f * real_dtype.type(factor * weight)
            if n - m > 0:
                term = term * ks_pow[n - m]
            if m - len(index) > 0:
                term = term * kl_pow[m - len(index)]
            accumulators[n] = (
                term if accumulators[n] is None else accumulators[n] + term
            )

    _walk_quadrature_chains(
        bloch_waves,
        float(thickness),
        incident_tilts,
        emit,
        nodes=nodes,
        max_events=max_events,
        num_depths=num_depths,
        _precomputed=_precomputed,
    )

    patterns = []
    for accumulator in accumulators:
        if accumulator is None:
            patterns.append(xp.zeros(gpts, dtype=real_dtype))
        else:
            patterns.append(xp.clip(ifft2(accumulator).real, 0, None))
    array = xp.stack(patterns)
    array = fft_crop(array, new_shape=array.shape[:-2] + tuple(new_gpts))
    array = xp.fft.fftshift(array, axes=(-2, -1))
    # the detectors of the multislice counterpart return host arrays (``to_cpu=True``)
    array = asnumpy(array)

    metadata = {
        "energy": bloch_waves.energy,
        "sg_max": bloch_waves.sg_max,
        "g_max": bloch_waves.g_max,
        "label": "intensity",
        "units": "arb. unit",
    }
    if plasmons is None:
        ensemble_axes_metadata = []
        array = array[0]
    else:
        ensemble_axes_metadata = [plasmons.order_axis]
        metadata["plasmon_weights"] = plasmons.excitation_weights(float(thickness))

    return DiffractionPatterns(
        array=array,
        sampling=probe_waves.reciprocal_space_sampling,
        fftshift=True,
        ensemble_axes_metadata=ensemble_axes_metadata,
        metadata=metadata,
    )
