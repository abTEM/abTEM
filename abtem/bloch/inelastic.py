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
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.complex import abs2
from abtem.core.energy import energy2wavelength
from abtem.core.utils import get_dtype

if TYPE_CHECKING:
    from abtem.bloch.dynamical import BlochWaves
    from abtem.inelastic.plasmons import (
        MonteCarloPlasmons,
        PlasmonScatteringEvents,
        QuadraturePlasmons,
    )


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
    """Eigendecompose the untilted structure matrix once for reuse across all configs."""
    A = A_base.copy()
    set_structure_matrix_diagonal(
        A, g, Mii, energy, beam_direction=None, use_wave_eq=use_wave_eq
    )
    v, C = xp.linalg.eigh(A)
    return (v, C)


import numba as _numba


@_numba.njit(parallel=True, cache=True)
def _numba_eigh_propagate(A_batch, Mii, wavelength, dz, psi_in):
    """Parallel eigh + propagate over a batch of tilted structure matrices."""
    B = A_batch.shape[0]
    N = A_batch.shape[1]
    intensities = np.empty((B, N))
    for b in _numba.prange(B):
        A = np.ascontiguousarray(A_batch[b])
        with _numba.objmode(res="float64[:]"):
            v, C = np.linalg.eigh(A)
            phase = np.exp(1j * np.pi * dz * wavelength * v)
            psi = psi_in / Mii
            beams_f = Mii * (C @ (phase * (C.conj().T @ psi)))
            res = beams_f.real ** 2 + beams_f.imag ** 2
        intensities[b] = res
    return intensities


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
        sg = -gn - wavelength * (g2[None, :] - gn ** 2) / 2.0
    else:
        sg = -gn - wavelength * g2[None, :] / 2.0
    return sg


def _batched_eigh_propagate(
    A_base, g, Mii, beam_dirs, wavelength, use_wave_eq, dz, psi_in, xp,
    batch_size=256,
):
    """Build tilted A matrices, eigendecompose, and propagate — all in batches.

    On GPU (CuPy) the batched ``eigh`` runs via cuSOLVER.  On CPU with numba
    available, ``numba.prange`` parallelises the per-matrix ``eigh`` calls across
    cores (numba's threading layer cooperates with OpenBLAS's GIL release during
    LAPACK).  Falls back to a sequential numpy loop otherwise.

    Parameters
    ----------
    A_base : array, shape (N, N)
    g : array, shape (N, 3)
    Mii : array, shape (N,)
    beam_dirs : array, shape (B, 3)
    wavelength : float
    use_wave_eq : bool
    dz : float — propagation distance.
    psi_in : array, shape (N,) — input Bloch amplitudes.
    batch_size : int — sub-batch size to limit GPU memory.

    Returns
    -------
    intensities : array, shape (B, N) — ``|phi_g|^2`` for each direction.
    """
    B = beam_dirs.shape[0]
    N = A_base.shape[0]

    sg_all = _batched_excitation_errors(g, beam_dirs, wavelength, use_wave_eq, xp)
    diags_all = (2.0 / wavelength) * sg_all * Mii[None, :]

    if xp is np:
        psi_np = np.ascontiguousarray(psi_in, dtype=np.complex128)
        Mii_np = np.ascontiguousarray(Mii, dtype=np.float64)
        all_intensities = np.empty((B, N))
        idx = np.arange(N)

        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            bs = end - start
            A_chunk = np.broadcast_to(A_base[None], (bs, N, N)).copy()
            A_chunk[:, idx, idx] = diags_all[start:end]
            A_chunk = np.ascontiguousarray(A_chunk, dtype=np.complex128)
            all_intensities[start:end] = _numba_eigh_propagate(
                A_chunk, Mii_np, wavelength, dz, psi_np,
            )

        return all_intensities

    psi = psi_in / Mii
    all_intensities = xp.empty((B, N), dtype=A_base.real.dtype)

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        bs = end - start

        A_batch = xp.broadcast_to(A_base[None], (bs, N, N)).copy()
        idx = xp.arange(N)
        A_batch[:, idx, idx] = diags_all[start:end]

        vals, vecs = xp.linalg.eigh(A_batch)

        phase = xp.exp(1.0j * xp.pi * dz * wavelength * vals)
        C_H_psi = xp.einsum("bji,j->bi", vecs.conj(), psi)
        beams_f = Mii[None, :] * xp.einsum(
            "bij,bj->bi", vecs, phase * C_H_psi
        )
        all_intensities[start:end] = abs2(beams_f)

    return all_intensities


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
        events.depths, events.radial_angles,
        events.azimuthal_angles, events.weights, num_excitations,
    )
    for depths, radial, azimuthal, weight, order in iterator:
        phi, _beam_dir = _chain_one_configuration(
            A_base=A_base, g=g, Mii=Mii, hkl=hkl,
            energy=energy, use_wave_eq=use_wave_eq, thickness=thickness,
            depths=depths, radial_angles=radial, azimuthal_angles=azimuthal,
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
        events.depths, events.radial_angles,
        events.azimuthal_angles, events.weights, num_excitations,
    )
    for depths, radial, azimuthal, weight, order in iterator:
        phi, beam_dir = _chain_one_configuration(
            A_base=A_base, g=g, Mii=Mii, hkl=hkl,
            energy=energy, use_wave_eq=use_wave_eq, thickness=thickness,
            depths=depths, radial_angles=radial, azimuthal_angles=azimuthal,
            _untilted_eig=untilted_eig,
        )
        intensities = asnumpy(abs2(phi))

        if beam_dir is None:
            dk_xy = np.array([0.0, 0.0])
        else:
            beam_dir_np = np.asarray(beam_dir)
            dk_xy = K * beam_dir_np[:2]

        shifted_xy = g_xy + dk_xy[None, :]
        ix = np.round(
            (shifted_xy[:, 0] + kx_max) / (2 * kx_max) * (nx - 1)
        ).astype(int)
        iy = np.round(
            (shifted_xy[:, 1] + ky_max) / (2 * ky_max) * (ny - 1)
        ).astype(int)
        mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        i = order_index[order]
        np.add.at(images[i], (iy[mask], ix[mask]), intensities[mask])
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
    nDP = len(dp_range_rad)
    dtheta = np.sqrt(2) * dp_step_rad
    theta_ratio_sq = (theta_c_rad / theta_E_rad) ** 2
    log_norm = np.log(1 + theta_ratio_sq)

    angle_x = dp_range_rad[:, None]
    angle_y = dp_range_rad[None, :]
    kt = np.sqrt(angle_x ** 2 + angle_y ** 2)

    mask = (kt > 0) & (kt < theta_c_rad)
    dphi = np.where(kt > 0, np.minimum(dtheta / kt, 2 * np.pi), 0.0)
    P_theta = 2 * kt * dtheta / (kt ** 2 + theta_E_rad ** 2) / log_norm
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

    wavelength = energy2wavelength(energy)
    K = 1.0 / wavelength

    qc_mrad = dp_max_rad * 1000
    theta_1d_mrad = np.arange(0.5, qc_mrad + 1.0, 1.0)
    theta_1d_rad = theta_1d_mrad / 1000

    dsigma = _tds_differential_cross_section(
        theta_1d_rad, scattering_factor_func, debye_waller_factor, energy,
    )
    sigma_p = 2 * np.pi * dsigma * np.sin(theta_1d_rad)
    sigma_total = float(np.sum(sigma_p) * 1e-3)

    nDP = len(dp_range_rad)
    dtheta = np.sqrt(2) * dp_step_rad

    angle_x = dp_range_rad[:, None]
    angle_y = dp_range_rad[None, :]
    kt = np.sqrt(angle_x ** 2 + angle_y ** 2)

    mask = (kt > 0) & (kt < dp_max_rad)
    dphi = np.where(kt > 0, np.minimum(dtheta / kt, 2 * np.pi), 0.0)

    sigma_interp = interp1d(
        theta_1d_mrad, sigma_p, bounds_error=False, fill_value=0.0,
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
        A_base, g, Mii, energy, use_wave_eq, xp,
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
    beam_dirs = xp.asarray(np.stack([
        sin_theta * np.cos(active_phi_angle),
        sin_theta * np.sin(active_phi_angle),
        np.cos(active_theta),
    ], axis=-1))

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
            np.add.at(
                dp_total, (py[valid], px[valid]), contribution[:, valid].sum(0)
            )

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
                radials.append(theta)
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
            A_base, g, Mii, hkl, energy, use_wave_eq, thickness,
            depths, radial, azimuthal, _untilted_eig=untilted_eig,
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


def _tilt_directions(tilts_mrad: np.ndarray) -> np.ndarray:
    """Unit beam directions for transverse tilt vectors [mrad]."""
    tilts_mrad = np.asarray(tilts_mrad, dtype=float).reshape(-1, 2)
    theta_x, theta_y = tilts_mrad[:, 0] * 1e-3, tilts_mrad[:, 1] * 1e-3
    theta = np.sqrt(theta_x**2 + theta_y**2)
    phi = np.arctan2(theta_y, theta_x)
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
    A[:, idx, idx] = diags
    return xp.linalg.eigh(A)


def _propagate_rows(vals, vecs, Mii, wavelength, dz, rows, xp):
    """Propagate a batch of Bloch amplitude vectors a per-row distance.

    Parameters
    ----------
    vals, vecs : arrays
        Eigenvalues ``(N,)`` and eigenvectors ``(N, N)`` of the structure matrix.
    dz : array, shape (k,)
        Propagation distance of every row [Å].
    rows : array, shape (k, N)
        Bloch amplitudes.
    """
    scaled = rows / Mii[None, :]
    coefficients = scaled @ vecs.conj()
    phase = xp.exp(1.0j * np.pi * xp.asarray(dz)[:, None] * wavelength * vals[None, :])
    return Mii[None, :] * ((coefficients * phase) @ vecs.T)


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
    :func:`calculate_quadrature_diffuse_pattern` for the laboratory frame.

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
    from math import comb

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

    min_angle = 0.0 if plasmons.min_angle is None else plasmons.min_angle

    # A tilt changes the excitation error of a beam by about ``g . theta``, so the
    # intensities vary on the angular scale ``1 / (g_max * thickness)``. Spacing the
    # rings by probability alone resolves this well in practice; ``max_angular_step``
    # is available when a rocking curve needs finer cells than that.
    nodes = plasmons._angular_nodes(
        energy,
        min_angle,
        max_angular_step=plasmons.max_angular_step,
        event_max_angular_step=plasmons.event_max_angular_step,
    )
    p_small, p_large = nodes["p_small"], nodes["p_large"]
    single_tilts = nodes["single"]["tilts"]
    extra_tilts = nodes["extra"]["tilts"]
    single_weights = nodes["single"]["weights"]
    extra_weights = nodes["extra"]["weights"]
    n_single, n_extra = len(single_tilts), len(extra_tilts)

    max_events = plasmons.max_tilt_events
    num_orders = plasmons.num_orders
    num_depths = plasmons.num_depths

    if p_small == 0.0 and max_events < num_orders - 1:
        raise ValueError(
            "with 'min_angle' zero every excitation changes the beam direction, so "
            f"'max_tilt_events' ({max_events}) must be at least 'max_loss_order' "
            f"({num_orders - 1}); raise it, or set a non-zero 'min_angle'"
        )

    depth_nodes = (np.arange(num_depths) + 0.5) * thickness / num_depths

    intensities = xp.zeros((num_orders, num_beams), dtype=get_dtype())

    def contribute(pattern, num_events, weight):
        """Add a path's pattern to every loss order it contributes to."""
        for n in range(0 if num_events == 0 else 1, num_orders):
            if num_events < max_events:
                orders = [num_events] if num_events <= n else []
            else:
                orders = range(num_events, n + 1)
            factor = 0.0
            for m in orders:
                factor += comb(n, m) * p_large**m * p_small ** (n - m)
            if factor:
                intensities[n] += float(factor * weight) * pattern

    v_elastic, C_elastic = _precompute_untilted_eig(
        A_base, g, Mii, energy, use_wave_eq, xp
    )
    phi_incident = plane_wave_coefficients(hkl, xp)[None]

    # zero loss, and the paths whose excitations are all pure momentum transfers
    exit_elastic = _propagate_rows(
        v_elastic, C_elastic, Mii, wavelength, np.array([thickness]), phi_incident, xp
    )
    contribute(abs2(exit_elastic)[0], 0, 1.0)

    if max_events == 0:
        return asnumpy(intensities)

    # elastic amplitudes at every depth node, the entry points of the first event
    entry_amplitudes = _propagate_rows(
        v_elastic,
        C_elastic,
        Mii,
        wavelength,
        depth_nodes,
        xp.repeat(phi_incident, num_depths, axis=0),
        xp,
    )
    entry_depths = np.arange(num_depths)
    entry_runs = np.ones(num_depths, dtype=int)
    entry_weights = np.full(num_depths, 1.0 / num_depths)

    tree_size = n_single * sum(n_extra**k for k in range(max_events))
    pbar = TqdmWrapper(
        enabled=None, total=tree_size, desc="Plasmon chains", leave=False
    )

    def visit(
        vals, vecs, amplitudes, depths, runs, weights, angular_weight, num_events
    ):
        """Exit-detect a chain and extend it with one more scattering event."""
        pbar.update_if_exists(1)

        exit_rows = _propagate_rows(
            vals, vecs, Mii, wavelength, thickness - depth_nodes[depths], amplitudes, xp
        )
        exit_intensity = abs2(exit_rows)
        for row, weight in enumerate(weights):
            contribute(exit_intensity[row], num_events, angular_weight * weight)

        if num_events == max_events:
            return

        # Advance to every depth node at or beyond the last event, still travelling in
        # the current direction. The result is shared by all child nodes.
        next_order = num_events + 1
        advanced, next_depths, next_runs, next_weights = [], [], [], []
        for node in range(num_depths):
            selected = np.where(depths <= node)[0]
            if len(selected) == 0:
                continue
            distance = depth_nodes[node] - depth_nodes[depths[selected]]
            advanced.append(
                _propagate_rows(
                    vals, vecs, Mii, wavelength, distance, amplitudes[selected], xp
                )
            )
            next_depths.append(np.full(len(selected), node))
            run = np.where(depths[selected] == node, runs[selected] + 1, 1)
            next_runs.append(run)
            next_weights.append(
                weights[selected] * (next_order / run) / num_depths
            )
        advanced = xp.concatenate(advanced, axis=0)
        next_depths = np.concatenate(next_depths)
        next_runs = np.concatenate(next_runs)
        next_weights = np.concatenate(next_weights)

        child_directions = xp.asarray(
            _tilt_directions(current_tilt[None] + extra_tilts)
        )
        child_vals, child_vecs = _batched_eig(
            A_base, g, Mii, child_directions, wavelength, use_wave_eq, xp
        )
        for child in range(n_extra):
            nonlocal_tilt = current_tilt + extra_tilts[child]
            visit_with_tilt(
                nonlocal_tilt,
                child_vals[child],
                child_vecs[child],
                advanced,
                next_depths,
                next_runs,
                next_weights,
                angular_weight * extra_weights[child],
                next_order,
            )

    def visit_with_tilt(tilt, vals, vecs, *args):
        nonlocal current_tilt
        previous, current_tilt = current_tilt, tilt
        try:
            visit(vals, vecs, *args)
        finally:
            current_tilt = previous

    current_tilt = np.zeros(2)

    first_directions = xp.asarray(_tilt_directions(single_tilts))
    first_vals, first_vecs = _batched_eig(
        A_base, g, Mii, first_directions, wavelength, use_wave_eq, xp
    )
    for node in range(n_single):
        visit_with_tilt(
            single_tilts[node],
            first_vals[node],
            first_vecs[node],
            entry_amplitudes,
            entry_depths,
            entry_runs,
            entry_weights,
            single_weights[node],
            1,
        )

    pbar.close_if_exists()
    return asnumpy(intensities)
