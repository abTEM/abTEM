from __future__ import annotations

import warnings

import numpy as np
from numba import jit
from scipy.special import erf

VALID_REGULARIZATIONS = frozenset(
    {"none", "yukawa", "rozzi_spherical", "rozzi_cylindrical", "rozzi_slab"}
)


@jit(nopython=True, nogil=True)
def scattering_factor(k, p):
    return (
        p[0, 0] * np.exp(-p[1, 0] * k**2.0)
        + p[0, 1] * np.exp(-p[1, 1] * k**2.0)
        + p[0, 2] * np.exp(-p[1, 2] * k**2.0)
        + p[0, 3] * np.exp(-p[1, 3] * k**2.0)
        + p[0, 4] * np.exp(-p[1, 4] * k**2.0)
    )


@jit(nopython=True, nogil=True)
def scattering_factor_k2(k2, p):
    return (
        p[0, 0] * np.exp(-p[1, 0] * k2)
        + p[0, 1] * np.exp(-p[1, 1] * k2)
        + p[0, 2] * np.exp(-p[1, 2] * k2)
        + p[0, 3] * np.exp(-p[1, 3] * k2)
        + p[0, 4] * np.exp(-p[1, 4] * k2)
    )


def ionic_coulomb_term(
    q2,
    delta_Z,
    regularization="none",
    kappa=None,
    R=None,
    L_cell=None,
):
    """Regularized Coulomb piece of the Peng (1998) ionic electron scattering factor.

    Returns ``delta_Z * K(q²)`` where ``K`` is the regularized kernel for the
    divergent 1/q² Coulomb term arising from the net ionic charge ΔZ.  The
    Mott–Bethe prefactor C = m₀e²/(2h²ε₀) is **not** applied here; it is the
    caller's responsibility to multiply by C (and any unit-conversion factors
    appropriate for the host code).

    In abTEM's convention ``q`` corresponds to ``k = 2s`` (where ``s = sin θ/λ``
    is Peng's scattering parameter), so ``q²`` passed in is ``k²``.

    Parameters
    ----------
    q2 : array-like
        Squared scattering wavevector magnitudes [1/Å²].  May be 1-D, 2-D
        (FFT grid), or any broadcastable shape.
    delta_Z : float
        Net ionic charge (ΔZ = Z − Z₀).  Positive for cations, negative for
        anions.  Returns zero immediately when delta_Z == 0.
    regularization : str, optional
        Regularization scheme for the q→0 Coulomb divergence.  Options:

        ``"none"``
            Returns the bare ΔZ/q² expression.  **Returns np.inf at q = 0.**
            Use only for testing or when the caller will mask q = 0 separately.
            This is the default and preserves backward-compatible behaviour in
            the imaging pipeline (the DC component is conventionally excluded).

        ``"yukawa"``
            Replaces 1/q² with 1/(q² + κ²), giving ΔZ/κ² at q = 0.  κ should
            match the physical screening length of the experiment (e.g.,
            2π/L_cell for a periodic box).  Requires ``kappa`` or ``L_cell``.

        ``"rozzi_spherical"``
            Spherically truncated Coulomb kernel of Rozzi et al. (2006):
            ΔZ·[1 − cos(qR)]/q².  At q = 0 the Taylor expansion gives ΔZ·R²/2,
            which is evaluated analytically to avoid 0/0 cancellation.  R should
            be chosen as half the smallest cell dimension so that the real-space
            interaction is zero beyond R.  Requires ``R`` or ``L_cell``.

        ``"rozzi_cylindrical"``  /  ``"rozzi_slab"``
            Cylindrical (2D-periodic) and slab (1D-periodic) variants from
            Rozzi et al. (2006) Eqs. (14)–(16).  **Not yet implemented**;
            raises ``NotImplementedError``.

    kappa : float or None
        Yukawa screening wavevector κ [1/Å].  Required when
        ``regularization="yukawa"`` unless ``L_cell`` is provided.  If both
        ``kappa`` and ``L_cell`` are given, ``kappa`` takes priority (warning
        emitted).
    R : float or None
        Spherical truncation radius [Å].  Required for ``"rozzi_spherical"``
        unless ``L_cell`` is provided.
    L_cell : float or None
        Simulation cell length [Å].  Convenience parameter: derives
        ``kappa = 2π/L_cell`` or ``R = L_cell/2`` when the explicit value is
        not supplied.

    Returns
    -------
    np.ndarray
        Array with same shape as ``q2`` containing ΔZ · K(q²).

    References
    ----------
    L.-M. Peng (1998). Acta Cryst. A54, 481–485.
    C. A. Rozzi, D. Varsano, A. Marini, E. K. U. Gross & A. Rubio (2006).
        Phys. Rev. B 73, 205119.
    """
    q2 = np.asarray(q2, dtype=float)

    if delta_Z == 0:
        return np.zeros_like(q2)

    if regularization not in VALID_REGULARIZATIONS:
        raise ValueError(
            f"Unknown regularization '{regularization}'. "
            f"Choose from: {sorted(VALID_REGULARIZATIONS)}."
        )

    if regularization == "yukawa":
        if kappa is not None and L_cell is not None:
            warnings.warn(
                "Both kappa and L_cell supplied for regularization='yukawa'; "
                "using kappa and ignoring L_cell.",
                UserWarning,
                stacklevel=2,
            )
        if kappa is None:
            if L_cell is not None:
                kappa = 2.0 * np.pi / L_cell
            else:
                raise ValueError(
                    "regularization='yukawa' requires kappa or L_cell."
                )

    elif regularization in ("rozzi_spherical", "rozzi_cylindrical", "rozzi_slab"):
        if R is None:
            if L_cell is not None:
                R = L_cell / 2.0
            else:
                raise ValueError(
                    f"regularization='{regularization}' requires R or L_cell."
                )

    if regularization in ("rozzi_cylindrical", "rozzi_slab"):
        raise NotImplementedError(
            f"regularization='{regularization}' is not yet implemented. "
            "It requires slab/wire geometry support (non-cubic periodic cell). "
            "See Rozzi et al. (2006) Phys. Rev. B 73, 205119, Eqs. (14)–(16)."
        )

    if regularization == "none":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(q2 > 0.0, delta_Z / q2, np.inf)

    elif regularization == "yukawa":
        assert kappa is not None  # guaranteed by validation above
        return delta_Z / (q2 + kappa**2)

    else:  # rozzi_spherical
        assert R is not None  # guaranteed by validation above
        # Use a safe q² to avoid 0/0 in the large-q branch; override those
        # elements with the analytically computed small-q limit R²/2.
        _Q2_THRESHOLD = 1e-16  # [1/Å²]  (q ~ 1e-8 Å⁻¹)
        small_q = q2 < _Q2_THRESHOLD
        safe_q2 = np.where(small_q, 1.0, q2)
        large_q_val = delta_Z * (1.0 - np.cos(np.sqrt(safe_q2) * R)) / safe_q2
        small_q_val = delta_Z * R**2 / 2.0
        return np.where(small_q, small_q_val, large_q_val)


def ionic_scattering_factor_k2(
    k2,
    p,
    mott_coeff,
    delta_Z,
    regularization="none",
    kappa=None,
    R=None,
    L_cell=None,
):
    """Peng 1999 ionic projected scattering factor: 5-Gaussian sum + regularized ΔZ correction.

    Evaluates

        f(k²) = Σᵢ aᵢ exp(−bᵢ k²)  +  mott_coeff · ΔZ · K(k²)

    where K is the regularized Coulomb kernel selected by ``regularization``
    (see :func:`ionic_coulomb_term`).  The Mott–Bethe prefactor and any
    unit-conversion factors are absorbed into ``mott_coeff``; ``delta_Z`` is
    the bare net ionic charge.

    At k² = 0 the Coulomb correction is always set to zero for the imaging
    pipeline (the DC component does not affect contrast); use
    :func:`ionic_coulomb_term` directly if you need the q = 0 value.

    Parameters
    ----------
    k2 : array-like
        Squared scattering wavevector [1/Å²] in abTEM's convention (k = 2s).
    p : np.ndarray, shape (2, 5)
        Scaled Peng parameters [a_i, b_i] as returned by
        ``PengParametrization.scaled_parameters``.
    mott_coeff : float
        Product of the Mott–Bethe constant and all unit-conversion factors,
        *excluding* delta_Z.  For the projected scattering factor this is
        ``4 · M / kappa_abtem``; for the raw scattering factor ``4 · M``.
    delta_Z : float
        Net ionic charge (ΔZ = Z − Z₀).
    regularization : str
        Regularization scheme; see :func:`ionic_coulomb_term`.
    kappa : float or None
        Yukawa screening wavevector [1/Å].
    R : float or None
        Spherical truncation radius [Å].
    L_cell : float or None
        Simulation cell length [Å] used to derive kappa or R.

    References
    ----------
    L.-M. Peng (1998). Acta Cryst. A54, 481–485.
    C. A. Rozzi, D. Varsano, A. Marini, E. K. U. Gross & A. Rubio (2006).
        Phys. Rev. B 73, 205119.
    """
    coulomb = mott_coeff * ionic_coulomb_term(
        k2, delta_Z, regularization, kappa, R, L_cell
    )
    # Replace inf/nan at k²=0 (DC component) with 0 — does not affect contrast.
    coulomb = np.where(np.isfinite(coulomb), coulomb, 0.0)
    return scattering_factor_k2(k2, p) + coulomb


def finite_projected_scattering_factor(r, p, a, b):
    p = np.expand_dims(p, tuple(range(2, 2 + len(r.shape))))
    return (
        np.abs(erf(p[2] * b) - erf(p[2] * a))
        * p[0]
        * np.exp(-p[1] * r[None, ...] ** 2.0)
    ).sum(0) / 2
