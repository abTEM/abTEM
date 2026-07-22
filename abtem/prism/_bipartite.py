"""Geometry and reconstruction helpers for BiP-PRISM core-loss EELS.

BiP-PRISM (paper Alg. 5, "bi-partitioned PRISM") accelerates dual-scattering-matrix
PRISM-EELS by building *both* the probe-forming matrix ``S1`` and the detector/exit
matrix ``S2`` on a sparse set of **parent beams** (a hex-ring subsample of the beam
set), then reconstructing the full beam set *locally at each ionized atom* by
**natural-neighbor interpolation** of the de-tilted parent columns, with an optional
**magnitude-preserving** correction (paper eq-magpreserve) that counters the
amplitude loss the convex complex average would otherwise introduce.

This module holds the geometry-only, backend-agnostic pieces:

- :func:`select_parent_beams` — hex-ring parent selection (indices into the beam set).
- :func:`natural_neighbor_weights` — partition-of-unity interpolation weights
  ``(n_query, n_parent)`` (vectorised Delaunay-barycentric by default; Sibson optional).
- :func:`windowed_reconstruct` — per-atom windowed reconstruction (de-tilt → NNW
  combine → magnitude-preserve → re-tilt), evaluated only on a crop window.

These mirror the scatterem reference (``_select_parent_beams``,
``natural_neighbor_weights``, ``PartitionedScatteringMatrix._nnw_window``) but are
expressed in abTEM's ``plane_waves`` phase convention:
``column_b(r) = exp(+2πi k_b · r)`` with ``r`` in Å.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.utils import get_dtype


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "get"):  # cupy
        array = array.get()
    return np.asarray(array)


def select_parent_beams(
    wave_vectors, n_radial: int, n_angular: int = 6
) -> np.ndarray:
    """Hex-ring parent-beam selection (port of scatterem ``_select_parent_beams``).

    Samples the beam set on the DC beam plus ``n_radial`` concentric hex rings
    (ring ``i`` carries ``n_angular * (1 + i)`` angular samples, rotated by
    ``i·π/n_angular``), snaps each sample to the nearest actual beam, and
    de-duplicates. Because parents are *indices into* ``wave_vectors``, the
    full-parent limit (parents = all beams) is exact.

    Parameters
    ----------
    wave_vectors : (N, 2) array_like
        Beam coordinates (any consistent units; only their geometry matters).
    n_radial : int
        Number of concentric parent rings (radial resolution). ``>= 1``.
    n_angular : int, optional
        Base azimuthal sampling; ring ``i`` gets ``n_angular * (1 + i)`` samples.

    Returns
    -------
    parent_indices : (Bp,) int ndarray
        Sorted, unique indices into ``wave_vectors``.
    """
    wv = _to_numpy(wave_vectors).astype(np.float64)
    if n_radial < 1:
        raise ValueError("n_radial must be >= 1")

    radius = float(np.linalg.norm(wv, axis=1).max())
    a_off = np.pi / n_angular
    samples = [[0.0, 0.0]]
    for i, r in enumerate(np.linspace(0.0, radius, n_radial + 1)[1:]):
        n_ang = n_angular * (1 + i)
        for a in np.linspace(-np.pi, np.pi, n_ang, endpoint=False):
            samples.append([r * np.sin(a + a_off * i), r * np.cos(a + a_off * i)])
    samples = np.asarray(samples, dtype=np.float64)

    d = np.linalg.norm(samples[:, None, :] - wv[None, :, :], axis=2)
    return np.unique(np.argmin(d, axis=1))


def _linear_barycentric_weights(
    parents: np.ndarray, queries: np.ndarray, minimum_weight_cutoff: float = 1e-2
) -> np.ndarray:
    """Vectorised piecewise-linear (Delaunay barycentric) weights ``(B, Bp)``.

    Each query's weights are the barycentric coordinates of the Delaunay triangle
    it falls in (3 parents); queries outside the convex hull fall back to the
    nearest parent (weight 1). Partition of unity, exact at parents. Port of
    scatterem ``_linear_barycentric_weights``.
    """
    known = np.asarray(parents, dtype=float)
    interp = np.asarray(queries, dtype=float)
    B, Bs = interp.shape[0], known.shape[0]
    weights = np.zeros((B, Bs))

    def _nearest_all():
        d = np.linalg.norm(known[None, :, :] - interp[:, None, :], axis=2)
        weights[np.arange(B), d.argmin(axis=1)] = 1.0
        return weights

    if Bs < 3:
        return _nearest_all()
    try:
        tri = Delaunay(known)
    except Exception:
        return _nearest_all()

    simplex = tri.find_simplex(interp)  # (B,), -1 outside the convex hull
    inside = simplex >= 0
    if inside.any():
        sx = simplex[inside]
        transform = tri.transform[sx]  # (Bin, 3, 2)
        r = interp[inside] - transform[:, 2, :]
        bary2 = np.einsum("bij,bj->bi", transform[:, :2, :], r)
        bary = np.concatenate([bary2, 1.0 - bary2.sum(1, keepdims=True)], axis=1)
        verts = tri.simplices[sx]  # (Bin, 3)
        rows = np.repeat(np.nonzero(inside)[0], 3)
        weights[rows, verts.reshape(-1)] = bary.reshape(-1)
    out = np.nonzero(~inside)[0]
    if out.size:
        d = np.linalg.norm(known[None, :, :] - interp[out][:, None, :], axis=2)
        weights[out, d.argmin(axis=1)] = 1.0

    weights[weights < minimum_weight_cutoff] = 0.0
    row_sum = weights.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return weights / row_sum


def natural_neighbor_weights(
    parents,
    queries,
    method: str = "linear",
    minimum_weight_cutoff: float = 1e-2,
) -> np.ndarray:
    """Interpolation weights expressing each query beam as a convex blend of parents.

    Parameters
    ----------
    parents : (Bp, 2) array_like
        Parent-beam coordinates.
    queries : (B, 2) array_like
        Query (target) beam coordinates.
    method : {"linear"}
        Only ``"linear"`` is supported: vectorised Delaunay-barycentric weights
        (C0, fully vectorised, nearest-fallback outside the convex hull) — this is
        the scatterem reference default and is "adequate for the smooth
        probe-reconstruction use in partitioned PRISM" (paper). abTEM's Sibson
        weights (:func:`abtem.prism._natural_neighbors.pairwise_weights`) are C1
        but raise ``QhullError`` on degenerate circumcenter polygons, so they are
        not exposed here.

    Returns
    -------
    weights : (B, Bp) ndarray
        ``weights[b, p]`` = weight of parent ``p`` in reconstructing query ``b``.
        Rows are a partition of unity; a query coinciding with a parent is one-hot.
    """
    parents = _to_numpy(parents).astype(np.float64)
    queries = _to_numpy(queries).astype(np.float64)

    if method == "linear":
        return _linear_barycentric_weights(parents, queries, minimum_weight_cutoff)
    raise ValueError(f"unknown method {method!r} (only 'linear' is supported)")


def focal_backprop_distance(traversed_depth, focal_backprop):
    """Back-propagation distance [Å] before NNW interpolation (paper Sec. 3.3.6).

    ``focal_backprop`` = ``"centroid"`` -> half the traversed depth (the scattering
    centroid of the entrance-to-plane path, where the de-tilted parent envelopes
    are most coherent so the convex NNW average decoheres least); a float ``f`` ->
    ``f * traversed_depth``; ``None``/``0`` -> ``0.0`` (no back-propagation).
    """
    if not focal_backprop:
        return 0.0
    if isinstance(focal_backprop, str):
        if focal_backprop != "centroid":
            raise ValueError(
                f"focal_backprop str must be 'centroid', got {focal_backprop!r}"
            )
        return 0.5 * float(traversed_depth)
    return float(focal_backprop) * float(traversed_depth)


def pad_window(corner, length, margin, n):
    """Pad a length-``length`` window starting at ``corner`` by ``margin`` on each
    side (capped to the full axis ``n``).

    Returns ``(padded_indices, inner_slice)`` where ``padded_indices`` are the
    un-wrapped pixel indices of the widened window and ``inner_slice`` selects the
    original window within it: ``padded_indices[inner_slice] == corner + arange(length)``.
    The margin gives the windowed forward-Fresnel enough room that its periodic
    wrap-around stays outside the inner window (paper: ``_fb_margin``).
    """
    length = int(length)
    lp = min(length + 2 * int(margin), int(n))
    left = (lp - length) // 2
    padded = (int(corner) - left) + np.arange(lp)
    return padded, slice(left, left + length)


def fresnel_margin(distance, wavelength, k_max, sampling, gpts):
    """Window padding [px] covering the Fresnel lateral spread of a propagation by
    ``distance`` [Å] for transverse frequencies up to ``k_max`` [1/Å].

    ``spread ~ distance * lambda * k_max`` in Å -> ``/dx`` in pixels; padded by 1.5x
    plus a small constant, capped at half the grid (paper: ``_fb_margin``).
    """
    if not distance:
        return 0
    dx = min(sampling)
    spread_px = abs(distance) * wavelength * k_max / dx
    return int(min(max(4, np.ceil(1.5 * spread_px) + 2), min(gpts) // 2))


def _window_tilt(k, iy, ix, extent, gpts, sign, xp, cdtype):
    """``exp(sign · 2πi k · r)`` on the window pixels ``(iy, ix)`` → ``(len(k), wy, wx)``.

    Matches :func:`abtem.prism.utils.plane_waves`: ``r = (iy·extent0/gpts0,
    ix·extent1/gpts1)``. The window indices may be un-wrapped (negative or
    ``>= gpts``); the phase is exactly ``gpts``-periodic because ``k · extent`` is
    integer for the PRISM beam grid, so this is consistent with the wrapped crop.
    """
    real_dtype = get_dtype(complex=False)
    dy = np.float32(extent[0] / gpts[0])
    dx = np.float32(extent[1] / gpts[1])
    x = xp.asarray(iy, dtype=real_dtype) * dy  # (wy,)
    y = xp.asarray(ix, dtype=real_dtype) * dx  # (wx,)
    k = xp.asarray(k, dtype=real_dtype)  # (M, 2)
    phase = (sign * 2.0 * np.float32(np.pi)) * (
        k[:, 0, None, None] * x[None, :, None] + k[:, 1, None, None] * y[None, None, :]
    )
    return complex_exponential(phase).astype(cdtype)


def windowed_reconstruct(
    parent_cols,
    weights,
    k_parents,
    k_targets,
    iy,
    ix,
    extent,
    gpts,
    mag_preserve: bool = True,
):
    """Reconstruct the target beam columns on a crop window from the parent columns.

    Implements paper eq-window-recon (+ eq-magpreserve): de-tilt each parent column
    by its own carrier ``exp(-2πi k_p · r)``, natural-neighbor combine to the target
    beams, optionally restore the interpolated magnitude, and re-tilt each target by
    its carrier ``exp(+2πi k_b · r)`` — all evaluated only on the window.

    Parameters
    ----------
    parent_cols : (Bp, wy, wx) array
        Parent scattering-matrix columns cropped to the window (backend-native).
    weights : (B, Bp) array_like
        Natural-neighbor weights (from :func:`natural_neighbor_weights`).
    k_parents : (Bp, 2) array_like
        Parent-beam wave vectors (1/Å).
    k_targets : (B, 2) array_like
        Target-beam wave vectors (1/Å).
    iy, ix : 1-D int array_like
        Window pixel rows / columns (un-wrapped indices into the full grid).
    extent, gpts : pair
        Full-grid extent (Å) and gpts, for the phase-ramp real coordinates.
    mag_preserve : bool, optional
        Apply the magnitude-preserving correction (default ``True``).

    Returns
    -------
    (B, wy, wx) array
        Reconstructed target columns on the window, same backend/dtype as
        ``parent_cols``.
    """
    xp = get_array_module(parent_cols)
    cdtype = parent_cols.dtype
    w = xp.asarray(_to_numpy(weights), dtype=cdtype)  # (B, Bp)

    detilt = _window_tilt(k_parents, iy, ix, extent, gpts, -1.0, xp, cdtype)
    Sd = parent_cols * detilt  # (Bp, wy, wx)

    recon = xp.einsum("bp,pwv->bwv", w, Sd)  # (B, wy, wx)
    if mag_preserve:
        mag = xp.einsum("bp,pwv->bwv", w.real.astype(cdtype), xp.abs(Sd).astype(cdtype))
        mag = mag.real
        denom = xp.abs(recon)
        denom = xp.where(denom > 1e-20, denom, xp.asarray(1e-20, dtype=denom.dtype))
        recon = mag * (recon / denom)

    tilt = _window_tilt(k_targets, iy, ix, extent, gpts, +1.0, xp, cdtype)
    return recon * tilt
