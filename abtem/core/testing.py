"""Array comparison helpers shared by the test suite and the benchmark harness.

``array_is_close`` is the boolean check the test suite has always used; it
compares relative error only on elements that carry real intensity (above
``check_above_abs`` and above ``check_above_rel`` times the reference maximum),
so a pattern spanning many decades is not judged on its noise floor. Both
tolerances default to infinity, so a call without an explicit tolerance passes
unconditionally: callers must always pass ``rel_tol`` or ``abs_tol``.

``close_stats`` is the vector form of the same comparison for reports: it
returns the individual quantities instead of a single verdict.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def array_is_close(
    a1,
    a2,
    rel_tol=np.inf,
    abs_tol=np.inf,
    check_above_abs=0.0,
    check_above_rel=0.0,
    mask=None,
):
    if mask is not None:
        a1 = a1[mask]
        a2 = a2[mask]

    if rel_tol < np.inf:
        element_is_checked = (a2 > check_above_abs) * (
            a2 > (a2.max() * check_above_rel)
        )
        rel_error = (a1[element_is_checked] - a2[element_is_checked]) / a2[
            element_is_checked
        ]
        if np.any(np.abs(rel_error) > rel_tol):
            return False

    if abs_tol < np.inf:
        if np.any(np.abs(a1 - a2) > abs_tol):
            return False

    return True


def close_stats(
    candidate: np.ndarray, reference: np.ndarray, above_rel: float = 1e-6
) -> dict[str, Any]:
    """Compare ``candidate`` against ``reference`` and return the metric vector.

    Parameters
    ----------
    candidate, reference : np.ndarray
        Arrays of the same shape. Complex arrays are compared on their real and
        imaginary parts jointly (the absolute difference is ``|c - r|``) and the
        integrated intensity is ``sum |x|^2`` instead of ``sum x``.
    above_rel : float
        Elements with ``|reference| <= above_rel * max|reference|`` are excluded
        from the relative error, following ``array_is_close``'s
        ``check_above_rel`` semantics.

    Returns
    -------
    dict
        ``identical`` (bit-for-bit equal, NaN-aware), ``max_abs_norm``
        (``max|c - r| / max|r|``), ``rel_above`` (maximum relative error over
        the checked elements, ``nan`` if none is checked), ``intensity``
        (relative change of the integrated intensity), ``n_checked`` (number of
        elements in the relative comparison) and ``shape_ok``.
    """
    c = np.asarray(candidate)
    r = np.asarray(reference)
    stats: dict[str, Any] = {"shape_ok": c.shape == r.shape and c.dtype == r.dtype}
    if c.shape != r.shape:
        stats.update(
            identical=False,
            max_abs_norm=np.nan,
            rel_above=np.nan,
            intensity=np.nan,
            n_checked=0,
        )
        return stats

    stats["identical"] = bool(np.array_equal(c, r, equal_nan=True)) and (
        c.dtype == r.dtype
    )

    diff = np.abs(c.astype(np.complex128) - r.astype(np.complex128))
    ref_abs = np.abs(r.astype(np.complex128))
    ref_max = float(ref_abs.max()) if ref_abs.size else 0.0
    stats["max_abs_norm"] = (
        float(diff.max() / ref_max)
        if ref_max > 0
        else (0.0 if diff.size == 0 or float(diff.max()) == 0.0 else np.inf)
    )

    checked = ref_abs > above_rel * ref_max
    stats["n_checked"] = int(checked.sum())
    if stats["n_checked"]:
        stats["rel_above"] = float((diff[checked] / ref_abs[checked]).max())
    else:
        stats["rel_above"] = np.nan

    if np.iscomplexobj(c) or np.iscomplexobj(r):
        i_c = float((np.abs(c) ** 2).sum())
        i_r = float((np.abs(r) ** 2).sum())
    else:
        i_c = float(c.astype(np.float64).sum())
        i_r = float(r.astype(np.float64).sum())
    stats["intensity"] = (
        (i_c - i_r) / i_r if i_r != 0 else (0.0 if i_c == 0 else np.inf)
    )
    return stats
