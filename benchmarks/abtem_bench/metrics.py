"""Array comparison for the benchmark harness.

``close_stats`` compares a candidate array with a reference and returns the
quantities a report shows (bit-identical flag, largest absolute difference
over the reference maximum, relative error above a floor, change of the
integrated intensity, shape and dtype).
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
        from the relative error.

    Returns
    -------
    dict
        ``identical`` (bit-for-bit equal, NaN-aware), ``max_abs_norm``
        (``max|c - r| / max|r|``), ``rel_above`` (maximum relative error over
        the checked elements, ``nan`` if none is checked), ``intensity``
        (relative change of the integrated intensity), ``n_checked`` (number of
        elements in the relative comparison), ``nonfinite_mismatch`` (number
        of elements that are NaN or infinite and not the same in both arrays)
        and ``shape_ok``.

    Only elements finite in both arrays enter ``max_abs_norm``, ``rel_above``
    and ``intensity``; an element that is NaN in both, or the same infinity in
    both, is left out. Any other non-finite element makes ``max_abs_norm`` and
    ``rel_above`` infinite.
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
            nonfinite_mismatch=0,
        )
        return stats

    stats["identical"] = bool(np.array_equal(c, r, equal_nan=True)) and (
        c.dtype == r.dtype
    )

    valid = np.isfinite(c) & np.isfinite(r)
    same = (c == r) | (np.isnan(c) & np.isnan(r))
    stats["nonfinite_mismatch"] = int((~valid & ~same).sum())
    c, r = c[valid], r[valid]

    diff = np.abs(c.astype(np.complex128) - r.astype(np.complex128))
    ref_abs = np.abs(r.astype(np.complex128))
    ref_max = float(ref_abs.max()) if ref_abs.size else 0.0
    diff_max = float(diff.max()) if diff.size else 0.0
    stats["max_abs_norm"] = (
        diff_max / ref_max if ref_max > 0 else (0.0 if diff_max == 0.0 else np.inf)
    )

    checked = ref_abs > above_rel * ref_max
    stats["n_checked"] = int(checked.sum())
    if stats["n_checked"]:
        stats["rel_above"] = float((diff[checked] / ref_abs[checked]).max())
    else:
        stats["rel_above"] = np.nan
    if stats["nonfinite_mismatch"]:
        stats["max_abs_norm"] = stats["rel_above"] = np.inf

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
