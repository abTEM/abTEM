"""Structures and helpers shared by cases.

Everything here must work on every abtem ref the suite is run against, so it
imports abtem lazily and probes for optional keyword arguments instead of
assuming them.
"""

from __future__ import annotations

import inspect
from typing import Any


def silicon(reps: tuple[int, int, int]):
    from ase.build import bulk

    return bulk("Si", cubic=True) * tuple(reps)


def srtio3(reps: tuple[int, int, int]):
    from ase.spacegroup import crystal

    a = 3.905
    unit = crystal(
        ["Sr", "Ti", "O"],
        [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[a, a, a, 90, 90, 90],
    )
    return unit * tuple(reps)


def supports_kwarg(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def multislice_kwargs(params: Any) -> dict[str, Any]:
    """Keyword arguments forwarded through ``**multislice_func_kwargs``.

    ``algorithm`` is passed only for a variant that fixes the propagator order,
    so the default variant uses each ref's own default (that difference is the
    point of the v1.1 comparison). ``potential_chunk_size`` is passed only when
    the ref accepts it.
    """
    from abtem.multislice import multislice_and_detect

    kwargs: dict[str, Any] = {}
    order = getattr(params, "algorithm_order", None)
    if order is not None:
        from abtem.multislice import FourierMultislice

        kwargs["algorithm"] = FourierMultislice(order=order)
    chunk = getattr(params, "chunk", None)
    if chunk is not None and supports_kwarg(
        multislice_and_detect, "potential_chunk_size"
    ):
        kwargs["potential_chunk_size"] = chunk
    return kwargs


def detector_angles(probe) -> float:
    """The largest angle every detector can use on this probe's grid, in mrad."""
    return 0.95 * float(min(probe.cutoff_angles))
