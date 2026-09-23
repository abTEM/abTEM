"""Determinism presets.

A preset pins every abTEM config key, dask setting and environment variable
that changes results or timing, so two captures differ only in the abtem
package under test. The worker applies the abTEM and dask parts after import;
the runner applies the environment part when it spawns the worker.

``accuracy``: float64, FFTW_ESTIMATE, one thread, synchronous scheduler. The
expectation on CPU is bit-identical output for the same case on the same
machine, across processes and days.

``speed``: shipped FFTW settings (FFTW_MEASURE) so the cold cost users pay is
represented, but fixed thread and worker counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ENV_COMMON: dict[str, str] = {
    "TQDM_DISABLE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


@dataclass(frozen=True)
class Preset:
    name: str
    precision: str
    abtem_config: dict[str, Any]
    dask_config: dict[str, Any]
    env: dict[str, str] = field(default_factory=lambda: dict(ENV_COMMON))
    repeats: int = 5


_ABTEM_COMMON: dict[str, Any] = {
    "fft": "fftw",
    "fftw.threads": 1,
    "mkl.threads": 1,
    "dask.chunk-size": "128 MB",
    "dask.chunk-size-gpu": "512 MB",
    "grid.round-to-fast-fft": False,
    "cupy.fft-cache-size": "1 GB",
    "cupy.fft-cache-entries": 64,
    "diagnostics.progress_bar": False,
    "diagnostics.task_progress": False,
    "warnings.overspecified-grid": False,
    "warnings.dask-blockwise-performance": False,
    "dask.multi-gpu": False,
}

PRESETS: dict[str, Preset] = {
    "accuracy": Preset(
        name="accuracy",
        precision="float64",
        abtem_config={
            **_ABTEM_COMMON,
            "precision": "float64",
            "fftw.planning_effort": "FFTW_ESTIMATE",
        },
        dask_config={"scheduler": "synchronous", "num_workers": 1},
        repeats=3,
    ),
    "speed": Preset(
        name="speed",
        precision="float32",
        abtem_config={
            **_ABTEM_COMMON,
            "precision": "float32",
            "fftw.planning_effort": "FFTW_MEASURE",
        },
        dask_config={"scheduler": "synchronous", "num_workers": 1},
        repeats=5,
    ),
}


def resolved_abtem_config(preset: Preset, params: Any, device: str) -> dict[str, Any]:
    """The abTEM config the worker sets, including per-case values."""
    cfg = dict(preset.abtem_config)
    cfg["device"] = device
    cfg["dask.lazy"] = bool(getattr(params, "lazy", True))
    chunk = getattr(params, "chunk", None)
    if chunk is not None:
        # Unknown on refs before potential chunking existed; abtem.config.set
        # accepts unknown keys silently, so pinning is harmless there.
        cfg["potential.slice-chunk-size"] = chunk
    return cfg


def apply(preset: Preset, params: Any, device: str) -> dict[str, Any]:
    """Apply the preset inside the worker process. Returns what was set."""
    import dask

    import abtem

    cfg = resolved_abtem_config(preset, params, device)
    abtem.config.set(cfg)  # non-context use: applies for the rest of the process
    dask.config.set(preset.dask_config)
    return {"abtem": cfg, "dask": dict(preset.dask_config)}
