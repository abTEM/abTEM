"""Bundle store: the commit-independent result format.

    <bundle>/
      manifest.json
      logs/<case-id>.log
      cases/<case-id>.json
      cases/<case-id>/<output>.npz

The format depends on numpy and json only, so compare works on bundles
produced by any abtem version. Schema changes are additive and versioned.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

STATUS_OK = "OK"
STATUS_UNSUPPORTED = "UNSUPPORTED"
STATUS_ERROR = "ERROR"
STATUS_OOM = "OOM"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_SKIPPED_MEMORY = "SKIPPED-MEMORY"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _version(module: str) -> str | None:
    try:
        return __import__(module).__version__
    except Exception:  # noqa: BLE001 -- optional package absent
        return None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _gpu_name() -> str | None:
    try:
        import cupy as cp  # type: ignore

        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # noqa: BLE001
        return None


def fingerprint() -> dict[str, Any]:
    """Machine and stack identity recorded in every manifest."""
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "gpu": _gpu_name(),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "dask": _version("dask"),
        "pyfftw": _version("pyfftw"),
        "numba": _version("numba"),
        "cupy": _version("cupy"),
        "gpaw": _version("gpaw"),
    }


def fingerprint_short(fp: dict[str, Any]) -> str:
    """Eight hex digits over the parts that decide bit-identity across machines."""
    import hashlib

    keys = ("cpu", "gpu", "python", "numpy", "pyfftw", "numba", "cupy")
    h = hashlib.sha256("|".join(str(fp.get(k)) for k in keys).encode())
    return h.hexdigest()[:8]


def mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def load_average() -> float | None:
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def invariants(array: np.ndarray) -> dict[str, Any]:
    a = np.asarray(array)
    if a.size == 0:
        return {"size": 0}
    if np.iscomplexobj(a):
        return {
            "size": int(a.size),
            "sum_real": float(a.real.sum()),
            "sum_imag": float(a.imag.sum()),
            "abs_max": float(np.abs(a).max()),
            "intensity": float((np.abs(a) ** 2).sum()),
        }
    a64 = a.astype(np.float64)
    return {
        "size": int(a.size),
        "sum": float(a64.sum()),
        "mean": float(a64.mean()),
        "min": float(a64.min()),
        "max": float(a64.max()),
        "abs_max": float(np.abs(a64).max()),
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


class Bundle:
    """One capture: a directory of case records and output arrays."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # -- layout -------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def case_json(self, case_id: str) -> Path:
        return self.path / "cases" / f"{_fname(case_id)}.json"

    def case_dir(self, case_id: str) -> Path:
        return self.path / "cases" / _fname(case_id)

    def log_path(self, case_id: str) -> Path:
        return self.path / "logs" / f"{_fname(case_id)}.log"

    # -- manifest -----------------------------------------------------------
    def write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest = dict(manifest)
        manifest.setdefault("schema_version", SCHEMA_VERSION)
        dump_json(self.manifest_path, manifest)

    def read_manifest(self) -> dict[str, Any]:
        return load_json(self.manifest_path)

    def exists(self) -> bool:
        return self.manifest_path.exists()

    # -- cases --------------------------------------------------------------
    def write_case(
        self,
        case_id: str,
        record: dict[str, Any],
        outputs: dict[str, tuple[np.ndarray, list[dict], dict]] | None = None,
    ) -> None:
        record = dict(record)
        record["case_id"] = case_id
        index = {}
        if outputs:
            d = self.case_dir(case_id)
            d.mkdir(parents=True, exist_ok=True)
            for name, (array, axes, metadata) in outputs.items():
                arr = np.ascontiguousarray(array)
                np.savez(
                    d / f"{name}.npz",
                    array=arr,
                    axes=np.array(json.dumps(_json_safe(axes))),
                    metadata=np.array(json.dumps(_json_safe(metadata))),
                )
                index[name] = {
                    "file": f"{_fname(case_id)}/{name}.npz",
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "invariants": invariants(arr),
                }
        record["outputs"] = index
        dump_json(self.case_json(case_id), record)

    def read_case(self, case_id: str) -> dict[str, Any]:
        return load_json(self.case_json(case_id))

    def has_case(self, case_id: str) -> bool:
        return self.case_json(case_id).exists()

    def load_output(
        self, case_id: str, name: str
    ) -> tuple[np.ndarray, list[dict], dict]:
        with np.load(self.case_dir(case_id) / f"{name}.npz", allow_pickle=False) as z:
            return (
                z["array"],
                json.loads(str(z["axes"])),
                json.loads(str(z["metadata"])),
            )

    def case_ids(self) -> list[str]:
        d = self.path / "cases"
        if not d.exists():
            return []
        ids = []
        for p in sorted(d.glob("*.json")):
            ids.append(load_json(p)["case_id"])
        return ids

    # -- packing ------------------------------------------------------------
    def pack(self, target: str | Path) -> Path:
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(target, "w:xz") as tar:
            tar.add(self.path, arcname=self.path.name)
        return target

    @classmethod
    def unpack(cls, archive: str | Path, into: str | Path) -> "Bundle":
        into = Path(into)
        into.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:xz") as tar:
            names = tar.getnames()
            root = names[0].split("/")[0]
            tar.extractall(into, filter="data")
        return cls(into / root)


def _fname(case_id: str) -> str:
    return case_id.replace("/", "__")
