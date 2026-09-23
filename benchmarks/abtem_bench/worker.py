"""Run exactly one case id in this process and write its record.

Invoked by the runner as ``python -P -m abtem_bench.worker ...`` with the
harness directory and the ref's worktree on PYTHONPATH. The first lines it
prints identify the abtem actually imported; it aborts if that is not the
expected worktree, so a mislabelled bundle cannot be produced silently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from abtem_bench import meters, presets, registry, store


def _identify(expected_root: str | None) -> dict[str, Any]:
    import abtem

    file = os.path.realpath(abtem.__file__)
    info = {"abtem_version": abtem.__version__, "abtem_file": file}
    print(f"abtem  <- {file}", flush=True)
    print(f"version <- {abtem.__version__}", flush=True)
    if expected_root:
        root = os.path.realpath(expected_root)
        if not file.startswith(root + os.sep):
            raise SystemExit(
                f"abtem imported from {file}, expected a module under {root}; "
                "refusing to record a mislabelled bundle"
            )
    return info


def _last_line(text: str) -> str:
    """The last non-empty line of a traceback: the exception itself."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1][:300] if lines else ""


def _materialize(result: Any) -> Any:
    """Compute lazy results so outputs are concrete arrays."""
    if isinstance(result, dict):
        return {k: _materialize(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        computed = result.compute() if hasattr(result, "compute") else result
        return [_materialize(v) for v in computed]
    if getattr(result, "is_lazy", False):
        return result.compute()
    return result


def _extract(obj: Any) -> tuple[np.ndarray, list[dict], dict]:
    """Host array, serialised axes and JSON-safe metadata of one output."""
    if isinstance(obj, np.ndarray):
        return obj, [], {}
    if hasattr(obj, "to_cpu"):
        obj = obj.to_cpu()
    array = np.asarray(obj.array)
    axes: list[dict] = []
    try:
        from abtem.core.axes import axis_to_dict

        axes = [axis_to_dict(a) for a in obj.axes_metadata]
    except Exception:  # noqa: BLE001 -- keep the array even if axes fail
        axes = [
            {"type": type(a).__name__, "repr": repr(a)}
            for a in getattr(obj, "axes_metadata", [])
        ]
    metadata = {}
    for k, v in dict(getattr(obj, "metadata", {}) or {}).items():
        if k == "data_origin":  # embeds the abtem version string
            continue
        try:
            json.dumps(v)
            metadata[k] = v
        except TypeError:
            metadata[k] = str(v)
    metadata["type"] = type(obj).__name__
    return array, axes, metadata


def _name_outputs(case: registry.Case, result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    if isinstance(result, (list, tuple)):
        names = case.outputs or tuple(f"output{i}" for i in range(len(result)))
        if len(names) != len(result):
            raise RuntimeError(
                f"case {case.name} returned {len(result)} outputs, "
                f"declared {len(names)}"
            )
        return dict(zip(names, result))
    name = case.outputs[0] if case.outputs else "output"
    return {name: result}


def run(args: argparse.Namespace) -> int:
    bundle = store.Bundle(args.out)
    cid = registry.CaseId.parse(args.case_id)
    t_start = time.perf_counter()
    record: dict[str, Any] = {
        "ref": args.ref_label,
        "sha": args.ref_sha,
        "preset": args.preset,
        "timestamp_utc": store.utc_now(),
        "status": store.STATUS_ERROR,
    }
    try:
        record.update(_identify(args.expected_root))
        import abtem

        reg = registry.load_cases()
        case = reg[cid.name]
        params = case.params(cid.tier, cid.variant, cid.device)
        record["params"] = vars(params)
        if cid.device not in case.tiers[cid.tier].devices:
            record["status"] = store.STATUS_UNSUPPORTED
            record["error"] = f"tier {cid.tier} does not run on {cid.device}"
            bundle.write_case(args.case_id, record)
            return 0
        if case.requires is not None and not case.requires(abtem):
            record["status"] = store.STATUS_UNSUPPORTED
            record["error"] = "requires() is false on this ref"
            bundle.write_case(args.case_id, record)
            return 0

        preset = presets.PRESETS[args.preset]
        record["config"] = presets.apply(preset, params, cid.device)
        record["config"]["abtem_resolved"] = abtem.config.config
        import dask

        record["config"]["dask_resolved"] = dask.config.config

        t0 = time.perf_counter()
        run_fn = case.func(params, cid.device)
        setup = time.perf_counter() - t0

        timings: dict[str, Any] = {"setup": setup, "warm": [], "warm_cpu": []}
        result = None
        with meters.VRAMSampler() as vram:
            if case.warmup in ("cold_and_warm", "cold_only"):
                result, t = meters.timed(run_fn, cid.device)
                timings["cold"] = t.wall
                timings["cold_cpu"] = t.cpu
            n_warm = 0 if case.warmup == "cold_only" else args.repeats
            for _ in range(n_warm):
                meters.gpu_cleanup(cid.device)
                result, t = meters.timed(run_fn, cid.device)
                timings["warm"].append(t.wall)
                timings["warm_cpu"].append(t.cpu)
        if timings["warm"]:
            timings["median"] = float(np.median(timings["warm"]))
            timings["min"] = float(min(timings["warm"]))
        else:
            timings["median"] = timings.get("cold")
            timings["min"] = timings.get("cold")
        record["timings"] = timings
        record["memory"] = vram.as_dict()
        if case.nominal_bytes is not None:
            try:
                record["memory"]["nominal_bytes"] = int(case.nominal_bytes(params))
            except Exception:  # noqa: BLE001
                record["memory"]["nominal_bytes"] = None

        outputs = {
            n: _extract(o) for n, o in _name_outputs(case, _materialize(result)).items()
        }
        record["status"] = store.STATUS_OK
        record["wall_total"] = time.perf_counter() - t_start
        bundle.write_case(args.case_id, record, outputs)
        return 0
    except SystemExit:
        raise
    except MemoryError:
        record["status"] = store.STATUS_OOM
        record["error"] = traceback.format_exc()
        record["error_summary"] = _last_line(record["error"])
        bundle.write_case(args.case_id, record)
        return 3
    except Exception:  # noqa: BLE001 -- the record is the error report
        record["status"] = store.STATUS_ERROR
        record["error"] = traceback.format_exc()
        record["error_summary"] = _last_line(record["error"])
        bundle.write_case(args.case_id, record)
        print(record["error"], file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="abtem_bench.worker")
    p.add_argument("--case-id", required=True)
    p.add_argument("--ref-label", required=True)
    p.add_argument("--ref-sha", required=True)
    p.add_argument("--expected-root", default=None)
    p.add_argument("--preset", default="accuracy", choices=sorted(presets.PRESETS))
    p.add_argument("--out", required=True, help="bundle directory")
    p.add_argument("--repeats", type=int, default=None)
    args = p.parse_args(argv)
    if args.repeats is None:
        args.repeats = presets.PRESETS[args.preset].repeats
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
