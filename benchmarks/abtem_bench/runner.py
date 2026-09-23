"""Runner: worktrees, environments, one fresh subprocess per case id.

The harness and cases come from the invoking checkout; only the ``abtem``
package is swapped by putting the ref's worktree on PYTHONPATH after the
harness directory. ``python -P`` keeps the worker's cwd off ``sys.path`` so a
checkout cannot shadow the worktree.

Peak host memory is ``ru_maxrss`` from ``os.wait4`` on the worker: exact,
per child, and costing the measured process nothing.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abtem_bench import presets, registry, store

HARNESS_DIR = Path(__file__).resolve().parents[1]  # .../benchmarks
MEM_FLOOR_BYTES = 8 * 1024**3
TIER_TIMEOUTS = {"quick": 300.0, "standard": 1800.0, "large": None}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def find_repo(start: Path | None = None) -> Path:
    """The abTEM checkout that contains this harness."""
    start = start or HARNESS_DIR
    return Path(_git(start, "rev-parse", "--show-toplevel"))


@dataclass
class Ref:
    label: str
    sha: str
    worktree: Path
    describe: str


def resolve_ref(repo: Path, ref: str) -> tuple[str, str]:
    sha = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    try:
        describe = _git(repo, "describe", "--tags", "--always", sha)
    except subprocess.CalledProcessError:
        describe = sha[:12]
    return sha, describe


def ensure_worktree(repo: Path, sha: str) -> Path:
    """A detached worktree of ``sha`` under ``.worktrees/bench/<sha>``."""
    path = repo / ".worktrees" / "bench" / sha
    if (path / "abtem" / "__init__.py").exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():  # stale registration without files
        subprocess.run(["git", "-C", str(repo), "worktree", "prune"], check=False)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(path), sha],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


def prepare_ref(repo: Path, ref: str) -> Ref:
    sha, describe = resolve_ref(repo, ref)
    return Ref(
        label=ref, sha=sha, worktree=ensure_worktree(repo, sha), describe=describe
    )


_OOM_PATTERNS = (
    "CUDA_ERROR_OUT_OF_MEMORY",
    "CUDA_ERROR_ILLEGAL_ADDRESS",
    "OutOfMemoryError",
    "cudaErrorMemoryAllocation",
    "MemoryError",
    "Cannot allocate memory",
    "hipErrorOutOfMemory",
)


def classify_failure(returncode: int, stderr: str, timed_out: bool) -> tuple[str, str]:
    """Map a worker's exit into a status and a one-line reason."""
    if timed_out:
        return store.STATUS_TIMEOUT, "timeout"
    if returncode == -signal.SIGKILL or returncode == 137:
        return store.STATUS_OOM, "killed (SIGKILL), most likely the OOM killer"
    for pat in _OOM_PATTERNS:
        if pat in stderr:
            return store.STATUS_OOM, pat
    m = re.search(r"(CUDA_ERROR_\w+|cudaError\w+|hipError\w+)", stderr)
    if m:
        return store.STATUS_ERROR, m.group(1)
    tail = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    return store.STATUS_ERROR, (tail[-1][:200] if tail else f"exit code {returncode}")


class WorkerResult(dict):
    pass


def run_case(
    ref: Ref,
    case_id: registry.CaseId,
    bundle: store.Bundle,
    preset: str,
    repeats: int | None = None,
    timeout: float | None = None,
    python: str = sys.executable,
) -> dict[str, Any]:
    """Run one case in a fresh worker process and return its record."""
    cid = str(case_id)
    avail = store.mem_available_bytes()
    if avail is not None and avail < MEM_FLOOR_BYTES:
        record: dict[str, Any] = {
            "case_id": cid,
            "ref": ref.label,
            "sha": ref.sha,
            "preset": preset,
            "status": store.STATUS_SKIPPED_MEMORY,
            "error": (
                f"MemAvailable {avail / 1024**3:.1f} GB below the "
                f"{MEM_FLOOR_BYTES / 1024**3:.0f} GB floor"
            ),
            "timestamp_utc": store.utc_now(),
        }
        bundle.write_case(cid, record)
        return record

    env = dict(os.environ)
    env.update(presets.PRESETS[preset].env)
    env["PYTHONPATH"] = os.pathsep.join([str(HARNESS_DIR), str(ref.worktree)])
    cmd = [
        python,
        "-P",
        "-m",
        "abtem_bench.worker",
        "--case-id",
        cid,
        "--ref-label",
        ref.label,
        "--ref-sha",
        ref.sha,
        "--expected-root",
        str(ref.worktree),
        "--preset",
        preset,
        "--out",
        str(bundle.path),
    ]
    if repeats is not None:
        cmd += ["--repeats", str(repeats)]

    bundle.log_path(cid).parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    t0 = time.perf_counter()
    with open(bundle.log_path(cid), "w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(bundle.path),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def _kill():
            nonlocal timed_out
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        timer = threading.Timer(timeout, _kill) if timeout else None
        if timer:
            timer.start()
        try:
            _, status, rusage = os.wait4(proc.pid, 0)
        finally:
            if timer:
                timer.cancel()
        proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.perf_counter() - t0
    peak_rss = rusage.ru_maxrss * 1024  # Linux reports kilobytes

    written_by_worker = bundle.has_case(cid)
    if written_by_worker:
        result = bundle.read_case(cid)
    else:
        log_text = bundle.log_path(cid).read_text(errors="replace")
        status_name, reason = classify_failure(proc.returncode, log_text, timed_out)
        result = {
            "case_id": cid,
            "ref": ref.label,
            "sha": ref.sha,
            "preset": preset,
            "status": status_name,
            "error": reason,
            "timestamp_utc": store.utc_now(),
            "log_tail": "\n".join(log_text.splitlines()[-20:]),
        }
    if timed_out and result["status"] == store.STATUS_OK:
        result["status"] = store.STATUS_TIMEOUT
    memory: dict[str, Any] = dict(result.get("memory") or {})
    memory["peak_rss_bytes"] = int(peak_rss)
    result["memory"] = memory
    result["worker_wall"] = wall
    result["returncode"] = proc.returncode
    if written_by_worker:
        # keep the worker's outputs index; only the record changes
        store.dump_json(bundle.case_json(cid), result)
    else:
        bundle.write_case(cid, result)
    return result


def new_bundle(
    out: Path, ref: Ref, preset: str, tier: str, devices: list[str], command: str
) -> store.Bundle:
    bundle = store.Bundle(out)
    bundle.path.mkdir(parents=True, exist_ok=True)
    fp = store.fingerprint()
    bundle.write_manifest(
        {
            "schema_version": store.SCHEMA_VERSION,
            "harness_version": __import__("abtem_bench").__version__,
            "case_hash": registry.case_hash(),
            "ref": {
                "label": ref.label,
                "sha": ref.sha,
                "describe": ref.describe,
                "worktree": str(ref.worktree),
            },
            "preset": preset,
            "tier": tier,
            "devices": devices,
            "timestamp_utc": store.utc_now(),
            "fingerprint": fp,
            "fingerprint_short": store.fingerprint_short(fp),
            "load_average_at_start": store.load_average(),
            "mem_available_at_start": store.mem_available_bytes(),
            "command": command,
        }
    )
    return bundle


def capture(
    repo: Path,
    refs: list[str],
    case_ids: list[registry.CaseId],
    out: Path,
    preset: str,
    tier: str,
    devices: list[str],
    repeats: int | None,
    rounds: int = 1,
    labels: list[str] | None = None,
    command: str = "",
    progress=print,
) -> list[store.Bundle]:
    """Capture every case for every ref, interleaved per case (A, B, A, B, ...).

    One bundle per ref, at ``out/<label>``. ``labels`` overrides the bundle
    directory names (needed when the same ref is captured twice for a
    self-check).
    """
    prepared = [prepare_ref(repo, r) for r in refs]
    labels = labels or [r.label.replace("/", "_") for r in prepared]
    bundles = [
        new_bundle(out / lab, ref, preset, tier, devices, command)
        for lab, ref in zip(labels, prepared)
    ]
    timeout = TIER_TIMEOUTS.get(tier)
    for rnd in range(rounds):
        for cid in case_ids:
            for ref, bundle in zip(prepared, bundles):
                if rounds > 1 and bundle.has_case(str(cid)):
                    # later rounds append to the timing lists
                    prev = bundle.read_case(str(cid))
                else:
                    prev = None
                rec = run_case(
                    ref, cid, bundle, preset, repeats=repeats, timeout=timeout
                )
                if (
                    prev
                    and prev.get("status") == store.STATUS_OK
                    and rec.get("status") == store.STATUS_OK
                ):
                    for key in ("warm", "warm_cpu"):
                        rec["timings"][key] = prev["timings"].get(key, []) + rec[
                            "timings"
                        ].get(key, [])
                    rec["timings"]["rounds"] = rnd + 1
                    store.dump_json(bundle.case_json(str(cid)), rec)
                t = rec.get("timings", {}).get("median")
                rss = (rec.get("memory") or {}).get("peak_rss_bytes")
                progress(
                    f"[{rnd + 1}/{rounds}] {ref.label:<12} {cid!s:<44} "
                    f"{rec['status']:<10}"
                    + (f" median {t:8.3f} s" if t else "")
                    + (f"  rss {rss / 1024**2:7.0f} MB" if rss else "")
                )
    for bundle in bundles:
        _record_abtem_identity(bundle)
    return bundles


def _record_abtem_identity(bundle: store.Bundle) -> None:
    """Copy the abtem version and file the workers reported into the manifest."""
    for cid in bundle.case_ids():
        rec = bundle.read_case(cid)
        if rec.get("abtem_version"):
            manifest = bundle.read_manifest()
            manifest["abtem_version"] = rec["abtem_version"]
            manifest["abtem_file"] = rec.get("abtem_file")
            bundle.write_manifest(manifest)
            return
