#!/usr/bin/env python3
"""Compare two benchmark JSON outputs from benchmark_potential_chunking.py.

Usage
-----
First, run the benchmark on each abTEM installation and save JSON output:

    # New / current version:
    python benchmarks/benchmark_potential_chunking.py --device gpu \\
        --output-json results_new.json

    # Old / baseline version (can be run with a different Python env):
    /path/to/old/python benchmarks/benchmark_potential_chunking.py --device gpu \\
        --output-json results_old.json

Then compare:

    python benchmarks/compare_benchmarks.py results_old.json results_new.json

Labels that appear in only one file are flagged as "old only" / "new only".
Entries where either side is an error or was killed are also highlighted.
"""

import json
import math
import sys


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def load(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # Support both the new {"meta": ..., "results": [...]} format and a plain
    # list (in case the user produced their own JSON).
    if isinstance(data, list):
        return {"meta": {}, "results": data}
    return data


def _time_str(r: dict | None) -> str:
    if r is None:
        return "    (absent)"
    if "error" in r:
        err = r["error"]
        tag = err[:20] if not err.startswith("KILLED") else "KILLED"
        return f"  ERR:{tag:<16s}"
    return f"{r['time']:9.3f}s"


def _vram_str(r: dict | None) -> str:
    if r is None or "peak_vram_mb" not in r:
        return "      "
    return f"{r['peak_vram_mb'] / 1000:5.1f}G"


def _speedup_str(old_r: dict | None, new_r: dict | None) -> str:
    old_t = old_r.get("time") if old_r and "error" not in old_r else None
    new_t = new_r.get("time") if new_r and "error" not in new_r else None
    if old_t is None or new_t is None or new_t == 0:
        return "     n/a"
    ratio = old_t / new_t
    sign = "▲" if ratio >= 1.0 else "▼"
    return f"{sign}{ratio:6.2f}×"


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        print(f"Usage: {sys.argv[0]} OLD.json NEW.json")
        sys.exit(1)

    old_path, new_path = sys.argv[1], sys.argv[2]
    old_data = load(old_path)
    new_data = load(new_path)

    old_results = old_data["results"]
    new_results = new_data["results"]
    old_meta = old_data.get("meta", {})
    new_meta = new_data.get("meta", {})

    old_by_label = {r["label"]: r for r in old_results}
    new_by_label = {r["label"]: r for r in new_results}

    # Preserve encounter order; prefer old ordering for matched entries.
    seen: dict[str, None] = {}
    for r in old_results:
        seen[r["label"]] = None
    for r in new_results:
        seen[r["label"]] = None
    all_labels = list(seen)

    # ── Header ──────────────────────────────────────────────────────
    W = 72
    print(f"\n{'=' * (W + 58)}")
    print(f"Benchmark comparison")
    print(f"  OLD ({old_path}): abTEM {old_meta.get('abtem_version', '?')}  "
          f"device={old_meta.get('device', '?')}  quick={old_meta.get('quick', '?')}")
    print(f"  NEW ({new_path}): abTEM {new_meta.get('abtem_version', '?')}  "
          f"device={new_meta.get('device', '?')}  quick={new_meta.get('quick', '?')}")
    print(f"{'=' * (W + 58)}")
    print(f"  {'Label':<{W}}  {'Old time':>11}  {'vRAM':>6}  "
          f"{'New time':>11}  {'vRAM':>6}  {'Speedup':>8}")
    print(f"  {'-' * W}  {'-' * 11}  {'-' * 6}  {'-' * 11}  {'-' * 6}  {'-' * 8}")

    # ── Per-entry rows ───────────────────────────────────────────────
    ratios: list[float] = []
    n_faster = n_slower = n_same = 0

    for label in all_labels:
        old_r = old_by_label.get(label)
        new_r = new_by_label.get(label)

        if old_r is None:
            marker = " [new only]"
        elif new_r is None:
            marker = " [old only]"
        else:
            marker = ""

        sp = _speedup_str(old_r, new_r)

        old_t = old_r.get("time") if old_r and "error" not in old_r else None
        new_t = new_r.get("time") if new_r and "error" not in new_r else None
        if old_t and new_t:
            ratio = old_t / new_t
            ratios.append(ratio)
            if ratio > 1.02:
                n_faster += 1
            elif ratio < 0.98:
                n_slower += 1
            else:
                n_same += 1

        display_label = label + marker
        print(
            f"  {display_label:<{W}}  {_time_str(old_r):>11}  {_vram_str(old_r):>6}  "
            f"{_time_str(new_r):>11}  {_vram_str(new_r):>6}  {sp:>8}"
        )

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * (W + 58)}")
    n_compared = len(ratios)
    if n_compared > 0:
        geo_mean = math.exp(sum(math.log(r) for r in ratios) / n_compared)
        print(f"  Compared {n_compared} matching timed entries  "
              f"(▲ faster: {n_faster}  ▼ slower: {n_slower}  ≈ same: {n_same})")
        print(f"  Geometric mean speedup (old/new): {geo_mean:.3f}×"
              f"  {'(new is faster overall)' if geo_mean > 1 else '(old is faster overall)'}")
    else:
        print("  No matching timed entries to compare.")

    n_old_err = sum(1 for r in old_results if "error" in r)
    n_new_err = sum(1 for r in new_results if "error" in r)
    if n_old_err or n_new_err:
        print(f"  Errors/killed — old: {n_old_err}  new: {n_new_err}")
    print()


if __name__ == "__main__":
    main()
