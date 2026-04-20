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

Output columns are tab-separated so the table can be piped through
``column -t -s $'\\t'`` for additional alignment, or imported into a
spreadsheet.
"""

import json
import math
import re
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


def _normalize_label(label: str) -> str:
    """Return a canonical label suitable for cross-version matching.

    Newer abTEM resolves 'chunk=auto' to 'chunk=auto(NxM)' when
    ``estimate_potential_chunk_size`` is available.  Strip the parenthetical
    so results from old and new installations can be matched.
    """
    return re.sub(r'\bchunk=auto\(\d+x\d+\)', 'chunk=auto', label)


def _time_str(r: dict | None) -> str:
    if r is None:
        return "(absent)"
    if "error" in r:
        err = r["error"]
        tag = err[:24] if not err.startswith("KILLED") else "KILLED"
        return f"ERR:{tag}"
    return f"{r['time']:.3f}s"


def _mem_str(r: dict | None) -> str:
    """Return a memory string for either GPU (vRAM) or CPU (RAM) results."""
    if r is None:
        return ""
    vram = r.get("peak_vram_mb")
    ram  = r.get("peak_ram_mb")
    if vram:
        return f"{vram / 1000:.1f}G"
    if ram:
        return f"{ram / 1000:.1f}G"
    return ""


def _gpu_str(r: dict | None) -> str:
    """Return GPU utilization string, or blank when absent (CPU runs, old JSON)."""
    if r is None:
        return ""
    util = r.get("mean_gpu_util")
    if util is None:
        return ""
    return f"{util:.0f}%"


def _speedup_str(old_r: dict | None, new_r: dict | None) -> str:
    old_t = old_r.get("time") if old_r and "error" not in old_r else None
    new_t = new_r.get("time") if new_r and "error" not in new_r else None
    if old_t is None or new_t is None or new_t == 0:
        return "n/a"
    ratio = old_t / new_t
    sign = "▲" if ratio >= 1.0 else "▼"
    return f"{sign}{ratio:.2f}×"


# ──────────────────────────────────────────────────────────────────────
# Pairing logic
# ──────────────────────────────────────────────────────────────────────

def _build_pairs(old_results: list[dict], new_results: list[dict]) -> list[tuple]:
    """Return an ordered list of (norm_label, display_label, old_r, new_r).

    Matching is done on *normalised* labels (chunk=auto(NxM) → chunk=auto)
    to handle version differences in label verbosity.  When multiple entries
    share the same normalised label (e.g. three different scan configs that all
    produce 'batch=auto(1)' after normalisation), they are paired positionally
    in encounter order rather than by label identity.
    """
    from collections import defaultdict

    # Group each side's results by normalised label, preserving order.
    old_groups: dict[str, list[dict]] = defaultdict(list)
    for r in old_results:
        old_groups[_normalize_label(r["label"])].append(r)

    new_groups: dict[str, list[dict]] = defaultdict(list)
    for r in new_results:
        new_groups[_normalize_label(r["label"])].append(r)

    # Track which normalised labels we have seen, preserving first-encounter order.
    seen_norms: dict[str, None] = {}
    for r in old_results:
        seen_norms[_normalize_label(r["label"])] = None
    for r in new_results:
        seen_norms[_normalize_label(r["label"])] = None

    # Consume from each group positionally.
    pairs: list[tuple] = []
    for norm in seen_norms:
        old_list = old_groups[norm]
        new_list = new_groups[norm]
        n_pairs = max(len(old_list), len(new_list))

        for i in range(n_pairs):
            old_r = old_list[i] if i < len(old_list) else None
            new_r = new_list[i] if i < len(new_list) else None
            # Display label: prefer new (more detailed) when available.
            display_label = (new_r or old_r)["label"]
            pairs.append((norm, display_label, old_r, new_r))

    return pairs


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

    pairs = _build_pairs(old_results, new_results)

    # ── Pre-collect rows so we can compute dynamic label width ───────
    rows: list[tuple[str, dict | None, dict | None]] = []
    for _norm, display_label, old_r, new_r in pairs:
        if old_r is None:
            marker = " [new only]"
        elif new_r is None:
            marker = " [old only]"
        else:
            marker = ""
        rows.append((display_label + marker, old_r, new_r))

    W = max((len(lbl) for lbl, _, _ in rows), default=40)
    W = max(W, 40)

    # Detect whether any result has GPU util so we can omit the column for
    # CPU-only comparisons.
    all_results = old_results + new_results
    has_gpu_util = any(r.get("mean_gpu_util") is not None for r in all_results)

    # ── Header ──────────────────────────────────────────────────────
    gpu_cols = 2 * (1 + 5) if has_gpu_util else 0   # two "\t{GPU%:>5}" columns
    SEP = "=" * (W + 2 + 14 + 8 + gpu_cols + 14 + 8 + 10)
    print(f"\n{SEP}")
    print("Benchmark comparison")
    print(f"  OLD ({old_path}): abTEM {old_meta.get('abtem_version', '?')}  "
          f"device={old_meta.get('device', '?')}  quick={old_meta.get('quick', '?')}")
    print(f"  NEW ({new_path}): abTEM {new_meta.get('abtem_version', '?')}  "
          f"device={new_meta.get('device', '?')}  quick={new_meta.get('quick', '?')}")
    print(SEP)
    # Tab-separated column headers; label is padded to W for alignment.
    gpu_hdr = f"\t{'GPU%':>5}" if has_gpu_util else ""
    print(f"  {'Label':{W}}\t{'Old time':>12}\t{'Mem':>6}{gpu_hdr}\t{'New time':>12}\t{'Mem':>6}{gpu_hdr}\t{'Speedup':>8}")
    print(f"  {'-'*W}\t{'-'*12}\t{'-'*6}{('-'*6 if has_gpu_util else '')}\t{'-'*12}\t{'-'*6}{('-'*6 if has_gpu_util else '')}\t{'-'*8}")

    # ── Per-entry rows ───────────────────────────────────────────────
    ratios: list[float] = []
    n_faster = n_slower = n_same = 0

    for label_col, old_r, new_r in rows:
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

        old_gpu = f"\t{_gpu_str(old_r):>5}" if has_gpu_util else ""
        new_gpu = f"\t{_gpu_str(new_r):>5}" if has_gpu_util else ""
        print(
            f"  {label_col:{W}}"
            f"\t{_time_str(old_r):>12}"
            f"\t{_mem_str(old_r):>6}"
            f"{old_gpu}"
            f"\t{_time_str(new_r):>12}"
            f"\t{_mem_str(new_r):>6}"
            f"{new_gpu}"
            f"\t{sp:>8}"
        )

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
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
