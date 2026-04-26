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

Entries are grouped by potential gpts size.  Within each group, PlaneWave and
Scan entries are shown with abbreviated labels (the repeated gpts/slices prefix
is omitted since it is already captured by the group header).

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
from collections import defaultdict


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

    Strips version-specific parentheticals so that results from different
    abTEM versions can be paired:

    * ``chunk=auto(NxM)``  → ``chunk=auto``   (new versions resolve the chunk
      count and size; old versions just say 'auto')
    * ``batch=auto(N)``    → ``batch=auto``   (auto-tuned batch size depends on
      available VRAM and therefore differs between versions)
    """
    label = re.sub(r'\bchunk=auto\(\d+x\d+\)', 'chunk=auto', label)
    label = re.sub(r'\bbatch=auto\(\d+\)', 'batch=auto', label)
    return label


def _err_tag(err: str) -> str:
    """Condense a benchmark error string to a short tag for table display."""
    if err.startswith("OOM") or "OutOfMemory" in err or "Out of memory" in err:
        return "OOM"
    if err.startswith("KILLED"):
        return "KILLED"
    # Map known CUDA error codes to short descriptive tags.
    m = re.search(r"CUDA_ERROR_(\w+)", err)
    if m:
        _CUDA_SHORT = {
            "OUT_OF_MEMORY":     "OOM",
            "ILLEGAL_ADDRESS":   "OOM",       # memory-pressure hard fault
            "INVALID_VALUE":     "CUDA:INVAL", # e.g. kernel launch grid too large
            "LAUNCH_FAILED":     "CUDA:LAUNCH",
        }
        return _CUDA_SHORT.get(m.group(1), f"CUDA:{m.group(1)[:8]}")
    if "CudaAPIError" in err or "cudaError" in err:
        return "CUDA_ERR"
    return "ERR"


def _time_str(r: dict | None) -> str:
    if r is None:
        return "(absent)"
    if "error" in r:
        return _err_tag(r["error"])
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


def _mem_val(r: dict | None) -> float | None:
    """Return memory in MB, or None when unavailable."""
    if r is None or "error" in r:
        return None
    v = r.get("peak_vram_mb") or r.get("peak_ram_mb")
    return v or None


def _mem_delta_str(old_r: dict | None, new_r: dict | None) -> str:
    """Return new/old memory ratio (positive = new uses more memory)."""
    old_m = _mem_val(old_r)
    new_m = _mem_val(new_r)
    if old_m is None or new_m is None or old_m == 0:
        return ""
    ratio = new_m / old_m
    sign = "▲" if ratio > 1.02 else ("▼" if ratio < 0.98 else "≈")
    return f"{sign}{ratio:.2f}×"


def _gpu_str(r: dict | None) -> str:
    """Return GPU utilization string, or blank when absent (CPU runs, old JSON)."""
    if r is None:
        return ""
    util = r.get("mean_gpu_util")
    if util is None:
        return ""
    return f"{util:.0f}%"


def _cpu_str(r: dict | None) -> str:
    """Return CPU utilization string, or blank when absent or psutil unavailable."""
    if r is None:
        return ""
    util = r.get("mean_cpu_util")
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
# Label parsing helpers
# ──────────────────────────────────────────────────────────────────────

def _extract_gpts(r: dict) -> str | None:
    """Return a gpts group key string for a result dict, or None.

    Priority:
    1. Structured ``potential_gpts`` field (set by current benchmark code,
       including in error returns).
    2. ``gpts=(N, M)`` token in the label string (PlaneWave entries).
    3. ``scan=(N, M)`` token as a fallback group for scan entries whose
       potential size is not recorded (old-format JSON).  The key is prefixed
       with ``scan=`` so the group header can display it differently.
    """
    pg = r.get("potential_gpts")
    if pg is not None:
        return f"({pg[0]}, {pg[1]})"
    label = r.get("label", "")
    m = re.search(r'\bgpts=\(\s*(\d+),\s*(\d+)\s*\)', label)
    if m:
        return f"({m.group(1)}, {m.group(2)})"
    # Fallback for scan entries without a recorded potential gpts.
    ms = re.search(r'\bscan=\(\s*(\d+),\s*(\d+)\s*\)', label)
    if ms:
        return f"scan=({ms.group(1)}, {ms.group(2)})"
    return None


def _row_type_and_short(label: str) -> tuple[str, str]:
    """Return (type_tag, shortened_label) for grouped display.

    * Scan entries  → tag = ``Scan(Nx,Ny)``, short label strips gpts/slices prefix
      and scan= field (both are captured by the group context).
    * PlaneWave entries → tag = ``PW``, short label strips the gpts prefix.
    """
    # Detect scan entries by the presence of scan=(N, M).
    m_scan = re.search(r'\bscan=\(\s*(\d+),\s*(\d+)\s*\)', label)
    if m_scan:
        tag = f"Scan({m_scan.group(1)},{m_scan.group(2)})"
        short = label
        # Strip leading gpts=(…) and optional slices=N, prefix.
        short = re.sub(r'^gpts=\([^)]+\),\s*(?:slices=\d+,\s*)?', '', short)
        # Strip the scan=(…) field (and surrounding punctuation).
        short = re.sub(r',?\s*scan=\([^)]+\)', '', short)
        short = short.strip().strip(',').strip()
        return tag, short

    # PlaneWave entry: strip leading gpts=(…), prefix.
    short = re.sub(r'^gpts=\([^)]+\),\s*', '', label)
    return "PW", short


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

    # ── Build display rows and assign to gpts groups ──────────────────
    # Each row: (gpts_key, type_tag, short_label, full_label, marker, old_r, new_r)
    # Rows without a gpts key go in the special "_ungrouped" bucket.
    group_order: list[str] = []   # preserves first-encounter order of gpts keys
    groups: dict[str, list] = defaultdict(list)

    for _norm, display_label, old_r, new_r in pairs:
        marker = ""
        if old_r is None:
            marker = " [new only]"
        elif new_r is None:
            marker = " [old only]"

        # Determine gpts group from whichever side has the metadata.
        gpts_key = _extract_gpts(new_r or {}) or _extract_gpts(old_r or {})
        if gpts_key is None:
            gpts_key = "_ungrouped"

        type_tag, short_label = _row_type_and_short(display_label)
        row = (type_tag, short_label, display_label, marker, old_r, new_r)

        if gpts_key not in groups:
            group_order.append(gpts_key)
        groups[gpts_key].append(row)

    # ── Compute column widths across all rows ─────────────────────────
    all_results = old_results + new_results
    has_gpu_util = any(r.get("mean_gpu_util") is not None for r in all_results)
    has_cpu_util = any(r.get("mean_cpu_util") is not None for r in all_results)
    has_mem_delta = any(
        _mem_val(old_r) is not None and _mem_val(new_r) is not None
        for _gk, rows in groups.items()
        for _tt, _sl, _fl, _mk, old_r, new_r in rows
    )

    # W = width of the abbreviated label column (type_tag + short_label + marker).
    W = 40
    for rows in groups.values():
        for type_tag, short_label, _fl, marker, _o, _n in rows:
            W = max(W, len(type_tag) + 2 + len(short_label) + len(marker))
    W += 1   # small breathing room

    SEP = "=" * (W + 2 + 14 + 8
                 + (7 if has_gpu_util else 0)
                 + (7 if has_cpu_util else 0)
                 + 14 + 8
                 + (7 if has_gpu_util else 0)
                 + (7 if has_cpu_util else 0)
                 + 10)

    # ── Global header ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("Benchmark comparison")
    print(f"  OLD ({old_path}): abTEM {old_meta.get('abtem_version', '?')}  "
          f"device={old_meta.get('device', '?')}  quick={old_meta.get('quick', '?')}")
    print(f"  NEW ({new_path}): abTEM {new_meta.get('abtem_version', '?')}  "
          f"device={new_meta.get('device', '?')}  quick={new_meta.get('quick', '?')}")
    print(SEP)

    gpu_hdr  = f"\t{'GPU%':>5}"  if has_gpu_util  else ""
    cpu_hdr  = f"\t{'CPU%':>5}"  if has_cpu_util  else ""
    dmem_hdr = f"\t{'ΔMem':>8}"  if has_mem_delta else ""
    print(f"  {'Label':{W}}\t{'Old time':>12}\t{'Mem':>6}{gpu_hdr}{cpu_hdr}"
          f"\t{'New time':>12}\t{'Mem':>6}{gpu_hdr}{cpu_hdr}\t{'Speedup':>8}{dmem_hdr}")
    dash6  = f"\t{'-'*6}" if has_gpu_util  else ""
    dash6c = f"\t{'-'*6}" if has_cpu_util  else ""
    dmem_dash = f"\t{'-'*8}" if has_mem_delta else ""
    print(f"  {'-'*W}\t{'-'*12}\t{'-'*6}{dash6}{dash6c}"
          f"\t{'-'*12}\t{'-'*6}{dash6}{dash6c}\t{'-'*8}{dmem_dash}")

    # ── Per-group output ──────────────────────────────────────────────
    ratios: list[float] = []
    n_faster = n_slower = n_same = 0

    for gpts_key in group_order:
        rows = groups[gpts_key]

        # Group header line.
        if gpts_key == "_ungrouped":
            hdr_text = "── ungrouped "
        elif gpts_key.startswith("scan="):
            # Scan entries whose potential gpts isn't recorded (old-format JSON).
            hdr_text = f"── {gpts_key} [potential gpts unknown] "
        else:
            hdr_text = f"── gpts={gpts_key} "
        print(f"\n  {hdr_text}{'─' * max(0, W - len(hdr_text))}")

        for type_tag, short_label, _display_label, marker, old_r, new_r in rows:
            abbreviated = f"{type_tag}  {short_label}{marker}"
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

            old_gpu  = f"\t{_gpu_str(old_r):>5}" if has_gpu_util  else ""
            new_gpu  = f"\t{_gpu_str(new_r):>5}" if has_gpu_util  else ""
            old_cpu  = f"\t{_cpu_str(old_r):>5}" if has_cpu_util  else ""
            new_cpu  = f"\t{_cpu_str(new_r):>5}" if has_cpu_util  else ""
            dmem_col = f"\t{_mem_delta_str(old_r, new_r):>8}" if has_mem_delta else ""
            print(
                f"  {abbreviated:{W}}"
                f"\t{_time_str(old_r):>12}"
                f"\t{_mem_str(old_r):>6}"
                f"{old_gpu}"
                f"{old_cpu}"
                f"\t{_time_str(new_r):>12}"
                f"\t{_mem_str(new_r):>6}"
                f"{new_gpu}"
                f"{new_cpu}"
                f"\t{sp:>8}"
                f"{dmem_col}"
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
