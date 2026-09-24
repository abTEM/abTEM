"""Compare two bundles: results, speed, memory.

Never imports the refs' abtem. It reads the store format and uses
``abtem.core.testing.close_stats`` from the invoking checkout for the metric
vector.
"""

from __future__ import annotations

import fnmatch
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from abtem_bench import registry, store

IDENTICAL = "IDENTICAL"
OK = "OK"
DRIFT = "DRIFT"
ACCEPTED = "ACCEPTED"
SHAPE = "SHAPE"
ONLY_A = "ONLY-A"
ONLY_B = "ONLY-B"

ACCEPTED_CHANGES_PATH = Path(__file__).resolve().parents[1] / "accepted_changes.toml"


class CompareError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# accepted_changes.toml


@dataclass
class Accepted:
    case: str
    since: str
    reason: str
    pr: int | None = None
    matched: list[str] = field(default_factory=list)

    def matches(self, case_id: str) -> bool:
        cid = registry.CaseId.parse(case_id)
        pattern = _literal_brackets(self.case)
        return fnmatch.fnmatchcase(case_id, pattern) or fnmatch.fnmatchcase(
            cid.name, pattern
        )


def _literal_brackets(glob: str) -> str:
    """Escape ``[`` and ``]`` so a variant like ``x[order1]@*`` matches literally.

    fnmatch treats brackets as character classes; case ids use them for the
    variant name.
    """
    return glob.replace("[", "[[]").replace("]", "[]]").replace("[[[]]", "[[]")


def load_accepted(
    path: Path | None = None, known_names: set[str] | None = None
) -> list[Accepted]:
    path = path or ACCEPTED_CHANGES_PATH
    if not path.exists():
        return []
    data = tomllib.loads(path.read_text())
    out = []
    for i, entry in enumerate(data.get("accepted", [])):
        for key in ("case", "since", "reason"):
            if key not in entry:
                raise CompareError(f"{path.name}: entry {i} lacks {key!r}")
        acc = Accepted(
            entry["case"], str(entry["since"]), entry["reason"], entry.get("pr")
        )
        if known_names is not None:
            name_glob = acc.case.split("@")[0].split("[")[0]
            if not any(fnmatch.fnmatchcase(n, name_glob) for n in known_names):
                raise CompareError(
                    f"{path.name}: entry {i} case glob {acc.case!r} "
                    "matches no registered case"
                )
        out.append(acc)
    return out


# ---------------------------------------------------------------------------
# pairing


SAME = "same"
ATTRIBUTION = "attribution"


def pair_ids(
    reference: store.Bundle,
    candidate: store.Bundle,
    reg: dict[str, registry.Case] | None,
) -> list[tuple[str | None, str | None, str]]:
    """(reference id, candidate id, kind) triples.

    Same ids pair first (kind ``same``). A candidate variant declaring
    ``compare_as`` additionally pairs with the reference's ``compare_as``
    variant (kind ``attribution``): ``x[order1]`` on the candidate against
    ``x`` on the reference isolates what the variant removes. Attribution rows
    are informational and never count as failures.
    """
    ref_ids = set(reference.case_ids())
    pairs: list[tuple[str | None, str | None, str]] = []
    seen_ref: set[str] = set()
    for cid_str in candidate.case_ids():
        cid = registry.CaseId.parse(cid_str)
        if cid_str in ref_ids:
            pairs.append((cid_str, cid_str, SAME))
            seen_ref.add(cid_str)
        target = None
        if reg and cid.name in reg and cid.variant in reg[cid.name].variants:
            ca = reg[cid.name].variants[cid.variant].compare_as
            if ca is not None:
                target = str(cid.with_variant(ca))
        if target is not None and target in ref_ids:
            pairs.append((target, cid_str, ATTRIBUTION))
        elif cid_str not in ref_ids:
            pairs.append((None, cid_str, SAME))
    for r in sorted(ref_ids - seen_ref):
        pairs.append((r, None, SAME))
    return pairs


# ---------------------------------------------------------------------------
# metrics


def output_stats(
    ref_arr: np.ndarray,
    cand_arr: np.ndarray,
    ref_axes: list,
    cand_axes: list,
    above_rel: float,
) -> dict[str, Any]:
    from abtem.core.testing import close_stats

    s = close_stats(cand_arr, ref_arr, above_rel=above_rel)
    s["dtype_ok"] = str(cand_arr.dtype) == str(ref_arr.dtype)
    s["axes_ok"] = _axes_equal(ref_axes, cand_axes)
    return s


def _error_note(record: dict[str, Any]) -> str:
    """One line naming a failed record's exception, for the report."""
    if record.get("status") == store.STATUS_OK:
        return ""
    summary = record.get("error_summary")
    if summary:
        return str(summary)
    err = (record.get("error") or "").strip()
    lines = [ln.strip() for ln in err.splitlines() if ln.strip()]
    return (lines[-1] if lines else "")[:300]


def _axes_equal(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        xs = {
            k: v for k, v in x.items() if k not in ("label", "tex_label", "tex_units")
        }
        ys = {
            k: v for k, v in y.items() if k not in ("label", "tex_label", "tex_units")
        }
        if xs != ys:
            return False
    return True


def verdict_for(stats: dict[str, Any], tol: registry.Tolerance) -> str:
    if not (
        stats.get("shape_ok")
        and stats.get("dtype_ok", True)
        and stats.get("axes_ok", True)
    ):
        return SHAPE
    if stats.get("identical"):
        return IDENTICAL
    rel = stats.get("rel_above")
    rel_ok = (
        (rel is None) or (isinstance(rel, float) and math.isnan(rel)) or rel <= tol.rel
    )
    inten = stats.get("intensity", 0.0)
    inten_ok = abs(inten) <= tol.intensity if not math.isnan(inten) else True
    mabs_ok = stats.get("max_abs_norm", 0.0) <= tol.max_abs_norm
    return OK if (rel_ok and inten_ok and mabs_ok) else DRIFT


# ---------------------------------------------------------------------------
# noise floor


def noise_floor(
    bundle_a: store.Bundle, bundle_b: store.Bundle
) -> dict[str, dict[str, float]]:
    """Per-case floors from a self-check pair: speed spread and accuracy spread."""
    floors: dict[str, dict[str, float]] = {}
    for cid in bundle_a.case_ids():
        if not bundle_b.has_case(cid):
            continue
        ra, rb = bundle_a.read_case(cid), bundle_b.read_case(cid)
        if ra.get("status") != store.STATUS_OK or rb.get("status") != store.STATUS_OK:
            continue
        ta, tb = ra["timings"].get("median"), rb["timings"].get("median")
        f: dict[str, float] = {}
        if ta and tb:
            f["speed"] = abs(ta / tb - 1.0)
        ma, mb = ra["memory"].get("peak_rss_bytes"), rb["memory"].get("peak_rss_bytes")
        if ma and mb:
            f["memory"] = abs(ma / mb - 1.0)
        rel = 0.0
        for name in ra.get("outputs", {}):
            if name not in rb.get("outputs", {}):
                continue
            a, ax_a, _ = bundle_a.load_output(cid, name)
            b, ax_b, _ = bundle_b.load_output(cid, name)
            s = output_stats(a, b, ax_a, ax_b, 1e-6)
            r = s.get("rel_above")
            if r is not None and not math.isnan(r):
                rel = max(rel, r)
        f["accuracy"] = rel
        floors[cid] = f
    return floors


# ---------------------------------------------------------------------------
# main comparison


@dataclass
class Row:
    ref_id: str | None
    cand_id: str | None
    verdict: str
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    time_ratio: float | None = None
    time_ref: float | None = None
    time_cand: float | None = None
    cold_ratio: float | None = None
    rss_ratio: float | None = None
    vram_ratio: float | None = None
    flags: list[str] = field(default_factory=list)
    accepted_by: str | None = None
    note: str = ""
    kind: str = SAME


@dataclass
class Report:
    reference: dict[str, Any]
    candidate: dict[str, Any]
    rows: list[Row]
    accepted: list[Accepted]
    stale_accepted: list[Accepted]
    case_hash_match: bool
    fingerprint_match: bool
    thresholds: dict[str, float]
    noise: dict[str, dict[str, float]]

    def failures(self, fail_on: list[str]) -> list[str]:
        out = []
        for r in self.rows:
            if r.kind != SAME:
                continue
            if "drift" in fail_on and r.verdict == DRIFT:
                out.append(f"{r.cand_id}: DRIFT")
            if "shape" in fail_on and r.verdict == SHAPE:
                out.append(f"{r.cand_id}: SHAPE")
            if "speed" in fail_on and "speed" in r.flags:
                out.append(f"{r.cand_id}: speed x{r.time_ratio:.2f}")
            if "memory" in fail_on and "memory" in r.flags:
                out.append(f"{r.cand_id}: rss x{r.rss_ratio:.2f}")
            if "error" in fail_on and r.verdict in (
                store.STATUS_ERROR,
                store.STATUS_OOM,
                store.STATUS_TIMEOUT,
            ):
                out.append(f"{r.cand_id}: {r.verdict}")
        return out


def compare(
    reference: store.Bundle,
    candidate: store.Bundle,
    reg: dict[str, registry.Case] | None = None,
    accepted_path: Path | None = None,
    noise: dict[str, dict[str, float]] | None = None,
    speed_threshold: float = 0.10,
    memory_threshold: float = 0.05,
    allow_case_mismatch: bool = False,
    min_time: float = 0.5,
) -> Report:
    man_r, man_c = reference.read_manifest(), candidate.read_manifest()
    hash_match = man_r.get("case_hash") == man_c.get("case_hash")
    if not hash_match and not allow_case_mismatch:
        raise CompareError(
            "case_hash differs between the bundles: the case code was not the same; "
            "pass --allow-case-mismatch to compare anyway"
        )
    fp_match = man_r.get("fingerprint_short") == man_c.get("fingerprint_short")
    accepted = load_accepted(accepted_path, set(reg) if reg else None)
    noise = noise or {}
    rows: list[Row] = []

    for ref_id, cand_id, kind in pair_ids(reference, candidate, reg):
        if ref_id is None:
            rows.append(Row(None, cand_id, ONLY_B))
            continue
        if cand_id is None:
            rows.append(Row(ref_id, None, ONLY_A))
            continue
        rr, rc = reference.read_case(ref_id), candidate.read_case(cand_id)
        if rr["status"] != store.STATUS_OK or rc["status"] != store.STATUS_OK:
            bad = rc["status"] if rc["status"] != store.STATUS_OK else rr["status"]
            rows.append(
                Row(
                    ref_id,
                    cand_id,
                    bad,
                    note=_error_note(rc) or _error_note(rr),
                    kind=kind,
                )
            )
            continue

        cid = registry.CaseId.parse(cand_id)
        tol = (
            reg[cid.name].tolerance if reg and cid.name in reg else registry.Tolerance()
        )
        flag_ok = kind == SAME
        if reg and cid.name in reg and cid.variant in reg[cid.name].variants:
            flag_ok = flag_ok and reg[cid.name].variants[cid.variant].flag
        row = Row(ref_id, cand_id, IDENTICAL, kind=kind)
        if kind == ATTRIBUTION:
            row.note = f"attribution: vs reference `{ref_id}`"
        worst = IDENTICAL
        order = {IDENTICAL: 0, OK: 1, DRIFT: 2, SHAPE: 3}
        for name in rr.get("outputs", {}):
            if name not in rc.get("outputs", {}):
                row.outputs[name] = {"verdict": SHAPE, "note": "missing in candidate"}
                worst = SHAPE
                continue
            a, ax_a, _ = reference.load_output(ref_id, name)
            b, ax_b, _ = candidate.load_output(cand_id, name)
            s = output_stats(a, b, ax_a, ax_b, tol.above_rel)
            v = verdict_for(s, tol)
            s["verdict"] = v
            row.outputs[name] = s
            if order[v] > order[worst]:
                worst = v
        row.verdict = worst
        if worst == DRIFT and kind == SAME:
            for acc in accepted:
                if acc.matches(cand_id):
                    row.verdict = ACCEPTED
                    row.accepted_by = acc.reason
                    acc.matched.append(cand_id)
                    break

        tr, tc = rr["timings"].get("median"), rc["timings"].get("median")
        if tr and tc:
            row.time_ref, row.time_cand, row.time_ratio = tr, tc, tc / tr
            floor = noise.get(cand_id, {}).get("speed", 0.0)
            beyond = abs(row.time_ratio - 1.0) > max(speed_threshold, 3 * floor)
            if beyond and min(tr, tc) < min_time:
                # sub-second runs are launch-overhead noise; say so, don't flag
                row.flags.append("short")
            elif flag_ok and beyond:
                row.flags.append("speed")
        cr, cc = rr["timings"].get("cold"), rc["timings"].get("cold")
        if cr and cc:
            row.cold_ratio = cc / cr
        mr, mc = rr["memory"].get("peak_rss_bytes"), rc["memory"].get("peak_rss_bytes")
        if mr and mc:
            row.rss_ratio = mc / mr
            floor = noise.get(cand_id, {}).get("memory", 0.0)
            if flag_ok and abs(row.rss_ratio - 1.0) > max(memory_threshold, 3 * floor):
                row.flags.append("memory")
        vr, vc = (
            rr["memory"].get("peak_vram_device_bytes"),
            rc["memory"].get("peak_vram_device_bytes"),
        )
        if vr and vc:
            row.vram_ratio = vc / vr
        rows.append(row)

    stale = [a for a in accepted if not a.matched]
    return Report(
        reference=man_r,
        candidate=man_c,
        rows=rows,
        accepted=accepted,
        stale_accepted=stale,
        case_hash_match=hash_match,
        fingerprint_match=fp_match,
        thresholds={
            "speed": speed_threshold,
            "memory": memory_threshold,
            "min_time": min_time,
        },
        noise=noise,
    )


# ---------------------------------------------------------------------------
# rendering


def _fmt(x: float | None, spec: str = ".2f", none: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return none
    return format(x, spec)


def _sci(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return "0" if x == 0 else f"{x:.1e}"


def _ratio(x: float | None) -> str:
    if x is None:
        return ""
    sign = "▲" if x > 1.02 else ("▼" if x < 0.98 else "≈")
    return f"{sign}{x:.2f}"


def to_markdown(report: Report) -> str:
    r, c = report.reference, report.candidate
    lines = []
    lines.append("# abtem-bench comparison")
    lines.append("")
    for label, m in (("Reference", r), ("Candidate", c)):
        lines.append(
            f"{label}: `{m['ref']['label']}` ({m['ref']['describe']}, "
            f"{m['ref']['sha'][:12]}), abtem {m.get('abtem_version', '?')}, "
            f"preset `{m['preset']}`, tier `{m['tier']}`, captured "
            f"{m['timestamp_utc']} on {m['fingerprint'].get('hostname')} "
            f"(fingerprint {m.get('fingerprint_short')})."
        )
    notes = []
    if not report.case_hash_match:
        notes.append(
            "case_hash differs: the two bundles did not run the same case code"
        )
    if not report.fingerprint_match:
        notes.append(
            "machine fingerprints differ: bit-identity is not expected, "
            "tolerances apply"
        )
    if report.noise:
        notes.append(
            f"noise floor from a self-check of {len(report.noise)} cases; "
            "speed and memory flags use max(threshold, 3 x floor)"
        )
    for n in notes:
        lines.append(f"Note: {n}.")
    lines.append("")
    lines.append(
        "Rows: one per paired case id; attribution rows (`x[variant]` vs `x`) "
        "compare a candidate variant against the reference default and are "
        "informational. Columns: verdict over all outputs; `identical` = every "
        "output bit-for-bit equal; `rel` = largest relative error over elements "
        "above the case's `above_rel` fraction of the reference maximum (worst "
        "output); `intensity` = relative change of the integrated intensity "
        "(worst output); `time` = candidate/reference warm median with the two "
        "medians in seconds; `cold` = candidate/reference first-call time; "
        "`rss` = candidate/reference peak resident memory of the worker "
        "process; flags mark ratios beyond threshold (`short`: a time ratio beyond "
        "threshold on a run too short to judge, below the minimum duration)."
    )
    lines.append("")
    lines.append(
        "| case | verdict | identical | rel | intensity | time | cold | rss "
        "| flags | note |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in report.rows:
        ident = (
            all(o.get("identical") for o in row.outputs.values())
            if row.outputs
            else None
        )
        rel = (
            max((o.get("rel_above") or 0.0) for o in row.outputs.values())
            if row.outputs
            else None
        )
        inten = (
            max(
                (abs(o.get("intensity") or 0.0) for o in row.outputs.values()),
                default=None,
            )
            if row.outputs
            else None
        )
        time_s = (
            f"{_ratio(row.time_ratio)} ({_fmt(row.time_ref, '.3f')} → "
            f"{_fmt(row.time_cand, '.3f')} s)"
            if row.time_ratio
            else ""
        )
        note = row.note or (f"accepted: {row.accepted_by}" if row.accepted_by else "")
        ident_s = "" if ident is None else ("yes" if ident else "no")
        if row.kind == ATTRIBUTION:
            name = f"`{row.cand_id}` vs `{row.ref_id}`"
        else:
            name = f"`{row.cand_id or row.ref_id}`"
        lines.append(
            f"| {name} | {row.verdict} | {ident_s} | {_sci(rel)} | {_sci(inten)} "
            f"| {time_s} | {_ratio(row.cold_ratio)} | {_ratio(row.rss_ratio)} "
            f"| {' '.join(row.flags)} | {note} |"
        )
    lines.append("")
    if report.accepted:
        lines.append("## Accepted changes (changelog)")
        lines.append("")
        lines.append("| case glob | since | PR | reason | matched |")
        lines.append("|---|---|---|---|---|")
        for a in report.accepted:
            lines.append(
                f"| `{a.case}` | {a.since} | {a.pr or ''} | {a.reason} "
                f"| {len(a.matched)} |"
            )
        lines.append("")
    if report.stale_accepted:
        lines.append(
            "Stale accepted entries (match no drift): "
            + ", ".join(f"`{a.case}`" for a in report.stale_accepted)
            + "."
        )
        lines.append("")
    counts: dict[str, int] = {}
    for row in report.rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
    lines.append(
        "Summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + "."
    )
    return "\n".join(lines) + "\n"


def to_json(report: Report) -> dict[str, Any]:
    return {
        "reference": {
            "ref": report.reference["ref"],
            "preset": report.reference["preset"],
            "tier": report.reference["tier"],
            "fingerprint_short": report.reference.get("fingerprint_short"),
        },
        "candidate": {
            "ref": report.candidate["ref"],
            "preset": report.candidate["preset"],
            "tier": report.candidate["tier"],
            "fingerprint_short": report.candidate.get("fingerprint_short"),
        },
        "case_hash_match": report.case_hash_match,
        "fingerprint_match": report.fingerprint_match,
        "thresholds": report.thresholds,
        "noise": report.noise,
        "rows": [
            {
                "reference_id": r.ref_id,
                "candidate_id": r.cand_id,
                "verdict": r.verdict,
                "outputs": r.outputs,
                "time_ratio": r.time_ratio,
                "time_ref": r.time_ref,
                "time_cand": r.time_cand,
                "cold_ratio": r.cold_ratio,
                "rss_ratio": r.rss_ratio,
                "vram_ratio": r.vram_ratio,
                "flags": r.flags,
                "accepted_by": r.accepted_by,
                "note": r.note,
                "kind": r.kind,
            }
            for r in report.rows
        ],
        "accepted": [
            {
                "case": a.case,
                "since": a.since,
                "pr": a.pr,
                "reason": a.reason,
                "matched": a.matched,
            }
            for a in report.accepted
        ],
        "stale_accepted": [a.case for a in report.stale_accepted],
    }
