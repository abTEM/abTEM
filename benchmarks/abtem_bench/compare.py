"""Compare two bundles: results, speed, memory.

Never imports the refs' abtem. It reads the store format and uses
``abtem.core.testing.close_stats`` from the invoking checkout for the metric
vector.
"""

from __future__ import annotations

import math
import string
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from abtem_bench import registry, store

IDENTICAL = "IDENTICAL"
OK = "OK"
DRIFT = "DRIFT"
ACCEPTED = "ACCEPTED"
SHAPE = "SHAPE"
ONLY_A = "ONLY-A"
ONLY_B = "ONLY-B"

#: Run statuses of a case that produced no result. ``UNSUPPORTED`` is not one:
#: it is a ref lacking an API the case declares in ``requires``.
FAILED_STATUSES = (
    store.STATUS_ERROR,
    store.STATUS_OOM,
    store.STATUS_TIMEOUT,
    store.STATUS_SKIPPED_MEMORY,
)

#: Gates ``--fail-on`` accepts. Those in THRESHOLD_GATES take ``:<N>%``.
GATES = ("drift", "shape", "error", "missing", "speed", "memory")
THRESHOLD_GATES = ("speed", "memory")

#: Bounds an accepted_changes entry may set, by the close_stats key they bound.
BOUND_KEYS = {
    "max_rel": "rel_above",
    "max_intensity": "intensity",
    "max_abs_norm": "max_abs_norm",
}
ENTRY_KEYS = {"case", "since", "reason", "pr", *BOUND_KEYS}

#: Axis fields that only name or typeset an axis; differences are reported but
#: are not a change of the result. Fields starting with ``_`` are abtem's
#: internal bookkeeping and are treated the same way.
AXIS_LABEL_KEYS = ("label", "tex_label", "tex_units")
#: Axis fields that define the grid; absent from one side, they are a mismatch.
AXIS_STRUCTURAL_KEYS = ("type", "sampling", "offset", "values", "units", "endpoint")
#: Output metadata fields that only name or typeset a quantity; ignored.
METADATA_LABEL_KEYS = ("label", "units", "tex_label", "tex_units")
AXIS_RTOL = 1e-9
AXIS_ATOL = 1e-12

ACCEPTED_CHANGES_PATH = Path(__file__).resolve().parents[1] / "accepted_changes.toml"


class CompareError(RuntimeError):
    """An input problem: the comparison cannot be made as asked."""


def _within(value: float | None, bound: float) -> bool:
    """``value <= bound``; a NaN or missing value is never within a bound."""
    return value is not None and value <= bound


# ---------------------------------------------------------------------------
# accepted_changes.toml


@dataclass
class Accepted:
    """One accepted_changes entry and what it matched in this comparison.

    ``since`` scopes the entry: it applies only when the reference bundle was
    captured at that ref (its label, its ``git describe``, or a prefix of at
    least 7 hex digits of its sha). The optional bounds cap how large the
    accepted drift may be, per output; a drift beyond a bound stays ``DRIFT``.
    A non-finite value that differs between the two outputs is never accepted.
    """

    case: str
    since: str
    reason: str
    pr: int | None = None
    max_rel: float | None = None
    max_intensity: float | None = None
    max_abs_norm: float | None = None
    matched: list[str] = field(default_factory=list)
    exceeded: list[str] = field(default_factory=list)
    worst: dict[str, float] = field(default_factory=dict)

    def matches(self, case_id: str) -> bool:
        cid = registry.CaseId.parse(case_id)
        return registry.glob_match(case_id, self.case) or registry.glob_match(
            cid.name, self.case
        )

    def applies_to(self, ref: dict[str, Any]) -> bool:
        if self.since in (ref.get("label"), ref.get("describe")):
            return True
        since = self.since.lower()
        sha = str(ref.get("sha") or "").lower()
        return (
            len(since) >= 7
            and all(ch in string.hexdigits for ch in since)
            and sha.startswith(since)
        )

    def bounds(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in BOUND_KEYS if getattr(self, k) is not None}

    def violations(self, outputs: dict[str, dict[str, Any]]) -> list[str]:
        """Outputs whose drift exceeds a bound or has non-finite mismatches."""
        out = []
        for name, s in outputs.items():
            if s.get("identical"):
                continue
            if s.get("nonfinite_mismatch"):
                out.append(
                    f"{name} has {s['nonfinite_mismatch']} non-finite values "
                    "that differ"
                )
            for key, bound in self.bounds().items():
                value = s.get(BOUND_KEYS[key])
                if value is not None and key == "max_intensity":
                    value = abs(value)
                if not _within(value, bound):
                    out.append(f"{name} {key[4:]} {_sci(value)} > {_sci(bound)}")
        return out

    def record(self, case_id: str, outputs: dict[str, dict[str, Any]]) -> None:
        """Credit a matched drift and fold its size into ``worst``."""
        self.matched.append(case_id)
        for s in outputs.values():
            for key in ("rel_above", "intensity", "max_abs_norm"):
                value = s.get(key)
                if value is None:
                    continue
                value = abs(value)
                prev = self.worst.get(key)
                if prev is None or not value <= prev:  # NaN wins
                    self.worst[key] = value


def _all_case_ids(reg: dict[str, registry.Case]) -> list[str]:
    return [str(cid) for c in reg.values() for tier in c.tiers for cid in c.ids(tier)]


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def load_accepted(
    path: Path | None = None, reg: dict[str, registry.Case] | None = None
) -> list[Accepted]:
    """Read and validate accepted_changes.toml.

    Every entry needs a non-empty ``case``, ``since`` and ``reason``; ``pr`` is
    a positive integer; bounds are positive finite numbers; unknown keys are
    refused (a misspelt bound would otherwise be silently unbounded).
    ``max_abs_norm`` is required: the integrated intensity is unchanged by a
    shift, a flip or a phase scramble of the output, so only a bound on the
    largest elementwise difference rejects a corrupted result. With a
    registry, the ``case`` glob must match at least one registered case id of
    any tier, device or variant.

    ``path=None`` reads the shipped file and returns no entries when it is
    absent; an explicit ``path`` must exist.
    """
    if path is None:
        path = ACCEPTED_CHANGES_PATH
        if not path.exists():
            return []
    elif not path.exists():
        raise CompareError(f"accepted changes file not found: {path}")
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise CompareError(f"{path.name}: {exc}") from exc
    extra = set(data) - {"accepted"}
    if extra:
        raise CompareError(f"{path.name}: unknown top-level keys {sorted(extra)}")
    known_ids = _all_case_ids(reg) if reg else None
    out = []
    entries = data.get("accepted", [])
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise CompareError(
            f"{path.name}: 'accepted' must be an array of tables ([[accepted]])"
        )
    for i, entry in enumerate(entries):
        where = f"{path.name}: entry {i}"
        unknown = set(entry) - ENTRY_KEYS
        if unknown:
            raise CompareError(f"{where}: unknown keys {sorted(unknown)}")
        for key in ("case", "since", "reason"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise CompareError(f"{where}: {key!r} must be a non-empty string")
        pr = entry.get("pr")
        if pr is not None and not (
            isinstance(pr, int) and not isinstance(pr, bool) and pr > 0
        ):
            raise CompareError(f"{where}: 'pr' must be a pull request number")
        for key in BOUND_KEYS:
            if key in entry and not _number(entry[key]):
                raise CompareError(f"{where}: {key!r} must be a positive number")
        if "max_abs_norm" not in entry:
            raise CompareError(
                f"{where}: 'max_abs_norm' is required: it is the bound a shifted, "
                "flipped or rescaled output cannot pass"
            )
        acc = Accepted(
            entry["case"].strip(), entry["since"].strip(), entry["reason"].strip(), pr
        )
        for key in BOUND_KEYS:
            if key in entry:
                setattr(acc, key, float(entry[key]))
        if known_ids is not None and not any(acc.matches(c) for c in known_ids):
            raise CompareError(
                f"{where}: case glob {acc.case!r} matches no registered case"
            )
        out.append(acc)
    return out


# ---------------------------------------------------------------------------
# --fail-on


def parse_fail_on(text: str | None) -> dict[str, float | None]:
    """``"drift,speed:10%"`` -> ``{"drift": None, "speed": 0.1}``.

    A gate's value is its threshold as a fraction, or None for the compare
    default. Unknown gates and malformed thresholds raise CompareError, and so
    does a given text that names no gate; None means no gates.
    """
    gates: dict[str, float | None] = {}
    if text is None:
        return gates
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        name, sep, value = (part.strip() for part in token.partition(":"))
        if name not in GATES:
            raise CompareError(
                f"--fail-on: unknown gate {name!r}; known: {', '.join(GATES)}"
            )
        if not sep:
            gates[name] = None
            continue
        if name not in THRESHOLD_GATES:
            raise CompareError(f"--fail-on: {name!r} takes no threshold")
        try:
            pct = float(value[:-1]) if value.endswith("%") else math.nan
        except ValueError:
            pct = math.nan
        if not (math.isfinite(pct) and pct > 0):
            raise CompareError(
                f"--fail-on: threshold {value!r} for {name!r} must be a positive "
                f"percentage, e.g. {name}:10%"
            )
        gates[name] = pct / 100
    if not gates:
        raise CompareError("--fail-on names no gate")
    return gates


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


def _close(x: Any, y: Any) -> bool:
    if isinstance(x, bool) or isinstance(y, bool):
        return x == y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        if math.isnan(x) and math.isnan(y):
            return True
        return math.isclose(x, y, rel_tol=AXIS_RTOL, abs_tol=AXIS_ATOL)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_close(p, q) for p, q in zip(x, y))
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_close(x[k], y[k]) for k in x)
    return x == y


def _short_repr(value: Any, limit: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def axes_diff(ref_axes: list, cand_axes: list) -> tuple[list[str], list[str]]:
    """(mismatches, informational differences) between two serialised axes.

    Numbers are compared to ``AXIS_RTOL`` and ``AXIS_ATOL``, NaN equal to NaN;
    other values exactly. A field in ``AXIS_STRUCTURAL_KEYS`` present on one
    side only is a mismatch. Labels, internal ``_`` fields and other fields
    present on one side only are informational.
    """
    if len(ref_axes) != len(cand_axes):
        return [f"{len(cand_axes)} axes, reference has {len(ref_axes)}"], []
    bad: list[str] = []
    info: list[str] = []
    for i, (a, b) in enumerate(zip(ref_axes, cand_axes)):
        for key in sorted(set(a) | set(b)):
            where = f"axis {i} {key}"
            if key not in a or key not in b:
                side = "candidate" if key in b else "reference"
                text = f"{where} only in the {side}"
                (bad if key in AXIS_STRUCTURAL_KEYS else info).append(text)
            elif key in AXIS_LABEL_KEYS or key.startswith("_"):
                if a[key] != b[key]:
                    info.append(
                        f"{where} {_short_repr(a[key])} → {_short_repr(b[key])}"
                    )
            elif not _close(a[key], b[key]):
                bad.append(f"{where} {_short_repr(a[key])} → {_short_repr(b[key])}")
    return bad, info


def metadata_diff(ref_meta: dict, cand_meta: dict) -> list[str]:
    """Differences between two outputs' metadata, by the rules of ``axes_diff``.

    Label fields and fields starting with ``_`` are left out. The result is
    informational: it never changes a verdict.
    """
    diffs: list[str] = []
    for key in sorted(set(ref_meta) | set(cand_meta)):
        if key in METADATA_LABEL_KEYS or key.startswith("_"):
            continue
        if key not in ref_meta or key not in cand_meta:
            side = "candidate" if key in cand_meta else "reference"
            diffs.append(f"{key} only in the {side}")
        elif not _close(ref_meta[key], cand_meta[key]):
            diffs.append(
                f"{key} {_short_repr(ref_meta[key])} → {_short_repr(cand_meta[key])}"
            )
    return diffs


def output_stats(
    ref_arr: np.ndarray,
    cand_arr: np.ndarray,
    ref_axes: list,
    cand_axes: list,
    above_rel: float,
    ref_meta: dict | None = None,
    cand_meta: dict | None = None,
) -> dict[str, Any]:
    from abtem.core.testing import close_stats

    s = close_stats(cand_arr, ref_arr, above_rel=above_rel)
    s["dtype_ok"] = str(cand_arr.dtype) == str(ref_arr.dtype)
    s["shape_ok"] = tuple(cand_arr.shape) == tuple(ref_arr.shape)
    bad, info = axes_diff(ref_axes, cand_axes)
    s["axes_ok"] = not bad
    notes = []
    if not s["shape_ok"]:
        notes.append(f"shape {tuple(ref_arr.shape)} → {tuple(cand_arr.shape)}")
    if not s["dtype_ok"]:
        notes.append(f"dtype {ref_arr.dtype} → {cand_arr.dtype}")
    notes.extend(bad)
    s["mismatch"] = notes
    s["axes_info"] = info
    s["metadata_info"] = metadata_diff(ref_meta or {}, cand_meta or {})
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


def verdict_for(stats: dict[str, Any], tol: registry.Tolerance) -> str:
    """Verdict of one output; any NaN in the vector counts as beyond tolerance."""
    if not (stats.get("shape_ok") and stats.get("dtype_ok") and stats.get("axes_ok")):
        return SHAPE
    if stats.get("identical"):
        return IDENTICAL
    rel = stats.get("rel_above")
    rel_ok = stats.get("n_checked", 1) == 0 or _within(rel, tol.rel)
    inten = stats.get("intensity")
    inten_ok = _within(None if inten is None else abs(inten), tol.intensity)
    mabs_ok = _within(stats.get("max_abs_norm"), tol.max_abs_norm)
    return OK if (rel_ok and inten_ok and mabs_ok) else DRIFT


def widened(
    tol: registry.Tolerance, floor: dict[str, float] | None
) -> registry.Tolerance:
    """The tolerance raised to three times a self-check's accuracy spread."""
    if not floor:
        return tol
    return replace(
        tol,
        rel=max(tol.rel, 3 * floor.get("accuracy", 0.0)),
        intensity=max(tol.intensity, 3 * floor.get("accuracy_intensity", 0.0)),
        max_abs_norm=max(tol.max_abs_norm, 3 * floor.get("accuracy_abs", 0.0)),
    )


# ---------------------------------------------------------------------------
# noise floor


def check_noise(noise: Any, source: str) -> dict[str, dict[str, float]]:
    """``noise`` if it is a dict of case id -> dict of numbers, else CompareError."""
    ok = isinstance(noise, dict) and all(
        isinstance(cid, str)
        and isinstance(floor, dict)
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in floor.values()
        )
        for cid, floor in noise.items()
    )
    if not ok:
        raise CompareError(
            f"{source}: a noise file maps case ids to dicts of numbers "
            '({"<case id>": {"memory": 0.01, ...}})'
        )
    return noise


def _spread(a: Any, b: Any) -> float | None:
    if not a or not b:
        return None
    return abs(a / b - 1.0)


def noise_floor(
    bundle_a: store.Bundle, bundle_b: store.Bundle
) -> dict[str, dict[str, float]]:
    """Per-case spreads between two captures of one ref (a self-check).

    ``speed``, ``memory`` and ``vram`` are ``|a / b - 1|`` of the warm median,
    the peak RSS and the peak CuPy pool usage, present when both captures
    measured them. ``accuracy``, ``accuracy_abs`` and ``accuracy_intensity``
    are the largest relative error, normalised maximum difference and
    integrated-intensity change over the case's outputs (0 when the two
    captures are bit-identical); non-finite values are left out.
    """
    for b in (bundle_a, bundle_b):
        if not b.exists():
            raise CompareError(f"not a bundle: {b.path}")
    floors: dict[str, dict[str, float]] = {}
    for cid in bundle_a.case_ids():
        if not bundle_b.has_case(cid):
            continue
        ra, rb = bundle_a.read_case(cid), bundle_b.read_case(cid)
        if ra.get("status") != store.STATUS_OK or rb.get("status") != store.STATUS_OK:
            continue
        f: dict[str, float] = {}
        mem_a, mem_b = ra.get("memory", {}), rb.get("memory", {})
        same_meter = rss_meters(mem_a, mem_b) is None
        for key, x, y in (
            ("speed", ra["timings"].get("median"), rb["timings"].get("median")),
            (
                "memory",
                mem_a.get("peak_rss_bytes") if same_meter else None,
                mem_b.get("peak_rss_bytes") if same_meter else None,
            ),
            (
                "vram",
                mem_a.get("peak_vram_pool_bytes"),
                mem_b.get("peak_vram_pool_bytes"),
            ),
        ):
            spread = _spread(x, y)
            if spread is not None:
                f[key] = spread
        acc = {"accuracy": 0.0, "accuracy_abs": 0.0, "accuracy_intensity": 0.0}
        for name in ra.get("outputs", {}):
            if name not in rb.get("outputs", {}):
                continue
            a, ax_a, _ = bundle_a.load_output(cid, name)
            b, ax_b, _ = bundle_b.load_output(cid, name)
            s = output_stats(a, b, ax_a, ax_b, 1e-6)
            for key, stat in (
                ("accuracy", "rel_above"),
                ("accuracy_abs", "max_abs_norm"),
                ("accuracy_intensity", "intensity"),
            ):
                value = s.get(stat)
                if value is not None and math.isfinite(value):
                    acc[key] = max(acc[key], abs(value))
        f.update(acc)
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
    accepted_by: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    kind: str = SAME
    #: which side's run failed: "reference", "candidate", "both" or None
    failed: str | None = None
    #: whether speed and memory ratios of this row may be flagged at all
    flaggable: bool = False
    floors: dict[str, float] = field(default_factory=dict)
    #: run status of each side's record, None where the side has no such id
    ref_status: str | None = None
    cand_status: str | None = None

    @property
    def case(self) -> str:
        return str(self.cand_id or self.ref_id)


def _speed_flag(row: Row, threshold: float, min_delta: float) -> str | None:
    """``speed`` beyond threshold, ``short`` when the time difference is tiny."""
    if not row.flaggable or row.time_ratio is None:
        return None
    if row.time_ref is None or row.time_cand is None:
        return None
    if abs(row.time_ratio - 1.0) <= max(threshold, 3 * row.floors.get("speed", 0.0)):
        return None
    if abs(row.time_cand - row.time_ref) < min_delta:
        return "short"
    return "speed"


def _ratio_flagged(
    row: Row, ratio: float | None, floor_key: str, threshold: float
) -> bool:
    """Beyond ``max(threshold, 3 x floor)``; never without a floor."""
    floor = row.floors.get(floor_key)
    if not row.flaggable or ratio is None or floor is None:
        return False
    return abs(ratio - 1.0) > max(threshold, 3 * floor)


@dataclass
class Report:
    reference: dict[str, Any]
    candidate: dict[str, Any]
    rows: list[Row]
    accepted: list[Accepted]
    not_applicable: list[Accepted]
    stale_accepted: list[Accepted]
    case_hash_match: bool
    preset_match: bool
    fingerprint_match: bool
    thresholds: dict[str, float]
    noise: dict[str, dict[str, float]]
    #: the accepted_changes.toml read (or looked for), and whether it exists
    accepted_path: Path | None = None
    accepted_exists: bool = True
    #: paired cases whose memory was not judged for lack of a noise floor entry
    memory_unjudged: int = 0

    def failures(self, gates: dict[str, float | None]) -> list[str]:
        """Messages for every gate that fails; empty when the comparison passes.

        ``speed`` and ``memory`` fail on increases only; a decrease beyond the
        threshold is flagged in the report but passes.
        """
        if "memory" in gates and not self.noise:
            raise CompareError("--fail-on memory needs a noise floor (--noise)")
        t = self.thresholds
        out: list[str] = []
        errored: set[str] = set()
        for r in self.rows:
            if "missing" in gates and r.verdict == ONLY_A:
                out.append(f"{r.ref_id}: missing from the candidate")
            if (
                "missing" in gates
                and r.kind == SAME
                and r.cand_status == store.STATUS_UNSUPPORTED
                and r.ref_status == store.STATUS_OK
            ):
                out.append(
                    f"{r.case}: UNSUPPORTED on the candidate, OK on the reference"
                )
            if r.kind == SAME and r.verdict == DRIFT and "drift" in gates:
                out.append(f"{r.case}: DRIFT")
            if r.kind == SAME and r.verdict == SHAPE and "shape" in gates:
                out.append(f"{r.case}: SHAPE")
            if (
                "error" in gates
                and r.failed in ("candidate", "both")
                and r.case not in errored
            ):
                errored.add(r.case)
                out.append(
                    f"{r.case}: {r.cand_status if r.verdict == ONLY_B else r.verdict}"
                )
            if "speed" in gates and r.time_ratio is not None and r.time_ratio > 1:
                limit = gates["speed"] if gates["speed"] is not None else t["speed"]
                if _speed_flag(r, limit, t["min_delta"]) == "speed":
                    out.append(f"{r.case}: time x{r.time_ratio:.2f}")
            if "memory" in gates and r.rss_ratio is not None and r.rss_ratio > 1:
                limit = gates["memory"] if gates["memory"] is not None else t["memory"]
                if _ratio_flagged(r, r.rss_ratio, "memory", limit):
                    out.append(f"{r.case}: rss x{r.rss_ratio:.2f}")
        return out


def _ratio_of(a: Any, b: Any) -> float | None:
    return b / a if a and b else None


def rss_meters(mem_a: dict[str, Any], mem_b: dict[str, Any]) -> tuple[str, str] | None:
    """The two meters of ``peak_rss_bytes`` when they differ and both are present.

    A record without ``rss_meter`` comes from a harness revision that stored the
    ``os.wait4`` figure.
    """
    if not (mem_a.get("peak_rss_bytes") and mem_b.get("peak_rss_bytes")):
        return None
    meters = (mem_a.get("rss_meter", "wait4"), mem_b.get("rss_meter", "wait4"))
    return meters if meters[0] != meters[1] else None


def _pair_row(
    reference: store.Bundle,
    candidate: store.Bundle,
    ref_id: str,
    cand_id: str,
    kind: str,
    reg: dict[str, registry.Case] | None,
    accepted: list[Accepted],
    noise: dict[str, dict[str, float]],
    thresholds: dict[str, float],
) -> Row:
    rr, rc = reference.read_case(ref_id), candidate.read_case(cand_id)
    row = Row(
        ref_id,
        cand_id,
        IDENTICAL,
        kind=kind,
        ref_status=rr["status"],
        cand_status=rc["status"],
    )
    if kind == ATTRIBUTION:
        row.notes.append(f"attribution: vs reference `{ref_id}`")
    if rr["status"] != store.STATUS_OK or rc["status"] != store.STATUS_OK:
        row.verdict = rc["status"] if rc["status"] != store.STATUS_OK else rr["status"]
        bad_ref = rr["status"] in FAILED_STATUSES
        bad_cand = rc["status"] in FAILED_STATUSES
        row.failed = (
            "both"
            if bad_ref and bad_cand
            else "candidate"
            if bad_cand
            else "reference"
            if bad_ref
            else None
        )
        for side, rec in (("reference", rr), ("candidate", rc)):
            note = _error_note(rec)
            if rec["status"] != store.STATUS_OK:
                row.notes.append(
                    f"{side} {rec['status']}" + (f": {note}" if note else "")
                )
        return row

    cid = registry.CaseId.parse(cand_id)
    case = reg.get(cid.name) if reg else None
    row.floors = dict(noise.get(cand_id, {}))
    tol = widened(case.tolerance if case else registry.Tolerance(), row.floors)
    row.flaggable = kind == SAME and (
        case is None
        or cid.variant not in case.variants
        or case.variants[cid.variant].flag
    )

    order = {IDENTICAL: 0, OK: 1, DRIFT: 2, SHAPE: 3}
    worst = IDENTICAL
    ref_outputs, cand_outputs = rr.get("outputs", {}), rc.get("outputs", {})
    for name in ref_outputs:
        if name not in cand_outputs:
            row.outputs[name] = {"verdict": SHAPE, "mismatch": ["missing"]}
            row.notes.append(f"{name}: missing in the candidate")
            worst = SHAPE
            continue
        a, ax_a, meta_a = reference.load_output(ref_id, name)
        b, ax_b, meta_b = candidate.load_output(cand_id, name)
        s = output_stats(a, b, ax_a, ax_b, tol.above_rel, meta_a, meta_b)
        s["verdict"] = verdict_for(s, tol)
        row.outputs[name] = s
        row.notes.extend(f"{name}: {m}" for m in s["mismatch"])
        if s["axes_info"]:
            row.notes.append(f"{name}: axis metadata differs ({len(s['axes_info'])})")
        if s["metadata_info"]:
            row.notes.append(f"{name}: metadata differs ({len(s['metadata_info'])})")
        if order[s["verdict"]] > order[worst]:
            worst = s["verdict"]
    new = [n for n in cand_outputs if n not in ref_outputs]
    if new:
        row.notes.append("new outputs, not compared: " + ", ".join(new))
    row.verdict = worst

    if worst == DRIFT and kind == SAME:
        matching = [a for a in accepted if a.matches(cand_id)]
        over = [v for a in matching for v in a.violations(row.outputs)]
        for a in matching:
            a.record(cand_id, row.outputs)
            if a.violations(row.outputs):
                a.exceeded.append(cand_id)
        if matching and not over:
            row.verdict = ACCEPTED
            row.accepted_by = [a.reason for a in matching]
        elif over:
            row.notes.append("exceeds accepted bound: " + "; ".join(over))

    tr, tc = rr["timings"].get("median"), rc["timings"].get("median")
    if tr and tc:
        row.time_ref, row.time_cand, row.time_ratio = tr, tc, tc / tr
        flag = _speed_flag(row, thresholds["speed"], thresholds["min_delta"])
        if flag:
            row.flags.append(flag)
    row.cold_ratio = _ratio_of(rr["timings"].get("cold"), rc["timings"].get("cold"))
    mem_r, mem_c = rr.get("memory", {}), rc.get("memory", {})
    meters = rss_meters(mem_r, mem_c)
    if meters:
        row.notes.append(
            "peak RSS measured with different meters "
            f"({meters[0]} vs {meters[1]}); not compared"
        )
    else:
        row.rss_ratio = _ratio_of(
            mem_r.get("peak_rss_bytes"), mem_c.get("peak_rss_bytes")
        )
    if _ratio_flagged(row, row.rss_ratio, "memory", thresholds["memory"]):
        row.flags.append("memory")
    row.vram_ratio = _ratio_of(
        mem_r.get("peak_vram_pool_bytes"), mem_c.get("peak_vram_pool_bytes")
    )
    if _ratio_flagged(row, row.vram_ratio, "vram", thresholds["memory"]):
        row.flags.append("vram")
    return row


def _single_row(
    bundle: store.Bundle, ref_id: str | None, cand_id: str | None, verdict: str
) -> Row:
    """A row for an id only one bundle holds; a failed run there is noted."""
    row = Row(ref_id, cand_id, verdict)
    side = "reference" if cand_id is None else "candidate"
    rec = bundle.read_case(str(ref_id or cand_id))
    status = rec["status"]
    if side == "reference":
        row.ref_status = status
    else:
        row.cand_status = status
    if status in FAILED_STATUSES:
        row.failed = side
        note = _error_note(rec)
        row.notes.append(f"{side} {status}" + (f": {note}" if note else ""))
    return row


def compare(
    reference: store.Bundle,
    candidate: store.Bundle,
    reg: dict[str, registry.Case] | None = None,
    accepted_path: Path | None = None,
    noise: dict[str, dict[str, float]] | None = None,
    speed_threshold: float = 0.10,
    memory_threshold: float = 0.05,
    min_delta: float = 0.05,
    allow_case_mismatch: bool = False,
    allow_preset_mismatch: bool = False,
) -> Report:
    """Compare ``candidate`` against ``reference``.

    Raises CompareError when the bundles ran different case code or presets
    (unless allowed), share no case id, or accepted_changes.toml is invalid.
    ``accepted_path=None`` reads the shipped accepted_changes.toml and accepts
    nothing when it is absent (the report says so); an explicit path must
    exist.
    Speed ratios beyond ``max(speed_threshold, 3 x floor)`` are flagged, or
    marked ``short`` when the two medians differ by less than ``min_delta``
    seconds. Memory and VRAM ratios are flagged only for cases with a noise
    floor.
    """
    man_r, man_c = reference.read_manifest(), candidate.read_manifest()
    hash_match = man_r.get("case_hash") == man_c.get("case_hash")
    if not hash_match and not allow_case_mismatch:
        raise CompareError(
            "case_hash differs between the bundles: the case code was not the same; "
            "pass --allow-case-mismatch to compare anyway"
        )
    preset_match = man_r.get("preset") == man_c.get("preset")
    if not preset_match and not allow_preset_mismatch:
        raise CompareError(
            f"presets differ (reference {man_r.get('preset')!r}, candidate "
            f"{man_c.get('preset')!r}); pass --allow-preset-mismatch to compare "
            "anyway"
        )
    fp_match = man_r.get("fingerprint_short") == man_c.get("fingerprint_short")
    entries = load_accepted(accepted_path, reg)
    used_path = accepted_path if accepted_path is not None else ACCEPTED_CHANGES_PATH
    ref_info = man_r.get("ref", {})
    accepted = [a for a in entries if a.applies_to(ref_info)]
    not_applicable = [a for a in entries if not a.applies_to(ref_info)]
    noise = noise or {}
    thresholds = {
        "speed": speed_threshold,
        "memory": memory_threshold,
        "min_delta": min_delta,
    }

    pairs = pair_ids(reference, candidate, reg)
    if not any(r is not None and c is not None for r, c, _ in pairs):
        raise CompareError(
            "the bundles share no case id (reference: tier "
            f"{man_r.get('tier')!r}, devices {man_r.get('devices')}; candidate: "
            f"tier {man_c.get('tier')!r}, devices {man_c.get('devices')})"
        )
    rows: list[Row] = []
    for ref_id, cand_id, kind in pairs:
        if ref_id is None:
            rows.append(_single_row(candidate, None, cand_id, ONLY_B))
        elif cand_id is None:
            rows.append(_single_row(reference, ref_id, None, ONLY_A))
        else:
            rows.append(
                _pair_row(
                    reference,
                    candidate,
                    ref_id,
                    cand_id,
                    kind,
                    reg,
                    accepted,
                    noise,
                    thresholds,
                )
            )

    unjudged = sum(1 for r in rows if r.flaggable and "memory" not in r.floors)
    return Report(
        reference=man_r,
        candidate=man_c,
        rows=rows,
        accepted=accepted,
        not_applicable=not_applicable,
        stale_accepted=[a for a in accepted if not a.matched],
        case_hash_match=hash_match,
        preset_match=preset_match,
        fingerprint_match=fp_match,
        thresholds=thresholds,
        noise=noise,
        accepted_path=used_path,
        accepted_exists=used_path.exists(),
        memory_unjudged=unjudged if noise else 0,
    )


# ---------------------------------------------------------------------------
# rendering


def _fmt(x: float | None, spec: str = ".2f", none: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return none
    return format(x, spec)


def _sci(x: float | None) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return "0" if x == 0 else f"{x:.1e}"


def _ratio(x: float | None) -> str:
    if x is None:
        return ""
    sign = "▲" if x > 1.02 else ("▼" if x < 0.98 else "≈")
    return f"{sign}{x:.2f}"


def _cell(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _worst(row: Row, key: str) -> str:
    """Largest ``|stat|`` over a row's outputs, naming the output if several."""
    best: tuple[float, str] | None = None
    for name, s in row.outputs.items():
        value = s.get(key)
        if value is None:
            continue
        value = abs(value)
        if best is None or math.isnan(value) or value > best[0]:
            best = (value, name)
            if math.isnan(value):
                break
    if best is None:
        return ""
    text = _sci(best[0])
    if len(row.outputs) > 1 and best[0] != 0:
        text += f" ({best[1]})"
    return text


def _bounds_text(a: Accepted) -> str:
    return ", ".join(f"{k[4:]} ≤ {_sci(v)}" for k, v in a.bounds().items()) or "none"


def _measured_text(a: Accepted) -> str:
    names = (
        ("rel_above", "rel"),
        ("intensity", "intensity"),
        ("max_abs_norm", "max_abs"),
    )
    return ", ".join(
        f"{label} {_sci(a.worst[k])}" for k, label in names if k in a.worst
    )


def to_markdown(report: Report) -> str:
    r, c = report.reference, report.candidate
    lines = []
    lines.append("# abtem-bench comparison")
    lines.append("")
    for label, m in (("Reference", r), ("Candidate", c)):
        ref = m.get("ref") or {}
        host = (m.get("fingerprint") or {}).get("hostname", "?")
        lines.append(
            f"{label}: `{ref.get('label', '?')}` ({ref.get('describe', '?')}, "
            f"{str(ref.get('sha', '?'))[:12]}), abtem {m.get('abtem_version', '?')}, "
            f"preset `{m.get('preset', '?')}`, tier `{m.get('tier', '?')}`, "
            f"captured {m.get('timestamp_utc', '?')} on {host} "
            f"(fingerprint {m.get('fingerprint_short', '?')})."
        )
    t = report.thresholds
    notes = []
    if not report.case_hash_match:
        notes.append(
            "case_hash differs: the two bundles did not run the same case code"
        )
    if not report.preset_match:
        notes.append("presets differ: the two bundles ran under different settings")
    if not report.accepted_exists:
        notes.append(
            f"no accepted_changes.toml at {report.accepted_path}; no drift is accepted"
        )
    if not report.fingerprint_match:
        notes.append(
            "machine fingerprints differ: bit-identity is not expected, "
            "tolerances apply"
        )
    if report.noise:
        notes.append(
            f"noise floor from a self-check of {len(report.noise)} cases: speed, "
            "memory and VRAM flags use max(threshold, 3 x floor), and accuracy "
            "tolerances widen to 3 x the self-check's spread"
        )
        if report.memory_unjudged:
            notes.append(
                f"memory not judged for {report.memory_unjudged} case ids without "
                "a floor in the noise file"
            )
    else:
        notes.append(
            "no noise floor: memory and VRAM ratios are shown but never flagged"
        )
    for n in notes:
        lines.append(f"Note: {n}.")
    lines.append("")
    lines.append(
        "Rows: one per paired case id; attribution rows (`x[variant]` vs `x`) "
        "compare a candidate variant against the reference default and are "
        "informational. Columns: verdict over all outputs; `identical` = every "
        "output bit-for-bit equal; `rel` = largest relative error over elements "
        "above the case's `above_rel` fraction of the reference maximum; "
        "`intensity` = largest relative change of the integrated intensity; both "
        "name the output when the case has several. `time` = candidate/reference "
        "warm median with the two medians in seconds; `cold` = "
        "candidate/reference first-call time; `rss` = candidate/reference peak "
        "resident memory of the worker process; `vram` = candidate/reference peak "
        "CuPy pool usage (GPU cases). Flags mark ratios beyond threshold "
        f"(speed {t['speed']:.0%}, memory {t['memory']:.0%}); `short` marks a time "
        f"ratio beyond threshold whose medians differ by less than "
        f"{t['min_delta']:g} s."
    )
    lines.append("")
    lines.append(
        "| case | verdict | identical | rel | intensity | time | cold | rss | vram "
        "| flags | note |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in report.rows:
        ident = (
            all(o.get("identical") for o in row.outputs.values())
            if row.outputs
            else None
        )
        time_s = (
            f"{_ratio(row.time_ratio)} ({_fmt(row.time_ref, '.3f')} → "
            f"{_fmt(row.time_cand, '.3f')} s)"
            if row.time_ratio
            else ""
        )
        notes_row = list(row.notes)
        if row.accepted_by:
            notes_row.append("accepted: " + " / ".join(row.accepted_by))
        ident_s = "" if ident is None else ("yes" if ident else "no")
        if row.kind == ATTRIBUTION:
            name = f"`{row.cand_id}` vs `{row.ref_id}`"
        else:
            name = f"`{row.case}`"
        lines.append(
            f"| {name} | {row.verdict} | {ident_s} | {_cell(_worst(row, 'rel_above'))} "
            f"| {_cell(_worst(row, 'intensity'))} | {time_s} "
            f"| {_ratio(row.cold_ratio)} | {_ratio(row.rss_ratio)} "
            f"| {_ratio(row.vram_ratio)} | {' '.join(row.flags)} "
            f"| {_cell('; '.join(notes_row))} |"
        )
    lines.append("")
    if report.accepted:
        lines.append("## Accepted changes (changelog)")
        lines.append("")
        lines.append(
            "Rows: accepted_changes.toml entries that apply to this reference. "
            "`bounds` = the largest drift the entry accepts per output; `matched` = "
            "drifting case ids the entry covers (in brackets: how many exceed a "
            "bound and stay DRIFT); `measured` = the largest drift over those ids."
        )
        lines.append("")
        lines.append(
            "| case glob | since | PR | reason | bounds | matched | measured |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for a in report.accepted:
            matched = str(len(a.matched))
            if a.exceeded:
                matched += f" ({len(a.exceeded)} over bound)"
            lines.append(
                f"| `{_cell(a.case)}` | {_cell(a.since)} | {a.pr or ''} "
                f"| {_cell(a.reason)} | {_bounds_text(a)} | {matched} "
                f"| {_measured_text(a)} |"
            )
        lines.append("")
    if report.stale_accepted:
        lines.append(
            "Stale accepted entries (apply to this reference but match no drift): "
            + ", ".join(f"`{a.case}`" for a in report.stale_accepted)
            + "."
        )
        lines.append("")
    if report.not_applicable:
        lines.append(
            "Accepted entries for other references (not applied): "
            + ", ".join(f"`{a.case}` since {a.since}" for a in report.not_applicable)
            + "."
        )
        lines.append("")
    summary = "Summary: " + _counts(r for r in report.rows if r.kind == SAME) + "."
    attribution = [r for r in report.rows if r.kind == ATTRIBUTION]
    if attribution:
        summary += f" Attribution rows: {_counts(attribution)}."
    lines.append(summary)
    return "\n".join(lines) + "\n"


def _counts(rows: Iterable[Row]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


def _entry_json(a: Accepted) -> dict[str, Any]:
    return {
        "case": a.case,
        "since": a.since,
        "pr": a.pr,
        "reason": a.reason,
        "bounds": a.bounds(),
        "matched": a.matched,
        "exceeded": a.exceeded,
        "measured": a.worst,
    }


def to_json(report: Report) -> dict[str, Any]:
    def side(m: dict[str, Any]) -> dict[str, Any]:
        return {
            "ref": m.get("ref"),
            "preset": m.get("preset"),
            "tier": m.get("tier"),
            "fingerprint_short": m.get("fingerprint_short"),
        }

    return {
        "reference": side(report.reference),
        "candidate": side(report.candidate),
        "case_hash_match": report.case_hash_match,
        "preset_match": report.preset_match,
        "fingerprint_match": report.fingerprint_match,
        "thresholds": report.thresholds,
        "noise": report.noise,
        "accepted_path": None
        if report.accepted_path is None
        else str(report.accepted_path),
        "accepted_exists": report.accepted_exists,
        "memory_unjudged": report.memory_unjudged,
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
                "notes": r.notes,
                "kind": r.kind,
                "failed": r.failed,
            }
            for r in report.rows
        ],
        "accepted": [_entry_json(a) for a in report.accepted],
        "not_applicable": [_entry_json(a) for a in report.not_applicable],
        "stale_accepted": [a.case for a in report.stale_accepted],
    }
