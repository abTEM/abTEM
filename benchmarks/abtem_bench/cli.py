"""Command line: list, capture, self-check, compare."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from abtem_bench import compare as cmp
from abtem_bench import prepare, presets, registry, runner, store


def _devices(text: str) -> list[str]:
    out = [d.strip() for d in text.split(",") if d.strip()]
    for d in out:
        if d not in registry.DEVICES:
            raise argparse.ArgumentTypeError(f"unknown device {d!r}")
    return out


def _add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tier", default="quick", choices=registry.TIER_NAMES)
    p.add_argument(
        "--device", type=_devices, default=["cpu"], help="comma-separated: cpu,gpu"
    )
    p.add_argument(
        "--only", action="append", default=[], help="case id or name glob (repeatable)"
    )
    p.add_argument(
        "--tag",
        action="append",
        default=[],
        help="restrict to cases carrying this tag (repeatable)",
    )


def cmd_list(args: argparse.Namespace) -> int:
    reg = registry.load_cases()
    ids = registry.select_ids(reg, args.tier, args.device, args.only, args.tag)
    print(f"case_hash {registry.case_hash()}")
    for cid in ids:
        c = reg[cid.name]
        print(f"{cid!s:<50} tags={','.join(sorted(c.tags))}")
    print(f"{len(ids)} case ids")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    reg = registry.load_cases()
    ids = registry.select_ids(reg, args.tier, args.device, args.only, args.tag)
    if not ids:
        print("no cases selected", file=sys.stderr)
        return 2
    repo = runner.find_repo()
    bundles = runner.capture(
        repo,
        [args.ref],
        ids,
        Path(args.out).parent,
        args.preset,
        args.tier,
        args.device,
        args.repeats,
        rounds=args.rounds,
        labels=[Path(args.out).name],
        command=shlex.join(sys.argv),
    )
    print(f"bundle: {bundles[0].path}")
    return 0


def cmd_self_check(args: argparse.Namespace) -> int:
    reg = registry.load_cases()
    ids = registry.select_ids(reg, args.tier, args.device, args.only, args.tag)
    if not ids:
        print("no cases selected", file=sys.stderr)
        return 2
    repo = runner.find_repo()
    out = Path(args.out)
    bundles = runner.capture(
        repo,
        [args.ref, args.ref],
        ids,
        out,
        args.preset,
        args.tier,
        args.device,
        args.repeats,
        rounds=args.rounds,
        labels=["a", "b"],
        command=shlex.join(sys.argv),
    )
    floors = cmp.noise_floor(bundles[0], bundles[1])
    store.dump_json(out / "noise.json", floors)
    report = cmp.compare(bundles[0], bundles[1], reg)
    md = cmp.to_markdown(report)
    (out / "self_check.md").write_text(md)
    print(md)
    not_identical = [
        str(r.cand_id)
        for r in report.rows
        if r.kind == cmp.SAME and r.cand_id and r.verdict != cmp.IDENTICAL
    ]
    if not_identical:
        print(
            f"self-check: {len(not_identical)} case(s) not bit-identical: "
            + ", ".join(not_identical)
        )
        return (
            1
            if args.preset == "accuracy"
            and "cpu" in args.device
            and args.device == ["cpu"]
            else 0
        )
    print("self-check: all cases bit-identical")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    reg = registry.load_cases()
    reference, candidate = store.Bundle(args.reference), store.Bundle(args.candidate)
    for b in (reference, candidate):
        if not b.exists():
            print(f"not a bundle: {b.path}", file=sys.stderr)
            return 2
    noise = None
    if args.noise:
        npath = Path(args.noise)
        if npath.is_dir():
            noise = cmp.noise_floor(
                store.Bundle(npath / "a"), store.Bundle(npath / "b")
            )
        else:
            noise = store.load_json(npath)
    report = cmp.compare(
        reference,
        candidate,
        reg,
        accepted_path=Path(args.accepted) if args.accepted else None,
        noise=noise,
        speed_threshold=args.speed_threshold,
        memory_threshold=args.memory_threshold,
        allow_case_mismatch=args.allow_case_mismatch,
        min_time=args.min_time,
    )
    md = cmp.to_markdown(report)
    if args.md:
        Path(args.md).write_text(md)
    if args.json:
        store.dump_json(Path(args.json), cmp.to_json(report))
    print(md)
    fail_on = [f.strip() for f in (args.fail_on or "").split(",") if f.strip()]
    failures = report.failures(fail_on)
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="abtem-bench", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list", help="list case ids for a tier")
    _add_selection(s)
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("capture", help="capture one ref into a bundle")
    s.add_argument(
        "--ref",
        required=True,
        help="git ref (tag, branch, sha) of the abtem to measure",
    )
    s.add_argument("--out", required=True, help="bundle directory to create")
    s.add_argument("--preset", default="accuracy", choices=sorted(presets.PRESETS))
    s.add_argument("--repeats", type=int, default=None)
    s.add_argument("--rounds", type=int, default=1)
    _add_selection(s)
    s.set_defaults(func=cmd_capture)

    s = sub.add_parser(
        "self-check",
        help="capture one ref twice, interleaved, and compare (noise floor)",
    )
    s.add_argument("--ref", required=True)
    s.add_argument(
        "--out",
        required=True,
        help="directory receiving bundles a/ and b/, noise.json, self_check.md",
    )
    s.add_argument("--preset", default="accuracy", choices=sorted(presets.PRESETS))
    s.add_argument("--repeats", type=int, default=None)
    s.add_argument("--rounds", type=int, default=1)
    _add_selection(s)
    s.set_defaults(func=cmd_self_check)

    s = sub.add_parser(
        "prepare",
        help="resolve refs and create their worktrees (needs git; run on the host)",
    )
    s.add_argument("--ref", action="append", required=True)
    s.set_defaults(
        func=lambda a: prepare.main([x for r in a.ref for x in ("--ref", r)])
    )

    s = sub.add_parser(
        "compare", help="compare a candidate bundle against a reference bundle"
    )
    s.add_argument("reference")
    s.add_argument("candidate")
    s.add_argument(
        "--noise", help="self-check directory (with a/ and b/) or noise.json"
    )
    s.add_argument(
        "--accepted",
        help="accepted_changes.toml (default: benchmarks/accepted_changes.toml)",
    )
    s.add_argument("--speed-threshold", type=float, default=0.10)
    s.add_argument("--memory-threshold", type=float, default=0.05)
    s.add_argument(
        "--min-time",
        type=float,
        default=0.5,
        help="seconds; a speed ratio beyond threshold on a shorter run is marked "
        "'short' instead of flagged",
    )
    s.add_argument("--allow-case-mismatch", action="store_true")
    s.add_argument("--md", help="write the Markdown report here")
    s.add_argument("--json", help="write the JSON report here")
    s.add_argument("--fail-on", help="comma-separated: drift,shape,speed,memory,error")
    s.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
