"""Command line: list, capture, self-check, compare."""

from __future__ import annotations

import argparse
import shlex
import sys
import traceback
from pathlib import Path

from abtem_bench import compare as cmp
from abtem_bench import prepare, presets, registry, runner, store


def _devices(text: str) -> list[str]:
    out = [d.strip() for d in text.split(",") if d.strip()]
    for d in out:
        if d not in registry.DEVICES:
            raise argparse.ArgumentTypeError(f"unknown device {d!r}")
    return out


def _at_least(minimum: int):
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be {minimum} or more")
        return value

    return parse


def _positive(text: str) -> float:
    value = float(text)
    if not (value > 0 and value != float("inf")):
        raise argparse.ArgumentTypeError("must be a positive number")
    return value


def _command() -> str:
    """The invocation as a command someone can run again."""
    return "python -P -m abtem_bench " + shlex.join(sys.argv[1:])


def _add_capture_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default="accuracy", choices=sorted(presets.PRESETS))
    p.add_argument("--repeats", type=_at_least(0), default=None)
    p.add_argument("--rounds", type=_at_least(1), default=1)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="delete an existing bundle at the output path first",
    )


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


def _select(reg: dict, args: argparse.Namespace) -> list[registry.CaseId]:
    """The selected case ids; warns about every ``--only`` pattern matching none."""
    for pattern in registry.unmatched_patterns(
        reg, args.tier, args.device, args.only, args.tag
    ):
        print(f"warning: --only {pattern} matches no case id", file=sys.stderr)
    return registry.select_ids(reg, args.tier, args.device, args.only, args.tag)


def _failed_cases(bundles: list[store.Bundle]) -> list[str]:
    """Ids of the cases that ended in a failed status in any of the bundles."""
    failed: set[str] = set()
    for b in bundles:
        for cid in b.case_ids():
            if b.read_case(cid).get("status") in cmp.FAILED_STATUSES:
                failed.add(cid)
    return sorted(failed)


def _report_failed(bundles: list[store.Bundle]) -> bool:
    """Print the failed cases of a capture to stderr; whether there were any."""
    failed = _failed_cases(bundles)
    if failed:
        print(
            f"capture: {len(failed)} case(s) failed: {', '.join(failed)}",
            file=sys.stderr,
        )
    return bool(failed)


def cmd_list(args: argparse.Namespace) -> int:
    reg = registry.load_cases()
    ids = _select(reg, args)
    print(f"case_hash {registry.case_hash()}")
    for cid in ids:
        c = reg[cid.name]
        print(f"{cid!s:<50} tags={','.join(sorted(c.tags))}")
    print(f"{len(ids)} case ids")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    reg = registry.load_cases()
    ids = _select(reg, args)
    if not ids:
        print("no cases selected", file=sys.stderr)
        return 2
    repo = runner.find_repo()
    bundles = runner.capture(
        repo,
        [args.ref],
        ids,
        Path(args.out).resolve().parent,
        args.preset,
        args.tier,
        args.device,
        args.repeats,
        rounds=args.rounds,
        labels=[Path(args.out).name],
        command=_command(),
        overwrite=args.overwrite,
    )
    print(f"bundle: {bundles[0].path}")
    return 1 if _report_failed(bundles) else 0


def cmd_self_check(args: argparse.Namespace) -> int:
    reg = registry.load_cases()
    ids = _select(reg, args)
    if not ids:
        print("no cases selected", file=sys.stderr)
        return 2
    repo = runner.find_repo()
    out = Path(args.out).resolve()
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
        command=_command(),
        overwrite=args.overwrite,
    )
    failed = _report_failed(bundles)
    out.mkdir(parents=True, exist_ok=True)
    floors = cmp.noise_floor(bundles[0], bundles[1], reg)
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
    else:
        print("self-check: all cases bit-identical")
    if failed:
        return 1
    # Bit identity is the expectation only for the float64 preset on CPU.
    if not_identical and args.preset == "accuracy" and args.device == ["cpu"]:
        return 1
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    gates = cmp.parse_fail_on(args.fail_on)
    if "memory" in gates and not args.noise:
        raise cmp.CompareError("--fail-on memory needs a noise floor (--noise)")
    reg = registry.load_cases()
    reference, candidate = store.Bundle(args.reference), store.Bundle(args.candidate)
    for b in (reference, candidate):
        if not b.exists():
            raise cmp.CompareError(f"not a bundle: {b.path}")
    noise = None
    if args.noise:
        npath = Path(args.noise)
        if npath.is_dir():
            noise = cmp.noise_floor(
                store.Bundle(npath / "a"), store.Bundle(npath / "b"), reg
            )
        elif npath.is_file():
            try:
                noise = store.load_json(npath)
            except ValueError as exc:
                raise cmp.CompareError(f"{npath}: not JSON: {exc}") from exc
            cmp.check_noise(noise, str(npath))
        else:
            raise cmp.CompareError(f"no self-check or noise.json at {npath}")
    report = cmp.compare(
        reference,
        candidate,
        reg,
        accepted_path=Path(args.accepted) if args.accepted else None,
        noise=noise,
        speed_threshold=args.speed_threshold,
        memory_threshold=args.memory_threshold,
        min_delta=args.min_delta,
        allow_case_mismatch=args.allow_case_mismatch,
        allow_preset_mismatch=args.allow_preset_mismatch,
    )
    md = cmp.to_markdown(report)
    if args.md:
        Path(args.md).write_text(md)
    if args.json:
        store.dump_json(Path(args.json), cmp.to_json(report))
    print(md)
    failures = report.failures(gates)
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
    _add_capture_options(s)
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
    _add_capture_options(s)
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
    s.add_argument(
        "--speed-threshold",
        type=_positive,
        default=0.10,
        help="fraction; flag time ratios beyond 1 +/- this (default 0.10)",
    )
    s.add_argument(
        "--memory-threshold",
        type=_positive,
        default=0.05,
        help="fraction; flag RSS and VRAM ratios beyond 1 +/- this (default 0.05); "
        "memory is flagged only for cases with a noise floor",
    )
    s.add_argument(
        "--min-delta",
        type=_positive,
        default=0.05,
        help="seconds; a time ratio beyond threshold whose medians differ by "
        "less than this is marked 'short' instead of flagged (default 0.05)",
    )
    s.add_argument("--allow-case-mismatch", action="store_true")
    s.add_argument("--allow-preset-mismatch", action="store_true")
    s.add_argument("--md", help="write the Markdown report here")
    s.add_argument("--json", help="write the JSON report here")
    s.add_argument(
        "--fail-on",
        help="comma-separated gates: drift, shape, error, missing, speed[:N%%], "
        "memory[:N%%] (memory needs --noise); exit 1 if any fails",
    )
    s.set_defaults(func=cmd_compare)
    return p


def use_invoking_checkout() -> Path | None:
    """Make ``import abtem`` in this process resolve to the harness's checkout.

    compare imports ``abtem.core.testing`` from the checkout the harness and the
    cases come from. Without this, the import resolves to whatever abtem the
    environment provides, typically an editable install of some other checkout
    at some other commit, which may not have the helper or may have a different
    one. Workers are unaffected: each gets its ref's worktree on PYTHONPATH.
    Returns the checkout put on ``sys.path``, or None when the harness is not in
    a checkout (installed from a wheel), where the environment's abtem is used.
    """
    try:
        root = prepare.repo_root()
    except FileNotFoundError:
        return None
    if not (root / "abtem" / "__init__.py").exists():
        return None
    if "abtem" in sys.modules:
        return None  # already imported; a later sys.path change cannot redirect it
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    use_invoking_checkout()
    try:
        return args.func(args)
    except (cmp.CompareError, runner.BundleExistsError, runner.RefError) as exc:
        # Input problems, not gate failures: exit 2, which --fail-on never uses.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 -- 1 is reserved for a failed gate
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
