"""Break down the gain of a series of commits on the multislice benchmark, commit by commit.

Runs `bench_multislice_3ded_payload.py` (always this checkout's copy) on the code of the
base commit and of each following commit that changes `abtem/`, every commit in a git
worktree of its own outside the repository, and reports for each step how much the
time of the compute changed against the step before, the same for the stages of the
instrumented run, and whether the diffraction patterns changed against those of the
base. The commits are cumulative: the gain of a step is that of the commit given the
commits before it.

Options that are not listed here go to the benchmark, so the run of a production size is
this script with the benchmark's options (the relative paths are those of the directory
the script is run from).

Usage
-----
    python benchmarks/bisect_multislice_gains.py --base REV [options] [benchmark options]

    --base REV           the commit to compare against (required)
    --tip REV            the last commit (default HEAD)
    --commits REV ...    the steps after the base, instead of every commit from the
                         base to the tip that changes `abtem/`
    --repeats N          runs of each step, interleaved (default 1); the median counts
    --stages             also run the instrumented benchmark, for the gain of each stage
                         (about twice the time per run; the default is the compute alone)
    --float64-reference  also run the base in double precision, and report how far the
                         pattern of every step is from it (the error of each commit's
                         arithmetic, to tell a change that is rounding from one that is
                         not); the run takes several times as long as the others
    --no-check-output    do not compare the diffraction patterns against the base's
                         (each is saved as a .npy file for the comparison, a few GB at
                         production size, and removed once compared)
    --out DIR            where the report goes (default benchmarks/results/)
    --list               list the steps and stop
    --keep               keep the worktrees

Example, on a GPU, for the commits since the base (one orientation of ZrO2):
    python benchmarks/bisect_multislice_gains.py --base 93fd1cdb --stages \\
        ../py3DED/examples/realistic_cases/ZrO2.json \\
        --cif ../py3DED/examples/realistic_cases/ZrO2_baddeleyit_VESTA.cif \\
        --device gpu --index
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parent.parent
BENCHMARK = Path(__file__).resolve().parent / "bench_multislice_3ded_payload.py"

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--base", required=True)
parser.add_argument("--tip", default="HEAD")
parser.add_argument("--commits", nargs="+")
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--stages", action="store_true")
parser.add_argument("--float64-reference", action="store_true")
parser.add_argument("--no-check-output", action="store_true")
parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
parser.add_argument("--list", action="store_true")
parser.add_argument("--keep", action="store_true")
args, benchmark_args = parser.parse_known_args()


def git(*arguments, check=True):
    return subprocess.run(
        ["git", "-C", str(REPOSITORY), *arguments],
        check=check,
        capture_output=True,
        text=True,
    ).stdout.strip()


def changes_the_package(commit):
    files = git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root", commit)
    return any(name.startswith("abtem/") for name in files.splitlines())


def find_steps():
    base = git("rev-parse", args.base)
    if args.commits:
        later = [git("rev-parse", commit) for commit in args.commits]
    else:
        everything = git(
            "rev-list", "--reverse", "--no-merges", f"{base}..{args.tip}"
        ).split()
        later = [commit for commit in everything if changes_the_package(commit)]
    return [
        {"commit": commit, "subject": git("log", "-1", "--format=%s", commit)}
        for commit in [base, *later]
    ]


def run_step(step, repeat, work, extra=(), label=""):
    """Run the benchmark on the code of `step`; its results, or why it failed. `extra`
    are further benchmark options, which make it the compute alone (as the first
    given, later options win)."""
    stem = f"{step['index']}_{step['commit'][:8]}_{repeat}{label}"
    output = work / f"{stem}.json"
    command = [
        sys.executable,
        str(BENCHMARK),
        *benchmark_args,
        *extra,
        "--output",
        str(output),
    ]
    if extra or not args.stages:
        command.append("--reference-only")
    saved = work / f"{stem}.npy"
    if not args.no_check_output:
        command += ["--save-result", str(saved)]

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [step["worktree"], *environment.get("PYTHONPATH", "").split(os.pathsep)]
    ).rstrip(os.pathsep)
    log = work / f"{stem}.log"
    print(
        f"  {step['commit'][:8]} run {repeat + 1}{label}: {step['subject']}",
        flush=True,
    )
    start = time.perf_counter()
    with open(log, "w") as stream:
        completed = subprocess.run(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT
        )
    seconds = time.perf_counter() - start
    if completed.returncode or not output.exists():
        print(f"    failed (exit {completed.returncode}); see {log}", flush=True)
        return {"failed": f"exit {completed.returncode}", "log": str(log)}

    results = json.loads(output.read_text())
    if not str(results.get("abtem_file", "")).startswith(step["worktree"]):
        # an installed copy shadowed the worktree: the numbers would be another commit's
        return {
            "failed": f"imported {results.get('abtem_file')}, not the worktree",
            "log": str(log),
        }
    case = results["cases"][0]
    record = {
        "log": str(log),
        "wall_seconds": round(seconds, 1),
        "reference": case.get("reference", {}),
        "instrumented": case.get("instrumented", {}),
        "after_compute": case.get("after_compute", {}),
        "n_slices": case.get("n_slices"),
        "saved": str(saved) if saved.exists() else None,
    }
    print(
        f"    compute {record['reference'].get('seconds')} s"
        + (
            f", instrumented {record['instrumented'].get('seconds')} s"
            if args.stages
            else ""
        ),
        flush=True,
    )
    return record


def compare_patterns(reference_file, other_file):
    """How far the saved diffraction patterns `other_file` are from `reference_file`
    (read a few planes at a time): the largest difference and the relative L2 distance,
    both against the reference, and the largest relative difference over the pixels
    above a thousandth of the largest value."""
    first = np.load(reference_file, mmap_mode="r")
    second = np.load(other_file, mmap_mode="r")
    if first.shape != second.shape:
        return {"shapes_differ": [list(first.shape), list(second.shape)]}
    first = first.reshape(-1, *first.shape[-2:])
    second = second.reshape(-1, *second.shape[-2:])

    def blocks():
        for start in range(0, first.shape[0], 4):
            yield (
                np.asarray(first[start : start + 4], dtype=np.float64),
                np.asarray(second[start : start + 4], dtype=np.float64),
            )

    scale = max(float(np.abs(a).max()) for a, _ in blocks())
    largest, squared, norm, strong = 0.0, 0.0, 0.0, 0.0
    for a, b in blocks():
        difference = np.abs(a - b)
        largest = max(largest, float(difference.max()))
        squared += float((difference**2).sum())
        norm += float((a**2).sum())
        mask = np.abs(a) > 1e-3 * scale
        if mask.any():
            strong = max(strong, float((difference[mask] / np.abs(a[mask])).max()))
    return {
        "max_relative_difference": largest / scale if scale else None,
        "relative_l2": (squared / norm) ** 0.5 if norm else None,
        "max_relative_difference_of_strong_pixels": strong,
    }


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def summarise(steps):
    """Per step: the medians over the repeats, and the changes against the step before."""
    rows = []
    for step in steps:
        runs = [run for run in step["runs"] if "failed" not in run]
        row = {
            "commit": step["commit"][:8],
            "subject": step["subject"],
            "failed": len(step["runs"]) - len(runs),
        }
        times = [run["reference"].get("seconds") for run in runs]
        row["compute"] = median(times)
        row["compute_range"] = (
            [min(t for t in times if t), max(t for t in times if t)]
            if any(times)
            else None
        )
        row["instrumented"] = median(
            [run["instrumented"].get("seconds") for run in runs]
        )
        row["categories"] = {}
        row["functions"] = {}
        for run in runs:
            instrumented = run["instrumented"]
            for name, value in instrumented.get("categories", {}).items():
                row["categories"].setdefault(name, []).append(value["seconds"])
            for name, value in instrumented.get("functions", {}).items():
                row["functions"].setdefault(name, []).append(value["exclusive_s"])
        row["categories"] = {k: median(v) for k, v in row["categories"].items()}
        row["functions"] = {k: median(v) for k, v in row["functions"].items()}
        row["sha"] = {run["after_compute"].get("result_sha256") for run in runs}
        row["output"] = step.get("output")
        row["error_against_float64"] = step.get("error_against_float64")
        rows.append(row)
    return rows


def render(rows, steps):
    base = rows[0]
    lines = ["# Gain of each commit on the multislice benchmark", ""]
    lines.append(f"Benchmark options: `{' '.join(benchmark_args)}`  ")
    lines.append(f"Runs per step: {args.repeats}; the median is used.")
    lines.append("")
    with_truth = any(row.get("error_against_float64") for row in rows)
    if with_truth:
        reference = steps[0].get("float64_reference", {}).get("reference", {})
        lines += [
            "The float32 patterns are compared with the base in double precision "
            f"(compute {reference.get('seconds')} s): the largest difference and the "
            "relative L2 distance, against the largest value and the norm of the "
            "double-precision pattern, and the largest relative difference of the "
            "pixels above a thousandth of its largest value.",
            "",
        ]
    lines.append(
        "| step | commit | change | compute [s] | against the step before | "
        "against the base | patterns against the base |"
        + (" error against float64 |" if with_truth else "")
    )
    lines.append("|---|---|---|---|---|---|---|" + ("---|" if with_truth else ""))
    previous = None
    for index, row in enumerate(rows):
        compute = row["compute"]
        if compute is None:
            lines.append(
                f"| {index} | {row['commit']} | {row['subject']} | failed | | | |"
            )
            continue
        spread = ""
        if row["compute_range"] and args.repeats > 1:
            low, high = row["compute_range"]
            spread = f" ({low:.1f}-{high:.1f})"
        before = against_base = ""
        if previous is not None:
            before = (
                f"{compute - previous:+.1f} s ({100 * (compute / previous - 1):+.1f} %)"
            )
        if index and base["compute"]:
            against_base = (
                f"{compute - base['compute']:+.1f} s "
                f"({100 * (compute / base['compute'] - 1):+.1f} %), "
                f"{base['compute'] / compute:.2f}x"
            )
        output = "base"
        if index:
            output = describe_output(row, base)
        truth = (
            f" {describe_difference(row.get('error_against_float64'))} |"
            if with_truth
            else ""
        )
        lines.append(
            f"| {index} | {row['commit']} | {row['subject']} | {compute:.1f}{spread} | "
            f"{before} | {against_base} | {output} |{truth}"
        )
        previous = compute
    lines.append("")

    if args.stages and any(row["categories"] for row in rows):
        lines += [
            "## Stages (instrumented run, device synchronised around every call)",
            "",
            "Seconds of exclusive time; the sums of the synchronised run are larger "
            "than the compute above, so read the changes, not the totals.",
            "",
        ]
        names = sorted(
            {name for row in rows for name in row["categories"]},
            key=lambda name: -(base["categories"].get(name) or 0),
        )
        lines.append("| category | " + " | ".join(r["commit"] for r in rows) + " |")
        lines.append("|---|" + "---|" * len(rows))
        for name in names:
            cells = [
                f"{row['categories'][name]:.1f}" if name in row["categories"] else "-"
                for row in rows
            ]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
        lines += ["### Calls that changed by more than 0.5 s between two steps", ""]
        lines.append("| call | " + " | ".join(r["commit"] for r in rows) + " |")
        lines.append("|---|" + "---|" * len(rows))
        names = sorted(
            {name for row in rows for name in row["functions"]},
            key=lambda name: -(base["functions"].get(name) or 0),
        )
        for name in names:
            values = [row["functions"].get(name) for row in rows]
            present = [value for value in values if value is not None]
            steps_changed = any(
                abs((a or 0) - (b or 0)) > 0.5 for a, b in zip(values, values[1:])
            )
            if len(present) and steps_changed:
                cells = [f"{v:.1f}" if v is not None else "-" for v in values]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def describe_difference(difference):
    if not difference:
        return "not compared"
    if "max_relative_difference" not in difference:
        return str(difference)
    if difference["max_relative_difference"] == 0:
        return "identical"
    return (
        f"largest {difference['max_relative_difference']:.1e}, "
        f"L2 {difference['relative_l2']:.1e}, strong pixels "
        f"{difference['max_relative_difference_of_strong_pixels']:.1e}"
    )


def describe_output(row, base):
    if row.get("output"):
        return describe_difference(row["output"])
    if row["sha"] and row["sha"] == base["sha"] and None not in row["sha"]:
        return "identical (SHA-256)"
    return "not compared"


def main():
    steps = find_steps()
    for index, step in enumerate(steps):
        step["index"] = index
    print(f"{len(steps)} step(s), the first is the base:")
    for step in steps:
        print(f"  {step['commit'][:8]}  {step['subject']}")
    if args.list:
        return

    work = Path(tempfile.mkdtemp(prefix="bisect_multislice_"))
    created = []
    try:
        for step in steps:
            path = work / f"worktree_{step['index']}_{step['commit'][:8]}"
            git("worktree", "add", "--detach", str(path), step["commit"])
            created.append(path)
            step["worktree"] = str(path)
            step["runs"] = []

        print(f"\nrunning in {work}")
        truth = None  # the patterns of the base in double precision
        if args.float64_reference:
            run = run_step(steps[0], 0, work, ["--precision", "float64"], "_float64")
            truth_run = {k: v for k, v in run.items() if k != "saved"}
            truth = run.get("saved")
            if "failed" in run:
                print("the float64 reference failed; going on without it", flush=True)
            else:
                steps[0]["float64_reference"] = truth_run

        reference = None  # the patterns of the base's first run
        for repeat in range(args.repeats):
            for step in steps:
                run = run_step(step, repeat, work)
                step["runs"].append(run)
                # compare and remove at once: a run's patterns are a few GB
                saved = run.pop("saved", None)
                if not saved:
                    continue
                if truth and "error_against_float64" not in step:
                    step["error_against_float64"] = compare_patterns(truth, saved)
                if reference is None and step is steps[0]:
                    reference = saved
                    continue
                if reference and "output" not in step:
                    step["output"] = compare_patterns(reference, saved)
                Path(saved).unlink(missing_ok=True)
        for leftover in (reference, truth):
            if leftover:
                Path(leftover).unlink(missing_ok=True)

        rows = summarise(steps)
        report = render(rows, steps)
        print("\n" + report)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        (out / f"bisect_{stamp}.md").write_text(report)
        (out / f"bisect_{stamp}.json").write_text(
            json.dumps(
                {"benchmark_args": benchmark_args, "rows": rows, "steps": steps},
                indent=2,
                default=lambda value: sorted(value, key=str),
            )
        )
        print(f"\nwritten to {out / f'bisect_{stamp}.md'} (logs in {work})")
    finally:
        if not args.keep:  # the logs stay: a failed step is diagnosed from them
            for path in created:
                git("worktree", "remove", "--force", str(path), check=False)
            git("worktree", "prune", check=False)


if __name__ == "__main__":
    main()
