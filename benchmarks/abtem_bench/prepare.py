"""Prepare ref worktrees on a machine that has git, for runs on one that has not.

Standard library only, so it runs with any Python:

    python3 -P -m abtem_bench.prepare --ref origin/dev --ref v1.0.10

creates ``.worktrees/bench/<sha>`` for each ref under the repository that
contains this harness and records label, sha and ``git describe`` in
``.worktrees/bench/index.json``. Inside a container without git, the runner
resolves ``--ref`` labels from that index and uses the prepared worktrees.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]  # .../benchmarks
INDEX_NAME = "index.json"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def repo_root(start: Path = HARNESS_DIR) -> Path:
    """The repository containing ``start``: the first parent with a ``.git`` entry.

    Works without git, for both a full clone (``.git`` directory) and a
    linked worktree (``.git`` file).
    """
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
    raise FileNotFoundError(f"no .git found above {start}")


def bench_dir(repo: Path) -> Path:
    return repo / ".worktrees" / "bench"


def load_index(repo: Path) -> dict[str, dict[str, str]]:
    path = bench_dir(repo) / INDEX_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_index(repo: Path, index: dict[str, dict[str, str]]) -> None:
    bench_dir(repo).mkdir(parents=True, exist_ok=True)
    (bench_dir(repo) / INDEX_NAME).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


def prepare(repo: Path, ref: str) -> dict[str, str]:
    """Resolve ``ref``, create its worktree if missing, record it in the index."""
    sha = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    try:
        describe = _git(repo, "describe", "--tags", "--always", sha)
    except subprocess.CalledProcessError:
        describe = sha[:12]
    worktree = bench_dir(repo) / sha
    if not (worktree / "abtem" / "__init__.py").exists():
        worktree.parent.mkdir(parents=True, exist_ok=True)
        if worktree.exists():
            subprocess.run(["git", "-C", str(repo), "worktree", "prune"], check=False)
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(worktree), sha],
            check=True,
            capture_output=True,
            text=True,
        )
    entry = {"sha": sha, "describe": describe, "worktree": str(worktree)}
    index = load_index(repo)
    index[ref] = entry
    index[sha] = entry
    save_index(repo, index)
    return entry


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="abtem_bench.prepare", description=__doc__)
    p.add_argument("--ref", action="append", required=True, help="git ref (repeatable)")
    args = p.parse_args(argv)
    repo = repo_root()
    for ref in args.ref:
        e = prepare(repo, ref)
        print(f"{ref:<20} {e['sha'][:12]}  {e['describe']:<28} {e['worktree']}")
    print(f"index: {bench_dir(repo) / INDEX_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
