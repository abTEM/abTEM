"""The CLI imports abtem from the checkout the harness lives in."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from abtem_bench import store

HARNESS = Path(__file__).resolve().parents[1]
CHECKOUT = HARNESS.parent


def _bundle(path: Path, label: str, array: np.ndarray) -> None:
    b = store.Bundle(path)
    b.write_manifest(
        {
            "case_hash": "h",
            "ref": {"label": label, "sha": "0" * 40, "describe": label},
            "preset": "accuracy",
            "tier": "quick",
            "devices": ["cpu"],
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "fingerprint": {"hostname": "box"},
            "fingerprint_short": "f",
        }
    )
    b.write_case(
        "demo.case@quick/cpu",
        {"status": store.STATUS_OK, "timings": {"median": 1.0}, "memory": {}},
        {"out": (array, [], {})},
    )


def test_compare_uses_the_harness_checkout_not_the_installed_abtem(tmp_path):
    # Only the harness directory on PYTHONPATH, and a cwd outside any checkout:
    # what the README's command line gives. The environment's own abtem (an
    # editable install of another checkout, or a wheel) must not be the one
    # whose abtem.core.testing compare uses.
    a = np.linspace(1.0, 2.0, 16).reshape(4, 4)
    _bundle(tmp_path / "ref", "ref", a)
    _bundle(tmp_path / "cand", "cand", a.copy())
    env = {**os.environ, "PYTHONPATH": str(HARNESS)}
    probe = (
        "import sys; from abtem_bench import cli; root = cli.use_invoking_checkout();"
        "import abtem, json; print(json.dumps([str(root), abtem.__file__]))"
    )
    out = subprocess.run(
        [sys.executable, "-P", "-c", probe],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    root, abtem_file = json.loads(out.stdout.strip().splitlines()[-1])
    assert Path(root) == CHECKOUT
    assert Path(abtem_file).resolve().is_relative_to(CHECKOUT)

    # an explicit, empty accepted-changes file: the shipped one is not under test
    accepted = tmp_path / "accepted.toml"
    accepted.write_text("")
    run = subprocess.run(
        [
            sys.executable,
            "-P",
            "-m",
            "abtem_bench",
            "compare",
            str(tmp_path / "ref"),
            str(tmp_path / "cand"),
            "--accepted",
            str(accepted),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert "IDENTICAL" in run.stdout
