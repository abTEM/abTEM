"""The CLI imports abtem from the checkout the harness lives in."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from abtem_bench import cli, runner, store

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


# ---------------------------------------------------------------------------
# exit codes and warnings

CID = "demo.case@quick/cpu"


def _pair(tmp_path, array=None):
    a = np.linspace(1.0, 2.0, 16).reshape(4, 4) if array is None else array
    _bundle(tmp_path / "ref", "ref", a)
    _bundle(tmp_path / "cand", "cand", a.copy())
    accepted = tmp_path / "accepted.toml"
    accepted.write_text("")
    return ["compare", str(tmp_path / "ref"), str(tmp_path / "cand")] + [
        "--accepted",
        str(accepted),
    ]


def test_a_truncated_case_record_exits_2_not_1(tmp_path, capsys):
    argv = _pair(tmp_path)
    case_json = store.Bundle(tmp_path / "cand").case_json(CID)
    case_json.write_text(case_json.read_text()[:30])
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert "Traceback" in err and "error:" in err


def test_a_missing_output_file_exits_2_not_1(tmp_path, capsys):
    argv = _pair(tmp_path)
    store.Bundle(tmp_path / "ref").case_dir(CID).joinpath("out.npz").unlink()
    assert cli.main(argv) == 2
    assert "error:" in capsys.readouterr().err


def test_keyboard_interrupt_and_system_exit_propagate(tmp_path, monkeypatch):
    argv = _pair(tmp_path)
    for exc in (KeyboardInterrupt, SystemExit):

        def boom(*a, **k):
            raise exc()

        monkeypatch.setattr(cli.cmp, "compare", boom)
        with pytest.raises(exc):
            cli.main(argv)


def test_an_empty_fail_on_exits_2(tmp_path, capsys):
    argv = _pair(tmp_path)
    assert cli.main(argv + ["--fail-on", " , "]) == 2
    assert "--fail-on names no gate" in capsys.readouterr().err
    assert cli.main(argv) == 0  # absent: no gates


def test_a_malformed_noise_json_exits_2(tmp_path, capsys):
    argv = _pair(tmp_path)
    noise = tmp_path / "noise.json"
    noise.write_text('{"demo.case@quick/cpu": 3}')
    assert cli.main(argv + ["--noise", str(noise)]) == 2
    assert "noise" in capsys.readouterr().err
    noise.write_text("{not json")
    assert cli.main(argv + ["--noise", str(noise)]) == 2


def test_a_missing_explicit_accepted_file_exits_2(tmp_path, capsys):
    argv = _pair(tmp_path)
    argv[-1] = str(tmp_path / "missing.toml")
    assert cli.main(argv) == 2
    assert "missing.toml" in capsys.readouterr().err


def _no_constants(token):
    raise ValueError(f"non-standard JSON constant {token}")


def test_the_json_report_is_strict_json(tmp_path):
    a = np.linspace(1.0, 2.0, 16).reshape(4, 4)
    b = a.copy()
    b[1, 1] = np.nan  # NaN and infinite statistics
    argv = _pair(tmp_path, a)
    _bundle(tmp_path / "cand", "cand", b)
    out = tmp_path / "report.json"
    assert cli.main(argv + ["--json", str(out)]) == 0
    report = json.loads(out.read_text(), parse_constant=_no_constants)
    stats = report["rows"][0]["outputs"]["out"]
    assert stats["max_abs_norm"] == "inf" and stats["nonfinite_mismatch"] == 1


def _failed_bundle(path: Path, status: str) -> store.Bundle:
    b = store.Bundle(path)
    b.write_manifest({"case_hash": "h"})
    b.write_case("potential.infinite@quick/cpu", {"status": store.STATUS_OK})
    b.write_case("stem.multidetector@quick/cpu", {"status": status, "error": "boom"})
    return b


@pytest.mark.parametrize(
    "status",
    [
        store.STATUS_ERROR,
        store.STATUS_OOM,
        store.STATUS_TIMEOUT,
        store.STATUS_SKIPPED_MEMORY,
    ],
)
def test_capture_and_self_check_exit_1_when_a_case_failed(
    tmp_path, monkeypatch, capsys, status
):
    bundles = [
        _failed_bundle(tmp_path / "a", status),
        _failed_bundle(tmp_path / "b", status),
    ]
    monkeypatch.setattr(runner, "capture", lambda *a, **k: bundles)
    monkeypatch.setattr(runner, "find_repo", lambda: tmp_path)
    only = ["--only", "potential.infinite*", "--only", "stem.multidetector*"]
    argv = ["--ref", "x", "--out", str(tmp_path / "out")] + only
    assert cli.main(["capture"] + argv) == 1
    captured = capsys.readouterr()
    assert "bundle:" in captured.out
    assert "capture: 1 case(s) failed: stem.multidetector@quick/cpu" in captured.err
    assert cli.main(["self-check"] + argv) == 1
    assert "capture: 1 case(s) failed" in capsys.readouterr().err


def test_capture_exits_0_when_every_case_is_ok(tmp_path, monkeypatch):
    b = store.Bundle(tmp_path / "a")
    b.write_manifest({"case_hash": "h"})
    b.write_case("potential.infinite@quick/cpu", {"status": store.STATUS_OK})
    monkeypatch.setattr(runner, "capture", lambda *a, **k: [b])
    monkeypatch.setattr(runner, "find_repo", lambda: tmp_path)
    argv = ["capture", "--ref", "x", "--out", str(tmp_path / "out")]
    assert cli.main(argv + ["--only", "potential.infinite*"]) == 0


def test_only_patterns_that_match_nothing_are_warned_about(
    tmp_path, monkeypatch, capsys
):
    ok = store.Bundle(tmp_path / "a")
    ok.write_manifest({"case_hash": "h"})
    ok.write_case("potential.infinite@quick/cpu", {"status": store.STATUS_OK})
    monkeypatch.setattr(runner, "capture", lambda *a, **k: [ok])
    monkeypatch.setattr(runner, "find_repo", lambda: tmp_path)
    warning = "warning: --only nosuch.* matches no case id"
    flags = ["--only", "potential.infinite*", "--only", "nosuch.*"]
    assert cli.main(["list"] + flags) == 0
    assert capsys.readouterr().err.splitlines() == [warning]
    out = ["--ref", "x", "--out", str(tmp_path / "o")]
    assert cli.main(["capture"] + out + flags) == 0
    assert capsys.readouterr().err.splitlines() == [warning]
    # nothing selected at all: the warning comes with the existing refusal
    for command in ("capture", "self-check"):
        assert cli.main([command] + out + ["--only", "nosuch.*"]) == 2
        err = capsys.readouterr().err
        assert warning in err and "no cases selected" in err
