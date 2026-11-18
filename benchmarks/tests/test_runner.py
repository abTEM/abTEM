"""Runner and worker end to end, and the runner's bookkeeping."""

import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from abtem_bench import registry, runner, store
from abtem_bench.registry import CaseId
from conftest import CHECKOUT

CID = CaseId.parse("potential.infinite@quick/cpu")

#: ``VmHWM`` is read from ``/proc/self/status``.
needs_proc = pytest.mark.skipif(
    sys.platform != "linux", reason="peak memory is read from /proc"
)


@pytest.fixture
def this_checkout(monkeypatch):
    """A Ref whose worktree is the checkout under test: no git, no other commit."""
    monkeypatch.setattr(runner, "MEM_FLOOR_BYTES", 0)
    registry.load_cases()
    return runner.Ref(label="checkout", sha="0" * 40, worktree=CHECKOUT, describe="x")


@needs_proc
def test_end_to_end_with_a_relative_bundle_path(tmp_path, monkeypatch, this_checkout):
    monkeypatch.chdir(tmp_path)
    bundle = runner.new_bundle(
        Path("rel/bundle"), this_checkout, "accuracy", "quick", ["cpu"], "t"
    )
    assert bundle.path == tmp_path / "rel" / "bundle"
    rec = runner.run_case(this_checkout, CID, bundle, "accuracy", repeats=0)
    assert rec["status"] == store.STATUS_OK, rec.get("error")
    assert bundle.case_ids() == [str(CID)]
    assert (bundle.case_dir(str(CID)) / "potential.npz").exists()
    mem = rec["memory"]
    # the worker's own peak; wait4's figure also carries the runner's high water
    assert 0 < mem["peak_rss_bytes"] <= mem["peak_rss_wait4_bytes"]
    assert mem["rss_meter"] == "VmHWM"
    # a CPU case never touches the GPU
    assert mem["peak_vram_pool_bytes"] is None and mem["samples"] == 0


def _crashing_python(tmp_path):
    script = tmp_path / "segv.sh"
    script.write_text("#!/bin/sh\nkill -SEGV $$\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_a_worker_that_dies_never_inherits_the_previous_record(tmp_path, this_checkout):
    bundle = store.Bundle(tmp_path / "b")
    bundle.write_manifest({"case_hash": "h"})
    bundle.write_case(
        str(CID),
        {"status": store.STATUS_OK, "timings": {"median": 1.0}, "memory": {}},
        {"potential": (np.ones(3), [], {})},
    )
    rec = runner.run_case(
        this_checkout, CID, bundle, "accuracy", python=_crashing_python(tmp_path)
    )
    assert rec["status"] == store.STATUS_ERROR
    assert "SIGSEGV" in rec["error"]
    assert bundle.read_case(str(CID))["outputs"] == {}
    assert not bundle.case_dir(str(CID)).exists()


def test_new_bundle_refuses_a_non_empty_directory_unless_overwriting(
    tmp_path, this_checkout
):
    out = tmp_path / "b"
    runner.new_bundle(out, this_checkout, "accuracy", "quick", ["cpu"], "t")
    (out / "cases").mkdir()
    (out / "cases" / "stale.json").write_text("{}")
    with pytest.raises(runner.BundleExistsError):
        runner.new_bundle(out, this_checkout, "accuracy", "quick", ["cpu"], "t")
    runner.new_bundle(out, this_checkout, "accuracy", "quick", ["cpu"], "t", True)
    assert not (out / "cases" / "stale.json").exists()


@pytest.mark.parametrize("overwrite", [False, True])
def test_a_directory_that_is_not_a_bundle_is_never_deleted(
    tmp_path, this_checkout, overwrite
):
    out = tmp_path / "results"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    with pytest.raises(runner.BundleExistsError, match="not a bundle"):
        runner.new_bundle(
            out, this_checkout, "accuracy", "quick", ["cpu"], "t", overwrite
        )
    assert (out / "keep.txt").read_text() == "mine"
    assert not (out / "manifest.json").exists()


def test_overwrite_replaces_a_bundle_and_creates_a_missing_directory(
    tmp_path, this_checkout
):
    out = tmp_path / "b"
    runner.new_bundle(out, this_checkout, "accuracy", "quick", ["cpu"], "t")
    (out / "stale.txt").write_text("old")
    runner.new_bundle(out, this_checkout, "accuracy", "quick", ["cpu"], "t", True)
    assert (out / "manifest.json").exists() and not (out / "stale.txt").exists()
    fresh = tmp_path / "new" / "b"
    runner.new_bundle(fresh, this_checkout, "accuracy", "quick", ["cpu"], "t", True)
    assert (fresh / "manifest.json").exists()


@pytest.mark.parametrize("target", ["cwd", "parent"])
def test_overwrite_never_deletes_the_current_directory_or_a_parent(
    tmp_path, monkeypatch, this_checkout, target
):
    out = tmp_path / "outer" / "b"
    runner.new_bundle(out, this_checkout, "accuracy", "quick", ["cpu"], "t")
    inner = out / "cases"
    inner.mkdir()
    monkeypatch.chdir(inner if target == "parent" else out)
    with pytest.raises(runner.BundleExistsError, match="current directory"):
        runner.new_bundle(
            out, this_checkout, "accuracy", "quick", ["cpu"], "t", overwrite=True
        )
    assert (out / "manifest.json").exists() and inner.exists()


@needs_proc
@pytest.mark.parametrize(
    "exc, status", [(MemoryError, store.STATUS_OOM), (RuntimeError, store.STATUS_ERROR)]
)
def test_failed_workers_name_the_rss_meter_too(tmp_path, monkeypatch, exc, status):
    import argparse

    from abtem_bench import worker

    def boom(expected_root):
        raise exc("boom")

    monkeypatch.setattr(worker, "_identify", boom)
    args = argparse.Namespace(
        out=str(tmp_path / "b"),
        case_id=str(CID),
        ref_label="x",
        ref_sha="0" * 40,
        preset="accuracy",
        expected_root=None,
        repeats=0,
    )
    worker.run(args)
    rec = store.Bundle(args.out).read_case(str(CID))
    assert rec["status"] == status
    assert rec["memory"]["rss_meter"] == "VmHWM"
    assert rec["memory"]["peak_rss_bytes"] > 0


def test_rounds_recompute_statistics_over_all_rounds():
    prev = {
        "timings": {"warm": [1.0, 2.0, 3.0], "warm_cpu": [1, 1, 1], "cold": 5.0},
        "memory": {"peak_rss_bytes": 100, "peak_rss_wait4_bytes": 300},
    }
    rec = {
        "timings": {"warm": [4.0, 5.0, 6.0], "warm_cpu": [1, 1, 1], "cold": 7.0},
        "memory": {"peak_rss_bytes": 120, "peak_rss_wait4_bytes": 250},
    }
    t = runner._merge_rounds(prev, rec, 2)["timings"]
    assert t["warm"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert t["median"] == pytest.approx(3.5) and t["min"] == 1.0
    assert t["cold_all"] == [5.0, 7.0] and t["cold"] == pytest.approx(6.0)
    assert rec["memory"]["peak_rss_bytes"] == 120
    assert rec["memory"]["peak_rss_wait4_bytes"] == 300


def test_failure_classification_reads_only_the_end_of_the_log():
    log = "hipErrorNoDevice: from an early device probe\n" + "ok\n" * 60
    log += "Traceback (most recent call last):\nValueError: boom\n"
    assert runner.classify_failure(1, log, False) == (
        store.STATUS_ERROR,
        "ValueError: boom",
    )
    assert runner.classify_failure(-9, "", False)[0] == store.STATUS_OOM
    assert runner.classify_failure(0, "", True)[0] == store.STATUS_TIMEOUT


def test_case_timeout_prefers_the_declared_tier_timeout():
    registry.load_cases()
    declared = registry.REGISTRY["stem.multidetector"].tiers["standard"].timeout
    assert (
        runner.case_timeout(CaseId("stem.multidetector", tier="standard")) == declared
    )
    assert runner.case_timeout(CaseId("nosuch.case", tier="quick")) == 120.0


def test_overwrite_refuses_a_manifest_that_is_not_a_bundles(tmp_path, this_checkout):
    out = tmp_path / "project"
    (out / "src").mkdir(parents=True)
    (out / "manifest.json").write_text('{"name": "my-extension"}')
    (out / "src" / "data.txt").write_text("keep")
    with pytest.raises(runner.BundleExistsError, match="not a bundle"):
        runner.new_bundle(out, this_checkout, "accuracy", "quick", ["cpu"], "t", True)
    assert (out / "src" / "data.txt").read_text() == "keep"


def _worker_args(tmp_path, cid):
    import argparse

    return argparse.Namespace(
        out=str(tmp_path / "b"),
        case_id=str(cid),
        ref_label="x",
        ref_sha="0" * 40,
        preset="accuracy",
        expected_root=None,
        repeats=0,
    )


@pytest.fixture
def restored_config():
    """``worker.run`` applies a preset to abtem's and dask's process-wide
    configuration; put both back so later tests see what they started with."""
    import copy

    import dask

    from abtem.core import config as abtem_config

    saved = [(d, copy.deepcopy(d)) for d in (abtem_config.config, dask.config.config)]
    yield
    for d, snapshot in saved:
        d.clear()
        d.update(snapshot)


def _run_demo_worker(tmp_path, monkeypatch, cid, case_requires=None, **variants):
    from abtem_bench import worker
    from abtem_bench.registry import Tier

    calls = []

    def func(p, device):
        calls.append(p.variant)
        return lambda: np.ones(3)

    reg = {}
    monkeypatch.setattr(registry, "REGISTRY", reg)
    monkeypatch.setattr(registry, "load_cases", lambda: reg)
    registry.case(
        "demo.case",
        tiers={t: Tier(gpts=(4, 4)) for t in registry.TIER_NAMES},
        variants=variants,
        requires=case_requires,
    )(func)
    args = _worker_args(tmp_path, cid)
    assert worker.run(args) == 0
    return store.Bundle(args.out).read_case(str(cid)), calls


@pytest.mark.parametrize(
    "cid, requires, status, error",
    [
        ("demo.case@quick/cpu", False, store.STATUS_OK, None),
        (
            "demo.case[alt]@quick/cpu",
            False,
            store.STATUS_UNSUPPORTED,
            "requires() of variant 'alt' is false on this ref",
        ),
        ("demo.case[alt]@quick/cpu", True, store.STATUS_OK, None),
    ],
)
def test_a_variant_requires_decides_only_that_variant(
    tmp_path, monkeypatch, restored_config, cid, requires, status, error
):
    from abtem_bench.registry import Variant

    cid = CaseId.parse(cid)
    rec, calls = _run_demo_worker(
        tmp_path, monkeypatch, cid, alt=Variant(requires=lambda abtem: requires)
    )
    assert rec["status"] == status, rec.get("error")
    assert rec.get("error") == error
    assert calls == ([cid.variant] if status == store.STATUS_OK else [])


def test_a_case_requires_makes_every_variant_unsupported(
    tmp_path, monkeypatch, restored_config
):
    from abtem_bench.registry import Variant

    cid = CaseId.parse("demo.case[alt]@quick/cpu")
    rec, calls = _run_demo_worker(
        tmp_path,
        monkeypatch,
        cid,
        case_requires=lambda abtem: False,
        alt=Variant(requires=lambda abtem: True),
    )
    assert rec["status"] == store.STATUS_UNSUPPORTED
    assert rec["error"] == "requires() is false on this ref"
    assert calls == []


def _capture_rounds(tmp_path, monkeypatch, preset, rounds, script, arrays_script=None):
    """Capture ``demo.case`` with a fake ``run_case`` over several rounds.

    ``script`` lists the status of each call to ``run_case``; ``arrays_script``
    the value of its single output ``out``. Returns the bundle and the calls made.
    """
    calls = []
    ref = runner.Ref("x", "0" * 40, tmp_path, "x")

    def fake_run_case(ref, cid, bundle, preset, repeats=None, timeout=None):
        i = len(calls)
        calls.append(str(cid))
        bundle.clear_case(str(cid))
        status = script[i]
        record = {"status": status, "ref": ref.label}
        outputs = None
        if status == store.STATUS_OK:
            record["timings"] = {"warm": [1.0], "median": 1.0, "cold": 2.0}
            arrays = (arrays_script or [np.ones(3)] * len(script))[i]
            outputs = {"out": (arrays, [], {})}
        else:
            record["error"] = "boom"
        bundle.write_case(str(cid), record, outputs)
        return bundle.read_case(str(cid))

    monkeypatch.setattr(runner, "prepare_ref", lambda repo, r: ref)
    monkeypatch.setattr(runner, "run_case", fake_run_case)
    [bundle] = runner.capture(
        tmp_path,
        ["x"],
        [CaseId.parse("demo.case@quick/cpu")],
        tmp_path / "out",
        preset,
        "quick",
        ["cpu"],
        None,
        rounds=rounds,
        progress=lambda *a: None,
    )
    return bundle, calls


def test_a_failed_round_keeps_its_record_and_ends_the_case(tmp_path, monkeypatch):
    bundle, calls = _capture_rounds(
        tmp_path,
        monkeypatch,
        "accuracy",
        3,
        [store.STATUS_OK, store.STATUS_ERROR, store.STATUS_OK],
    )
    rec = bundle.read_case("demo.case@quick/cpu")
    assert rec["status"] == store.STATUS_ERROR and rec["error"] == "boom"
    assert len(calls) == 2  # round 3 did not run


def test_a_failed_first_round_runs_no_further_round(tmp_path, monkeypatch):
    bundle, calls = _capture_rounds(
        tmp_path, monkeypatch, "speed", 2, [store.STATUS_ERROR, store.STATUS_OK]
    )
    assert bundle.read_case("demo.case@quick/cpu")["status"] == store.STATUS_ERROR
    assert len(calls) == 1


@pytest.mark.parametrize("preset", ["accuracy", "speed"])
def test_identical_rounds_are_merged_and_counted(tmp_path, monkeypatch, preset):
    nan = np.array([1.0, np.nan])
    bundle, calls = _capture_rounds(
        tmp_path,
        monkeypatch,
        preset,
        3,
        [store.STATUS_OK] * 3,
        [nan, nan.copy(), nan.copy()],
    )
    rec = bundle.read_case("demo.case@quick/cpu")
    assert rec["status"] == store.STATUS_OK and "notes" not in rec
    assert rec["timings"]["rounds"] == 3 and rec["timings"]["warm"] == [1.0] * 3
    assert len(calls) == 3


@pytest.mark.parametrize(
    "second",
    [
        np.array([1.0, 1.0, 1.5]),  # a value
        np.ones(4),  # a shape
        np.ones(3, dtype=np.float32),  # a dtype
    ],
    ids=["value", "shape", "dtype"],
)
def test_outputs_that_differ_between_rounds_are_an_error_under_accuracy(
    tmp_path, monkeypatch, second
):
    bundle, _ = _capture_rounds(
        tmp_path,
        monkeypatch,
        "accuracy",
        3,
        [store.STATUS_OK] * 3,
        [np.ones(3), second, second],
    )
    rec = bundle.read_case("demo.case@quick/cpu")
    assert rec["status"] == store.STATUS_ERROR
    assert "out" in rec["error"] and "round 1 and round 2" in rec["error"]
    assert rec["timings"]["rounds"] == 2  # round 3 did not run


def test_outputs_that_differ_between_rounds_are_a_note_under_speed(
    tmp_path, monkeypatch
):
    bundle, calls = _capture_rounds(
        tmp_path,
        monkeypatch,
        "speed",
        3,
        [store.STATUS_OK] * 3,
        [np.ones(3), np.zeros(3), np.zeros(3)],
    )
    rec = bundle.read_case("demo.case@quick/cpu")
    assert rec["status"] == store.STATUS_OK and len(calls) == 3
    assert rec["timings"]["rounds"] == 3
    assert len(rec["notes"]) == 1 and "round 1 and round 2: out" in rec["notes"][0]


def test_compare_reports_the_round_note(tmp_path, registry_with_demo):
    from abtem_bench import compare as cmp
    from conftest import CID, bundle, record

    note = "outputs differ between round 1 and round 2: out"
    cases = {CID: (record(notes=[note]), {"out": (np.ones(3), [], {})})}
    ref = bundle(tmp_path, "ref", cases)
    cand = bundle(tmp_path, "cand", cases)
    report = cmp.compare(ref, cand, registry_with_demo)
    assert f"candidate: {note}" in report.rows[0].notes


def test_user_configuration_does_not_reach_the_captured_config(
    tmp_path, monkeypatch, this_checkout
):
    """YAML under ``~/.config`` and ``ABTEM_*``/``DASK_*`` variables are dropped."""
    import yaml

    leak = {"antialias": {"cutoff": 0.123}}
    for package in ("abtem", "dask"):
        (tmp_path / "home" / ".config" / package).mkdir(parents=True)
        (tmp_path / "home" / ".config" / package / "x.yaml").write_text(
            yaml.safe_dump(leak)
        )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ABTEM_ANTIALIAS__CUTOFF", "0.234")
    monkeypatch.setenv("DASK_ANTIALIAS__CUTOFF", "0.345")
    bundle = runner.new_bundle(
        tmp_path / "b", this_checkout, "accuracy", "quick", ["cpu"], "t"
    )
    rec = runner.run_case(this_checkout, CID, bundle, "accuracy", repeats=0)
    assert rec["status"] == store.STATUS_OK, rec.get("error")
    default = yaml.safe_load((CHECKOUT / "abtem" / "core" / "abtem.yaml").read_text())
    assert rec["config"]["abtem_resolved"]["antialias"] == default["antialias"]
    # dask's own defaults and the preset's settings remain
    assert "chunk-size" in rec["config"]["dask_resolved"]["array"]
    assert rec["config"]["dask_resolved"]["scheduler"] == "synchronous"
    assert rec["config"]["abtem_resolved"]["precision"] == "float64"


@pytest.mark.parametrize(
    "platform, expected", [("linux", 5 * 1024), ("darwin", 5), ("win32", 5 * 1024)]
)
def test_ru_maxrss_is_converted_to_bytes_per_platform(platform, expected):
    assert runner._maxrss_bytes(5, platform) == expected
