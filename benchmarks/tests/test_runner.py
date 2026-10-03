"""Runner and worker end to end, and the runner's bookkeeping."""

import stat
from pathlib import Path

import numpy as np
import pytest
from abtem_bench import registry, runner, store
from abtem_bench.registry import CaseId
from conftest import CHECKOUT

CID = CaseId.parse("potential.infinite@quick/cpu")


@pytest.fixture
def this_checkout(monkeypatch):
    """A Ref whose worktree is the checkout under test: no git, no other commit."""
    monkeypatch.setattr(runner, "MEM_FLOOR_BYTES", 0)
    registry.load_cases()
    return runner.Ref(label="checkout", sha="0" * 40, worktree=CHECKOUT, describe="x")


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
