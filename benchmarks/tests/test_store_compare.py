"""Store round trip and the compare verdict matrix on synthetic bundles."""

import numpy as np
import pytest
from abtem_bench import compare as cmp
from abtem_bench import registry, store
from abtem_bench.registry import Tier, Tolerance, Variant


def _manifest(label="ref", case_hash="h", fp="ffff0000"):
    return {
        "schema_version": store.SCHEMA_VERSION,
        "harness_version": "0.0",
        "case_hash": case_hash,
        "ref": {"label": label, "sha": "0" * 40, "describe": label},
        "preset": "accuracy",
        "tier": "quick",
        "devices": ["cpu"],
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "fingerprint": {"hostname": "box"},
        "fingerprint_short": fp,
    }


def _record(median=1.0, cold=2.0, rss=100 * 1024**2, status=store.STATUS_OK):
    return {
        "status": status,
        "timings": {"median": median, "cold": cold, "warm": [median]},
        "memory": {"peak_rss_bytes": rss},
    }


def _bundle(tmp_path, name, cases, **manifest):
    b = store.Bundle(tmp_path / name)
    b.write_manifest(_manifest(label=name, **manifest))
    for cid, (record, outputs) in cases.items():
        b.write_case(cid, record, outputs)
    return b


@pytest.fixture
def registry_with_demo(monkeypatch, tmp_path):
    reg = {}
    monkeypatch.setattr(registry, "REGISTRY", reg)
    # tests must not read the repository's real accepted_changes.toml
    monkeypatch.setattr(cmp, "ACCEPTED_CHANGES_PATH", tmp_path / "no-accepted.toml")
    tiers = {t: Tier(gpts=(4, 4)) for t in registry.TIER_NAMES}
    registry.case(
        "demo.case",
        tiers=tiers,
        variants={"alt": Variant(compare_as="default"), "auto": Variant(flag=False)},
        tolerance=Tolerance(rel=1e-10, intensity=1e-12, max_abs_norm=1e-10),
    )(lambda p, d: lambda: None)
    return reg


def test_store_roundtrip(tmp_path):
    b = store.Bundle(tmp_path / "b")
    b.write_manifest(_manifest())
    arr = np.arange(12, dtype=np.complex128).reshape(3, 4) * (1 + 1j)
    axes = [{"type": "RealSpaceAxis", "sampling": 0.1, "offset": 0.0, "units": "Å"}]
    b.write_case(
        "demo.case@quick/cpu", _record(), {"wave": (arr, axes, {"energy": 1.0})}
    )
    assert b.case_ids() == ["demo.case@quick/cpu"]
    rec = b.read_case("demo.case@quick/cpu")
    assert rec["outputs"]["wave"]["dtype"] == "complex128"
    assert rec["outputs"]["wave"]["invariants"]["size"] == 12
    a, ax, meta = b.load_output("demo.case@quick/cpu", "wave")
    np.testing.assert_array_equal(a, arr)
    assert ax == axes and meta == {"energy": 1.0}
    packed = b.pack(tmp_path / "b.tar.xz")
    b2 = store.Bundle.unpack(packed, tmp_path / "unpacked")
    assert b2.read_manifest()["case_hash"] == "h"
    np.testing.assert_array_equal(b2.load_output("demo.case@quick/cpu", "wave")[0], arr)


def test_compare_verdicts(tmp_path, registry_with_demo):
    ref_arr = np.linspace(1.0, 2.0, 16).reshape(4, 4)
    cid = "demo.case@quick/cpu"
    ref = _bundle(
        tmp_path, "ref", {cid: (_record(median=1.0), {"out": (ref_arr, [], {})})}
    )

    def cand(name, arr, median=1.0, rss=100 * 1024**2):
        return _bundle(
            tmp_path,
            name,
            {cid: (_record(median=median, rss=rss), {"out": (arr, [], {})})},
        )

    identical = cmp.compare(ref, cand("c1", ref_arr.copy()), registry_with_demo)
    assert identical.rows[0].verdict == cmp.IDENTICAL

    tiny = cmp.compare(ref, cand("c2", ref_arr * (1 + 1e-13)), registry_with_demo)
    assert tiny.rows[0].verdict == cmp.OK

    drift = cmp.compare(ref, cand("c3", ref_arr * 1.01), registry_with_demo)
    assert drift.rows[0].verdict == cmp.DRIFT
    assert drift.failures(["drift"]) == [f"{cid}: DRIFT"]

    shape = cmp.compare(ref, cand("c4", ref_arr[:2]), registry_with_demo)
    assert shape.rows[0].verdict == cmp.SHAPE

    slow = cmp.compare(ref, cand("c5", ref_arr.copy(), median=1.5), registry_with_demo)
    assert "speed" in slow.rows[0].flags and slow.rows[0].time_ratio == pytest.approx(
        1.5
    )

    # a large ratio on a sub-second run is marked short, not flagged as speed
    brief_ref = _bundle(
        tmp_path, "bref", {cid: (_record(median=0.09), {"out": (ref_arr, [], {})})}
    )
    brief = cmp.compare(
        brief_ref, cand("c7", ref_arr.copy(), median=0.13), registry_with_demo
    )
    assert brief.rows[0].flags == ["short"] and brief.failures(["speed"]) == []

    fat = cmp.compare(
        ref, cand("c6", ref_arr.copy(), rss=120 * 1024**2), registry_with_demo
    )
    assert "memory" in fat.rows[0].flags and fat.rows[0].rss_ratio == pytest.approx(1.2)


def test_noise_floor_raises_flag_threshold(tmp_path, registry_with_demo):
    ref_arr = np.ones((4, 4))
    cid = "demo.case@quick/cpu"
    ref = _bundle(
        tmp_path, "ref", {cid: (_record(median=1.0), {"out": (ref_arr, [], {})})}
    )
    cand = _bundle(
        tmp_path, "cand", {cid: (_record(median=1.15), {"out": (ref_arr, [], {})})}
    )
    assert "speed" in cmp.compare(ref, cand, registry_with_demo).rows[0].flags
    noisy = cmp.compare(ref, cand, registry_with_demo, noise={cid: {"speed": 0.06}})
    assert "speed" not in noisy.rows[0].flags  # 3 x 0.06 = 0.18 > 0.15


def test_auto_variant_is_never_flagged(tmp_path, registry_with_demo):
    cid = "demo.case[auto]@quick/cpu"
    arr = np.ones((4, 4))
    ref = _bundle(tmp_path, "ref", {cid: (_record(median=1.0), {"out": (arr, [], {})})})
    cand = _bundle(
        tmp_path,
        "cand",
        {cid: (_record(median=3.0, rss=10**9), {"out": (arr, [], {})})},
    )
    row = cmp.compare(ref, cand, registry_with_demo).rows[0]
    assert row.flags == [] and row.time_ratio == pytest.approx(3.0)


def test_compare_as_pairs_variant_with_reference_default(tmp_path, registry_with_demo):
    arr = np.ones((4, 4))
    ref = _bundle(
        tmp_path, "ref", {"demo.case@quick/cpu": (_record(), {"out": (arr, [], {})})}
    )
    cand = _bundle(
        tmp_path,
        "cand",
        {"demo.case[alt]@quick/cpu": (_record(), {"out": (arr, [], {})})},
    )
    rows = cmp.compare(ref, cand, registry_with_demo).rows
    assert [(r.ref_id, r.cand_id, r.verdict, r.kind) for r in rows] == [
        (
            "demo.case@quick/cpu",
            "demo.case[alt]@quick/cpu",
            cmp.IDENTICAL,
            cmp.ATTRIBUTION,
        ),
        ("demo.case@quick/cpu", None, cmp.ONLY_A, cmp.SAME),
    ]
    # an attribution row never fails the comparison, even when it drifts
    cand2 = _bundle(
        tmp_path,
        "cand2",
        {"demo.case[alt]@quick/cpu": (_record(), {"out": (arr * 2, [], {})})},
    )
    report = cmp.compare(ref, cand2, registry_with_demo)
    assert report.rows[0].verdict == cmp.DRIFT and report.failures(["drift"]) == []


def test_unpaired_and_failed_cases(tmp_path, registry_with_demo):
    arr = np.ones((4, 4))
    ref = _bundle(
        tmp_path,
        "ref",
        {
            "demo.case@quick/cpu": (_record(), {"out": (arr, [], {})}),
            "demo.case@quick/gpu": (_record(), {"out": (arr, [], {})}),
        },
    )
    cand = _bundle(
        tmp_path,
        "cand",
        {
            "demo.case@quick/cpu": (_record(status=store.STATUS_UNSUPPORTED), None),
            "demo.case[auto]@quick/cpu": (_record(), {"out": (arr, [], {})}),
        },
    )
    verdicts = {
        (r.ref_id, r.cand_id): r.verdict
        for r in cmp.compare(ref, cand, registry_with_demo).rows
    }
    assert (
        verdicts[("demo.case@quick/cpu", "demo.case@quick/cpu")]
        == store.STATUS_UNSUPPORTED
    )
    assert verdicts[(None, "demo.case[auto]@quick/cpu")] == cmp.ONLY_B
    assert verdicts[("demo.case@quick/gpu", None)] == cmp.ONLY_A


def test_case_hash_mismatch_is_refused_unless_allowed(tmp_path, registry_with_demo):
    arr = np.ones((4, 4))
    cid = "demo.case@quick/cpu"
    ref = _bundle(
        tmp_path, "ref", {cid: (_record(), {"out": (arr, [], {})})}, case_hash="aaa"
    )
    cand = _bundle(
        tmp_path, "cand", {cid: (_record(), {"out": (arr, [], {})})}, case_hash="bbb"
    )
    with pytest.raises(cmp.CompareError, match="case_hash"):
        cmp.compare(ref, cand, registry_with_demo)
    report = cmp.compare(ref, cand, registry_with_demo, allow_case_mismatch=True)
    assert not report.case_hash_match
    assert "case_hash differs" in cmp.to_markdown(report)


def test_accepted_changes_turn_drift_into_accepted_and_report_stale(
    tmp_path, registry_with_demo
):
    arr = np.ones((4, 4))
    cid = "demo.case@quick/cpu"
    ref = _bundle(tmp_path, "ref", {cid: (_record(), {"out": (arr, [], {})})})
    cand = _bundle(tmp_path, "cand", {cid: (_record(), {"out": (arr * 1.5, [], {})})})
    toml = tmp_path / "accepted.toml"
    toml.write_text(
        '[[accepted]]\ncase = "demo.*"\nsince = "v0"\nreason = "intended"\npr = 1\n'
        '[[accepted]]\ncase = "demo.case[auto]*"\nsince = "v0"\n'
        'reason = "never matches"\n'
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.rows[0].verdict == cmp.ACCEPTED
    assert report.rows[0].accepted_by == "intended"
    assert [a.case for a in report.stale_accepted] == ["demo.case[auto]*"]
    # brackets in a glob are literal (variant names), not fnmatch classes
    assert cmp.Accepted("demo.case[auto]@*", "v0", "r").matches(
        "demo.case[auto]@quick/cpu"
    )
    assert not cmp.Accepted("demo.case[auto]@*", "v0", "r").matches(
        "demo.case@quick/cpu"
    )
    assert report.failures(["drift"]) == []
    md = cmp.to_markdown(report)
    assert "Accepted changes" in md and "Stale accepted" in md

    toml.write_text('[[accepted]]\ncase = "nosuch.case*"\nsince = "v0"\nreason = "x"\n')
    with pytest.raises(cmp.CompareError, match="matches no registered case"):
        cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)


def test_close_stats_vector():
    from abtem.core.testing import array_is_close, close_stats

    r = np.array([1.0, 1e-9, 2.0])
    c = np.array([1.0 + 1e-12, 5e-9, 2.0])
    s = close_stats(c, r, above_rel=1e-6)
    assert not s["identical"] and s["shape_ok"]
    assert s["n_checked"] == 2  # the 1e-9 element is below 1e-6 of max and excluded
    assert s["rel_above"] == pytest.approx(1e-12, rel=0.5)
    assert array_is_close(c, r, rel_tol=1e-9, check_above_rel=1e-6)
    assert not array_is_close(
        c, r, rel_tol=1e-9
    )  # the tiny element fails without the mask
    z = np.zeros(3)
    assert close_stats(z, z)["identical"] and close_stats(z, z)["max_abs_norm"] == 0.0
