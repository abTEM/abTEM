"""Store round trip and the compare verdict matrix on synthetic bundles."""

import numpy as np
import pytest
from abtem_bench import compare as cmp
from abtem_bench import store
from conftest import CID, bundle, manifest, one, record


def test_store_roundtrip(tmp_path):
    b = store.Bundle(tmp_path / "b")
    b.write_manifest(manifest())
    arr = np.arange(12, dtype=np.complex128).reshape(3, 4) * (1 + 1j)
    axes = [{"type": "RealSpaceAxis", "sampling": 0.1, "offset": 0.0, "units": "Å"}]
    b.write_case(CID, record(), {"wave": (arr, axes, {"energy": 1.0})})
    assert b.case_ids() == [CID]
    rec = b.read_case(CID)
    assert rec["outputs"]["wave"]["dtype"] == "complex128"
    assert rec["outputs"]["wave"]["invariants"]["size"] == 12
    a, ax, meta = b.load_output(CID, "wave")
    np.testing.assert_array_equal(a, arr)
    assert ax == axes and meta == {"energy": 1.0}
    packed = b.pack(tmp_path / "b.tar.xz")
    b2 = store.Bundle.unpack(packed, tmp_path / "unpacked")
    assert b2.read_manifest()["case_hash"] == "h"
    np.testing.assert_array_equal(b2.load_output(CID, "wave")[0], arr)


def test_compare_verdicts(tmp_path, registry_with_demo):
    ref_arr = np.linspace(1.0, 2.0, 16).reshape(4, 4)
    ref = one(tmp_path, "ref", ref_arr)

    def verdict(name, arr):
        return cmp.compare(ref, one(tmp_path, name, arr), registry_with_demo).rows[0]

    assert verdict("c1", ref_arr.copy()).verdict == cmp.IDENTICAL
    assert verdict("c2", ref_arr * (1 + 1e-13)).verdict == cmp.OK
    drift = cmp.compare(ref, one(tmp_path, "c3", ref_arr * 1.01), registry_with_demo)
    assert drift.rows[0].verdict == cmp.DRIFT
    assert drift.failures({"drift": None}) == [f"{CID}: DRIFT"]
    shape = verdict("c4", ref_arr[:2])
    assert shape.verdict == cmp.SHAPE and "out: shape (4, 4) → (2, 4)" in shape.notes
    dtype = verdict("c5", ref_arr.astype(np.float32))
    assert dtype.verdict == cmp.SHAPE and "out: dtype float64 → float32" in dtype.notes


def test_a_nan_in_the_candidate_is_drift(tmp_path, registry_with_demo):
    ref_arr = np.linspace(1.0, 2.0, 16).reshape(4, 4)
    cand_arr = ref_arr.copy()
    cand_arr[1, 1] = np.nan
    row = cmp.compare(
        one(tmp_path, "ref", ref_arr),
        one(tmp_path, "cand", cand_arr),
        registry_with_demo,
    ).rows[0]
    assert row.verdict == cmp.DRIFT


def test_speed_flags_and_the_short_marker(tmp_path, registry_with_demo):
    arr = np.ones((4, 4))

    def row(ref_median, cand_median, **kw):
        ref = one(tmp_path, f"r{ref_median}", arr, median=ref_median)
        cand = one(tmp_path, f"c{cand_median}", arr, median=cand_median)
        report = cmp.compare(ref, cand, registry_with_demo, **kw)
        return report, report.rows[0]

    report, slow = row(1.0, 1.5)
    assert slow.flags == ["speed"] and slow.time_ratio == pytest.approx(1.5)
    assert report.failures({"speed": None}) == [f"{CID}: time x1.50"]
    # a threshold on the gate overrides the report's
    assert report.failures({"speed": 0.6}) == []

    # a speed-up is flagged but never fails the gate
    report, fast = row(1.0, 0.5)
    assert fast.flags == ["speed"] and report.failures({"speed": None}) == []

    # a large ratio between two medians a few milliseconds apart is 'short'
    report, brief = row(0.09, 0.13)
    assert brief.flags == ["short"] and report.failures({"speed": None}) == []

    # a ten-fold regression of a sub-second case is a regression
    report, regressed = row(0.09, 0.9)
    assert regressed.flags == ["speed"]
    assert report.failures({"speed": None}) == [f"{CID}: time x10.00"]

    _, wider = row(0.09, 0.9, min_delta=1.0)
    assert wider.flags == ["short"]


def test_noise_floor_raises_flag_threshold(tmp_path, registry_with_demo):
    arr = np.ones((4, 4))
    ref = one(tmp_path, "ref", arr, median=1.0)
    cand = one(tmp_path, "cand", arr, median=1.15)
    assert "speed" in cmp.compare(ref, cand, registry_with_demo).rows[0].flags
    noisy = cmp.compare(ref, cand, registry_with_demo, noise={CID: {"speed": 0.06}})
    assert "speed" not in noisy.rows[0].flags  # 3 x 0.06 = 0.18 > 0.15


def test_memory_is_flagged_only_against_a_noise_floor(tmp_path, registry_with_demo):
    arr = np.ones((4, 4))
    ref = one(tmp_path, "ref", arr, rss=100 * 1024**2)
    cand = one(tmp_path, "cand", arr, rss=120 * 1024**2)
    bare = cmp.compare(ref, cand, registry_with_demo)
    assert bare.rows[0].rss_ratio == pytest.approx(1.2)
    assert "memory" not in bare.rows[0].flags
    assert "no noise floor" in cmp.to_markdown(bare)
    with pytest.raises(cmp.CompareError, match="noise floor"):
        bare.failures({"memory": None})

    floored = cmp.compare(ref, cand, registry_with_demo, noise={CID: {"memory": 0.01}})
    assert "memory" in floored.rows[0].flags
    assert floored.failures({"memory": None}) == [f"{CID}: rss x1.20"]
    assert floored.failures({"memory": 0.25}) == []


def test_vram_ratio_and_flag(tmp_path, registry_with_demo):
    arr = np.ones((4, 4))

    def b(name, pool):
        rec = record()
        rec["memory"]["peak_vram_pool_bytes"] = pool
        return bundle(tmp_path, name, {CID: (rec, {"out": (arr, [], {})})})

    ref, cand = b("ref", 1000), b("cand", 1500)
    row = cmp.compare(ref, cand, registry_with_demo).rows[0]
    assert row.vram_ratio == pytest.approx(1.5) and "vram" not in row.flags
    row = cmp.compare(ref, cand, registry_with_demo, noise={CID: {"vram": 0.0}}).rows[0]
    assert "vram" in row.flags
    assert "| ▲1.50 |" in cmp.to_markdown(cmp.compare(ref, cand, registry_with_demo))


def test_accuracy_floor_widens_the_tolerance(tmp_path, registry_with_demo):
    ref_arr = np.linspace(1.0, 2.0, 16).reshape(4, 4)
    ref = one(tmp_path, "ref", ref_arr)
    cand = one(tmp_path, "cand", ref_arr * (1 + 1e-8))
    assert cmp.compare(ref, cand, registry_with_demo).rows[0].verdict == cmp.DRIFT
    floor = {CID: {"accuracy": 1e-8, "accuracy_abs": 1e-8, "accuracy_intensity": 1e-8}}
    row = cmp.compare(ref, cand, registry_with_demo, noise=floor).rows[0]
    assert row.verdict == cmp.OK


def test_noise_floor_records_accuracy_spreads(tmp_path, registry_with_demo):
    arr = np.linspace(1.0, 2.0, 16).reshape(4, 4)
    a = bundle(
        tmp_path,
        "a",
        {CID: (record(median=1.0, rss=100), {"out": (arr, [], {})})},
    )
    b = bundle(
        tmp_path,
        "b",
        {CID: (record(median=1.1, rss=110), {"out": (arr * (1 + 1e-9), [], {})})},
    )
    f = cmp.noise_floor(a, b)[CID]
    assert f["speed"] == pytest.approx(abs(1.0 / 1.1 - 1.0))
    assert f["memory"] == pytest.approx(abs(100 / 110 - 1.0)) and "vram" not in f
    assert f["accuracy"] == pytest.approx(1e-9, rel=1e-3)
    assert f["accuracy_abs"] == pytest.approx(1e-9, rel=1e-3)
    assert f["accuracy_intensity"] == pytest.approx(1e-9, rel=1e-3)


def test_auto_variant_is_never_flagged(tmp_path, registry_with_demo):
    cid = "demo.case[auto]@quick/cpu"
    arr = np.ones((4, 4))
    ref = one(tmp_path, "ref", arr, cid=cid, median=1.0)
    cand = one(tmp_path, "cand", arr, cid=cid, median=3.0, rss=10**9)
    row = cmp.compare(ref, cand, registry_with_demo, noise={cid: {"memory": 0}}).rows[0]
    assert row.flags == [] and row.time_ratio == pytest.approx(3.0)


def test_compare_as_pairs_variant_with_reference_default(tmp_path, registry_with_demo):
    arr = np.ones((4, 4))
    alt = "demo.case[alt]@quick/cpu"
    ref = one(tmp_path, "ref", arr)
    cand = one(tmp_path, "cand", arr, cid=alt)
    rows = cmp.compare(ref, cand, registry_with_demo).rows
    assert [(r.ref_id, r.cand_id, r.verdict, r.kind) for r in rows] == [
        (CID, alt, cmp.IDENTICAL, cmp.ATTRIBUTION),
        (CID, None, cmp.ONLY_A, cmp.SAME),
    ]
    # an attribution row never fails the comparison, even when it drifts
    report = cmp.compare(
        ref, one(tmp_path, "cand2", arr * 2, cid=alt), registry_with_demo
    )
    assert report.rows[0].verdict == cmp.DRIFT
    assert report.failures({"drift": None, "speed": None}) == []


def test_unpaired_and_failed_cases(tmp_path, registry_with_demo):
    arr = np.ones((4, 4))
    ref = bundle(
        tmp_path,
        "ref",
        {
            CID: (record(), {"out": (arr, [], {})}),
            "demo.case@quick/gpu": (record(), {"out": (arr, [], {})}),
        },
    )
    cand = bundle(
        tmp_path,
        "cand",
        {
            CID: (record(status=store.STATUS_UNSUPPORTED), None),
            "demo.case[auto]@quick/cpu": (record(), {"out": (arr, [], {})}),
        },
    )
    report = cmp.compare(ref, cand, registry_with_demo)
    verdicts = {(r.ref_id, r.cand_id): r.verdict for r in report.rows}
    assert verdicts[(CID, CID)] == store.STATUS_UNSUPPORTED
    assert verdicts[(None, "demo.case[auto]@quick/cpu")] == cmp.ONLY_B
    assert verdicts[("demo.case@quick/gpu", None)] == cmp.ONLY_A
    # UNSUPPORTED is not a failed run; a case the candidate lacks is 'missing'
    assert report.failures({"error": None}) == []
    assert report.failures({"missing": None}) == [
        "demo.case@quick/gpu: missing from the candidate"
    ]


def test_case_hash_mismatch_is_refused_unless_allowed(tmp_path, registry_with_demo):
    arr = np.ones((4, 4))
    cases = {CID: (record(), {"out": (arr, [], {})})}
    ref = bundle(tmp_path, "ref", cases, case_hash="aaa")
    cand = bundle(tmp_path, "cand", cases, case_hash="bbb")
    with pytest.raises(cmp.CompareError, match="case_hash"):
        cmp.compare(ref, cand, registry_with_demo)
    report = cmp.compare(ref, cand, registry_with_demo, allow_case_mismatch=True)
    assert not report.case_hash_match
    assert "case_hash differs" in cmp.to_markdown(report)


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
