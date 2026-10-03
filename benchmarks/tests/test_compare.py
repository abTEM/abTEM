"""Compare's inputs and gates: accepted changes, --fail-on, refusals, rendering."""

import numpy as np
import pytest
from abtem_bench import cli, store
from abtem_bench import compare as cmp
from conftest import CID, bundle, one, record

ARR = np.linspace(1.0, 2.0, 16).reshape(4, 4)


def _toml(tmp_path, *entries):
    path = tmp_path / "accepted.toml"
    path.write_text("\n".join(f"[[accepted]]\n{e}" for e in entries))
    return path


def _drift(tmp_path, factor=1.5, label="v0", sha="0" * 40):
    ref = bundle(
        tmp_path, "ref", {CID: (record(), {"out": (ARR, [], {})})}, label=label, sha=sha
    )
    return ref, one(tmp_path, "cand", ARR * factor)


# ---------------------------------------------------------------------------
# accepted_changes.toml


def test_accepted_entry_turns_drift_into_accepted(tmp_path, registry_with_demo):
    ref, cand = _drift(tmp_path)
    toml = _toml(
        tmp_path,
        'case = "demo.*"\nsince = "v0"\nreason = "intended"\npr = 1',
        'case = "demo.case[auto]*"\nsince = "v0"\nreason = "never matches"',
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.rows[0].verdict == cmp.ACCEPTED
    assert report.rows[0].accepted_by == ["intended"]
    assert [a.case for a in report.stale_accepted] == ["demo.case[auto]*"]
    assert report.failures({"drift": None}) == []
    md = cmp.to_markdown(report)
    assert "Accepted changes" in md and "Stale accepted" in md
    # the changelog states how large the accepted drift is
    assert report.accepted[0].worst["rel_above"] == pytest.approx(0.5)
    assert "rel 5.0e-01" in md


def test_brackets_in_an_entry_glob_are_literal(registry_with_demo):
    entry = cmp.Accepted("demo.case[auto]@*", "v0", "r")
    assert entry.matches("demo.case[auto]@quick/cpu")
    assert not entry.matches("demo.case@quick/cpu")


def test_an_entry_applies_only_against_its_since_reference(
    tmp_path, registry_with_demo
):
    toml = _toml(tmp_path, 'case = "demo.*"\nsince = "v0"\nreason = "intended"')
    ref, cand = _drift(tmp_path, label="v1")
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.rows[0].verdict == cmp.DRIFT
    assert report.failures({"drift": None}) == [f"{CID}: DRIFT"]
    assert [a.case for a in report.not_applicable] == ["demo.*"]
    assert report.stale_accepted == []
    assert "other references (not applied): `demo.*` since v0" in cmp.to_markdown(
        report
    )


def test_since_matches_a_sha_prefix_of_seven_or_more(tmp_path, registry_with_demo):
    sha = "abcdef0123" + "0" * 30
    ref = {"label": "main", "describe": "x", "sha": sha}
    assert cmp.Accepted("demo.*", "abcdef0", "r").applies_to(ref)
    assert cmp.Accepted("demo.*", "ABCDEF01", "r").applies_to(ref)
    assert not cmp.Accepted("demo.*", "abcdef", "r").applies_to(ref)
    assert not cmp.Accepted("demo.*", "abcdef1", "r").applies_to(ref)
    assert cmp.Accepted("demo.*", "x", "r").applies_to(ref)  # git describe


def test_a_drift_beyond_an_entry_bound_stays_drift(tmp_path, registry_with_demo):
    ref, cand = _drift(tmp_path, factor=1.5)
    within = _toml(
        tmp_path,
        'case = "demo.*"\nsince = "v0"\nreason = "r"\nmax_rel = 0.6\n'
        "max_intensity = 0.6\nmax_abs_norm = 0.6",
    )
    assert (
        cmp.compare(ref, cand, registry_with_demo, accepted_path=within).rows[0].verdict
        == cmp.ACCEPTED
    )
    beyond = _toml(
        tmp_path, 'case = "demo.*"\nsince = "v0"\nreason = "r"\nmax_rel = 0.1'
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=beyond)
    row = report.rows[0]
    assert row.verdict == cmp.DRIFT and row.accepted_by == []
    assert "exceeds accepted bound: out rel 5.0e-01 > 1.0e-01" in row.notes
    assert report.failures({"drift": None}) == [f"{CID}: DRIFT"]
    assert report.accepted[0].exceeded == [CID] and report.stale_accepted == []
    assert "1 (1 over bound)" in cmp.to_markdown(report)


def test_a_bound_on_any_matching_entry_holds(tmp_path, registry_with_demo):
    # an unbounded broad entry does not override a narrower bounded one
    ref, cand = _drift(tmp_path, factor=1.5)
    toml = _toml(
        tmp_path,
        'case = "demo.*"\nsince = "v0"\nreason = "broad"',
        'case = "demo.case@*"\nsince = "v0"\nreason = "narrow"\nmax_rel = 0.1',
    )
    assert (
        cmp.compare(ref, cand, registry_with_demo, accepted_path=toml).rows[0].verdict
        == cmp.DRIFT
    )


def test_every_matching_entry_is_credited(tmp_path, registry_with_demo):
    ref, cand = _drift(tmp_path)
    toml = _toml(
        tmp_path,
        'case = "demo.*"\nsince = "v0"\nreason = "first"',
        'case = "demo.case@*"\nsince = "v0"\nreason = "second"',
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.rows[0].accepted_by == ["first", "second"]
    assert [a.matched for a in report.accepted] == [[CID], [CID]]
    assert report.stale_accepted == []


@pytest.mark.parametrize(
    "entry, message",
    [
        ('case = "nosuch.case*"\nsince = "v0"\nreason = "x"', "matches no registered"),
        ('case = "demo.case[nosuch]@*"\nsince = "v0"\nreason = "x"', "matches no"),
        ('case = "demo.*"\nsince = "v0"\nreason = " "', "'reason' must be"),
        ('case = "demo.*"\nsince = ""\nreason = "x"', "'since' must be"),
        ('case = "demo.*"\nreason = "x"', "'since' must be"),
        ('case = "demo.*"\nsince = 1.0\nreason = "x"', "'since' must be"),
        ('case = "demo.*"\nsince = "v0"\nreason = "x"\npr = true', "'pr' must be"),
        ('case = "demo.*"\nsince = "v0"\nreason = "x"\npr = "12"', "'pr' must be"),
        ('case = "demo.*"\nsince = "v0"\nreason = "x"\nmax_rel = -1', "'max_rel'"),
        ('case = "demo.*"\nsince = "v0"\nreason = "x"\nmax_rel = nan', "'max_rel'"),
        ('case = "demo.*"\nsince = "v0"\nreason = "x"\nmax_rell = 1', "unknown keys"),
    ],
)
def test_invalid_entries_are_refused(tmp_path, registry_with_demo, entry, message):
    ref, cand = _drift(tmp_path)
    with pytest.raises(cmp.CompareError, match=message):
        cmp.compare(ref, cand, registry_with_demo, accepted_path=_toml(tmp_path, entry))


def test_the_shipped_accepted_changes_file_is_valid():
    from abtem_bench import registry

    entries = cmp.load_accepted(cmp.ACCEPTED_CHANGES_PATH, registry.load_cases())
    assert entries and all(a.pr for a in entries)


# ---------------------------------------------------------------------------
# --fail-on and refusals


def test_parse_fail_on():
    assert cmp.parse_fail_on(None) == {}
    assert cmp.parse_fail_on(" drift , speed:10% ,memory:2.5%") == {
        "drift": None,
        "speed": 0.1,
        "memory": 0.025,
    }
    for bad, message in (
        ("drfit", "unknown gate"),
        ("drift:5%", "takes no threshold"),
        ("speed:10", "positive percentage"),
        ("speed:x%", "positive percentage"),
        ("speed:-5%", "positive percentage"),
        ("speed:nan%", "positive percentage"),
    ):
        with pytest.raises(cmp.CompareError, match=message):
            cmp.parse_fail_on(bad)


def test_error_gate_counts_only_candidate_failures(tmp_path, registry_with_demo):
    ok = (record(), {"out": (ARR, [], {})})

    def failed(status):
        return (record(status=status, error="Traceback\nValueError: boom"), None)

    gpu = "demo.case@quick/gpu"
    ref = bundle(tmp_path, "ref", {CID: failed(store.STATUS_ERROR), gpu: ok})
    cand = bundle(tmp_path, "cand", {CID: ok, gpu: failed(store.STATUS_SKIPPED_MEMORY)})
    report = cmp.compare(ref, cand, registry_with_demo)
    rows = {r.cand_id: r for r in report.rows}
    assert rows[CID].verdict == store.STATUS_ERROR and rows[CID].failed == "reference"
    assert rows[CID].notes == ["reference ERROR: ValueError: boom"]
    assert rows[gpu].failed == "candidate"
    assert report.failures({"error": None}) == [f"{gpu}: SKIPPED-MEMORY"]


def test_bundles_without_a_common_case_are_refused(tmp_path, registry_with_demo):
    ref = one(tmp_path, "ref", ARR)
    cand = one(tmp_path, "cand", ARR, cid="demo.case@quick/gpu")
    with pytest.raises(cmp.CompareError, match="share no case id"):
        cmp.compare(ref, cand, registry_with_demo)


def test_a_preset_mismatch_is_refused_unless_allowed(tmp_path, registry_with_demo):
    cases = {CID: (record(), {"out": (ARR, [], {})})}
    ref = bundle(tmp_path, "ref", cases)
    cand = bundle(tmp_path, "cand", cases, preset="speed")
    with pytest.raises(cmp.CompareError, match="presets differ"):
        cmp.compare(ref, cand, registry_with_demo)
    report = cmp.compare(ref, cand, registry_with_demo, allow_preset_mismatch=True)
    assert not report.preset_match and "presets differ" in cmp.to_markdown(report)


def test_cli_input_errors_exit_2(tmp_path, monkeypatch, capsys):
    # The real registry: the CLI loads the shipped cases, which must not register
    # into a test's temporary registry.
    monkeypatch.setattr(cmp, "ACCEPTED_CHANGES_PATH", tmp_path / "none.toml")
    ref, cand = _drift(tmp_path)
    base = ["compare", str(ref.path), str(cand.path)]
    assert cli.main(base + ["--fail-on", "drfit"]) == 2
    assert "unknown gate" in capsys.readouterr().err
    assert cli.main(base + ["--fail-on", "memory"]) == 2
    assert "needs a noise floor" in capsys.readouterr().err
    assert cli.main(base + ["--noise", str(tmp_path / "nowhere")]) == 2
    assert cli.main(["compare", str(ref.path), str(tmp_path / "nothing")]) == 2
    assert "not a bundle" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# axes and rendering

AXIS = {
    "type": "RealSpaceAxis",
    "label": "x",
    "units": "Å",
    "sampling": 0.1,
    "offset": 0.0,
    "_concatenate": True,
}


def _with_axes(tmp_path, name, axes):
    return one(tmp_path, name, ARR, axes=axes)


def test_axis_labels_are_informational(tmp_path, registry_with_demo):
    ref = _with_axes(tmp_path, "ref", [AXIS, AXIS])
    relabelled = {**AXIS, "label": "y", "_concatenate": False, "endpoint": False}
    row = cmp.compare(
        ref, _with_axes(tmp_path, "cand", [AXIS, relabelled]), registry_with_demo
    ).rows[0]
    assert row.verdict == cmp.IDENTICAL
    assert row.notes == ["out: axis metadata differs (3)"]


def test_axis_numbers_compare_to_a_relative_tolerance(tmp_path, registry_with_demo):
    ref = _with_axes(tmp_path, "ref", [AXIS, AXIS])
    rounded = {**AXIS, "sampling": 0.1 * (1 + 1e-13), "offset": 1e-17}
    row = cmp.compare(
        ref, _with_axes(tmp_path, "c1", [AXIS, rounded]), registry_with_demo
    ).rows[0]
    assert row.verdict == cmp.IDENTICAL
    resampled = {**AXIS, "sampling": 0.2}
    row = cmp.compare(
        ref, _with_axes(tmp_path, "c2", [AXIS, resampled]), registry_with_demo
    ).rows[0]
    assert row.verdict == cmp.SHAPE
    assert row.notes == ["out: axis 1 sampling 0.1 → 0.2"]
    units = {**AXIS, "units": "nm"}
    row = cmp.compare(
        ref, _with_axes(tmp_path, "c3", [AXIS, units]), registry_with_demo
    ).rows[0]
    assert row.verdict == cmp.SHAPE


def test_worst_output_is_named_and_cells_are_escaped(tmp_path, registry_with_demo):
    ref = bundle(
        tmp_path,
        "ref",
        {CID: (record(), {"a": (ARR, [], {}), "b": (ARR, [], {})})},
        label="v0",
    )
    cand = bundle(
        tmp_path,
        "cand",
        {CID: (record(), {"a": (ARR * (1 + 1e-3), [], {}), "b": (ARR * 1.5, [], {})})},
    )
    toml = _toml(tmp_path, 'case = "demo.*"\nsince = "v0"\nreason = "a | b"')
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    md = cmp.to_markdown(report)
    assert "| 5.0e-01 (b) |" in md
    assert "accepted: a \\| b" in md and "| a \\| b |" in md
