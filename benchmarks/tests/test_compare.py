"""Compare's inputs and gates: accepted changes, --fail-on, refusals, rendering."""

import numpy as np
import pytest
from abtem_bench import cli, store
from abtem_bench import compare as cmp
from conftest import CID, bundle, one, record

ARR = np.linspace(1.0, 2.0, 16).reshape(4, 4)
#: the bound every accepted entry must set; loose enough for the drifts below
LOOSE = "max_abs_norm = 10"


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
        f'case = "demo.*"\nsince = "v0"\nreason = "intended"\npr = 1\n{LOOSE}',
        f'case = "demo.case[auto]*"\nsince = "v0"\nreason = "never matches"\n{LOOSE}',
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
    toml = _toml(
        tmp_path, f'case = "demo.*"\nsince = "v0"\nreason = "intended"\n{LOOSE}'
    )
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
        tmp_path,
        f'case = "demo.*"\nsince = "v0"\nreason = "r"\nmax_rel = 0.1\n{LOOSE}',
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
        f'case = "demo.*"\nsince = "v0"\nreason = "broad"\n{LOOSE}',
        f'case = "demo.case@*"\nsince = "v0"\nreason = "narrow"\nmax_rel = 0.1\n'
        f"{LOOSE}",
    )
    assert (
        cmp.compare(ref, cand, registry_with_demo, accepted_path=toml).rows[0].verdict
        == cmp.DRIFT
    )


def test_every_matching_entry_is_credited(tmp_path, registry_with_demo):
    ref, cand = _drift(tmp_path)
    toml = _toml(
        tmp_path,
        f'case = "demo.*"\nsince = "v0"\nreason = "first"\n{LOOSE}',
        f'case = "demo.case@*"\nsince = "v0"\nreason = "second"\n{LOOSE}',
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.rows[0].accepted_by == ["first", "second"]
    assert [a.matched for a in report.accepted] == [[CID], [CID]]
    assert report.stale_accepted == []


@pytest.mark.parametrize(
    "entry, message",
    [
        (
            f'case = "nosuch.case*"\nsince = "v0"\nreason = "x"\n{LOOSE}',
            "matches no registered",
        ),
        (
            f'case = "demo.case[nosuch]@*"\nsince = "v0"\nreason = "x"\n{LOOSE}',
            "matches no",
        ),
        ('case = "demo.*"\nsince = "v0"\nreason = " "', "'reason' must be"),
        ('case = "demo.*"\nsince = ""\nreason = "x"', "'since' must be"),
        ('case = "demo.*"\nreason = "x"', "'since' must be"),
        ('case = "demo.*"\nsince = 1.0\nreason = "x"', "'since' must be"),
        ('case = "demo.*"\nsince = "v0"\nreason = "x"\npr = true', "'pr' must be"),
        ('case = "demo.*"\nsince = "v0"\nreason = "x"\npr = "12"', "'pr' must be"),
        ('case = "demo.*"\nsince = "v0"\nreason = "x"\nmax_rel = -1', "'max_rel'"),
        ('case = "demo.*"\nsince = "v0"\nreason = "x"\nmax_rel = nan', "'max_rel'"),
        ('case = "demo.*"\nsince = "v0"\nreason = "x"\nmax_rell = 1', "unknown keys"),
        ('case = "demo.*"\nsince = "v0"\nreason = "x"', "'max_abs_norm' is required"),
        (
            'case = "demo.*"\nsince = "v0"\nreason = "x"\nmax_rel = 1\n'
            "max_intensity = 1",
            "'max_abs_norm' is required",
        ),
    ],
)
def test_invalid_entries_are_refused(tmp_path, registry_with_demo, entry, message):
    ref, cand = _drift(tmp_path)
    with pytest.raises(cmp.CompareError, match=message):
        cmp.compare(ref, cand, registry_with_demo, accepted_path=_toml(tmp_path, entry))


@pytest.mark.xfail(
    strict=False, reason="shipped file gains max_abs_norm bounds separately"
)
def test_the_shipped_accepted_changes_file_is_valid():
    from abtem_bench import registry

    entries = cmp.load_accepted(cmp.ACCEPTED_CHANGES_PATH, registry.load_cases())
    assert entries and all(a.pr for a in entries)


def _arrays(tmp_path, ref_arr, cand_arr):
    """Reference captured at ``v0`` and a candidate holding one output each."""
    ref = bundle(
        tmp_path, "ref", {CID: (record(), {"out": (ref_arr, [], {})})}, label="v0"
    )
    return ref, one(tmp_path, "cand", cand_arr)


def _corrupted(kind):
    """(reference, candidate) arrays whose integrated intensity is unchanged."""
    if kind == "roll":
        return ARR, np.roll(ARR, 1, axis=0)
    if kind == "flip":
        return ARR, ARR[::-1].copy()
    if kind == "phase":
        wave = ARR * np.exp(1j * ARR)
        return wave, wave * np.exp(1j * np.linspace(0.0, 2.0, 16).reshape(4, 4))
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["roll", "flip", "phase"])
def test_an_entry_without_max_abs_norm_is_refused_not_applied(
    tmp_path, registry_with_demo, kind
):
    # The integrated intensity is invariant under a shift, a flip or a phase
    # scramble, so an intensity bound alone would accept the corrupted output.
    ref_arr, cand_arr = _corrupted(kind)
    ref, cand = _arrays(tmp_path, ref_arr, cand_arr)
    intensity_only = _toml(
        tmp_path,
        'case = "demo.*"\nsince = "v0"\nreason = "r"\nmax_intensity = 0.5',
    )
    with pytest.raises(cmp.CompareError, match="'max_abs_norm' is required"):
        cmp.compare(ref, cand, registry_with_demo, accepted_path=intensity_only)
    # with the required bound, the corruption exceeds it and stays DRIFT
    bounded = _toml(
        tmp_path,
        'case = "demo.*"\nsince = "v0"\nreason = "r"\nmax_intensity = 0.5\n'
        "max_abs_norm = 0.05",
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=bounded)
    assert report.rows[0].verdict == cmp.DRIFT
    assert any("exceeds accepted bound" in n for n in report.rows[0].notes)
    assert report.failures({"drift": None}) == [f"{CID}: DRIFT"]


def test_a_non_finite_mismatch_is_never_accepted(tmp_path, registry_with_demo):
    ref_arr = ARR.copy()
    nan_arr = ARR.copy()
    nan_arr[1, 1] = np.nan
    toml = _toml(
        tmp_path,
        'case = "demo.*"\nsince = "v0"\nreason = "r"\nmax_rel = 1e30\n'
        "max_intensity = 1e30\nmax_abs_norm = 1e30",
    )
    ref, cand = _arrays(tmp_path, ref_arr, nan_arr)
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.rows[0].verdict == cmp.DRIFT
    # the entry's bounds as such do not see it: a NaN is masked out of the stats
    entry = cmp.Accepted("demo.*", "v0", "r", max_abs_norm=1e30, max_intensity=1e30)
    stats = {
        "identical": False,
        "nonfinite_mismatch": 1,
        "max_abs_norm": 0.0,
        "intensity": 0.0,
    }
    assert entry.violations({"out": stats}) == [
        "out has 1 non-finite values that differ"
    ]
    stats["nonfinite_mismatch"] = 0
    assert entry.violations({"out": stats}) == []


@pytest.mark.parametrize(
    "text", ["[accepted]\ncase = 'demo.*'", "accepted = 3", "accepted = [1, 2]"]
)
def test_accepted_must_be_an_array_of_tables(tmp_path, text):
    path = tmp_path / "accepted.toml"
    path.write_text(text)
    with pytest.raises(cmp.CompareError, match=r"must be an array of tables"):
        cmp.load_accepted(path)


def test_an_explicit_accepted_path_must_exist(tmp_path, registry_with_demo):
    ref, cand = _drift(tmp_path)
    with pytest.raises(cmp.CompareError, match="no-such.toml"):
        cmp.compare(
            ref, cand, registry_with_demo, accepted_path=tmp_path / "no-such.toml"
        )


def test_a_missing_default_accepted_file_is_reported(tmp_path, registry_with_demo):
    # registry_with_demo points the default at a file that does not exist
    ref, cand = _drift(tmp_path)
    report = cmp.compare(ref, cand, registry_with_demo)
    assert report.rows[0].verdict == cmp.DRIFT
    assert not report.accepted_exists
    md = cmp.to_markdown(report)
    assert f"Note: no accepted_changes.toml at {report.accepted_path}; " in md
    assert "no drift is accepted." in md
    out = cmp.to_json(report)
    assert out["accepted_path"] == str(report.accepted_path)
    assert out["accepted_exists"] is False
    present = _toml(tmp_path, f'case = "demo.*"\nsince = "v0"\nreason = "r"\n{LOOSE}')
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=present)
    assert report.accepted_exists and "no accepted_changes.toml" not in cmp.to_markdown(
        report
    )


# ---------------------------------------------------------------------------
# --fail-on and refusals


@pytest.mark.parametrize("text", ["", " ", ",", " , ,"])
def test_a_fail_on_that_names_no_gate_is_refused(text):
    with pytest.raises(cmp.CompareError, match="--fail-on names no gate"):
        cmp.parse_fail_on(text)


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


def test_candidate_only_and_reference_only_failures_fail_the_error_gate(
    tmp_path, registry_with_demo
):
    ok = (record(), {"out": (ARR, [], {})})

    def failed(status):
        return (record(status=status, error="Traceback\nValueError: boom"), None)

    ref_only, cand_only = "demo.case@quick/gpu", "demo.case[auto]@quick/cpu"
    ref = bundle(tmp_path, "ref", {CID: ok, ref_only: failed(store.STATUS_OOM)})
    cand = bundle(tmp_path, "cand", {CID: ok, cand_only: failed(store.STATUS_ERROR)})
    report = cmp.compare(ref, cand, registry_with_demo)
    rows = {r.case: r for r in report.rows}
    assert rows[cand_only].verdict == cmp.ONLY_B
    assert rows[cand_only].failed == "candidate"
    assert rows[cand_only].notes == ["candidate ERROR: ValueError: boom"]
    assert rows[ref_only].verdict == cmp.ONLY_A and rows[ref_only].failed == "reference"
    assert rows[ref_only].notes == ["reference OOM: ValueError: boom"]
    assert report.failures({"error": None}) == [f"{cand_only}: ERROR"]
    # a reference-only failure is a missing result, not a candidate error
    assert report.failures({"missing": None}) == [
        f"{ref_only}: missing from the candidate"
    ]


def test_an_unsupported_candidate_fails_the_missing_gate(tmp_path, registry_with_demo):
    ok = (record(), {"out": (ARR, [], {})})
    unsupported = (record(status=store.STATUS_UNSUPPORTED), None)
    ref = bundle(tmp_path, "ref", {CID: ok})
    cand = bundle(tmp_path, "cand", {CID: unsupported})
    report = cmp.compare(ref, cand, registry_with_demo)
    assert report.failures({"missing": None}) == [
        f"{CID}: UNSUPPORTED on the candidate, OK on the reference"
    ]
    assert report.failures({"error": None, "drift": None}) == []
    # an unsupported reference is not a loss
    report = cmp.compare(cand, ref, registry_with_demo)
    assert report.failures({"missing": None}) == []


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
    toml = _toml(tmp_path, f'case = "demo.*"\nsince = "v0"\nreason = "a | b"\n{LOOSE}')
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    md = cmp.to_markdown(report)
    assert "| 5.0e-01 (b) |" in md
    assert "accepted: a \\| b" in md and "| a \\| b |" in md


# ---------------------------------------------------------------------------
# noise file, manifests, memory not judged


@pytest.mark.parametrize(
    "noise",
    [[1, 2], {"a": 1}, {"a": [1]}, {"a": {"memory": "x"}}, {"a": {"memory": None}}],
)
def test_a_malformed_noise_file_is_refused(noise):
    with pytest.raises(cmp.CompareError, match="noise"):
        cmp.check_noise(noise, "noise.json")


def test_a_well_formed_noise_file_passes():
    noise = {CID: {"memory": 0.01, "speed": 0}}
    assert cmp.check_noise(noise, "noise.json") == noise


def test_the_report_renders_a_manifest_without_optional_fields(
    tmp_path, registry_with_demo
):
    ref, cand = _drift(tmp_path)
    report = cmp.compare(ref, cand, registry_with_demo)
    for m in (report.reference, report.candidate):
        m["ref"] = {"label": "x"}
        for key in ("fingerprint", "timestamp_utc"):
            m.pop(key)
    md = cmp.to_markdown(report)
    assert "`x` (?, ?)" in md and "captured ? on ?" in md
    assert cmp.to_json(report)["reference"]["ref"] == {"label": "x"}


def test_memory_is_reported_as_not_judged_for_cases_without_a_floor(
    tmp_path, registry_with_demo
):
    arr = np.ones((4, 4))
    other = "demo.case@standard/cpu"
    cases = {
        CID: (record(), {"out": (arr, [], {})}),
        other: (record(), {"out": (arr, [], {})}),
    }
    ref, cand = bundle(tmp_path, "ref", cases), bundle(tmp_path, "cand", cases)
    note = "memory not judged for 1 case ids without a floor in the noise file"
    report = cmp.compare(ref, cand, registry_with_demo, noise={CID: {"memory": 0.01}})
    assert report.memory_unjudged == 1 and note in cmp.to_markdown(report)
    assert cmp.to_json(report)["memory_unjudged"] == 1
    # every case has a floor, or there is no noise file at all: nothing to say
    both = {CID: {"memory": 0.01}, other: {"memory": 0.01}}
    report = cmp.compare(ref, cand, registry_with_demo, noise=both)
    assert report.memory_unjudged == 0 and "not judged" not in cmp.to_markdown(report)
    assert "not judged" not in cmp.to_markdown(
        cmp.compare(ref, cand, registry_with_demo)
    )
