"""Compare's inputs and gates: accepted changes, --fail-on, refusals, rendering."""

import subprocess

import numpy as np
import pytest
from abtem_bench import cli, registry, store
from abtem_bench import compare as cmp
from abtem_bench.compare import _contains as real_contains  # before patching
from abtem_bench.compare import _resolve as real_resolve
from abtem_bench.registry import Tier
from conftest import CID, REFERENCE_SHA, bundle, one, record

ARR = np.linspace(1.0, 2.0, 16).reshape(4, 4)
#: the bound every accepted entry must set; loose enough for the drifts below
LOOSE = "max_abs_norm = 10"


#: the commit an entry names when a test does not name one
COMMIT = "abcdef0"


def _toml(tmp_path, *entries):
    """An accepted-changes file; entries without a ``commit`` get ``COMMIT``."""
    path = tmp_path / "accepted.toml"
    path.write_text(
        "\n".join(
            f"[[accepted]]\n{e}" + ("" if "commit =" in e else f'\ncommit = "{COMMIT}"')
            for e in entries
        )
    )
    return path


def _drift(tmp_path, factor=1.5, label="v0"):
    ref = bundle(
        tmp_path, "ref", {CID: (record(), {"out": (ARR, [], {})})}, label=label
    )
    return ref, one(tmp_path, "cand", ARR * factor)


# ---------------------------------------------------------------------------
# accepted_changes.toml


def test_accepted_entry_turns_drift_into_accepted(tmp_path, registry_with_demo):
    ref, cand = _drift(tmp_path)
    toml = _toml(
        tmp_path,
        f'case = "demo.*"\nreason = "intended"\npr = 1\n{LOOSE}',
        f'case = "demo.case[auto]*"\nreason = "never matches"\n{LOOSE}',
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.rows[0].verdict == cmp.ACCEPTED
    assert report.rows[0].accepted_by == ["intended"]
    # no [auto] id was compared, so its entry is not stale
    assert report.stale_accepted == []
    assert report.failures({"drift": None}) == []
    md = cmp.to_markdown(report)
    assert "Accepted changes" in md and "Stale accepted" not in md
    # the changelog states how large the accepted drift is
    assert report.accepted[0].worst["rel_above"] == pytest.approx(0.5)
    assert "rel 5.0e-01" in md


def test_an_entry_for_a_compared_case_without_drift_is_stale(
    tmp_path, registry_with_demo
):
    gpu = "demo.case@quick/gpu"
    ref = bundle(
        tmp_path,
        "ref",
        {
            CID: (record(), {"out": (ARR, [], {})}),
            gpu: (record(), {"out": (ARR, [], {})}),
        },
        label="v0",
    )
    cand = bundle(
        tmp_path,
        "cand",
        {
            CID: (record(), {"out": (ARR * 1.5, [], {})}),
            gpu: (record(), {"out": (ARR, [], {})}),
        },
    )
    toml = _toml(
        tmp_path,
        f'case = "demo.case@quick/cpu"\nreason = "r"\n{LOOSE}',
        f'case = "demo.case@quick/gpu"\nreason = "gone"\n{LOOSE}',
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert [a.case for a in report.stale_accepted] == ["demo.case@quick/gpu"]
    assert "Stale accepted" in cmp.to_markdown(report)


def test_the_changelog_lists_only_entries_for_compared_cases(
    tmp_path, registry_with_demo
):
    ref, cand = _drift(tmp_path)
    toml = _toml(
        tmp_path,
        f'case = "demo.case@quick/*"\nreason = "here"\n{LOOSE}',
        f'case = "demo.case@standard/*"\nreason = "elsewhere"\n{LOOSE}',
    )
    md = cmp.to_markdown(cmp.compare(ref, cand, registry_with_demo, accepted_path=toml))
    assert "| here |" in md and "elsewhere" not in md
    assert "max_abs ≤ 1.0e+01" in md


def test_brackets_in_an_entry_glob_are_literal(registry_with_demo):
    entry = cmp.Accepted("demo.case[auto]@*", COMMIT, "r")
    assert entry.matches("demo.case[auto]@quick/cpu")
    assert not entry.matches("demo.case@quick/cpu")


_CANDIDATE_LACKS = "the candidate does not contain abcdef0"
_CANDIDATE_UNKNOWN = "git could not tell whether the candidate contains abcdef0"
_REFERENCE_UNKNOWN = "git could not tell whether the reference contains abcdef0"
_UNRESOLVED = "git could not resolve abcdef0 in the harness's repository"


@pytest.mark.parametrize(
    "resolves, in_cand, in_ref, note",
    [
        (True, False, False, _CANDIDATE_LACKS),
        (True, None, False, _CANDIDATE_UNKNOWN),
        (False, None, None, _UNRESOLVED),
        (True, True, None, _REFERENCE_UNKNOWN),
        (True, True, True, "the reference already contains abcdef0"),
    ],
)
def test_an_entry_applies_only_to_a_candidate_with_its_commit_and_a_reference_without(
    tmp_path, registry_with_demo, monkeypatch, resolves, in_cand, in_ref, note
):
    resolved, asked = [], []

    def fake_resolve(commit):
        resolved.append(commit)
        return commit if resolves else None

    def fake_contains(commit, sha):
        asked.append((commit, sha))
        return in_ref if sha == REFERENCE_SHA else in_cand

    monkeypatch.setattr(cmp, "_resolve", fake_resolve)
    monkeypatch.setattr(cmp, "_contains", fake_contains)
    ref, cand = _drift(tmp_path)
    # two entries of one commit: git is asked once for the prefix and once per bundle
    toml = _toml(
        tmp_path,
        f'case = "demo.*"\nreason = "r"\n{LOOSE}',
        f'case = "demo.case*"\nreason = "s"\n{LOOSE}',
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    cand_sha = cand.read_manifest()["ref"]["sha"]
    assert resolved == [COMMIT]
    expected = [(COMMIT, cand_sha)] if resolves else []
    if resolves and in_cand:
        expected.append((COMMIT, REFERENCE_SHA))
    assert asked == expected
    assert report.rows[0].verdict == cmp.DRIFT
    assert [a.scope_note for a in report.not_applicable] == [note] * 2
    unchecked = None in (in_cand, in_ref) or not resolves
    assert [a.unchecked for a in report.not_applicable] == [unchecked] * 2
    assert f"`demo.*` ({note})" in cmp.to_markdown(report)
    assert cmp.to_json(report)["not_applicable"][0]["scope_note"] == note


@pytest.fixture
def history(tmp_path, monkeypatch):
    """Four commits of a real repository standing in for the harness's, and git's
    own answers in place of the fakes: ``[first, second, third, fourth]``."""

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
            + ["-c", "commit.gpgsign=false", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    shas = []
    for msg in "abcd":
        git("commit", "-q", "--allow-empty", "-m", msg)
        shas.append(git("rev-parse", "HEAD"))
    monkeypatch.setattr(cmp.prepare, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(cmp, "_resolve", real_resolve)
    monkeypatch.setattr(cmp, "_contains", real_contains)
    return shas


def _compare_at(tmp_path, registry, toml, label, ref_sha, cand_sha):
    ref = bundle(
        tmp_path,
        f"ref-{label}",
        {CID: (record(), {"out": (ARR, [], {})})},
        label=label,
        sha=ref_sha,
    )
    cand = bundle(
        tmp_path,
        f"cand-{label}",
        {CID: (record(), {"out": (ARR * 1.5, [], {})})},
        sha=cand_sha,
    )
    return cmp.compare(ref, cand, registry, accepted_path=toml)


def test_an_entry_applies_against_every_reference_that_lacks_its_commit(
    tmp_path, registry_with_demo, history
):
    first, second, third, fourth = history
    toml = _toml(
        tmp_path, f'case = "demo.*"\nreason = "r"\ncommit = "{third[:8]}"\n{LOOSE}'
    )
    for label, ref_sha in (("v1.0.10", first), ("v1.0.11", second)):
        report = _compare_at(tmp_path, registry_with_demo, toml, label, ref_sha, fourth)
        assert report.rows[0].verdict == cmp.ACCEPTED, label
        assert report.not_applicable == [] and report.accepted[0].change == third


def test_an_entry_does_not_apply_when_the_reference_contains_its_commit(
    tmp_path, registry_with_demo, history
):
    first, second, third, fourth = history
    toml = _toml(
        tmp_path, f'case = "demo.*"\nreason = "r"\ncommit = "{third[:8]}"\n{LOOSE}'
    )
    for label, ref_sha in (("at", third), ("after", fourth)):
        report = _compare_at(tmp_path, registry_with_demo, toml, label, ref_sha, fourth)
        assert report.rows[0].verdict == cmp.DRIFT, label
        note = f"the reference already contains {third[:8]}"
        assert [a.scope_note for a in report.not_applicable] == [note]
        assert not report.not_applicable[0].unchecked
    # and not when the candidate lacks it
    report = _compare_at(tmp_path, registry_with_demo, toml, "old", first, second)
    assert report.rows[0].verdict == cmp.DRIFT
    assert report.not_applicable[0].scope_note == (
        f"the candidate does not contain {third[:8]}"
    )


def test_an_entry_is_not_applied_when_git_cannot_check_the_reference(
    tmp_path, registry_with_demo, history
):
    first, second, third, fourth = history
    toml = _toml(
        tmp_path, f'case = "demo.*"\nreason = "r"\ncommit = "{third[:8]}"\n{LOOSE}'
    )
    unknown = "1234567" + "0" * 33  # a sha this repository does not have
    report = _compare_at(tmp_path, registry_with_demo, toml, "x", unknown, fourth)
    assert report.rows[0].verdict == cmp.DRIFT
    (entry,) = report.not_applicable
    assert entry.unchecked
    assert entry.scope_note == (
        f"git could not tell whether the reference contains {third[:8]}"
    )


def test_git_answers_for_a_prefix_and_for_ancestry(tmp_path, history, monkeypatch):
    first, second = history[:2]
    assert cmp._resolve(first[:7]) == first
    assert cmp._resolve("1234567") is None  # unknown to git
    assert cmp._contains(first, second) is True
    assert cmp._contains(second, first) is False

    def no_repo():
        raise FileNotFoundError("no .git")

    monkeypatch.setattr(cmp.prepare, "repo_root", no_repo)
    assert cmp._resolve(first) is None
    assert cmp._contains(first, second) is None


def test_of_two_changes_the_one_whose_entries_hold_accepts_the_drift(
    tmp_path, registry_with_demo
):
    # dev holds both a large change and a small ported one: the small one's tight
    # bound does not turn the large one's accepted drift back into DRIFT
    ref, cand = _drift(tmp_path, factor=1.5)
    toml = _toml(
        tmp_path,
        f'case = "demo.*"\ncommit = "aaaaaaa"\nreason = "large"\n{LOOSE}',
        'case = "demo.*"\ncommit = "bbbbbbb"\nreason = "small"\nmax_abs_norm = 1e-3',
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.rows[0].verdict == cmp.ACCEPTED
    assert report.rows[0].accepted_by == ["large"]
    large, small = report.accepted
    assert large.matched == [CID] and small.matched == []
    assert small.dominated == [CID] and small.exceeded == []
    assert report.stale_accepted == []
    assert "| 0 (1 dominated) |" in cmp.to_markdown(report)
    assert cmp.to_json(report)["accepted"][1]["dominated"] == [CID]


def test_prefixes_of_one_commit_are_one_change(
    tmp_path, registry_with_demo, monkeypatch
):
    # a loose entry naming the commit with a longer prefix is still bound by the
    # tight entry of the same change
    monkeypatch.setattr(cmp, "_resolve", lambda commit: "abcdef0" + "1" * 33)
    ref, cand = _drift(tmp_path, factor=1.5)
    toml = _toml(
        tmp_path,
        'case = "demo.*"\ncommit = "abcdef0"\nreason = "tight"\nmax_abs_norm = 1e-3',
        f'case = "demo.*"\ncommit = "abcdef01"\nreason = "loose"\n{LOOSE}',
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.rows[0].verdict == cmp.DRIFT
    assert [a.exceeded for a in report.accepted] == [[CID], [CID]]


def test_a_drift_beyond_every_change_stays_drift(tmp_path, registry_with_demo):
    ref, cand = _drift(tmp_path, factor=1.5)
    toml = _toml(
        tmp_path,
        'case = "demo.*"\ncommit = "aaaaaaa"\nreason = "a"\nmax_abs_norm = 1e-3',
        'case = "demo.*"\ncommit = "bbbbbbb"\nreason = "b"\nmax_abs_norm = 1e-2',
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.rows[0].verdict == cmp.DRIFT
    assert [a.exceeded for a in report.accepted] == [[CID], [CID]]


def test_a_drift_beyond_an_entry_bound_stays_drift(tmp_path, registry_with_demo):
    ref, cand = _drift(tmp_path, factor=1.5)
    within = _toml(
        tmp_path,
        'case = "demo.*"\nreason = "r"\nmax_rel = 0.6\n'
        "max_intensity = 0.6\nmax_abs_norm = 0.6",
    )
    assert (
        cmp.compare(ref, cand, registry_with_demo, accepted_path=within).rows[0].verdict
        == cmp.ACCEPTED
    )
    beyond = _toml(
        tmp_path,
        f'case = "demo.*"\nreason = "r"\nmax_rel = 0.1\n{LOOSE}',
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
        f'case = "demo.*"\nreason = "broad"\n{LOOSE}',
        f'case = "demo.case@*"\nreason = "narrow"\nmax_rel = 0.1\n{LOOSE}',
    )
    assert (
        cmp.compare(ref, cand, registry_with_demo, accepted_path=toml).rows[0].verdict
        == cmp.DRIFT
    )


def test_every_matching_entry_is_credited(tmp_path, registry_with_demo):
    ref, cand = _drift(tmp_path)
    toml = _toml(
        tmp_path,
        f'case = "demo.*"\nreason = "first"\n{LOOSE}',
        f'case = "demo.case@*"\nreason = "second"\n{LOOSE}',
    )
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.rows[0].accepted_by == ["first", "second"]
    assert [a.matched for a in report.accepted] == [[CID], [CID]]
    assert report.stale_accepted == []


def test_an_entry_without_a_commit_is_refused(tmp_path, registry_with_demo):
    ref, cand = _drift(tmp_path)
    path = tmp_path / "accepted.toml"
    path.write_text(f'[[accepted]]\ncase = "demo.*"\nreason = "x"\n{LOOSE}')
    with pytest.raises(cmp.CompareError, match="'commit' must be the sha"):
        cmp.compare(ref, cand, registry_with_demo, accepted_path=path)


@pytest.mark.parametrize(
    "entry, message",
    [
        (
            f'case = "nosuch.case*"\nreason = "x"\n{LOOSE}',
            "matches no registered",
        ),
        (
            f'case = "demo.case[nosuch]@*"\nreason = "x"\n{LOOSE}',
            "matches no",
        ),
        ('case = "demo.*"\nreason = " "', "'reason' must be"),
        ('case = "demo.*"\nsince = "v0"\nreason = "x"', "unknown keys \\['since'\\]"),
        ('case = "demo.*"\nreason = "x"\npr = true', "'pr' must be"),
        ('case = "demo.*"\nreason = "x"\npr = "12"', "'pr' must be"),
        ('case = "demo.*"\nreason = "x"\nmax_rel = -1', "'max_rel'"),
        ('case = "demo.*"\nreason = "x"\nmax_rel = nan', "'max_rel'"),
        ('case = "demo.*"\nreason = "x"\nmax_rell = 1', "unknown keys"),
        ('case = "demo.*"\nreason = "x"', "'max_abs_norm' is required"),
        (
            f'case = "demo.*"\ncommit = ""\nreason = "x"\n{LOOSE}',
            "'commit'",
        ),
        (
            f'case = "demo.*"\ncommit = "abc"\nreason = "x"\n{LOOSE}',
            "'commit'",
        ),
        (
            f'case = "demo.*"\ncommit = "{"a" * 41}"\nreason = "x"\n{LOOSE}',
            "'commit'",
        ),
        (
            f'case = "demo.*"\ncommit = "v1.0.11"\nreason = "x"\n{LOOSE}',
            "'commit'",
        ),
        (
            f'case = "demo.*"\ncommit = 1234567\nreason = "x"\n{LOOSE}',
            "'commit'",
        ),
        (
            'case = "demo.*"\nreason = "x"\nmax_rel = 1\nmax_intensity = 1',
            "'max_abs_norm' is required",
        ),
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
        'case = "demo.*"\nreason = "r"\nmax_intensity = 0.5',
    )
    with pytest.raises(cmp.CompareError, match="'max_abs_norm' is required"):
        cmp.compare(ref, cand, registry_with_demo, accepted_path=intensity_only)
    # with the required bound, the corruption exceeds it and stays DRIFT
    bounded = _toml(
        tmp_path,
        'case = "demo.*"\nreason = "r"\nmax_intensity = 0.5\nmax_abs_norm = 0.05',
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
        'case = "demo.*"\nreason = "r"\nmax_rel = 1e30\n'
        "max_intensity = 1e30\nmax_abs_norm = 1e30",
    )
    ref, cand = _arrays(tmp_path, ref_arr, nan_arr)
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.rows[0].verdict == cmp.DRIFT
    # the entry's bounds as such do not see it: a NaN is masked out of the stats
    entry = cmp.Accepted("demo.*", COMMIT, "r", max_abs_norm=1e30, max_intensity=1e30)
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
    present = _toml(tmp_path, f'case = "demo.*"\nreason = "r"\n{LOOSE}')
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=present)
    assert report.accepted_exists and "no accepted_changes.toml" not in cmp.to_markdown(
        report
    )


# ---------------------------------------------------------------------------
# accepted_changes.toml: declared shapes

SID = "demo.shape@quick/cpu"
#: the reference's output arrays; a candidate changes the shape along axis 1
POT = np.linspace(1.0, 2.0, 72).reshape(3, 4, 6)
AUX = np.linspace(1.0, 2.0, 12).reshape(3, 4)
#: ``POT`` with its last row split into two halves: one more row, equal sums over axis 1
RECUT = np.concatenate([POT[:, :3], POT[:, 3:] / 2, POT[:, 3:] / 2], axis=1)
DECLARE = "shape = { potential = [3, 5, 6] }"


@pytest.fixture
def registry_with_shape(registry_with_demo):
    tiers = {t: Tier(gpts=(4, 4)) for t in registry.TIER_NAMES}
    for name, outputs in (("demo.shape", ("potential", "aux")), ("demo.other", ("x",))):
        registry.case(name, tiers=tiers, outputs=outputs)(lambda p, d: lambda: None)
    return registry_with_demo


def _grown(tmp_path, cand=None, ref_axes=(), cand_axes=()):
    """Reference bundle and a candidate whose ``potential`` is ``RECUT``, with one
    more row along axis 1 (``cand`` replaces the candidate's outputs)."""
    ref = bundle(
        tmp_path,
        "ref",
        {
            SID: (
                record(),
                {"potential": (POT, list(ref_axes), {}), "aux": (AUX, [], {})},
            )
        },
        label="v0",
    )
    if cand is None:
        cand = {
            "potential": (RECUT, list(cand_axes), {}),
            "aux": (AUX, [], {}),
        }
    return ref, bundle(tmp_path, "cand", {SID: (record(), cand)})


def _shape_entry(declare=DECLARE, case="demo.shape@quick/*", extra="", bound=LOOSE):
    return f'case = "{case}"\nreason = "grown"\npr = 7\n{bound}\n{declare}\n{extra}'


def _compare_shape(tmp_path, reg, ref, cand, *entries):
    return cmp.compare(ref, cand, reg, accepted_path=_toml(tmp_path, *entries))


def test_a_declared_shape_is_accepted(tmp_path, registry_with_shape):
    ref, cand = _grown(tmp_path)
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, _shape_entry())
    row = report.rows[0]
    assert row.verdict == cmp.ACCEPTED and row.accepted_by == ["grown"]
    assert row.outputs["potential"]["verdict"] == cmp.SHAPE
    assert report.failures(cmp.parse_fail_on("drift,shape")) == []
    entry = report.accepted[0]
    assert entry.matched == [SID] and report.stale_accepted == []
    # the sums of the shape-accepted output are folded into worst
    assert set(entry.worst) == {"rel_above", "intensity", "max_abs_norm"}
    assert max(entry.worst.values()) < 1e-15
    assert entry.shape_change == {"potential": ((3, 4, 6), (3, 5, 6))}


@pytest.mark.parametrize(
    "declared", [[3, 5, 7], [3, 6, 6], [4, 5, 6], [3, 4, 6], [5, 3, 6], [3, 5]]
)
def test_a_different_declared_shape_stays_shape(
    tmp_path, registry_with_shape, declared
):
    ref, cand = _grown(tmp_path)
    entry = _shape_entry(f"shape = {{ potential = {declared} }}")
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, entry)
    row = report.rows[0]
    assert row.verdict == cmp.SHAPE and row.accepted_by == []
    assert (
        "exceeds accepted bound: potential shape (3, 5, 6) but the change declares "
        f"{tuple(declared)}"
    ) in row.notes
    assert report.failures({"shape": None}) == [f"{SID}: SHAPE"]
    assert report.accepted[0].exceeded == [SID] and report.stale_accepted == []


def test_an_undeclared_output_that_changes_shape_stays_shape(
    tmp_path, registry_with_shape
):
    cand = {
        "potential": (RECUT, [], {}),
        "aux": (np.resize(AUX, (3, 5)), [], {}),
    }
    ref, cand = _grown(tmp_path, cand)
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, _shape_entry())
    note = next(n for n in report.rows[0].notes if n.startswith("exceeds"))
    assert report.rows[0].verdict == cmp.SHAPE
    assert note == (
        "exceeds accepted bound: aux has no declared shape (shape (3, 4) → (3, 5))"
    )
    both = _shape_entry("shape = { potential = [3, 5, 6], aux = [3, 5] }")
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, both)
    assert report.rows[0].verdict == cmp.ACCEPTED


def test_an_entry_of_the_change_may_declare_what_another_does_not(
    tmp_path, registry_with_shape
):
    ref, cand = _grown(tmp_path)
    report = _compare_shape(
        tmp_path,
        registry_with_shape,
        ref,
        cand,
        _shape_entry(),
        _shape_entry("", case="demo.shape@*"),
    )
    assert report.rows[0].verdict == cmp.ACCEPTED
    assert [a.matched for a in report.accepted] == [[SID], [SID]]


@pytest.mark.parametrize("same_shape", [False, True])
def test_a_dtype_change_is_never_accepted(tmp_path, registry_with_shape, same_shape):
    pot = (POT if same_shape else RECUT).astype(np.float32)
    ref, cand = _grown(tmp_path, {"potential": (pot, [], {}), "aux": (AUX, [], {})})
    declare = f"shape = {{ potential = {list(pot.shape)} }}"
    report = _compare_shape(
        tmp_path, registry_with_shape, ref, cand, _shape_entry(declare)
    )
    assert report.rows[0].verdict == cmp.SHAPE
    assert "exceeds accepted bound: potential dtype changed" in report.rows[0].notes


def test_a_missing_output_is_never_accepted(tmp_path, registry_with_shape):
    ref, cand = _grown(tmp_path, {"aux": (AUX, [], {})})
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, _shape_entry())
    assert report.rows[0].verdict == cmp.SHAPE
    assert (
        "exceeds accepted bound: potential missing in the candidate"
        in report.rows[0].notes
    )


@pytest.mark.parametrize("bound, verdict", [("1", cmp.ACCEPTED), ("0.1", cmp.SHAPE)])
def test_drift_of_another_output_must_hold_the_bounds(
    tmp_path, registry_with_shape, bound, verdict
):
    cand = {
        "potential": (RECUT, [], {}),
        "aux": (AUX * 1.5, [], {}),
    }
    ref, cand = _grown(tmp_path, cand)
    entry = _shape_entry(extra=f"max_rel = {bound}")
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, entry)
    assert report.rows[0].verdict == verdict
    assert report.accepted[0].worst["rel_above"] == pytest.approx(0.5)
    if verdict == cmp.SHAPE:
        assert (
            "exceeds accepted bound: aux rel 5.0e-01 > 1.0e-01" in report.rows[0].notes
        )


def _moved():
    """``RECUT`` with mass moved along axis 2: the total is kept, the sums over
    axis 1 are not."""
    moved = RECUT.copy()
    moved[:, :, 0] += 0.5
    moved[:, :, 1] -= 0.5
    return moved


TIGHT = "max_abs_norm = 1e-6"


@pytest.mark.parametrize(
    "candidate",
    [
        RECUT * 1000,
        np.zeros_like(RECUT),
        -RECUT,
        np.random.default_rng(0).uniform(1.0, 2.0, RECUT.shape),
        _moved(),
    ],
    ids=["x1000", "zeros", "negated", "random", "moved"],
)
def test_the_values_of_a_shape_accepted_output_are_compared(
    tmp_path, registry_with_shape, candidate
):
    cand = {"potential": (candidate, [], {}), "aux": (AUX, [], {})}
    ref, cand = _grown(tmp_path, cand)
    entry = _shape_entry(bound=TIGHT)
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, entry)
    row = report.rows[0]
    assert row.verdict == cmp.SHAPE and row.accepted_by == []
    assert row.outputs["potential"]["verdict"] == cmp.SHAPE
    [note] = [n for n in row.notes if n.startswith("exceeds")]
    assert "potential summed over axes [1] max_abs" in note
    assert "> 1.0e-06" in note
    assert report.failures({"shape": None}) == [f"{SID}: SHAPE"]
    assert report.accepted[0].exceeded == [SID]


def test_a_recut_axis_with_equal_sums_is_accepted(tmp_path, registry_with_shape):
    ref, cand = _grown(tmp_path)
    report = _compare_shape(
        tmp_path, registry_with_shape, ref, cand, _shape_entry(bound=TIGHT)
    )
    assert report.rows[0].verdict == cmp.ACCEPTED
    summed = report.rows[0].outputs["potential"]["summed"]
    assert summed["axes"] == [1] and summed["max_abs_norm"] < 1e-15


def test_every_entry_of_the_change_bounds_the_sums(tmp_path, registry_with_shape):
    cand = {"potential": (RECUT * 1.001, [], {}), "aux": (AUX, [], {})}
    ref, cand = _grown(tmp_path, cand)
    kwargs = {"bound": "max_abs_norm = 1e-2", "extra": "max_intensity = 1e-2"}
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, _shape_entry())
    assert report.rows[0].verdict == cmp.ACCEPTED
    report = _compare_shape(
        tmp_path, registry_with_shape, ref, cand, _shape_entry(**kwargs)
    )
    assert report.rows[0].verdict == cmp.ACCEPTED
    kwargs["extra"] = "max_intensity = 1e-4"
    report = _compare_shape(
        tmp_path, registry_with_shape, ref, cand, _shape_entry(**kwargs)
    )
    assert report.rows[0].verdict == cmp.SHAPE
    assert (
        "exceeds accepted bound: potential summed over axes [1] intensity "
        "1.0e-03 > 1.0e-04"
    ) in report.rows[0].notes


def test_the_sums_run_over_every_changed_axis(tmp_path, registry_with_shape):
    cand = np.zeros((3, 5, 7))
    cand[:, 0, 0] = POT.sum(axis=(1, 2))
    ref, cand_b = _grown(tmp_path, {"potential": (cand, [], {}), "aux": (AUX, [], {})})
    entry = _shape_entry("shape = { potential = [3, 5, 7] }", bound=TIGHT)
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand_b, entry)
    assert report.rows[0].verdict == cmp.ACCEPTED
    assert report.rows[0].outputs["potential"]["summed"]["axes"] == [1, 2]
    cand[:, 0, 0] *= 2
    ref, cand_b = _grown(tmp_path, {"potential": (cand, [], {}), "aux": (AUX, [], {})})
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand_b, entry)
    assert report.rows[0].verdict == cmp.SHAPE
    assert "potential summed over axes [1, 2] max_abs" in report.rows[0].notes[-1]


@pytest.mark.parametrize("factor, verdict", [(1, cmp.ACCEPTED), (0, cmp.SHAPE)])
def test_the_sums_over_every_axis_are_compared(
    tmp_path, registry_with_shape, factor, verdict
):
    cand = np.zeros((2, 5, 7))
    cand[0, 0, 0] = factor * POT.sum()
    ref, cand_b = _grown(tmp_path, {"potential": (cand, [], {}), "aux": (AUX, [], {})})
    entry = _shape_entry("shape = { potential = [2, 5, 7] }", bound=TIGHT)
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand_b, entry)
    assert report.rows[0].verdict == verdict
    assert report.rows[0].outputs["potential"]["summed"]["axes"] == [0, 1, 2]


def test_a_change_of_rank_is_never_accepted(tmp_path, registry_with_shape):
    cand = {"potential": (POT[..., None], [], {}), "aux": (AUX, [], {})}
    ref, cand = _grown(tmp_path, cand)
    entry = _shape_entry("shape = { potential = [3, 4, 6, 1] }")
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, entry)
    row = report.rows[0]
    assert row.verdict == cmp.SHAPE
    assert (
        "exceeds accepted bound: potential rank changed; values cannot be compared"
    ) in row.notes
    assert "summed" not in row.outputs["potential"]


def test_the_sums_are_in_the_json_and_the_changelog(tmp_path, registry_with_shape):
    cand = {"potential": (RECUT * 1.001, [], {}), "aux": (AUX, [], {})}
    ref, cand = _grown(tmp_path, cand)
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, _shape_entry())
    assert report.accepted[0].worst["intensity"] == pytest.approx(1e-3)
    assert ", intensity 1.0e-03, " in cmp.to_markdown(report)
    summed = cmp.to_json(report)["rows"][0]["outputs"]["potential"]["summed"]
    assert summed["axes"] == [1]
    assert summed["intensity"] == pytest.approx(1e-3)


def test_an_axis_mismatch_follows_the_declared_shape(tmp_path, registry_with_shape):
    def axes(n):
        return [{"type": "x", "sampling": 1.0, "values": list(range(n))}] * 3

    ref, cand = _grown(tmp_path, ref_axes=axes(4), cand_axes=axes(5))
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, _shape_entry())
    assert report.rows[0].outputs["potential"]["axes_ok"] is False
    assert report.rows[0].verdict == cmp.ACCEPTED
    wrong = _shape_entry("shape = { potential = [3, 5, 7] }")
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, wrong)
    assert report.rows[0].verdict == cmp.SHAPE


def test_a_change_without_a_shape_leaves_a_shape_row_untouched(
    tmp_path, registry_with_shape
):
    ref, cand = _grown(tmp_path)
    old_style = _shape_entry("", extra='commit = "1111111"')
    # alone, it is as on a base that knows no shape: no note, not matched, stale
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, old_style)
    row = report.rows[0]
    assert row.verdict == cmp.SHAPE
    assert row.notes == ["potential: shape (3, 4, 6) → (3, 5, 6)"]
    [entry] = report.accepted
    assert entry.matched == entry.exceeded == entry.dominated == []
    assert report.stale_accepted == [entry]
    # next to a change that declares a wrong shape, it takes no part either
    wrong = _shape_entry(DECLARE.replace("5", "6"))
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, old_style, wrong)
    assert report.rows[0].verdict == cmp.SHAPE
    assert report.rows[0].notes[-1] == (
        "exceeds accepted bound: potential shape (3, 5, 6) but the change "
        "declares (3, 6, 6)"
    )
    first, second = report.accepted
    assert first.matched == first.exceeded == first.dominated == []
    assert second.exceeded == [SID] and report.stale_accepted == [first]
    # and it is not dominated by a change that accepts the row
    report = _compare_shape(
        tmp_path, registry_with_shape, ref, cand, old_style, _shape_entry()
    )
    assert report.rows[0].verdict == cmp.ACCEPTED
    first, second = report.accepted
    assert first.matched == first.dominated == [] and second.matched == [SID]


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_value_in_a_shape_accepted_output_is_refused(
    tmp_path, registry_with_shape, bad
):
    pot = RECUT.copy()
    pot[0, 0, :2] = bad
    ref, cand = _grown(tmp_path, {"potential": (pot, [], {}), "aux": (AUX, [], {})})
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, _shape_entry())
    assert report.rows[0].verdict == cmp.SHAPE
    [note] = [n for n in report.rows[0].notes if n.startswith("exceeds")]
    assert "potential has 2 non-finite values" in note


def test_an_output_of_unchanged_shape_is_never_accepted_by_shape(
    tmp_path, registry_with_shape
):
    def axes(sampling):
        return [{"type": "x", "sampling": sampling}] * 3

    cand = {"potential": (POT, axes(2.0), {}), "aux": (AUX, [], {})}
    ref, cand = _grown(tmp_path, cand, ref_axes=axes(1.0))
    entry = _shape_entry("shape = { potential = [3, 4, 6] }")
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, entry)
    row = report.rows[0]
    assert row.verdict == cmp.SHAPE
    assert row.notes[-1].startswith(
        "exceeds accepted bound: potential shape is unchanged; only its axes "
        "differ (axis 0 sampling 1.0 → 2.0"
    )


def test_an_entry_whose_change_the_candidate_lacks_is_not_applied(
    tmp_path, registry_with_shape, monkeypatch
):
    monkeypatch.setattr(cmp, "_contains", lambda commit, sha: False)
    ref, cand = _grown(tmp_path)
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, _shape_entry())
    assert report.rows[0].verdict == cmp.SHAPE
    assert report.accepted == [] and len(report.not_applicable) == 1


def test_the_changelog_and_json_show_the_declared_shape(tmp_path, registry_with_shape):
    ref, cand = _grown(tmp_path)
    report = _compare_shape(tmp_path, registry_with_shape, ref, cand, _shape_entry())
    md = cmp.to_markdown(report)
    assert "max_abs ≤ 1.0e+01, shape potential (3, 5, 6) |" in md
    assert ", potential (3, 4, 6) → (3, 5, 6) |" in md
    assert cmp.to_json(report)["accepted"][0]["shape"] == {"potential": [3, 5, 6]}
    assert cmp.to_json(report)["rows"][0]["verdict"] == cmp.ACCEPTED


@pytest.mark.parametrize(
    "declare",
    [
        "shape = 3",
        "shape = [3, 5, 6]",
        "shape = {}",
        "shape = { potential = [] }",
        "shape = { potential = 5 }",
        'shape = { potential = "3" }',
        "shape = { potential = [3.0, 5, 6] }",
        "shape = { potential = [true, 5, 6] }",
        "shape = { potential = [0, 5, 6] }",
        "shape = { potential = [3, -5, 6] }",
        "shape = { potential = [[3], 5, 6] }",
        "shape = { potential = [3, 5, 6], aux = [] }",
    ],
)
def test_an_invalid_shape_is_refused(tmp_path, registry_with_shape, declare):
    ref, cand = _grown(tmp_path)
    with pytest.raises(cmp.CompareError, match="entry 0: 'shape' must be a non-empty"):
        _compare_shape(tmp_path, registry_with_shape, ref, cand, _shape_entry(declare))


@pytest.mark.parametrize(
    "case, declare, unknown",
    [
        ("demo.shape@*", "shape = { nosuch = [3] }", "nosuch"),
        ("demo.shape@*", "shape = { potential = [3], x = [3] }", "x"),
        # an output of a case the glob does not match
        ("demo.shape@*", "shape = { x = [3] }", "x"),
        # the demo case declares no outputs
        ("demo.case@*", "shape = { output = [3] }", "output"),
    ],
)
def test_a_shape_naming_no_output_of_a_matched_case_is_refused(
    tmp_path, registry_with_shape, case, declare, unknown
):
    path = _toml(tmp_path, _shape_entry(declare, case))
    with pytest.raises(
        cmp.CompareError, match=f"entry 0: 'shape' names \\['{unknown}'\\]"
    ):
        cmp.load_accepted(path, registry_with_shape)


def test_a_shape_may_name_an_output_of_any_matched_case(tmp_path, registry_with_shape):
    declare = "shape = { potential = [3], x = [3] }"
    path = _toml(tmp_path, _shape_entry(declare, "demo.*"))
    [entry] = cmp.load_accepted(path, registry_with_shape)
    assert entry.shape == {"potential": (3,), "x": (3,)}


# ---------------------------------------------------------------------------
# --fail-on and refusals


@pytest.mark.parametrize("text", ["", " ", ",", " , ,"])
def test_a_fail_on_that_names_no_gate_is_refused(text):
    with pytest.raises(cmp.CompareError, match="--fail-on names no gate"):
        cmp.parse_fail_on(text)


def test_parse_fail_on():
    assert cmp.parse_fail_on(None) == {
        "drift": None,
        "shape": None,
        "error": None,
        "missing": None,
    }
    assert cmp.parse_fail_on(" none ") == {}
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
    relabelled = {**AXIS, "label": "y", "_concatenate": False, "novel": False}
    row = cmp.compare(
        ref, _with_axes(tmp_path, "cand", [AXIS, relabelled]), registry_with_demo
    ).rows[0]
    assert row.verdict == cmp.IDENTICAL
    assert row.notes == [
        "out: axis metadata differs (axis 1 _concatenate, axis 1 label, axis 1 novel)"
    ]


@pytest.mark.parametrize(
    "key", ["type", "sampling", "offset", "values", "units", "endpoint"]
)
def test_a_structural_axis_field_missing_on_one_side_is_a_mismatch(
    tmp_path, registry_with_demo, key
):
    full = {**AXIS, "values": [0.0, 0.1], "endpoint": True}
    ref = _with_axes(tmp_path, "ref", [full, full])
    dropped = {k: v for k, v in full.items() if k != key}
    row = cmp.compare(
        ref, _with_axes(tmp_path, "cand", [full, dropped]), registry_with_demo
    ).rows[0]
    assert row.verdict == cmp.SHAPE
    assert row.notes == [f"out: axis 1 {key} only in the reference"]
    other = cmp.compare(
        _with_axes(tmp_path, "ref2", [full, dropped]), ref, registry_with_demo
    ).rows[0]
    assert other.verdict == cmp.SHAPE
    assert other.notes == [f"out: axis 1 {key} only in the candidate"]


def test_an_unknown_axis_field_on_one_side_is_informational(
    tmp_path, registry_with_demo
):
    ref = _with_axes(tmp_path, "ref", [AXIS, AXIS])
    extra = {**AXIS, "novel": 3}
    row = cmp.compare(
        ref, _with_axes(tmp_path, "cand", [AXIS, extra]), registry_with_demo
    ).rows[0]
    assert row.verdict == cmp.IDENTICAL
    assert row.notes == ["out: axis metadata differs (axis 1 novel)"]


def test_nan_axis_values_on_both_sides_are_equal(tmp_path, registry_with_demo):
    nan_axis = {**AXIS, "offset": float("nan"), "values": [float("nan"), 1.0]}
    ref = _with_axes(tmp_path, "ref", [nan_axis, AXIS])
    same = cmp.compare(
        ref, _with_axes(tmp_path, "c1", [dict(nan_axis), AXIS]), registry_with_demo
    ).rows[0]
    assert same.verdict == cmp.IDENTICAL and same.notes == []
    other = cmp.compare(
        ref,
        _with_axes(tmp_path, "c2", [{**nan_axis, "offset": 0.0}, AXIS]),
        registry_with_demo,
    ).rows[0]
    assert other.verdict == cmp.SHAPE


def _with_metadata(tmp_path, name, metadata):
    return bundle(tmp_path, name, {CID: (record(), {"out": (ARR, [], metadata)})})


def test_output_metadata_is_informational_only(tmp_path, registry_with_demo):
    base = {"energy": 1.0e5, "label": "a", "units": "eV", "_x": 1, "type": "Waves"}
    ref = _with_metadata(tmp_path, "ref", base)
    # ignored keys and numbers within tolerance change nothing
    quiet = {
        **base,
        "label": "b",
        "units": "keV",
        "_x": 2,
        "energy": 1.0e5 * (1 + 1e-12),
    }
    row = cmp.compare(
        ref, _with_metadata(tmp_path, "c1", quiet), registry_with_demo
    ).rows[0]
    assert row.verdict == cmp.IDENTICAL and row.notes == []
    # a changed value, a missing key and an extra key are reported, not judged
    changed = {"energy": 2.0e5, "type": "Waves", "new": 1}
    row = cmp.compare(
        ref, _with_metadata(tmp_path, "c2", changed), registry_with_demo
    ).rows[0]
    assert row.verdict == cmp.IDENTICAL
    assert row.notes == ["out: metadata differs (energy, new)"]
    assert row.outputs["out"]["metadata_info"] == [
        "energy 100000.0 → 200000.0",
        "new only in the candidate",
    ]


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
    toml = _toml(tmp_path, f'case = "demo.*"\nreason = "a | b"\n{LOOSE}')
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


def test_to_json_replaces_non_finite_floats_with_strings():
    out = cmp._finite_json(
        {
            "a": [float("nan"), float("inf"), -float("inf"), 1.5],
            "b": {"c": np.float64("nan")},
            "d": "nan",
            "e": 3,
        }
    )
    assert out == {
        "a": ["nan", "inf", "-inf", 1.5],
        "b": {"c": "nan"},
        "d": "nan",
        "e": 3,
    }


def test_a_structural_axis_field_set_to_none_on_one_side_is_informational(
    tmp_path, registry_with_demo
):
    ref = _with_axes(tmp_path, "ref", [AXIS, AXIS])
    cand = _with_axes(tmp_path, "cand", [AXIS, {**AXIS, "values": None}])
    row = cmp.compare(ref, cand, registry_with_demo).rows[0]
    assert row.verdict == cmp.IDENTICAL


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -0.1])
def test_a_noise_floor_must_be_finite_and_non_negative(value):
    with pytest.raises(cmp.CompareError, match="finite"):
        cmp.check_noise({CID: {"accuracy": value}}, "noise.json")


def test_an_entry_for_a_case_that_failed_is_not_stale(tmp_path, registry_with_demo):
    ref = bundle(tmp_path, "ref", {CID: (record(), {"out": (ARR, [], {})})}, label="v0")
    cand = bundle(
        tmp_path, "cand", {CID: (record(status=store.STATUS_ERROR, error="x"), None)}
    )
    toml = _toml(tmp_path, f'case = "demo.*"\nreason = "r"\n{LOOSE}')
    report = cmp.compare(ref, cand, registry_with_demo, accepted_path=toml)
    assert report.stale_accepted == []


def test_an_unjudged_memory_gate_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cmp, "ACCEPTED_CHANGES_PATH", tmp_path / "none.toml")
    ref, cand = one(tmp_path, "ref", ARR), one(tmp_path, "cand", ARR, rss=10**9)
    noise = tmp_path / "noise.json"
    noise.write_text('{"other.case@quick/cpu": {"memory": 0.01}}')
    argv = ["compare", str(ref.path), str(cand.path), "--noise", str(noise)]
    assert cli.main(argv + ["--fail-on", "memory"]) == 0
    assert "did not judge 1 case id" in capsys.readouterr().err
