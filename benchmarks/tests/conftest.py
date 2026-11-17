"""Synthetic-bundle helpers the compare tests share."""

from pathlib import Path

import pytest
from abtem_bench import compare as cmp
from abtem_bench import registry, store
from abtem_bench.registry import Tier, Tolerance, Variant

CHECKOUT = Path(__file__).resolve().parents[2]

CID = "demo.case@quick/cpu"
#: the sha of a reference bundle built by ``bundle(tmp_path, "ref", ...)``; every
#: other bundle carries the default sha, and ``candidate_contains_every_commit``
#: tells them apart
REFERENCE_SHA = "1" * 40


def manifest(
    label="ref", case_hash="h", fp="ffff0000", preset="accuracy", sha="0" * 40
):
    return {
        "schema_version": store.SCHEMA_VERSION,
        "harness_version": "0.0",
        "case_hash": case_hash,
        "ref": {"label": label, "sha": sha, "describe": label},
        "preset": preset,
        "tier": "quick",
        "devices": ["cpu"],
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "fingerprint": {"hostname": "box"},
        "fingerprint_short": fp,
    }


def record(median=1.0, cold=2.0, rss=100 * 1024**2, status=store.STATUS_OK, **extra):
    return {
        "status": status,
        "timings": {"median": median, "cold": cold, "warm": [median]},
        "memory": {"peak_rss_bytes": rss},
        **extra,
    }


def bundle(tmp_path, name, cases, **manifest_kw):
    """A bundle ``name`` under ``tmp_path``; ``cases`` maps id -> (record, outputs)."""
    manifest_kw.setdefault("label", name)
    if name == "ref":
        manifest_kw.setdefault("sha", REFERENCE_SHA)
    b = store.Bundle(tmp_path / name)
    b.write_manifest(manifest(**manifest_kw))
    for cid, (rec, outputs) in cases.items():
        b.write_case(cid, rec, outputs)
    return b


def one(tmp_path, name, array, axes=(), cid=CID, **record_kw):
    """A bundle holding one case with one output ``out``."""
    return bundle(
        tmp_path, name, {cid: (record(**record_kw), {"out": (array, list(axes), {})})}
    )


@pytest.fixture(autouse=True)
def candidate_alone_contains_every_commit(monkeypatch):
    """Synthetic bundles carry made-up shas that git cannot resolve, so every
    accepted entry's commit resolves to itself and counts as contained in every
    bundle but the reference built by ``bundle(tmp_path, "ref", ...)``; tests of
    the scope patch ``cmp._resolve`` and ``cmp._contains`` themselves."""
    monkeypatch.setattr(cmp, "_resolve", lambda commit: commit)
    monkeypatch.setattr(cmp, "_contains", lambda commit, sha: sha != REFERENCE_SHA)


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
