"""Registry contract: case ids, params, source validation, case hash."""

import textwrap

import pytest
from abtem_bench import registry
from abtem_bench.registry import CaseDefinitionError, CaseId, Tier, Variant, case


def _tiers(**extra):
    return {
        "quick": Tier(gpts=(8, 8), reps=(1, 1, 1), **extra),
        "standard": Tier(gpts=(16, 16), reps=(1, 1, 2), **extra),
        "large": Tier(gpts=(32, 32), reps=(2, 2, 2), devices=("gpu",), **extra),
    }


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY", {})
    return registry.REGISTRY


def test_case_id_roundtrip():
    for text in ("stem.haadf@quick/cpu", "stem.haadf[order1]@large/gpu"):
        assert str(CaseId.parse(text)) == text
    assert CaseId.parse("a.b@quick/cpu").variant == "default"
    with pytest.raises(ValueError):
        CaseId.parse("not an id")


def test_case_filename_is_filesystem_safe():
    cid = CaseId.parse("stem.haadf[order1]@quick/gpu")
    assert "/" not in cid.filename


def test_register_merges_tier_and_variant_params(clean_registry):
    @case("demo.one", tiers=_tiers(chunk=4), variants={"eager": Variant(lazy=False)})
    def demo(p, device):
        return lambda: None

    c = clean_registry["demo.one"]
    p = c.params("quick", "eager", "cpu")
    assert (p.gpts, p.chunk, p.lazy, p.device, p.tier, p.variant) == (
        (8, 8),
        4,
        False,
        "cpu",
        "quick",
        "eager",
    )
    assert c.params("quick", "default", "cpu").lazy is True
    ids = [str(i) for i in c.ids("large", ("cpu", "gpu"))]
    assert ids == ["demo.one@large/gpu", "demo.one[eager]@large/gpu"]


def test_register_rejects_bad_declarations(clean_registry):
    with pytest.raises(CaseDefinitionError, match="missing tiers"):
        case("demo.two", tiers={"quick": Tier()})(lambda p, d: None)
    with pytest.raises(CaseDefinitionError, match="group.name"):
        case("nodot", tiers=_tiers())(lambda p, d: None)
    with pytest.raises(CaseDefinitionError, match="warmup"):
        case("demo.three", tiers=_tiers(), warmup="sometimes")(lambda p, d: None)
    with pytest.raises(CaseDefinitionError, match="compares as unknown"):
        case("demo.four", tiers=_tiers(), variants={"x": Variant(compare_as="y")})(
            lambda p, d: None
        )
    case("demo.five", tiers=_tiers())(lambda p, d: None)
    with pytest.raises(CaseDefinitionError, match="twice"):
        case("demo.five", tiers=_tiers())(lambda p, d: None)


def test_validate_sources_rejects_sampling_keyword_but_not_prose(tmp_path):
    good = tmp_path / "good.py"
    good.write_text(
        textwrap.dedent(
            '''
            """Never pass sampling= here; the docstring may say so."""
            def f(Potential):
                return Potential(gpts=(8, 8))  # sampling= in a comment is fine
            '''
        )
    )
    registry.validate_case_sources(tmp_path)
    bad = tmp_path / "bad.py"
    bad.write_text("def f(Potential):\n    return Potential(sampling=0.05)\n")
    with pytest.raises(CaseDefinitionError, match="bad.py:2"):
        registry.validate_case_sources(tmp_path)


def test_case_hash_tracks_case_sources(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    h1 = registry.case_hash(tmp_path)
    assert registry.case_hash(tmp_path) == h1
    (tmp_path / "a.py").write_text("x = 2\n")
    assert registry.case_hash(tmp_path) != h1


def test_shipped_cases_load_and_pass_validation():
    reg = registry.load_cases()
    assert {
        "potential.infinite",
        "hrtem.exitwave",
        "stem.multidetector",
        "diffraction.cbed",
    } <= set(reg)
    ids = registry.select_ids(reg, "quick", ["cpu"], only=["stem.*"])
    assert all(i.name == "stem.multidetector" for i in ids)
    assert len(registry.select_ids(reg, "quick", ["cpu"], tags=["v1.1"])) >= 4
