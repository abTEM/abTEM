"""Case registry: the declarative contract every benchmark case follows.

A case is a function ``(params, device) -> run`` decorated with ``@case``. The
function body is the untimed setup; the returned zero-argument ``run`` is the
only timed region and returns the abTEM objects (or numpy arrays) that become
the case's outputs.

Case ids are ``name[variant]@tier/device``; the ``[variant]`` part is omitted
for the default variant.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib
import pkgutil
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import abtem_bench

DEFAULT_VARIANT = "default"
TIER_NAMES = ("quick", "standard", "large")
DEVICES = ("cpu", "gpu")
WARMUP_POLICIES = ("cold_and_warm", "cold_only", "warm_only")


class CaseDefinitionError(ValueError):
    """A case declaration violates the contract."""


class Tier:
    """Parameters of one size tier, plus which devices it runs on."""

    def __init__(
        self,
        *,
        devices: tuple[str, ...] = DEVICES,
        timeout: float | None = None,
        **params: Any,
    ):
        for d in devices:
            if d not in DEVICES:
                raise CaseDefinitionError(f"unknown device {d!r}")
        self.devices = tuple(devices)
        self.timeout = timeout
        self.params: dict[str, Any] = dict(params)

    def __repr__(self) -> str:
        return f"Tier(devices={self.devices}, timeout={self.timeout}, {self.params})"


class Variant:
    """Parameter overrides producing a sibling case id.

    ``compare_as`` names the variant of the *other* bundle this one is paired
    with (typically ``"default"``), so ``x[order1]`` on a candidate can be
    compared to ``x`` on a reference. ``flag=False`` keeps speed and memory
    ratios in the report but never flags them (used for ``auto`` sizing).
    """

    def __init__(
        self, *, compare_as: str | None = None, flag: bool = True, **params: Any
    ):
        self.compare_as = compare_as
        self.flag = flag
        self.params: dict[str, Any] = dict(params)


@dataclass(frozen=True)
class Tolerance:
    """Accuracy tolerance for the float64 accuracy preset.

    ``rel`` bounds the relative error on elements above ``above_rel`` times the
    reference maximum; ``intensity`` bounds the relative change of the
    integrated intensity; ``max_abs_norm`` bounds ``max|diff| / max|ref|``.
    """

    rel: float = 1e-10
    above_rel: float = 1e-6
    intensity: float = 1e-12
    max_abs_norm: float = 1e-10


@dataclass(frozen=True)
class CaseId:
    name: str
    variant: str = DEFAULT_VARIANT
    tier: str = "quick"
    device: str = "cpu"

    def __str__(self) -> str:
        v = "" if self.variant == DEFAULT_VARIANT else f"[{self.variant}]"
        return f"{self.name}{v}@{self.tier}/{self.device}"

    @property
    def filename(self) -> str:
        return str(self).replace("/", "__")

    @classmethod
    def parse(cls, text: str) -> "CaseId":
        m = re.fullmatch(
            r"([A-Za-z0-9_.]+)(?:\[([A-Za-z0-9_]+)\])?@([a-z]+)/([a-z]+)", text
        )
        if not m:
            raise ValueError(f"not a case id: {text!r}")
        name, variant, tier, device = m.groups()
        return cls(name, variant or DEFAULT_VARIANT, tier, device)

    def with_variant(self, variant: str) -> "CaseId":
        return CaseId(self.name, variant, self.tier, self.device)


@dataclass
class Case:
    name: str
    func: Callable[[Any, str], Callable[[], Any]]
    tiers: dict[str, Tier]
    tags: frozenset[str] = frozenset()
    variants: dict[str, Variant] = field(default_factory=dict)
    outputs: tuple[str, ...] | None = None
    tolerance: Tolerance = Tolerance()
    requires: Callable[[Any], bool] | None = None
    warmup: str = "cold_and_warm"
    nominal_bytes: Callable[[Any], int] | None = None
    consistency: tuple[tuple[str, str], ...] = ()
    module: str = ""

    def params(self, tier: str, variant: str, device: str) -> SimpleNamespace:
        t = self.tiers[tier]
        merged: dict[str, Any] = dict(t.params)
        if variant != DEFAULT_VARIANT:
            merged.update(self.variants[variant].params)
        merged.setdefault("lazy", True)
        merged["device"] = device
        merged["tier"] = tier
        merged["variant"] = variant
        return SimpleNamespace(**merged)

    def ids(self, tier: str, devices: Iterable[str] = DEVICES) -> list[CaseId]:
        t = self.tiers[tier]
        out = []
        for device in devices:
            if device not in t.devices:
                continue
            out.append(CaseId(self.name, DEFAULT_VARIANT, tier, device))
            for v in self.variants:
                out.append(CaseId(self.name, v, tier, device))
        return out


REGISTRY: dict[str, Case] = {}


def case(
    name: str,
    *,
    tiers: Mapping[str, Tier],
    tags: Iterable[str] = (),
    variants: Mapping[str, Variant] | None = None,
    outputs: Iterable[str] | None = None,
    tolerance: Tolerance = Tolerance(),
    requires: Callable[[Any], bool] | None = None,
    warmup: str = "cold_and_warm",
    nominal_bytes: Callable[[Any], int] | None = None,
    consistency: Iterable[tuple[str, str]] = (),
) -> Callable[[Callable], Callable]:
    """Register a benchmark case. See the module docstring for the contract."""
    if not re.fullmatch(r"[a-z0-9_]+(\.[a-z0-9_]+)+", name):
        raise CaseDefinitionError(f"case name {name!r} must look like 'group.name'")
    missing = [t for t in TIER_NAMES if t not in tiers]
    if missing:
        raise CaseDefinitionError(f"case {name}: missing tiers {missing}")
    unknown = [t for t in tiers if t not in TIER_NAMES]
    if unknown:
        raise CaseDefinitionError(f"case {name}: unknown tiers {unknown}")
    if warmup not in WARMUP_POLICIES:
        raise CaseDefinitionError(
            f"case {name}: warmup must be one of {WARMUP_POLICIES}"
        )
    variants = dict(variants or {})
    if DEFAULT_VARIANT in variants:
        raise CaseDefinitionError(f"case {name}: 'default' is implicit, not a variant")
    for v, var in variants.items():
        if (
            var.compare_as is not None
            and var.compare_as != DEFAULT_VARIANT
            and var.compare_as not in variants
        ):
            raise CaseDefinitionError(
                f"case {name}: variant {v} compares as unknown variant {var.compare_as}"
            )

    def decorator(func: Callable) -> Callable:
        if name in REGISTRY:
            raise CaseDefinitionError(f"case {name} registered twice")
        REGISTRY[name] = Case(
            name=name,
            func=func,
            tiers=dict(tiers),
            tags=frozenset(tags),
            variants=variants,
            outputs=tuple(outputs) if outputs is not None else None,
            tolerance=tolerance,
            requires=requires,
            warmup=warmup,
            nominal_bytes=nominal_bytes,
            consistency=tuple(consistency),
            module=func.__module__,
        )
        return func

    return decorator


# ---------------------------------------------------------------------------
# Loading and validating the cases package


def cases_package_dir() -> Path:
    return Path(abtem_bench.__file__).resolve().parents[1] / "cases"


_FORBIDDEN_KEYWORDS = {
    "sampling": (
        "pass explicit gpts, never sampling= "
        "(grid.round-to-fast-fft changes derived grids between commits)"
    ),
}


def validate_case_sources(package_dir: Path | None = None) -> None:
    """Static checks on the case sources; raise on the first violation.

    Works on the syntax tree, so a mention in a comment or docstring is fine;
    only a real ``sampling=`` keyword argument in a call is rejected.
    """
    import ast

    package_dir = package_dir or cases_package_dir()
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg in _FORBIDDEN_KEYWORDS:
                        raise CaseDefinitionError(
                            f"{path.name}:{kw.lineno}: {_FORBIDDEN_KEYWORDS[kw.arg]}"
                        )


def load_cases() -> dict[str, Case]:
    """Import every module of the ``cases`` package, populating REGISTRY."""
    validate_case_sources()
    import cases  # the sibling package under benchmarks/

    for info in pkgutil.iter_modules(cases.__path__):
        importlib.import_module(f"cases.{info.name}")
    return REGISTRY


def case_hash(package_dir: Path | None = None) -> str:
    """sha256 over the sorted case sources plus the harness version.

    Two bundles are only comparable when this matches: it is the proof that
    both refs ran the same case code.
    """
    package_dir = package_dir or cases_package_dir()
    h = hashlib.sha256()
    h.update(f"abtem_bench {abtem_bench.__version__}\n".encode())
    for path in sorted(package_dir.glob("*.py")):
        h.update(f"--- {path.name}\n".encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def select_ids(
    registry: Mapping[str, Case],
    tier: str,
    devices: Iterable[str],
    only: Iterable[str] = (),
    tags: Iterable[str] = (),
) -> list[CaseId]:
    """Case ids for one tier, filtered by id globs (``only``) and tags."""
    only = list(only)
    tags = set(tags)
    out: list[CaseId] = []
    for c in registry.values():
        if tags and not tags & c.tags:
            continue
        for cid in c.ids(tier, devices):
            if only and not any(
                fnmatch.fnmatchcase(str(cid), g) or fnmatch.fnmatchcase(c.name, g)
                for g in only
            ):
                continue
            out.append(cid)
    return out
