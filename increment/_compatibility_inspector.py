"""Private compatibility inspector: mechanical observation and fixed
aggregation over capability-owned compatibility decisions.

This module never decides anything itself. Each ``CompatibilityCheck``
pairs a capability-owned pure decision function (``evaluate``, e.g. an
arm/contrast contract's ``runtime_support``) with a mechanical ``observe``
adapter (``default_support_observe`` / ``contextual_decision_observe``)
that copies already-decided facts -- runtime status, family participation,
refusal code, assumptions, sampling floor, warning, reference -- into the
common ``CompatibilityFinding`` shape. An observation adapter contains no
domain predicate and no ``if capability == ...`` branch; all substantive
judgment must already live in the ``DecisionT`` a capability contract
returned.

``aggregate_overall``/``aggregate_family`` combine multiple findings with a
fixed, capability-agnostic precedence -- never by comparing capability
names or special-casing a scenario:

- runtime precedence (most to least severe): ``refused`` > ``unknown`` >
  ``limited`` > ``supported``; a report whose findings are all
  ``not_applicable`` collapses to ``not_applicable``.
- family precedence: ``excluded`` > ``unknown`` > ``participates``;
  ``not_applicable`` findings are ignored unless every family finding is
  ``not_applicable``.

``CompatibilityReport`` validates its own ``overall``/``overall_family``
fields against this same fixed aggregation on construction, so a caller
can never build a report whose summary contradicts its findings.

Production runtime and capability modules never import this module. The
exact ``tach.toml`` graph declares its dependencies without granting any
runtime module a dependency on it. It serves test-side catalogs and
documentation generators that need a mechanically-derived finding.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from increment.compatibility import SamplingFloor, Support, Unsupported
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "facade.compatibility_inspector.compatibility_report.overall_fields_equal": "overall fields must equal fixed aggregation {expected!r}, got {actual!r}",
    },
)
_raise = raiser(_REFUSALS)

RuntimeStatus = Literal["supported", "limited", "refused", "not_applicable", "unknown"]
FamilyStatus = Literal["participates", "excluded", "not_applicable", "unknown"]
EvidenceSource = Literal["contract", "probe", "none"]


class CompatibilityFinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    capability: str
    runtime: RuntimeStatus
    family: FamilyStatus
    refusal_code: str | None = None
    assumptions: tuple[str, ...] = ()
    floor: SamplingFloor | None = None
    warning: str | None = None
    reference: str | None = None
    evidence_source: EvidenceSource


class ContextualDecision(Protocol):
    @property
    def runtime_effect(self) -> Literal["limited", "not_applicable"]: ...

    @property
    def participation(self) -> Literal["participates", "excluded", "not_applicable"]: ...

    @property
    def warning(self) -> str | None: ...


@dataclass(frozen=True)
class CompatibilityCheck[RequestT, DecisionT]:
    capability: str
    request: RequestT
    evaluate: Callable[[RequestT], DecisionT]
    observe: Callable[[DecisionT], CompatibilityFinding]

    def run(self) -> CompatibilityFinding:
        return self.observe(self.evaluate(self.request))


_RUNTIME_PRECEDENCE: tuple[RuntimeStatus, ...] = (
    "refused",
    "unknown",
    "limited",
    "supported",
)
_FAMILY_PRECEDENCE: tuple[FamilyStatus, ...] = (
    "excluded",
    "unknown",
    "participates",
)


def aggregate_overall(findings: Sequence[CompatibilityFinding]) -> RuntimeStatus:
    applicable = {finding.runtime for finding in findings if finding.runtime != "not_applicable"}
    if not applicable:
        return "not_applicable"
    return next(status for status in _RUNTIME_PRECEDENCE if status in applicable)


def aggregate_family(findings: Sequence[CompatibilityFinding]) -> FamilyStatus:
    applicable = {finding.family for finding in findings if finding.family != "not_applicable"}
    if not applicable:
        return "not_applicable"
    return next(status for status in _FAMILY_PRECEDENCE if status in applicable)


class CompatibilityReport(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    overall: RuntimeStatus
    overall_family: FamilyStatus
    findings: tuple[CompatibilityFinding, ...]

    @model_validator(mode="after")
    def _aggregates_match_findings(self) -> CompatibilityReport:
        expected = (aggregate_overall(self.findings), aggregate_family(self.findings))
        actual = (self.overall, self.overall_family)
        if actual != expected:
            _raise(
                "facade.compatibility_inspector.compatibility_report.overall_fields_equal",
                actual=actual,
                expected=expected,
            )
        return self

    @classmethod
    def from_findings(cls, findings: Sequence[CompatibilityFinding]) -> CompatibilityReport:
        frozen = tuple(findings)
        return cls(
            overall=aggregate_overall(frozen),
            overall_family=aggregate_family(frozen),
            findings=frozen,
        )


def default_support_observe(
    support: Support,
    *,
    capability: str,
    evidence_source: EvidenceSource,
) -> CompatibilityFinding:
    if isinstance(support, Unsupported):
        return CompatibilityFinding(
            capability=capability,
            runtime="refused",
            family="not_applicable",
            refusal_code=support.refusal_code,
            evidence_source=evidence_source,
        )
    return CompatibilityFinding(
        capability=capability,
        runtime="supported",
        family="not_applicable",
        assumptions=support.assumptions,
        floor=support.floor,
        reference=support.reference,
        evidence_source=evidence_source,
    )


def contextual_decision_observe(
    decision: ContextualDecision,
    *,
    capability: str,
    evidence_source: EvidenceSource,
    reference: str,
) -> CompatibilityFinding:
    return CompatibilityFinding(
        capability=capability,
        runtime=decision.runtime_effect,
        family=decision.participation,
        warning=decision.warning,
        reference=reference,
        evidence_source=evidence_source,
    )


class RunnableCheck(Protocol):
    def run(self) -> CompatibilityFinding: ...


def inspect_compatibility(checks: Sequence[RunnableCheck]) -> CompatibilityReport:
    return CompatibilityReport.from_findings(tuple(check.run() for check in checks))
