"""Collect experiment health checks from one pinned source snapshot."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from increment._evidence_types import EvidenceScope, HealthDiagnosticEvidence
from increment.decision import ContrastAnalysisState, DefinitionsArmAnalysisState
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.health import ExperimentHealth, HealthCheck
from increment.readouts._evidence import (
    acquire_health_scope,
    health_population,
    load_health_diagnostics,
)
from increment.sources import _health_snapshot_id

if TYPE_CHECKING:
    from increment._binding import SourceBinding
    from increment.estimation.diagnostics import NotApplicable, SRMResult

type Population = Literal["assigned", "triggered"]
type SrmObservation = tuple[
    SRMResult | NotApplicable | None, tuple[str, Mapping[str, Any], str] | None
]
# Allocation results recorded within one pinned capture, keyed by request.
type SrmObservations = dict[tuple[Any, ...], SrmObservation]

_CHECK_ORDER = (
    "assigned_allocation",
    "triggered_balance",
    "trigger_rates",
    "trigger_maturity",
    "feed_completeness",
    "trigger_assumptions",
)

# Context a check renders for readers: refusal text, remedies, descriptions and
# assumption statements. The codes, statuses and values beside them are its evidence.
_PROSE_CONTEXT = frozenset(
    {
        "reason",
        "route",
        "description",
        "risk",
        "counterfactual_trigger_invariance",
        "trigger_order_stationarity",
        "no_effect_before_trigger",
        "triggered_look_predictability",
        "family_guarantee_status",
    }
)


def health_identity(check: HealthCheck) -> dict[str, Any]:
    """The part of *check* a snapshot id covers: codes, statuses and structured evidence."""
    identity = check.model_dump(mode="json")
    identity["context"] = {
        key: value for key, value in identity["context"].items() if key not in _PROSE_CONTEXT
    }
    return identity


def srm_observation_key(
    population: Population,
    expected: Mapping[str, float] | None,
    alpha: float,
    inference: str,
) -> tuple[Any, ...]:
    """Identify one allocation request within a pinned capture."""
    return (
        population,
        None if expected is None else tuple(sorted(expected.items())),
        alpha,
        inference,
    )


def check_health(
    binding: SourceBinding,
    *,
    expected_allocation: Mapping[str, float] | None,
    srm: Callable[[SourceBinding, Population], SRMResult | NotApplicable],
    trigger_rates: Callable[[SourceBinding], dict[str, float]],
    observations: SrmObservations | None = None,
) -> ExperimentHealth:
    """Collect every health check against one source snapshot.

    *observations* marks a capture that has already pinned its source; the report
    then reads that source directly and reuses the default allocation results the
    capture recorded. Otherwise every check reads one pinned snapshot, so the
    report, its cutoff and its fingerprint describe the same evidence.
    """
    if observations is not None or isinstance(binding.state, ContrastAnalysisState):
        return _collect(
            binding,
            expected_allocation,
            srm=srm,
            trigger_rates=trigger_rates,
            observations=observations,
        )
    population = health_population(binding.source)
    with acquire_health_scope(
        binding.source,
        EvidenceScope(
            metrics=tuple(binding.metrics),
            population=population,
            snapshot=True,
            include_breakouts=True,
        ),
    ) as source:
        return _collect(
            binding if source is binding.source else binding.with_source(source),
            expected_allocation,
            srm=srm,
            trigger_rates=trigger_rates,
            observations=None,
        )


def _collect(
    binding: SourceBinding,
    expected_allocation: Mapping[str, float] | None,
    *,
    srm: Callable[[SourceBinding, Population], SRMResult | NotApplicable],
    trigger_rates: Callable[[SourceBinding], dict[str, float]],
    observations: SrmObservations | None,
) -> ExperimentHealth:
    experiment = getattr(binding.source, "artifact_experiment", None)
    trigger_name = getattr(experiment, "trigger", None) or getattr(
        binding.context, "trigger_name", None
    )
    diagnostics = load_health_diagnostics(
        binding.source,
        binding.metrics,
        include_trigger_checks=trigger_name is not None,
    )
    allocation: dict[str, SrmObservation] | None = None
    if observations is not None:
        allocation = {
            population: observations[key]
            for population in ("assigned", "triggered")
            if (key := srm_observation_key(population, expected_allocation, 0.001, "always_valid"))
            in observations
        } or None
    checks = [_allocation_check(binding, "assigned", expected_allocation, srm, allocation)]
    if trigger_name is not None:
        checks.append(_allocation_check(binding, "triggered", expected_allocation, srm, allocation))
        checks.append(_trigger_rates_check(binding, trigger_rates))
        checks.extend(_trigger_maturity(binding.metrics, diagnostics))
        checks.extend(_feed_completeness(diagnostics))
        checks.append(_trigger_assumptions(binding))
    else:
        checks.extend(
            HealthCheck(
                check=check_name,
                analysis_population="triggered",
                availability="not_applicable",
                context={"reason": "This experiment does not declare a trigger."},
            )
            for check_name in _CHECK_ORDER[1:]
        )
    checks.sort(
        key=lambda check: (
            _CHECK_ORDER.index(check.check),
            check.analysis_population,
            check.metric or "",
            check.feed or "",
        )
    )
    cutoff = _health_cutoff(binding, diagnostics)
    return ExperimentHealth(
        checks=tuple(checks),
        source_snapshot_id=_health_snapshot_id(
            binding.source, cutoff, [health_identity(check) for check in checks]
        ),
        source_cutoff=cutoff,
    )


def _allocation_check(
    binding: SourceBinding,
    population: Population,
    expected_allocation: Mapping[str, float] | None,
    srm: Callable[[SourceBinding, Population], SRMResult | NotApplicable],
    allocation: Mapping[str, SrmObservation] | None,
) -> HealthCheck:
    from increment.estimation.assignment_integrity import assignment_integrity
    from increment.estimation.diagnostics import NotApplicable

    check_name = "assigned_allocation" if population == "assigned" else "triggered_balance"
    if population == "assigned" and isinstance(binding.state, ContrastAnalysisState):
        return HealthCheck(
            check=check_name,
            analysis_population=population,
            availability="unavailable",
            code="integrity.switchback_assignment_law",
            context={"reason": "Switchback order randomization is not a unit-arm allocation law."},
        )
    if population == "assigned":
        admission_design = binding.design
        if (
            admission_design is not None
            and expected_allocation is not None
            and getattr(admission_design, "allocation_scheme", None) == "independent"
            and getattr(admission_design, "allocation", None) is None
        ):
            admission_design = admission_design.model_copy(
                update={"allocation": dict(expected_allocation)}
            )
        integrity = assignment_integrity(admission_design, None)
        if integrity.status == "not_applicable":
            return HealthCheck(
                check=check_name,
                analysis_population=population,
                availability="not_applicable",
                context={"reason": "The declared design has no assignment allocation to test."},
            )
        if integrity.status != "not_checked_missing_counts":
            return HealthCheck(
                check=check_name,
                analysis_population=population,
                availability="unavailable",
                code=integrity.code,
                context={
                    **dict(integrity.context),
                    "reason": f"Assignment-law admission is {integrity.status}.",
                    **(
                        {
                            "route": (
                                "Declare allocation_scheme='independent' on a design "
                                "that uses an independent constant-probability draw, or "
                                "pass the target shares as workflows.srm(experiment, reader, "
                                "inference='fixed', expected={...}) for one predeclared look."
                            )
                        }
                        if integrity.code == "integrity.allocation_scheme_missing"
                        else {}
                    ),
                },
            )
    if allocation is not None and population in allocation:
        result, refusal = allocation[population]
        if result is None:
            code, refusal_context, reason = refusal or (
                "",
                {},
                "No allocation result was captured.",
            )
            return HealthCheck(
                check=check_name,
                analysis_population=population,
                availability=(
                    "not_applicable"
                    if code == "dashboard.allocation_not_applicable"
                    else "unavailable"
                ),
                code=code or None,
                context={**dict(refusal_context), "reason": reason},
            )
    else:
        try:
            result = srm(binding, population)
        except CapabilityError as refusal:
            return HealthCheck(
                check=check_name,
                analysis_population=population,
                availability="unavailable",
                code=refusal.code,
                context={**dict(refusal.context), "reason": str(refusal)},
            )
        except InvalidRequestError as refusal:
            if refusal.code not in {
                "estimation.diagnostics.total_observed_count",
                "estimation.diagnostics.always_srm_predeclared",
            } or (
                refusal.code == "estimation.diagnostics.always_srm_predeclared"
                and population != "triggered"
            ):
                raise
            context = {**dict(refusal.context), "reason": str(refusal)}
            if refusal.code == "estimation.diagnostics.always_srm_predeclared":
                context["route"] = (
                    "Declare allocation on the design or pass expected_allocation "
                    "to health_check()."
                )
            return HealthCheck(
                check=check_name,
                analysis_population=population,
                availability="unavailable",
                code=refusal.code,
                context=context,
            )
    if isinstance(result, NotApplicable):
        return HealthCheck(
            check=check_name,
            analysis_population=population,
            availability="not_applicable",
            context={"reason": result.reason},
        )
    return HealthCheck(
        check=check_name,
        analysis_population=population,
        availability="available",
        conclusion=(
            "warning"
            if result.is_srm
            or result.low_expected_count
            or result.unassigned_units
            or result.mixed_assignment_units
            else "not_rejected"
        ),
        context={
            "grain": result.grain,
            "observed": dict(result.observed),
            "expected": dict(result.expected),
            "unassigned_units": result.unassigned_units,
            "mixed_assignment_units": result.mixed_assignment_units,
            "low_expected_count": result.low_expected_count,
        },
        srm=result,
    )


def _trigger_rates_check(
    binding: SourceBinding, trigger_rates: Callable[[SourceBinding], dict[str, float]]
) -> HealthCheck:
    try:
        rates = trigger_rates(binding)
    except CapabilityError as refusal:
        return HealthCheck(
            check="trigger_rates",
            analysis_population="triggered",
            availability="unavailable",
            code=refusal.code,
            context={**dict(refusal.context), "reason": str(refusal)},
        )
    return HealthCheck(
        check="trigger_rates",
        analysis_population="triggered",
        availability="available",
        conclusion="descriptive",
        context={"description": "Observed trigger share by assigned arm."},
        rates=rates,
    )


def _trigger_maturity(
    metrics: Sequence[Any], evidence: HealthDiagnosticEvidence
) -> list[HealthCheck]:
    checks: list[HealthCheck] = []
    for metric in sorted(metrics, key=lambda item: item.name):
        answer = evidence.edges[metric.name]
        if answer.failure is not None:
            refusal = answer.failure
            checks.append(
                HealthCheck(
                    check="trigger_maturity",
                    analysis_population="triggered",
                    metric=metric.name,
                    availability="unavailable",
                    code=refusal.code,
                    context={**dict(refusal.context), "reason": str(refusal)},
                )
            )
            continue
        observed = answer.observed
        finalized = answer.finalized
        assert observed is not None
        checks.append(
            HealthCheck(
                check="trigger_maturity",
                analysis_population="triggered",
                metric=metric.name,
                availability="available",
                conclusion="warning" if finalized is None else "descriptive",
                context={
                    "observed_edge": observed.isoformat(),
                    "finalized_edge": None if finalized is None else finalized.isoformat(),
                    "reason": "The metric window is not certified complete."
                    if finalized is None
                    else "The retained finalized prefix is shown; no later horizon is inferred.",
                },
            )
        )
    return checks


def _feed_reason(evidence_kind: str, watermark: object | None) -> str:
    if evidence_kind == "native_sql_assignment":
        return (
            "A SQL-backed assignment exposure has no fact feed whose completeness can be certified."
        )
    if evidence_kind == "native_sql_trigger":
        return "A SQL-backed trigger has no fact feed whose completeness can be certified."
    if watermark is None:
        return "No certified completeness watermark is retained for this feed."
    return "Certified completeness watermark at the source cutoff."


def _feed_completeness(evidence: HealthDiagnosticEvidence) -> list[HealthCheck]:
    checks = []
    if evidence.feed_operation_available:
        for item in evidence.feeds:
            feed = item.feed or item.evidence_kind
            if item.metric is not None:
                feed = f"{feed}:{item.metric}"
            watermark = item.complete_through
            checks.append(
                HealthCheck(
                    check="feed_completeness",
                    analysis_population="triggered",
                    metric=item.metric,
                    feed=feed,
                    availability="unavailable" if watermark is None else "available",
                    conclusion=None if watermark is None else "descriptive",
                    context={
                        "evidence_kind": item.evidence_kind,
                        "watermark": None if watermark is None else watermark.isoformat(),
                        "reason": _feed_reason(item.evidence_kind, watermark),
                    },
                )
            )
        if checks:
            return checks
    return [
        HealthCheck(
            check="feed_completeness",
            analysis_population="triggered",
            availability="unavailable",
            context={
                "reason": "This source retains no per-feed completeness certificates.",
                "watermark": None,
            },
        )
    ]


def _trigger_assumptions(binding: SourceBinding) -> HealthCheck:
    from increment.estimation.family import sequential_family_guarantee

    runtime_inference = binding.plan.inference
    declared_plan = getattr(getattr(binding.source, "artifact_experiment", None), "plan", None)
    if declared_plan is None and isinstance(binding.state, DefinitionsArmAnalysisState):
        declared_plan = binding.state.experiment.plan
    inference = getattr(declared_plan, "inference", runtime_inference)
    registration = getattr(inference, "triggered_registration", None)
    family_guarantee = (
        sequential_family_guarantee(registration) if registration is not None else None
    )
    return HealthCheck(
        check="trigger_assumptions",
        analysis_population="triggered",
        availability="available",
        conclusion="descriptive",
        context={
            "counterfactual_trigger_invariance": "assumed, not established by balance or rates",
            "trigger_order_stationarity": "assumed, not established by balance or rates",
            "no_effect_before_trigger": "assumed, not established by balance or rates",
            "triggered_look_predictability": (
                "assumed, not verified: looks, stops and freezes use only already-revealed "
                "triggered-chain information, not assigned results that reveal future "
                "triggered outcomes, including the same metric"
                if family_guarantee is not None
                else None
            ),
            "family_guarantee": family_guarantee,
            "family_guarantee_status": (
                "assumed, not verified" if family_guarantee is not None else None
            ),
            "risk": (
                "Balanced trigger rates do not verify counterfactual trigger invariance.",
                "Trigger-order stationarity remains an assumption.",
                "Assigned results can reveal future triggered outcomes, including "
                "the same metric; they must not inform triggered family stops or freezes.",
            ),
        },
    )


def _health_cutoff(binding: SourceBinding, diagnostics: HealthDiagnosticEvidence) -> str | None:
    if diagnostics.feed_operation_available:
        cutoff = diagnostics.observation_cutoff
        return None if cutoff is None else cutoff.isoformat()
    session = binding.session
    evidence = session.source_snapshot_evidence if session is not None else None
    return None if evidence is None else evidence.observation_cutoff_ts.isoformat()


__all__ = ["SrmObservations", "check_health", "srm_observation_key"]
