from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import polars as pl

from increment._evidence_types import MetricEvidenceRequest
from increment.estimation.adjust import ObservationalEvidence, _estimate_ate_data
from increment.estimation.encouragement import estimate_compliance, estimate_encouragement
from increment.estimation.engine import Method
from increment.estimation.quantile import quantile_lift_from_rows
from increment.estimation.results import LiftEstimate
from increment.estimation.winsor import estimate_winsor_lift
from increment.evidence import (
    BreakoutMethodMomentsRequest,
    ComplianceRequest,
    ComplianceSummary,
    CountsRequest,
    UnitRowsRequest,
)
from increment.semantics.design import Encouragement, Observational
from increment.semantics.models import MeanMetric, Metric
from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState, WinsorSupport
from tests.evidence_fixtures.replay import (
    _assert_projected_value,
    assert_estimator_matches_old_row,
    estimate_categorical_cuped_transcript,
    estimate_fitted_cuped_transcript,
    estimate_scalar_transcript,
    estimate_unit_cuped_transcript,
)
from tests.evidence_fixtures.transcript import Transcript
from tests.parity_harness.expected_differences import without_gained_arm_facts


@dataclass(frozen=True, slots=True)
class ReadoutDeclaration:
    """Captured construction state, with no evidence acquisition interface."""

    context: Any
    _source_identity_record: Mapping[str, Any]

    def __post_init__(self):
        from increment._source_identity import source_identity_is_canonical
        from increment.declarations import freeze
        from increment.evidence import SourceContext

        if not isinstance(self.context, SourceContext):
            raise TypeError("readout declarations require an actual SourceContext")
        if not source_identity_is_canonical(self._source_identity_record):
            raise ValueError("readout source identity is not canonical")
        object.__setattr__(self, "_source_identity_record", freeze(self._source_identity_record))


@dataclass(frozen=True, slots=True)
class QuantileReplayConfig:
    """Complete estimator options for a materialized quantile input."""

    control_group: str
    methods: tuple[Method, ...]
    prior: Any
    alpha: float
    inference: Any
    method_roles: Mapping[str, str]
    design: Any


@dataclass(frozen=True, slots=True)
class ObservationalReplayConfig:
    """Fully resolved options for source-free observational estimation."""

    design: Observational
    metrics: tuple[Metric, ...]
    cluster: str | None
    methods: tuple[Method, ...]
    prior: Any
    alpha: float
    alternative: str
    value_scale: Mapping[str, Literal["relative", "absolute"]] | None
    null_lifts: Mapping[str, float] | None
    null_abs: Mapping[str, float] | None
    alternatives: Mapping[str, str] | None
    scales: Mapping[str, Literal["relative", "absolute"]]
    raise_if_empty: bool
    resolved_method_roles: Mapping[str, Literal["decision", "sensitivity"]]
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None
    family_member: bool
    route_tail: float | None
    prewarned_unsupported: frozenset[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class WinsorReplayConfig:
    """Scenario-owned metadata and complete estimator options for raw winsor."""

    study_id: str
    missingness: str
    population: Literal["assigned", "triggered"]
    control_group: str
    treatment_group: str
    inference: WinsorInferenceSpec
    support: WinsorSupport | None
    alpha: float
    method: str
    method_role: Literal["decision", "sensitivity"]
    null_lift: float
    null_abs: float | None
    preferred_direction: Any
    reference: Any


@dataclass(frozen=True, slots=True)
class EncouragementReplayConfig:
    """Full statistical identity/options for encouragement consumers."""

    design: Encouragement
    metrics: tuple[Metric, ...]
    estimands: tuple[str, ...]
    methods: tuple[Method, ...]
    prior: Any
    alpha: float
    alternative: str | None
    null_lift: float | None
    null_abs: float | None
    inference: Any
    cluster: str | None
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None
    compliance_requested: bool
    population: Literal["assigned", "triggered"]
    as_of: Any | None
    completed_windows_only: bool


def estimate_quantile_transcript(
    transcript: Transcript,
    request: UnitRowsRequest,
    metric: Metric,
    config: QuantileReplayConfig,
) -> Any:
    """Run the production quantile row consumer on captured unit rows only."""
    if request.metric != metric:
        raise ValueError("quantile request metric differs from the declared metric")
    if request.outcome_stage != "raw":
        raise ValueError("quantile replay requires raw-stage unit rows")
    answer = transcript.answer_for(request)
    frame = answer["frame"]
    columns = frame["columns"]
    if not {"unit_id", "group_id", "y"}.issubset(columns):
        raise ValueError("quantile unit rows lack unit_id, group_id, or y")
    rows = frame["rows"]
    unit_ids = [row["unit_id"] for row in rows]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("quantile unit rows contain duplicate unit identities")
    count_request = CountsRequest(kind="unit", population=request.population)
    counts = transcript.answer_for(count_request)
    observed_counts: dict[str, int] = {}
    for row in rows:
        group_id = str(row["group_id"])
        observed_counts[group_id] = observed_counts.get(group_id, 0) + 1
    if observed_counts != counts["counts"]:
        raise ValueError("quantile unit-row arm counts differ from typed count evidence")
    return quantile_lift_from_rows(
        pl.DataFrame(rows, schema=columns, orient="row"),
        metric,
        config.control_group,
        design=config.design,
        authenticate=None,
        methods=list(config.methods),
        prior=config.prior,
        alpha=config.alpha,
        inference=config.inference,
        method_roles=dict(config.method_roles),
    )


def estimate_observational_transcript(
    transcript: Transcript,
    metric_requests: tuple[MetricEvidenceRequest, ...],
    unit_requests: tuple[UnitRowsRequest, ...],
    config: ObservationalReplayConfig,
) -> Any:
    """Run the existing source-free observational data consumer on captured evidence."""
    import pandas as pd

    metrics = {metric.name: metric for metric in config.metrics}
    if len(metrics) != len(config.metrics):
        raise ValueError("observational metrics contain duplicate names")
    if {request.metric.name for request in metric_requests} != set(metrics):
        raise ValueError("observational metric evidence roster differs from configuration")
    if {request.metric.name for request in unit_requests} != set(metrics):
        raise ValueError("observational unit-row roster differs from configuration")
    expected_covariates = tuple(config.design.adjustment.covariates)
    evidence_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    frames: dict[str, Any] = {}
    for request in metric_requests:
        metric = metrics[request.metric.name]
        if request.metric != metric or request.control_group != config.design.control_group:
            raise ValueError("observational metric request differs from declared design")
        if tuple(request.methods) != config.methods:
            raise ValueError("observational request methods differ from estimator configuration")
        if tuple(request.covariates) != expected_covariates:
            raise ValueError("observational request covariates differ from declared design")
        evidence = transcript.answer_for(request)
        evidence_rows[metric.name] = tuple(evidence["rows"])
    for request in unit_requests:
        metric = metrics[request.metric.name]
        if request.metric != metric or request.population != metric_requests[0].population:
            raise ValueError("observational unit request differs from metric/population identity")
        if tuple(request.covariates) != expected_covariates:
            raise ValueError("observational unit request covariates differ from declared design")
        frame = transcript.answer_for(request)["frame"]
        columns = frame["columns"]
        required = {"unit_id", "group_id", "y", *expected_covariates}
        if config.cluster is not None:
            required.add("cluster_id")
        if not required.issubset(columns):
            raise ValueError(
                f"observational unit frame lacks columns: {sorted(required - set(columns))}"
            )
        frames[metric.name] = pd.DataFrame(frame["rows"], columns=columns)

    def unit_frame_memo(metric: Metric, covariates: Any) -> Any:
        if tuple(covariates) != expected_covariates:
            raise ValueError("observational consumer requested undeclared covariates")
        try:
            return frames[metric.name]
        except KeyError as exc:
            raise ValueError(f"observational unit rows missing for metric {metric.name!r}") from exc

    evidence = ObservationalEvidence(rows=evidence_rows, cluster=config.cluster)
    return _estimate_ate_data(
        config.design,
        metrics=config.metrics,
        cluster=config.cluster,
        methods=list(config.methods),
        prior=config.prior,
        alpha=config.alpha,
        alternative=config.alternative,
        value_scale=config.value_scale,
        null_lifts=config.null_lifts,
        null_abs=config.null_abs,
        alternatives=config.alternatives,
        scales=config.scales,
        _raise_if_empty=config.raise_if_empty,
        resolved_method_roles=config.resolved_method_roles,
        method_roles=config.method_roles,
        evidence=evidence,
        family_member=config.family_member,
        route_tail=config.route_tail,
        unit_frame_memo=unit_frame_memo,
        prewarned_unsupported=config.prewarned_unsupported,
    )


def estimate_winsor_transcript(
    transcript: Transcript,
    request: UnitRowsRequest,
    metric: Metric,
    config: WinsorReplayConfig,
) -> LiftEstimate:
    """Derive the actual winsor input DTO from raw unit rows, then estimate it."""
    if request.metric != metric or request.outcome_stage != "raw":
        raise ValueError("winsor replay requires the declared metric's raw unit rows")
    if request.population != config.population:
        raise ValueError("winsor request population differs from configured population")
    if not isinstance(metric, MeanMetric):
        raise ValueError("winsor replay requires a mean metric")
    win = metric.winsorization
    if win is None or win.upper_percentile is None or win.lower_percentile is not None:
        raise ValueError("winsor replay requires a declared upper-percentile-only metric")
    frame = transcript.answer_for(request)["frame"]
    columns = frame["columns"]
    if not {"unit_id", "group_id", "y"}.issubset(columns):
        raise ValueError("winsor raw rows lack unit_id, group_id, or y")
    rows = frame["rows"]
    unit_ids = [row["unit_id"] for row in rows]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("winsor raw rows contain duplicate unit identities")
    count_request = CountsRequest(kind="unit", population=request.population)
    counts = transcript.answer_for(count_request)
    observed_counts: dict[str, int] = {}
    values: dict[str, list[float]] = {}
    for row in rows:
        group = str(row["group_id"])
        observed_counts[group] = observed_counts.get(group, 0) + 1
        values.setdefault(group, []).append(float(row["y"]))
    if observed_counts != counts["counts"]:
        raise ValueError("winsor raw-row arm counts differ from typed count evidence")
    raw = WinsorRawState(
        metric=metric.name,
        study_id=config.study_id,
        population=config.population,
        missingness=config.missingness,
        quantile=win.upper_percentile,
        support=config.support,
        inference=config.inference,
        arms=tuple(
            RawArm(group_id=group, values=tuple(sorted(arm_values)))
            for group, arm_values in sorted(values.items())
        ),
    )
    return estimate_winsor_lift(
        raw,
        config.control_group,
        config.treatment_group,
        alpha=config.alpha,
        method=config.method,
        method_role=config.method_role,
        null_lift=config.null_lift,
        null_abs=config.null_abs,
        preferred_direction=config.preferred_direction,
        reference=config.reference,
    )


def estimate_compliance_transcript(
    transcript: Transcript,
    request: ComplianceRequest,
    config: EncouragementReplayConfig,
) -> Any:
    """Replay the design-level compliance consumer from its captured summary."""
    if (
        request.design != config.design
        or request.population != config.population
        or request.as_of != config.as_of
        or request.completed_windows_only != config.completed_windows_only
    ):
        raise ValueError("compliance request differs from declared design/cohort/time")
    payload = transcript.answer_for(request)
    summary = ComplianceSummary.from_wire(payload, study_id=str(payload["study_id"]))
    if summary.as_of != request.as_of:
        raise ValueError("compliance summary as-of identity differs from request")
    return estimate_compliance(
        summary,
        config.design,
        alpha=config.alpha,
        inference=config.inference,
        prior=config.prior,
        compliance_requested=config.compliance_requested,
    )


def estimate_encouragement_transcript(
    transcript: Transcript,
    requests: tuple[MetricEvidenceRequest, ...],
    config: EncouragementReplayConfig,
) -> Any:
    """Replay actual encouragement moment rows through the production estimator."""
    metrics = {metric.name: metric for metric in config.metrics}
    if len(metrics) != len(config.metrics):
        raise ValueError("encouragement metrics contain duplicate names")
    if {request.metric.name for request in requests} != set(metrics):
        raise ValueError("encouragement request roster differs from configuration")
    rows: list[Mapping[str, Any]] = []
    for request in requests:
        metric = metrics[request.metric.name]
        if request.metric != metric or request.control_group != config.design.control_group:
            raise ValueError("encouragement request differs from declared metric/design")
        if request.population != config.population or request.by:
            raise ValueError("encouragement request population/breakout differs from configuration")
        if tuple(request.methods) != config.methods:
            raise ValueError("encouragement request methods differ from configuration")
        answer = transcript.answer_for(request)
        rows.extend(answer["rows"])
    return estimate_encouragement(
        config.metrics,
        rows,
        config.design,
        estimands=config.estimands,
        methods=list(config.methods),
        prior=config.prior,
        alpha=config.alpha,
        alternative=config.alternative,
        null_lift=config.null_lift,
        null_abs=config.null_abs,
        inference=config.inference,
        cluster=config.cluster,
        method_roles=config.method_roles,
    )


def estimate_encouragement_breakout_transcript(
    transcript: Transcript,
    requests: tuple[BreakoutMethodMomentsRequest, ...],
    config: EncouragementReplayConfig,
) -> Mapping[tuple[BreakoutMethodMomentsRequest, str], Any]:
    """Estimate source-qualified encouragement segment inputs; assembly is separate."""
    metrics = {metric.name: metric for metric in config.metrics}
    if len(metrics) != len(config.metrics):
        raise ValueError("encouragement metrics contain duplicate names")
    output: dict[tuple[BreakoutMethodMomentsRequest, str], Any] = {}
    for request in requests:
        metric = metrics.get(request.metric.name)
        if (
            metric != request.metric
            or request.method not in config.methods
            or request.population != config.population
        ):
            raise ValueError("breakout request differs from configured metric/method/population")
        if request.breakout.property == "":
            raise ValueError("breakout request has an empty dimension")
        answer = transcript.answer_for(request)
        segments: dict[str, list[Mapping[str, Any]]] = {}
        for row in answer["rows"]:
            if request.breakout.property not in row:
                raise ValueError("breakout moment row lacks its declared dimension")
            value = (
                "__null__"
                if row[request.breakout.property] is None
                else str(row[request.breakout.property])
            )
            segments.setdefault(value, []).append(row)
        for value, segment_rows in segments.items():
            key = (request, value)
            if key in output:
                raise ValueError("duplicate breakout segment evidence")
            output[key] = estimate_encouragement(
                [metric],
                segment_rows,
                config.design,
                estimands=config.estimands,
                methods=[request.method],
                prior=config.prior,
                alpha=config.alpha,
                alternative=config.alternative,
                null_lift=config.null_lift,
                null_abs=config.null_abs,
                inference=config.inference,
                cluster=config.cluster,
                method_roles=config.method_roles,
            )
    return MappingProxyType(output)


def _assert_method_family_row(result: Any, old_row: Mapping[str, Any]) -> None:
    assert_estimator_matches_old_row(result, old_row, consumer="scalar")


def assert_quantile_matches_old_row(result: Any, old_row: Mapping[str, Any]) -> None:
    """Compare the complete estimator-owned quantile row to frozen output."""
    _assert_method_family_row(result, old_row)


def assert_observational_matches_old_row(result: Any, old_row: Mapping[str, Any]) -> None:
    """Compare the complete estimator-owned observational row to frozen output."""
    _assert_method_family_row(result, old_row)


def assert_winsor_matches_old_row(result: Any, old_row: Mapping[str, Any]) -> None:
    """Compare the complete estimator-owned winsor row to frozen output."""
    _assert_method_family_row(result, old_row)


def _project_consumer_row(result: Any) -> Any:
    """Apply the production readout's data-only status/diagnostic projection."""
    from increment.readouts._design_scope import _project_scoped_row

    return _project_scoped_row(result, None, {})


def assert_compliance_matches_old_row(result: Any, old_row: Mapping[str, Any]) -> None:
    """Compare pure compliance output after the production data-only projection."""
    _assert_method_family_row(_project_consumer_row(result), old_row)


def assert_encouragement_matches_old_row(result: Any, old_row: Mapping[str, Any]) -> None:
    """Compare pure encouragement output after the production data-only projection."""
    _assert_method_family_row(_project_consumer_row(result), old_row)


_BREAKOUT_ASSEMBLY_FIELDS = frozenset(
    {
        "dimension",
        "dimension_value",
        "excluded",
        "family_axes",
        "family_q",
        "family_threshold",
        "discovery",
        "low_reliability",
        "n_control",
        "n_treat",
        "policy_name",
        "role",
        "source",
    }
)
_BREAKOUT_READOUT_FIELDS = frozenset({"metric_type", "multiplicity_status", "sampling_available"})
_BREAKOUT_LATER_ASSEMBLY_FIELDS = frozenset({"population", "source_snapshot_id"})
_BREAKOUT_FIELDS_WITHOUT_OLD_AUTHORITY = frozenset(
    {"delta_evidence", "posterior_prob_favorable_reason_code"}
)
_BREAKOUT_OPTIONAL_OLD_FIELDS = frozenset({"mean_control", "mean_treat", "metric_type"})


def assert_encouragement_breakout_matches_old_row(
    result: Any,
    old_row: Mapping[str, Any],
    *,
    request: BreakoutMethodMomentsRequest,
    segment: str,
    source: str | None,
    n_treat: float | None,
    n_control: float | None,
) -> Any:
    """Compare pure estimate fields and explicit production breakout assembly."""
    from increment.breakout.estimates import (
        DEFAULT_RELIABILITY_FLOOR,
        BreakoutEstimate,
        _breakout_estimate_row,
    )

    if not isinstance(result, LiftEstimate):
        raise TypeError(f"expected a pure LiftEstimate, got {type(result).__name__}")
    if request.breakout.source is not None and request.breakout.source != source:
        raise ValueError("resolved breakout source differs from the typed request")
    pure_fields = set(LiftEstimate.model_fields)
    breakout_fields = set(BreakoutEstimate.model_fields)
    owned_fields = (
        (pure_fields & breakout_fields) | _BREAKOUT_ASSEMBLY_FIELDS | _BREAKOUT_READOUT_FIELDS
    ) - (_BREAKOUT_LATER_ASSEMBLY_FIELDS | _BREAKOUT_FIELDS_WITHOUT_OLD_AUTHORITY)
    unclassified = (
        breakout_fields
        - owned_fields
        - _BREAKOUT_LATER_ASSEMBLY_FIELDS
        - _BREAKOUT_FIELDS_WITHOUT_OLD_AUTHORITY
    )
    # The public encouragement breakout stamps compliance rows with the
    # requested outcome metric before calling the shared row projector.
    if result.metric != request.metric.name:
        result = result.model_copy(update={"metric": request.metric.name})
    if (
        unclassified
        or not _BREAKOUT_ASSEMBLY_FIELDS <= breakout_fields
        or not _BREAKOUT_READOUT_FIELDS <= breakout_fields
    ):
        raise AssertionError(
            "breakout ownership map differs from production model: "
            f"unmapped={sorted(unclassified)}, "
            f"missing_assembly={sorted(_BREAKOUT_ASSEMBLY_FIELDS - breakout_fields)}, "
            f"missing_readout={sorted(_BREAKOUT_READOUT_FIELDS - breakout_fields)}"
        )
    assembled = _breakout_estimate_row(
        result,
        dimension=request.breakout.property,
        dimension_value=segment,
        source=source,
        n_treat=n_treat,
        n_control=n_control,
        low_reliability=(
            (n_treat is not None and n_treat < DEFAULT_RELIABILITY_FLOOR)
            or (n_control is not None and n_control < DEFAULT_RELIABILITY_FLOOR)
        ),
    )
    assembled = assembled.model_copy(update={"policy_name": "compiled_plan"})
    from increment.estimation.multiplicity import stamp_multiplicity_status
    from increment.readouts._metric_type import declared_metric_type

    assembled = stamp_multiplicity_status([assembled])[0]
    metric_type = declared_metric_type(assembled, {request.metric.name: request.metric.type})
    readout_updates: dict[str, Any] = {"metric_type": metric_type}
    if assembled.sampling_available is None:
        readout_updates["sampling_available"] = True
    assembled = assembled.model_copy(update=readout_updates)
    actual = without_gained_arm_facts(old_row, assembled.model_dump(mode="python"))
    legacy_absent = breakout_fields - old_row.keys() - _BREAKOUT_LATER_ASSEMBLY_FIELDS
    permitted_absent = _BREAKOUT_FIELDS_WITHOUT_OLD_AUTHORITY | _BREAKOUT_OPTIONAL_OLD_FIELDS
    if (
        not _BREAKOUT_FIELDS_WITHOUT_OLD_AUTHORITY <= legacy_absent
        or legacy_absent - permitted_absent
    ):
        raise AssertionError(
            "breakout fields without OLD authority changed: "
            f"actual={sorted(legacy_absent)}, permitted={sorted(permitted_absent)}"
        )
    missing_old_authority = (owned_fields - _BREAKOUT_OPTIONAL_OLD_FIELDS) - old_row.keys()
    if missing_old_authority:
        raise AssertionError(
            f"OLD breakout row lacks required owned fields: {sorted(missing_old_authority)}"
        )
    missing_assembly = _BREAKOUT_ASSEMBLY_FIELDS - old_row.keys()
    if missing_assembly:
        raise AssertionError(f"OLD breakout assembly fields missing: {sorted(missing_assembly)}")
    old_unowned = old_row.keys() - owned_fields
    if old_unowned - _BREAKOUT_LATER_ASSEMBLY_FIELDS:
        raise AssertionError(
            f"OLD breakout row has unclassified later fields: "
            f"{sorted(old_unowned - _BREAKOUT_LATER_ASSEMBLY_FIELDS)}"
        )
    missing_actual = (owned_fields - _BREAKOUT_OPTIONAL_OLD_FIELDS) - actual.keys()
    if missing_actual:
        raise AssertionError(
            f"owned breakout fields lack production values: {sorted(missing_actual)}"
        )
    for field in owned_fields:
        if field in old_row:
            _assert_projected_value(field, actual.get(field), old_row[field])
        elif actual.get(field) is not None:
            raise AssertionError(
                f"breakout field {field!r} has a value but no OLD authority: {actual[field]!r}"
            )
    return assembled


METHOD_REPLAYERS: Mapping[str, Callable[..., Any]] = MappingProxyType(
    {
        "scalar": estimate_scalar_transcript,
        "fitted_cuped": estimate_fitted_cuped_transcript,
        "categorical_cuped": estimate_categorical_cuped_transcript,
        "unit_cuped": estimate_unit_cuped_transcript,
        "quantile": estimate_quantile_transcript,
        "observational": estimate_observational_transcript,
        "winsor": estimate_winsor_transcript,
        "compliance": estimate_compliance_transcript,
        "encouragement": estimate_encouragement_transcript,
        "encouragement_breakout": estimate_encouragement_breakout_transcript,
    }
)
