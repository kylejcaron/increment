from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, cast

from increment._evidence_types import ContrastStatsRequest, DayAxisRequest
from increment.evidence import EvidenceScope, SequentialCaptureRequest
from tests.evidence_fixtures.transcript import _plain, _restore, _validate_record


def _validate_day_axis_scope(request: DayAxisRequest, scope: EvidenceScope) -> None:
    allowed_populations = set(scope.populations) or {scope.population}
    if request.population not in allowed_populations:
        raise ValueError("day-axis request population is outside its declared scope")
    if scope.grain is not None and request.grain != scope.grain:
        raise ValueError("day-axis request grain does not match its declared scope")
    declared_metrics = tuple(_plain(metric) for metric in scope.metrics)
    if any(_plain(metric) not in declared_metrics for metric in request.metrics):
        raise ValueError("day-axis request metric is outside its declared scope")


def acquire_day_axis(
    source: Any,
    request: DayAxisRequest,
    scope: EvidenceScope,
    execution_context: Any,
    *,
    options: Any = None,
) -> tuple[dict[str, Any], ...]:
    """Materialize acquired and completed daily inputs within one borrowed scope."""
    from increment._analysis_config import UNSET
    from increment._day_axis import (
        _fill_triggered_early_look_rows,
        _resolve_lift_estimands,
        _triggered_group_roster,
    )
    from increment._lift_options import LiftOptions
    from increment._source_identity import source_identity
    from increment.evidence import (
        ComplianceSeriesRequest,
        ComplianceSummary,
        evidence_provider,
        request_evidence,
    )
    from increment.readouts._design_scope import _config_snapshot
    from increment.readouts._evidence import (
        _builtin_source,
        artifact_provenance,
        day_axis_route,
        load_day_axis_rows,
    )

    if execution_context.scope != scope:
        raise ValueError("day-axis scope does not match its execution context")
    _validate_day_axis_scope(request, scope)
    lift = request.caller in {"run_daily_lift", "run_asof_lift"}
    design = source.context.design
    encouragement = None
    estimands = ()
    if lift:
        _, estimands, encouragement = _resolve_lift_estimands(request.estimands, design)
    design_compliance = encouragement is not None and request.dimension is None
    compliance_only = design_compliance and estimands == ("compliance",)
    if lift and options is None:
        options = LiftOptions.resolve(
            source,
            request.metrics,
            decision_method=UNSET,
            sensitivity_methods=UNSET,
            prior=UNSET,
            alpha=source.context.plan.alpha,
        )
    configs = options.configs if lift else ()
    if lift:
        covariates = frozenset(
            metric.name for metric in request.metrics if options.needs_covariate(metric)
        )
        if request.covariate_metrics != covariates:
            raise ValueError("day-axis request must retain its effective covariate metrics")

    with execution_context.borrow(
        source=source, metrics=request.metrics, configs=configs
    ) as borrowed:
        pinned = _builtin_source(borrowed)
        compliance_series = None
        if design_compliance and "compliance" in estimands:
            compliance_request = ComplianceSeriesRequest(
                design=encouragement,
                completed_windows_only=request.completed_windows_only,
                population=request.population,
            )
            series = cast(
                tuple[tuple[object, ComplianceSummary], ...],
                request_evidence(evidence_provider(pinned), compliance_request),
            )
            compliance_series = {
                "request": _plain(compliance_request),
                "answer": [
                    {"day": _plain(day), "summary": _plain(summary.to_wire())}
                    for day, summary in series
                ],
            }
        route = day_axis_route(pinned)
        evidence_slices = (
            ()
            if compliance_only
            else load_day_axis_rows(borrowed, request, scope_is_pinned=True).slices
        )
        slices = []
        for item in evidence_slices:
            rows = _plain(item.rows)
            consumer_rows = _fill_triggered_early_look_rows(
                item.rows,
                req=request,
                metrics=item.metrics,
                experiment=request.experiment or item.experiment,
                source=pinned,
                observation_edges=item.observation_edges,
                source_name=item.source_name,
            )
            slices.append(
                {
                    "metrics": _plain(item.metrics),
                    "view": item.view,
                    "dimension": item.dimension,
                    "source_name": item.source_name,
                    "observation_edges": _plain(item.observation_edges),
                    "experiment": _plain(item.experiment),
                    "rows": rows,
                    "consumer_rows": _plain(consumer_rows),
                }
            )
        arm_roster = None
        if lift and request.experiment is not None and (request.metrics or design_compliance):
            arm_roster = _triggered_group_roster(pinned, request.experiment)
        scope_request = {
            "kind": "increment.readout.day_axis",
            "caller": request.caller,
            "view": request.grain,
            "dimension": request.dimension,
            "completed_windows_only": request.completed_windows_only,
            "estimands": request.estimands,
            "population": request.population,
            "source_identity": source_identity(source),
            "metrics": [
                metric.model_dump(mode="json")
                for metric in sorted(request.metrics, key=lambda metric: metric.name)
            ],
            "configs": [
                _config_snapshot(config, design)
                for config in sorted(configs, key=lambda config: config.metric.name)
            ],
        }
        answer = {
            "route": route,
            "scope": _plain(scope),
            "slices": slices,
            "arm_roster": _plain(arm_roster),
            "compliance_series": compliance_series,
            "scope_request": _plain(scope_request),
            "artifact_provenance": _plain(artifact_provenance(source)),
        }
    return (
        {
            "request_type": "DayAxisRequest",
            "request": _plain(request),
            "answer_type": "MaterializedDayAxisRows",
            "answer": answer,
        },
    )


def _capture_sequential_breakout(
    source: Any,
    borrowed: Any,
    checkpoint: Any,
    readout_scope: Any,
    configs: Sequence[Any],
    estimands: Sequence[str] | None,
) -> Any:
    from increment._analysis_config import normalize_display_correction
    from increment._source_identity import source_identity
    from increment.readouts._breakout import _load_cuped_breakout_frames
    from increment.readouts._common import _runtime_methods
    from increment.readouts._design_scope import _config_snapshot
    from increment.readouts._evidence import _builtin_source, artifact_provenance, day_axis_route
    from increment.sequential_source import registered_breakout_dimension

    if readout_scope is None or readout_scope.view != "breakout":
        return None
    dimension = registered_breakout_dimension(
        checkpoint.registration, population=readout_scope.population
    )
    if readout_scope.segment and (
        len(readout_scope.segment) != 1 or readout_scope.segment[0].property != dimension
    ):
        raise ValueError("sequential breakout scope differs from its registered dimension")
    context = source.context
    policy = context.plan.view_policies.for_view(
        "breakout", mechanism=context.design.mechanism, segmented=True
    )
    selected = tuple(config.metric for config in configs)
    methods = {config.metric.name: _runtime_methods(config, context.design) for config in configs}
    _, _, _, components = _load_cuped_breakout_frames(
        _builtin_source(borrowed),
        configs,
        methods,
        dimension,
        population=readout_scope.population,
        source_name=None,
    )
    scope_request = {
        "source_identity": source_identity(source),
        "metrics": [
            metric.model_dump(mode="json") for metric in sorted(selected, key=lambda m: m.name)
        ],
        "configs": [
            _config_snapshot(config, context.design)
            for config in sorted(configs, key=lambda c: c.metric.name)
        ],
        "dimension": dimension,
        "estimands": None if estimands is None else tuple(estimands),
        "correction": normalize_display_correction(policy.correction),
        "q": policy.q if policy.q is not None else context.plan.q,
        "source_name": None,
        "analysis_population": readout_scope.population,
    }
    return {
        "route": day_axis_route(source),
        "artifact_provenance": _plain(artifact_provenance(source)),
        "scope_request": _plain(scope_request),
        "source_components": _plain(components),
    }


def acquire_sequential(
    source: Any,
    request: SequentialCaptureRequest,
    scope: EvidenceScope,
    execution_context: Any,
    *,
    readout_scope: Any = None,
    configs: Sequence[Any] | None = None,
    estimands: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Capture ordered checkpoint inputs and declaration-owned scope data."""
    from increment._source_identity import source_identity
    from increment.evidence import request_evidence
    from increment.readouts._sequential_scope import _checkpoint_roster

    if execution_context.scope != scope:
        raise ValueError("sequential scope does not match its execution context")

    with execution_context.borrow(source=source) as borrowed:
        checkpoint = cast(Any, request_evidence(borrowed, request))
        roster = _checkpoint_roster(
            source, source.context.design, checkpoint.registration.roster, checkpoint
        )
        answer = {
            "checkpoint": _plain(checkpoint),
            "scope": _plain(scope),
            "roster": _plain(roster),
            "source_identity": _plain(source_identity(source)),
            "breakout": _capture_sequential_breakout(
                source,
                borrowed,
                checkpoint,
                readout_scope,
                source.context.configs if configs is None else configs,
                estimands,
            ),
        }
    return (
        {
            "request_type": "SequentialCaptureRequest",
            "request": _plain(request),
            "answer_type": "SequentialCapture",
            "answer": answer,
        },
    )


def acquire_contrast(
    source: object, request: ContrastStatsRequest, execution_context: Any
) -> dict[str, Any]:
    """Acquire switchback statistics through the typed borrowed-scope loader."""
    from increment._evidence_loaders import load_contrast_stats

    statistics = load_contrast_stats(source, request, execution_context=execution_context)
    return {
        "request_type": "ContrastStatsRequest",
        "request": _plain(request),
        "answer_type": "ContrastStats",
        "answer": {"value": _plain(statistics), "scope": _plain(execution_context.scope)},
    }


def sequential_record_snapshot(record: Mapping[str, Any]) -> Any:
    """Decode the stored order and verify any captured continuation parent."""
    from increment.sequential_state import snapshot_from_json

    _validate_record(record)
    if record["request_type"] != "SequentialCaptureRequest":
        raise ValueError("sequential replay requires a SequentialCaptureRequest record")
    request, answer = record["request"], record["answer"]
    if request.get("finalized") is not True:
        raise ValueError("sequential replay requires a finalized captured request")
    checkpoint = answer["checkpoint"]
    snapshot = snapshot_from_json(json.dumps(checkpoint, sort_keys=True, separators=(",", ":")))
    if request["previous"] is not None:
        parent = snapshot_from_json(
            json.dumps(request["previous"], sort_keys=True, separators=(",", ":"))
        )
        snapshot.verify_parent(parent)
    return snapshot


def scope_sequential_record(
    record: Mapping[str, Any],
    declarations: Any,
    *,
    metrics: Sequence[Any] | None = None,
    estimands: Sequence[str] | None = None,
    configs: Sequence[Any] | None = None,
) -> Any:
    """Execute registered selection and scope assembly without source acquisition."""
    from increment._sequential_readouts import sequential_readout
    from increment.readouts._design_scope import ResolvedRoster
    from increment.readouts._sequential_scope import scope_sequential_results
    from tests.evidence_fixtures.method_replay import ReadoutDeclaration

    if not isinstance(declarations, ReadoutDeclaration):
        raise TypeError("sequential assembly requires captured readout declarations")
    snapshot = sequential_record_snapshot(record)
    answer = record["answer"]
    if _plain(declarations._source_identity_record) != answer["source_identity"]:
        raise ValueError("sequential identity differs from its captured declaration")
    if declarations.context.plan.inference.registration != snapshot.registration:
        raise ValueError("sequential registration differs from its captured checkpoint")
    roster_payload = _restore(answer["roster"])
    roster = ResolvedRoster(
        arms=tuple(roster_payload["arms"]),
        source=roster_payload["source"],
        complete=roster_payload["complete"],
        counts=roster_payload["counts"],
        integrity_counts=roster_payload["integrity_counts"],
        arms_by_metric={
            metric: set(arms) for metric, arms in roster_payload["arms_by_metric"].items()
        },
        known_arms=frozenset(roster_payload["known_arms"]),
    )
    rows = sequential_readout(
        declarations,
        snapshot=snapshot,
        _include_unrequested=True,
        configs=configs,
    )
    return scope_sequential_results(
        declarations,
        rows,
        snapshot,
        metrics=None if metrics is None else [metric.name for metric in metrics],
        estimands=estimands,
        base_roster=roster,
        configs=configs,
    )


def _captured_artifact_provenance(value: Mapping[str, Any]) -> Any:
    from increment._evidence_types import ArtifactProvenance

    payload = _restore(value)
    return ArtifactProvenance(
        manifest_sha256=payload["manifest_sha256"],
        relation_digests=tuple(tuple(pair) for pair in payload["relation_digests"]),
    )


def scope_sequential_breakout_record(
    record: Mapping[str, Any],
    declarations: Any,
    *,
    metrics: Sequence[Any],
    configs: Sequence[Any],
    assembly_inputs: list[dict[str, Any]],
    estimands: Sequence[str] | None = None,
) -> Any:
    """Project the stored segmented checkpoint through registered breakout assembly."""
    from increment._sequential_readouts import sequential_readout
    from increment.breakout.estimates import BreakoutEstimates, _breakout_estimate_row
    from increment.estimation.multiplicity import stamp_multiplicity_status
    from increment.readouts._design_scope import _config_snapshot
    from increment.readouts._multiplicity_scope import scoped_collection
    from increment.sequential_source import registered_breakout_dimension
    from tests.evidence_fixtures.method_replay import ReadoutDeclaration

    if not isinstance(declarations, ReadoutDeclaration):
        raise TypeError("sequential breakout requires captured readout declarations")
    snapshot = sequential_record_snapshot(record)
    captured = record["answer"]["breakout"]
    if captured is None:
        raise ValueError("sequential breakout requires its captured scope inputs")
    request = _restore(captured["scope_request"])
    context = declarations.context
    if context.plan.inference.registration != snapshot.registration:
        raise ValueError("sequential breakout registration differs from its checkpoint")
    if _plain(declarations._source_identity_record) != record["answer"]["source_identity"]:
        raise ValueError("sequential breakout identity differs from its captured declaration")
    dimension = registered_breakout_dimension(
        snapshot.registration, population=request["analysis_population"]
    )
    if (
        request["dimension"] != dimension
        or _plain(request["source_identity"]) != record["answer"]["source_identity"]
        or _plain(request["estimands"]) != _plain(estimands)
        or request["metrics"]
        != [metric.model_dump(mode="json") for metric in sorted(metrics, key=lambda m: m.name)]
        or request["configs"]
        != _plain(
            [
                _config_snapshot(config, context.design)
                for config in sorted(configs, key=lambda c: c.metric.name)
            ]
        )
    ):
        raise ValueError("sequential breakout scope differs from its captured declarations")
    rows = sequential_readout(
        declarations, metrics=metrics, estimands=estimands, snapshot=snapshot, configs=configs
    )
    output = []
    for row in rows:
        checkpoint = row.require_sequential_result().checkpoint
        if len(checkpoint.cell.segment) != 1 or checkpoint.cell.segment[0][0] != dimension:
            raise ValueError("sequential breakout cell differs from its registered dimension")
        output.append(
            _breakout_estimate_row(
                row,
                dimension=dimension,
                dimension_value=checkpoint.cell.segment[0][1],
                source=request["source_name"],
                n_treat=checkpoint.treatment.n,
                n_control=checkpoint.control.n,
                low_reliability=False,
                discovery=row.discovery,
                family_axes=row.family_axes,
                family_q=row.family_q,
                family_threshold=row.family_threshold,
            )
        )
    scope_rows = stamp_multiplicity_status(output)
    assembly_inputs.append(
        {
            "identity_rows": [
                row.model_dump(mode="json", exclude={"source_snapshot_id"}) for row in scope_rows
            ],
            "request": _plain(request),
            "plan": _plain(context.plan),
            "source_components": captured["source_components"],
        }
    )
    scoped = scoped_collection(
        scope_rows,
        BreakoutEstimates,
        context.plan,
        configs,
        request,
        metrics=metrics,
        route=captured["route"],
        view="breakout",
        design=context.design,
        dimension=dimension,
        source=request["source_name"],
        sequential_snapshot=snapshot,
        source_components=_restore(captured["source_components"]),
        artifact_provenance=_captured_artifact_provenance(captured["artifact_provenance"]),
    )
    return BreakoutEstimates(
        [
            stamp_multiplicity_status(
                (row,),
                correction=request["correction"] if row.family_id is not None else None,
            )[0]
            for row in scoped
        ],
        metadata=scoped.metadata,
    )


def scope_contrast_records(
    records: Sequence[Mapping[str, Any]],
    procedures: Mapping[str, Any],
    *,
    source_identity_record: Mapping[str, Any],
    coherence: Any = None,
    effective_modes: Mapping[str, Any] | None = None,
) -> Any:
    """Run the contrast consumer and its complete production scope assembler."""
    from pydantic import TypeAdapter

    from increment.estimation.contrast import ContrastStats, compute_contrast
    from increment.readouts._contrast_scope import scope_contrast_results
    from increment.semantics.models import Metric

    metrics = []
    statistics = {}
    results = []
    for record in records:
        _validate_record(record)
        if record["request_type"] != "ContrastStatsRequest":
            raise ValueError("contrast assembly requires contrast records only")
        metric = TypeAdapter(Metric).validate_python(record["request"]["metric"])
        if metric.name in statistics:
            raise ValueError("contrast assembly requires unique metric records")
        metrics.append(metric)
        stats = ContrastStats.model_validate(_restore(record["answer"]["value"]))
        statistics[metric.name] = stats
        results.extend(compute_contrast(stats, procedures[metric.name]).results)
    if not metrics or set(statistics) != set(procedures):
        raise ValueError("contrast procedures must cover exactly the captured metrics")
    return scope_contrast_results(
        results,
        metrics=metrics,
        stats=statistics,
        procedures=procedures,
        source_identity_record=source_identity_record,
        coherence=coherence,
        effective_modes=effective_modes,
    )


def _day_axis_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    restored = _restore(rows)
    for row in restored:
        day = row.get("ds")
        if isinstance(day, str):
            try:
                row["ds"] = dt.date.fromisoformat(day)
            except ValueError:
                try:
                    row["ds"] = dt.datetime.fromisoformat(day)
                except ValueError:
                    pass
    return restored


def _day_axis_lift_context(request: DayAxisRequest, plan: Any, design: Any) -> tuple[Any, ...]:
    from increment._analysis_config import normalize_display_correction
    from increment._day_axis import _resolve_lift_estimands
    from increment.plan import with_unassigned_procedures
    from increment.readouts._common import _sequential_inference

    _, estimands, encouragement = _resolve_lift_estimands(request.estimands, design)
    correction = None
    if request.grain == "asof":
        if getattr(plan.inference, "registration", None) is not None:
            raise ValueError("registered as-of history requires its captured stopped states")
        policy = plan.view_policies.for_view(
            "asof",
            mechanism=getattr(design, "mechanism", None),
            segmented=request.dimension is not None,
        )
        correction = normalize_display_correction(policy.correction)
        extra = {
            "design": encouragement,
            "inference": _sequential_inference(plan),
            "estimands": estimands,
            "correction": correction,
        }
    else:
        extra = {
            "design": None,
            "estimands": tuple(name for name in estimands if name not in ("compliance", "late")),
        }
    design_compliance = encouragement is not None and request.dimension is None
    if design_compliance:
        extra["estimands"] = tuple(name for name in estimands if name != "compliance")
    effective_plan = (
        with_unassigned_procedures(plan, request.metrics, design=design)
        if request.population == "triggered"
        else plan
    )
    return estimands, encouragement, extra, correction, effective_plan, design_compliance


def estimate_day_axis_record(
    record: Mapping[str, Any],
    request: DayAxisRequest,
    *,
    plan: Any,
    design: Any,
    assembly_inputs: list[dict[str, Any]],
    lift_kwargs: Mapping[str, Any] | None = None,
    configs: Sequence[Any] = (),
) -> Any:
    """Consume completed day-axis inputs through the production pure consumers."""
    from increment._day_axis import (
        _append_compliance_rows,
        _estimate_day_axis_lift_slice,
        _fill_daily_domain_gaps,
        _order_day_axis_rows,
        _stamp_triggered_cell_failures,
    )
    from increment.breakout.estimates import run_daily, run_daily_lift
    from increment.estimation.multiplicity import stamp_multiplicity_status
    from increment.evidence import ComplianceSummary
    from increment.plan import with_unassigned_procedures
    from increment.readouts._evidence import DayAxisSlice

    _validate_record(record)
    if record["request_type"] != "DayAxisRequest" or _plain(request) != record["request"]:
        raise ValueError("day-axis consumer requires the exact captured typed request")
    lift = request.caller in {"run_daily_lift", "run_asof_lift"}
    if lift and lift_kwargs is None:
        raise ValueError("day-axis lift requires its complete resolved options")
    answer = record["answer"]
    results = []
    failure_rows = []
    correction = None
    if lift:
        estimands, encouragement, extra, correction, effective_plan, design_compliance = (
            _day_axis_lift_context(request, plan, design)
        )
    declared = {_plain(metric)["name"]: metric for metric in request.metrics}
    for item in answer["slices"]:
        slice_metrics = tuple(declared[metric["name"]] for metric in item["metrics"])
        if _plain(slice_metrics) != item["metrics"]:
            raise ValueError("day-axis slice metrics differ from the captured request")
        raw_rows = _day_axis_rows(item["consumer_rows"])
        evidence_slice = DayAxisSlice(
            rows=raw_rows,
            metrics=slice_metrics,
            view=item["view"],
            dimension=item["dimension"],
            source_name=item["source_name"],
        )
        if lift:
            kwargs = dict(lift_kwargs or {})
            names = {metric.name for metric in slice_metrics}
            for key in ("methods_by_metric", "prior_by_metric", "method_roles_by_metric"):
                kwargs[key] = {name: value for name, value in kwargs[key].items() if name in names}
            slice_plan = with_unassigned_procedures(plan, slice_metrics, design=design)
            results.extend(
                _estimate_day_axis_lift_slice(
                    partial(run_daily_lift, plan=slice_plan),
                    raw_rows,
                    evidence_slice,
                    request,
                    kwargs,
                    extra,
                    design.control_group,
                    estimands,
                    effective_plan,
                    design,
                )
            )
        else:
            results.extend(
                _stamp_triggered_cell_failures(
                    run_daily(
                        summary=raw_rows,
                        metrics=list(slice_metrics),
                        dimension=item["dimension"],
                        source=item["source_name"],
                        view=item["view"],
                    ),
                    raw_rows,
                    request,
                )
            )
        failure_rows.extend(raw_rows)
    if not lift:
        results = _stamp_triggered_cell_failures(results, (), request)
    if lift:
        compliance = {}
        series = answer["compliance_series"]
        if series is not None:
            from increment.evidence import ComplianceSeriesRequest

            expected_request = ComplianceSeriesRequest(
                design=encouragement,
                completed_windows_only=request.completed_windows_only,
                population=request.population,
            )
            if series["request"] != _plain(expected_request):
                raise ValueError("day-axis compliance series request differs from the consumer")
            compliance = {
                dt.date.fromisoformat(point["day"]): ComplianceSummary.from_wire(
                    point["summary"], study_id=point["summary"]["study_id"]
                )
                for point in series["answer"]
            }
        elif design_compliance and "compliance" in estimands:
            raise ValueError("day-axis consumer requires its captured compliance series")
        if (
            request.experiment is not None
            and (request.metrics or design_compliance)
            and answer["arm_roster"] is None
        ):
            raise ValueError("day-axis consumer requires its captured arm roster")
        _append_compliance_rows(
            results,
            compliance,
            enabled=design_compliance,
            req=request,
            design=encouragement,
            control_group=design.control_group,
            requested_estimands=estimands,
            plan=plan,
            arm_roster=() if answer["arm_roster"] is None else tuple(answer["arm_roster"]),
        )
        results = _stamp_triggered_cell_failures(
            results, failure_rows, request, control_group=design.control_group
        )
    results = [
        row.model_copy(update={"analysis_population": request.population}) for row in results
    ]
    identity_rows = tuple(results)
    results = _fill_daily_domain_gaps(results, request)
    if lift:
        results = stamp_multiplicity_status(results, correction=correction)
    return scope_day_axis_record(
        record,
        request,
        _order_day_axis_rows(results, request),
        identity_rows=identity_rows,
        plan=plan,
        design=design,
        configs=configs,
        assembly_inputs=assembly_inputs,
    )


def scope_day_axis_record(
    record: Mapping[str, Any],
    request: DayAxisRequest,
    rows: Sequence[Any],
    *,
    identity_rows: Sequence[Any],
    plan: Any,
    design: Any,
    assembly_inputs: list[dict[str, Any]],
    configs: Sequence[Any] = (),
) -> Any:
    """Attach scope using captured request data, never a source-shaped adapter."""
    from increment.breakout.estimates import DailyLiftEstimates, DailyMetricValues
    from increment.readouts._design_scope import _config_snapshot
    from increment.readouts._multiplicity_scope import scoped_collection

    lift = request.caller in {"run_daily_lift", "run_asof_lift"}
    expected_configs = [
        _config_snapshot(config, design)
        for config in sorted(configs, key=lambda config: config.metric.name)
    ]
    scope_request = record["answer"]["scope_request"]
    expected_scope = {
        "kind": "increment.readout.day_axis",
        "caller": request.caller,
        "view": request.grain,
        "dimension": request.dimension,
        "completed_windows_only": request.completed_windows_only,
        "estimands": _plain(request.estimands),
        "population": request.population,
        "source_identity": scope_request["source_identity"],
        "metrics": [
            metric.model_dump(mode="json")
            for metric in sorted(request.metrics, key=lambda metric: metric.name)
        ],
        "configs": _plain(expected_configs),
    }
    if scope_request != _plain(expected_scope):
        raise ValueError("day-axis scope request differs from its captured declarations")
    assembly_inputs.append(
        {
            "identity_rows": [
                row.model_dump(mode="json", exclude={"source_snapshot_id"}) for row in identity_rows
            ],
            "request": scope_request,
            "plan": _plain(plan),
            "source_components": [],
        }
    )
    return scoped_collection(
        rows,
        DailyLiftEstimates if lift else DailyMetricValues,
        plan,
        configs,
        _restore(scope_request),
        metrics=request.metrics,
        route=record["answer"]["route"],
        view=request.grain,
        design=design,
        dimension=request.dimension,
        identity_rows=identity_rows,
        artifact_provenance=_captured_artifact_provenance(record["answer"]["artifact_provenance"]),
    )


def assert_switchback_unit_t_reference(
    record: Mapping[str, Any],
    result: Any,
    unit_deltas: Mapping[str, Any],
) -> None:
    """Check unit-t statistics against independently formed exact raw contrasts."""
    import math
    from fractions import Fraction

    from scipy.stats import t

    from tests.evidence_fixtures.replay import _assert_projected_value

    _validate_record(record)
    if record["request_type"] != "ContrastStatsRequest":
        raise ValueError("switchback reference requires a captured contrast record")
    stats = record["answer"]["value"]
    if (
        stats["independence_grain"] != "unit_cycle"
        or result.method != "switchback_unit_t_approximation"
    ):
        raise ValueError("raw unit-t reference requires independent unit-t contrasts")
    if len(unit_deltas) < 2 or any(
        not isinstance(value, Fraction) for value in unit_deltas.values()
    ):
        raise ValueError("raw contrasts must retain exact Fractions for at least two units")
    count = len(unit_deltas)
    mean = sum(unit_deltas.values(), Fraction()) / count
    m2 = sum(((value - mean) ** 2 for value in unit_deltas.values()), Fraction())
    assert stats["n_units"] == result.n_units == count
    _assert_projected_value("contrast.reference_point", result.estimate.value, float(mean))
    _assert_projected_value("contrast.reference_m2", stats["m2_delta"], float(m2))
    _assert_projected_value(
        "contrast.reference_residual",
        stats["mean_residual"],
        float(mean - Fraction(stats["reference_delta"])),
    )
    standard_error = math.sqrt(float(m2 / ((count - 1) * count)))
    _assert_projected_value("contrast.reference_se", result.standard_error, standard_error)
    assert result.dof == count - 1
    alpha = result.alpha if result.alternative == "two-sided" else 2 * result.alpha
    radius = standard_error * t.isf(alpha / 2, count - 1)
    _assert_projected_value("contrast.reference_lb", result.estimate.lb, float(mean) - radius)
    _assert_projected_value("contrast.reference_ub", result.estimate.ub, float(mean) + radius)
