from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from increment._analysis_config import UNSET, _Unset
from increment._literals import ValueScale
from increment._readout_request import _raise as _raise_readout_request
from increment.breakout.estimates import LiftEstimates, reject_quantile_metrics
from increment.estimation.adjust import observational_evidence
from increment.estimation.assignment_integrity import assignment_integrity
from increment.estimation.engine import Method
from increment.estimation.readout_types import CellKey, ReadoutScope, cell_order, freeze
from increment.estimation.results import LiftEstimate
from increment.estimation.sequential import (
    SEQUENTIAL_POLICIES,
    AlwaysValid,
    AsymptoticMean,
    MixedFamily,
)
from increment.readouts._common import (
    _raise,
    _refuse_if_no_treatment_arm,
    _refuse_segmented_registration,
    _require_design,
    _sequential_inference,
)
from increment.readouts._encouragement import encouragement_rows
from increment.readouts._metric_rows import _load_metric_rows
from increment.readouts._multiplicity_scope import attach_multiplicity_scope
from increment.readouts._observational import _estimate_observational
from increment.readouts._passes import _raise_if_all_lift_cells_refused
from increment.readouts._randomized import (
    _estimate_randomized_non_secondary,
    _estimate_randomized_secondary_family,
)
from increment.readouts._requests import _design_compliance_only, _prepare_run_request
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment.estimation.inference import Prior
    from increment.semantics.models import Metric


# Scheduled decomposition target; public parameter count is the API.
def run(
    src: MomentSource,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
    _population: Literal["assigned", "triggered"] = "assigned",
) -> list[LiftEstimate]:
    """Whole-window relative lift, one row per (metric, method, non-control arm).

    Design and statistical plan are both read off *src* (``src.design``,
    ``src.plan``) rather than passed in - a `MomentSource` owns both as
    construction state. Every row is stamped with its resolved
    `role` (None only when `src.plan.declared` is False, i.e. no
    `AnalysisPlan` was ever declared).

    On the randomized fixed-horizon-or-sequential path: a primary's alpha
    (compiled procedure ``alpha``, already split evenly across every declared
    primary) is split again across this metric's own non-control arms; a
    guardrail/unassigned metric estimates at its compiled ``alpha`` unsplit. A
    secondary estimates every in-family (metric x arm) cell at the plan's
    nominal `alpha` first; under a fixed-horizon plan those p-values feed
    one `bh_select` across the whole secondary family, and the selected
    cells' intervals are re-estimated (a second pass over the same
    retained moments) at the Benjamini-Yekutieli FCR level; under an
    AlwaysValid plan, registered raw likelihood evidence drives `e_bh_select`.
    Selected intervals are reinverted at the capped FCR allocation from the
    same stopped checkpoint; unselected cells retain their nominal intervals.
    A non-family secondary (explicitly outside the declared family, or a
    quantile metric under an AlwaysValid plan) estimates once at nominal, with no discovery verdict.

    Under Encouragement, dispatches to encouragement_rows, which delegates
    per metric group to estimate_encouragement instead of estimate_lift;
    estimands (default itt/compliance/late) picks which rows to report.
    Registered sequential plans instead evaluate their declared ITT and
    compliance cells; binary-uptake LATE is refused. Fixed-horizon per-metric
    shifted nulls (declared or plan-bound margins) are refused. A primary's alpha splits across
    its own treatment arms, an in-family secondary faces the same
    BH/e-BH selection every other design gets, and the design-level
    compliance row is estimated once at the plan's own alpha. A
    cluster on src uses cluster-level covariance: additive ITT and LATE
    intervals use Welch references, while relative ITT uses fixed-t Fieller
    sets with ``min(K_T - 1, K_C - 1)`` degrees of freedom. These are working
    approximations. Compliance reports the first-stage lift on the absolute
    scale; complier-relative LATE and any CUPED method are refused under a cluster.

    Under Observational, dispatches to estimate_ate (IPTW by default); a
    nonempty by= (per-segment IPTW) raises NotImplementedError. Only the
    absolute axis (a declared/plan-bound margin_abs) and value_scale are
    supported there; a relative-axis margin is refused. value_scale
    switches a metric to reporting the additive ATE in its own units, for
    when relative lift is unidentified because the control mean sits
    within 4 SEs of 0. This path applies the same role machinery the
    randomized path does: a primary's alpha is Bonferroni-split across its
    treatment arms, secondaries join a BH family at ``plan.q`` with
    Benjamini-Yekutieli FCR re-estimation for selected cells, and a
    guardrail keeps its full compiled ``alpha`` outside any family. Each
    metric's moments are reduced once, and a source that owns a snapshot is
    read through one: the arm gate, family size and routing level, estimates
    and FCR re-estimates all come from that one execution.

    decision_method, sensitivity_methods, and prior are call-wide overrides; without one, each metric
    falls back to its own declared value, then the design default.
    """
    design = _require_design(src, "run")
    selected, configs = _prepare_run_request(
        src,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        metrics=metrics,
        by=by,
        estimands=estimands,
        value_scale=value_scale,
        population=_population,
    )
    # Percentile readouts need the raw source snapshot, and an observational readout reads
    # unit frames and moments that must be one execution. An empty selection reads nothing, so
    # it never opens a snapshot (a definitions-backed capture writes TEMP tables). Metadata/
    # request validation above is intentionally complete before capture.
    needs_snapshot = bool(selected) and (
        design.mechanism == "observational"
        or any(
            getattr(getattr(metric, "winsorization", None), "has_percentile", False)
            for metric in selected
        )
    )
    if needs_snapshot:
        from increment._source_operations import ReadoutSnapshotOperation

        if isinstance(src, ReadoutSnapshotOperation):
            uptake = getattr(design, "uptake", None)
            uptake_facts = () if uptake is None else (uptake.fact,)
            with src.readout_snapshot(
                metrics=selected, population=_population, uptake_facts=uptake_facts
            ) as pinned:
                return _as_collection(
                    _run_prepared(
                        pinned,
                        selected,
                        configs,
                        prior=prior,
                        by=by,
                        estimands=estimands,
                        value_scale=value_scale,
                        population=_population,
                    )
                )
    return _as_collection(
        _run_prepared(
            src,
            selected,
            configs,
            prior=prior,
            by=by,
            estimands=estimands,
            value_scale=value_scale,
            population=_population,
        )
    )


def _as_collection(rows: list[LiftEstimate]) -> LiftEstimates:
    return rows if isinstance(rows, LiftEstimates) else LiftEstimates(rows)


def _config_snapshot(config):
    from increment.estimation.readout_types import refuse_readout

    try:
        return {
            "metric": config.metric.model_dump(mode="json"),
            "decision_method": config.decision_method.model_dump(mode="json"),
            "sensitivity_methods": [
                method.model_dump(mode="json") for method in config.sensitivity_methods
            ],
            "prior": None if config.prior is None else config.prior.model_dump(mode="json"),
            "prior_is_global": config.prior_is_global,
            "methods_explicitly_empty": config.methods_explicitly_empty,
            "decision_defaulted": config.decision_defaulted,
        }
    except (TypeError, ValueError):
        refuse_readout("readout.scope.request_not_canonical", fields=["configs"])


def _randomized_source(selected, evidence_components, assignment_counts, population):
    from hashlib import sha256

    from increment._canonical import canonical_json_bytes

    components = [
        {"kind": kind, "metric": metric.name, "population": population, "sha256": digest}
        for metric in selected
        for kind, digest, _count in (evidence_components[metric.name],)
    ]
    if assignment_counts is not None:
        components.append(
            {
                "kind": "assignment_counts",
                "metric": None,
                "population": population,
                "sha256": sha256(canonical_json_bytes(assignment_counts)).hexdigest(),
            }
        )
    components.sort(
        key=lambda component: (
            component["kind"],
            component["metric"] or "",
            component["population"],
        )
    )
    return {
        "kind": "composite",
        "sha256": sha256(canonical_json_bytes(components)).hexdigest(),
        "components": components,
    }


def _randomized_snapshot_id(source, request):
    from hashlib import sha256

    from increment._canonical import canonical_json_bytes

    preimage = {
        "kind": "increment.readout.snapshot",
        "version": 1,
        "collection": "LiftEstimates",
        "source": source,
        "request": request,
    }
    return "sha256:" + sha256(canonical_json_bytes(preimage)).hexdigest()


def _randomized_cell_record(cell, row, failure, config, snapshot_id):
    from increment.estimation.decision_types import PValueEvidence
    from increment.estimation.readout_types import (
        CellRecord,
        PosteriorInference,
        SamplingInference,
    )

    evidence = None
    if row is not None and failure is None and row.lift is not None:
        try:
            evidence = PValueEvidence(
                cell.hypothesis(), cell.method, row.p_value(), row.reference_kind
            )
        except (AttributeError, ValueError):
            evidence = None
    sampling = SamplingInference(
        available=failure is None and row is not None,
        evidence=evidence,
        reason_code=None if failure is None else failure.code,
        reason_context=None if failure is None else failure.context,
    )
    posterior = (
        None
        if config.prior is None
        else PosteriorInference(
            available=None if row is None else row.posterior_available,
            model=None if row is None else row.posterior_model,
            scale=None if row is None else row.posterior_scale,
            prior=config.prior,
            reason_code=None if row is None else row.posterior_reason_code,
            reason_context=None if row is None else row.posterior_reason_context,
        )
    )
    return CellRecord(
        cell=cell,
        source_snapshot_id=snapshot_id,
        failure=failure,
        sampling=sampling,
        posterior=posterior,
    )


def _scoped_randomized_results(
    src,
    selected,
    configs,
    design,
    plan,
    rows,
    computations,
    observed_by_metric,
    evidence_components,
    population,
):
    """Attach the complete randomized arm roster and captured cell outcomes."""
    from increment.estimation.readout_types import (
        CellFailure,
        PopulationRoster,
        ReadoutMetadata,
        SourceReadoutScope,
    )
    from increment.readouts._common import _runtime_method_roles
    from increment.readouts._design_scope import _roster_evidence_arms, resolve_roster

    roster_arms, roster_source, roster_complete, assignment_counts, integrity_counts = (
        resolve_roster(src, design, population, observed_by_metric)
    )
    known_roster_arms = _roster_evidence_arms(observed_by_metric, assignment_counts)
    source = _randomized_source(selected, evidence_components, assignment_counts, population)
    request = {
        "view": "run",
        "population": population,
        "metrics": [metric.model_dump(mode="json") for metric in selected],
        "configs": [_config_snapshot(config) for config in configs],
        "design": design.model_dump(mode="json"),
        "alpha": plan.alpha,
        "q": plan.q,
        "declared": plan.declared,
        "procedures": {
            name: value.model_dump(mode="json") for name, value in plan.procedures.items()
        },
    }
    trigger_name = src.context.trigger_name
    if trigger_name is not None:
        request["trigger_declared"] = True
        request["trigger_name"] = trigger_name
    snapshot_id = _randomized_snapshot_id(source, request)

    failure_by_cell = {}
    for computation in computations:
        for key, failure in computation.failures.items():
            method = str(failure.context.get("method", ""))
            failure_by_cell[(key.metric, key.group_id, method)] = failure
    method_info = {}
    cells = []
    decision_cells = []
    for metric, config in zip(selected, configs, strict=True):
        methods = (
            ()
            if config.methods_explicitly_empty
            else (config.decision_method, *config.sensitivity_methods)
        )
        roles = _runtime_method_roles(methods)
        for group_id in roster_arms:
            if group_id == design.control_group:
                continue
            for method in methods:
                method_role = roles[method.name]
                test = plan.procedures[metric.name]
                cell = CellKey(
                    kind="arm",
                    metric=metric.name,
                    group_id=group_id,
                    method=method.name,
                    method_role=method_role,
                    estimand="itt",
                    analysis_population=population,
                    value_scale="relative",
                    inference="always_valid"
                    if isinstance(plan.inference, AlwaysValid)
                    else "asymptotic_mean"
                    if isinstance(plan.inference, AsymptoticMean)
                    else "fixed",
                    alternative=getattr(test, "alternative", "two-sided"),
                )
                cells.append(cell)
                if method_role == "decision":
                    decision_cells.append(cell)
                method_info[(metric.name, group_id, method.name, method_role)] = (
                    cell,
                    config,
                    test,
                )
    cell_failures = {}
    for key, (cell, _config, _test) in method_info.items():
        metric_name, group_id, method_name, _method_role = key
        failure = failure_by_cell.get((metric_name, group_id, method_name))
        if failure is not None:
            cell_failures[cell] = CellFailure(
                hypothesis=failure.hypothesis, code=failure.code, context=failure.context
            )
        elif group_id not in observed_by_metric.get(metric_name, set()):
            observed = sorted(observed_by_metric.get(metric_name, set()))
            code = (
                "readout.cell.missing_metric_observations"
                if group_id in known_roster_arms
                else "readout.cell.missing_arm"
            )
            cell_failures[cell] = CellFailure(
                hypothesis=cell.hypothesis(),
                code=code,
                context={
                    "metric": metric_name,
                    "group_id": group_id,
                    "observed_arms": observed,
                    "roster_source": roster_source,
                },
            )
    complete = (
        bool(decision_cells)
        and roster_complete
        and all(cell not in cell_failures for cell in decision_cells)
    )
    missing_cells = [
        cell.model_dump(mode="json") for cell in decision_cells if cell in cell_failures
    ]
    scope_reason_context = None if complete else freeze({"missing_cells": missing_cells})

    result_by_cell = {}
    for row in rows:
        result_by_cell[(row.metric, row.group_id, row.method, row.method_role)] = row
    output = []
    records = []
    for cell in sorted(cells, key=cell_order):
        failure = cell_failures.get(cell)
        row = result_by_cell.get((cell.metric, cell.group_id, cell.method, cell.method_role))
        if row is None and failure is not None:
            row = LiftEstimate(
                metric=cell.metric,
                group_id=cell.group_id,
                method=cell.method,
                method_role=cell.method_role,
                estimand=cell.estimand,
                analysis_population=population,
                value_scale=cell.value_scale,
                alternative=cell.alternative,
                inference=cell.inference,
                role=plan.procedures[cell.metric].role if plan.declared else None,
                lift=None,
                failure_code=failure.code,
                failure_context=failure.context,
                sampling_available=False,
                sampling_reason_code=failure.code,
                sampling_reason_context=failure.context,
                posterior_available=False
                if method_info[(cell.metric, cell.group_id, cell.method, cell.method_role)][1].prior
                is not None
                else None,
            )
        if row is not None:
            row = row.model_copy(
                update={
                    "source_snapshot_id": snapshot_id,
                    "analysis_population": population,
                    "sampling_available": failure is None,
                    "sampling_reason_code": None if failure is None else failure.code,
                    "sampling_reason_context": None if failure is None else failure.context,
                    "failure_code": None if failure is None else failure.code,
                    "failure_context": None if failure is None else failure.context,
                    "decision_scope_complete": complete,
                    "decision_scope_reason_code": None
                    if complete
                    else "readout.scope.decision_incomplete",
                    "decision_scope_reason_context": scope_reason_context,
                }
            )
            output.append(row)
        config = method_info[(cell.metric, cell.group_id, cell.method, cell.method_role)][1]
        records.append(_randomized_cell_record(cell, row, failure, config, snapshot_id))

    ordered_cells = tuple(sorted(set(cells), key=cell_order))
    ordered_decisions = tuple(sorted(set(decision_cells), key=cell_order))
    integrity = assignment_integrity(
        design,
        integrity_counts,
        population=population,
        randomization_grain="cluster" if getattr(src.context, "cluster", None) else "unit",
    )
    source_scope = SourceReadoutScope(
        source_snapshot_id=snapshot_id,
        cells=ordered_cells,
        decision_cells=ordered_decisions,
        rosters=(
            PopulationRoster(
                analysis_population=population,
                arms=roster_arms,
                source=roster_source,
                complete=roster_complete,
            ),
        ),
        decision_complete_by_population={population: complete},
        integrity=(integrity,),
    )
    output, families = attach_multiplicity_scope(output, ordered_cells, plan, configs, snapshot_id)
    scope = ReadoutScope(
        snapshot_id=snapshot_id,
        cells=ordered_cells,
        decision_cells=ordered_decisions,
        populations=(population,),
        families=families,
        by_source={snapshot_id: source_scope},
    )
    metadata = ReadoutMetadata(scope=scope, cells=tuple(records))
    return LiftEstimates(output, metadata=metadata, source=source)


def _run_prepared(
    src: MomentSource,
    selected: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    *,
    prior: Prior | None | _Unset,
    by: Sequence[str],
    estimands: Sequence[str] | None,
    value_scale: Mapping[str, ValueScale] | None,
    population: Literal["assigned", "triggered"],
) -> list[LiftEstimate]:
    """Estimate a validated request inside its source-owned snapshot lifetime."""
    design = _require_design(src, "run")
    plan = src.context.plan

    if isinstance(plan.inference, SEQUENTIAL_POLICIES):
        from increment._sequential_readouts import sequential_readout
        from increment.readouts._sequential_scope import scope_sequential_results
        from increment.sequential_source import source_snapshot

        _refuse_segmented_registration(plan.inference)
        rows = sequential_readout(src, metrics=selected, estimands=estimands)
        return scope_sequential_results(
            src,
            rows,
            source_snapshot(src),
            metrics=[metric.name for metric in selected],
            estimands=estimands,
        )
    call_prior = cast("Prior | None", None if prior is UNSET else prior)
    cluster = src.context.cluster
    if design.mechanism == "encouragement":
        if value_scale:
            _raise("readout.encouragement.value_scale")
        if isinstance(plan.inference, SEQUENTIAL_POLICIES):
            _raise("readout.sequential_inference_supported")
        compliance_only = _design_compliance_only(src, estimands)
        if by and compliance_only:
            _raise_readout_request("readout.run.segment_unsupported", metric="uptake")
        rows_by_metric: dict[str, list[Mapping[str, Any]]] = {}
        if not compliance_only:
            reject_quantile_metrics(
                selected,
                "run() under an encouragement design",
                reason="estimate_encouragement consumes mean-based group_summary moments, "
                "which a quantile metric has none of",
                remedy="Drop the encouragement design or the quantile metric.",
            )
            rows_by_metric = {
                metric.name: list(
                    _load_metric_rows(src, metric, by=by, control_group=design.control_group).rows
                )
                for metric in selected
            }
            if selected:
                _refuse_if_no_treatment_arm(
                    {str(row["group_id"]) for rows in rows_by_metric.values() for row in rows},
                    design.control_group,
                )
        from increment.readouts._design_scope import (
            eligible_late_cells,
            encouragement_expected,
            scope_design_results,
        )

        eligible_late = (
            eligible_late_cells(rows_by_metric, design, cluster)
            if "late" in (estimands or ("itt", "compliance", "late"))
            else set()
        )
        capture: dict[str, Any] = {}
        rows = encouragement_rows(
            src=src,
            metrics=selected,
            rows_by_metric=rows_by_metric,
            configs=configs,
            design=design,
            plan=plan,
            estimands=estimands,
            cluster=cluster,
            caller="run() under an encouragement design",
            capture=capture,
        )
        observed = {
            name: {str(row["group_id"]) for row in metric_rows}
            for name, metric_rows in rows_by_metric.items()
        }
        summary = capture.get("compliance_summary")
        if summary is not None:
            observed["uptake"] = {arm.group_id for arm in summary.arms}
        return scope_design_results(
            src,
            design,
            plan,
            rows,
            capture.get("computations", []),
            configs=configs,
            population=population,
            expected_for=lambda arms: encouragement_expected(
                selected,
                configs,
                design,
                plan,
                estimands=estimands,
                arms=arms,
                eligible_late=eligible_late,
                cluster=cluster,
            ),
            evidence_rows=rows_by_metric,
            observed_by_metric=observed,
            compliance_summary=summary,
            request_extra={
                "metrics": [metric.model_dump(mode="json") for metric in selected],
                "configs": [_config_snapshot(config) for config in configs],
            },
        )
    if design.mechanism == "observational":
        from increment.readouts._design_scope import (
            observational_expected,
            scope_design_results,
        )

        # The one moments reduction per metric: it gates on a treatment arm, and sizes every
        # family and routing level, and every estimate and FCR re-estimate reads it.
        evidence = observational_evidence(src, selected)
        if selected:
            _refuse_if_no_treatment_arm(
                set().union(*(evidence.arms(metric) for metric in selected)),
                design.control_group,
            )
        obs_computations: list[Any] = []
        rows = _estimate_observational(
            src,
            selected,
            configs,
            design,
            plan,
            value_scale=value_scale,
            call_prior=call_prior,
            evidence=evidence,
            computations=obs_computations,
        )
        return scope_design_results(
            src,
            design,
            plan,
            rows,
            obs_computations,
            configs=configs,
            population=population,
            expected_for=lambda arms: observational_expected(
                selected, configs, design, plan, value_scale=value_scale, arms=arms
            ),
            evidence_rows=evidence.rows,
            observed_by_metric={m.name: set(evidence.arms(m)) for m in selected},
            use_counts=False,
            request_extra={
                "metrics": [metric.model_dump(mode="json") for metric in selected],
                "configs": [_config_snapshot(config) for config in configs],
                "value_scale": dict(value_scale or {}),
            },
        )
    if value_scale:
        _raise("readout.value_scale_randomized_absolute")
    effective_inference: AsymptoticMean | AlwaysValid | MixedFamily | None = _sequential_inference(
        plan
    )
    advisory_seen: set[tuple[str, str]] = set()
    observed_arms: set[str] = set()
    observed_by_metric: dict[str, set[str]] = {}
    evidence_components: dict[str, tuple[str, str, int]] = {}
    computations: list[Any] = []
    out, refused_cells, secondary_entries, computations = _estimate_randomized_non_secondary(
        src,
        selected,
        configs,
        design,
        plan,
        by=by,
        effective_inference=effective_inference,
        advisory_seen=advisory_seen,
        observed_arms=observed_arms,
        computations=computations,
        observed_by_metric=observed_by_metric,
        evidence_components=evidence_components,
    )
    secondary_out, secondary_refused = _estimate_randomized_secondary_family(
        src,
        secondary_entries,
        design,
        plan,
        by=by,
        effective_inference=effective_inference,
        advisory_seen=advisory_seen,
        observed_arms=observed_arms,
        computations=computations,
        observed_by_metric=observed_by_metric,
        evidence_components=evidence_components,
    )
    out.extend(secondary_out)
    refused_cells.extend(secondary_refused)
    if selected:
        _refuse_if_no_treatment_arm(observed_arms, design.control_group)
    if not out and refused_cells:
        _raise_if_all_lift_cells_refused(refused_cells, computations)
    order = {metric.name: i for i, metric in enumerate(selected)}
    out.sort(key=lambda row: order[row.metric])
    return _scoped_randomized_results(
        src,
        selected,
        configs,
        design,
        plan,
        out,
        computations,
        observed_by_metric,
        evidence_components,
        population,
    )


def arm_moments(
    src: MomentSource,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
    population: Literal["assigned", "triggered"] = "assigned",
) -> list[LiftEstimate]:
    """Run the existing parallel arm-moment readout implementation."""
    return run(
        src,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        metrics=metrics,
        by=by,
        estimands=estimands,
        value_scale=value_scale,
        _population=population,
    )
