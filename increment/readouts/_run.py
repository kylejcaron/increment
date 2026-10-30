from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, cast

from increment._analysis_config import UNSET, _Unset, effective_methods
from increment._evidence_types import EvidenceScope, MetricEvidence, MetricEvidenceRequest
from increment._literals import ValueScale
from increment._readout_request import _raise as _raise_readout_request
from increment._source_types import is_custom_source
from increment.breakout.estimates import LiftEstimates, reject_quantile_metrics
from increment.declarations import Method
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
from increment.readouts._design_scope import _config_snapshot
from increment.readouts._encouragement import encouragement_rows
from increment.readouts._evidence import (
    _builtin_source,
    acquire_observational_inputs,
    artifact_provenance,
    has_verified_sequential_checkpoint,
    prior_exclusions,
)
from increment.readouts._metric_rows import load_metric_evidence
from increment.readouts._observational import _estimate_observational
from increment.readouts._passes import _raise_if_all_lift_cells_refused
from increment.readouts._randomized import (
    _estimate_randomized_non_secondary,
    _estimate_randomized_secondary_family,
)
from increment.readouts._randomized_scope import _scoped_randomized_results
from increment.readouts._requests import _design_compliance_only, _prepare_run_request
from increment.sources import EvidenceSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment._source_types import _BuiltinMomentSource
    from increment.estimation.inference import Prior
    from increment.kernel.context import EvidenceExecutionContext
    from increment.semantics.models import Metric


# Scheduled decomposition target; public parameter count is the API.
def run(
    src: EvidenceSource,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
    _population: Literal["assigned", "triggered"] = "assigned",
    coherence: Literal["captured", "live"] = "captured",
    execution_context: EvidenceExecutionContext | None = None,
) -> LiftEstimates:
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

    Every arm row carries an explicit ``analysis_population`` axis (``assigned`` or
    ``triggered``); rows from distinct populations are separate identities.
    Registered triggered cells that a source cannot estimate are retained as typed
    unavailable rows and make the triggered decision scope incomplete.
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
    # Snapshot validation is pure and must refuse non-portable learner factories
    # before any source snapshot or evidence read begins.
    for config in configs:
        _config_snapshot(config, design)
    if design.mechanism == "observational" and not isinstance(
        src.context.plan.inference, SEQUENTIAL_POLICIES
    ):
        from increment.estimation.adjust import _validate_adjustment_methods
        from increment.readouts._common import _runtime_methods

        for config in configs:
            _validate_adjustment_methods(_runtime_methods(config, design))

    # Raw frames and method-scoped covariates share one capture across the
    # selected metrics, including their common observation horizon. An empty
    # selection reads nothing. Request validation is complete before capture.
    design_summary_only = _design_compliance_only(src.context, estimands)
    sequential = isinstance(src.context.plan.inference, SEQUENTIAL_POLICIES)
    needs_live_evidence = (bool(selected) or design_summary_only) and not sequential
    needs_snapshot = sequential or (
        needs_live_evidence
        and (
            design.mechanism == "observational"
            or (
                not design_summary_only
                and any(
                    method.variance_reduction in ("cuped", "cuped_multicovariate")
                    for config in configs
                    for method in effective_methods(config, design=design)
                )
            )
            or any(
                getattr(getattr(metric, "winsorization", None), "has_percentile", False)
                for metric in selected
            )
        )
    )
    uptake = getattr(design, "uptake", None)
    capture_all_sequential_family = sequential and not design_summary_only
    scope_metrics = tuple(src.context.metrics) if capture_all_sequential_family else tuple(selected)
    scope_configs = tuple(src.context.configs) if capture_all_sequential_family else tuple(configs)
    checkpoint_only = (
        sequential
        and not design_summary_only
        and bool(selected)
        and has_verified_sequential_checkpoint(src)
    )
    scope = EvidenceScope(
        metrics=scope_metrics,
        population=_population,
        configs=scope_configs,
        uptake_facts=() if uptake is None else (uptake.fact,),
        snapshot=needs_snapshot,
        execution=bool(selected) or design_summary_only,
        checkpoint_only=checkpoint_only,
    )
    if not selected and not design_summary_only:
        return _as_collection(
            _run_prepared(
                _builtin_source(src),
                selected,
                configs,
                prior=prior,
                by=by,
                estimands=estimands,
                value_scale=value_scale,
                population=_population,
                scope_active=False,
            )
        )
    from uuid import uuid4

    from increment.kernel.context import EvidenceExecutionContext
    from increment.kernel.contracts import _refuse
    from increment.readouts._provider import native_scope_provider

    if execution_context is None:
        context = EvidenceExecutionContext(
            src if is_custom_source(src) else native_scope_provider(src),
            scope,
            run_id=uuid4().hex,
            coherence=coherence,
        )
        manager = context.evidence_scope()
    else:
        context = execution_context
        if context.requested_coherence != coherence:
            _refuse("kernel.request.invalid", field="coherence")
        context.validate_request(src, selected, configs, population=_population)
        uptake = getattr(design, "uptake", None)
        if uptake is not None and (
            estimands is None or {"compliance", "late"}.intersection(estimands)
        ):
            uptake_fact = uptake.fact
            if uptake_fact not in context.scope.uptake_facts:
                _refuse(
                    "kernel.context.undeclared_stream",
                    stream=f"fact:{uptake_fact}",
                )
        manager = context.borrow()
    with manager as scoped:
        context.validate_observational_streams(src, design)
        rows = _run_prepared(
            _builtin_source(scoped),
            selected,
            configs,
            prior=prior,
            by=by,
            estimands=estimands,
            value_scale=value_scale,
            population=_population,
            scope_active=True,
            snapshot_source=src,
        )
    result = _as_collection(rows)
    if result.metadata is not None:
        from increment._immutable import _FrozenMapping

        result._metadata = result.metadata.model_copy(
            update={
                "coherence": (
                    None if "provider" in context.effective_modes.values() else context.coherence
                ),
                "effective_modes": _FrozenMapping(context.effective_modes),
            }
        )
    return result


def _as_collection(rows: list[LiftEstimate]) -> LiftEstimates:
    return rows if isinstance(rows, LiftEstimates) else LiftEstimates(rows)


def _observational_adjustment_identity(src, selected, configs, design, rows, unit_frame_memo):
    from increment._analysis_config import effective_methods

    adjusted_methods = {
        (config.metric.name, method.name)
        for config in configs
        for method in effective_methods(config, design=design)
        if method.name in {"iptw", "dml", "aipw"}
    }
    used = {
        (row.metric, row.method)
        for row in rows
        if row.failure_code is None and (row.metric, row.method) in adjusted_methods
    }
    adjusted = {metric for metric, _method in used}
    if not adjusted:
        return None

    import narwhals as nw

    from increment.estimation.readout_types import StreamingDigest

    identity = {}
    for metric in selected:
        if metric.name not in adjusted:
            continue
        frame = nw.from_native(
            unit_frame_memo(metric, design.adjustment.covariates),
            eager_only=True,
        )
        digest = StreamingDigest()
        rows = sorted(
            frame.iter_rows(named=True),
            key=lambda row: (str(row.get("group_id")), str(row.get("unit_id"))),
        )
        for row in rows:
            digest.update({key: _digestable(value) for key, value in row.items()})
        identity[metric.name] = {
            "columns": tuple(frame.columns),
            "sha256": digest.hexdigest(),
            "n_rows": digest.count,
        }
    return identity


def _digestable(value):
    """Missing covariates arrive as NaN; canonical JSON admits only finite
    numbers, so non-finite floats are encoded as tagged strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _preflight_roster(src, design, population, observed_by_metric=None, base_roster=None):
    if design.mechanism == "observational":
        return None
    from increment.readouts._design_scope import resolve_roster

    return resolve_roster(
        src,
        design,
        population,
        {} if observed_by_metric is None else observed_by_metric,
        base_roster=base_roster,
    )


def _restore_legacy_prior_exclusion(src, configs, plan):
    """Reapply an old wire exclusion only while its bound prior remains active."""
    from increment.decision import MultiplicityFamily

    base_configs = {config.metric.name: config for config in src.context.configs}
    procedures = dict(plan.procedures)
    changed = False
    for config in configs:
        procedure = procedures.get(config.metric.name)
        base = base_configs.get(config.metric.name)
        if (
            config.metric.name not in prior_exclusions(src)
            or procedure is None
            or base is None
            or base.prior is None
            or config.prior is not None
            or not config.prior_is_global
            or procedure.role != "secondary"
            or procedure.family.member
            or not isinstance(procedure.family.family, MultiplicityFamily)
        ):
            continue
        family = procedure.family.model_copy(update={"member": True})
        procedures[config.metric.name] = procedure.model_copy(update={"family": family})
        changed = True
    return plan.model_copy(update={"procedures": procedures}) if changed else plan


def _run_plan_and_roster(src, design, configs, population):
    plan = _restore_legacy_prior_exclusion(src, configs, src.context.plan)
    if isinstance(plan.inference, SEQUENTIAL_POLICIES):
        return plan, None
    return plan, _preflight_roster(src, design, population)


def _load_encouragement_evidence(src, selected, configs, design, by, *, population, scope_active):
    config_by_metric = {config.metric.name: config for config in configs}
    return {
        metric.name: load_metric_evidence(
            src,
            MetricEvidenceRequest(
                metric=metric,
                control_group=design.control_group,
                by=tuple(by),
                methods=tuple(effective_methods(config_by_metric[metric.name], design=design)),
                population=population,
            ),
            scope_active=scope_active,
        )
        for metric in selected
    }


def _run_prepared(  # noqa: PLR0915
    src: _BuiltinMomentSource,
    selected: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    *,
    prior: Prior | None | _Unset,
    by: Sequence[str],
    estimands: Sequence[str] | None,
    value_scale: Mapping[str, ValueScale] | None,
    population: Literal["assigned", "triggered"],
    scope_active: bool = False,
    snapshot_source: Any = None,
) -> list[LiftEstimate]:
    """Estimate a validated request inside its source-owned snapshot lifetime."""
    design = _require_design(src, "run")
    plan, preflight_roster = _run_plan_and_roster(src, design, configs, population)

    if isinstance(plan.inference, SEQUENTIAL_POLICIES):
        from increment._sequential_readouts import sequential_readout
        from increment.readouts._evidence import current_source_snapshot
        from increment.readouts._sequential_scope import scope_sequential_results

        _refuse_segmented_registration(plan.inference)
        snapshot = current_source_snapshot(src if snapshot_source is None else snapshot_source)
        from dataclasses import replace

        selected_names = {config.metric.name for config in configs}
        sequential_configs = (
            *configs,
            *(
                replace(
                    config,
                    sensitivity_methods=(),
                    source_methods=(config.decision_method,),
                )
                for config in src.context.configs
                if config.metric.name not in selected_names
            ),
        )
        rows = sequential_readout(
            src,
            metrics=selected,
            estimands=estimands,
            snapshot=snapshot,
            _include_unrequested=True,
            configs=sequential_configs,
        )
        return scope_sequential_results(
            src,
            rows,
            snapshot,
            base_roster=preflight_roster,
            metrics=[metric.name for metric in selected],
            estimands=estimands,
            configs=sequential_configs,
        )
    call_prior = cast("Prior | None", None if prior is UNSET else prior)
    cluster = src.context.cluster
    if design.mechanism == "encouragement":
        if value_scale:
            _raise("readout.encouragement.value_scale")
        if isinstance(plan.inference, SEQUENTIAL_POLICIES):
            _raise("readout.sequential_inference_supported")
        compliance_only = _design_compliance_only(src.context, estimands)
        if by and compliance_only:
            _raise_readout_request("readout.run.segment_unsupported", metric="uptake")
        rows_by_metric: dict[str, list[Mapping[str, Any]]] = {}
        evidence_by_metric: dict[str, MetricEvidence] = {}
        if not compliance_only:
            reject_quantile_metrics(
                selected,
                "run() under an encouragement design",
                reason="estimate_encouragement consumes mean-based group_summary moments, "
                "which a quantile metric has none of",
                remedy="Drop the encouragement design or the quantile metric.",
            )
            evidence_by_metric = _load_encouragement_evidence(
                src,
                selected,
                configs,
                design,
                by,
                population=population,
                scope_active=scope_active,
            )
            rows_by_metric = {
                name: list(evidence.rows) for name, evidence in evidence_by_metric.items()
            }
            if selected:
                _refuse_if_no_treatment_arm(
                    {
                        str(row["group_id"])
                        for rows in rows_by_metric.values()
                        for row in rows
                        if row.get("group_id") is not None
                    },
                    design.control_group,
                )
                preflight_roster = _preflight_roster(
                    src,
                    design,
                    population,
                    {
                        metric: {
                            str(row["group_id"])
                            for row in metric_rows
                            if row.get("group_id") is not None
                        }
                        for metric, metric_rows in rows_by_metric.items()
                    },
                    preflight_roster,
                )
        from increment.readouts._design_scope import (
            encouragement_expected,
            scope_design_results,
        )

        capture: dict[str, Any] = {}
        rows = encouragement_rows(
            src=src,
            metrics=selected,
            rows_by_metric=rows_by_metric,
            configs=configs,
            evidence_by_metric=evidence_by_metric,
            design=design,
            plan=plan,
            estimands=estimands,
            population=population,
            cluster=cluster,
            caller="run() under an encouragement design",
            capture=capture,
        )
        observed = {
            name: {str(row["group_id"]) for row in metric_rows if row.get("group_id") is not None}
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
            estimands=estimands,
            expected_for=lambda arms: encouragement_expected(
                selected,
                configs,
                design,
                plan,
                estimands=estimands,
                arms=arms,
                cluster=cluster,
            ),
            evidence_rows=rows_by_metric,
            method_evidence=evidence_by_metric,
            observed_by_metric=observed,
            compliance_summary=summary,
            base_roster=preflight_roster,
            request_extra={
                "metrics": [
                    m.model_dump(mode="json") for m in sorted(selected, key=lambda m: m.name)
                ],
                "configs": [
                    _config_snapshot(config, design)
                    for config in sorted(configs, key=lambda c: c.metric.name)
                ],
            },
        )
    if design.mechanism == "observational":
        from increment.readouts._design_scope import (
            observational_expected,
            scope_design_results,
        )

        # The one moments reduction per metric: it gates on a treatment arm, and sizes every
        # family and routing level, and every estimate and FCR re-estimate reads it.
        evidence, unit_frame_memo = acquire_observational_inputs(
            src, selected, population=population
        )
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
            unit_frame_memo=unit_frame_memo,
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
                "metrics": [
                    m.model_dump(mode="json") for m in sorted(selected, key=lambda m: m.name)
                ],
                "configs": [
                    _config_snapshot(config, design)
                    for config in sorted(configs, key=lambda c: c.metric.name)
                ],
                "value_scale": value_scale,
                "adjustment_input_identity": _observational_adjustment_identity(
                    src, selected, configs, design, rows, unit_frame_memo
                ),
            },
        )
    if value_scale:
        _raise("readout.value_scale_randomized_absolute")
    effective_inference: AsymptoticMean | AlwaysValid | MixedFamily | None = _sequential_inference(
        plan
    )
    advisory_seen: set[tuple[object, ...]] = set()
    observed_arms: set[str] = set()
    observed_by_metric: dict[str, set[str]] = {}
    evidence_by_metric = {}
    computations: list[Any] = []
    out, refused_cells, secondary_entries, computations = _estimate_randomized_non_secondary(
        src,
        selected,
        configs,
        design,
        plan,
        by=by,
        population=population,
        effective_inference=effective_inference,
        advisory_seen=advisory_seen,
        observed_arms=observed_arms,
        computations=computations,
        observed_by_metric=observed_by_metric,
        evidence_by_metric=evidence_by_metric,
        scope_active=scope_active,
    )
    secondary_out, secondary_refused = _estimate_randomized_secondary_family(
        src,
        secondary_entries,
        design,
        plan,
        by=by,
        population=population,
        effective_inference=effective_inference,
        advisory_seen=advisory_seen,
        observed_arms=observed_arms,
        computations=computations,
        observed_by_metric=observed_by_metric,
        evidence_by_metric=evidence_by_metric,
        scope_active=scope_active,
    )
    out.extend(secondary_out)
    refused_cells.extend(secondary_refused)
    if selected:
        _refuse_if_no_treatment_arm(observed_arms, design.control_group)
    if refused_cells and not any(row.failure_code is None for row in out):
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
        evidence_by_metric,
        artifact_provenance(src),
        population,
        preflight_roster,
    )


def arm_moments(
    src: EvidenceSource,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
    population: Literal["assigned", "triggered"] = "assigned",
    coherence: Literal["captured", "live"] = "captured",
    execution_context: EvidenceExecutionContext | None = None,
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
        coherence=coherence,
        execution_context=execution_context,
    )
