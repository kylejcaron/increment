from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from increment._analysis_config import (
    UNSET,
    _Unset,
    normalize_display_correction,
    overlay_configs,
    select_metrics,
)
from increment._evidence_types import BreakoutIdentity, BreakoutUnitRowsRequest, EvidenceScope
from increment._frame_validation import SCALAR_COVARIATE_REFUSALS
from increment._literals import Correction
from increment._readout_request import ReadoutRequest, validate_request
from increment._source_types import is_custom_source
from increment.breakout.estimates import (
    DEFAULT_RELIABILITY_FLOOR,
    BreakoutEstimate,
    BreakoutEstimates,
    _breakout_estimate_row,
    run_breakout,
)
from increment.errors import CodedError
from increment.estimation.encouragement import ESTIMANDS, estimate_encouragement
from increment.estimation.engine import Method, _validate_methods
from increment.estimation.multiplicity import stamp_multiplicity_status
from increment.estimation.sequential import SEQUENTIAL_POLICIES
from increment.readouts._common import (
    _raise,
    _require_design,
    _runtime_method_roles,
    _runtime_methods,
    _sequential_inference,
    _warn,
)
from increment.readouts._encouragement import _encouragement_cell_identities
from increment.readouts._evidence import (
    day_axis_route,
    evidence_scope,
    fits_uncovered_cells_locally,
    load_breakout_evidence,
    retains_cuped_relation,
)
from increment.readouts._metric_rows import _load_breakout_metric_rows
from increment.readouts._passes import _append_lift_cell_failure
from increment.sources import EvidenceSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._evidence_types import MetricEvidence
    from increment._literals import ValueScale
    from increment.estimation.inference import Prior
    from increment.semantics.design import Encouragement
    from increment.semantics.models import Metric


def _missing_encouragement_observation(
    *,
    metric: str,
    group_id: str,
    estimand: str,
    value_scale: ValueScale,
    method: str,
    method_role: Literal["decision", "sensitivity"],
    dimension: str,
    value: str,
    source: str | None,
    n_treat: float,
    n_control: float,
) -> BreakoutEstimate:
    context = {
        "metric": metric,
        "group_id": group_id,
        "dimension": dimension,
        "dimension_value": value,
        "reason": "no usable metric observations",
    }
    failure_code = "readout.cell.missing_metric_observations"
    return BreakoutEstimate(
        metric=metric,
        group_id=group_id,
        estimand=estimand,
        value_scale=value_scale,
        method=method,
        method_role=method_role,
        role="exploratory",
        dimension=dimension,
        dimension_value=value,
        source=source,
        n_treat=n_treat,
        n_control=n_control,
        lift=None,
        excluded="missing_metric_observations",
        failure_code=failure_code,
        failure_context=context,
        sampling_available=False,
        sampling_reason_code=failure_code,
        sampling_reason_context=context,
        posterior_available=False,
        posterior_reason_code=failure_code,
        posterior_reason_context=context,
    )


def _encouragement_breakout_rows(
    selected: Sequence[Metric],
    *,
    dimension: str,
    source_name: str | None,
    design: Encouragement,
    configs: Mapping[str, Any],
    methods_by_metric: Mapping[str, Any],
    roles_by_metric: Mapping[str, Any],
    estimands: Sequence[str],
    alpha: float,
    method_evidence: Mapping[str, MetricEvidence],
) -> list[BreakoutEstimate]:
    for metric_methods in methods_by_metric.values():
        _validate_methods(metric_methods)
    results: list[BreakoutEstimate] = []
    missing_cells: list[tuple[str, str, str, float, float]] = []
    compliance_scales_by_metric: dict[str, set[ValueScale]] = {
        metric.name: set() for metric in selected
    }
    for metric in selected:
        metric_methods = methods_by_metric[metric.name]
        evidence = method_evidence[metric.name]
        rows = evidence.rows
        segments: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            value = "__null__" if row[dimension] is None else str(row[dimension])
            segments.setdefault(value, []).append(row)
        for value in sorted(segments):
            segment_rows = segments[value]
            n_by_group = {
                str(row.get("group_id")): float(row["n"])
                for row in segment_rows
                if row.get("n") is not None
            }
            empty_groups = {
                str(row["group_id"])
                for row in segment_rows
                if row.get("n") is not None and float(row["n"]) == 0
            }
            missing_groups = (
                {str(row["group_id"]) for row in segment_rows}
                if design.control_group in empty_groups
                else empty_groups
            )
            missing_cells.extend(
                (
                    metric.name,
                    value,
                    group_id,
                    n_by_group.get(group_id, 0.0),
                    n_by_group.get(design.control_group, 0.0),
                )
                for group_id in sorted(missing_groups)
                if group_id != design.control_group
            )
            usable_rows = [row for row in segment_rows if str(row["group_id"]) not in empty_groups]
            if design.control_group in empty_groups or not usable_rows:
                continue
            if not any(row.get("group_id") == design.control_group for row in usable_rows):
                _warn(
                    "readouts.breakout.segment_no_control_arm",
                    dimension=dimension,
                    value=value,
                    control_group=design.control_group,
                    stacklevel=2,
                )
                continue
            n_control_arm = n_by_group.get(design.control_group)
            lift_estimates = []
            diagnostic_keys = set()
            for method in metric_methods:
                try:
                    method_rows = (
                        evidence.rows_for_method(method, dimension=dimension, dimension_value=value)
                        if method.variance_reduction == "cuped"
                        else usable_rows
                    )
                except CodedError as exc:
                    if exc.code not in SCALAR_COVARIATE_REFUSALS:
                        raise
                    for arm in sorted(n_by_group.keys() - {design.control_group}):
                        for estimand in ("itt", "late"):
                            if estimand in estimands:
                                _append_lift_cell_failure(
                                    results=lift_estimates,
                                    computations=None,
                                    metric=metric,
                                    group_id=arm,
                                    method=method,
                                    method_roles=roles_by_metric[metric.name],
                                    alternative="two-sided",
                                    code=exc.code,
                                    context=exc.context,
                                    prior=configs[metric.name].prior,
                                    estimand=estimand,
                                )
                    continue
                computation = estimate_encouragement(
                    [metric],
                    [row for row in method_rows if str(row["group_id"]) not in empty_groups],
                    design,
                    estimands=estimands,
                    methods=[method],
                    prior=configs[metric.name].prior,
                    alpha=alpha,
                    method_roles=roles_by_metric[metric.name],
                )
                for estimate in computation.results:
                    if estimate.estimand == "compliance":
                        key = (estimate.group_id, estimate.method, estimate.value_scale)
                        if key in diagnostic_keys:
                            continue
                        diagnostic_keys.add(key)
                    lift_estimates.append(estimate)
            for lift_estimate in lift_estimates:
                if lift_estimate.estimand == "compliance":
                    compliance_scales_by_metric[metric.name].add(lift_estimate.value_scale)
                if lift_estimate.metric != metric.name:
                    lift_estimate = lift_estimate.model_copy(update={"metric": metric.name})
                n_treat = n_by_group.get(str(lift_estimate.group_id))
                results.append(
                    _breakout_estimate_row(
                        lift_estimate,
                        dimension=dimension,
                        dimension_value=value,
                        source=source_name,
                        n_treat=n_treat,
                        n_control=n_control_arm,
                        low_reliability=(
                            (n_treat is not None and n_treat < DEFAULT_RELIABILITY_FLOOR)
                            or (
                                n_control_arm is not None
                                and n_control_arm < DEFAULT_RELIABILITY_FLOOR
                            )
                        ),
                    )
                )
    for metric_name, value, group_id, n_treat, n_control in missing_cells:
        for estimand, value_scale, method in _encouragement_cell_identities(
            estimands,
            methods_by_metric[metric_name],
            tuple(sorted(compliance_scales_by_metric[metric_name])),
        ):
            results.append(
                _missing_encouragement_observation(
                    metric=metric_name,
                    group_id=group_id,
                    estimand=estimand,
                    value_scale=value_scale,
                    method=method,
                    method_role=(
                        "decision"
                        if estimand == "compliance"
                        else roles_by_metric[metric_name][method]
                    ),
                    dimension=dimension,
                    value=value,
                    source=source_name,
                    n_treat=n_treat,
                    n_control=n_control,
                )
            )
    return [row.model_copy(update={"policy_name": "compiled_plan"}) for row in results]


def _scoped_breakout_results(
    rows: Sequence[BreakoutEstimate],
    *,
    src: EvidenceSource,
    analysis_population: Literal["assigned", "triggered"],
    plan: Any,
    metrics: Sequence[Metric],
    resolved_configs: Any,
    scope_request: Mapping[str, Any],
    design: Any,
    dimension: str,
    source_name: str | None,
    sequential_snapshot: Any = None,
    source_components: Sequence[Mapping[str, Any]] = (),
) -> BreakoutEstimates:
    route = day_axis_route(src)
    from increment.readouts._multiplicity_scope import scoped_collection

    population_rows = [
        row.model_copy(update={"analysis_population": analysis_population}) for row in rows
    ]
    return scoped_collection(
        population_rows,
        BreakoutEstimates,
        plan,
        resolved_configs,
        scope_request,
        metrics=metrics,
        route=route,
        view="breakout",
        design=design,
        dimension=dimension,
        source=source_name,
        sequential_snapshot=sequential_snapshot,
        source_components=source_components,
    )


def _segment_unit_frames(frame: Any, column: str) -> dict[str, Any]:
    """Split per-unit evidence by its segment label, retaining every declared covariate.

    Labels follow the moments wire: string identity, lower-cased booleans and
    ``"__null__"`` for a missing value. The segment column itself is dropped
    so a cell fits only its declared covariates.
    """
    import narwhals as nw

    units = nw.from_native(frame, eager_only=True)
    label = nw.col(column).cast(nw.String)
    if isinstance(units.schema[column], nw.Boolean):
        label = nw.when(nw.col(column)).then(nw.lit("true")).otherwise(nw.lit("false"))
    label = nw.when(nw.col(column).is_null()).then(nw.lit("__null__")).otherwise(label)
    labelled = units.with_columns(label.alias(column))
    return {
        value: labelled.filter(nw.col(column) == value).drop(column).to_native()
        for value in labelled.get_column(column).unique().to_list()
    }


def _load_cuped_breakout_frames(
    src: EvidenceSource,
    configs: Sequence[Any],
    methods_by_metric: Mapping[str, list[Method]],
    dimension: str,
    *,
    population: Literal["assigned", "triggered"],
    source_name: str | None,
    local_fit_metrics: frozenset[str] | None = None,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Sequence[Any]],
    dict[str, dict[str, Literal["decision", "sensitivity"]]],
    list[dict[str, Any]],
]:
    """Retain per-segment fit evidence for mixed-CUPED metrics.

    Local sources without a relational CUPED route retain their historical
    frame path; warehouse sources fit each segment through their scoped
    relation and transfer only compact evidence.
    """
    from increment.readouts._source_digest import component

    frames: dict[str, Mapping[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    covariates = {config.metric.name: config.covariates for config in configs}
    roles = {
        config.metric.name: _runtime_method_roles(methods_by_metric[config.metric.name])
        for config in configs
    }
    import narwhals as nw

    from increment._evidence_types import unit_frame_covariate_columns
    from increment._source_types import breakout_dimension_column
    from increment.readouts._metric_rows import _frame_digest_and_arms

    for config in configs:
        name = config.metric.name
        if any(
            method.variance_reduction == "cuped_multicovariate"
            for method in methods_by_metric[name]
        ):
            if local_fit_metrics is not None:
                if name not in local_fit_metrics:
                    continue
            elif retains_cuped_relation(src):
                continue
            names = tuple(item.name for item in config.covariates)
            column = breakout_dimension_column(names)
            loaded = load_breakout_evidence(
                src,
                BreakoutUnitRowsRequest(
                    metric=config.metric,
                    breakout=BreakoutIdentity(dimension, source_name),
                    population=population,
                    covariates=names,
                ),
            )
            frame = loaded.frame
            if loaded.unit_frame_has_dimension:
                requested = tuple(dict.fromkeys((dimension, *names)))
                frame = (
                    nw.from_native(frame, eager_only=True)
                    .with_columns(
                        nw.col(unit_frame_covariate_columns(requested)[dimension]).alias(column)
                    )
                    .to_native()
                )
            _arms, sha256, count = _frame_digest_and_arms(nw.from_native(frame, eager_only=True))
            evidence.append(
                {
                    **component(
                        kind="unit_evidence_rows",
                        metric=name,
                        population=population,
                        sha256=sha256,
                        source=source_name,
                        dimension=dimension,
                    ),
                    "count": count,
                }
            )
            frames[name] = _segment_unit_frames(frame, column)
    return frames, covariates, roles, evidence


def _validated_breakout_request(
    src: EvidenceSource,
    dimension: str,
    *,
    decision_method: Method | _Unset,
    sensitivity_methods: Sequence[Method] | _Unset,
    prior: Prior | None | _Unset,
    metrics: Sequence[str] | None,
    correction: Correction | None,
    q: float | None,
    estimands: Sequence[str] | None,
) -> tuple[ReadoutRequest, Correction, float]:
    design = _require_design(src, "breakout")
    plan = src.context.plan
    policy = plan.view_policies.for_view(
        "breakout",
        mechanism=design.mechanism if design is not None else None,
        segmented=True,
    )
    if correction is None:
        correction = normalize_display_correction(policy.correction)
    if q is None:
        q = policy.q if policy.q is not None else plan.q
    resolved_correction: Correction = "none" if correction is None else correction
    selected = select_metrics(
        cast("Sequence[Metric]", src.context.metrics), metrics, caller="breakout"
    )
    resolved_configs = overlay_configs(
        selected,
        src.context.configs,
        methods=None,
        prior=None,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior_override=prior,
    )
    request = ReadoutRequest.from_source(
        src,
        metrics=selected,
        configs=resolved_configs,
        view="breakout",
        grain="total",
        by=(dimension,),
        dimension=dimension,
        estimands=estimands,
        correction=resolved_correction,
        q=q,
    )
    validate_request(request)
    return request, resolved_correction, q


def _breakout_impl(
    src: EvidenceSource,
    dimension: str,
    *,
    source_name: str | None = None,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    correction: Correction | None = None,
    q: float | None = None,
    estimands: Sequence[str] | None = None,
    analysis_population: Literal["assigned", "triggered"] = "assigned",
) -> BreakoutEstimates:
    """Relative lift estimated separately for every distinct value of dimension.

    Design and alpha/inference are all read off *src* (``src.design``,
    ``src.plan.alpha``, ``src.plan.inference``) rather than passed in -
    a `MomentSource` owns all three as construction state. `src.plan.alpha`/
    `.inference` are populated regardless of whether an `AnalysisPlan` was
    ever declared (`src.plan.declared`): an undeclared plan resolves to
    alpha=0.05 and inference=None.

    Delegates the randomized per-segment split to run_breakout, supplying
    each metric's moments broken out by dimension; a segment with no
    usable control arm comes back as an excluded row. Every row is
    stamped `role="exploratory"` - breakout has no plan-declared
    primary/secondary/guardrail role concept, unlike run().

    Under Encouragement, partitions moments the same way and runs
    estimate_encouragement once per (segment, metric) pair, at the
    plan's alpha directly (no split, no BH/FCR family machinery - only
    role stamping). A control-free segment warns and is skipped;
    siblings remain estimable, and a weak first stage only suppresses
    that segment's late row. correction (other than "none") and a
    declared inference are refused there.

    Refused under Observational: a per-segment contrast splits the
    adjustment set's confounding structure too, which is not identified -
    raises rather than emitting a confounded estimate.

    correction="bonferroni" divides alpha by the number of distinct
    dimension values present in each metric's own moments, computed per
    metric. correction=None (the default) resolves to "bh" under a
    randomized design, or "none" under Encouragement, which does not
    support correction - a caller EXPLICITLY passing a non-"none"
    correction under Encouragement is refused. correction="bh"
    runs run_breakout's flat BH/e-BH family across every (metric, arm,
    segment) cell this call produces in ONE run_breakout call (all
    selected metrics together, not one call per metric) at *q*,
    stamping `discovery` and re-estimating selected fixed-horizon intervals
    at the FCR level. Exact registered sequential breakouts select over their
    retained roster and reinvert selected intervals at the capped FCR
    allocation from the same stopped checkpoints. Corrected asymptotic-mean
    families instead use fixed-roster Bonferroni familywise inference. A declared informative
    prior changes posterior fields only; BH selection still consumes the row's sampling evidence.


    Under a fixed-horizon plan, a metric with a non-inferiority margin --
    declared on the Metric (margin/margin_abs) or bound on the plan
    (ExperimentMetric.margin/margin_abs) -- is refused with
    ``readout.margin.breakout``: fixed-horizon segment rows build no shifted
    null, so a margin would change what the same metric's stat_sig means
    between views. The whole-window guardrail read is run(). A registered
    sequential breakout instead tests each segment cell against its
    registered null, which must equal the compiled margin null.

    metrics narrows the reported names; an unknown one raises before any
    moments run.
    """
    request, correction, q = _validated_breakout_request(
        src,
        dimension,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        metrics=metrics,
        correction=correction,
        q=q,
        estimands=estimands,
    )
    design = _require_design(src, "breakout")
    plan = src.context.plan
    alpha = plan.alpha
    inference = _sequential_inference(plan)
    selected = request.metrics
    resolved_configs = request.configs
    configs = {config.metric.name: config for config in resolved_configs}
    from increment._evidence_types import BreakoutIdentity
    from increment._source_identity import source_identity
    from increment.kernel.contracts import Scope
    from increment.readouts._design_scope import _config_snapshot

    operation_scope = Scope(
        view="breakout",
        grain="total",
        population=analysis_population,
        segment=BreakoutIdentity(dimension, source_name),
    )

    segment = BreakoutIdentity(dimension, source_name)
    operation_scope = Scope(
        view="breakout",
        grain="total",
        population=analysis_population,
        segment=segment,
    )

    methods_by_metric = {
        config.metric.name: _runtime_methods(config, design) for config in resolved_configs
    }
    unit_frames_by_metric, covariates_by_metric, roles_by_metric, unit_evidence = (
        _load_cuped_breakout_frames(
            src,
            resolved_configs,
            methods_by_metric,
            dimension,
            population=analysis_population,
            source_name=source_name,
        )
    )
    scope_request = {
        "source_identity": source_identity(src),
        "metrics": [
            metric.model_dump(mode="json") for metric in sorted(selected, key=lambda m: m.name)
        ],
        "configs": [
            _config_snapshot(config, design)
            for config in sorted(resolved_configs, key=lambda c: c.metric.name)
        ],
        "dimension": segment.property,
        "estimands": None if estimands is None else tuple(estimands),
        "correction": correction,
        "q": q,
        "source_name": segment.source,
        "analysis_population": operation_scope.population,
    }

    def scoped(rows, *, sequential_snapshot=None):
        return _scoped_breakout_results(
            rows,
            src=src,
            analysis_population=analysis_population,
            plan=plan,
            metrics=selected,
            resolved_configs=resolved_configs,
            scope_request=scope_request,
            design=design,
            dimension=dimension,
            source_name=source_name,
            sequential_snapshot=sequential_snapshot,
            # Local fallback frames contribute their own provenance; warehouse
            # fits are bound to the source and breakout request identity.
            source_components=unit_evidence,
        )

    if isinstance(plan.inference, SEQUENTIAL_POLICIES):
        from increment._sequential_readouts import sequential_readout
        from increment.readouts._evidence import current_source_snapshot
        from increment.sequential_state import sequential_refuse

        snapshot = current_source_snapshot(src)
        rows = sequential_readout(
            src,
            metrics=selected,
            estimands=estimands,
            snapshot=snapshot,
            configs=resolved_configs,
        )
        output = []
        for row in rows:
            cp = row.require_sequential_result().checkpoint
            if len(cp.cell.segment) != 1 or cp.cell.segment[0][0] != dimension:
                sequential_refuse(
                    "source.invalid", "breakout dimension differs from retained segment roster"
                )
            output.append(
                _breakout_estimate_row(
                    row,
                    dimension=dimension,
                    dimension_value=cp.cell.segment[0][1],
                    source=source_name,
                    n_treat=cp.treatment.n,
                    n_control=cp.control.n,
                    low_reliability=False,
                    discovery=row.discovery,
                    family_axes=row.family_axes,
                    family_q=row.family_q,
                    family_threshold=row.family_threshold,
                )
            )
        return scoped(stamp_multiplicity_status(output), sequential_snapshot=snapshot)

    method_evidence = {
        metric.name: _load_breakout_metric_rows(
            src,
            metric,
            methods_by_metric[metric.name],
            dimension=dimension,
            control_group=design.control_group,
            source_name=source_name,
            population=analysis_population,
        )
        for metric in selected
    }
    if fits_uncovered_cells_locally(src):
        # Cells without a retained or warehouse fit fall back to a local fit from unit rows.
        local_fit_metrics = frozenset(
            name
            for name, evidence in method_evidence.items()
            if any(answer.origin == "unavailable" for _, answer in evidence.method_fit_answers)
        )
        if local_fit_metrics:
            local_frames, _, _, local_evidence = _load_cuped_breakout_frames(
                src,
                resolved_configs,
                methods_by_metric,
                dimension,
                population=analysis_population,
                source_name=source_name,
                local_fit_metrics=local_fit_metrics,
            )
            unit_frames_by_metric.update(local_frames)
            unit_evidence.extend(local_evidence)
    from increment.readouts._source_digest import component

    for metric_name, evidence in method_evidence.items():
        for name, sha256, count in evidence.evidence_components:
            if not name.startswith(("warehouse_qr_breakout:", "categorical_cells_breakout:")):
                continue
            unit_evidence.append(
                {
                    **component(
                        kind=name,
                        metric=metric_name,
                        population=analysis_population,
                        source=source_name,
                        dimension=dimension,
                        sha256=sha256,
                    ),
                    "count": count,
                }
            )
    if design.mechanism == "encouragement":
        requested_estimands = estimands if estimands is not None else ESTIMANDS
        unknown = set(requested_estimands) - set(ESTIMANDS)
        if unknown:
            _raise(
                "estimation.encouragement.unknown_estimand_supported",
                estimands=ESTIMANDS,
                unknown=sorted(unknown),
            )
        encouragement_rows = _encouragement_breakout_rows(
            selected,
            dimension=dimension,
            source_name=source_name,
            design=design,
            configs=configs,
            methods_by_metric=methods_by_metric,
            roles_by_metric=roles_by_metric,
            estimands=requested_estimands,
            alpha=alpha,
            method_evidence=method_evidence,
        )
        return scoped(stamp_multiplicity_status(encouragement_rows))
    results: list[BreakoutEstimate] = []
    if correction == "bh":
        # One flat family across every (metric, arm, segment) cell this
        # call produces: a single run_breakout call over every selected
        # metric together, not one call per metric (which would scope
        # the family to one metric at a time).
        combined_rows = None
        results.extend(
            run_breakout(
                combined_rows,
                selected,
                control_group=design.control_group,
                source=source_name,
                dimension=dimension,
                methods=None,
                prior=None,
                _prior_by_metric={name: config.prior for name, config in configs.items()},
                _method_evidence_by_metric=method_evidence,
                alpha=alpha,
                correction=correction,
                q=q,
                inference=inference,
                method_roles_by_metric=roles_by_metric,
                methods_by_metric=methods_by_metric,
                policy_name="compiled_plan",
                unit_frames_by_metric=unit_frames_by_metric,
                covariates_by_metric=covariates_by_metric,
            )
        )
    else:
        for metric in selected:
            rows = None
            metric_methods = methods_by_metric[metric.name]
            results.extend(
                run_breakout(
                    rows,
                    [metric],
                    control_group=design.control_group,
                    source=source_name,
                    dimension=dimension,
                    methods=metric_methods,
                    prior=configs[metric.name].prior,
                    _method_evidence_by_metric=method_evidence,
                    alpha=alpha,
                    correction=correction,
                    inference=inference,
                    method_roles=roles_by_metric[metric.name],
                    policy_name="compiled_plan",
                    unit_frames_by_metric=unit_frames_by_metric,
                    covariates_by_metric=covariates_by_metric,
                )
            )
    return scoped(stamp_multiplicity_status(results))


def breakout(
    src: EvidenceSource,
    dimension: str,
    *,
    source_name: str | None = None,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    correction: Correction | None = None,
    q: float | None = None,
    estimands: Sequence[str] | None = None,
    analysis_population: Literal["assigned", "triggered"] = "assigned",
) -> BreakoutEstimates:
    """Run breakout on one pinned custom-provider snapshot."""
    if not is_custom_source(src):
        return _breakout_impl(
            src,
            dimension,
            source_name=source_name,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            metrics=metrics,
            correction=correction,
            q=q,
            estimands=estimands,
            analysis_population=analysis_population,
        )

    validated, _, _ = _validated_breakout_request(
        src,
        dimension,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        metrics=metrics,
        correction=correction,
        q=q,
        estimands=estimands,
    )
    request = EvidenceScope(
        metrics=validated.metrics,
        population=analysis_population,
        breakouts=(BreakoutIdentity(dimension, source_name),),
        configs=validated.configs,
        snapshot=True,
        include_breakouts=True,
    )
    with evidence_scope(src, request) as pinned:
        return _breakout_impl(
            pinned,
            dimension,
            source_name=source_name,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            metrics=metrics,
            correction=correction,
            q=q,
            estimands=estimands,
            analysis_population=analysis_population,
        )
