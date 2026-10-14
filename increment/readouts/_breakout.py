from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from increment._analysis_config import (
    UNSET,
    _Unset,
    normalize_display_correction,
    overlay_configs,
    select_metrics,
)
from increment._literals import Correction
from increment._readout_request import ReadoutRequest, validate_request
from increment._source_types import classify_source
from increment.breakout.estimates import (
    DEFAULT_RELIABILITY_FLOOR,
    BreakoutEstimate,
    BreakoutEstimates,
    _breakout_estimate_row,
    run_breakout,
)
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
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    src: MomentSource,
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
        rows = cast("list[Mapping[str, Any]]", src.moments(metric, grain="total", by=[dimension]))
        segments: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            segments.setdefault(str(row[dimension]), []).append(row)
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
            for lift_estimate in estimate_encouragement(
                [metric],
                usable_rows,
                design,
                estimands=estimands,
                methods=metric_methods,
                prior=configs[metric.name].prior,
                alpha=alpha,
                method_roles=roles_by_metric[metric.name],
            ).results:
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
    src: MomentSource,
    analysis_population: Literal["assigned", "triggered"],
    plan: Any,
    resolved_configs: Any,
    scope_request: Mapping[str, Any],
    design: Any,
    dimension: str,
    source_name: str | None,
    sequential_snapshot: Any = None,
    source_components: Sequence[Mapping[str, Any]] = (),
) -> BreakoutEstimates:
    route = classify_source(src)
    if route == "panel":
        route = "moments"
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
    src: MomentSource,
    configs: Sequence[Any],
    methods_by_metric: Mapping[str, list[Method]],
    dimension: str,
    *,
    population: str,
    source_name: str | None,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Sequence[Any]],
    dict[str, dict[str, Literal["decision", "sensitivity"]]],
    list[dict[str, Any]],
]:
    """Per-unit evidence for every mixed-CUPED metric, split by segment.

    Also returns one provenance component per acquired frame (its canonical
    digest and row count), so the readout's source identity covers the
    per-unit outcomes, covariate values and segment labels the cells were
    fitted on, as the headline route does.
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

    from increment._source_types import breakout_dimension_column, unit_frame_covariate_columns
    from increment.readouts._metric_rows import _frame_digest_and_arms
    from increment.sources import BreakoutMomentsSource

    for config in configs:
        name = config.metric.name
        if any(method.variance_reduction == "cuped_linear" for method in methods_by_metric[name]):
            names = tuple(item.name for item in config.covariates)
            column = breakout_dimension_column(names)
            if isinstance(src, BreakoutMomentsSource):
                frame = src.unit_frame(config.metric, covariates=names)
            else:
                requested = tuple(dict.fromkeys((dimension, *names)))
                frame = (
                    nw.from_native(
                        src.unit_frame(config.metric, covariates=requested),
                        eager_only=True,
                    )
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


def breakout(
    src: MomentSource,
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
        # Omitted q always inherits the plan's, whether or not correction
        # was passed explicitly -- an explicit correction with no q is
        # NOT the same request as an explicit correction AND q.
        q = policy.q if policy.q is not None else plan.q
    correction = correction or "none"
    alpha = plan.alpha
    inference = _sequential_inference(plan)
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
    configs = {config.metric.name: config for config in resolved_configs}
    request = ReadoutRequest.from_source(
        src,
        metrics=selected,
        configs=resolved_configs,
        view="breakout",
        grain="total",
        by=(dimension,),
        dimension=dimension,
        estimands=estimands,
        correction=correction,
        q=q,
    )
    validate_request(request)
    from increment._source_identity import source_identity
    from increment.readouts._design_scope import _config_snapshot

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
        "dimension": dimension,
        "estimands": None if estimands is None else tuple(estimands),
        "correction": correction,
        "q": q,
        "source_name": source_name,
        "analysis_population": analysis_population,
    }

    def scoped(rows, *, sequential_snapshot=None):
        return _scoped_breakout_results(
            rows,
            src=src,
            analysis_population=analysis_population,
            plan=plan,
            resolved_configs=resolved_configs,
            scope_request=scope_request,
            design=design,
            dimension=dimension,
            source_name=source_name,
            sequential_snapshot=sequential_snapshot,
            # Mixed-CUPED cells are fitted on per-unit evidence whose float
            # values the row digest excludes; its own digest is source provenance.
            source_components=unit_evidence,
        )

    if isinstance(plan.inference, SEQUENTIAL_POLICIES):
        from increment._sequential_readouts import sequential_readout
        from increment.sequential_source import source_snapshot
        from increment.sequential_state import sequential_refuse

        rows = sequential_readout(src, metrics=selected, estimands=estimands)
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
        return scoped(stamp_multiplicity_status(output), sequential_snapshot=source_snapshot(src))

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
            src,
            selected,
            dimension=dimension,
            source_name=source_name,
            design=design,
            configs=configs,
            methods_by_metric=methods_by_metric,
            roles_by_metric=roles_by_metric,
            estimands=requested_estimands,
            alpha=alpha,
        )
        return scoped(stamp_multiplicity_status(encouragement_rows))
    results: list[BreakoutEstimate] = []
    if correction == "bh":
        # One flat family across every (metric, arm, segment) cell this
        # call produces: a single run_breakout call over every selected
        # metric together, not one call per metric (which would scope
        # the family to one metric at a time).
        combined_rows = [
            row
            for metric in selected
            for row in cast(
                "list[Mapping[str, Any]]", src.moments(metric, grain="total", by=[dimension])
            )
        ]
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
            rows = cast(
                "list[Mapping[str, Any]]", src.moments(metric, grain="total", by=[dimension])
            )
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
