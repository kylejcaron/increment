from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from increment._analysis_config import (
    UNSET,
    _Unset,
    normalize_display_correction,
    overlay_configs,
    select_metrics,
)
from increment._literals import Correction
from increment._readout_request import ReadoutRequest, validate_request
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
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment.estimation.inference import Prior
    from increment.semantics.models import Metric


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
    from increment._day_axis import _day_axis_source_route
    from increment.readouts._multiplicity_scope import scoped_collection
    from increment.readouts._run import _config_snapshot

    scope_request = {
        "metrics": [metric.model_dump(mode="json") for metric in selected],
        "configs": [_config_snapshot(config) for config in resolved_configs],
        "dimension": dimension,
        "estimands": None if estimands is None else tuple(estimands),
        "correction": correction,
        "q": q,
        "source_name": source_name,
    }

    def scoped(rows):
        return scoped_collection(
            rows,
            BreakoutEstimates,
            plan,
            resolved_configs,
            scope_request,
            route=_day_axis_source_route(src),
            view="breakout",
            design=design,
            dimension=dimension,
            source=source_name,
        )

    if isinstance(plan.inference, SEQUENTIAL_POLICIES):
        from increment._sequential_readouts import sequential_readout
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
        return scoped(stamp_multiplicity_status(output))

    methods_by_metric = {
        config.metric.name: _runtime_methods(config, design) for config in resolved_configs
    }
    roles_by_metric = {
        name: _runtime_method_roles(methods) for name, methods in methods_by_metric.items()
    }
    if design.mechanism == "encouragement":
        requested_estimands = estimands if estimands is not None else ESTIMANDS
        unknown = set(requested_estimands) - set(ESTIMANDS)
        if unknown:
            _raise(
                "estimation.encouragement.unknown_estimand_supported",
                estimands=ESTIMANDS,
                unknown=sorted(unknown),
            )
        for metric_methods in methods_by_metric.values():
            _validate_methods(metric_methods)
        enc_results: list[BreakoutEstimate] = []
        for metric in selected:
            metric_methods = methods_by_metric[metric.name]
            rows = cast(
                "list[Mapping[str, Any]]", src.moments(metric, grain="total", by=[dimension])
            )
            segments: dict[str, list[Mapping[str, Any]]] = {}
            for row in rows:
                segments.setdefault(str(row[dimension]), []).append(row)
            for value in sorted(segments):
                if not any(row.get("group_id") == design.control_group for row in segments[value]):
                    _warn(
                        "readouts.breakout.segment_no_control_arm",
                        dimension=dimension,
                        value=value,
                        control_group=design.control_group,
                        stacklevel=2,
                    )
                    continue
                n_by_group = {
                    str(row.get("group_id")): float(row["n"])
                    for row in segments[value]
                    if row.get("n") is not None
                }
                n_control_arm = n_by_group.get(design.control_group)
                for lift_estimate in estimate_encouragement(
                    [metric],
                    segments[value],
                    design,
                    estimands=requested_estimands,
                    methods=metric_methods,
                    prior=configs[metric.name].prior,
                    alpha=alpha,
                    method_roles=roles_by_metric[metric.name],
                ).results:
                    if lift_estimate.metric != metric.name:
                        lift_estimate = lift_estimate.model_copy(update={"metric": metric.name})
                    n_treat = n_by_group.get(str(lift_estimate.group_id))
                    enc_results.append(
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
        return scoped(
            stamp_multiplicity_status(
                [row.model_copy(update={"policy_name": "compiled_plan"}) for row in enc_results]
            )
        )
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
                )
            )
    return scoped(stamp_multiplicity_status(results))
