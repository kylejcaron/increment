from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from increment._analysis_config import UNSET, _Unset
from increment._literals import ValueScale
from increment._readout_request import _raise as _raise_readout_request
from increment.breakout.estimates import reject_quantile_metrics
from increment.estimation.adjust import observational_evidence
from increment.estimation.engine import Method
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
    A non-family secondary (prior-bound, or a quantile metric under an
    AlwaysValid plan) estimates once at nominal, with no discovery verdict.

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
                return _run_prepared(
                    pinned,
                    selected,
                    configs,
                    prior=prior,
                    by=by,
                    estimands=estimands,
                    value_scale=value_scale,
                )
    return _run_prepared(
        src,
        selected,
        configs,
        prior=prior,
        by=by,
        estimands=estimands,
        value_scale=value_scale,
    )


def _run_prepared(
    src: MomentSource,
    selected: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    *,
    prior: Prior | None | _Unset,
    by: Sequence[str],
    estimands: Sequence[str] | None,
    value_scale: Mapping[str, ValueScale] | None,
) -> list[LiftEstimate]:
    """Estimate a validated request inside its source-owned snapshot lifetime."""
    design = _require_design(src, "run")
    plan = src.context.plan

    if isinstance(plan.inference, SEQUENTIAL_POLICIES):
        from increment._sequential_readouts import sequential_readout

        _refuse_segmented_registration(plan.inference)
        return sequential_readout(src, metrics=selected, estimands=estimands)
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
        return encouragement_rows(
            src=src,
            metrics=selected,
            rows_by_metric=rows_by_metric,
            configs=configs,
            design=design,
            plan=plan,
            estimands=estimands,
            cluster=cluster,
            caller="run() under an encouragement design",
        )
    if design.mechanism == "observational":
        # The one moments reduction per metric: it gates on a treatment arm, and sizes every
        # family and routing level, and every estimate and FCR re-estimate reads it.
        evidence = observational_evidence(src, selected)
        if selected:
            _refuse_if_no_treatment_arm(
                set().union(*(evidence.arms(metric) for metric in selected)),
                design.control_group,
            )
        return _estimate_observational(
            src,
            selected,
            configs,
            design,
            plan,
            value_scale=value_scale,
            call_prior=call_prior,
            evidence=evidence,
        )
    if value_scale:
        _raise("readout.value_scale_randomized_absolute")
    effective_inference: AsymptoticMean | AlwaysValid | MixedFamily | None = _sequential_inference(
        plan
    )
    advisory_seen: set[tuple[str, str]] = set()
    observed_arms: set[str] = set()
    out, refused_cells, secondary_entries = _estimate_randomized_non_secondary(
        src,
        selected,
        configs,
        design,
        plan,
        by=by,
        effective_inference=effective_inference,
        advisory_seen=advisory_seen,
        observed_arms=observed_arms,
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
    )
    out.extend(secondary_out)
    refused_cells.extend(secondary_refused)
    if selected:
        _refuse_if_no_treatment_arm(observed_arms, design.control_group)
    if not out and refused_cells:
        _raise_if_all_lift_cells_refused(refused_cells)
    order = {metric.name: i for i, metric in enumerate(selected)}
    out.sort(key=lambda row: order[row.metric])
    return out


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
