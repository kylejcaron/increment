"""Analysis.planning_baseline for arm analyses: a power-solver Baseline
derived from a pilot's own control arm and declared design.

The analyzed population is the one the solvers read ``Baseline.mean``/``var``
as: under a declared trigger that is the triggered population, with
``trigger_rate`` its observed share of the assigned control arm; otherwise the
assigned population at the default rate. Outcome moments and score correlation
use that analyzed population; recruitment metadata retains assigned membership:

- mean/var from the control arm's moments (a ratio metric's linearized
  moments; a quantile metric's control-arm values instead). Under a declared
  cluster they come from the control arm's per-unit rows, since a clustered
  source's moments are cluster totals;
- ``cuped_rho`` from the runtime's own CUPED fit over the pilot's arms,
  evaluated on the control arm, when the metric declares CUPED;
- ``compliance`` from the source's design-level uptake state under an
  encouragement;
- ``cluster_icc``/``cluster_size_cv`` from contributing control clusters,
  with ICC measured on the analyzed score (a ratio's linearized score);
- ``avg_cluster_size`` from assigned control units per randomized cluster,
  and ``cluster_participation`` from contributing versus assigned clusters.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from increment._analysis_config import select_metrics
from increment._source_operations import TriggeredPopulationOperation
from increment.errors import CapabilityError, RefusalSpec, raiser, refusals
from increment.query.integrity import validate_trigger_fires_in_every_arm
from increment.sources import require_operation

if TYPE_CHECKING:
    import numpy as np

    from increment.estimation.armstats import ArmStats
    from increment.power import Baseline
    from increment.semantics.design import Encouragement
    from increment.semantics.models import Metric, QuantileMetric
    from increment.sources import MomentSource


_REFUSALS = refusals(
    CapabilityError,
    {
        "analysis.planning_baseline.quantile_source_unavailable": "metric {metric!r} has no per-unit control-arm values on this source -- quantile planning baselines need from_definitions, from_unit_day_artifact, or from_unit_summary",
        "analysis.planning_baseline.quantile_insufficient_data": "metric {metric!r}: fewer than 2 control-arm values on this source -- a quantile planning baseline needs at least 2 observations",
        "analysis.planning_baseline.design_unavailable": "metric {metric!r}: no declared design on this source -- planning_baseline needs a control group",
        "analysis.planning_baseline.covariate_source_unavailable": "metric {metric!r} declares CUPED but this source's moments do not carry the covariate",
        "analysis.planning_baseline.cuped_no_residual_variance": "metric {metric!r}: its CUPED covariate explains all of the control arm's variance, which usually means the covariate repeats the outcome -- declare a pre-period covariate, or plan without CUPED",
        "analysis.planning_baseline.cluster_icc_not_estimable": "cluster ICC is not estimable from the control arm's {control_clusters} cluster(s) over {control_units} units -- needs at least 2 clusters, and not exactly one unit per cluster; pass cluster_icc explicitly to Baseline instead",
        "analysis.planning_baseline.trigger_membership_unavailable": RefusalSpec(
            "analysis.planning_baseline.trigger_membership_unavailable",
            CapabilityError,
            template="metric {metric!r}: clustered triggered planning needs assigned and contributing cluster memberships to derive participation; republish source evidence with assigned and triggered counts",
            keys=frozenset({"control_group", "assigned_clusters", "analyzed_clusters", "route"}),
        ),
        "analysis.planning_baseline.trigger_population_inconsistent": RefusalSpec(
            "analysis.planning_baseline.trigger_population_inconsistent",
            CapabilityError,
            template="metric {metric!r}: assigned/analyzed cluster memberships imply analyzed mean size {derived_mean}, but analyzed rows report {observed_mean}; retain both memberships from the same pilot",
            keys=frozenset(
                {
                    "assigned_units",
                    "assigned_clusters",
                    "analyzed_clusters",
                    "trigger_rate",
                    "route",
                }
            ),
        ),
        "analysis.planning_baseline.trigger_metric_population_unavailable": RefusalSpec(
            "analysis.planning_baseline.trigger_metric_population_unavailable",
            CapabilityError,
            template="metric {metric!r}: this source supplies outcomes for {observed_units} of {eligible_units} trigger-eligible control units; {route}",
            keys=frozenset({"control_group", "trigger", "route"}),
        ),
        "analysis.planning_baseline.trigger_evidence_unavailable": RefusalSpec(
            "analysis.planning_baseline.trigger_evidence_unavailable",
            CapabilityError,
            lambda *, metric, trigger, missing: (
                f"metric {metric!r}: the experiment declares trigger {trigger!r}, but this "
                f"source carries no {' or '.join(missing)} evidence, so neither the triggered "
                "population's moments nor its trigger rate can be read. Republish the unit-day "
                f"artifact with TriggerPopulationRequest(trigger_name={trigger!r}) and "
                "AssignmentCountsRequest(populations=('assigned', 'triggered')), or plan from "
                "Analysis.from_definitions"
            ),
        ),
    },
)
_raise = raiser(_REFUSALS)

# Source operations the triggered population needs, and the artifact
# extension that supplies each.
_TRIGGER_EVIDENCE = (
    ("triggered_source", "trigger_population"),
    ("triggered_counts", "assignment_counts"),
)


def planning_baseline(
    source: MomentSource,
    metric: str,
    *,
    trigger: str | None,
    trigger_rates: Callable[[], dict[str, float]],
) -> Baseline:
    """The Baseline for *metric* from *source*'s control arm.

    *trigger* is the experiment's declared trigger (``None`` when it declares
    none); *trigger_rates* is ``Analysis.trigger_rates``, read only under a
    declared trigger.
    """
    context = source.context
    spec = select_metrics(
        cast("Sequence[Metric]", context.metrics), [metric], caller="planning_baseline"
    )[0]
    design = context.design
    if design is None:
        _raise("analysis.planning_baseline.design_unavailable", metric=metric)
    control_group = design.control_group
    population, trigger_rate = _analyzed_population(
        source, metric=metric, control_group=control_group, trigger=trigger, rates=trigger_rates
    )
    if spec.type == "quantile":
        quantile = _quantile_baseline(population, spec, control_group)
        return _with_trigger_rate(quantile, trigger_rate)
    if context.cluster is not None:
        baseline = _clustered_baseline(population, spec, control_group, trigger=trigger)
        if trigger is not None:
            baseline = _with_cluster_participation(
                source,
                population,
                baseline,
                metric=metric,
                control_group=control_group,
                trigger_rate=trigger_rate,
            )
    else:
        config = next(c for c in context.configs if c.metric.name == spec.name)
        wants_cuped = config.decision_method.variance_reduction == "cuped" or any(
            m.variance_reduction == "cuped" for m in config.sensitivity_methods
        )
        baseline = _unit_baseline(population, spec, control_group, wants_cuped=wants_cuped)
    if design.mechanism == "encouragement":
        compliance = _compliance(population, design)
        baseline = baseline.model_copy(update={"compliance": compliance})
    return type(baseline).model_validate(_with_trigger_rate(baseline, trigger_rate).model_dump())


def _with_cluster_participation(
    assigned: MomentSource,
    analyzed: MomentSource,
    baseline: Baseline,
    *,
    metric: str,
    control_group: str,
    trigger_rate: float,
) -> Baseline:
    """Attach assigned recruitment moments and represented-cluster evidence."""
    from increment.power.core import _CLUSTER_MEAN_ROUNDOFF, _analyzed_cluster_mean

    assigned_clusters: int | None = None
    analyzed_clusters: int | None = None
    try:
        assigned_units = assigned.unit_counts()[control_group]
        assigned_clusters = assigned.cluster_counts()[control_group]
        analyzed_clusters = analyzed.cluster_counts()[control_group]
    except (AttributeError, CapabilityError, KeyError):
        _raise(
            "analysis.planning_baseline.trigger_membership_unavailable",
            metric=metric,
            control_group=control_group,
            assigned_clusters=assigned_clusters,
            analyzed_clusters=analyzed_clusters,
            route="retain assigned and triggered unit/cluster counts in the source or artifact",
        )
    if assigned_clusters <= 0 or analyzed_clusters <= 0:
        _raise(
            "analysis.planning_baseline.trigger_membership_unavailable",
            metric=metric,
            control_group=control_group,
            assigned_clusters=assigned_clusters,
            analyzed_clusters=analyzed_clusters,
            route="supply a pilot with assigned and contributing clusters",
        )
    participation = analyzed_clusters / assigned_clusters
    assigned_mean = assigned_units / assigned_clusters
    derived_mean = _analyzed_cluster_mean(assigned_mean, trigger_rate, participation)
    roundoff = _CLUSTER_MEAN_ROUNDOFF * max(derived_mean, baseline.avg_cluster_size)
    if not math.isfinite(derived_mean) or abs(derived_mean - baseline.avg_cluster_size) > roundoff:
        _raise(
            "analysis.planning_baseline.trigger_population_inconsistent",
            metric=metric,
            derived_mean=derived_mean,
            observed_mean=baseline.avg_cluster_size,
            assigned_units=assigned_units,
            assigned_clusters=assigned_clusters,
            analyzed_clusters=analyzed_clusters,
            trigger_rate=trigger_rate,
            route="retain assigned and analyzed memberships from the same pilot",
        )
    return baseline.model_copy(
        update={
            "avg_cluster_size": assigned_mean,
            "cluster_participation": participation,
            "trigger_rate": trigger_rate,
        }
    )


def _with_trigger_rate(baseline: Baseline, trigger_rate: float) -> Baseline:
    if trigger_rate == 1.0:
        return baseline
    return baseline.model_copy(update={"trigger_rate": trigger_rate})


def _analyzed_population(
    source: MomentSource,
    *,
    metric: str,
    control_group: str,
    trigger: str | None,
    rates: Callable[[], dict[str, float]],
) -> tuple[MomentSource, float]:
    """``(population source, control arm's trigger rate)``: the assigned
    population at rate 1.0 unless a trigger is declared."""
    if trigger is None:
        return source, 1.0
    missing = tuple(
        extension
        for operation, extension in _TRIGGER_EVIDENCE
        if operation not in source.operations
    )
    if missing:
        _raise(
            "analysis.planning_baseline.trigger_evidence_unavailable",
            metric=metric,
            trigger=trigger,
            missing=missing,
        )
    observed = rates()
    validate_trigger_fires_in_every_arm(observed, trigger)
    triggered = require_operation(source, "triggered_source", TriggeredPopulationOperation)
    return triggered.triggered_source(), observed[control_group]


def _quantile_baseline(source: MomentSource, spec: QuantileMetric, control_group: str) -> Baseline:
    """The pilot's real per-unit control-arm values, or a SOURCE refusal
    when the source cannot supply them."""
    import narwhals as nw

    from increment.power.core import QuantileBaseline

    try:
        rows = source.unit_frame(spec)
    except CapabilityError as exc:
        if exc.code != "source.moments.unit_grain":
            raise
        _raise("analysis.planning_baseline.quantile_source_unavailable", metric=spec.name)
    frame = nw.from_native(rows, eager_only=True)
    control_values = frame.filter(nw.col("group_id") == control_group)["y"].to_numpy()
    if control_values.size < 2:
        _raise("analysis.planning_baseline.quantile_insufficient_data", metric=spec.name)
    return QuantileBaseline.from_control_values(spec, control_values)


def _unit_baseline(
    source: MomentSource, spec: Metric, control_group: str, *, wants_cuped: bool
) -> Baseline:
    """mean/var (and ``cuped_rho``) from the control arm's unit-grain moments."""
    from increment.estimation.armstats import SummaryStats
    from increment.estimation.engine import _df_to_arms
    from increment.power import Baseline

    rows = cast("list[Mapping[str, Any]]", source.moments(spec, include_covariate=wants_cuped))
    arms = _df_to_arms(rows)
    control = next(a for a in arms if a.group_id == control_group)
    if wants_cuped and control.ref_x is None:
        _raise("analysis.planning_baseline.covariate_source_unavailable", metric=spec.name)
    if spec.type == "ratio":
        baseline = _ratio_baseline(control)
    else:
        baseline = Baseline.from_summary(
            SummaryStats(n=control.n, mean=control.mean_y(), var=control.var_y())
        )
    if not wants_cuped:
        return baseline
    rho = _cuped_rho(arms, control, ratio=spec.type == "ratio", metric=spec.name)
    return baseline.model_copy(update={"cuped_rho": rho})


def _ratio_baseline(
    control: ArmStats,
    *,
    var_num: float | None = None,
    var_den: float | None = None,
    cov_num_den: float | None = None,
) -> Baseline:
    """The control arm's linearized ratio Baseline, optionally with adjusted
    second moments in place of the raw ones."""
    from increment.power import Baseline

    return Baseline.from_ratio(
        mean_num=control.mean_y(),
        mean_den=control.mean_den(),
        var_num=control.var_y() if var_num is None else var_num,
        var_den=control.var_den() if var_den is None else var_den,
        cov_num_den=control.cov_yden() if cov_num_den is None else cov_num_den,
    )


def _cuped_rho(arms: list[ArmStats], control: ArmStats, *, ratio: bool, metric: str) -> float:
    """``sqrt(1 - adjusted/raw)`` for the control arm's variance under the
    runtime's own CUPED fit over the pilot's arms: exactly ``|corr(Y, X)|``
    when both arms share one slope. A fit that would not reduce the control
    arm's variance plans no reduction, since ``cuped_rho`` cannot express an
    increase. A covariate that reproduces the outcome up to scale drives the
    adjusted variance to (floating-point) zero rather than exactly negative,
    so this refuses whenever the adjusted variance is within
    ``variance_slack`` of zero relative to the terms that cancel in it -- the same
    magnitude-relative rounding tolerance ``clamp_negative_variance`` uses
    for a centered sum of squares -- not only when it rounds below zero."""
    from increment.estimation.armstats import variance_slack
    from increment.estimation.cuped import fit_cuped, fit_ratio_cuped
    from increment.power.core import ratio_linearized_variance

    if ratio:
        adjusted = fit_ratio_cuped(arms).at_fitted_slopes(control)
        adjusted_var = ratio_linearized_variance(
            control.mean_y(),
            control.mean_den(),
            adjusted.var_num,
            adjusted.var_den,
            adjusted.cov_num_den,
        )
        raw_var = ratio_linearized_variance(
            control.mean_y(),
            control.mean_den(),
            control.var_y(),
            control.var_den(),
            control.cov_yden(),
        )
        # Rounding scales with the terms that cancel, not with their small difference.
        r = control.mean_y() / control.mean_den()
        magnitude = (
            control.var_y() + r * r * control.var_den() + 2.0 * abs(r * control.cov_yden())
        ) / control.mean_den() ** 2
    else:
        fit = fit_cuped(arms)
        adjusted_var = fit.residual_var(control, fit.theta)
        raw_var = magnitude = control.var_y()
    if adjusted_var <= variance_slack(magnitude, control.n):
        _raise("analysis.planning_baseline.cuped_no_residual_variance", metric=metric)
    return math.sqrt(max(0.0, 1.0 - adjusted_var / raw_var))


def _clustered_baseline(
    source: MomentSource, spec: Metric, control_group: str, *, trigger: str | None
) -> Baseline:
    """Per-unit mean/var and the design-effect fields from the control arm's
    per-unit rows: a clustered source's moments are cluster totals, not units."""
    import narwhals as nw
    import numpy as np

    from increment.estimation.armstats import SummaryStats
    from increment.power import Baseline

    frame = nw.from_native(source.unit_frame(spec), eager_only=True)
    rows = frame.filter(nw.col("group_id") == control_group)
    if trigger is not None:
        eligible_units = source.unit_counts()[control_group]
        if len(rows) != eligible_units:
            _raise(
                "analysis.planning_baseline.trigger_metric_population_unavailable",
                metric=spec.name,
                control_group=control_group,
                trigger=trigger,
                eligible_units=eligible_units,
                observed_units=len(rows),
                route=(
                    "complete unfinished metric windows; for structurally undefined outcomes, "
                    "choose a metric defined for every trigger-eligible unit"
                ),
            )
    y = rows["y"].to_numpy()
    clusters = rows["cluster_id"].to_numpy()
    if spec.type == "ratio":
        den = rows["y_den"].to_numpy()
        cov = np.cov(y, den, ddof=1)
        baseline = Baseline.from_ratio(
            mean_num=float(y.mean()),
            mean_den=float(den.mean()),
            var_num=float(cov[0, 0]),
            var_den=float(cov[1, 1]),
            cov_num_den=float(cov[0, 1]),
        )
        # The ICC of the linearized score y - R*y_den, whose variance from_ratio carries.
        fields = _one_way_cluster_design(y - baseline.mean * den, clusters)
    else:
        fields = _one_way_cluster_design(y, clusters)
        baseline = Baseline.from_summary(
            SummaryStats(n=y.size, mean=float(y.mean()), var=float(y.var(ddof=1)))
        )
    return baseline.model_copy(update=fields)


def _compliance(source: MomentSource, design: Encouragement) -> float:
    """The first treatment arm's uptake rate minus control's, from the
    source's design-level compliance state -- the same state the runtime's
    compliance row reads, at unit grain under a cluster too."""
    rates = {
        arm.group_id: arm.uptake_total / arm.n_units
        for arm in source.compliance_summary(design).arms
    }
    control = rates.pop(design.control_group)
    return next(iter(rates.values())) - control


def _one_way_cluster_design(values: np.ndarray, cluster_ids: np.ndarray) -> dict[str, float]:
    """Mean cluster size, Fisher's one-way random-effects ICC with the
    unequal-group-size n0 correction, and the coefficient of variation of
    cluster sizes, from one per-cluster grouping."""
    import numpy as np

    unique, inverse, counts = np.unique(cluster_ids, return_inverse=True, return_counts=True)
    k = unique.size
    n_total = values.size
    if k < 2 or n_total <= k:
        _raise(
            "analysis.planning_baseline.cluster_icc_not_estimable",
            control_clusters=k,
            control_units=n_total,
        )
    grand_mean = values.mean()
    cluster_means = np.bincount(inverse, weights=values, minlength=k) / counts
    ssb = float(np.sum(counts * (cluster_means - grand_mean) ** 2))
    ssw = float(np.sum((values - cluster_means[inverse]) ** 2))
    msb = ssb / (k - 1)
    msw = ssw / (n_total - k)
    n0 = (n_total - np.sum(counts**2) / n_total) / (k - 1)
    icc = 0.0 if msb <= msw else (msb - msw) / (msb + (n0 - 1) * msw)
    return {
        "avg_cluster_size": n_total / k,
        "cluster_icc": max(0.0, min(icc, 0.999)),
        "cluster_size_cv": float(counts.std(ddof=0) / counts.mean()),
    }


__all__ = ["planning_baseline"]
