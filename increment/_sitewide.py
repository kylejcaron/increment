"""Analysis.sitewide behind one SitewideReadouts object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from increment._evidence_types import CountsRequest
from increment._source_operations import SitewideEvidenceOperation
from increment._source_types import declared_evidence
from increment._whole_window import (
    _UNKNOWN_METRIC,
    _is_artifact_source,
    _require_analysis_operation,
)
from increment.errors import CapabilityError, InvalidRequestError, RefusalSpec
from increment.errors import refuse as _refuse
from increment.estimation.sitewide import (
    AllocationBalance,
    SitewideContrast,
    SitewideRatioContrast,
    _validate_alpha,
    sitewide_impact,
    sitewide_impact_ratio,
)
from increment.evidence import evidence_provider
from increment.query.builders import check_cluster_size_balance, validate_site_volume_metric
from increment.semantics.models import Metric, RatioMetric

if TYPE_CHECKING:
    from increment.estimation.sitewide import SitewideImpact, SitewideRatioImpact
    from increment.query.native_contract import SitewideEvidence
    from increment.semantics.models import Experiment
    from increment.sources import _BuiltinMomentSource

_SITEWIDE_WINSORIZED_METRIC = RefusalSpec(
    "facade.analysis.sitewide_winsorized_metric",
    CapabilityError,
    template="sitewide() does not support winsorized metric {metric!r}; whole-site "
    "outcomes cannot reuse the per-unit transformed experiment metric",
)
_SITEWIDE_CLUSTER_METRIC_TYPE = RefusalSpec(
    "facade.analysis.sitewide_cluster_metric_type",
    CapabilityError,
    template="metric '{metric}' (type '{metric_type}') cannot be served by sitewide() "
    "with a declared cluster ('{cluster}') -- it does not decompose over the "
    "cluster-grain moments the clustered sitewide collapse reads.",
)
_SITEWIDE_CLUSTER_COUNTS_MISSING = RefusalSpec(
    "facade.analysis.sitewide_cluster_counts_missing",
    CapabilityError,
    template="sitewide() cluster checks need native evidence with cluster-grain "
    "assignment counts; this source's evidence for cluster '{cluster}' has none",
)
_SITEWIDE_RATIO_DENOMINATOR_MISSING = RefusalSpec(
    "facade.analysis.sitewide_ratio_denominator_missing",
    CapabilityError,
    template="sitewide() ratio evidence for metric {metric!r} is missing its denominator total",
)
_SITEWIDE_CONTROL_MISSING = RefusalSpec(
    "facade.analysis.sitewide_control_missing",
    InvalidRequestError,
    template="control_group {control_group!r} not found in the arms of metric {metric!r}. Available groups: {available!r}",
)

_SITEWIDE_NO_TREATMENT_ARM = RefusalSpec(
    "facade.analysis.sitewide_no_treatment_arm",
    InvalidRequestError,
    template="no treatment arm is enrolled for metric {metric!r} -- only control_group {control_group!r} has any units",
)

_SITEWIDE_ARM_REQUIRED = RefusalSpec(
    "facade.analysis.sitewide_arm_required",
    InvalidRequestError,
    lambda *, metric, arms: (
        f"metric {metric!r} has {len(arms)} enrolled non-control arms ({arms!r}) -- "
        f"pass arm='<group_id>' to say which one's ship-to-all impact you want. Every "
        f"other enrolled arm is netted out of the counterfactual baseline either way."
    ),
)

_SITEWIDE_ARM_NOT_ENROLLED = RefusalSpec(
    "facade.analysis.sitewide_arm_not_enrolled",
    InvalidRequestError,
    template="arm {arm!r} is not an enrolled non-control arm for metric {metric!r} (control_group is {control_group!r}). Enrolled non-control arms: {arms!r}",
)

_SITEWIDE_METRIC_UNALIGNED = RefusalSpec(
    "facade.analysis.sitewide_metric_unaligned",
    CapabilityError,
    template="sitewide() cannot align metric {metric!r} (type={metric_type!r}, "
    "window_days={window_days!r}, aggregation={aggregation!r}) with the calendar-window "
    "site total. Route: {route}",
)

_TRIGGERED_SITEWIDE_END_MISSING = RefusalSpec(
    "facade.analysis.sitewide_triggered_end_missing",
    CapabilityError,
    template="triggered sitewide requires a declared experiment end; provide an end aligned "
    "to the calendar window or use assigned sitewide.",
)

_TRIGGERED_SITEWIDE_CLUSTER = RefusalSpec(
    "facade.analysis.sitewide_triggered_cluster_unsupported",
    CapabilityError,
    template="triggered sitewide does not support cluster-randomized experiment {cluster!r}; use unit-randomized definitions.",
)
_TRIGGERED_SITEWIDE_SEQUENTIAL = RefusalSpec(
    "facade.analysis.sitewide_triggered_sequential_unsupported",
    CapabilityError,
    template="triggered sitewide is fixed-horizon and cannot use a registered sequential plan; use a fixed-horizon plan.",
)
_SITEWIDE_REQUIRES_RANDOMIZED = RefusalSpec(
    "facade.analysis.sitewide_requires_randomized",
    CapabilityError,
    template="sitewide() requires a randomized identification design; observational and encouragement designs are unsupported. Use run() for their estimands.",
)
# Retention and quantile are refused by validate_site_volume_metric.
_CLUSTER_COMPATIBLE_METRIC_TYPES = ("mean", "conversion", "ratio")


@dataclass(frozen=True, slots=True)
class SitewideRequest:
    metric_name: str
    arm: str | None
    alpha: float
    population: Literal["assigned", "triggered"] = "assigned"


class SitewideReadouts:
    def __init__(self, *, src: _BuiltinMomentSource, experiment: Experiment | None) -> None:
        self._src = src
        self._experiment = experiment

    def _with_provenance(
        self,
        result: SitewideImpact | SitewideRatioImpact,
        evidence: SitewideEvidence,
        population: Literal["assigned", "triggered"],
    ) -> SitewideImpact | SitewideRatioImpact:
        allocation_balance = None
        allocation_balance_unavailable_reason = None
        population_counts = evidence.population_counts or {}
        if population == "triggered":
            design = self._src.context.design
            expected = getattr(design, "allocation", None)
            if expected is None:
                allocation_balance_unavailable_reason = (
                    "integrity.allocation_scheme_missing"
                    if getattr(design, "allocation_scheme", None) is None
                    else "integrity.allocation_missing_or_nonconstant"
                )
            else:
                from increment.estimation.diagnostics import (
                    complete_srm_support,
                    sample_ratio_mismatch,
                )

                triggered = evidence_provider(self._src).counts(CountsRequest(kind="triggered"))
                grain, triggered_counts, unit_counts = (
                    cast("Literal['unit', 'cluster']", triggered.grain),
                    dict(triggered.counts),
                    dict(triggered.unit_counts),
                )
                # An assigned arm nobody in which triggered is a zero, not a missing arm.
                support = {**dict.fromkeys(evidence.assigned_counts or {}, 0), **triggered_counts}
                srm = sample_ratio_mismatch(
                    complete_srm_support(support, expected=expected),
                    expected=expected,
                    inference="fixed",
                    alpha=0.001,
                    grain=grain,
                    unit_counts=unit_counts,
                )
                allocation_balance = AllocationBalance(
                    statistic=srm.chi2_stat,
                    p_value=srm.fixed_p_value,
                    status="failed" if srm.is_srm else "passed",
                )
        assumptions = (
            (
                "membership_independent_of_treatment",
                "never_triggered_unaffected",
                "enrolled_units_exhaust_exposure",
            )
            if population == "triggered"
            else ("enrolled_units_exhaust_exposure",)
        )
        # Every enrolled arm is scaled (n_enrolled); cluster-grain populations are
        # normalized floats, so round before reporting a unit count.
        scaled_units = round(result.n_enrolled)
        return result.model_copy(
            update={
                "analysis_population": population,
                "window": evidence.window,
                "membership_cutoff": evidence.membership_cutoff,
                "snapshot_id": evidence.snapshot_id,
                "scaled_units": scaled_units,
                "unobserved_members": max(0, sum(population_counts.values()) - scaled_units),
                "multiplicity_status": "exploratory_unadjusted",
                "adjustment": "unadjusted",
                "assumptions": assumptions,
                "allocation_balance": allocation_balance,
                "allocation_balance_unavailable_reason": allocation_balance_unavailable_reason,
            }
        )

    def _validate_metric_alignment(self, metric: Metric) -> None:
        # Site volume adds each event over the calendar window; unit-level
        # outcomes must be additive totals over that same window.
        components = (
            (metric.numerator, metric.denominator) if isinstance(metric, RatioMetric) else (metric,)
        )
        windows = tuple(getattr(component, "window_days", None) for component in components)
        aggregations = tuple(getattr(component, "aggregation", None) for component in components)
        additive = all(aggregation in ("sum", "count") for aggregation in aggregations)
        if (
            metric.type == "conversion"
            or any(window is not None for window in windows)
            or not additive
        ):
            _refuse(
                _SITEWIDE_METRIC_UNALIGNED,
                metric=metric.name,
                metric_type=metric.type,
                window_days=windows if isinstance(metric, RatioMetric) else windows[0],
                aggregation=aggregations if isinstance(metric, RatioMetric) else aggregations[0],
                route="declare an unwindowed sum or count metric on the experiment",
            )

    def _validate_triggered_metric(self) -> None:
        experiment = self._experiment
        assert experiment is not None
        if experiment.cluster is not None:
            _refuse(
                _TRIGGERED_SITEWIDE_CLUSTER,
                cluster=experiment.cluster,
            )
        if experiment.plan.inference is not None:
            _refuse(_TRIGGERED_SITEWIDE_SEQUENTIAL)
        if experiment.end is None:
            _refuse(_TRIGGERED_SITEWIDE_END_MISSING)

    def run(self, req: SitewideRequest) -> SitewideImpact | SitewideRatioImpact:
        _validate_alpha(req.alpha)
        if self._experiment is not None and (
            self._experiment.resolved_design().mechanism != "randomized"
        ):
            _refuse(_SITEWIDE_REQUIRES_RANDOMIZED)
        source = self._src
        if _is_artifact_source(source):
            if "sitewide_evidence" not in declared_evidence(source).operations:
                from increment._evidence_refusals import refuse_operation

                refuse_operation(
                    source,
                    "sitewide_evidence",
                    tuple(sorted(declared_evidence(source).operations)),
                    route="sitewide() requires a published site-volume evidence extension",
                )
            src = cast("SitewideEvidenceOperation", source)
        else:
            src = _require_analysis_operation(
                source,
                "sitewide_evidence",
                SitewideEvidenceOperation,
                message=(
                    "sitewide() needs a native Analysis.from_definitions instance "
                    "-- a frame/warehouse/moments-backed analysis retains no raw "
                    "event stream to sum site-wide over (it starts from a "
                    "pre-melted unit summary or a precomputed moments cube, "
                    "never a raw fact table); build one with Analysis.from_definitions."
                ),
            )
        metrics_by_name = {metric.name: metric for metric in source.context.metrics}
        metric = metrics_by_name.get(req.metric_name)
        if metric is None:
            _refuse(_UNKNOWN_METRIC, metric=req.metric_name, declared=sorted(metrics_by_name))
        if getattr(metric, "winsorization", None) is not None:
            _refuse(_SITEWIDE_WINSORIZED_METRIC, metric=req.metric_name)
        assert self._experiment is not None
        cluster = self._experiment.cluster
        if cluster is not None and metric.type not in _CLUSTER_COMPATIBLE_METRIC_TYPES:
            _refuse(
                _SITEWIDE_CLUSTER_METRIC_TYPE,
                metric=req.metric_name,
                metric_type=metric.type,
                cluster=cluster,
            )
        validate_site_volume_metric(metric)
        self._validate_metric_alignment(metric)
        if req.population == "triggered":
            self._validate_triggered_metric()
        evidence = src.sitewide_evidence(
            metric,
            include_ratio=isinstance(metric, RatioMetric),
            population=req.population,
        )
        arms = evidence.arm_stats
        control_group = evidence.control_group
        control_arms = [item for item in arms if item.group_id == control_group]
        if not control_arms:
            _refuse(
                _SITEWIDE_CONTROL_MISSING,
                metric=req.metric_name,
                control_group=control_group,
                available=sorted({item.group_id for item in arms}),
            )
        treatment_arms = [item for item in arms if item.group_id != control_group]
        if not treatment_arms:
            _refuse(
                _SITEWIDE_NO_TREATMENT_ARM,
                metric=req.metric_name,
                control_group=control_group,
            )
        treatment_arms.sort(key=lambda item: item.group_id)
        arm_ids = [item.group_id for item in treatment_arms]
        if req.arm is None:
            if len(treatment_arms) > 1:
                _refuse(_SITEWIDE_ARM_REQUIRED, metric=req.metric_name, arms=arm_ids)
            target = treatment_arms[0]
        else:
            selected = [item for item in treatment_arms if item.group_id == req.arm]
            if not selected:
                _refuse(
                    _SITEWIDE_ARM_NOT_ENROLLED,
                    arm=req.arm,
                    metric=req.metric_name,
                    control_group=control_group,
                    arms=arm_ids,
                )
            target = selected[0]
        others = [item for item in treatment_arms if item.group_id != target.group_id]
        if cluster is not None:
            if evidence.cluster_counts is None or evidence.unit_counts is None:
                _refuse(_SITEWIDE_CLUSTER_COUNTS_MISSING, cluster=cluster)
            check_cluster_size_balance(
                evidence.cluster_counts,
                evidence.unit_counts,
                experiment_name=self._experiment.name,
                cluster=cluster,
                control_group=control_group,
                target_group=target.group_id,
            )
        if isinstance(metric, RatioMetric):
            if evidence.site_total_denominator is None:
                _refuse(_SITEWIDE_RATIO_DENOMINATOR_MISSING, metric=req.metric_name)
            ratio_contrast = (
                SitewideRatioContrast.from_clusters(
                    control_arms[0], target, others, cluster=cluster
                )
                if cluster is not None
                else SitewideRatioContrast.from_iid(control_arms[0], target, others)
            )
            result = sitewide_impact_ratio(
                ratio_contrast,
                site_total_numerator=evidence.site_total,
                site_total_denominator=evidence.site_total_denominator,
                alpha=req.alpha,
            )
            return self._with_provenance(result, evidence, req.population)
        sum_contrast = (
            SitewideContrast.from_clusters(control_arms[0], target, others, cluster=cluster)
            if cluster is not None
            else SitewideContrast.from_iid(control_arms[0], target, others)
        )
        result = sitewide_impact(
            sum_contrast,
            site_total_volume=evidence.site_total,
            alpha=req.alpha,
        )
        return self._with_provenance(result, evidence, req.population)


__all__ = ["SitewideReadouts", "SitewideRequest"]
