"""Analysis.sitewide behind one SitewideReadouts object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from increment._source_operations import SitewideEvidenceOperation
from increment._source_types import classify_source
from increment._whole_window import (
    _ANALYSIS_OPERATION,
    _UNKNOWN_METRIC,
    _require_analysis_operation,
)
from increment.errors import CapabilityError, InvalidRequestError, RefusalSpec
from increment.errors import refuse as _refuse
from increment.estimation.sitewide import (
    SitewideContrast,
    SitewideRatioContrast,
    _validate_alpha,
    sitewide_impact,
    sitewide_impact_ratio,
)
from increment.query.builders import check_cluster_size_balance, validate_site_volume_metric
from increment.semantics.models import RatioMetric

if TYPE_CHECKING:
    from increment.estimation.sitewide import SitewideImpact, SitewideRatioImpact
    from increment.semantics.models import Experiment
    from increment.sources import MomentSource

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

# Retention and quantile are refused by validate_site_volume_metric.
_CLUSTER_COMPATIBLE_METRIC_TYPES = ("mean", "conversion", "ratio")


@dataclass(frozen=True, slots=True)
class SitewideRequest:
    metric_name: str
    arm: str | None
    alpha: float


class SitewideReadouts:
    def __init__(self, *, src: MomentSource, experiment: Experiment | None) -> None:
        self._src = src
        self._experiment = experiment

    def run(self, req: SitewideRequest) -> SitewideImpact | SitewideRatioImpact:
        _validate_alpha(req.alpha)
        source = self._src
        if classify_source(source) == "artifact":
            if "sitewide_evidence" not in source.operations:
                _refuse(
                    _ANALYSIS_OPERATION,
                    message="sitewide() requires a published site-volume evidence extension",
                    operation="sitewide_evidence",
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
        evidence = src.sitewide_evidence(metric, include_ratio=isinstance(metric, RatioMetric))
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
            return sitewide_impact_ratio(
                ratio_contrast,
                site_total_numerator=evidence.site_total,
                site_total_denominator=evidence.site_total_denominator,
                alpha=req.alpha,
            )
        sum_contrast = (
            SitewideContrast.from_clusters(control_arms[0], target, others, cluster=cluster)
            if cluster is not None
            else SitewideContrast.from_iid(control_arms[0], target, others)
        )
        return sitewide_impact(
            sum_contrast,
            site_total_volume=evidence.site_total,
            alpha=req.alpha,
        )


__all__ = ["SitewideReadouts", "SitewideRequest"]
