"""Conservative marginal risk-ratio sensitivity to omitted confounding.

The per-arm counterfactual bounds, averaged over measured covariates C,
give RR_obs / B**2 <= RR_true <= RR_obs * B**2, where
B = RR_EU * RR_UD / (RR_EU + RR_UD - 1). This remains marginal-valid
without homogeneous conditional effects, but is conservative.
See Ding & VanderWeele (2016), Epidemiology 27:368–377, eAppendix
Proposition A.8 and §5.2; VanderWeele & Ding (2017), Ann Intern Med
167:268–274, for the equal-strength inversion of B.
Assumptions: consistency, positivity and correct nuisance models in the
reported population, exchangeability given (C, U), and strength maxima
over strata and every cohort arm pair, in both directions. This addresses
confounding, not selection or measurement bias. The binary outcome is
read from the row's declared metric type, never asserted by the caller.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from increment.errors import (
    CodedModel,
    InvalidRequestError,
    UnsupportedRequestError,
    raiser,
    refusals,
)
from increment.estimation.results import LiftEstimate

#: Declared metric types whose per-unit outcome is 0/1, so a relative lift is a risk ratio.
BinaryMetricType = Literal["conversion", "retention"]

_REFUSALS = refusals(
    UnsupportedRequestError,
    {
        "sensitivity.confounding.metric_type_unavailable": "Confounding sensitivity requires the row's declared metric type, and metric={metric!r} carries none. Re-run the analysis with a version that records metric type.",
        "sensitivity.confounding.outcome_unsupported": "Confounding sensitivity requires a binary outcome (a conversion or retention metric); metric_type={metric_type!r} is not supported.",
        "sensitivity.confounding.estimand_unsupported": "Confounding sensitivity does not cover estimand={estimand!r} from method={method!r}. Use observational IPTW or AIPW for a marginal risk ratio; randomized contrasts do not require an omitted-confounder bound.",
        "sensitivity.confounding.scale_unsupported": "Confounding sensitivity requires value_scale='relative'; value_scale={value_scale!r} is not supported. Recompute the estimate on the relative scale.",
        "sensitivity.confounding.sequential_unsupported": "Confounding sensitivity requires fixed-horizon inference; inference={inference!r} is not supported. Use a fixed-horizon observational estimate.",
        "sensitivity.confounding.risk_ratio_unavailable": "Confounding sensitivity requires an available positive finite risk ratio; relative_unavailable_reason={relative_unavailable_reason!r}. Recompute an observational relative estimate with an available point.",
    },
)
_REFUSALS.update(
    refusals(
        InvalidRequestError,
        {
            "sensitivity.confounding.threshold_invalid": "threshold_risk_ratio={threshold_risk_ratio!r} must be finite and strictly positive.",
            "sensitivity.confounding.ratio_not_representable": "The sensitivity ratio or equal-strength value is not representable for observed_risk_ratio={observed_risk_ratio!r} and threshold_risk_ratio={threshold_risk_ratio!r}.",
        },
    )
)
_raise = raiser(_REFUSALS)

_ASSUMPTIONS = (
    "The row's identification assumptions other than exchangeability hold: consistency, "
    "positivity in the reported (possibly trimmed) population, and nuisance-model correctness.",
    "An unmeasured U exists such that measured C together with U suffices for exchangeability.",
    "Exposure–confounder risk-ratio strengths are maxima over measured-covariate strata "
    "and every pair of arms in the analysis cohort, in both directions; "
    "confounder–outcome strengths are maxima over strata and arms.",
    "The marginal squared bounding factor is conservative: a factor at or above the "
    "tipping point means confounding could explain the effect, not that it does.",
    "Only confounding is addressed, not selection or measurement bias.",
    "The binary outcome is the row's declared metric type (conversion or retention), "
    "not a property verified from the data.",
)


class ConfoundingSensitivity(CodedModel, BaseModel):
    """A deterministic conservative tipping point, not causal identification.

    Interval sensitivity inherits the row's reported interval and level,
    including any family adjustment; it does not supply a new coverage guarantee.
    Analysis population and source snapshot identity are copied from the row.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    group_id: str
    method: str
    estimand: Literal["ate", "overlap_subpopulation_ate"]
    population: str | None
    analysis_population: Literal["assigned", "triggered"]
    source_snapshot_id: str | None
    metric_type: BinaryMetricType
    bound: Literal["marginal_squared_bounding_factor"]
    observed_risk_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    interval_lower_risk_ratio: float | None = Field(allow_inf_nan=False)
    interval_upper_risk_ratio: float | None = Field(allow_inf_nan=False)
    interval_level: float | None = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    threshold_risk_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    direction: Literal["above", "below", "at"]
    bias_factor_point: float = Field(ge=1.0, allow_inf_nan=False)
    e_value_point: float = Field(ge=1.0, allow_inf_nan=False)
    bias_factor_interval: float | None = Field(ge=1.0, allow_inf_nan=False)
    e_value_interval: float | None = Field(ge=1.0, allow_inf_nan=False)
    interval_unavailable_reason: Literal["interval_limit_unavailable"] | None
    assumptions: tuple[str, ...]
    robustness_not_identification: Literal[True] = True

    def as_cell(self, *, created_at=None):
        from datetime import UTC, datetime
        from importlib import import_module

        declarations = import_module("increment.declarations")
        contracts = import_module("increment.kernel.contracts")
        ResultCell = import_module("increment.result_envelope").ResultCell

        identity = contracts.CellIdentity(
            key=declarations.CellKey(
                kind="arm",
                metric=self.metric,
                method=self.method,
                method_role="sensitivity",
                estimand=self.estimand,
                analysis_population=self.analysis_population,
                value_scale="relative",
                inference="fixed",
                alternative="two-sided",
                group_id=self.group_id,
            ),
            source_snapshot_id=self.source_snapshot_id,
        )
        return ResultCell.from_report_payload(
            row_kind="sensitivity",
            identity=identity,
            label=self.method,
            row=self,
            source_snapshot_id=self.source_snapshot_id,
            created_at=created_at or datetime.now(UTC),
        )


def e_value(bias_factor: float) -> float:
    """Invert B(m, m) for a finite bias factor at least one, not an observed RR."""
    if not math.isfinite(bias_factor) or bias_factor < 1.0:
        _raise(
            "sensitivity.confounding.ratio_not_representable",
            observed_risk_ratio=bias_factor,
            threshold_risk_ratio=1.0,
        )
    value = bias_factor + math.sqrt(bias_factor) * math.sqrt(bias_factor - 1.0)
    if not math.isfinite(value):
        _raise(
            "sensitivity.confounding.ratio_not_representable",
            observed_risk_ratio=bias_factor,
            threshold_risk_ratio=1.0,
        )
    return value


def tipping_bias_factor(observed_risk_ratio: float, threshold_risk_ratio: float) -> float:
    """Return sqrt(max(r/t, t/r)); refuse a non-representable underlying ratio."""
    if not math.isfinite(threshold_risk_ratio) or threshold_risk_ratio <= 0.0:
        _raise(
            "sensitivity.confounding.threshold_invalid",
            threshold_risk_ratio=threshold_risk_ratio,
        )
    if not math.isfinite(observed_risk_ratio) or observed_risk_ratio <= 0.0:
        _raise(
            "sensitivity.confounding.risk_ratio_unavailable",
            relative_unavailable_reason=None,
        )
    ratio = (
        observed_risk_ratio / threshold_risk_ratio
        if observed_risk_ratio >= threshold_risk_ratio
        else threshold_risk_ratio / observed_risk_ratio
    )
    if not math.isfinite(ratio):
        _raise(
            "sensitivity.confounding.ratio_not_representable",
            observed_risk_ratio=observed_risk_ratio,
            threshold_risk_ratio=threshold_risk_ratio,
        )
    return math.exp(0.5 * math.log(ratio))


def _interval_bias_factor(
    estimate: LiftEstimate,
    observed_risk_ratio: float,
    threshold_risk_ratio: float,
) -> float | None:
    relative = estimate.relative_confidence_set
    if estimate.relative_unavailable_reason is not None or (
        relative is not None and relative.geometry in ("unavailable", "empty")
    ):
        return None
    lift = estimate.lift
    assert lift is not None
    all_real = relative is not None and relative.geometry == "all_real"
    if observed_risk_ratio > threshold_risk_ratio:
        if lift.lb is None:
            return 1.0 if all_real else None
        limit = 1.0 + lift.lb
        return (
            1.0
            if limit <= threshold_risk_ratio
            else tipping_bias_factor(limit, threshold_risk_ratio)
        )
    if observed_risk_ratio < threshold_risk_ratio:
        if lift.ub is None:
            return 1.0 if all_real else None
        limit = 1.0 + lift.ub
        return (
            1.0
            if limit >= threshold_risk_ratio
            else tipping_bias_factor(limit, threshold_risk_ratio)
        )
    return 1.0 if lift.lb is not None or lift.ub is not None or all_real else None


def confounding_sensitivity(
    estimate: LiftEstimate,
    *,
    threshold_risk_ratio: float = 1.0,
) -> ConfoundingSensitivity:
    """Transform an observational relative row without altering it.

    Method names are user labels; eligibility follows the persisted estimand
    and the row's declared metric type.
    """
    if not isinstance(estimate, LiftEstimate):
        _raise(
            "sensitivity.confounding.risk_ratio_unavailable",
            relative_unavailable_reason=None,
        )
    if estimate.estimand not in ("ate", "overlap_subpopulation_ate"):
        _raise(
            "sensitivity.confounding.estimand_unsupported",
            estimand=estimate.estimand,
            method=estimate.method,
        )
    metric_type = estimate.metric_type
    if metric_type is None:
        _raise("sensitivity.confounding.metric_type_unavailable", metric=estimate.metric)
    if metric_type != "conversion" and metric_type != "retention":
        _raise("sensitivity.confounding.outcome_unsupported", metric_type=metric_type)
    if estimate.value_scale != "relative":
        _raise("sensitivity.confounding.scale_unsupported", value_scale=estimate.value_scale)
    if estimate.inference != "fixed":
        _raise("sensitivity.confounding.sequential_unsupported", inference=estimate.inference)
    if not math.isfinite(threshold_risk_ratio) or threshold_risk_ratio <= 0.0:
        _raise(
            "sensitivity.confounding.threshold_invalid",
            threshold_risk_ratio=threshold_risk_ratio,
        )
    lift = estimate.lift
    if lift is None or not math.isfinite(1.0 + lift.value) or 1.0 + lift.value <= 0.0:
        _raise(
            "sensitivity.confounding.risk_ratio_unavailable",
            relative_unavailable_reason=estimate.relative_unavailable_reason,
        )
    observed_risk_ratio = 1.0 + lift.value
    bias_factor_point = tipping_bias_factor(observed_risk_ratio, threshold_risk_ratio)
    bias_factor_interval = _interval_bias_factor(
        estimate, observed_risk_ratio, threshold_risk_ratio
    )
    direction: Literal["above", "below", "at"] = (
        "above"
        if observed_risk_ratio > threshold_risk_ratio
        else "below"
        if observed_risk_ratio < threshold_risk_ratio
        else "at"
    )
    interval_level = lift.level
    relative = estimate.relative_confidence_set
    if relative is not None and relative.geometry == "all_real":
        interval_level = math.fsum((1.0, -relative.alpha_eff))
    return ConfoundingSensitivity(
        metric=estimate.metric,
        group_id=estimate.group_id,
        method=estimate.method,
        estimand=estimate.estimand,
        population=estimate.population,
        analysis_population=estimate.analysis_population,
        source_snapshot_id=estimate.source_snapshot_id,
        metric_type=metric_type,
        bound="marginal_squared_bounding_factor",
        observed_risk_ratio=observed_risk_ratio,
        interval_lower_risk_ratio=None if lift.lb is None else 1.0 + lift.lb,
        interval_upper_risk_ratio=None if lift.ub is None else 1.0 + lift.ub,
        interval_level=interval_level,
        threshold_risk_ratio=threshold_risk_ratio,
        direction=direction,
        bias_factor_point=bias_factor_point,
        e_value_point=e_value(bias_factor_point),
        bias_factor_interval=bias_factor_interval,
        e_value_interval=None if bias_factor_interval is None else e_value(bias_factor_interval),
        interval_unavailable_reason="interval_limit_unavailable"
        if bias_factor_interval is None
        else None,
        assumptions=_ASSUMPTIONS,
    )
