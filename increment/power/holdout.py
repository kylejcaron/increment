"""Fixed-horizon precision for a registered persistent holdout."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from increment._holdout_look import check_design
from increment._holdout_records import (
    Count,
    Finite,
    HoldoutPrecision,
    Nonnegative,
    require_registration,
)
from increment.errors import (
    MODEL_FIELD_REFUSALS,
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    _safe_error_value,
    refuse,
)
from increment.estimation._tails import student_t_isf, two_sided_critical_value
from increment.estimation.armstats import welch_satterthwaite_df
from increment.power._noncentral_t import _scalar_power_from_nc
from increment.power._shared import _POWER_UNRESOLVED

if TYPE_CHECKING:
    from increment._holdout_records import HoldoutStudy, HoldoutWindow


_ALPHA = 0.05
_PRECISION_UNAVAILABLE = "holdout.precision_unavailable"
_PRECISION_UNREPRESENTABLE = RefusalSpec(
    "holdout.precision_unrepresentable",
    InvalidRequestError,
    template="The projected mature counts for {window_id!r} exceed the numerical range for precision.",
)


class HoldoutProjection(CodedModel, BaseModel):
    """Projected counts and outcome-scale assumptions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    projected_assignments: Count
    projected_by_freeze: Count
    variance_eligible: Nonnegative
    variance_held_out: Nonnegative
    effect: Finite


def _welch_plan(
    projection: HoldoutProjection, n_eligible: int, n_held_out: int, window_id: str
) -> tuple[float, float, float, float]:
    # Take roots before dividing so subnormal variances can still contribute to SE.
    try:
        s1 = math.sqrt(projection.variance_eligible) / math.sqrt(n_eligible)
        s0 = math.sqrt(projection.variance_held_out) / math.sqrt(n_held_out)
    except OverflowError:
        refuse(_PRECISION_UNREPRESENTABLE, window_id=window_id)
    se = math.hypot(s1, s0)
    scale = max(s1, s0)
    df = welch_satterthwaite_df(
        (s1 / scale) ** 2, float(n_eligible - 1), (s0 / scale) ** 2, float(n_held_out - 1)
    )
    if not math.isfinite(df):
        refuse(_PRECISION_UNREPRESENTABLE, window_id=window_id)
    critical = two_sided_critical_value(student_t_isf, _ALPHA, df, what="holdout interval")
    nc = projection.effect / se
    power = _scalar_power_from_nc(nc, alternative="two-sided", tail_alpha=_ALPHA / 2, dof=df)
    if math.isnan(power):
        refuse(_POWER_UNRESOLVED, nc=nc, dof=df, tail_alpha=_ALPHA / 2)
    return se, df, 2 * critical * se, power


def plan_holdout(
    study: HoldoutStudy,
    window: HoldoutWindow,
    *,
    projected_assignments: int,
    projected_by_freeze: int,
    variance_eligible: float,
    variance_held_out: float,
    effect: float,
) -> HoldoutPrecision:
    """Forecast precision conditional on the projected, completely measured cohort.

    Variances and the signed additive effect are for the entire outcome window,
    in units of the declared outcome (squared for variances). The Welch reference
    provides a plug-in interval width and noncentral-t power at projected counts,
    not the expected width or power averaged over random allocations.
    """
    require_registration(study, window)
    if check_design(study) is not None:
        raise ValueError(
            "Holdout precision requires independent assignment and a supported design."
        )
    projection = HoldoutProjection(
        projected_assignments=projected_assignments,
        projected_by_freeze=projected_by_freeze,
        variance_eligible=variance_eligible,
        variance_held_out=variance_held_out,
        effect=effect,
    )
    if projection.projected_by_freeze > projection.projected_assignments:
        refuse(
            MODEL_FIELD_REFUSALS["model.field.range"],
            model="HoldoutProjection",
            field="projected_by_freeze",
            loc=("projected_by_freeze",),
            value=_safe_error_value(projection.projected_by_freeze),
            constraint="must not exceed projected_assignments",
        )

    p0 = study.holdout_probability
    numerator, denominator = p0.as_integer_ratio()
    n_assigned_held_out = projection.projected_assignments * numerator // denominator
    n_mature_held_out = projection.projected_by_freeze * numerator // denominator
    n_mature_eligible = projection.projected_by_freeze - n_mature_held_out
    reason: Literal["arm_too_small", "zero_variance"] | None = None
    se = df = width = power = None
    if min(n_mature_eligible, n_mature_held_out) < 2:
        reason = "arm_too_small"
    elif projection.variance_held_out == projection.variance_eligible == 0:
        reason = "zero_variance"
    else:
        se, df, width, power = _welch_plan(
            projection, n_mature_eligible, n_mature_held_out, window.window_id
        )

    terms = window.terms
    return HoldoutPrecision(
        window_id=window.window_id,
        start=terms.start,
        end=terms.end,
        assignment_unit=study.assignment_unit,
        outcome_unit=terms.outcome_unit,
        p0=p0,
        projected_assignments=projection.projected_assignments,
        projected_by_freeze=projection.projected_by_freeze,
        n_assigned_held_out=n_assigned_held_out,
        n_assigned_eligible=projection.projected_assignments - n_assigned_held_out,
        n_mature_held_out=n_mature_held_out,
        n_mature_eligible=n_mature_eligible,
        variance_held_out=projection.variance_held_out,
        variance_eligible=projection.variance_eligible,
        effect=projection.effect,
        alpha=_ALPHA,
        standard_error=se,
        df=df,
        interval_width=width,
        power=power,
        reason_code=_PRECISION_UNAVAILABLE if reason is not None else None,
        reason=reason,
    )
