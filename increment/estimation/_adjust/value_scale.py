"""Value-scale reporting helpers for observational adjustment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from increment._literals import ValueScale
from increment.estimation.armstats import ScoreStats

_ABSOLUTE_REMEDY = (
    "The ADDITIVE effect is identified: request it with "
    "value_scale={{{name!r}: 'absolute'}} on estimate_ate / readouts.run / Analysis.run."
)


@dataclass(frozen=True, slots=True)
class ResolvedValueScale:
    point: float | None
    scores: ScoreStats
    abs_diff: float | None
    abs_se: float | None
    posterior_point: float | None = None
    posterior_scores: ScoreStats | None = None


def resolve_value_scale(
    *,
    value_scale: ValueScale,
    prior: object | None,
    raw_point: float,
    raw_scores: ScoreStats,
    mu0: float,
    mu0_se: Callable[[], float],
    lift_fn: Callable[[float], float],
    lift_scores_fn: Callable[[float], ScoreStats],
    refuse_near_zero: Callable[[float, float], None],
) -> ResolvedValueScale:
    """Keep the sampling point/scores prior-free; return the prior likelihood separately."""
    if value_scale == "absolute":
        return ResolvedValueScale(
            point=raw_point,
            scores=raw_scores,
            abs_diff=None,
            abs_se=None,
        )
    posterior_point = None
    posterior_scores = None
    if prior is not None:
        refuse_near_zero(mu0, mu0_se())
        posterior_point = float(lift_fn(mu0))
        posterior_scores = lift_scores_fn(mu0)
    return ResolvedValueScale(
        point=None,
        scores=raw_scores,
        abs_diff=raw_point,
        abs_se=raw_scores.se(),
        posterior_point=posterior_point,
        posterior_scores=posterior_scores,
    )


def _compose_absolute_note(
    note: str | None, metric_name: str, value_scale: ValueScale
) -> str | None:
    """Append the additive-units caveat to whatever caveat the contrast
    already carries (e.g. the impute-indicator note); never overwrite it.
    """
    if value_scale != "absolute":
        return note
    return "; ".join(
        filter(
            None,
            [
                note,
                "additive ATE in the metric's own units "
                f"(value_scale='absolute' requested for metric {metric_name!r})",
            ],
        )
    )
