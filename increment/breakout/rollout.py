"""Segment rollout recommendation over a ``run_breakout`` result.

Thin model-based adapter over ``increment.estimation.rollout``'s pure
array math, living here for the same reason
:mod:`increment.breakout.heterogeneity` does: it consumes
``BreakoutEstimate``, which ``increment.estimation`` cannot import
without inverting the dependency.

``segment_rollout_recommendation`` answers "which segments clear the
cost of rolling out, and what is it honestly worth?" per grouping key,
with the cost resolved from a pre-declared ``Metric.rollout_cost``.

Relative scale only: Normal rows use ``lift.log_mean`` and
``lift.log_se**2``; point-backed exact-binomial rows use the corresponding
independent-binomial delta moments. The declared cost enters the estimator
as ``log1p(cost)``. Rows are
grouped by estimand and value scale, so an absolute-scale family
(an encouragement design's LATE rows) is priced on its own or not at all --
it never counts against a relative family's exclusions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

from increment._literals import ValueScale
from increment.breakout.estimates import (
    DESIGN_BASED_REASONS,
    BreakoutEstimate,
    BreakoutEstimates,
    EstimateList,
    ExclusionReason,
    _relative_meta_moments,
)
from increment.errors import InvalidRequestError, raiser, refusals
from increment.estimation.meta import _TAU_PRIOR_SCALE_DEFAULT
from increment.estimation.rollout import segment_rollout
from increment.semantics.models import Metric, MetricBase

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "breakout.rollout_cost_names": "rollout_cost= names metrics these estimates do not carry: {unknown} (known: {known}) -- refusing rather than silently pricing a rollout at the wrong threshold",
        "breakout.rollout_cost_rollout": "rollout_cost={{{name!r}: {value}}}: a rollout cost is a relative lift and must be finite and > -1",
        "breakout.price_rollout_metrics": "cannot price a rollout for metrics whose preferred_direction is not 'increase': {misdirected} -- the selection rule keeps segments whose lift EXCEEDS the cost, which presumes larger is better, so pricing a decrease-preferred metric would recommend the segments that moved worst. Drop it from metrics=, and from the estimates too if they carry it: pricing follows the estimates, while the declaration this refusal reads comes from metrics=, so removing it from only one of the two either leaves it priced at 0.0 or re-raises this same error",
    },
)
_raise = raiser(_REFUSALS)


class RolloutRecommendation(BaseModel):
    """One row per ``(analysis_population, metric, method, group_id,
    dimension, source, estimand, value_scale)`` grouping key: which of that
    key's segments to roll out, and an honest price on doing so.

    UNWEIGHTED, and deliberately so: every value field treats each
    usable segment as an equal contributor regardless of exposure, so
    ``policy_value`` is a plain sum over the rolled-out segments, not
    traffic-weighted. An exposure-weighted variant was measured and
    dropped - its selection-bias correction did not meet the accuracy
    bar the unweighted correction is held to.

    ``recommendation`` carries the estimator's verdict verbatim:
    ``"rollout"`` (corrected value is positive), ``"no_net_benefit"``
    (evidence cannot demonstrate positive value), or ``"refuse"`` (the
    offset guard fired - every value field is withheld).

    ``rollout_cost`` is the relative-lift break-even actually used.
    ``k``/``n_excluded_design``/``n_excluded_outcome`` count how many of
    the declared segments the decision could and could not use, so a
    reported value can never silently understate its coverage.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    method: str
    group_id: str
    analysis_population: Literal["assigned", "triggered"] = "assigned"
    dimension: str
    source: str | None
    estimand: str = "itt"  # "itt" | "compliance" | "late" (mirrors BreakoutEstimate)
    value_scale: ValueScale = "relative"  # scale of the rows' own lift values
    rollout_cost: float
    recommendation: Literal["rollout", "no_net_benefit", "refuse"]
    k: int
    n_selected: int
    n_excluded_design: int
    n_excluded_outcome: int
    estimated_offset: float
    policy_value: float | None
    policy_value_raw: float | None
    selection_bias: float | None
    selection_bias_share: float | None


class RolloutSegment(BaseModel):
    """One row per segment declared for a :class:`RolloutRecommendation` key -
    dense: every segment of a key that ran appears, usable or not.

    ``selected`` is the estimator's subset membership on a usable
    segment, ``None`` on one this call could not use - populated even
    behind a ``"refuse"`` recommendation as evidence about the
    selection rule, not a deployable decision.

    ``excluded`` is an upstream ``BreakoutEstimate.excluded``, or
    ``"zero_variance"`` for a live segment whose log-scale statistic or
    variance is unusable here - the same tag ``segment_heterogeneity``
    uses for its own per-scale drop.

    ``estimand``/``value_scale`` identify the upstream rows' own estimand
    and value scale; rows are grouped by both, so a family is priced on
    its own or not at all. A real LATE family (additive ``lift``, no
    log-scale moments) never clears the pricing gate below.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    method: str
    group_id: str
    analysis_population: Literal["assigned", "triggered"] = "assigned"
    dimension: str
    dimension_value: str
    source: str | None
    estimand: str = "itt"  # "itt" | "compliance" | "late" (mirrors BreakoutEstimate)
    value_scale: ValueScale = "relative"  # scale of the rows' own lift values
    selected: bool | None
    excluded: ExclusionReason | None


class RolloutRecommendations(EstimateList[RolloutRecommendation]):
    """``list[RolloutRecommendation]`` with ``.to_frame()`` - see
    :class:`~increment.breakout.estimates.EstimateList`."""

    _model = RolloutRecommendation


class RolloutSegments(EstimateList[RolloutSegment]):
    """``list[RolloutSegment]`` with ``.to_frame()`` - see
    :class:`~increment.breakout.estimates.EstimateList`."""

    _model = RolloutSegment


class SegmentRolloutResult(NamedTuple):
    """:func:`segment_rollout_recommendation`'s return value - two
    independent, row-aligned-by-grouping-key result sets."""

    recommendations: RolloutRecommendations
    segments: RolloutSegments


def _rollout_group_key(
    e: BreakoutEstimate,
) -> tuple[Literal["assigned", "triggered"], str, str, str, str, str | None, str, ValueScale]:
    """Group by the complete identity of one breakout population."""
    return (
        e.analysis_population,
        e.metric,
        e.method,
        e.group_id,
        e.dimension,
        e.source,
        e.estimand,
        e.value_scale,
    )


def _resolve_rollout_costs(
    estimates: BreakoutEstimates,
    metrics: Sequence[Metric] | None,
    rollout_cost: float | Mapping[str, float] | None,
) -> dict[str, float]:
    """Per-metric break-even, explicit > declared > ``0.0``.

    A mapping key naming no known metric is REFUSED rather than
    silently dropped, so a typo'd key cannot leave the metric priced at
    its declared cost.
    """
    # TotalMetric/ActiveMetric carry no MetricBase guardrail fields at all,
    # so they can never declare a rollout cost.
    declared: dict[str, float] = {
        m.name: m.rollout_cost
        for m in (metrics or ())
        if isinstance(m, MetricBase) and m.rollout_cost is not None
    }
    known: set[str] = {e.metric for e in estimates} | {m.name for m in (metrics or ())}

    explicit: dict[str, float]
    if rollout_cost is None:
        explicit = {}
    elif isinstance(rollout_cost, int | float):
        explicit = dict.fromkeys(known, float(rollout_cost))
    else:
        unknown = set(rollout_cost) - known
        if unknown:
            _raise("breakout.rollout_cost_names", known=sorted(known), unknown=sorted(unknown))
        explicit = {name: float(value) for name, value in rollout_cost.items()}

    for name, value in explicit.items():
        # Same domain as the declared Metric.rollout_cost field: a relative
        # lift must exceed -1 for log1p to be defined.
        if not math.isfinite(value) or value <= -1.0:
            _raise("breakout.rollout_cost_rollout", name=name, value=value)

    # Selection keeps segments whose lift EXCEEDS the cost, so a priced
    # metric must be increase-preferred; gated only when metrics= is given.
    directions = {
        m.name: m.preferred_direction for m in (metrics or ()) if isinstance(m, MetricBase)
    }
    misdirected = sorted(name for name in known if directions.get(name, "increase") != "increase")
    if misdirected:
        _raise("breakout.price_rollout_metrics", misdirected=misdirected)

    return {name: float(explicit.get(name, declared.get(name, 0.0))) for name in known}


def segment_rollout_recommendation(
    estimates: BreakoutEstimates,
    *,
    metrics: Sequence[Metric] | None = None,
    rollout_cost: float | Mapping[str, float] | None = None,
    tau_prior_scale: float = _TAU_PRIOR_SCALE_DEFAULT,
) -> SegmentRolloutResult:
    """Recommend which segments to roll out, and price the rollout, for
    each ``(analysis_population, metric, method, group_id, dimension, source,
    estimand, value_scale)`` grouping key found in *estimates*.

    *estimates* must come from a SINGLE ``run_breakout`` call - a
    standalone call stamps ``source=None``, so independent calls on one
    dimension would silently merge into a single selection problem.

    Cost resolution, highest precedence first: an explicit
    *rollout_cost* argument (a mapping's unknown name is REFUSED); then
    the metric's own declared ``rollout_cost``; then ``0.0``.

    Only RELATIVE moments are read; see the module docstring. A key
    needs at least 2 usable segments; fewer are silently skipped -- so an
    encouragement breakout's LATE rows (additive ``lift``, no log-scale
    moments) form their own key and emit nothing, rather than counting
    against the ITT key's exclusions. A segment with unusable statistics
    is dropped, reported ``excluded="zero_variance"``, ``selected=None``.
    Values are UNWEIGHTED by exposure - read :class:`RolloutRecommendation`
    before quoting one.
    Parameters
    ----------
    estimates : BreakoutEstimates
        A single ``run_breakout`` call's dense output.
    metrics : Sequence[Metric], optional
        Metric declarations, read only for ``rollout_cost``.
    rollout_cost : float or Mapping[str, float], optional
        Explicit break-even override(s), as relative lift.
    tau_prior_scale : float
        HalfNormal prior scale on tau, forwarded to
        :func:`~increment.estimation.rollout.segment_rollout`.
    Raises
    ------
    ValueError
        An explicit *rollout_cost* key naming no known metric, a
        non-finite / ``<= -1`` cost, or the estimator's own guards --
        including ``estimation.meta.posterior_integration_unresolved``,
        propagated unchanged when a key's tau posterior cannot be
        integrated within the numerical budget: a recommendation row has
        no field to carry that reason, and ``"refuse"`` means the offset
        guard fired, so the failure is never reported as one.
    Returns
    -------
    SegmentRolloutResult
        ``(recommendations, segments)`` - see
        :class:`RolloutRecommendation` and :class:`RolloutSegment`.
    """
    costs = _resolve_rollout_costs(estimates, metrics, rollout_cost)

    groups: dict[
        tuple[Literal["assigned", "triggered"], str, str, str, str, str | None, str, ValueScale],
        list[BreakoutEstimate],
    ] = {}
    for e in estimates:
        groups.setdefault(_rollout_group_key(e), []).append(e)

    recommendation_rows: list[RolloutRecommendation] = []
    segment_rows: list[RolloutSegment] = []

    for (
        analysis_population,
        metric,
        method,
        group_id,
        dimension,
        source,
        estimand,
        value_scale,
    ), rows in groups.items():
        n_design = sum(1 for r in rows if r.excluded in DESIGN_BASED_REASONS)
        n_outcome_upstream = sum(
            1 for r in rows if r.excluded is not None and r.excluded not in DESIGN_BASED_REASONS
        )
        live = [r for r in rows if r.excluded is None]
        excluded = [r for r in rows if r.excluded is not None]

        # Twin of segment_heterogeneity's per-scale usability gate (its
        # relative-scale branch); kept local because only that scale exists here.
        good_rows: list[BreakoutEstimate] = []
        bad_rows: list[BreakoutEstimate] = []
        est: list[float] = []
        var: list[float] = []
        for row in live:
            e_k, se_k = _relative_meta_moments(row)
            v_k = se_k**2 if se_k is not None else None
            if (
                e_k is None
                or v_k is None
                or not math.isfinite(e_k)
                or not (v_k > 0 and math.isfinite(v_k))
            ):
                bad_rows.append(row)
            else:
                good_rows.append(row)
                est.append(e_k)
                var.append(v_k)

        if len(good_rows) < 2:
            continue

        cost = costs.get(metric, 0.0)
        decision = segment_rollout(
            est, var, cost_threshold=math.log1p(cost), tau_prior_scale=tau_prior_scale
        )

        recommendation_rows.append(
            RolloutRecommendation(
                metric=metric,
                method=method,
                group_id=group_id,
                analysis_population=analysis_population,
                dimension=dimension,
                source=source,
                estimand=estimand,
                value_scale=value_scale,
                rollout_cost=cost,
                recommendation=decision.recommendation,
                k=decision.k,
                n_selected=int(decision.selected.sum()),
                n_excluded_design=n_design,
                n_excluded_outcome=n_outcome_upstream + len(bad_rows),
                estimated_offset=decision.estimated_offset,
                policy_value=decision.policy_value,
                policy_value_raw=decision.policy_value_raw,
                selection_bias=decision.selection_bias,
                selection_bias_share=decision.selection_bias_share,
            )
        )

        for row, is_selected in zip(good_rows, decision.selected, strict=True):
            segment_rows.append(
                RolloutSegment(
                    metric=metric,
                    method=method,
                    group_id=group_id,
                    analysis_population=analysis_population,
                    dimension=dimension,
                    dimension_value=row.dimension_value,
                    source=source,
                    estimand=estimand,
                    value_scale=value_scale,
                    selected=bool(is_selected),
                    excluded=None,
                )
            )
        for row in bad_rows:
            segment_rows.append(
                RolloutSegment(
                    metric=metric,
                    method=method,
                    group_id=group_id,
                    analysis_population=analysis_population,
                    dimension=dimension,
                    dimension_value=row.dimension_value,
                    source=source,
                    estimand=estimand,
                    value_scale=value_scale,
                    selected=None,
                    excluded="zero_variance",
                )
            )
        for row in excluded:
            segment_rows.append(
                RolloutSegment(
                    metric=metric,
                    method=method,
                    group_id=group_id,
                    analysis_population=analysis_population,
                    dimension=dimension,
                    dimension_value=row.dimension_value,
                    source=source,
                    estimand=estimand,
                    value_scale=value_scale,
                    selected=None,
                    excluded=row.excluded,
                )
            )

    return SegmentRolloutResult(
        RolloutRecommendations(recommendation_rows), RolloutSegments(segment_rows)
    )
