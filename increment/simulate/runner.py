"""End-to-end runner for ground-truth evaluation of the increment pipeline.

Chains the DGP, query-layer builders, estimation engine, and SRM diagnostic
for synthetic experiments with known ground truth.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import numpy as np
import pyarrow as pa
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from increment.errors import CodedError, InvalidRequestError, raiser, refusals
from increment.estimation.diagnostics import sample_ratio_mismatch
from increment.estimation.engine import estimate_lift
from increment.estimation.results import BinomialConfidenceSet, Estimate, LiftEstimate
from increment.semantics.models import (
    AnalysisPlan,
    ConversionMetric,
    Experiment,
    MeanMetric,
    Metric,
)
from increment.simulate.dgp import (
    _START_DATE,
    Scenario,
    SwitchbackScenario,
    simulate_raw_logs,
    simulate_switchback_panel,
)
from increment.winsor import WinsorConfidenceSet

if TYPE_CHECKING:
    from increment.breakout.estimates import BreakoutEstimate
    from increment.estimation.cate import CateResult
    from increment.estimation.contrast_results import ContrastResult
    from increment.estimation.decision_types import DecisionFailure


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "simulate.runner.cate_routing_does": "CATE routing does not support metric {metric_name!r}",
        "simulate.runner.replications_invalid": "replications must be a positive integer, got {replications!r}",
    },
)
_raise = raiser(_REFUSALS)

# terse from_unit_summary MetricSpec type per metric, for the CATE routing.
_METRIC_SPEC_TYPE: dict[str, str] = {"conversion": "conversion", "count": "mean", "revenue": "mean"}


def _validate_replications(replications: object) -> int:
    """Reject Boolean, noninteger, and nonpositive replication counts.

    Validation runs before seeding, data generation, or source loading, so
    invalid counts return the runner's coded refusal instead of an empty result
    or an uncoded lower-level exception.
    """
    if isinstance(replications, bool) or not isinstance(replications, int) or replications < 1:
        _raise("simulate.runner.replications_invalid", replications=replications)
    return cast("int", replications)


_NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]

_KEYED_COUNT_FIELDS = (
    "attempted",
    "point_estimable",
    "interval_estimable",
    "confidence_set_estimable",
    "set_only",
    "excluded",
    "failed",
)
_KEYED_REASON_FIELDS = (
    "failure_reasons",
    "exclusion_reasons",
    "interval_unavailable_reasons",
)
_POINT_AGGREGATE_FIELDS = (
    "bias",
    "bias_mcse",
    "coverage_conditional",
    "coverage_conditional_mcse",
    "coverage_unconditional",
    "coverage_unconditional_mcse",
)
_SET_AGGREGATE_FIELDS = (
    "set_coverage_conditional",
    "set_coverage_conditional_mcse",
    "set_coverage_unconditional",
    "set_coverage_unconditional_mcse",
)
# EvalResult's per-key validation (_validate_keyed_fields) covers both;
# SwitchbackEvalResult never touches the exact binomial method (see its
# own _validate_result) and so declares/validates only the point subset.
_KEYED_AGGREGATE_FIELDS = _POINT_AGGREGATE_FIELDS + _SET_AGGREGATE_FIELDS


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            key: _freeze_mapping(item) if isinstance(item, Mapping) else item
            for key, item in value.items()
        }
    )


def _plain_mapping(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_mapping(item) for key, item in value.items()}
    return value


class _FrozenResultModel(BaseModel):
    model_config = ConfigDict(frozen=True, validate_default=True)

    @field_validator("*")
    @classmethod
    def _copy_and_freeze_mappings(cls, value: Any) -> Any:
        return _freeze_mapping(value) if isinstance(value, Mapping) else value

    @field_serializer("*")
    def _serialize_mappings(self, value: Any) -> Any:
        return _plain_mapping(value)


def _invalid_result(message: str) -> PydanticCustomError:
    return PydanticCustomError(
        "simulate.runner.result_contract",
        "{message}",
        {"message": message},
    )


def _validate_aggregate_value(
    field_name: str,
    value: float | None,
    *,
    point_estimable: int,
    interval_estimable: int,
    confidence_set_estimable: int,
    attempted: int,
    location: str,
) -> None:
    if value is not None and not math.isfinite(value):
        raise _invalid_result(f"{location} must be finite or None")

    if (
        field_name
        in (
            "coverage_conditional",
            "coverage_unconditional",
            "set_coverage_conditional",
            "set_coverage_unconditional",
        )
        and value is not None
    ):
        if not 0.0 <= value <= 1.0:
            raise _invalid_result(f"{location} must be in [0, 1], or None")
    if field_name.endswith("_mcse") and value is not None and value < 0.0:
        raise _invalid_result(f"{location} must be nonnegative or None")

    if field_name == "bias":
        if point_estimable == 0 and value is not None:
            raise _invalid_result(f"{location} must be None when its count is 0")
        return

    required_count = {
        "bias_mcse": (point_estimable, 2),
        "coverage_conditional": (interval_estimable, 1),
        "coverage_conditional_mcse": (interval_estimable, 2),
        "coverage_unconditional": (attempted, 1),
        "coverage_unconditional_mcse": (attempted, 2),
        "set_coverage_conditional": (confidence_set_estimable, 1),
        "set_coverage_conditional_mcse": (confidence_set_estimable, 2),
        "set_coverage_unconditional": (attempted, 1),
        "set_coverage_unconditional_mcse": (attempted, 2),
    }.get(field_name)
    if required_count is None:
        return
    count, minimum = required_count
    if (value is not None) != (count >= minimum):
        state = "non-None" if count >= minimum else "None"
        raise _invalid_result(f"{location} must be {state} when its count is {count}")


def _validate_keyed_fields(result: Any, prefix: str = "") -> None:
    field_names = tuple(
        f"{prefix}{name}"
        for name in (*_KEYED_COUNT_FIELDS, *_KEYED_REASON_FIELDS, *_KEYED_AGGREGATE_FIELDS)
    )
    expected_keys = set(getattr(result, f"{prefix}attempted"))
    for field_name in field_names:
        actual_keys = set(getattr(result, field_name))
        if actual_keys != expected_keys:
            raise _invalid_result(
                f"{field_name} keys must match {prefix}attempted keys; "
                f"expected {sorted(expected_keys)!r}, got {sorted(actual_keys)!r}"
            )

    for key in expected_keys:
        attempted = getattr(result, f"{prefix}attempted")[key]
        point_estimable = getattr(result, f"{prefix}point_estimable")[key]
        interval_estimable = getattr(result, f"{prefix}interval_estimable")[key]
        confidence_set_estimable = getattr(result, f"{prefix}confidence_set_estimable")[key]
        set_only = getattr(result, f"{prefix}set_only")[key]
        excluded = getattr(result, f"{prefix}excluded")[key]
        failed = getattr(result, f"{prefix}failed")[key]
        if attempted != point_estimable + set_only + excluded + failed:
            raise _invalid_result(
                f"{prefix}attempted[{key!r}] must equal "
                "point_estimable + set_only + excluded + failed"
            )
        if interval_estimable > point_estimable:
            raise _invalid_result(
                f"{prefix}interval_estimable[{key!r}] must not exceed point_estimable"
            )
        if confidence_set_estimable != interval_estimable + set_only:
            raise _invalid_result(
                f"{prefix}confidence_set_estimable[{key!r}] must equal "
                "interval_estimable + set_only"
            )

        reason_totals = (
            ("failure_reasons", failed),
            ("exclusion_reasons", excluded),
            ("interval_unavailable_reasons", point_estimable - interval_estimable),
        )
        for reason_field, expected_total in reason_totals:
            reasons = getattr(result, f"{prefix}{reason_field}")[key]
            if sum(reasons.values()) != expected_total:
                raise _invalid_result(
                    f"{prefix}{reason_field}[{key!r}] must total {expected_total}"
                )

        for aggregate_field in _KEYED_AGGREGATE_FIELDS:
            value = getattr(result, f"{prefix}{aggregate_field}")[key]
            _validate_aggregate_value(
                aggregate_field,
                value,
                point_estimable=point_estimable,
                interval_estimable=interval_estimable,
                confidence_set_estimable=confidence_set_estimable,
                attempted=attempted,
                location=f"{prefix}{aggregate_field}[{key!r}]",
            )


class EvalResult(_FrozenResultModel):
    """Aggregated metrics from a simulation run.

    Every per-key numeric aggregate (``bias``/``bias_mcse``/
    ``coverage_conditional``/``coverage_conditional_mcse``/
    ``coverage_unconditional``/``coverage_unconditional_mcse``/
    ``set_coverage_conditional``/``set_coverage_conditional_mcse``/
    ``set_coverage_unconditional``/``set_coverage_unconditional_mcse``, and
    their ``breakout_`` mirrors) is nullable. ``None`` normally means the
    underlying population is undefined (e.g. bias with zero finite points,
    or conditional coverage with zero available intervals). Bias may also
    be ``None`` when ``point_estimable`` is positive but its finite
    point-minus-truth mean is outside the finite float range. Numeric
    absence is never ``NaN`` or a fake zero.
    ``coverage_unconditional``/``set_coverage_unconditional`` count a
    missing or failed interval/set as a miss, so they are defined (and may
    be exactly ``0.0``) as soon as at least one replication was attempted.

    Every per-key count is a real integer, for each prespecified
    evaluation key (a core metric name, or -- when the scenario declares
    ``n_segments > 1`` -- a ``"{metric}:{segment}"`` breakout cell):
    ``attempted = point_estimable + set_only + excluded + failed`` and
    ``interval_estimable <= point_estimable <= attempted`` and
    ``confidence_set_estimable = interval_estimable + set_only`` hold for
    every key. ``confidence_set_estimable`` generalizes
    ``interval_estimable``. Point-backed rows normally contribute their own
    interval as the confidence set, so ``set_only`` is 0. Exact-binomial
    risk-ratio sets and rank-projected winsor sets can remain available
    without a finite point; ``set_only`` counts those rows. They contribute
    to confidence-set coverage without contributing to point estimability or
    bias.
    A whole-replication failure (an uncaught coded refusal anywhere in that
    replication's pipeline) counts as ``failed`` for every attempted key,
    core and breakout alike -- not only the core metrics. A breakout cell
    absent from a replication's own output (as opposed to one the
    pipeline explicitly excluded) is still counted, as an explicit
    exclusion, never silently dropped from the denominator.

    ``failure_reasons``/``exclusion_reasons``/``interval_unavailable_reasons``
    are per-key ``{reason_code: count}`` mappings, so exceptions,
    exclusions, and degenerate/unavailable uncertainty stay distinguishable.
    """

    attempted: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    point_estimable: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    interval_estimable: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    confidence_set_estimable: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    set_only: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    excluded: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    failed: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    failure_reasons: Mapping[str, Mapping[str, _NonnegativeInt]] = Field(default_factory=dict)
    exclusion_reasons: Mapping[str, Mapping[str, _NonnegativeInt]] = Field(default_factory=dict)
    interval_unavailable_reasons: Mapping[str, Mapping[str, _NonnegativeInt]] = Field(
        default_factory=dict
    )

    bias: Mapping[str, float | None] = Field(default_factory=dict)  # mean(est - truth) per metric
    bias_mcse: Mapping[str, float | None] = Field(default_factory=dict)
    coverage_conditional: Mapping[str, float | None] = Field(default_factory=dict)
    coverage_conditional_mcse: Mapping[str, float | None] = Field(default_factory=dict)
    coverage_unconditional: Mapping[str, float | None] = Field(default_factory=dict)
    coverage_unconditional_mcse: Mapping[str, float | None] = Field(default_factory=dict)
    set_coverage_conditional: Mapping[str, float | None] = Field(default_factory=dict)
    set_coverage_conditional_mcse: Mapping[str, float | None] = Field(default_factory=dict)
    set_coverage_unconditional: Mapping[str, float | None] = Field(default_factory=dict)
    set_coverage_unconditional_mcse: Mapping[str, float | None] = Field(default_factory=dict)

    # Fraction of SUCCESSFUL replications flagging SRM; None when no
    # replication succeeded (an empty population is None, never a fake 0.0).
    srm_rate: float | None = None

    # Populated only when the scenario declares n_segments > 1; keyed
    # "{metric}:{segment}" -- every declared (metric, segment) cell is
    # present for every key-count/reason/aggregate field above.
    breakout_attempted: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    breakout_point_estimable: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    breakout_interval_estimable: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    breakout_confidence_set_estimable: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    breakout_set_only: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    breakout_excluded: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    breakout_failed: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    breakout_failure_reasons: Mapping[str, Mapping[str, _NonnegativeInt]] = Field(
        default_factory=dict
    )
    breakout_exclusion_reasons: Mapping[str, Mapping[str, _NonnegativeInt]] = Field(
        default_factory=dict
    )
    breakout_interval_unavailable_reasons: Mapping[str, Mapping[str, _NonnegativeInt]] = Field(
        default_factory=dict
    )

    breakout_bias: Mapping[str, float | None] = Field(default_factory=dict)
    breakout_bias_mcse: Mapping[str, float | None] = Field(default_factory=dict)
    breakout_coverage_conditional: Mapping[str, float | None] = Field(default_factory=dict)
    breakout_coverage_conditional_mcse: Mapping[str, float | None] = Field(default_factory=dict)
    breakout_coverage_unconditional: Mapping[str, float | None] = Field(default_factory=dict)
    breakout_coverage_unconditional_mcse: Mapping[str, float | None] = Field(default_factory=dict)
    breakout_set_coverage_conditional: Mapping[str, float | None] = Field(default_factory=dict)
    breakout_set_coverage_conditional_mcse: Mapping[str, float | None] = Field(default_factory=dict)
    breakout_set_coverage_unconditional: Mapping[str, float | None] = Field(default_factory=dict)
    breakout_set_coverage_unconditional_mcse: Mapping[str, float | None] = Field(
        default_factory=dict
    )

    # Present only when covariates are enabled. Missing keys mean no successful
    # CATE fit produced a finite value; the reducer never inserts NaN or zero.
    cate_ate: Mapping[str, float] = Field(default_factory=dict)
    cate_interaction: Mapping[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_result(self) -> EvalResult:
        _validate_keyed_fields(self)
        _validate_keyed_fields(self, "breakout_")
        core_keys = set(self.attempted)
        if not set(self.cate_ate) <= core_keys:
            raise _invalid_result("cate_ate keys must be a subset of attempted keys")
        if not set(self.cate_interaction) <= set(self.cate_ate):
            raise _invalid_result("cate_interaction keys must be a subset of cate_ate keys")
        for field_name in ("cate_ate", "cate_interaction"):
            if any(not math.isfinite(value) for value in getattr(self, field_name).values()):
                raise _invalid_result(f"{field_name} values must be finite")
        if self.srm_rate is not None and (
            not math.isfinite(self.srm_rate) or not 0.0 <= self.srm_rate <= 1.0
        ):
            raise _invalid_result("srm_rate must be finite and in [0, 1], or None")
        return self


# ---------------------------------------------------------------------------
# Shared reducer for attempted/estimable/excluded/failed, bias, and coverage.
# It also serves breakout cells and the switchback outcome key.


@dataclass(frozen=True, slots=True)
class _KeyOutcome:
    """One replication's contribution to one evaluation key's reducer state.

    ``status="ok"`` always carries either a finite ``point`` (``lb``/
    ``ub``/``open_side`` describe its interval; all ``None`` -- with
    ``interval_reason`` set -- when the point has no available interval)
    OR no point at all plus an available set (``set_lower``/``set_upper``).
    Rank sets retain endpoint statuses and map unbounded ends to infinities
    for membership. They can lack a point; exact binomial zero-control-count
    observations can additionally carry an unbounded upper set endpoint:
    ``set_lower`` is always finite (the relative-lift scale's natural
    floor, -1) and ``set_upper is None`` means genuinely unbounded above,
    never unavailable -- the exact method never admits a row with neither
    a point nor a set. ``status="excluded"``/``"failed"`` never carry a
    point or a set. ``reason`` is the exclusion/failure reason code for
    those two statuses.
    """

    status: Literal["ok", "excluded", "failed"]
    point: float | None = None
    lb: float | None = None
    ub: float | None = None
    open_side: Literal["lower", "upper"] | None = None
    interval_reason: str | None = None
    reason: str | None = None
    set_lower: float | None = None
    set_upper: float | None = None
    lower_status: Literal["finite", "unbounded", "undefined"] | None = None
    upper_status: Literal["finite", "unbounded", "undefined"] | None = None


@dataclass(frozen=True, slots=True)
class _KeyStats:
    attempted: int
    point_estimable: int
    interval_estimable: int
    confidence_set_estimable: int
    set_only: int
    excluded: int
    failed: int
    failure_reasons: dict[str, int]
    exclusion_reasons: dict[str, int]
    interval_unavailable_reasons: dict[str, int]
    bias: float | None
    bias_mcse: float | None
    coverage_conditional: float | None
    coverage_conditional_mcse: float | None
    coverage_unconditional: float | None
    coverage_unconditional_mcse: float | None
    set_coverage_conditional: float | None
    set_coverage_conditional_mcse: float | None
    set_coverage_unconditional: float | None
    set_coverage_unconditional_mcse: float | None


def _interval_hit(
    lb: float | None,
    ub: float | None,
    open_side: Literal["lower", "upper"] | None,
    truth: float,
) -> bool | None:
    """Confidence-set membership under the ``Estimate.open_side`` contract.

    ``open_side`` identifies the genuinely unbounded endpoint. If both bounds
    are absent, the interval is unavailable rather than one-sided, so this
    function returns ``None`` instead of counting a false hit or miss.
    """
    if open_side == "lower":
        return None if ub is None else truth <= ub
    if open_side == "upper":
        return None if lb is None else lb <= truth
    if lb is None or ub is None:
        return None
    return lb <= truth <= ub


def _lift_outcome(
    lift: Estimate | None,
    *,
    binomial_set: BinomialConfidenceSet | None = None,
    confidence_set: WinsorConfidenceSet | None = None,
    note: str | None = None,
    winsor_scale: Literal["relative", "additive"] = "relative",
) -> _KeyOutcome:
    """Build the reducer outcome for one successfully produced ``LiftEstimate``/
    ``BreakoutEstimate`` row (``lift``/``binomial_set``/``note`` are that
    row's own fields of the same name).

    A point-backed row (``lift is not None``) always carries a finite
    ``value``; ``lb``/``ub`` both ``None`` is a finite point with no
    available interval -- a real contract boundary, not double-counted as
    an exclusion -- labelled by *note* when the caller has one, else a
    generic ``"no_interval"``. A binomial point-unavailable row
    (``lift is None``: an observed zero control count, see
    ``binomial_rr.py``) instead reports only its persisted
    ``binomial_set`` -- never treated as an exclusion, since the row WAS
    admitted, just without a finite point.
    """
    if confidence_set is not None:
        endpoints = (
            confidence_set.relative if winsor_scale == "relative" else confidence_set.additive
        )
        point = (
            confidence_set.point if winsor_scale == "relative" else confidence_set.additive_point
        )
        undefined = any(e.status == "undefined" for e in (endpoints.lower, endpoints.upper))
        reason = next(
            (e.reason for e in (endpoints.lower, endpoints.upper) if e.status == "undefined"), None
        )
        lower = endpoints.lower.value if endpoints.lower.status == "finite" else -math.inf
        upper = endpoints.upper.value if endpoints.upper.status == "finite" else math.inf
        if point is None:
            if not undefined:
                return _KeyOutcome(
                    status="ok",
                    set_lower=lower,
                    set_upper=upper,
                    lower_status=endpoints.lower.status,
                    upper_status=endpoints.upper.status,
                )
            return _KeyOutcome(
                status="excluded",
                reason=reason or "winsor_point_unavailable",
                lower_status=endpoints.lower.status,
                upper_status=endpoints.upper.status,
            )
        return _KeyOutcome(
            status="ok",
            point=point,
            lb=None if undefined else lower,
            ub=None if undefined else upper,
            interval_reason=reason,
            lower_status=endpoints.lower.status,
            upper_status=endpoints.upper.status,
        )
    if lift is None:
        assert binomial_set is not None, "validated: lift=None implies a binomial_set"
        return _KeyOutcome(status="ok", set_lower=binomial_set.lower, set_upper=binomial_set.upper)
    interval_reason = (
        None if (lift.lb is not None or lift.ub is not None) else (note or "no_interval")
    )
    return _KeyOutcome(
        status="ok",
        point=lift.value,
        lb=lift.lb,
        ub=lift.ub,
        open_side=lift.open_side,
        interval_reason=interval_reason,
    )


def _bernoulli_mcse(rate: float, n: int) -> float:
    """Binomial Monte-Carlo SE of an observed rate over *n* observations.

    Reports the plain plug-in formula, including its zero value at
    ``rate`` in ``{0, 1}``, as the repository's uncertainty CONVENTION;
    that zero is not proof of literally zero uncertainty -- calibration
    gates reading this field use an independently derived MC bound
    instead of trusting a plug-in zero at an extreme rate (see
    ``tests/mc.py``).
    """
    return math.sqrt(rate * (1.0 - rate) / n)


def _scaled_bias_stats(points: Sequence[float], truth: float) -> tuple[float | None, float | None]:
    """Compute mean bias and its MCSE without overflowing intermediate sums."""
    n = len(points)
    bias_scale = max(abs(truth), *(abs(point) for point in points))
    if bias_scale == 0.0:
        bias = 0.0
    else:
        scaled_truth = truth / bias_scale
        scaled_bias = math.fsum(point / bias_scale - scaled_truth for point in points) / n
        candidate = bias_scale * scaled_bias
        bias = candidate if math.isfinite(candidate) else None

    if n < 2:
        return bias, None

    point_scale = max(abs(point) for point in points)
    if point_scale == 0.0:
        return bias, 0.0
    anchor = points[0]
    deviations = [
        (point - anchor) / point_scale
        if (point >= 0.0) == (anchor >= 0.0)
        else point / point_scale - anchor / point_scale
        for point in points
    ]
    mean_deviation = math.fsum(deviations) / n
    scaled_sq_sum = math.fsum((deviation - mean_deviation) ** 2 for deviation in deviations)
    candidate_mcse = point_scale * math.sqrt((scaled_sq_sum / (n - 1)) / n)
    return bias, candidate_mcse if math.isfinite(candidate_mcse) else None


def _coverage_stats(hits: Sequence[bool], denominator: int) -> tuple[float | None, float | None]:
    if denominator == 0:
        return None, None
    coverage = sum(hits) / denominator
    mcse = _bernoulli_mcse(coverage, denominator) if denominator > 1 else None
    return coverage, mcse


def _reduce_key(outcomes: Sequence[_KeyOutcome], truth: float) -> _KeyStats:
    """The one shared attempted/estimable/excluded/failed + bias/coverage
    reducer, for a core metric key, a breakout cell key, or the
    switchback estimand's single "outcome" key alike.

    For R attempted replications, P finite points, I available intervals,
    H hits: bias = mean(point - truth) over P (None if P=0, or if the
    finite mean is outside the finite float range); bias MCSE = sample
    SD(point - truth, ddof=1)/sqrt(P) (None if P<2); conditional
    coverage = H/I (None if I=0); unconditional coverage = H/R, counting
    every missing/failed interval as a miss (None if R=0 -- accepted here
    so this reducer stays total, but every public runner refuses R=0
    before ever reaching it); coverage MCSE uses the matching denominator
    (I or R) and needs at least two observations.

    Confidence-SET accounting generalizes the point-interval accounting
    above with the same conventions, over S (``confidence_set_estimable``)
    available sets and H_S set hits. Point-backed outcomes contribute their
    intervals to both I/H and S/H_S. A point-free binomial or rank confidence
    set contributes only to S/H_S via ``set_lower``/``set_upper``, never to
    P/I/bias; ``set_only`` counts those outcomes.
    set_coverage_conditional = H_S/S (None if S=0);
    set_coverage_unconditional = H_S/R (None if R=0).
    """
    attempted = len(outcomes)
    points: list[float] = []
    hits: list[bool] = []
    set_hits: list[bool] = []
    set_only = 0
    excluded = 0
    failed = 0
    failure_reasons: dict[str, int] = defaultdict(int)
    exclusion_reasons: dict[str, int] = defaultdict(int)
    interval_unavailable_reasons: dict[str, int] = defaultdict(int)

    for outcome in outcomes:
        if outcome.status == "failed":
            failed += 1
            failure_reasons[outcome.reason or "unknown"] += 1
            continue
        if outcome.status == "excluded":
            excluded += 1
            exclusion_reasons[outcome.reason or "unknown"] += 1
            continue
        if outcome.point is None:
            # Point-free observations admitted here carry an actual set;
            # undefined rank sets remain in the excluded denominator.
            assert outcome.set_lower is not None, (
                "an 'ok' outcome with no point always carries an available set"
            )
            set_only += 1
            set_hits.append(
                outcome.set_lower <= truth
                and (outcome.set_upper is None or truth <= outcome.set_upper)
            )
            continue
        assert math.isfinite(outcome.point), "an 'ok' outcome's point is always finite"
        points.append(outcome.point)
        hit = _interval_hit(outcome.lb, outcome.ub, outcome.open_side, truth)
        if hit is None:
            interval_unavailable_reasons[outcome.interval_reason or "no_interval"] += 1
        else:
            hits.append(hit)
            set_hits.append(hit)

    point_estimable = len(points)
    interval_estimable = len(hits)
    confidence_set_estimable = len(set_hits)

    if point_estimable == 0:
        bias = None
        bias_mcse = None
    else:
        bias, bias_mcse = _scaled_bias_stats(points, truth)

    coverage_conditional, coverage_conditional_mcse = _coverage_stats(hits, interval_estimable)
    coverage_unconditional, coverage_unconditional_mcse = _coverage_stats(hits, attempted)
    set_coverage_conditional, set_coverage_conditional_mcse = _coverage_stats(
        set_hits, confidence_set_estimable
    )
    set_coverage_unconditional, set_coverage_unconditional_mcse = _coverage_stats(
        set_hits, attempted
    )

    return _KeyStats(
        attempted=attempted,
        point_estimable=point_estimable,
        interval_estimable=interval_estimable,
        confidence_set_estimable=confidence_set_estimable,
        set_only=set_only,
        excluded=excluded,
        failed=failed,
        failure_reasons=dict(failure_reasons),
        exclusion_reasons=dict(exclusion_reasons),
        interval_unavailable_reasons=dict(interval_unavailable_reasons),
        bias=bias,
        bias_mcse=bias_mcse,
        coverage_conditional=coverage_conditional,
        coverage_conditional_mcse=coverage_conditional_mcse,
        coverage_unconditional=coverage_unconditional,
        coverage_unconditional_mcse=coverage_unconditional_mcse,
        set_coverage_conditional=set_coverage_conditional,
        set_coverage_conditional_mcse=set_coverage_conditional_mcse,
        set_coverage_unconditional=set_coverage_unconditional,
        set_coverage_unconditional_mcse=set_coverage_unconditional_mcse,
    )


@dataclass(frozen=True, slots=True)
class _ReducedFields:
    """``_reduce_key``'s output, fanned out across every key into the
    dict-of-dicts shape ``EvalResult``'s core/breakout field groups share."""

    attempted: dict[str, int]
    point_estimable: dict[str, int]
    interval_estimable: dict[str, int]
    confidence_set_estimable: dict[str, int]
    set_only: dict[str, int]
    excluded: dict[str, int]
    failed: dict[str, int]
    failure_reasons: dict[str, dict[str, int]]
    exclusion_reasons: dict[str, dict[str, int]]
    interval_unavailable_reasons: dict[str, dict[str, int]]
    bias: dict[str, float | None]
    bias_mcse: dict[str, float | None]
    coverage_conditional: dict[str, float | None]
    coverage_conditional_mcse: dict[str, float | None]
    coverage_unconditional: dict[str, float | None]
    coverage_unconditional_mcse: dict[str, float | None]
    set_coverage_conditional: dict[str, float | None]
    set_coverage_conditional_mcse: dict[str, float | None]
    set_coverage_unconditional: dict[str, float | None]
    set_coverage_unconditional_mcse: dict[str, float | None]


def _reduce_all(
    outcomes_by_key: dict[str, list[_KeyOutcome]], truth_by_key: dict[str, float]
) -> _ReducedFields:
    stats = {
        key: _reduce_key(outcomes, truth_by_key[key]) for key, outcomes in outcomes_by_key.items()
    }
    return _ReducedFields(
        attempted={k: s.attempted for k, s in stats.items()},
        point_estimable={k: s.point_estimable for k, s in stats.items()},
        interval_estimable={k: s.interval_estimable for k, s in stats.items()},
        confidence_set_estimable={k: s.confidence_set_estimable for k, s in stats.items()},
        set_only={k: s.set_only for k, s in stats.items()},
        excluded={k: s.excluded for k, s in stats.items()},
        failed={k: s.failed for k, s in stats.items()},
        failure_reasons={k: s.failure_reasons for k, s in stats.items()},
        exclusion_reasons={k: s.exclusion_reasons for k, s in stats.items()},
        interval_unavailable_reasons={k: s.interval_unavailable_reasons for k, s in stats.items()},
        bias={k: s.bias for k, s in stats.items()},
        bias_mcse={k: s.bias_mcse for k, s in stats.items()},
        coverage_conditional={k: s.coverage_conditional for k, s in stats.items()},
        coverage_conditional_mcse={k: s.coverage_conditional_mcse for k, s in stats.items()},
        coverage_unconditional={k: s.coverage_unconditional for k, s in stats.items()},
        coverage_unconditional_mcse={k: s.coverage_unconditional_mcse for k, s in stats.items()},
        set_coverage_conditional={k: s.set_coverage_conditional for k, s in stats.items()},
        set_coverage_conditional_mcse={
            k: s.set_coverage_conditional_mcse for k, s in stats.items()
        },
        set_coverage_unconditional={k: s.set_coverage_unconditional for k, s in stats.items()},
        set_coverage_unconditional_mcse={
            k: s.set_coverage_unconditional_mcse for k, s in stats.items()
        },
    )


# ---------------------------------------------------------------------------
# Default scenario helpers


def _make_metrics() -> list[Metric]:
    """Create the three canonical Metric objects."""
    return [
        ConversionMetric(name="conversion", entity="user", fact="conversion"),
        MeanMetric(name="count", entity="user", fact="visit", aggregation="count"),
        MeanMetric(name="revenue", entity="user", fact="revenue", aggregation="sum"),
    ]


# ---------------------------------------------------------------------------
# Breakout routing (segment dimension)


def _run_breakout_for_replication(
    scenario: Scenario,
    ibis_conn: Any,
    raw_log: pa.Table,
    exposures: Any,
    fact_tbl: Any,
    experiment: Experiment,
    metrics: list[Metric],
) -> list[BreakoutEstimate]:
    """Segment-broken-out lift for every metric in ``scenario.true_lift``,
    when the scenario declares ``n_segments > 1``. ``[]`` otherwise.

    Mirrors the native path's own breakout construction (see
    ``increment.query.native_source.DefinitionsMomentSource._breakout_moments_source``):
    per-metric totals grouped ``by=["segment"]``, wrapped in a
    ``BreakoutMomentsSource`` scoped to that one dimension, then read
    through the public ``readouts.breakout`` entry point.
    """
    if scenario.n_segments <= 1:
        return []

    import pyarrow.compute as pc

    from increment.query.builders import (
        group_summary,
        metric_events,
        unit_day_spine_stats,
        unit_totals,
    )
    from increment.readouts import breakout as breakout_readout
    from increment.semantics.design import Randomized
    from increment.sources import BreakoutMomentsSource

    is_segment = pc.equal(raw_log.column("event"), "segment")  # ty: ignore[unresolved-attribute]
    segment_data = raw_log.filter(is_segment).select(["unit_id", "value"])
    if segment_data.num_rows == 0:
        return []
    segment_pdf = segment_data.to_pandas()
    segment_pdf["segment"] = segment_pdf["value"].astype("int64").astype(str)
    properties_tbl = ibis_conn.create_table("segment_props", segment_pdf[["unit_id", "segment"]])

    by = ["segment"]
    selected_metrics = [m for m in metrics if m.name in scenario.true_lift]
    rows: list[dict[str, Any]] = []
    for metric in selected_metrics:
        value_col = None
        if isinstance(metric, MeanMetric) and metric.aggregation == "sum":
            value_col = "value"
        m_events = metric_events(fact_tbl, metric, value_column=value_col)
        spine, stats = unit_day_spine_stats(exposures, m_events, experiment, metric.name)
        totals_by = unit_totals(
            spine,
            stats,
            metric,
            experiment,
            by=by,
            properties_table=properties_tbl,
            warn_on_censoring=False,
        )
        summary_by = group_summary(totals_by, by=by)
        rows.extend(summary_by.to_pyarrow().to_pylist())

    if not rows:
        return []

    src = BreakoutMomentsSource(
        rows,
        dimension="segment",
        metrics=selected_metrics,
        study_id="sim",
        design=Randomized(control_group="control"),
    )
    return list(breakout_readout(src, "segment", correction="none"))


# ---------------------------------------------------------------------------
# CATE routing (continuous covariate interaction)


def _cate_source_frame(pdf: Any, metric_name: str) -> Any:
    """One row per unit -- ``unit_id``, ``group_id``, ``covariate``, and a
    column named *metric_name* (the terse ``from_unit_summary`` metrics
    dict reads the value column by that name) -- built purely from the raw
    simulated log (no ibis, no warehouse query).
    """
    exposures = (
        pdf[pdf["event"] == "exposure"][["unit_id", "group_id"]]
        .drop_duplicates()
        .set_index("unit_id")
    )
    exposure_ts = pdf[pdf["event"] == "exposure"].groupby("unit_id")["ts"].min()

    def post_exposure(event: str) -> Any:
        rows = pdf[pdf["event"] == event].copy()
        if rows.empty:
            return rows
        return rows[rows["ts"] > rows["unit_id"].map(exposure_ts)]

    exposures["covariate"] = pdf[pdf["event"] == "covariate"].set_index("unit_id")["value"]
    if metric_name == "conversion":
        converted = set(post_exposure("conversion")["unit_id"])
        exposures[metric_name] = exposures.index.isin(converted).astype(float)
    elif metric_name == "count":
        counts = post_exposure("visit").groupby("unit_id").size()
        exposures[metric_name] = counts.reindex(exposures.index).fillna(0.0)
    elif metric_name == "revenue":
        revenue = post_exposure("revenue").groupby("unit_id")["value"].sum()
        exposures[metric_name] = revenue.reindex(exposures.index).fillna(0.0)
    else:
        _raise("simulate.runner.cate_routing_does", metric_name=metric_name)
    return exposures.reset_index()


def _run_cate_for_replication(
    scenario: Scenario, raw_log: pa.Table, metrics: list[Metric]
) -> dict[str, CateResult]:
    """Fit a CATE model interacting the observable "covariate" with
    treatment, for every metric in ``scenario.true_lift`` this routing
    supports, when the scenario declares ``covariate_sd > 0``. ``{}``
    otherwise.

    Routes entirely through ``increment.frame.from_unit_summary`` -- a
    public in-memory API, never a warehouse query -- since the native
    ibis path's ``unit_frame()`` explicitly refuses covariates.
    """
    if scenario.covariate_sd <= 0:
        return {}

    from increment.cate import estimate_cate
    from increment.estimation.cate import Covariate
    from increment.frame import from_unit_summary

    pdf = raw_log.to_pandas()
    if not (pdf["event"] == "covariate").any():
        return {}

    results: dict[str, CateResult] = {}
    for metric in metrics:
        if metric.name not in scenario.true_lift or metric.name not in _METRIC_SPEC_TYPE:
            continue
        frame_df = _cate_source_frame(pdf, metric.name)
        src = from_unit_summary(
            frame_df,
            unit="unit_id",
            group="group_id",
            control="control",
            metrics={metric.name: _METRIC_SPEC_TYPE[metric.name]},
            experiment_id="sim",
        )
        results[metric.name] = estimate_cate(
            src, metric.name, control="control", interact=[Covariate(name="covariate")]
        )
    return results


# ---------------------------------------------------------------------------
# Per-replication runner


def _run_one_replication(
    scenario: Scenario,
    ibis_conn: Any,  # ibis.BaseBackend
    metrics: list[Metric],
) -> dict[str, Any]:
    """Run the full pipeline for one replication.

    Returns a dict with keys ``metric_estimates`` (list of LiftEstimate),
    ``metric_failures`` (dict[ArmHypothesisKey, DecisionFailure] --
    ``estimate_lift``'s guard failures, one entry per metric that could not
    be estimated), ``group_counts`` (dict[str,int]), ``experiment``
    (Experiment), ``breakout_estimates`` (list of BreakoutEstimate,
    possibly empty), and ``cate_results`` (dict[str, CateResult],
    possibly empty).
    """
    # Deferred: pulling in the query builders imports ibis, which is heavy
    # and unnecessary until a replication actually runs.
    from increment.query.builders import (
        first_exposures,
        group_summary,
        metric_events,
        unit_day_spine_stats,
        unit_totals,
    )

    start_dt = _START_DATE
    end_dt = start_dt + timedelta(days=scenario.n_days)
    experiment = Experiment(
        name="sim",
        exposure="exposure",
        unit="user",
        start=start_dt,
        end=end_dt,
        plan=AnalysisPlan(secondaries=list(scenario.true_lift.keys())),
        control_group="control",
    )

    # Generate raw data
    raw_log: pa.Table = simulate_raw_logs(scenario)

    # Filter into exposure events and fact events using pyarrow
    import pyarrow.compute as pc

    is_exposure = pc.equal(raw_log.column("event"), "exposure")  # ty: ignore[unresolved-attribute] - pyarrow.compute is dynamically generated, no stubs
    exposure_data = raw_log.filter(is_exposure).select(
        ["unit_id", "ts", "experiment_id", "group_id"]
    )
    is_fact = pc.invert(is_exposure)  # ty: ignore[unresolved-attribute]
    fact_data = raw_log.filter(is_fact).select(["unit_id", "ts", "event", "value"])

    # Write to ibis tables
    exposure_tbl = ibis_conn.create_table("exposures", exposure_data)
    fact_tbl = ibis_conn.create_table("facts", fact_data)

    # First exposures
    exposures = first_exposures(exposure_tbl, experiment)

    # Group counts for SRM
    group_counts_df = exposures.group_by("group_id").agg(n=exposures.count()).execute()
    group_counts = dict(zip(group_counts_df["group_id"], group_counts_df["n"], strict=True))

    # Summaries stay Arrow, not pandas: pandas coerces SQL NULL moments (e.g.
    # an unmaterialised CUPED covariate sum) to NaN, which estimation rejects as corrupt.
    summaries: list[pa.Table] = []
    for metric in metrics:
        if metric.name not in scenario.true_lift:
            continue  # skip metrics not in this scenario

        value_col = None
        if isinstance(metric, MeanMetric) and metric.aggregation == "sum":
            value_col = "value"

        m_events = metric_events(fact_tbl, metric, value_column=value_col)
        spine, stats = unit_day_spine_stats(exposures, m_events, experiment, metric.name)
        totals = unit_totals(spine, stats, metric, experiment, warn_on_censoring=False)
        summ = group_summary(totals).to_pyarrow()
        summaries.append(summ)

    combined = pa.concat_tables(summaries)

    # Estimate lift
    metrics_by_name = {m.name: m for m in metrics}
    computation = estimate_lift(
        metrics=metrics,
        summary=combined,
        control_group="control",
    )
    estimates = [
        e.model_copy(
            update={"preferred_direction": metrics_by_name[e.metric].declared_preferred_direction}
        )
        for e in computation.results
    ]
    # estimate_lift's failures are keyed by the general HypothesisKey union;
    # this randomized-ITT core path only ever produces ArmHypothesisKey, but
    # the static return type doesn't narrow that for us.
    metric_failures: dict[Any, DecisionFailure] = dict(computation.failures)

    breakout_estimates = _run_breakout_for_replication(
        scenario, ibis_conn, raw_log, exposures, fact_tbl, experiment, metrics
    )
    cate_results = _run_cate_for_replication(scenario, raw_log, metrics)

    return {
        "metric_estimates": estimates,
        "metric_failures": metric_failures,
        "group_counts": group_counts,
        "experiment": experiment,
        "breakout_estimates": breakout_estimates,
        "cate_results": cate_results,
    }


# ---------------------------------------------------------------------------
# Aggregation


def _aggregate_cate(
    cate_results: list[dict[str, CateResult]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean fitted ATE and mean "covariate" interaction coefficient across
    replications, per metric -- proof CATE was genuinely fit each rep."""
    ate: dict[str, list[float]] = defaultdict(list)
    interaction: dict[str, list[float]] = defaultdict(list)
    for rep_results in cate_results:
        for metric_name, fit in rep_results.items():
            ate[metric_name].append(fit.ate)
            if fit.interactions:
                interaction[metric_name].append(fit.interactions[0].coef)
    return (
        {k: float(np.mean(v)) for k, v in ate.items() if v},
        {k: float(np.mean(v)) for k, v in interaction.items() if v},
    )


# ---------------------------------------------------------------------------
# Public API


def run_end_to_end(
    scenario: Scenario,
    replications: int = 1,
) -> EvalResult:
    """Run the full increment pipeline on synthetic data across
    ``replications`` independent Monte Carlo replications.

    Returns, for every core metric key and (when the scenario declares
    ``n_segments > 1``) every "{metric}:{segment}" breakout key, coherent
    attempted/point_estimable/interval_estimable/excluded/failed counts
    plus nullable bias/coverage aggregates and their Monte Carlo SE (see
    ``EvalResult``); the SRM detection rate across successful
    replications; and, when the scenario declares ``covariate_sd > 0``,
    the fitted CATE summary -- all exercised end-to-end through this one
    entry point.

    A whole-replication failure (any coded refusal anywhere in that
    replication's pipeline) is caught and counted as ``failed`` for every
    attempted key rather than aborting the entire run.
    """
    replications = _validate_replications(replications)

    import ibis

    metrics = _make_metrics()
    metric_keys = list(scenario.true_lift)
    breakout_keys = (
        [f"{metric}:{segment}" for metric in metric_keys for segment in range(scenario.n_segments)]
        if scenario.n_segments > 1
        else []
    )

    metric_outcomes: dict[str, list[_KeyOutcome]] = {key: [] for key in metric_keys}
    breakout_outcomes: dict[str, list[_KeyOutcome]] = {key: [] for key in breakout_keys}
    all_cate: list[dict[str, CateResult]] = []
    srm_flags: list[bool] = []

    # Spawn per-replication seeds from a single SeedSequence for collision-free
    # streams; an arithmetic scheme like seed + rep*K can collide across scenarios.
    rep_seeds = np.random.SeedSequence(scenario.seed).spawn(replications)
    for rep in range(replications):
        rep_seed = int(rep_seeds[rep].generate_state(1, dtype=np.uint64)[0])
        rep_scenario = scenario.model_copy(update={"seed": rep_seed})
        con = ibis.duckdb.connect()
        try:
            result = _run_one_replication(rep_scenario, con, metrics)
        except CodedError as exc:
            for key in metric_keys:
                metric_outcomes[key].append(_KeyOutcome(status="failed", reason=exc.code))
            for key in breakout_keys:
                breakout_outcomes[key].append(_KeyOutcome(status="failed", reason=exc.code))
            continue
        finally:
            con.disconnect()

        estimates_by_metric: dict[str, LiftEstimate] = {
            e.metric: e for e in result["metric_estimates"]
        }
        failures_by_metric: dict[str, DecisionFailure] = {
            hypothesis.metric: failure for hypothesis, failure in result["metric_failures"].items()
        }
        for key in metric_keys:
            estimate = estimates_by_metric.get(key)
            if estimate is not None:
                metric_outcomes[key].append(
                    _lift_outcome(
                        estimate.lift,
                        binomial_set=estimate.binomial_set,
                        confidence_set=estimate.confidence_set,
                        note=estimate.note,
                    )
                )
                continue
            failure = failures_by_metric.get(key)
            metric_outcomes[key].append(
                _KeyOutcome(
                    status="excluded", reason=failure.code if failure is not None else "no_result"
                )
            )

        breakout_by_key = {
            f"{row.metric}:{row.dimension_value}": row for row in result["breakout_estimates"]
        }
        for key in breakout_keys:
            row = breakout_by_key.get(key)
            if row is None:
                breakout_outcomes[key].append(
                    _KeyOutcome(status="excluded", reason="missing_breakout_cell")
                )
            elif row.excluded is not None:
                breakout_outcomes[key].append(
                    _KeyOutcome(status="excluded", reason=str(row.excluded))
                )
            else:
                breakout_outcomes[key].append(
                    _lift_outcome(row.lift, binomial_set=row.binomial_set, note=row.note)
                )

        all_cate.append(result["cate_results"])

        # SRM check on every successful replication (against the configured design).
        r = scenario.assignment_ratio
        srm_result = sample_ratio_mismatch(
            result["group_counts"],
            expected={"control": 1.0 - r, "treatment": r},
        )
        srm_flags.append(srm_result.is_srm)

    truth_by_metric = {key: scenario.true_lift[key] for key in metric_keys}
    truth_by_breakout = {key: scenario.true_lift[key.rsplit(":", 1)[0]] for key in breakout_keys}

    core = _reduce_all(metric_outcomes, truth_by_metric)
    breakout = _reduce_all(breakout_outcomes, truth_by_breakout)
    cate_ate, cate_interaction = _aggregate_cate(all_cate)
    srm_rate = float(np.mean(srm_flags)) if srm_flags else None

    return EvalResult(
        attempted=core.attempted,
        point_estimable=core.point_estimable,
        interval_estimable=core.interval_estimable,
        confidence_set_estimable=core.confidence_set_estimable,
        set_only=core.set_only,
        excluded=core.excluded,
        failed=core.failed,
        failure_reasons=core.failure_reasons,
        exclusion_reasons=core.exclusion_reasons,
        interval_unavailable_reasons=core.interval_unavailable_reasons,
        bias=core.bias,
        bias_mcse=core.bias_mcse,
        coverage_conditional=core.coverage_conditional,
        coverage_conditional_mcse=core.coverage_conditional_mcse,
        coverage_unconditional=core.coverage_unconditional,
        coverage_unconditional_mcse=core.coverage_unconditional_mcse,
        set_coverage_conditional=core.set_coverage_conditional,
        set_coverage_conditional_mcse=core.set_coverage_conditional_mcse,
        set_coverage_unconditional=core.set_coverage_unconditional,
        set_coverage_unconditional_mcse=core.set_coverage_unconditional_mcse,
        srm_rate=srm_rate,
        breakout_attempted=breakout.attempted,
        breakout_point_estimable=breakout.point_estimable,
        breakout_interval_estimable=breakout.interval_estimable,
        breakout_confidence_set_estimable=breakout.confidence_set_estimable,
        breakout_set_only=breakout.set_only,
        breakout_excluded=breakout.excluded,
        breakout_failed=breakout.failed,
        breakout_failure_reasons=breakout.failure_reasons,
        breakout_exclusion_reasons=breakout.exclusion_reasons,
        breakout_interval_unavailable_reasons=breakout.interval_unavailable_reasons,
        breakout_bias=breakout.bias,
        breakout_bias_mcse=breakout.bias_mcse,
        breakout_coverage_conditional=breakout.coverage_conditional,
        breakout_coverage_conditional_mcse=breakout.coverage_conditional_mcse,
        breakout_coverage_unconditional=breakout.coverage_unconditional,
        breakout_coverage_unconditional_mcse=breakout.coverage_unconditional_mcse,
        breakout_set_coverage_conditional=breakout.set_coverage_conditional,
        breakout_set_coverage_conditional_mcse=breakout.set_coverage_conditional_mcse,
        breakout_set_coverage_unconditional=breakout.set_coverage_unconditional,
        breakout_set_coverage_unconditional_mcse=breakout.set_coverage_unconditional_mcse,
        cate_ate=cate_ate,
        cate_interaction=cate_interaction,
    )


# ---------------------------------------------------------------------------
# CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ground-truth evaluation of the increment pipeline."
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=200,
        help="Number of Monte Carlo replications (default: 200)",
    )
    args = parser.parse_args()

    scenario = Scenario(
        n_units=1000,
        n_days=14,
        true_lift={"conversion": 0.0, "count": 0.0, "revenue": 0.0},
        seed=42,
    )
    try:
        result = run_end_to_end(scenario, replications=args.replications)
    except CodedError as exc:
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(result.model_dump_json(indent=2))


class SwitchbackEvalResult(_FrozenResultModel):
    """Monte Carlo summaries for the qualified unit-t switchback approximation.

    Retains one-outcome numeric dictionaries (keyed ``"outcome"``, the
    switchback panel's sole metric) for consumer compatibility, with the
    same nullable bias/coverage semantics as ``EvalResult``.
    ``attempted``/``point_estimable``/``interval_estimable``/``excluded``/
    ``failed`` are scalar (not per-key dicts) because this evaluator
    always scores exactly one outcome.
    """

    reference_kind: Literal["unit_t_approximation"] = "unit_t_approximation"
    bias: Mapping[str, float | None]
    bias_mcse: Mapping[str, float | None]
    coverage_conditional: Mapping[str, float | None]
    coverage_conditional_mcse: Mapping[str, float | None]
    coverage_unconditional: Mapping[str, float | None]
    coverage_unconditional_mcse: Mapping[str, float | None]
    attempted: _NonnegativeInt
    point_estimable: _NonnegativeInt
    interval_estimable: _NonnegativeInt
    excluded: _NonnegativeInt
    failed: _NonnegativeInt
    failure_reasons: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    exclusion_reasons: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    interval_unavailable_reasons: Mapping[str, _NonnegativeInt] = Field(default_factory=dict)
    supported: bool
    assumption_violation: bool

    @model_validator(mode="after")
    def _validate_result(self) -> SwitchbackEvalResult:
        expected_keys = {"outcome"}
        for field_name in _POINT_AGGREGATE_FIELDS:
            values = getattr(self, field_name)
            if set(values) != expected_keys:
                raise _invalid_result(f"{field_name} must be keyed exactly by 'outcome'")
            value = values["outcome"]
            _validate_aggregate_value(
                field_name,
                value,
                point_estimable=self.point_estimable,
                interval_estimable=self.interval_estimable,
                confidence_set_estimable=self.interval_estimable,
                attempted=self.attempted,
                location=f"{field_name}['outcome']",
            )

        if self.attempted != self.point_estimable + self.excluded + self.failed:
            raise _invalid_result("attempted must equal point_estimable + excluded + failed")
        if self.interval_estimable > self.point_estimable:
            raise _invalid_result("interval_estimable must not exceed point_estimable")

        reason_totals = (
            ("failure_reasons", self.failed),
            ("exclusion_reasons", self.excluded),
            (
                "interval_unavailable_reasons",
                self.point_estimable - self.interval_estimable,
            ),
        )
        for field_name, expected_total in reason_totals:
            if sum(getattr(self, field_name).values()) != expected_total:
                raise _invalid_result(f"{field_name} must total {expected_total}")
        return self


def _run_switchback_replication(scenario: SwitchbackScenario) -> Estimate:
    from increment.analysis import Analysis
    from increment.frame import MetricSpec
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SharedScheduleOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized

    panel = simulate_switchback_panel(scenario)
    sequence = (
        SharedScheduleOrder(probability_ct=scenario.probability_ct)
        if scenario.shared_schedule
        else IndependentBernoulliOrder(probability_ct=scenario.probability_ct)
    )
    analysis = Analysis.from_switchback_panel(
        panel,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics=[MetricSpec(name="outcome", type="mean", value_column="outcome")],
        identification=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
        ),
        assignment=SwitchbackAssignment(
            sequence=sequence,
            window=SwitchbackWindow(
                washout_steps=scenario.washout_steps,
                observation_steps=scenario.observation_steps,
                carryover_order=scenario.declared_carryover_order,
            ),
        ),
    )
    result = cast("ContrastResult", analysis.run()[0])
    return result.estimate


def run_switchback_end_to_end(
    scenario: SwitchbackScenario,
    replications: int = 1,
) -> SwitchbackEvalResult:
    """Evaluate the qualified unit-t approximation on deterministic switchback data.

    Bias and coverage target the treatment effect over the retained steps.
    A whole-replication failure (any coded refusal building or analyzing
    that replication's panel) is caught and counted as ``failed`` rather
    than aborting the entire run.
    """
    replications = _validate_replications(replications)
    rep_seeds = np.random.SeedSequence(scenario.seed).spawn(replications)
    outcomes: list[_KeyOutcome] = []
    for child in rep_seeds:
        seed = int(child.generate_state(1, dtype=np.uint64)[0])
        try:
            estimate = _run_switchback_replication(scenario.model_copy(update={"seed": seed}))
        except CodedError as exc:
            outcomes.append(_KeyOutcome(status="failed", reason=exc.code))
            continue
        outcomes.append(_lift_outcome(estimate))

    retained_fraction = (
        scenario.observation_steps - scenario.declared_carryover_order
    ) / scenario.observation_steps
    stats = _reduce_key(outcomes, scenario.treatment_effect * retained_fraction)
    assumption_violation = (
        scenario.carryover_amplitude > 0.0
        and scenario.true_carryover_order > scenario.declared_carryover_order
    )
    return SwitchbackEvalResult(
        bias={"outcome": stats.bias},
        bias_mcse={"outcome": stats.bias_mcse},
        coverage_conditional={"outcome": stats.coverage_conditional},
        coverage_conditional_mcse={"outcome": stats.coverage_conditional_mcse},
        coverage_unconditional={"outcome": stats.coverage_unconditional},
        coverage_unconditional_mcse={"outcome": stats.coverage_unconditional_mcse},
        attempted=stats.attempted,
        point_estimable=stats.point_estimable,
        interval_estimable=stats.interval_estimable,
        excluded=stats.excluded,
        failed=stats.failed,
        failure_reasons=stats.failure_reasons,
        exclusion_reasons=stats.exclusion_reasons,
        interval_unavailable_reasons=stats.interval_unavailable_reasons,
        supported=not assumption_violation,
        assumption_violation=assumption_violation,
    )
