"""Confounding sensitivity for a declared unadjusted cohort risk ratio."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from increment.errors import (
    InvalidRequestError,
    UnsupportedRequestError,
    raiser,
    refusals,
)

__all__ = ["CohortRiskRatio", "RiskRatioSensitivity", "risk_ratio_sensitivity"]

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "sensitivity.input.number": "{field} must be a finite number in {domain}; received {value!r}",
        "sensitivity.input.interval": "confidence_interval must contain two positive finite bounds bracketing the risk ratio; received {value!r}",
        "sensitivity.input.identity": "{field} must be a nonempty string; received {value!r}",
        "sensitivity.input.study": "study must be a CohortRiskRatio; received {value_type}",
        "sensitivity.decision.direction": "direction must be 'above', 'below', or None; received {value!r}",
        "sensitivity.decision.threshold": "threshold {threshold} must lie toward the null: >= 1 for 'above', <= 1 for 'below'; received direction={direction!r}",
        "sensitivity.numerical.range": "{quantity} cannot be represented as a positive finite float for these inputs; no finite sensitivity result is reported",
    },
)
_REFUSALS.update(
    refusals(
        UnsupportedRequestError,
        {
            "sensitivity.study.unsupported": "unsupported {field}={value!r}; this construction accepts {supported!r}. Supply an unadjusted fixed-horizon cohort risk ratio for a binary exposure and outcome; no conversion from other estimands or scales is performed",
            "sensitivity.study.adjusted": "adjustment_covariates={value!r} declares an adjusted estimate; this implementation accepts unadjusted cohort risks only. Conditional or standardized adjusted-risk sensitivity needs a construction matched to that estimand",
        },
    )
)
_raise = raiser(_REFUSALS)

_ASSUMPTIONS = (
    "Binary exposure and outcome; cohort risks refer to a common follow-up and the declared population.",
    "Consistency, positivity and no interference hold; conditioning on the hypothetical confounders would establish exchangeability.",
    "Confounding strengths bound the exposure-confounder prevalence ratio and the maximum confounder-outcome risk ratio in either exposure arm; below decisions reverse the exposure comparison.",
    "Selection bias, measurement error and estimator misspecification are outside this construction.",
    "E-values are equal-strength (minimax) tipping points; asymmetric confounding need not put both associations above the E-value.",
    "Confidence-limit sensitivity inherits the supplied fixed-horizon interval; it is not a newly calibrated causal confidence interval.",
    "Sensitivity quantifies robustness under assumptions and does not establish causal identification or absence of omitted confounding.",
)


def _refusal_value(value: object, *, depth: int = 0) -> object:
    """Snapshot invalid input without unsafe reprs or foreign scalar recursion."""
    if value is None or type(value) in (bool, float):
        return value
    if type(value) is str:
        return value[:256]
    if type(value) is int:
        return value if value.bit_length() <= 800 else f"<int of {value.bit_length()} bits>"
    if depth < 3 and isinstance(value, (tuple, list)) and type(value) in (tuple, list):
        return tuple(_refusal_value(item, depth=depth + 1) for item in value)
    try:
        return repr(value)[:256]
    except Exception:
        return f"<{type(value).__name__} with a failing repr>"


def _number(value: object, field: str, minimum: float, *, inclusive: bool = False) -> float:
    valid = isinstance(value, Real) and not isinstance(value, bool)
    try:
        number = float(value) if valid else math.nan
    except (OverflowError, TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number) or (number < minimum if inclusive else number <= minimum):
        _raise(
            "sensitivity.input.number",
            field=field,
            domain=f"[{minimum}, infinity)" if inclusive else f"({minimum}, infinity)",
            value=_refusal_value(value),
        )
    return number


@dataclass(frozen=True, kw_only=True)
class CohortRiskRatio:
    """A supplied unadjusted cohort estimate and its declared causal target.

    Parameters
    ----------
    risk_ratio : float
        Observed exposed/unexposed risk ratio for a binary outcome.
    population, exposure, outcome : str
        Identities of the target population, binary exposure and binary outcome.
    confidence_interval : tuple of float, optional
        Positive finite lower/upper bounds containing the estimate. Omit when
        uncertainty is unavailable; it is never reconstructed.
    confidence_level : float
        Coverage level of the supplied fixed-horizon interval, default 0.95.
    estimator, estimand, effect_scale, inference : str
        Declarations checked against the supported construction. An observed
        association is not by itself an identified population causal effect.
    adjustment_covariates : tuple of str
        Must be empty for this unadjusted construction.
    """

    risk_ratio: float
    population: str
    exposure: str
    outcome: str
    confidence_interval: tuple[float, float] | None = None
    confidence_level: float = 0.95
    estimator: Literal["cohort_risk_ratio"] = "cohort_risk_ratio"
    estimand: Literal["population_risk_ratio"] = "population_risk_ratio"
    effect_scale: Literal["risk_ratio"] = "risk_ratio"
    inference: Literal["fixed_horizon"] = "fixed_horizon"
    adjustment_covariates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field, supported in (
            ("estimator", "cohort_risk_ratio"),
            ("estimand", "population_risk_ratio"),
            ("effect_scale", "risk_ratio"),
            ("inference", "fixed_horizon"),
        ):
            value = getattr(self, field)
            if not isinstance(value, str):
                _raise(
                    "sensitivity.study.unsupported",
                    field=field,
                    value=_refusal_value(value),
                    supported=supported,
                )
            value = str.__str__(value)
            if value != supported:
                _raise(
                    "sensitivity.study.unsupported",
                    field=field,
                    value=_refusal_value(value),
                    supported=supported,
                )
            object.__setattr__(self, field, value)
        if type(self.adjustment_covariates) is not tuple or self.adjustment_covariates:
            _raise("sensitivity.study.adjusted", value=_refusal_value(self.adjustment_covariates))
        for field in ("population", "exposure", "outcome"):
            value = getattr(self, field)
            if not isinstance(value, str):
                _raise("sensitivity.input.identity", field=field, value=_refusal_value(value))
            value = str.__str__(value)
            if not value.strip():
                _raise("sensitivity.input.identity", field=field, value=_refusal_value(value))
            object.__setattr__(self, field, value)
        rr = _number(self.risk_ratio, "risk_ratio", 0.0)
        level = _number(self.confidence_level, "confidence_level", 0.0)
        if level >= 1:
            _raise(
                "sensitivity.input.number", field="confidence_level", domain="(0, 1)", value=level
            )
        object.__setattr__(self, "risk_ratio", rr)
        object.__setattr__(self, "confidence_level", level)
        if self.confidence_interval is not None:
            interval = self.confidence_interval
            if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                _raise("sensitivity.input.interval", value=_refusal_value(interval))
            low = _number(interval[0], "confidence_interval.lower", 0.0)
            high = _number(interval[1], "confidence_interval.upper", 0.0)
            if not low <= rr <= high:
                _raise("sensitivity.input.interval", value=_refusal_value(interval))
            object.__setattr__(self, "confidence_interval", (low, high))


@dataclass(frozen=True, kw_only=True)
class RiskRatioSensitivity:
    """Receive-only tipping points and one-sided confounding bounds.

    ``study`` retains the original summary. ``point_evalue`` and
    ``confidence_evalue`` report equal-strength tipping points for the strict
    declared threshold decision. A value of one means that decision already
    fails. Missing uncertainty retains a numeric null and its exact reason.
    ``point_bound`` is a lower bound for ``above`` and an upper bound for
    ``below``, under the supplied strength limits. ``confidence_limit_bound``
    transforms the corresponding original interval endpoint; it is not a new
    causal interval. ``guarantee`` describes population algebra only.
    """

    study: CohortRiskRatio
    threshold: float
    direction: Literal["above", "below"]
    point_evalue: float
    confidence_evalue: float | None
    confidence_unavailable_reason: str | None
    exposure_confounder_rr: float
    confounder_outcome_rr: float
    bias_factor: float
    point_bound: float
    confidence_limit_bound: float | None
    assumptions: tuple[str, ...] = _ASSUMPTIONS
    guarantee: Literal["exact"] = "exact"
    reference: str = "Ding and VanderWeele (2016); VanderWeele and Ding (2017)"


def _finite_result(value: float, quantity: str) -> float:
    if not math.isfinite(value) or value <= 0:
        _raise("sensitivity.numerical.range", quantity=quantity)
    return value


def _tipping(value: float, threshold: float, direction: Literal["above", "below"]) -> float:
    if value <= threshold if direction == "above" else value >= threshold:
        return 1.0
    delta = (value - threshold) / threshold if direction == "above" else (threshold - value) / value
    ratio = _finite_result(1.0 + delta, "tipping ratio")
    # Direct differences retain neighboring floats; split the radical to avoid overflow.
    return _finite_result(ratio + math.sqrt(ratio) * math.sqrt(delta), "E-value")


def risk_ratio_sensitivity(
    study: CohortRiskRatio,
    *,
    threshold: float = 1.0,
    direction: Literal["above", "below"] | None = None,
    exposure_confounder_rr: float = 1.0,
    confounder_outcome_rr: float = 1.0,
) -> RiskRatioSensitivity:
    """Quantify confounding needed to overturn a cohort risk-ratio decision.

    Invalid inputs or unrepresentable results raise ``InvalidRequestError``.

    Parameters
    ----------
    study : CohortRiskRatio
        Declared unadjusted cohort estimate, target and optional uncertainty.
    threshold : float
        Decision threshold on the original risk-ratio scale, default 1.
        For above decisions it must be >= 1; for below decisions <= 1.
    direction : Literal["above", "below"] | None
        Strict decision to challenge. Defaults to the estimate's side of one
        (``above`` at one). Equality already fails the decision.
    exposure_confounder_rr : float
        Maximum prevalence ratio for hypothetical confounder categories between
        exposure arms (reverse the arms for a below decision), at least one.
    confounder_outcome_rr : float
        Maximum outcome risk ratio between hypothetical confounder categories
        within either exposure arm, at least one.

    Returns
    -------
    RiskRatioSensitivity
        Original summary, tipping E-values, scenario bounds and assumptions.
        Default strengths of one reproduce the no-confounding bounds.
    """
    if not isinstance(study, CohortRiskRatio):
        _raise("sensitivity.input.study", value_type=type(study).__name__)
    if direction is None:
        direction = "above" if study.risk_ratio >= 1 else "below"
    if not isinstance(direction, str):
        _raise("sensitivity.decision.direction", value=_refusal_value(direction))
    normalized_direction = str.__str__(direction)
    if normalized_direction not in ("above", "below"):
        _raise("sensitivity.decision.direction", value=_refusal_value(normalized_direction))
    direction = "above" if normalized_direction == "above" else "below"
    threshold = _number(threshold, "threshold", 0.0)
    if threshold < 1 if direction == "above" else threshold > 1:
        _raise("sensitivity.decision.threshold", threshold=threshold, direction=direction)
    a = _number(exposure_confounder_rr, "exposure_confounder_rr", 1.0, inclusive=True)
    b = _number(confounder_outcome_rr, "confounder_outcome_rr", 1.0, inclusive=True)
    small, large = sorted((a, b))
    # Algebraically ab/(a+b-1), without overflowing the product or sum.
    factor = small / (1.0 + (small - 1.0) / large)
    endpoint = None
    if study.confidence_interval is not None:
        endpoint = study.confidence_interval[0 if direction == "above" else 1]
    point_bound = study.risk_ratio / factor if direction == "above" else study.risk_ratio * factor
    confidence_bound = None
    if endpoint is not None:
        confidence_bound = _finite_result(
            endpoint / factor if direction == "above" else endpoint * factor,
            "confidence-limit bound",
        )
    return RiskRatioSensitivity(
        study=study,
        threshold=threshold,
        direction=direction,
        point_evalue=_tipping(study.risk_ratio, threshold, direction),
        confidence_evalue=None if endpoint is None else _tipping(endpoint, threshold, direction),
        confidence_unavailable_reason="confidence_interval_not_supplied"
        if endpoint is None
        else None,
        exposure_confounder_rr=a,
        confounder_outcome_rr=b,
        bias_factor=factor,
        point_bound=_finite_result(point_bound, "point bound"),
        confidence_limit_bound=confidence_bound,
    )
