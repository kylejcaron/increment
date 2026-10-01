"""Shared admission and prospective residual-envelope arithmetic.

Only mathematical design inputs enter these cached calculations. No observed
variance is used to create or replace a scientific envelope.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Context, Decimal, DecimalException, localcontext
from fractions import Fraction
from functools import lru_cache
from typing import NoReturn

from scipy.stats import binomtest

from increment._literals import ALTERNATIVE_VALUES
from increment.errors import InvalidRequestError, RefusalSpec, refuse
from increment.semantics.unit_cycle import UnitCycleVarianceEnvelope

_REFUSALS = {
    code: RefusalSpec(code, InvalidRequestError, lambda *, reason, **_: reason)
    for code in (
        "unit_cycle.invalid_design",
        "unit_cycle.envelope_required",
        "unit_cycle.reference_mismatch",
        "unit_cycle.error_budget_exhausted",
        "unit_cycle.numerical",
        "unit_cycle.implausible_realized_split",
        "unit_cycle.sufficient_state",
    )
}


def _refuse(code: str, *, reason: str, **context: object) -> NoReturn:
    refuse(_REFUSALS[code], reason=reason, **context)


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _refuse("unit_cycle.invalid_design", reason=f"{name} must be a positive integer")


def _float_outward(value: Fraction, *, upper: bool) -> float:
    """Round an exact rational in the requested direction, including subnormals."""
    try:
        result = float(value)
    except OverflowError:
        _refuse("unit_cycle.numerical", reason="finite_float_unrepresentable")
    if not math.isfinite(result):
        _refuse("unit_cycle.numerical", reason="finite_float_unrepresentable")
    represented = Fraction(result)
    if (upper and represented < value) or (not upper and represented > value):
        result = math.nextafter(result, math.inf if upper else -math.inf)
    if not math.isfinite(result):
        _refuse("unit_cycle.numerical", reason="outward_float_unrepresentable")
    return result


FINITE_MAX = Fraction(sys.float_info.max)
NEAREST_OVERFLOW = FINITE_MAX + 2**970


def unit_cycle_reporting_bounds(
    slope: Fraction, cutoff: float, alternative: str
) -> tuple[tuple[Fraction, bool], tuple[Fraction, bool]]:
    """Exact interval for the HT mean whose point and required endpoints convert.

    Nearest conversion has range (-B,B); downward has [-M,B), upward (-B,M].
    Each endpoint carries its inclusion flag.
    """
    c = Fraction(cutoff)
    lowers = [(-NEAREST_OVERFLOW, False)]
    uppers = [(NEAREST_OVERFLOW, False)]
    if alternative != "less":
        lowers.append((c - FINITE_MAX * slope, True))
        uppers.append((c + NEAREST_OVERFLOW * slope, False))
    if alternative != "greater":
        lowers.append((-c - NEAREST_OVERFLOW * slope, False))
        uppers.append((FINITE_MAX * slope - c, True))
    low = max(value for value, _ in lowers)
    high = min(value for value, _ in uppers)
    return (
        (low, all(closed for value, closed in lowers if value == low)),
        (high, all(closed for value, closed in uppers if value == high)),
    )


def unit_cycle_report(
    point: Fraction, slope: Fraction, cutoff: float, alternative: str
) -> tuple[float, float | None, float | None, float]:
    """Convert the required report with the same rounding and reasons everywhere."""
    if not -NEAREST_OVERFLOW < point < NEAREST_OVERFLOW:
        _refuse("unit_cycle.numerical", reason="point_unrepresentable")
    c = Fraction(cutoff)
    lower = None if alternative == "less" else _float_outward((point - c) / slope, upper=False)
    upper = None if alternative == "greater" else _float_outward((point + c) / slope, upper=True)
    if not 0 < slope < NEAREST_OVERFLOW or float(slope) == 0:
        _refuse("unit_cycle.numerical", reason="slope_unrepresentable")
    return float(point), lower, upper, float(slope)


def _sqrt_upper(value: Fraction) -> float:
    if not value:
        return 0.0
    # Decimal supplies only an initial guess; exact squared comparisons certify it.
    try:
        with localcontext(Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)):
            candidate = float((Decimal(value.numerator) / Decimal(value.denominator)).sqrt())
    except (DecimalException, OverflowError):
        _refuse("unit_cycle.numerical", reason="cutoff_unrepresentable")
    if not math.isfinite(candidate):
        _refuse("unit_cycle.numerical", reason="cutoff_unrepresentable")
    while Fraction(candidate) ** 2 < value:
        candidate = math.nextafter(candidate, math.inf)
        if not math.isfinite(candidate):
            _refuse("unit_cycle.numerical", reason="cutoff_unrepresentable")
    while candidate > 0:
        previous = math.nextafter(candidate, 0.0)
        if Fraction(previous) ** 2 < value:
            break
        candidate = previous
    return candidate


@dataclass(frozen=True, slots=True)
class UnitCycleAdmissionRule:
    minimum_ct: int
    maximum_ct: int
    refusal_probability_upper: float


@lru_cache(maxsize=4096, typed=True)
def unit_cycle_admission_outcome(n_cycles: int, probability_ct: float, ct_cycles: int) -> bool:
    """Verify the realized binomial decision against the cached interval mask."""
    rule = unit_cycle_admission_rule(n_cycles, probability_ct)
    try:
        value = float(binomtest(ct_cycles, n_cycles, probability_ct).pvalue)
    except (OverflowError, ValueError, TypeError) as exc:
        _refuse(
            "unit_cycle.numerical", reason="binomial_admission_unrepresentable", detail=str(exc)
        )
    if not math.isfinite(value):
        _refuse("unit_cycle.numerical", reason="binomial_admission_nonfinite")
    admitted = value >= 1e-6
    if admitted != (rule.minimum_ct <= ct_cycles <= rule.maximum_ct):
        _refuse("unit_cycle.numerical", reason="binomial_mask_not_interval")
    return admitted


def _chernoff_tail(n: int, k: int, p: float) -> Fraction:
    """Bound the tail ending at k on its side of n*p by exp(-n KL(k/n,p)).

    Decimal ln/exp are correctly rounded. Adjacent decimal values bracket
    their exact values; directed additions/products propagate the upper bound.
    """
    with localcontext(Context(prec=80, rounding=ROUND_CEILING, Emin=-999999, Emax=999999)) as ctx:
        pd = Decimal.from_float(p)
        ctx.prec = max(80, len(str(n)) + 32, -int(pd.as_tuple().exponent) + 4)
        ctx.rounding = ROUND_CEILING
        qd = Decimal(1) - pd  # Exact with the precision chosen above.
        nd = Decimal(n)
        # Signed entropy terms avoid division before taking logarithms.
        terms = [(n, nd), (k, pd), (n - k, qd), (-k, Decimal(k)), (-(n - k), Decimal(n - k))]
        log_upper = Decimal(0)
        for coefficient, argument in terms:
            if coefficient == 0:
                continue
            rounded = argument.ln()
            bound = rounded.next_plus() if coefficient > 0 else rounded.next_minus()
            log_upper += Decimal(coefficient) * bound
        if log_upper >= 0:
            return Fraction(1)
        # Float output cannot be below its smallest positive subnormal.
        if log_upper < -800:
            return Fraction(math.ulp(0.0))
        upper = log_upper.exp().next_plus()
        return min(Fraction(1), Fraction(upper))


@lru_cache(maxsize=4096, typed=True)
def unit_cycle_admission_rule(n_cycles: int, probability_ct: float) -> UnitCycleAdmissionRule:
    """Inclusive accepted CT counts for binomtest(k,n,p).pvalue >= 1e-6.

    Binomial masses increase then decrease, so the two-sided probability-order
    acceptance set is an interval containing the mode. Search both tails.
    """
    _positive_int(n_cycles, "n_cycles")
    if not math.isfinite(probability_ct) or not 0 < probability_ct < 1:
        _refuse("unit_cycle.invalid_design", reason="probability_ct must be in (0,1)")
    n = n_cycles
    mode = min(n, int((n + 1) * Fraction(probability_ct)))

    def accepted(k: int) -> bool:
        try:
            value = float(binomtest(k, n, probability_ct).pvalue)
        except (OverflowError, ValueError, TypeError) as exc:
            _refuse(
                "unit_cycle.numerical", reason="binomial_admission_unrepresentable", detail=str(exc)
            )
        if not math.isfinite(value):
            _refuse("unit_cycle.numerical", reason="binomial_admission_nonfinite")
        return value >= 1e-6

    if not accepted(mode):
        _refuse("unit_cycle.numerical", reason="binomial_mode_not_admitted")
    low, high = -1, mode
    while high - low > 1:
        middle = (low + high) // 2
        if accepted(middle):
            high = middle
        else:
            low = middle
    minimum = high
    low, high = mode, n + 1
    while high - low > 1:
        middle = (low + high) // 2
        if accepted(middle):
            low = middle
        else:
            high = middle
    maximum = low
    bound = Fraction(0)
    if minimum > 0:
        bound += _chernoff_tail(n, minimum - 1, probability_ct)
    if maximum < n:
        bound += _chernoff_tail(n, maximum + 1, probability_ct)
    return UnitCycleAdmissionRule(
        minimum, maximum, _float_outward(min(Fraction(1), bound), upper=True)
    )


@dataclass(frozen=True, slots=True)
class UnitCycleEnvelopeCutoff:
    value: float
    effective_alpha: float
    refusal_probability_upper: float


def unit_cycle_envelope_cutoff(
    envelope: UnitCycleVarianceEnvelope, *, n: int, alpha: float, alternative: str
) -> UnitCycleEnvelopeCutoff:
    """Outward Chebyshev/Cantelli cutoff after reserving admission error."""
    envelope = UnitCycleVarianceEnvelope.model_validate(envelope)
    _positive_int(n, "n")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        _refuse("unit_cycle.invalid_design", reason="alpha must be in (0,1)")
    if alternative not in ALTERNATIVE_VALUES:
        _refuse("unit_cycle.invalid_design", reason="unsupported alternative")
    rule = unit_cycle_admission_rule(
        n * envelope.cycles_per_unit, envelope.assignment.sequence.probability_ct
    )
    remaining = Fraction(alpha) - Fraction(rule.refusal_probability_upper)
    if remaining <= 0:
        _refuse(
            "unit_cycle.error_budget_exhausted",
            reason="admission_exhausts_alpha",
            alpha=alpha,
            refusal_probability_upper=rule.refusal_probability_upper,
        )
    effective = _float_outward(remaining, upper=False)
    if effective <= 0:
        _refuse("unit_cycle.numerical", reason="effective_alpha_underflow")
    squared = Fraction(envelope.residual_variance_upper) / n / Fraction(effective)
    if alternative != "two-sided":
        squared *= 1 - Fraction(effective)
    return UnitCycleEnvelopeCutoff(_sqrt_upper(squared), effective, rule.refusal_probability_upper)


__all__ = [
    "UnitCycleAdmissionRule",
    "UnitCycleEnvelopeCutoff",
    "unit_cycle_admission_rule",
    "unit_cycle_envelope_cutoff",
]
