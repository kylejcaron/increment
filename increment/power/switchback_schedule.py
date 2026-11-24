"""Prospective schedule-only refusal probabilities for switchback orders."""

from __future__ import annotations

import math
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, DecimalException, localcontext

from scipy.stats import binom

from increment._unit_cycle import unit_cycle_admission_rule
from increment.errors import InvalidRequestError
from increment.power._validation import _as_count
from increment.power.switchback import _check_n, _invalid
from increment.semantics.assignment import FixedCountOrder, SwitchbackAssignment


def _fixed_count_probability(k: int, m: int, draws: int) -> float:
    precision = math.ceil(k * math.log10(3)) + 30
    exponent_limit = max(999999, k * draws + precision)
    if exponent_limit > min(MAX_EMAX, -MIN_EMIN):
        _invalid("n", "The draw count exceeds the decimal exponent range.", value=draws)
    with localcontext(Context(prec=precision, Emin=-exponent_limit, Emax=exponent_limit)):
        denominator = Decimal(math.comb(k, m))
        probability = Decimal(0)
        for a in range(k - m + 1):
            choose_control = math.comb(k, a)
            for b in range(m + 1):
                if a == b == 0:
                    continue
                mass = (Decimal(math.comb(k - a - b, m - b)) / denominator) ** draws
                term = choose_control * math.comb(k - a, b) * mass
                probability += term if (a + b) % 2 else -term
        result = float(probability)
        if not math.isfinite(result) or not 0 <= result <= 1:
            _invalid("n", "The schedule refusal probability could not be resolved.", value=draws)
        return result


def switchback_schedule_refusal_probability(
    assignment: SwitchbackAssignment, n: int, *, cycles_per_unit: int = 1
) -> float:
    """Schedule-only refusal for n >= 2 units or blocks; N=n*cycles_per_unit at unit grain.

    For 3<=K<=512, sum (-1)^(a+b+1) C(K,a)C(K-a,b)[C(K-a-b,m-b)/C(K,m)]^N.
    Precision ceil(K log10(3))+30 gives <1e-15 relative error before float underflow;
    paired orders sum the refused binomial tails.
    """
    if not isinstance(assignment, SwitchbackAssignment):
        _invalid("baseline", "Provide a switchback assignment with its window.", field="assignment")
    n = _check_n(n)
    cycles = _as_count(cycles_per_unit)
    shared = assignment.sequence.independence_unit == "shared_block"
    if cycles is None or not 1 <= cycles <= 2**53 or (shared and cycles != 1):
        _invalid(
            "n",
            "Provide a positive integer cycle count, equal to one for a shared schedule.",
            field="cycles_per_unit",
            value=cycles_per_unit,
        )
    draws = _check_n(n * cycles)
    sequence = assignment.sequence
    if isinstance(sequence, FixedCountOrder):
        k, m = assignment.periods_per_cycle, sequence.treated_periods
        if not 3 <= k <= 512 or not 0 < m < k:
            _invalid(
                "baseline",
                "Provide 3 to 512 positions and an interior treated count.",
                field="periods_per_cycle/treated_periods",
                value=(k, m),
            )
        try:
            return _fixed_count_probability(k, m, draws)
        except (DecimalException, OverflowError):
            _invalid("n", "The draw count cannot be represented by this calculation.", value=draws)
    try:
        rule = unit_cycle_admission_rule(draws, sequence.probability_ct)
        minimum = max(rule.minimum_ct, 1) if shared else rule.minimum_ct
        maximum = min(rule.maximum_ct, draws - 1) if shared else rule.maximum_ct
        probability = float(
            binom.cdf(minimum - 1, draws, sequence.probability_ct)
            + binom.sf(maximum, draws, sequence.probability_ct)
        )
    except (InvalidRequestError, OverflowError, ValueError, TypeError):
        _invalid("n", "The draw count cannot be represented by this calculation.", value=draws)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        _invalid("n", "The schedule refusal probability could not be resolved.", value=draws)
    return probability


__all__ = ["switchback_schedule_refusal_probability"]
