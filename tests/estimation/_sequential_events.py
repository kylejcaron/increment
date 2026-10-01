"""Certified event screens for the bounded sequential certification evaluator.

A feasible-null likelihood supplies a lower bound on the composite-null
supremum. Therefore Q/L(feasible) is an upper bound on deployed evidence and
can prove noncrossing without solving the global null optimization.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import Literal

from increment.estimation._certified import Interval, log_interval, log_rising
from increment.estimation._sequential_likelihood import BernoulliState, BetaPrior

Alternative = Literal["two-sided", "greater", "less"]
F = Fraction


@dataclass(frozen=True, slots=True)
class NoncrossingScreen:
    """Outward log bounds used by a strict noncrossing decision."""

    log_e_upper: Fraction
    log_threshold_lower: Fraction

    @property
    def proves_noncrossing(self) -> bool:
        return self.log_e_upper < self.log_threshold_lower


@cache
def _predictive_log(state: BernoulliState, prior: BetaPrior, precision: int) -> Interval:
    if not state.n:
        return Interval.exact(0)
    return (
        log_rising(prior.a, state.successes, precision=precision)
        + log_rising(prior.b, state.n - state.successes, precision=precision)
        - log_rising(prior.a + prior.b, state.n, precision=precision)
    )


@cache
def _likelihood_log(
    state: BernoulliState, probability: Fraction, precision: int
) -> Interval | None:
    result = Interval.exact(0)
    for count, value in (
        (state.successes, probability),
        (state.n - state.successes, 1 - probability),
    ):
        if not count:
            continue
        if value == 0:
            return None
        result += count * log_interval(value, precision=precision)
    return result


def bernoulli_noncrossing_screen(
    control: BernoulliState,
    treatment: BernoulliState,
    prior_control: BetaPrior,
    prior_treatment: BetaPrior,
    *,
    ratio: Fraction,
    alternative: Alternative,
    feasible_control: Fraction,
    feasible_treatment: Fraction,
    alpha: Fraction,
    precision: int = 60,
) -> NoncrossingScreen:
    """Prove deployed Bernoulli evidence is below ``1 / alpha`` when possible.

    The caller supplies one predeclared or fitted point in the tested null. A
    false result is unresolved, never evidence of crossing.
    """

    ratio = F(ratio)
    feasible_control = F(feasible_control)
    feasible_treatment = F(feasible_treatment)
    alpha = F(alpha)
    if ratio < 0 or not 0 <= feasible_control <= 1 or not 0 <= feasible_treatment <= 1:
        raise ValueError("ratio and feasible probabilities must be in their valid ranges")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    feasible = (
        (alternative == "two-sided" and feasible_treatment == ratio * feasible_control)
        or (alternative == "greater" and feasible_treatment <= ratio * feasible_control)
        or (alternative == "less" and feasible_treatment >= ratio * feasible_control)
    )
    if not feasible:
        raise ValueError("supplied point does not satisfy the null constraint")

    log_likelihood_control = _likelihood_log(control, feasible_control, precision)
    log_likelihood_treatment = _likelihood_log(treatment, feasible_treatment, precision)
    if log_likelihood_control is None or log_likelihood_treatment is None:
        raise ValueError("supplied null point must have positive likelihood")
    log_predictive = _predictive_log(control, prior_control, precision) + _predictive_log(
        treatment, prior_treatment, precision
    )
    log_feasible = log_likelihood_control + log_likelihood_treatment
    return NoncrossingScreen(
        log_e_upper=log_predictive.hi - log_feasible.lo,
        log_threshold_lower=log_interval(1 / alpha, precision=precision).lo,
    )
