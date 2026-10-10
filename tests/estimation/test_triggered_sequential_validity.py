"""Validity sentinels for sequential monitoring of a growing triggered cohort.

The triggered chain reveals one fixed i.i.d. sequence in trigger order; a
calendar look evaluates the prefix of units whose trigger-anchored window has
closed. The null sentinel checks that the package's own count-clock set,
evaluated at every daily look of a growing cohort, keeps its any-look false
rejection rate below alpha. The stopping-policy test separates the per-cell
crossing bound, which no look rule can inflate, from the stopped e-value's
expectation that a family selection needs: a rule that peeks at information
outside the chain's own filtration inflates the latter while the former holds.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from increment.estimation._sequential_likelihood import (
    BernoulliState,
    BetaPrior,
    GaussianState,
    bernoulli_evidence,
)
from increment.estimation.asymptotic_mean import asymptotic_mean_set, mixture_r_star
from tests.asymptotic_cases import mean_model

_ALPHA = Fraction(1, 20)
_UNITS = 1200
_TRIGGER_RATE = 0.3
_WINDOW_DAYS = 3
_LOOKS = 30


def _declaration() -> object:
    """Boundary tuned for the planned triggered count, as the assigned route tunes its own."""
    planned = int(_UNITS * _TRIGGER_RATE)
    rho = Fraction(math.sqrt(float(mixture_r_star(_ALPHA) / planned)))
    return mean_model(rho=rho, start_count=2)


def _any_look_rejection(rng: np.random.Generator, declaration, *, effect: float = 0.0) -> bool:
    """One growing-cohort path evaluated at daily calendar looks.

    Exposure day uniform on 0-19, trigger delay 0-5 days, lognormal outcomes
    rounded to a fine grid so the exact kernels stay cheap; the effect applies
    to treated units only (zero under the null).
    """
    exposure = rng.integers(0, 20, size=_UNITS)
    treated = rng.random(_UNITS) < 0.5
    triggered = rng.random(_UNITS) < _TRIGGER_RATE
    trigger_day = exposure + rng.integers(0, 6, size=_UNITS)
    outcome = np.round(rng.lognormal(0.0, 0.6, size=_UNITS) * (1.0 + effect * treated), 3)
    members = np.flatnonzero(triggered)
    order = members[np.lexsort((members, trigger_day[members]))]
    final_day = trigger_day[order] + (_WINDOW_DAYS - 1)
    control = GaussianState.empty(1)
    treatment = GaussianState.empty(1)
    revealed = 0
    for look in range(_LOOKS):
        upto = int(np.searchsorted(final_day, look, side="right"))
        if upto == revealed:
            continue
        block = order[revealed:upto]
        revealed = upto
        for arm_state, mask in ((treatment, treated[block]), (control, ~treated[block])):
            rows = [(Fraction(str(outcome[unit])),) for unit in block[mask]]
            if not rows:
                continue
            merged = arm_state.merge(GaussianState.from_rows(rows, dimension=1))
            if arm_state is treatment:
                treatment = merged
            else:
                control = merged
        evaluated = asymptotic_mean_set(
            control,
            treatment,
            declaration=declaration,
            alpha=_ALPHA,
            null_lift=Fraction(0),
            alternative="two-sided",
        )
        if evaluated.rejects():
            return True
    return False


def _binomial_margin(paths: int, rate: float) -> float:
    return 3.0 * math.sqrt(rate * (1.0 - rate) / paths)


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_growing_triggered_cohort_null_any_look_rejection_stays_below_alpha():
    rng = np.random.default_rng(20261009)
    declaration = _declaration()
    paths = 1000
    rejections = sum(_any_look_rejection(rng, declaration) for _ in range(paths))
    rate = rejections / paths
    assert rate <= float(_ALPHA) + _binomial_margin(paths, float(_ALPHA)), rate
    powered = sum(_any_look_rejection(rng, declaration, effect=0.25) for _ in range(200))
    assert powered / 200 > 0.5


def test_growing_triggered_cohort_null_smoke():
    rng = np.random.default_rng(7)
    declaration = _declaration()
    rejections = sum(_any_look_rejection(rng, declaration) for _ in range(20))
    assert rejections <= 3


def _log_e(control: tuple[int, int], treatment: tuple[int, int], prior: BetaPrior) -> float:
    certificate = bernoulli_evidence(
        BernoulliState(*control),
        BernoulliState(*treatment),
        prior,
        prior,
        ratio=Fraction(1),
        alternative="two-sided",
    )
    if certificate.status == "finite":
        assert certificate.log_e is not None
        return float(certificate.log_e.lo)
    return math.inf if certificate.status == "infinite" else -math.inf


@pytest.mark.slow
def test_leak_informed_stopping_inflates_the_stopped_e_value_but_not_the_crossing_bound():
    """A look rule using information outside the chain's filtration breaks E[e_tau] <= 1
    while P(sup e >= 1/alpha) <= alpha stays intact, which is why the family verdict
    (not the per-cell threshold) carries the outcome-independent look-time condition.
    """
    rng = np.random.default_rng(7)
    prior = BetaPrior(Fraction(3), Fraction(7))
    block, blocks, rate, paths = 20, 20, 0.3, 400
    threshold = -math.log(float(_ALPHA))
    crossed = 0
    scheduled: list[float] = []
    leaked: list[float] = []
    for _ in range(paths):
        treated = rng.random(block * blocks) < 0.5
        success = rng.random(block * blocks) < rate
        control = treatment = (0, 0)
        logs = []
        for index in range(blocks):
            window = slice(index * block, (index + 1) * block)
            control = (
                control[0] + int((~treated[window]).sum()),
                control[1] + int((success[window] & ~treated[window]).sum()),
            )
            treatment = (
                treatment[0] + int(treated[window].sum()),
                treatment[1] + int((success[window] & treated[window]).sum()),
            )
            logs.append(_log_e(control, treatment, prior))
        path = np.array(logs)
        crossed += bool((path >= threshold).any())
        scheduled.append(math.exp(path[-1]))
        stop = blocks - 1
        for index in range(blocks - 1):
            if path[index + 1] < path[index]:
                stop = index
                break
        leaked.append(math.exp(path[stop]))
    assert crossed / paths <= float(_ALPHA) + _binomial_margin(paths, float(_ALPHA))
    difference = np.array(leaked) - 2.0 * np.array(scheduled)
    lower = difference.mean() - 3.0 * difference.std(ddof=1) / math.sqrt(paths)
    assert lower > 0.0, (np.mean(leaked), np.mean(scheduled))
    assert np.mean(scheduled) <= 1.0
