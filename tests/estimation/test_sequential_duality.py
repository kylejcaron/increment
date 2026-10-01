"""Two-sided always-valid decisions and confidence sequences reject the same null."""

from __future__ import annotations

import math
from fractions import Fraction as F

import pytest

from increment.estimation._certified import log_interval
from increment.estimation._sequential_inversion import bernoulli_confidence_sequence
from increment.estimation._sequential_likelihood import (
    BernoulliState,
    BetaPrior,
    bernoulli_evidence,
)

FLAT = BetaPrior(F(1), F(1))
ALPHA = F(1, 20)


def _rejects(control: BernoulliState, treatment: BernoulliState) -> bool:
    """The rule SequentialInferenceResult.rejects applies at the registered null ratio 1."""
    certificate = bernoulli_evidence(
        control, treatment, FLAT, FLAT, ratio=F(1), alternative="two-sided"
    )
    if certificate.status != "finite":
        return certificate.status == "infinite"
    assert certificate.log_e is not None
    return certificate.log_e.lo >= (-log_interval(ALPHA)).hi


def _excludes_null(control: BernoulliState, treatment: BernoulliState) -> bool:
    bounds = bernoulli_confidence_sequence(control, treatment, FLAT, FLAT, alpha=ALPHA)
    return (
        bounds.empty
        or (bounds.lower is not None and bounds.lower > 1)
        or (bounds.upper is not None and bounds.upper < 1)
    )


@pytest.mark.slow
def test_significant_two_sided_decision_excludes_the_null_from_its_interval():
    # 150/500 vs 215/500 has log evidence 3.288 >= log 20 at the equality null.
    control, treatment = BernoulliState(500, 150), BernoulliState(500, 215)
    assert _rejects(control, treatment)
    bounds = bernoulli_confidence_sequence(control, treatment, FLAT, FLAT, alpha=ALPHA)
    assert bounds.log_threshold == -log_interval(ALPHA)
    assert bounds.lower is not None and bounds.lower > 1
    assert _excludes_null(control, treatment)


@pytest.mark.slow
def test_nonsignificant_two_sided_decision_keeps_the_null_inside_its_interval():
    control, treatment = BernoulliState(500, 150), BernoulliState(500, 165)
    assert not _rejects(control, treatment)
    assert not _excludes_null(control, treatment)


def _null_crossings(reps: int, looks: int, *, batch: int = 100, rate: float = 0.3, seed: int):
    """Replications whose two-sided decision ever rejects, with the first crossing states."""
    import numpy as np

    rng = np.random.default_rng(seed)
    crossings, witnesses = 0, []
    for _ in range(reps):
        nc = nt = sc = st = 0
        for _ in range(looks):
            sc += int(rng.binomial(batch, rate))
            st += int(rng.binomial(batch, rate))
            nc += batch
            nt += batch
            control, treatment = BernoulliState(nc, sc), BernoulliState(nt, st)
            if _rejects(control, treatment):
                crossings += 1
                witnesses.append((control, treatment))
                break
    return crossings, witnesses


def _crossing_bound(reps: int) -> float:
    return 0.05 + 3 * math.sqrt(0.05 * 0.95 / reps)


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_null_two_sided_crossing_rate_is_within_alpha_and_duality_holds_at_crossings():
    reps, looks = 200, 10
    crossings, witnesses = _null_crossings(reps, looks, seed=20260922)
    assert crossings / reps <= _crossing_bound(reps)
    for control, treatment in witnesses[:3]:
        assert _excludes_null(control, treatment)


def test_null_two_sided_crossing_smoke():
    reps, looks = 4, 3
    crossings, witnesses = _null_crossings(reps, looks, seed=7)
    assert crossings / reps <= _crossing_bound(reps)
    for control, treatment in witnesses:
        assert _excludes_null(control, treatment)
