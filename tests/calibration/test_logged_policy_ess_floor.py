"""Coverage of the logged-policy contrast interval by band of ``min_t ESS``, fixed logger.

Pre-registered design (fixed before any interval was inspected)
---------------------------------------------------------------
Mechanism
    ``simulate_logged_run("fixed_random", n_units=n, horizon=T, seed=...,
    context_dependence=0.8)``: the uniform 0.5/0.5 logger, one policy version,
    rewards and contexts as in ``test_logged_policy_adaptive.py``.
Target
    ``P(B | x=0) = p``, ``P(B | x=1) = 1 - p``, so each step multiplies the
    cumulative weight by ``2p`` or ``2(1-p)`` and the expected ``ESS_T / n``
    is about ``(2(p^2 + (1-p)^2))^-T``; ``p``, ``T`` and ``n`` together place
    a cell's realized ``min_t ESS`` in one band. The reference is the uniform
    ``REFERENCE_POLICY_V1`` (ESS ``n`` at every step).
Truth
    ``true_policy_value(target) - true_policy_value(reference)`` from the
    two-state Markov recursion.
Statistic
    Coverage of the truth by the two-sided 95% interval, pooled over cells
    into bands of ``min_t ESS`` (the same ``ess_by_time`` the result reports).
    Bands at and above the floor use the shipped estimator. Bands below the
    floor are refused by the shipped estimator (that is the floor's contract,
    asserted by code) and are measured with the inline trajectory Hajek
    comparator ``hajek_contrast`` -- the same arithmetic without the gate --
    so the reason for the floor stays on record.
Tolerance
    A band at or above the floor covers at least ``0.95 - 3 * MCSE`` over its
    emitted intervals (binomial MCSE ``sqrt(0.95 * 0.05 / m)``); the band well
    below the floor, [2,10), covers less than that. Every band needs at least
    200 intervals so the tolerance is not vacuous. The band immediately below
    the floor, [10,20), measured 0.933 over 2190 intervals in the derivation
    run: short of nominal by about two points, which is inside the 3 MCSE
    tolerance at any replication count this test's budget allows, so it is
    reported here rather than pinned.
Floor derivation
    The floor itself was derived from the larger run recorded beside
    ``ESS_FLOOR`` (26 cells, 9188 intervals): coverage 0.862 / 0.892 / 0.933 /
    0.933 / 0.943 / 0.947 / 0.949 in the bands [2,5), [5,10), [10,20), [20,40),
    [40,80), [80,160), [160,inf), and [20,40) is the smallest band from which
    every band stays within 3 MCSE of nominal.
Seeds
    ``7_000_000 + 1000 * round(1000 p) + 100 * T + n``, plus the replication.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from tests.calibration.test_logged_policy_adaptive import _arrays, hajek_contrast
from tests.mc import mcse

CONTEXT_DEPENDENCE = 0.8
# (p, T, n, reps): cells whose realized min_t ESS lands below, at and above the floor.
CELLS: tuple[tuple[float, int, int, int], ...] = (
    (0.9, 5, 50, 400),  # median ESS ~5: well below the floor, refused
    (0.8, 5, 100, 400),  # median ESS ~23: straddles the floor
    (0.9, 3, 100, 400),  # median ESS ~23
    (0.9, 3, 200, 300),  # median ESS ~45
)
BANDS = ((2.0, 10.0), (20.0, 40.0), (40.0, 80.0))


def _target(p: float):
    from increment.logged_policy import TabularPolicy

    return TabularPolicy(
        policy_id="diverging",
        version=f"p{p}",
        probabilities={0: {"A": 1 - p, "B": p}, 1: {"A": p, "B": 1 - p}},
    )


def _cell_seed(p: float, horizon: int, n_units: int) -> int:
    return 7_000_000 + 1000 * round(1000 * p) + 100 * horizon + n_units


def _min_ess(trace, target) -> float:
    """``min_t ESS_t`` of the target's cumulative weights, as the result reports it."""
    from increment.logged_policy import REFERENCE_POLICY_V1

    ratio, _, _ = _arrays(trace, target, REFERENCE_POLICY_V1)
    w = np.cumprod(ratio, axis=1)
    return float((w.sum(axis=0) ** 2 / (w * w).sum(axis=0)).min())


def run_cells(cells, *, reps_scale: float = 1.0) -> dict[str, list[tuple[float, bool]]]:
    """``{"shipped": [(min ESS, covered)], "unfloored": [...]}`` over every cell.

    ``shipped`` holds the estimator's own intervals, each checked against the
    comparator's estimate; ``unfloored`` holds the comparator's interval for
    every replication the ESS floor refused.
    """
    from scipy.stats import t as t_dist

    from increment.errors import CodedError
    from increment.logged_policy import (
        ESS_FLOOR,
        REFERENCE_POLICY_V1,
        estimate_policy_contrast,
    )
    from increment.simulate.bandit_loggers import simulate_logged_run, true_policy_value

    shipped: list[tuple[float, bool]] = []
    unfloored: list[tuple[float, bool]] = []
    for p, horizon, n_units, reps in cells:
        target = _target(p)
        seed = _cell_seed(p, horizon, n_units)
        truth = None
        for rep in range(max(1, round(reps * reps_scale))):
            run = simulate_logged_run(
                "fixed_random",
                n_units=n_units,
                horizon=horizon,
                seed=seed + rep,
                context_dependence=CONTEXT_DEPENDENCE,
            )
            if truth is None:
                truth = true_policy_value(
                    target, run.reward_model, horizon=horizon
                ) - true_policy_value(REFERENCE_POLICY_V1, run.reward_model, horizon=horizon)
            try:
                result = estimate_policy_contrast(run.trace, target, REFERENCE_POLICY_V1)
            except CodedError as exc:
                assert exc.code == "logged_policy.support.ess_floor", exc.code
                assert cast("float", exc.context["ess"]) < ESS_FLOOR
                ess = _min_ess(run.trace, target)
                assert ess < ESS_FLOOR
                estimate, se = hajek_contrast(
                    run.trace, target, REFERENCE_POLICY_V1, cumulative=True
                )
                half = float(t_dist.isf(0.025, n_units - 1)) * se
                unfloored.append((ess, estimate - half <= truth <= estimate + half))
                continue
            ess = min(result.ess_by_time)
            assert ess >= ESS_FLOOR
            estimate, _ = hajek_contrast(run.trace, target, REFERENCE_POLICY_V1, cumulative=True)
            assert abs(estimate - result.estimate) <= 1e-9
            shipped.append((ess, result.lb <= truth <= result.ub))
    return {"shipped": shipped, "unfloored": unfloored}


def band_coverage(rows: list[tuple[float, bool]], lo: float, hi: float) -> tuple[int, float]:
    inside = [covered for ess, covered in rows if lo <= ess < hi]
    return len(inside), (float(np.mean(inside)) if inside else math.nan)


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_coverage_is_nominal_from_the_ess_floor_up_and_short_just_below_it():
    from increment.logged_policy import ESS_FLOOR

    assert ESS_FLOOR == BANDS[1][0]
    rows = run_cells(CELLS)
    below, (lo_floor, hi_floor), above = BANDS
    # Every emitted interval sits at or above the floor; every refused one below.
    assert all(ess >= ESS_FLOOR for ess, _ in rows["shipped"])
    assert all(ess < ESS_FLOOR for ess, _ in rows["unfloored"])
    for lo, hi in ((lo_floor, hi_floor), above):
        m, rate = band_coverage(rows["shipped"], lo, hi)
        assert m >= 200, (lo, hi, m)
        assert rate >= 0.95 - 3 * mcse(0.95, m), ((lo, hi), rate, m)
    m, rate = band_coverage(rows["unfloored"], *below)
    assert m >= 200, (below, m)
    assert rate < 0.95 - 3 * mcse(0.95, m), (below, rate, m)


def test_ess_floor_study_smoke():
    """Fast twin: one replication per cell keeps the pipeline honest --
    refusals land below the floor, emitted intervals at or above it, and the
    comparator reproduces every shipped estimate."""
    from increment.logged_policy import ESS_FLOOR

    rows = run_cells(CELLS, reps_scale=0.0025)
    assert rows["shipped"] and rows["unfloored"]
    assert all(ess >= ESS_FLOOR for ess, _ in rows["shipped"])
    assert all(ess < ESS_FLOOR for ess, _ in rows["unfloored"])
