"""Off-policy contrast calibration under a fixed logger, and refusal under adaptive ones.

Pre-registered design (fixed before any interval was inspected)
---------------------------------------------------------------
Mechanism
    ``simulate_logged_run(logger, n_units=n, horizon=T, seed=..., epsilon=0.2,
    batch_size=50, reward_effect=0.1, context_dependence=0.8)``. Every logger
    draws rewards from the one reward model with mean ``base[x] + sign[x] *
    0.1 * 1[action == "B"]``, ``base = (0.30, 0.50)``, ``sign = (+1, -1)``;
    ``x_1 ~ Bernoulli(0.5)`` and ``P(x_t = 1 | a_{t-1}) = clip(0.5 + 0.4 *
    1[a_{t-1} == "B"], 0.05, 0.95)``, so later contexts depend on earlier
    actions and the cumulative-product weight is genuinely required from
    ``t = 3`` on. Loggers: ``fixed_random`` (0.5/0.5 always, one policy
    version), ``epsilon_greedy``, ``thompson`` and ``contextual_thompson``
    (batch-refit adaptive loggers registering a new version per batch).
Truth
    The finite-horizon dynamic policy-value contrast of ``TARGET_POLICY_V1``
    (``P(B | x=0) = 0.8``, ``P(B | x=1) = 0.2``) over ``REFERENCE_POLICY_V1``
    (``P(B) = 0.5``), computed exactly by the two-state Markov recursion in
    ``true_policy_value(policy, run.reward_model, horizon=T)``: 0.03 at T=1
    and T=2, 0.0275255040 at T=5 (context dependence 0.8).
Contract under test
    The estimator admits one fixed logging law per trace. Fixed-logger cells
    (T in {1, 2} x n in {50, 200, 1000} and T=5 x n in {100, 200, 1000}, the
    five-step cell at 50 units sitting entirely below the ESS floor) measure
    coverage of the truth by the unit-clustered sandwich interval over
    emitted intervals and assert it inside the k=3 binomial band around 0.95.
    Coverage MCSE uses the emitted-interval count, not all replications; the
    selected-cell denominator makes this a conditional coverage result.
    An ESS-floor refusal is counted, never pinned. Adaptive-logger cells
    (three loggers x T in {1, 2, 5} x n in
    {50, 200, 1000}, minus the one-batch cell n=50, T=1 whose logger never
    refits and so logs one fixed version) assert every replication is refused: by
    ``logged_policy.inference.adaptive_logging_unsupported`` once a trace with
    more than one logging version reaches the estimator, or earlier by
    ``logged_policy.trace.propensity_below_floor`` when the logger itself
    drove a propensity under the design-time floor. A separate fixed-logger
    study at T=5, n=1000 measures the bias of a current-step-propensity
    comparator against the shipped cumulative product (MCSE = SD / sqrt(reps)).
Replications
    800 (n=50), 600 (n=100), 400 (n=200), 160 (n=1000) for the fixed logger;
    eight per adaptive cell, since the refusal is a deterministic property of
    the trace.
Seeds
    Cell seed ``6_000_000 + 100_000 * logger_index + 10_000 * T + n``; each
    replication builds its trace with ``cell_seed + rep``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from tests.mc import Coverage, mcse, nominal_band

if TYPE_CHECKING:
    from increment.simulate.bandit_loggers import Logger

ALPHA = 0.05
REWARD_EFFECT = 0.1
CONTEXT_DEPENDENCE = 0.8
EPSILON = 0.2
BATCH_SIZE = 50
LOGGERS: tuple[Logger, ...] = ("fixed_random", "epsilon_greedy", "thompson", "contextual_thompson")
ADAPTIVE_LOGGERS: tuple[Logger, ...] = LOGGERS[1:]
HORIZONS = (1, 2, 5)
UNITS = (50, 200, 1000)
REPS_BY_N = {50: 800, 100: 600, 200: 400, 1000: 160}
ADAPTIVE_REPS = 8
ADAPTIVE_CODES = frozenset(
    {
        "logged_policy.inference.adaptive_logging_unsupported",
        "logged_policy.trace.propensity_below_floor",
    }
)

type Cell = tuple[Logger, int, int]
FIXED_GRID: tuple[Cell, ...] = tuple(
    ("fixed_random", t, n) for t in HORIZONS for n in ((100, 200, 1000) if t == 5 else UNITS)
)
# An adaptive logger refits after every BATCH_SIZE decisions, so a trace with
# no more decisions than one batch is logged under a single version.
ADAPTIVE_GRID: tuple[Cell, ...] = tuple(
    (lg, t, n) for lg in ADAPTIVE_LOGGERS for t in HORIZONS for n in UNITS if n * t > BATCH_SIZE
)


def cell_seed(logger: Logger, horizon: int, n_units: int) -> int:
    return 6_000_000 + 100_000 * LOGGERS.index(logger) + 10_000 * horizon + n_units


def _arrays(trace, target, reference) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(target ratio, reference ratio, reward)`` as ``(n_units, horizon)``
    arrays from the trace records; ratio = policy probability / propensity."""
    index = {u: i for i, u in enumerate(trace.unit_ids)}
    n, horizon = trace.n_units, trace.horizon
    ratio_t = np.empty((n, horizon))
    ratio_0 = np.empty((n, horizon))
    reward = np.empty((n, horizon))
    for r in trace.records:
        i, t = index[r.unit_id], r.decision_index - 1
        ratio_t[i, t] = target.probability(r.chosen_action, r.pre_decision_context) / r.propensity
        ratio_0[i, t] = (
            reference.probability(r.chosen_action, r.pre_decision_context) / r.propensity
        )
        reward[i, t] = r.reward
    return ratio_t, ratio_0, reward


def hajek_contrast(trace, target, reference, *, cumulative: bool) -> tuple[float, float]:
    """``(estimate, se)``: the trajectory Hájek contrast with one
    normalization per decision time and a unit-clustered sandwich.
    ``cumulative=False`` swaps the cumulative likelihood ratio for the
    current-step ratio -- the comparator production must never expose."""
    ratio_t, ratio_0, reward = _arrays(trace, target, reference)
    n = reward.shape[0]
    w_t = np.cumprod(ratio_t, axis=1) if cumulative else ratio_t
    w_0 = np.cumprod(ratio_0, axis=1) if cumulative else ratio_0
    v_t = (w_t * reward).sum(axis=0) / w_t.sum(axis=0)
    v_0 = (w_0 * reward).sum(axis=0) / w_0.sum(axis=0)
    phi = (
        w_t * (reward - v_t) / (w_t.sum(axis=0) / n) - w_0 * (reward - v_0) / (w_0.sum(axis=0) / n)
    ).mean(axis=1)
    return float(v_t.mean() - v_0.mean()), float(math.sqrt(np.sum(phi**2) / (n * (n - 1))))


def _truth(run, horizon: int) -> float:
    from increment.logged_policy import REFERENCE_POLICY_V1, TARGET_POLICY_V1
    from increment.simulate.bandit_loggers import true_policy_value

    return true_policy_value(
        TARGET_POLICY_V1, run.reward_model, horizon=horizon
    ) - true_policy_value(REFERENCE_POLICY_V1, run.reward_model, horizon=horizon)


def _one_replication(
    logger: Logger, horizon: int, n_units: int, seed: int, context_dependence: float
):
    """``(run, contrast, refusal code)``; exactly one of contrast / code is set."""
    from increment.errors import CodedError
    from increment.logged_policy import (
        REFERENCE_POLICY_V1,
        TARGET_POLICY_V1,
        estimate_policy_contrast,
    )
    from increment.simulate.bandit_loggers import simulate_logged_run

    try:
        run = simulate_logged_run(
            logger,
            n_units=n_units,
            horizon=horizon,
            seed=seed,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            reward_effect=REWARD_EFFECT,
            context_dependence=context_dependence,
        )
    except CodedError as exc:
        assert exc.code == "logged_policy.trace.propensity_below_floor", exc.code
        return None, None, exc.code
    try:
        contrast = estimate_policy_contrast(
            run.trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1, alpha=ALPHA
        )
    except CodedError as exc:
        assert exc.code in (
            "logged_policy.support.ess_floor",
            "logged_policy.inference.cluster_floor",
            "logged_policy.inference.adaptive_logging_unsupported",
        ), exc.code
        return run, None, exc.code
    return run, contrast, None


def study(cell: Cell, reps: int, seed: int) -> dict[str, Any]:
    from increment.logged_policy import REFERENCE_POLICY_V1, TARGET_POLICY_V1

    logger, horizon, n_units = cell
    coverage = Coverage()
    codes: dict[str, int] = {}
    max_parity_gap = 0.0
    truth = None
    for rep in range(reps):
        run, contrast, code = _one_replication(
            logger, horizon, n_units, seed + rep, CONTEXT_DEPENDENCE
        )
        if code is not None:
            codes[code] = codes.get(code, 0) + 1
            continue
        if truth is None:
            truth = _truth(run, horizon)
        assert contrast.horizon == horizon and contrast.n_units == n_units
        assert len(contrast.logging_policy_versions) == 1, contrast.logging_policy_versions
        estimate, _se = hajek_contrast(
            run.trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1, cumulative=True
        )
        max_parity_gap = max(max_parity_gap, abs(estimate - contrast.estimate))
        coverage.record(contrast.lb <= truth <= contrast.ub)
    return {
        "truth": truth,
        "emitted": coverage.reps,
        "coverage": coverage.rate if coverage.reps else None,
        "codes": codes,
        "parity_gap": max_parity_gap,
    }


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("cell", FIXED_GRID, ids=lambda c: f"{c[0]}-T{c[1]}-n{c[2]}")
def test_fixed_logger_contrast_interval_is_nominal(cell: Cell):
    logger, horizon, n_units = cell
    reps = REPS_BY_N[n_units]
    result = study(cell, reps, cell_seed(*cell))
    assert result["parity_gap"] <= 1e-9, (cell, result["parity_gap"])
    # A fixed 0.5/0.5 logger never breaches the propensity floor and every
    # cell has more than twenty units; only the ESS floor can screen a trace.
    assert set(result["codes"]) <= {"logged_policy.support.ess_floor"}, result["codes"]
    emitted = int(result["emitted"])
    if horizon == 1 or n_units >= 200:
        # A one-step trace has no product to concentrate; at n >= 200 the
        # five-step product keeps every replication above the ESS floor.
        assert result["codes"] == {}, result["codes"]
    assert emitted >= 100, (cell, "too few emitted intervals", emitted, result["codes"])
    lo, hi = nominal_band(0.95, emitted, k=3.0)
    assert lo <= result["coverage"] <= hi, (cell, result["coverage"], lo, hi, emitted)


@pytest.mark.slow
@pytest.mark.parametrize("cell", ADAPTIVE_GRID, ids=lambda c: f"{c[0]}-T{c[1]}-n{c[2]}")
def test_adaptive_logger_traces_are_refused(cell: Cell):
    result = study(cell, ADAPTIVE_REPS, cell_seed(*cell))
    assert result["emitted"] == 0, result
    assert sum(result["codes"].values()) == ADAPTIVE_REPS
    assert set(result["codes"]) <= ADAPTIVE_CODES, result["codes"]


@pytest.mark.parametrize(
    "code",
    (
        "logged_policy.support.policy_not_normalized",
        "logged_policy.support.context_unsupported",
        "logged_policy.support.action_unsupported",
    ),
)
def test_study_rejects_unexpected_support_refusals(monkeypatch, code):
    def refuse(*args, **kwargs):
        from increment.errors import CodedError

        raise CodedError("unexpected refusal", code=code, context={})

    monkeypatch.setattr("increment.logged_policy.estimate_policy_contrast", refuse)
    with pytest.raises(AssertionError):
        study(("fixed_random", 1, 50), 1, 0)


def test_study_counts_refusals_and_covers_only_emitted_samples(monkeypatch):
    from types import SimpleNamespace

    codes = (
        "logged_policy.support.ess_floor",
        "logged_policy.inference.adaptive_logging_unsupported",
        "logged_policy.trace.propensity_below_floor",
    )

    def replication(logger, horizon, n_units, seed, context_dependence):
        if seed >= 4:
            return None, None, codes[seed - 4]
        contrast = SimpleNamespace(
            horizon=horizon,
            n_units=n_units,
            logging_policy_versions=("fixed-random/v1",),
            estimate=0.0,
            lb=-0.1 if seed < 3 else 0.5,
            ub=0.1 if seed < 3 else 0.6,
        )
        return SimpleNamespace(trace=None), contrast, None

    monkeypatch.setattr(__name__ + "._one_replication", replication)
    monkeypatch.setattr(__name__ + "._truth", lambda *args: 0.0)
    monkeypatch.setattr(__name__ + ".hajek_contrast", lambda *args, **kwargs: (0.0, 0.0))
    result = study(("fixed_random", 1, 50), 4 + len(codes), 0)
    assert result["emitted"] == 4
    assert result["coverage"] == 0.75
    assert result["codes"] == dict.fromkeys(codes, 1)


# (current-step bias, cumulative-product bias) at fixed_random, T=5, n=1000
# for context dependence 0.0 and 0.8, measured when this study was written.
PINNED_COMPARATOR: dict[float, tuple[float, float]] = {
    0.0: (-0.00005, 0.00061),
    0.8: (0.00264, 0.00057),
}


def _comparator(context_dependence: float, reps: int) -> dict[str, float]:
    from increment.logged_policy import REFERENCE_POLICY_V1, TARGET_POLICY_V1

    seed = cell_seed("fixed_random", 5, 1000) + 500_000 + int(round(10 * context_dependence))
    truth = None
    cumulative: list[float] = []
    current_step: list[float] = []
    for rep in range(reps):
        run, contrast, code = _one_replication(
            "fixed_random", 5, 1000, seed + rep, context_dependence
        )
        assert code is None, code
        if truth is None:
            truth = _truth(run, 5)
        est_cum, _ = hajek_contrast(
            run.trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1, cumulative=True
        )
        est_cur, _ = hajek_contrast(
            run.trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1, cumulative=False
        )
        assert abs(est_cum - contrast.estimate) <= 1e-9
        cumulative.append(est_cum - truth)
        current_step.append(est_cur - truth)
    assert truth is not None
    return {
        "truth": truth,
        "cumulative_bias": float(np.mean(cumulative)),
        "cumulative_se": float(np.std(cumulative, ddof=1) / math.sqrt(reps)),
        "current_step_bias": float(np.mean(current_step)),
        "current_step_se": float(np.std(current_step, ddof=1) / math.sqrt(reps)),
    }


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("context_dependence", (0.0, 0.8))
def test_current_step_propensity_is_biased_only_under_context_dependence(context_dependence: float):
    """The cumulative product is what makes the estimator target the dynamic
    policy value: with history-independent contexts both weightings are
    unbiased (within 3 MCSE of zero); with context dependence 0.8 the
    current-step weighting is biased by more than 3 MCSE (the recursion puts
    its target at 0.03 against a truth of 0.0275) while the cumulative
    product stays within 3 MCSE of zero."""
    result = _comparator(context_dependence, REPS_BY_N[1000])
    assert abs(result["cumulative_bias"]) <= 3 * result["cumulative_se"], result
    if context_dependence == 0.0:
        assert result["truth"] == pytest.approx(0.03)
        assert abs(result["current_step_bias"]) <= 3 * result["current_step_se"], result
    else:
        assert result["truth"] == pytest.approx(0.0275255040)
        assert result["current_step_bias"] > 3 * result["current_step_se"], result
    current_pin, cumulative_pin = PINNED_COMPARATOR[context_dependence]
    assert abs(result["current_step_bias"] - current_pin) <= 3 * result["current_step_se"] + 5e-6
    assert abs(result["cumulative_bias"] - cumulative_pin) <= 3 * result["cumulative_se"] + 5e-6


def test_logged_policy_adaptive_smoke():
    """Fast twin: twelve replications of the fixed logger at T=2, n=50 keep
    the pipeline honest -- nothing refused and coverage inside the nominal
    k=3 band -- one adaptive trace per logger is refused by code, and a
    Thompson logger that never refit inside the trace (one batch of fifty
    decisions under its uniform first posterior) is admitted as the single
    fixed version it logged under."""
    cell: Cell = ("fixed_random", 2, 50)
    reps = 12
    result = study(cell, reps, cell_seed(*cell) + 900_000)
    assert result["codes"] == {}, result["codes"]
    assert result["coverage"] >= 0.95 - 3 * mcse(0.95, reps), result["coverage"]
    for logger in ADAPTIVE_LOGGERS:
        refused = study((logger, 2, 50), 1, cell_seed(logger, 2, 50) + 900_000)
        assert refused["emitted"] == 0 and set(refused["codes"]) <= ADAPTIVE_CODES, refused
    single_batch = study(("thompson", 1, BATCH_SIZE), 1, cell_seed("thompson", 1, BATCH_SIZE))
    assert single_batch["emitted"] == 1 and single_batch["codes"] == {}, single_batch
