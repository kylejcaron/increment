"""Monte-Carlo evidence for the CUPED-adjusted LATE.

Two pinned DGPs with cov(X, D) != 0 within arm, making the delta method's
theta*cov_xd cross term load-bearing. Calibration DGP (X~N(10,2), true LATE
0.8) has CUPED reduce variance (SD ratio ~0.69). Backfire DGP (X~N(0,1),
true LATE 5.0, X predicts uptake but not Y) has CUPED cost variance (SD
ratio ~1.6); the estimator stays calibrated and flags the inflation.

Bound-asserting tests pin a seed and run >=4000 reps (coverage MC se
~0.0034); fast smoke variants cover small-N in the default suite.
"""

from __future__ import annotations

import pytest

from increment.estimation.armstats import centered_row_from_raw_sums
from increment.estimation.encouragement import estimate_encouragement
from increment.estimation.engine import Method
from increment.semantics.design import Encouragement
from increment.semantics.models import MeanMetric

METRIC = MeanMetric(name="rev", entity="user_id", fact="orders", aggregation="sum")
CUPED = [Method(name="cuped", variance_reduction="cuped")]
SEED = 20260811
REPS = 4000
_Z = 1.959963984540054

DESIGN = Encouragement.model_validate(
    {
        "mechanism": "encouragement",
        "control_group": "control",
        "uptake": {"fact": "help_click"},
        "exclusion_restriction": {
            "acknowledged": True,
            "justification": "unclicked button assumed inert",
        },
        # Two-sided: always-takers give the control arm real uptake, which is
        # what puts a nonzero cov(X, D) in BOTH arms.
        "one_sided": False,
    }
)


def _moment_row(gid, x, y, d):
    raw = {
        "experiment_id": "s",
        "metric": "rev",
        "group_id": gid,
        "n": int(x.size),
        "sum_y": float(y.sum()),
        "sum_y2": float((y * y).sum()),
        "sum_x": float(x.sum()),
        "sum_x2": float((x * x).sum()),
        "sum_xy": float((x * y).sum()),
        "sum_d": float(d.sum()),
        "sum_yd": float((y * d).sum()),
        "sum_y2d": float((y * y * d).sum()),
        "sum_xd": float((x * d).sum()),
    }
    return centered_row_from_raw_sums(raw)


def calibration_rows(n, rng):
    """X-dependent latent types; true LATE 0.8, homogeneous."""
    import numpy as np

    rows = []
    for gid, z in (("control", 0), ("treat", 1)):
        x = rng.normal(10.0, 2.0, size=n)
        p_at = np.clip(0.03 + 0.015 * (x - 10.0), 0.0, 1.0)
        p_c = np.clip(0.30 + 0.04 * (x - 10.0), 0.0, 1.0)
        u = rng.random(n)
        at = u < p_at
        co = (u >= p_at) & (u < p_at + p_c)
        d = (at | (co & bool(z))).astype(float)
        y = 1.0 + 0.5 * x + 0.6 * at + 0.2 * co + 0.8 * d + rng.normal(0, 1.0, size=n)
        rows.append(_moment_row(gid, x, y, d))
    return rows


def backfire_rows(n, rng):
    """X drives uptake, barely predicts Y net of uptake; true LATE 5.0."""
    import numpy as np

    rows = []
    for gid, z in (("control", 0), ("treat", 1)):
        x = rng.normal(0.0, 1.0, size=n)
        p_c = np.clip(0.5 + 0.3 * x, 0.0, 0.9)
        u = rng.random(n)
        at = u < 0.05
        co = (u >= 0.05) & (u < 0.05 + p_c)
        d = (at | (co & bool(z))).astype(float)
        y = 1.0 + 0.05 * x + 0.3 * at + 0.1 * co + 5.0 * d + rng.normal(0, 0.5, size=n)
        rows.append(_moment_row(gid, x, y, d))
    return rows


def _late_row(rows, *, cuped):
    (row,) = [
        r
        for r in estimate_encouragement(
            [METRIC],
            rows,
            DESIGN,
            estimands=("late",),
            methods=CUPED if cuped else None,
        ).results
        if r.estimand == "late" and r.value_scale == "absolute"
    ]
    # Default near-flat prior makes the conjugate posterior the raw
    # delta-method interval to ~1e-15 relative, so half-width recovers the SE.
    lift = row.require_lift()
    assert lift.lb is not None and lift.ub is not None
    return row, (lift.ub - lift.lb) / (2 * _Z)


_MEMO: dict[tuple, tuple] = {}


def _replicate(dgp, *, n, reps, seed, true_late):
    """(cuped stats, unadjusted stats, annotation trigger rate).

    Each stat block is (mean point, sd of points, mean SE, coverage).

    Memoized: several of the bounds below read the same 4000-rep run, and
    re-simulating it per test would multiply this suite's cost for no added
    evidence.
    """
    memo_key = (dgp.__name__, n, reps, seed, true_late)
    if memo_key in _MEMO:
        return _MEMO[memo_key]
    import numpy as np

    pts: dict[str, list[float]] = {"cuped": [], "raw": []}
    ses: dict[str, list[float]] = {"cuped": [], "raw": []}
    hits = {"cuped": 0, "raw": 0}
    triggered = 0
    for i in range(reps):
        rows = dgp(n, np.random.default_rng(seed + i))
        for arm_key, cuped in (("cuped", True), ("raw", False)):
            row, se = _late_row(rows, cuped=cuped)
            pts[arm_key].append(row.require_lift().value)
            ses[arm_key].append(se)
            hits[arm_key] += row.require_lift().lb <= true_late <= row.require_lift().ub
            if cuped and "INFLATED" in (row.note or ""):
                triggered += 1

    def _block(k):
        p, s = np.array(pts[k]), np.array(ses[k])
        return p.mean(), p.std(ddof=1), s.mean(), hits[k] / reps

    _MEMO[memo_key] = (_block("cuped"), _block("raw"), triggered / reps)
    return _MEMO[memo_key]


# 1. Parameter recovery, with and without cuped


def test_late_recovery_smoke():
    cuped, raw, trigger = _replicate(calibration_rows, n=2000, reps=25, seed=SEED, true_late=0.8)
    assert abs(cuped[0] - 0.8) < 0.15
    assert abs(raw[0] - 0.8) < 0.15
    assert cuped[1] < raw[1], "cuped must be the tighter estimator on this DGP"
    assert trigger == 0.0


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_late_unbiased_with_and_without_cuped():
    cuped, raw, _ = _replicate(calibration_rows, n=4000, reps=REPS, seed=SEED, true_late=0.8)
    assert abs(cuped[0] - 0.8) < 0.01
    assert abs(raw[0] - 0.8) < 0.01


# 3. SE calibration + coverage; the cross term is what buys them


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_se_calibration_and_coverage_on_the_cov_xd_dgp():
    cuped, _, _ = _replicate(calibration_rows, n=4000, reps=REPS, seed=SEED, true_late=0.8)
    mean_pt, sd, mean_se, coverage = cuped
    assert 0.95 <= mean_se / sd <= 1.05
    assert coverage >= 0.935


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_dropping_the_theta_cov_xd_term_undercovers():
    """The regression test that keeps ``cxd`` load-bearing: the naive
    covariance (``cov_yd`` alone, no ``theta * cov_xd`` correction) fails the
    coverage bound the correct form clears on the same reps and seed."""
    import numpy as np

    from increment.estimation.armstats import ArmStats
    from increment.estimation.cuped import cuped_adjust, pooled_theta
    from increment.estimation.encouragement import _first_stage

    hits = 0
    for i in range(REPS):
        rows = calibration_rows(4000, np.random.default_rng(SEED + i))
        c, t = (
            ArmStats(study_id="s", **{k: v for k, v in r.items() if k != "experiment_id"})
            for r in rows
        )
        pooled_theta([c, t])  # same refusals as the real path
        c_adj, t_adj = cuped_adjust([c, t])
        b, var_b = _first_stage(t, c)
        tau = (t_adj.mean - c_adj.mean) / b
        var_a = t_adj.var / t.n + c_adj.var / c.n
        naive_cov = t.cov_yd() / t.n + c.cov_yd() / c.n
        se = np.sqrt(max((var_a - 2 * tau * naive_cov + tau**2 * var_b) / b**2, 0.0))
        hits += abs(tau - 0.8) <= _Z * se
    assert hits / REPS < 0.943, "naive covariance must FAIL the coverage bound"


# 4. The realized variance reduction matches the moment prediction


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_realized_sd_ratio_matches_the_moment_predicted_ratio():
    cuped, raw, _ = _replicate(calibration_rows, n=4000, reps=REPS, seed=SEED, true_late=0.8)
    realized = cuped[1] / raw[1]
    predicted = cuped[2] / raw[2]
    assert realized == pytest.approx(predicted, abs=0.03)
    assert realized < 0.75, "CUPED must actually reduce variance on this DGP"


# 5. Small-n: theta-estimation error absorbed


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_small_n_calibration_absorbs_theta_estimation_error():
    """theta is treated as fixed in the delta method (the standard CUPED
    argument: its error is second order). At n=500 that approximation still
    holds - if it did not, the SE would run optimistic here first."""
    cuped, _, _ = _replicate(calibration_rows, n=500, reps=REPS, seed=SEED, true_late=0.8)
    _, sd, mean_se, coverage = cuped
    assert mean_se / sd >= 0.95
    assert coverage >= 0.935


# 9. The backfire regime is exercised, not merely documented


def test_backfire_smoke():
    cuped, raw, trigger = _replicate(backfire_rows, n=2000, reps=25, seed=SEED, true_late=5.0)
    assert cuped[1] > raw[1], "backfire DGP must inflate the cuped SD"
    assert trigger == 1.0


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_backfire_stays_calibrated_while_inflating_the_variance():
    cuped, raw, trigger = _replicate(backfire_rows, n=4000, reps=REPS, seed=SEED, true_late=5.0)
    mean_pt, sd, mean_se, coverage = cuped
    assert abs(mean_pt - 5.0) < 0.02
    assert 0.95 <= mean_se / sd <= 1.05, "the SE must honestly track the inflation"
    assert coverage >= 0.935
    assert sd / raw[1] > 1.3, "the inflation regime must actually be inflating"
    assert trigger >= 0.99


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_annotation_never_fires_when_cuped_is_helping():
    _, _, trigger = _replicate(calibration_rows, n=4000, reps=REPS, seed=SEED, true_late=0.8)
    assert trigger == 0.0
