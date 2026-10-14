"""Checks for the issue #73 nonlinear-targeting evidence harness.

Fast tests pin the harness's own contracts: the boosted base regressor and
R-learner recover a known surface, splits never separate a unit's rows or a
cluster, nothing fitted reads evaluation rows, the joint subgroup audit matches
an independent covariance construction, and the recommendation rule applies
its declared thresholds. The slow sentinel reruns a small campaign and checks
recovery, false-targeting and coverage with wide Monte Carlo bands; the full
campaign is ``python -m calibration.nonlinear_targeting``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm

from calibration._rlearner import BoostedRegressor, RLearner
from calibration.nonlinear_targeting import (
    ALPHA,
    AUDIT_GROUPS,
    MATERIAL_GAP_SHARE,
    SCENARIOS,
    Draw,
    Reference,
    assert_no_leakage,
    audit_coverage,
    audit_groups,
    correction_cell,
    design_matrix,
    fit_cell_correction,
    fit_learners,
    heldout_score,
    joint_mean_cov,
    max_abs_critical,
    recommend,
    run_campaign,
    simulate,
    three_way_split,
)
from increment.errors import CodedError


def _small(name: str, n: int = 4_000):
    scenario = SCENARIOS[name]
    return replace(scenario, n_units=n // scenario.repeats)


def test_boosted_regressor_recovers_a_weighted_step_and_is_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3_000, 3))
    truth = np.where((X[:, 0] > 0) & (X[:, 1] > 0), 1.0, -1.0)
    y = truth + rng.normal(scale=0.3, size=truth.size)
    w = rng.uniform(0.2, 1.0, size=truth.size)
    first = BoostedRegressor(n_rounds=60).fit(X, y, sample_weight=w).predict(X)
    second = BoostedRegressor(n_rounds=60).fit(X, y, sample_weight=w).predict(X)
    np.testing.assert_array_equal(first, second)
    assert np.sqrt(np.mean((first - truth) ** 2)) < 0.25
    staged = BoostedRegressor(n_rounds=60).fit(X, y).staged_predict(X, (10, 60))
    assert np.mean((staged[1] - truth) ** 2) < np.mean((staged[0] - truth) ** 2)


def test_rlearner_requires_selection_and_recovers_a_threshold_effect():
    draw = simulate(_small("nonlinear", 10_000), np.random.default_rng(2))
    labels = three_way_split(draw, seed=2)
    X = design_matrix(draw.cols)
    fit, sel = labels == 0, labels == 1
    learner = RLearner(seed=2).fit(X[fit], draw.y[fit], draw.d[fit], draw.unit_ids[fit])
    with pytest.raises(RuntimeError, match="select"):
        learner.predict(X)
    assert learner.select(X[sel], draw.y[sel], draw.d[sel]) in learner.rounds_grid
    ev = labels == 2
    tau = draw.tau[ev]
    assert np.mean((learner.predict(X[ev]) - tau) ** 2) < 0.6 * tau.var()


@pytest.mark.parametrize("name", ["nonlinear", "nonlinear_clustered", "nonlinear_repeated"])
def test_three_way_split_keeps_units_and_clusters_together(name):
    draw = simulate(_small(name), np.random.default_rng(3))
    labels = three_way_split(draw, seed=3)
    shares = np.bincount(labels, minlength=3) / labels.size
    np.testing.assert_allclose(shares, [0.4, 0.3, 0.3], atol=0.02)
    for ids in (draw.unit_ids, draw.cluster_ids):
        if ids is None:
            continue
        for label_set in (set(labels[ids == i]) for i in np.unique(ids)[:200]):
            assert len(label_set) == 1


def test_leakage_check_refuses_a_split_cluster_and_a_split_unit():
    clustered = simulate(_small("nonlinear_clustered"), np.random.default_rng(4))
    labels = three_way_split(clustered, seed=4)
    assert clustered.cluster_ids is not None
    member = np.flatnonzero(clustered.cluster_ids == clustered.cluster_ids[0])
    broken = labels.copy()
    broken[member[0]] = (labels[member[1]] + 1) % 3
    with pytest.raises(CodedError):
        assert_no_leakage(replace(clustered, unit_ids=np.arange(labels.size).astype(str)), broken)
    repeated = simulate(_small("nonlinear_repeated"), np.random.default_rng(4))
    labels = three_way_split(repeated, seed=4)
    rows = np.flatnonzero(repeated.unit_ids == repeated.unit_ids[0])
    broken = labels.copy()
    broken[rows[0]] = (labels[rows[1]] + 1) % 3
    with pytest.raises(CodedError):
        assert_no_leakage(repeated, broken)


def test_nothing_fitted_reads_evaluation_rows():
    scenario = _small("subgroup_miscalibration", 3_000)
    draw = simulate(scenario, np.random.default_rng(5))
    labels = three_way_split(draw, seed=5)
    ev = labels == 2
    poisoned = replace(draw, y=np.where(ev, draw.y + 50.0 * draw.d, draw.y))
    probe = {k: v[:500] for k, v in draw.cols.items()}
    clean, _ = fit_learners(draw, labels, 5, scenario)
    dirty, _ = fit_learners(poisoned, labels, 5, scenario)
    assert clean.simple_treat_all == dirty.simple_treat_all
    assert set(clean.predictors) == set(dirty.predictors)
    for name, predictor in clean.predictors.items():
        np.testing.assert_array_equal(predictor(probe), dirty.predictors[name](probe))


def test_heldout_score_is_unbiased_for_the_effect_whatever_the_control_variate():
    rng = np.random.default_rng(6)
    n = 400_000
    x = rng.normal(size=n)
    tau = 0.3 + 0.5 * (x > 0)
    d = rng.binomial(1, 0.5, size=n).astype(float)
    y = 2.0 * np.sin(x) + d * tau + rng.normal(size=n)
    for m in (np.zeros(n), 2.0 * np.sin(x), 5.0 + x):
        psi = heldout_score(y, d, m)
        for mask in (x > 0, x <= 0):
            assert abs(psi[mask].mean() - tau[mask].mean()) < 0.03


def test_joint_mean_cov_matches_the_independent_and_clustered_formulas():
    rng = np.random.default_rng(7)
    u = rng.normal(size=(500, 2))
    means, cov = joint_mean_cov(u, np.arange(500))
    np.testing.assert_allclose(means, u.mean(axis=0))
    np.testing.assert_allclose(cov, np.cov(u, rowvar=False) / 500)
    groups = np.repeat(np.arange(100), 5)
    _, cov = joint_mean_cov(u[:, :1], groups)
    totals = np.bincount(groups, weights=u[:, 0] - u[:, 0].mean())
    assert math.isclose(cov[0, 0], 100 / 99 * np.sum(totals**2) / 500**2, rel_tol=1e-12)


def test_max_abs_critical_spans_pointwise_to_sidak():
    z = norm.isf(ALPHA / 2)
    assert max_abs_critical(np.eye(1), ALPHA) == pytest.approx(z)
    assert max_abs_critical(np.ones((3, 3)), ALPHA) == pytest.approx(z, abs=0.02)
    sidak = norm.isf((1 - (1 - ALPHA) ** (1 / 4)) / 2)
    assert max_abs_critical(np.eye(4), ALPHA) == pytest.approx(sidak, abs=0.02)


def _audit_inputs(seed: int = 8, n: int = 6_000):
    rng = np.random.default_rng(seed)
    cols = {
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
        "seg": rng.choice(np.array(["a", "b", "c"]), size=n, p=[0.5, 0.3, 0.2]),
    }
    d = rng.binomial(1, 0.5, size=n).astype(float)
    tau = 0.2 + 0.3 * (cols["x1"] > 0)
    y = d * tau + rng.normal(size=n)
    return cols, d, heldout_score(y, d, np.zeros(n)), tau


def test_audit_reports_overlapping_groups_jointly_and_thin_groups_as_unavailable():
    cols, d, psi, tau = _audit_inputs()
    tau_hat = np.full(psi.size, 0.35)
    report = audit_groups(tau_hat, psi, d, cols, np.arange(psi.size))
    by_name = {r.group: r for r in report.rows}
    assert [r.group for r in report.rows] == list(AUDIT_GROUPS)
    thin = by_name["seg=c&x1>1.5&x2>1"]
    assert thin.discrepancy is None and thin.unavailable_reason is not None
    assert thin.unavailable_reason.startswith("insufficient_evidence")
    assert report.wald_df == len(report.auditable)
    for row in report.auditable:
        mask = AUDIT_GROUPS[row.group](cols)
        assert row.discrepancy is not None
        assert row.discrepancy.estimate == pytest.approx(np.mean(tau_hat[mask] - psi[mask]))
        assert row.predicted == pytest.approx(0.35)
    # Independent construction of one overlapping covariance entry.
    a, b = AUDIT_GROUPS["x1>0"](cols), AUDIT_GROUPS["x1>0&x2>0"](cols)
    r = tau_hat - psi
    ia = a * (r - r[a].mean()) / a.mean()
    ib = b * (r - r[b].mean()) / b.mean()
    assert np.cov(ia, ib)[0, 1] > 0
    z = norm.isf(ALPHA / 2)
    bonferroni = norm.isf(ALPHA / (2 * len(report.auditable)))
    assert z < report.critical_value < bonferroni
    # Predictions miss the x1 > 0 effect by 0.35 - 0.5 = -0.15: covered and detected.
    detected = by_name["x1>0"].discrepancy
    assert detected is not None
    assert detected.lower <= -0.15 <= detected.upper < 0


def test_shrunk_correction_never_exceeds_naive_and_leaves_thin_cells_alone():
    cols, d, psi, tau = _audit_inputs(seed=9)
    tau_hat = tau + 0.3 * (cols["seg"] == "b")
    naive = fit_cell_correction(tau_hat, psi, d, cols, np.arange(psi.size), shrink=False)
    shrunk = fit_cell_correction(tau_hat, psi, d, cols, np.arange(psi.size), shrink=True)
    assert np.all(np.abs(shrunk.offsets) <= np.abs(naive.offsets) + 1e-15)
    assert np.all((shrunk.shrinkage >= 0) & (shrunk.shrinkage <= 1))
    b_cells = np.unique(correction_cell(cols)[cols["seg"] == "b"])
    corrected = shrunk.apply(tau_hat, cols)
    seg_b = cols["seg"] == "b"
    assert abs(np.mean(corrected[seg_b] - tau[seg_b])) < abs(np.mean(tau_hat[seg_b] - tau[seg_b]))
    assert np.all(naive.offsets[b_cells] > 0)


def _summary(role: str, *, current: float, candidate: float, oracle: float, coverage=0.95):
    comparison = {"coverage": coverage, "false_gain_claim": 0.0}
    audit = {"simultaneous_coverage": coverage}
    return {
        "scenario": role,
        "role": role,
        "true_value": {"current": current, "candidate": candidate},
        "oracle_value": oracle,
        "candidate_primary_win_rate": 0.9,
        "comparisons": {"candidate-simple": comparison, "current-simple": comparison},
        "audits": {"current": audit, "candidate": audit},
    }


def test_recommendation_rule_applies_its_declared_thresholds():
    safe = [_summary("linear", current=0.10, candidate=0.10, oracle=0.10)]
    gap = 0.10
    material = _summary("nonlinear", current=0.0, candidate=MATERIAL_GAP_SHARE * gap, oracle=gap)
    small = _summary("nonlinear", current=0.0, candidate=0.9 * MATERIAL_GAP_SHARE * gap, oracle=gap)
    assert recommend([material, *safe])["verdict"] == "implement"
    assert recommend([small, *safe])["verdict"] == "reject"
    undercovered = _summary("linear", current=0.1, candidate=0.1, oracle=0.1, coverage=0.80)
    deferred = recommend([material, undercovered])
    assert deferred["verdict"] == "defer" and deferred["failures"]


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_small_campaign_recovers_gain_without_false_targeting():
    """Small campaign (12 replicates per scenario, seed 1730) with wide bands.

    The nonlinear surface must show a positive true candidate gain, the
    constant surface must not produce candidate gain claims beyond chance,
    and simultaneous audit coverage must stay well above collapse.
    """
    result = run_campaign(["nonlinear", "constant"], reps=12, seed=1730, log=lambda _: None)
    nonlinear, constant = result["summaries"]
    assert nonlinear["true_value"]["candidate"] > nonlinear["true_value"]["current"]
    assert constant["comparisons"]["candidate-simple"]["false_gain_claim"] <= 0.25
    for summary in (nonlinear, constant):
        for audit in summary["audits"].values():
            assert audit["simultaneous_coverage"] >= 0.75


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_isolated_audit_coverage_stays_near_nominal_under_clustering():
    """Fixed predictions, 200 clustered evaluation draws (base seed 1000).

    The full 500-replicate check measures about 0.94 simultaneous coverage at
    nominal 0.95; the band here catches a broken variance, not a 1-point drift.
    """
    result = audit_coverage("nonlinear_clustered", reps=200)
    assert result["simultaneous_coverage"] >= 0.88
    for z in result["z"].values():
        assert abs(z["mean"]) < 0.25 and 0.8 < z["sd"] < 1.2


def test_reference_truth_matches_the_effect_surface():
    reference = Reference.build(SCENARIOS["nonlinear"], seed=10, n=50_000)
    assert reference.tau.mean() == pytest.approx(-0.2 + 0.7 * 0.25, abs=0.01)
    draw: Draw = simulate(_small("constant"), np.random.default_rng(10))
    np.testing.assert_allclose(draw.tau, 0.25)
