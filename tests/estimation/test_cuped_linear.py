from __future__ import annotations

import copy
import math
import pickle
from typing import cast

import numpy as np
import pytest
from scipy import stats

from increment.errors import InvalidRequestError
from increment.estimation.cuped_linear import fit_cuped_linear


def _fit_repeated_design(y, arms, *, repetitions=4, **options):
    """Repeat a numerical witness's units to keep it within the support region."""
    for family in ("numerator", "denominator"):
        if family in options:
            options[family] = np.repeat(options[family], repetitions)
    if "covariates" in options:
        options["covariates"] = {
            name: np.repeat(values, repetitions) for name, values in options["covariates"].items()
        }
    return fit_cuped_linear(
        None if y is None else np.repeat(y, repetitions), np.repeat(arms, repetitions), **options
    )


def _reference(y, arm, x, arm_order, treatment, control):
    """Independent ordinary-matrix WLS and BM reference for one contrast."""
    n = len(y)
    groups = [np.flatnonzero(arm == label) for label in arm_order]
    means_x = [x[idx].mean(axis=0) for idx in groups]
    pi = np.array([len(idx) / n for idx in groups])
    anchor = sum(p * m for p, m in zip(pi, means_x, strict=True))
    x_centered = np.vstack([x[idx] - m for idx, m in zip(groups, means_x, strict=True)])
    y_centered = np.concatenate([y[idx] - y[idx].mean() for idx in groups])
    lam_parts = [np.full(len(idx), 1 / (len(idx) * (len(idx) - 1))) for idx in groups]
    lam = np.concatenate(lam_parts)
    theta = np.linalg.solve(
        x_centered.T @ (x_centered * lam[:, None]), x_centered.T @ (y_centered * lam)
    )
    adjusted = np.array(
        [y[idx].mean() - theta @ (m - anchor) for idx, m in zip(groups, means_x, strict=True)]
    )
    residual = y_centered - x_centered @ theta
    gram_inv = np.linalg.inv(x_centered.T @ (x_centered * lam[:, None]))
    centered_groups = [x[idx] - m for idx, m in zip(groups, means_x, strict=True)]
    leverage = np.concatenate(
        [
            np.full(len(idx), 1 / len(idx)) + w * np.einsum("ij,jk,ik->i", xc, gram_inv, xc)
            for idx, xc, w in zip(groups, centered_groups, lam_parts, strict=True)
        ]
    )
    j = np.zeros(len(groups))
    j[arm_order.index(treatment)] = 1
    j[arm_order.index(control)] = -1
    d = sum(j[a] * (means_x[a] - anchor) for a in range(len(groups)))
    c = np.concatenate(
        [
            np.full(len(idx), j[a] / len(idx)) - w * (xc @ gram_inv @ d)
            for a, (idx, xc, w) in enumerate(zip(groups, centered_groups, lam_parts, strict=True))
        ]
    )
    se = np.sqrt(np.sum((c * residual / np.sqrt(1 - leverage)) ** 2))
    X = np.zeros((n, len(groups) + x.shape[1]))
    cursor = 0
    for a, idx in enumerate(groups):
        X[cursor : cursor + len(idx), a] = 1
        X[cursor : cursor + len(idx), len(groups) :] = x[idx] - means_x[a]
        cursor += len(idx)
    xtwx_inv = np.linalg.inv(X.T @ (X * lam[:, None]))
    H = X @ xtwx_inv @ (X * lam[:, None]).T
    contrast = np.concatenate([j, -d])
    ai = X @ (xtwx_inv @ contrast) * lam
    gm = (np.eye(n) - H).T * (ai / np.sqrt(1 - np.diag(H)))[None, :]
    eigenvalues = np.linalg.eigvalsh(gm.T @ gm)
    dof = eigenvalues.sum() ** 2 / (eigenvalues @ eigenvalues)
    point = adjusted[arm_order.index(treatment)] - adjusted[arm_order.index(control)]
    return point, se, dof, theta, anchor, leverage


def test_absolute_projection_matches_independent_wls_hc2_and_bm_reference():
    rng = np.random.default_rng(219)
    arm = np.array(["t"] * 17 + ["c"] * 13)
    x = rng.normal(size=(len(arm), 2))
    y = 1.2 + 0.8 * (arm == "t") + x @ np.array([0.7, -0.4]) + rng.normal(size=len(arm))
    order = ("c", "t")
    fit = fit_cuped_linear(
        y,
        arm,
        control="c",
        arm_order=order,
        covariates={"x1": x[:, 0], "x2": x[:, 1]},
        kinds={"x1": "continuous", "x2": "continuous"},
    )
    expected = _reference(y, arm, x, order, "t", "c")
    result = fit.absolute("t", alpha=0.05)
    assert result.estimate == pytest.approx(expected[0], abs=1e-11)
    assert result.se == pytest.approx(expected[1], rel=1e-10)
    assert result.dof == pytest.approx(expected[2], rel=1e-9)
    assert fit.theta == pytest.approx(expected[3], abs=1e-11)
    assert fit.anchor == pytest.approx(expected[4], abs=1e-11)
    assert fit.leverage_trace == pytest.approx(2 + fit.p, abs=1e-11)
    critical = stats.t.isf(0.025, expected[2])
    assert result.lower == pytest.approx(expected[0] - critical * expected[1])
    assert result.upper == pytest.approx(expected[0] + critical * expected[1])
    assert fit.n_min == 13 and fit.p == 2
    assert fit.max_leverage == pytest.approx(expected[5].max())
    assert fit.mean_leverage == pytest.approx(expected[5].mean())


def test_fit_and_refusal_are_pickle_and_deepcopy_safe():
    y = np.array([1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 6.0, 4.0])
    arm = np.array(["c"] * 4 + ["t"] * 4)
    fit = fit_cuped_linear(y, arm, control="c", arm_order=("c", "t"))
    for restored in (pickle.loads(pickle.dumps(fit)), copy.deepcopy(fit)):
        assert restored.absolute("t").estimate == pytest.approx(fit.absolute("t").estimate)
    x = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 1.0, 4.0])
    with pytest.raises(InvalidRequestError) as raised:
        fit_cuped_linear(
            y,
            arm,
            control="c",
            arm_order=("c", "t"),
            covariates={"x": x, "x_copy": x.copy()},
            kinds={"x": "continuous", "x_copy": "continuous"},
        )
    for restored in (pickle.loads(pickle.dumps(raised.value)), copy.deepcopy(raised.value)):
        assert restored.code == "estimation.cuped_linear.rank_deficiency"
        assert restored.context["columns"] and restored.context["partners"]


def test_unadjusted_hc2_is_welch_and_alpha_only_changes_endpoints():
    rng = np.random.default_rng(42)
    arm = np.array(["c"] * 11 + ["t"] * 15)
    y = rng.normal(size=len(arm))
    fit = fit_cuped_linear(y, arm, control="c", arm_order=("c", "t"))
    result = fit.absolute("t", alpha=0.05)
    welch = y[arm == "t"].var(ddof=1) / 15 + y[arm == "c"].var(ddof=1) / 11
    assert result.se**2 == pytest.approx(welch, rel=1e-12)
    narrow = fit.absolute("t", alpha=0.01)
    assert (narrow.estimate, narrow.se, narrow.dof) == pytest.approx(
        (result.estimate, result.se, result.dof)
    )
    assert narrow.lower < result.lower and narrow.upper > result.upper


def test_scalar_point_matches_scalar_cuped_moment_fit():
    from increment.estimation.armstats import ArmStats
    from increment.estimation.cuped import fit_cuped

    x_c = np.array([0.0, 1.0, 2.0, 4.0, 5.0])
    x_t = np.array([1.0, 2.0, 3.0, 6.0, 7.0, 8.0])
    y_c = 2 + 0.8 * x_c + np.array([1.0, -1.0, 0.0, 1.0, -1.0])
    y_t = 3 + 0.8 * x_t + np.array([-1.0, 1.0, 0.0, -1.0, 1.0, 0.0])
    fit = fit_cuped_linear(
        np.r_[y_c, y_t],
        ["c"] * 5 + ["t"] * 6,
        control="c",
        arm_order=("c", "t"),
        covariates={"x": np.r_[x_c, x_t]},
        kinds={"x": "continuous"},
    )
    stats_ = [
        ArmStats.from_raw_sums(
            study_id="s",
            metric="m",
            group_id=g,
            n=len(xx),
            sum_y=float(yy.sum()),
            sum_y2=float(yy @ yy),
            sum_x=float(xx.sum()),
            sum_x2=float(xx @ xx),
            sum_xy=float(xx @ yy),
        )
        for g, xx, yy in (("c", x_c, y_c), ("t", x_t, y_t))
    ]
    scalar = fit_cuped(stats_)
    assert fit.theta[0] == pytest.approx(scalar.theta, abs=1e-12)
    assert fit.anchor[0] == pytest.approx(scalar.mean_x_pooled, abs=1e-12)
    for index, item in enumerate(stats_):
        assert fit.adjusted_means[index] == pytest.approx(scalar.adjusted_mean(item), abs=1e-12)


def test_categorical_missing_token_is_distinct_and_rows_are_retained():
    arm = np.array(["c"] * 8 + ["t"] * 8)
    category = np.array(["<null>", "a", None, "a", "<null>", "a", None, "a"] * 2, dtype=object)
    fit = _fit_repeated_design(
        np.arange(16.0),
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"category": category},
        kinds={"category": "categorical"},
        missing={"category": "impute-indicator"},
    )
    from increment.estimation.cuped_linear import MISSING_CATEGORY

    category_levels = fit.layout.levels[0]
    assert category_levels is not None
    assert category_levels[:2] == ("<null>", "a")
    assert category_levels[2] is MISSING_CATEGORY
    assert category_levels[2] != "<null>"
    assert fit.p == 2 and sum(fit.n_by_arm) == 64


def test_numeric_missing_history_is_retained_and_identical_masks_share_indicator():
    arm = np.array(["c"] * 8 + ["t"] * 8)
    history = np.array([1.0, 3.0, 2.0, 5.0, None, None, None, None] * 2, dtype=object)
    second = np.array([4.0, 1.0, 5.0, 2.0, None, None, None, None] * 2, dtype=object)
    fit = _fit_repeated_design(
        np.arange(16.0),
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"history": history, "second": second},
        kinds={"history": "continuous", "second": "continuous"},
        missing={"history": "impute-indicator", "second": "impute-indicator"},
    )
    assert sum(fit.n_by_arm) == 64
    assert fit.layout.shared_indicators == (("history", "second"),)
    assert fit.layout.encoded_columns[-1] == "missing[history,second]"


def test_categorical_only_fit_equals_target_weighted_stratum_contrast():
    strata = np.repeat(["a", "b"], 200)
    arm = np.concatenate([np.array(["t"] * 120 + ["c"] * 80)] * 2)
    y = np.concatenate(
        [
            np.r_[np.full(120, 11.0), np.full(80, 10.0)],
            np.r_[np.full(120, 20.0), np.full(80, 20.0)],
        ]
    )
    # Stratum proportions and allocation probabilities are equal by construction.
    fit = fit_cuped_linear(
        y,
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"stratum": strata},
        kinds={"stratum": "categorical"},
    )
    target = 0.5 * (1.0 + 0.0)
    assert fit.absolute("t").estimate == pytest.approx(target, abs=1e-12)


def test_repeated_fit_has_stable_results():
    rng = np.random.default_rng(617)
    arm = np.array(["c"] * 30 + ["t"] * 30)
    x = rng.normal(size=60)
    y = 3 + (arm == "t") + 2 * x + rng.normal(size=60)
    fit = fit_cuped_linear(
        y,
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"x": x},
        kinds={"x": "continuous"},
    )
    altered = fit_cuped_linear(
        y,
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"x": x},
        kinds={"x": "continuous"},
    )
    assert altered.theta == fit.theta
    assert altered.anchor == fit.anchor
    assert altered.absolute("t").estimate == fit.absolute("t").estimate
    assert altered.absolute("t").se == fit.absolute("t").se


def test_ratio_projection_uses_joint_component_covariance():
    rng = np.random.default_rng(711)
    arm = np.array(["c"] * 40 + ["t"] * 40)
    x = rng.normal(size=80)
    num = 5 + 0.6 * x + rng.normal(scale=0.7, size=80) + (arm == "t") * 0.4
    den = 3 + 0.3 * x + 0.5 * (num - 5 - 0.6 * x) + rng.normal(scale=0.4, size=80)
    fit = fit_cuped_linear(
        None,
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"x": x},
        kinds={"x": "continuous"},
        numerator=num,
        denominator=den,
    )
    result = fit.ratio("t", scale="log")
    assert np.isfinite(result.se) and result.se > 0
    assert result.dof > 0
    assert fit.covariance[0][len(fit.arm_order)] != 0
    means = np.asarray(fit.adjusted_means).reshape(2, 2)
    jac = np.r_[-1 / means[0, 0], 1 / means[0, 1], 1 / means[1, 0], -1 / means[1, 1]]
    assert result.se**2 == pytest.approx(jac @ np.asarray(fit.covariance) @ jac, rel=1e-10)


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("arm_count", "estimation.cuped_linear.arm_counts"),
        ("missing", "estimation.cuped_linear.missing_refused"),
        ("level", "estimation.cuped_linear.level_support"),
        ("leverage", "estimation.cuped_linear.leverage"),
        ("df", "estimation.cuped_linear.residual_df"),
        ("arm_constant", "estimation.cuped_linear.rank_deficiency"),
        ("support_l", "estimation.cuped_linear.support_region"),
    ],
)
def test_fit_refusals_have_stable_codes(case, code):
    arm = np.array(["c"] * 6 + ["t"] * 6)
    y = np.arange(12, dtype=float)
    covariates = None
    kinds = None
    if case == "arm_count":
        arm = np.array(["c"] * 1 + ["t"] * 3)
        y = np.arange(4, dtype=float)
    elif case == "missing":
        x = np.array([1.0, 2.0, None, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=object)
        covariates = {"x": x}
        kinds = {"x": "continuous"}
    elif case == "level":
        category = np.array(["a", "b", "b", "b", "b", "b", "a", "a", "b", "b", "b", "b"])
        covariates = {"segment": category}
        kinds = {"segment": "categorical"}
    elif case == "leverage":
        x = np.r_[np.zeros(5), 100.0, np.ones(6)]
        covariates = {"x": x}
        kinds = {"x": "continuous"}
    elif case == "arm_constant":
        covariates = {"arm_intercept": np.r_[np.zeros(6), np.ones(6)]}
        kinds = {"arm_intercept": "continuous"}
    elif case == "df":
        rng = np.random.default_rng(88)
        arm = np.array(["c"] * 3 + ["t"] * 3)
        y = np.arange(6, dtype=float)
        x = rng.normal(size=(6, 4))
        covariates = {f"x{i}": x[:, i] for i in range(4)}
        kinds = {f"x{i}": "continuous" for i in range(4)}
    elif case == "support_l":
        rng = np.random.default_rng(101)
        arm = np.array(["c"] * 8 + ["t"] * 8)
        y = rng.normal(size=16)
        x = rng.normal(size=(16, 4))
        covariates = {f"x{i}": x[:, i] for i in range(4)}
        kinds = {f"x{i}": "continuous" for i in range(4)}
    elif case == "nonfinite":
        y = y.copy()
        y[2] = np.inf
    with pytest.raises(InvalidRequestError) as raised:
        fit_cuped_linear(
            y,
            arm,
            control="c",
            arm_order=tuple(dict.fromkeys(arm)),
            covariates=covariates,
            kinds=kinds,
        )
    assert raised.value.code == code


def test_relative_projection_refuses_nonpositive_adjusted_means():
    arm = np.array(["c"] * 6 + ["t"] * 6)
    y = np.r_[np.full(6, -1.0), np.full(6, 2.0)]
    fit = fit_cuped_linear(y, arm, control="c", arm_order=("c", "t"))
    with pytest.raises(InvalidRequestError) as raised:
        fit.relative("t")
    assert raised.value.code == "estimation.cuped_linear.nonpositive_component"


def _ratio_reference(num, den, arm, x, order):
    groups = [np.flatnonzero(arm == a) for a in order]
    counts = [len(idx) for idx in groups]
    means_x = [x[idx].mean() for idx in groups]
    anchor = sum((n / len(arm)) * mean for n, mean in zip(counts, means_x, strict=True))
    w = [1 / (n * (n - 1)) for n in counts]
    gram = sum(
        weight * np.sum((x[idx] - mean) ** 2)
        for idx, mean, weight in zip(groups, means_x, w, strict=True)
    )
    slopes, adjusted, residuals = [], [], []
    for outcome in (num, den):
        theta = (
            sum(
                weight * np.sum((x[idx] - mean) * (outcome[idx] - outcome[idx].mean()))
                for idx, mean, weight in zip(groups, means_x, w, strict=True)
            )
            / gram
        )
        slopes.append(theta)
        adjusted.append(
            np.array(
                [
                    outcome[idx].mean() - theta * (mean - anchor)
                    for idx, mean in zip(groups, means_x, strict=True)
                ]
            )
        )
        residuals.append(
            [
                outcome[idx] - outcome[idx].mean() - theta * (x[idx] - mean)
                for idx, mean in zip(groups, means_x, strict=True)
            ]
        )
    nt, nc = adjusted[0][1], adjusted[0][0]
    dt, dc = adjusted[1][1], adjusted[1][0]
    jacobians = (np.array([-1 / nc, 1 / nt]), np.array([1 / dc, -1 / dt]))
    component_scores, contrasts = [], []
    for family, jac in enumerate(jacobians):
        d = sum(jac[a] * (means_x[a] - anchor) for a in range(2))
        k = jac[1] * counts[0] / sum(counts) - jac[0] * counts[1] / sum(counts)
        s = (-slopes[family] * k / jac[0], slopes[family] * k / jac[1])
        score = []
        for a, idx in enumerate(groups):
            xc = x[idx] - means_x[a]
            c = jac[a] / counts[a] - w[a] * xc * d / gram
            h = 1 / counts[a] + w[a] * xc**2 / gram
            score.extend(
                c * residuals[family][a] / np.sqrt(1 - h)
                + jac[a] / counts[a] * (slopes[family] - s[a]) * xc
            )
        component_scores.append(np.asarray(score))
        contrasts.append(np.r_[jac, -d])
    se = np.linalg.norm(component_scores[0] + component_scores[1])
    rows, weight_rows = [], []
    for a, idx in enumerate(groups):
        design = np.zeros((len(idx), 3))
        design[:, a] = 1
        design[:, 2] = x[idx] - means_x[a]
        rows.append(design)
        weight_rows.append(np.full(len(idx), w[a]))
    one, one_w = np.vstack(rows), np.concatenate(weight_rows)
    design = np.block([[one, np.zeros_like(one)], [np.zeros_like(one), one]])
    weights = np.tile(one_w, 2)
    inverse = np.linalg.inv(design.T @ (design * weights[:, None]))
    hat = design @ inverse @ (design * weights[:, None]).T
    leverage = np.diag(hat)
    contrast = np.r_[contrasts[0], contrasts[1]]
    ai = design @ (inverse @ contrast) * weights
    gm = (np.eye(len(ai)) - hat).T * (ai / np.sqrt(1 - leverage))[None, :]
    eig = np.linalg.eigvalsh(gm.T @ gm)
    dof = eig.sum() ** 2 / (eig @ eig)
    return np.log((nt / dt) / (nc / dc)), se, dof


def test_ratio_log_projection_matches_independent_stacked_reference():
    rng = np.random.default_rng(711)
    arm = np.array(["c"] * 40 + ["t"] * 40)
    x = rng.normal(size=80)
    num = 5 + 0.6 * x + rng.normal(scale=0.7, size=80) + (arm == "t") * 0.4
    den = 3 + 0.3 * x + 0.5 * (num - 5 - 0.6 * x) + rng.normal(scale=0.4, size=80)
    fit = fit_cuped_linear(
        None,
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"x": x},
        kinds={"x": "continuous"},
        numerator=num,
        denominator=den,
    )
    expected = _ratio_reference(num, den, arm, x, ("c", "t"))
    result = fit.ratio("t", scale="log")
    assert result.estimate == pytest.approx(expected[0], abs=1e-11)
    assert result.se == pytest.approx(expected[1], rel=1e-10)
    assert result.dof == pytest.approx(expected[2], rel=1e-9)


def test_constant_covariates_are_disclosed_without_losing_missingness_indicators():
    arm = np.array(["c"] * 6 + ["t"] * 6)
    history = np.array([0.0, 0.0, None, None, 0.0, 0.0] * 2, dtype=object)
    fit = fit_cuped_linear(
        np.arange(12.0),
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"constant": np.ones(12), "history": history},
        kinds={"constant": "continuous", "history": "continuous"},
        missing={"history": "impute-indicator"},
    )
    assert fit.layout.structural_omissions == ("constant", "history")
    assert fit.layout.levels == (None, None)
    assert sum(fit.n_by_arm) == len(arm)


def test_three_arm_absolute_projection_uses_joint_covariance():
    rng = np.random.default_rng(872)
    arm = np.repeat(["c", "t", "other"], [9, 13, 11])
    x = rng.normal(size=33)
    y = 2.0 + (arm == "t") * 0.7 + (arm == "other") * 0.2 + 1.4 * x + rng.normal(size=33)
    fit = fit_cuped_linear(
        y,
        arm,
        control="c",
        arm_order=("c", "t", "other"),
        covariates={"x": x},
        kinds={"x": "continuous"},
    )
    result = fit.absolute("t")
    contrast = np.array([-1.0, 1.0, 0.0])
    assert result.se**2 == pytest.approx(contrast @ np.asarray(fit.covariance) @ contrast)


def test_three_arm_relative_and_ratio_projections_remain_finite():
    rng = np.random.default_rng(941)
    arm = np.repeat(["c", "t", "other"], [9, 13, 11])
    x = rng.normal(size=33)
    y = 5 + (arm == "t") * 0.8 + 0.3 * x + rng.normal(scale=0.2, size=33)
    numerator = 8 + (arm == "t") * 0.7 + 0.5 * x + rng.normal(scale=0.2, size=33)
    denominator = 4 + (arm == "other") * 0.2 + 0.1 * x + rng.normal(scale=0.1, size=33)
    mean_fit = fit_cuped_linear(
        y,
        arm,
        control="c",
        arm_order=("c", "t", "other"),
        covariates={"x": x},
        kinds={"x": "continuous"},
    )
    fit = fit_cuped_linear(
        None,
        arm,
        control="c",
        arm_order=("c", "t", "other"),
        covariates={"x": x},
        kinds={"x": "continuous"},
        numerator=numerator,
        denominator=denominator,
    )
    means = np.asarray(fit.adjusted_means).reshape(2, len(fit.arm_order))
    relative_means = np.asarray(mean_fit.adjusted_means)
    relative = mean_fit.relative("t")
    assert relative.estimate == pytest.approx(
        math.log(relative_means[1]) - math.log(relative_means[0])
    )
    assert relative.se > 0 and np.isfinite(relative.se)
    for scale in ("log", "absolute"):
        if scale == "log":
            expected = (
                math.log(means[0, 1])
                - math.log(means[1, 1])
                - math.log(means[0, 0])
                + math.log(means[1, 0])
            )
        else:
            expected = means[0, 1] / means[1, 1] - means[0, 0] / means[1, 0]
        result = fit.ratio("t", scale=scale)
        assert result.estimate == pytest.approx(expected)
        assert result.se > 0 and np.isfinite(result.se)


@pytest.mark.parametrize("epsilon", [1e-7])
def test_near_collinear_accepted_fit_matches_weighted_qr(epsilon):
    rng = np.random.default_rng(809)
    arm = np.repeat(["c", "t"], 20)
    z = rng.normal(size=40)
    v = rng.normal(size=40)
    x = np.column_stack([z, z + epsilon * v])
    y = 2 + (arm == "t") * 0.5 + x @ np.array([0.7, -0.2]) + rng.normal(size=40)
    fit = fit_cuped_linear(
        y,
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"z": x[:, 0], "near": x[:, 1]},
        kinds={"z": "continuous", "near": "continuous"},
    )
    groups = [np.flatnonzero(arm == label) for label in ("c", "t")]
    means = [x[group].mean(axis=0) for group in groups]
    design = np.zeros((len(y), 4))
    for a, group in enumerate(groups):
        design[group, a] = 1
        design[group, 2:] = x[group] - means[a]
    weights = np.concatenate(
        [np.full(len(group), 1 / (len(group) * (len(group) - 1))) for group in groups]
    )
    sqrt_weights = np.sqrt(weights)
    q, r = np.linalg.qr(design * sqrt_weights[:, None], mode="reduced")
    beta = np.linalg.solve(r, q.T @ (sqrt_weights * y))
    pooled = (20 * means[0] + 20 * means[1]) / 40
    adjusted = np.array(
        [y[group].mean() - beta[2:] @ (means[a] - pooled) for a, group in enumerate(groups)]
    )
    assert fit.absolute("t").estimate == pytest.approx(adjusted[1] - adjusted[0], rel=1e-7)
    assert fit.leverage_trace == pytest.approx(4.0, abs=1e-8)
    d = means[1] - means[0]
    contrast = np.r_[-1.0, 1.0, -d]
    coefficients = np.linalg.solve(r, np.linalg.solve(r.T, contrast))
    leverage = np.sum(q * q, axis=1)
    residual = y - design @ beta
    influence = design @ coefficients * weights
    expected_se = np.linalg.norm(influence * residual / np.sqrt(1 - leverage))
    assert fit.absolute("t").se == pytest.approx(expected_se, rel=5e-4)


def _dense_bm_dof(design, weights, contrast):
    inverse = np.linalg.inv(design.T @ (design * weights[:, None]))
    hat = design @ inverse @ (design * weights[:, None]).T
    leverage = np.diag(hat)
    influence = design @ (inverse @ contrast) * weights
    matrix = (np.eye(len(weights)) - hat).T * (influence / np.sqrt(1 - leverage))[None, :]
    eigenvalues = np.linalg.eigvalsh(matrix.T @ matrix)
    return eigenvalues.sum() ** 2 / (eigenvalues @ eigenvalues)


@pytest.mark.parametrize("arm_order", [("c", "t"), ("c", "t", "other")])
def test_all_projection_dofs_match_dense_eigenvalue_reference(arm_order):
    rng = np.random.default_rng(1401)
    counts = [11 + 2 * i for i in range(len(arm_order))]
    arm = np.repeat(arm_order, counts)
    x = rng.normal(size=len(arm))
    groups = [np.flatnonzero(arm == label) for label in arm_order]
    means_x = np.array([x[group].mean() for group in groups])
    anchor = np.average(means_x, weights=counts)
    design = np.zeros((len(arm), len(arm_order) + 1))
    weights = np.empty(len(arm))
    for i, group in enumerate(groups):
        design[group, i] = 1
        design[group, -1] = x[group] - means_x[i]
        weights[group] = 1 / (len(group) * (len(group) - 1))
    y = 5 + (arm == "t") * 0.7 + 0.3 * x + rng.normal(scale=0.5, size=len(arm))
    numerator = 9 + (arm == "t") * 0.8 + 0.4 * x + rng.normal(scale=0.4, size=len(arm))
    denominator = 4 + (arm == "other") * 0.2 + 0.1 * x + rng.normal(scale=0.2, size=len(arm))
    mean_fit = fit_cuped_linear(
        y, arm, control="c", arm_order=arm_order, covariates={"x": x}, kinds={"x": "continuous"}
    )
    ratio_fit = fit_cuped_linear(
        None,
        arm,
        control="c",
        arm_order=arm_order,
        covariates={"x": x},
        kinds={"x": "continuous"},
        numerator=numerator,
        denominator=denominator,
    )
    treatment = "t"
    ti, ci = arm_order.index(treatment), arm_order.index("c")
    j = np.zeros(len(arm_order))
    j[ti], j[ci] = 1, -1
    d = j @ (means_x - anchor)
    contrast = np.r_[j, -d]
    assert mean_fit.absolute(treatment).dof == pytest.approx(
        _dense_bm_dof(design, weights, contrast), rel=1e-9
    )
    mean_values = np.asarray(mean_fit.adjusted_means)
    relative_j = np.zeros(len(arm_order))
    relative_j[ti], relative_j[ci] = 1 / mean_values[ti], -1 / mean_values[ci]
    relative_d = relative_j @ (means_x - anchor)
    assert mean_fit.relative(treatment).dof == pytest.approx(
        _dense_bm_dof(design, weights, np.r_[relative_j, -relative_d]), rel=1e-9
    )
    component_means = np.asarray(ratio_fit.adjusted_means).reshape(2, -1)
    for scale in ("log", "absolute"):
        jac = np.zeros(2 * len(arm_order))
        if scale == "log":
            jac[ti], jac[len(arm_order) + ti] = (
                1 / component_means[0, ti],
                -1 / component_means[1, ti],
            )
            jac[ci], jac[len(arm_order) + ci] = (
                -1 / component_means[0, ci],
                1 / component_means[1, ci],
            )
        else:
            jac[ti], jac[len(arm_order) + ti] = (
                1 / component_means[1, ti],
                -component_means[0, ti] / component_means[1, ti] ** 2,
            )
            jac[ci], jac[len(arm_order) + ci] = (
                -1 / component_means[1, ci],
                component_means[0, ci] / component_means[1, ci] ** 2,
            )
        stacked_design = np.block(
            [[design, np.zeros_like(design)], [np.zeros_like(design), design]]
        )
        stacked_weights = np.tile(weights, 2)
        stacked_contrast = []
        for family in range(2):
            component_j = jac[family * len(arm_order) : (family + 1) * len(arm_order)]
            component_d = component_j @ (means_x - anchor)
            stacked_contrast.extend(np.r_[component_j, -component_d])
        actual = ratio_fit.ratio(treatment, scale=scale).dof
        assert actual == pytest.approx(
            _dense_bm_dof(stacked_design, stacked_weights, np.asarray(stacked_contrast)),
            rel=1e-9,
        )


def test_rank_deficient_covariates_refuse_exact_linear_combinations():
    arm = np.repeat(["c", "t"], 8)
    x = np.arange(len(arm), dtype=float)
    with pytest.raises(InvalidRequestError) as raised:
        fit_cuped_linear(
            x,
            arm,
            control="c",
            arm_order=("c", "t"),
            covariates={"x": x, "twice_x": 2 * x},
            kinds={"x": "continuous", "twice_x": "continuous"},
        )
    assert raised.value.code == "estimation.cuped_linear.rank_deficiency"


@pytest.mark.slow
def test_large_fit_and_projections_use_linear_memory():
    import tracemalloc

    rng = np.random.default_rng(1402)
    n_arm = 200_000
    arm = np.repeat(["c", "t"], n_arm)
    covariates = {f"x{i}": rng.normal(size=2 * n_arm) for i in range(10)}
    y = covariates["x0"] + rng.normal(size=2 * n_arm)
    tracemalloc.start()
    fit = fit_cuped_linear(
        y,
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates=covariates,
        kinds=dict.fromkeys(covariates, "continuous"),
    )
    fit.absolute("t")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 512 * 1024 * 1024


@pytest.mark.parametrize("seed", [6, 7, 8])
def test_rank_refuses_seeded_exact_covariate_sum(seed):
    rng = np.random.default_rng(seed)
    arm = np.repeat(["c", "t"], 12)
    x, z = rng.integers(-5, 6, size=(2, len(arm)))
    with pytest.raises(InvalidRequestError) as raised:
        fit_cuped_linear(
            np.arange(len(arm), dtype=float),
            arm,
            control="c",
            arm_order=("c", "t"),
            covariates={"x": x, "z": z, "sum": x + z},
            kinds={"x": "continuous", "z": "continuous", "sum": "continuous"},
        )
    assert raised.value.code == "estimation.cuped_linear.rank_deficiency"


def test_three_arm_family_projections_match_joint_covariance():
    arm = np.repeat(["c", "t", "other"], 2)
    x = np.tile([-1.0, 1.0], 3)
    y = np.array([1.0, 3.0, 3.0, 5.0, 5.0, 7.0])
    mean_fit = _fit_repeated_design(
        y,
        arm,
        control="c",
        arm_order=("c", "t", "other"),
        covariates={"x": x},
        kinds={"x": "continuous"},
    )
    absolute_j = np.array([-1.0, 1.0, 0.0])
    assert mean_fit.absolute("t").se ** 2 == pytest.approx(
        absolute_j @ np.asarray(mean_fit.covariance) @ absolute_j, rel=1e-12
    )
    means = np.asarray(mean_fit.adjusted_means)
    relative_j = np.array([-1 / means[0], 1 / means[1], 0.0])
    relative = mean_fit.relative("t")
    assert relative.estimate == pytest.approx(math.log1p((means[1] - means[0]) / means[0]))
    assert relative.se**2 == pytest.approx(
        relative_j @ np.asarray(mean_fit.covariance) @ relative_j, rel=1e-12
    )

    numerator = np.array([1.0, 3.0, 3.0, 5.0, 5.0, 7.0])
    denominator = np.full(6, 2.0)
    fit = _fit_repeated_design(
        None,
        arm,
        control="c",
        arm_order=("c", "t", "other"),
        covariates={"x": x},
        kinds={"x": "continuous"},
        numerator=numerator,
        denominator=denominator,
    )
    component_means = np.asarray(fit.adjusted_means).reshape(2, 3)
    covariance = np.asarray(fit.covariance)
    nc, nt, ndc, ndt = (
        component_means[0, 0],
        component_means[0, 1],
        component_means[1, 0],
        component_means[1, 1],
    )
    log_jac = np.array([-1 / nc, 1 / nt, 0, 1 / ndc, -1 / ndt, 0])
    log_result = fit.ratio("t", scale="log")
    expected_log = math.log(nt / ndt) - math.log(nc / ndc)
    assert log_result.estimate == pytest.approx(expected_log)
    assert log_result.se**2 == pytest.approx(log_jac @ covariance @ log_jac, rel=1e-12)

    absolute_jac = np.array([-1 / ndc, 1 / ndt, 0, nc / ndc**2, -nt / ndt**2, 0])
    absolute_result = fit.ratio("t", scale="absolute")
    expected_absolute = nt / ndt - nc / ndc
    assert absolute_result.estimate == pytest.approx(expected_absolute)
    assert absolute_result.se**2 == pytest.approx(
        absolute_jac @ covariance @ absolute_jac, rel=1e-12
    )


def test_hc2_projection_retains_residual_variance_under_cancellation():
    arm = np.array(["c", "c", "t", "t"])
    x = np.array([-1.0, 1.0, -1.0, 1.0])
    y = np.array([-(2**30), 2**30, -(2**30), 2**30 + 2], dtype=float)
    fit = _fit_repeated_design(
        y, arm, control="c", arm_order=("c", "t"), covariates={"x": x}, kinds={"x": "continuous"}
    )
    assert fit.absolute("t").se == pytest.approx(1 / math.sqrt(13))


def test_small_relative_effect_survives_log_rounding():
    arm = np.array(["c", "c", "t", "t"])
    fit = _fit_repeated_design(
        np.array([2**50, 2**50, 2**50 + 0.25, 2**50 + 0.25]), arm, control="c", arm_order=("c", "t")
    )
    assert fit.relative("t").estimate == pytest.approx(math.log1p(2**-52), rel=1e-12, abs=0)


def test_arm_local_outcome_offsets_preserve_contrasts_and_row_order():
    base = 2**54
    arm = np.array(["c"] * 4 + ["t"] * 4)
    y = np.r_[base + np.array([0.0, 0.0, 0.0, 4.0]), base + np.array([0.0, 0.0, 4.0, 4.0])]
    for permutation in (np.arange(8), np.array([7, 3, 1, 5, 0, 6, 4, 2])):
        fit = fit_cuped_linear(y[permutation], arm[permutation], control="c", arm_order=("c", "t"))
        assert fit.absolute("t").estimate == pytest.approx(1.0)


def test_small_positive_arm_is_not_erased_by_large_other_arm():
    arm = np.array(["c"] * 4 + ["t"] * 4)
    y = np.r_[np.full(4, 2**54), [0.25, 0.75, 0.25, 0.75]]
    fit = fit_cuped_linear(y, arm, control="c", arm_order=("c", "t"))
    assert fit.adjusted_means[1] == pytest.approx(0.5)
    assert fit.absolute("t").se > 0


def test_large_covariate_offset_preserves_slope_and_zero_residual_uncertainty():
    arm = np.array(["c"] * 4 + ["t"] * 4)
    x = np.tile(1e16 + np.array([0.0, 2.0, 4.0, 6.0]), 2)
    y = np.r_[np.array([0.0, 2.0, 4.0, 6.0]), np.array([1.0, 3.0, 5.0, 7.0])]
    fit = _fit_repeated_design(
        y, arm, control="c", arm_order=("c", "t"), covariates={"x": x}, kinds={"x": "continuous"}
    )
    assert fit.theta[0] == pytest.approx(1.0)
    assert fit.absolute("t").se == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("denominator_scale", [1.0, 3.0, 5.0, 2**100])
def test_ratio_log_effect_survives_component_log_rounding(denominator_scale):
    arm = np.array(["c", "c", "t", "t"])
    base = 2**50
    fit = _fit_repeated_design(
        None,
        arm,
        control="c",
        arm_order=("c", "t"),
        numerator=np.array([base, base, base + 0.25, base + 0.25]),
        denominator=np.full(4, denominator_scale),
    )
    assert fit.ratio("t", scale="log").estimate == pytest.approx(
        math.log1p(2**-52), rel=1e-12, abs=0
    )


@pytest.mark.parametrize("epsilon", [1e-7, 1e-8])
def test_near_collinear_bm_dof_matches_dense_eigenvalue_reference(epsilon):
    rng = np.random.default_rng(105)
    arm = np.repeat(["c", "t"], [11, 15])
    z = rng.normal(size=len(arm))
    v = 4 * rng.normal(size=len(arm))
    x = np.column_stack([z, z + epsilon * v])
    y = 4 + (arm == "t") + 0.3 * z + rng.normal(size=len(arm))
    fit = fit_cuped_linear(
        y,
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"z": x[:, 0], "near": x[:, 1]},
        kinds={"z": "continuous", "near": "continuous"},
    )
    groups = [np.flatnonzero(arm == label) for label in ("c", "t")]
    means = [x[group].mean(axis=0) for group in groups]
    design = np.zeros((len(arm), 4))
    weights = np.empty(len(arm))
    for i, group in enumerate(groups):
        design[group, i] = 1
        design[group, 2:] = x[group] - means[i]
        weights[group] = 1 / (len(group) * (len(group) - 1))
    d = means[1] - means[0]
    contrast = np.r_[-1.0, 1.0, -d]
    sqrt_weights = np.sqrt(weights)
    q, r = np.linalg.qr(design * sqrt_weights[:, None], mode="reduced")
    ai = sqrt_weights * (q @ np.linalg.solve(r.T, contrast))
    leverage = np.sum(q * q, axis=1)
    hat_transpose = sqrt_weights[:, None] * (q @ q.T) / sqrt_weights[None, :]
    score_matrix = (np.eye(len(weights)) - hat_transpose) * (ai / np.sqrt(1 - leverage))[None, :]
    eigenvalues = np.linalg.eigvalsh(score_matrix.T @ score_matrix)
    expected = eigenvalues.sum() ** 2 / (eigenvalues @ eigenvalues)
    assert fit.dof == pytest.approx(expected, rel=1e-8)
    assert fit.absolute("t").dof == pytest.approx(expected, rel=1e-8)


def test_large_between_arm_covariate_offset_preserves_within_arm_design():
    arm = np.array(["c"] * 4 + ["t"] * 4)
    x = np.r_[np.full(4, 2**54), [0.25, 0.75, 0.25, 0.75]]
    fit = _fit_repeated_design(
        np.arange(8.0),
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"x": x},
        kinds={"x": "continuous"},
    )
    assert fit.p == 1
    assert fit.max_leverage == pytest.approx(0.125)


def test_correlation_scaled_pivot_refuses_numerically_dependent_column():
    arm = np.repeat(["c", "t"], 4)
    z = np.tile([-1.0, 1.0], 4)
    near = z + 1e-9 * np.tile([1.0, 1.0, -1.0, -1.0], 2)
    with pytest.raises(InvalidRequestError) as raised:
        fit_cuped_linear(
            np.arange(8.0),
            arm,
            control="c",
            arm_order=("c", "t"),
            covariates={"z": z, "near": near},
            kinds={"z": "continuous", "near": "continuous"},
        )
    assert raised.value.code == "estimation.cuped_linear.rank_deficiency"


def test_categorical_sequence_detects_nan_before_string_coercion():
    arm = np.repeat(["c", "t"], 4)
    category = ["nan", "nan", float("nan"), float("nan")] * 2
    with pytest.raises(InvalidRequestError) as raised:
        fit_cuped_linear(
            np.arange(8.0),
            arm,
            control="c",
            arm_order=("c", "t"),
            covariates={"category": category},
            kinds={"category": "categorical"},
        )
    assert raised.value.code == "estimation.cuped_linear.missing_refused"


def test_covariate_constant_within_each_arm_is_rank_deficient():
    arm = np.repeat(["c", "t"], 4)
    x = np.repeat([1.0, 2.0], 4)
    with pytest.raises(InvalidRequestError) as raised:
        fit_cuped_linear(
            np.arange(8.0),
            arm,
            control="c",
            arm_order=("c", "t"),
            covariates={"x": x},
            kinds={"x": "continuous"},
        )
    assert raised.value.code == "estimation.cuped_linear.rank_deficiency"


@pytest.mark.parametrize("counts", [(64, 65), (75, 75)])
def test_large_near_collinear_bm_dof_matches_qr_reference(counts):
    rng = np.random.default_rng(105)
    arm = np.repeat(["c", "t"], counts)
    z = rng.normal(size=len(arm))
    v = rng.normal(size=len(arm))
    x = np.column_stack([z, z + 1e-7 * v])
    y = 4 + (arm == "t") + 0.3 * z + rng.normal(size=len(arm))
    fit = fit_cuped_linear(
        y,
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"z": x[:, 0], "near": x[:, 1]},
        kinds={"z": "continuous", "near": "continuous"},
    )
    groups = [np.flatnonzero(arm == label) for label in ("c", "t")]
    means = [x[group].mean(axis=0) for group in groups]
    design = np.zeros((len(arm), 4))
    weights = np.empty(len(arm))
    for a, group in enumerate(groups):
        design[group, a] = 1
        design[group, 2:] = x[group] - means[a]
        weights[group] = 1 / (len(group) * (len(group) - 1))
    contrast = np.r_[-1.0, 1.0, -(means[1] - means[0])]
    sqrt_weights = np.sqrt(weights)
    q, r = np.linalg.qr(design * sqrt_weights[:, None], mode="reduced")
    ai = sqrt_weights * (q @ np.linalg.solve(r.T, contrast))
    leverage = np.sum(q * q, axis=1)
    hat_transpose = sqrt_weights[:, None] * (q @ q.T) / sqrt_weights[None, :]
    score_matrix = (np.eye(len(arm)) - hat_transpose) * (ai / np.sqrt(1 - leverage))[None, :]
    singular = np.linalg.svd(score_matrix, compute_uv=False)
    expected = (singular @ singular) ** 2 / ((singular**2) @ (singular**2))
    assert fit.dof == pytest.approx(expected, rel=1e-8)


def test_large_well_conditioned_bm_dof_matches_closed_form_reference():
    counts = (35_000, 35_001)
    arm = np.repeat(["c", "t"], counts)
    fit = fit_cuped_linear(np.arange(len(arm), dtype=float), arm, control="c", arm_order=("c", "t"))
    trace = sum(1 / count for count in counts)
    trace_square = sum(1 / (count**2 * (count - 1)) for count in counts)
    assert fit.dof == pytest.approx(trace**2 / trace_square, rel=1e-8)


def test_covariate_arm_offset_difference_survives_scaling():
    base = 2**54
    arm = np.array(["c"] * 4 + ["t"] * 4)
    x = np.r_[base + np.array([0.0, 4.0, 8.0, 12.0]), base + np.array([4.0, 8.0, 12.0, 16.0])]
    y = np.array([0.0, 4.0, 8.0, 12.0, 5.0, 9.0, 13.0, 17.0])
    fit = _fit_repeated_design(
        y,
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"x": x},
        kinds={"x": "continuous"},
    )
    assert fit.absolute("t").estimate == pytest.approx(1.0, abs=1e-12)


def test_ratio_log_effect_survives_large_within_arm_ratio():
    arm = np.array(["c", "c", "t", "t"])
    base = 2**50
    fit = _fit_repeated_design(
        None,
        arm,
        control="c",
        arm_order=("c", "t"),
        numerator=np.array([1.0, 1.0, base + 0.25, base + 0.25]),
        denominator=np.array([1.0, 1.0, base, base]),
    )
    assert fit.ratio("t", scale="log").estimate == pytest.approx(
        math.log1p(2**-52), rel=1e-12, abs=0
    )


def test_homogeneous_numpy_bytes_categories_remain_supported():
    arm = np.repeat(["c", "t"], 4)
    category = np.array([b"a", b"a", b"b", b"b"] * 2, dtype="S1")
    fit = _fit_repeated_design(
        np.arange(8.0),
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"category": category},
        kinds={"category": "categorical"},
    )
    assert fit.layout.levels == (("a", "b"),)


@pytest.mark.parametrize("denominator_scale", [1.0, 3.0, 5.0, 2**100])
def test_ratio_cross_product_log_effect_is_denominator_unit_invariant(denominator_scale):
    arm = np.array(["c", "c", "t", "t"])
    base = 2**50
    fit = _fit_repeated_design(
        None,
        arm,
        control="c",
        arm_order=("c", "t"),
        numerator=np.array([1.0, 1.0, base + 0.25, base + 0.25]),
        denominator=np.array(
            [
                denominator_scale,
                denominator_scale,
                base * denominator_scale,
                base * denominator_scale,
            ]
        ),
    )
    assert fit.ratio("t", scale="log").estimate == pytest.approx(
        math.log1p(2**-52), rel=1e-12, abs=0
    )


@pytest.mark.parametrize(
    "observable", ["p", "k_max", "leverage_ratio", "max_leverage", "ratio_denominator_precision"]
)
def test_m1_support_refusal_retains_exact_observable_contract(observable):
    rng = np.random.default_rng(501)
    n_arm, width = (1030, 204) if observable == "p" else (40, 1)
    if observable == "k_max":
        n_arm, width = 1206, 0
    elif observable == "leverage_ratio":
        n_arm, width = 10, 3
    arms = np.repeat(["c", "t"], n_arm)
    x = rng.normal(size=(2 * n_arm, width))
    if observable == "leverage_ratio":
        phase = np.tile(np.arange(n_arm) * 2 * np.pi / n_arm, 2)
        x = np.column_stack((np.sin(phase), np.cos(phase), np.sin(2 * phase)))
    covariates = {f"x{i}": x[:, i] for i in range(width)}
    kinds = dict.fromkeys(covariates, "continuous")
    if observable == "k_max":
        covariates = {"category": np.tile(np.repeat(np.arange(201).astype(str), 6), 2)}
        kinds = {"category": "categorical"}
    if observable == "max_leverage":
        covariates["x0"][:] = 0
        covariates["x0"][[0, n_arm]] = 100
    y = 10 + rng.normal(size=2 * n_arm)
    options = {}
    if observable == "ratio_denominator_precision":
        denominator = np.tile(np.r_[np.ones(n_arm // 2), np.full(n_arm // 2, 1000.0)], 2)
        covariates = {"x0": denominator}
        options = {"numerator": y, "denominator": denominator}
        y = None
    with pytest.raises(InvalidRequestError) as raised:
        fit_cuped_linear(
            y,
            arms,
            control="c",
            arm_order=("c", "t"),
            covariates=covariates,
            kinds=kinds,
            numerator=options.get("numerator"),
            denominator=options.get("denominator"),
        )
    error = raised.value
    assert error.code == "estimation.cuped_linear.support_region"
    assert set(error.context) == {"region", "violations"}
    assert error.context["region"] == "M1"
    raw_violations = error.context["violations"]
    assert isinstance(raw_violations, tuple)
    violations = {
        name: (value, bound)
        for name, value, bound in cast(tuple[tuple[str, float, float], ...], raw_violations)
    }
    assert set(violations) == {observable}
    assert (
        violations[observable][1]
        == {
            "p": 203,
            "k_max": 200,
            "leverage_ratio": 0.20,
            "max_leverage": 0.45,
            "ratio_denominator_precision": 0.15,
        }[observable]
    )
    assert violations[observable][0] > violations[observable][1]
    for restored in (copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
        assert restored.code == error.code and restored.context == error.context
        with pytest.raises(TypeError):
            cast(dict[str, object], restored.context)["region"] = "changed"


def test_m1_inside_every_bound_preserves_mean_and_ratio_inference():
    rng = np.random.default_rng(502)
    arms = np.repeat(["c", "t"], 40)
    category = np.tile(np.repeat(["a", "b", "c", "d"], 10), 2)
    covariates = {"category": category, "x": rng.normal(size=80)}
    kinds = {"category": "categorical", "x": "continuous"}
    y = 10 + (arms == "t") + rng.normal(size=80)
    for options in ({}, {"numerator": y, "denominator": 5 + rng.uniform(size=80)}):
        fit = fit_cuped_linear(
            None if options else y,
            arms,
            control="c",
            arm_order=("c", "t"),
            covariates=covariates,
            kinds=kinds,
            numerator=options.get("numerator"),
            denominator=options.get("denominator"),
        )
        assert fit.p <= 203
        assert max(len(levels or ()) for levels in fit.layout.levels) <= 200
        assert fit.p / fit.n_min <= 0.20
        assert fit.max_leverage <= 0.45
        projection = fit.ratio("t") if options else fit.relative("t")
        assert projection.lower < projection.estimate < projection.upper


def test_nonpositive_raw_denominator_arm_is_refused_under_the_fit_code():
    arms = np.repeat(["c", "t", "z"], 8)
    numerator = np.arange(24, dtype=float) + 1
    denominator = np.r_[np.full(16, 2.0), np.zeros(8)]
    with pytest.raises(InvalidRequestError) as raised:
        fit_cuped_linear(
            None,
            arms,
            control="c",
            arm_order=("c", "t", "z"),
            numerator=numerator,
            denominator=denominator,
        )
    assert raised.value.code == "estimation.cuped_linear.nonpositive_component"
    assert raised.value.context == {"family": "denominator", "arm": "z", "value": 0.0}


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_median_centered_arm_mean_near_overflow_keeps_zero_contrast(sign):
    arm = np.array(["c"] * 4 + ["t"] * 4)
    y = sign * np.tile([0.0, 0.0, 1e308, 1e308], 2)
    fit = fit_cuped_linear(y, arm, control="c", arm_order=("c", "t"))
    result = fit.absolute("t")
    assert result.estimate == 0
    assert result.se == pytest.approx(1e308 / math.sqrt(6), rel=1e-12)
    assert result.dof == pytest.approx(6.0)
    assert fit.adjusted_means == pytest.approx((sign * 5e307, sign * 5e307), rel=1e-12)
    if sign > 0:
        assert fit.relative("t").estimate == 0


def test_slope_centering_near_overflow_is_power_of_two_scale_equivariant():
    arm = np.array(["c"] * 4 + ["t"] * 4)
    x = np.tile([0.0, 1.0, 2.0, 3.0], 2)
    y = np.r_[[0.0, 0.0, 1e308, 1e308], [0.0, 1e308, 1e308, 1e308]]
    fits = [
        _fit_repeated_design(
            y * factor,
            arm,
            control="c",
            arm_order=("c", "t"),
            covariates={"x": x},
            kinds={"x": "continuous"},
        )
        for factor in (1.0, 2.0**-200)
    ]
    large, small = (fit.absolute("t") for fit in fits)
    assert large.estimate == pytest.approx(small.estimate * 2.0**200, rel=1e-12)
    assert large.se == pytest.approx(small.se * 2.0**200, rel=1e-12)
    assert large.dof == pytest.approx(small.dof, rel=1e-12)
    assert fits[0].theta[0] == pytest.approx(fits[1].theta[0] * 2.0**200, rel=1e-12)


@pytest.mark.parametrize("denominator", [1.0, 3.0, 2.0**100])
def test_constant_denominator_ratio_contrasts_match_mean_contrasts(denominator):
    base = 2**54
    arm = np.array(["c"] * 4 + ["t"] * 4)
    numerator = np.r_[base + np.array([0.0, 0.0, 0.0, 4.0]), base + np.array([0.0, 0.0, 4.0, 4.0])]
    ratio_fit = fit_cuped_linear(
        None,
        arm,
        control="c",
        arm_order=("c", "t"),
        numerator=numerator,
        denominator=np.full(8, denominator),
    )
    mean_fit = fit_cuped_linear(numerator, arm, control="c", arm_order=("c", "t"))
    absolute, log = ratio_fit.ratio("t", scale="absolute"), ratio_fit.ratio("t", scale="log")
    expected_absolute, expected_log = mean_fit.absolute("t"), mean_fit.relative("t")
    assert expected_absolute.estimate == pytest.approx(1.0)
    assert absolute.estimate == pytest.approx(1.0 / denominator, rel=1e-12, abs=0)
    assert absolute.se == pytest.approx(expected_absolute.se / denominator, rel=1e-12)
    assert absolute.dof == pytest.approx(expected_absolute.dof, rel=1e-9)
    assert log.estimate == pytest.approx(math.log1p(1 / (base + 1)), rel=1e-12, abs=0)
    assert log.estimate == pytest.approx(expected_log.estimate, rel=1e-12, abs=0)
    assert log.se == pytest.approx(expected_log.se, rel=1e-12)
    assert log.dof == pytest.approx(expected_log.dof, rel=1e-9)


def test_constant_within_arm_subnormal_outcomes_project_a_finite_log_ratio():
    arm = np.repeat(["c", "t"], 3)
    y = np.r_[np.full(3, 1e-310), np.full(3, 2e-310)]
    fit = fit_cuped_linear(y, arm, control="c", arm_order=("c", "t"))
    result = fit.relative("t")
    assert result.estimate == pytest.approx(math.log(y[3] / y[0]), rel=1e-12, abs=0)
    assert result.se == 0
    assert math.isfinite(result.dof) and result.dof > 0
    ratio_fit = fit_cuped_linear(
        None,
        arm,
        control="c",
        arm_order=("c", "t"),
        numerator=y,
        denominator=np.full(6, 1e-310),
    )
    assert ratio_fit.ratio("t", scale="log").estimate == pytest.approx(
        math.log(y[3] / y[0]), rel=1e-12, abs=0
    )
    assert ratio_fit.ratio("t", scale="absolute").estimate == pytest.approx(
        y[3] / y[0] - 1, rel=1e-12, abs=0
    )
    assert ratio_fit.ratio("t", scale="log").se == 0
    assert ratio_fit.ratio("t", scale="absolute").se == 0


# Subnormal outcomes carry at most 44 significant bits at 1e-310 and lose
# further bits in each arm-mean subtraction, so a relative tolerance near
# 1e-10 is the representability bound, not an estimator slack.
@pytest.mark.parametrize("factor", [2.0**-1030, 2.0**1000])
def test_log_scale_projections_are_invariant_through_extreme_outcome_scales(factor):
    arm = np.repeat(["c", "t"], 4)
    base = np.array([1.0, 1.1, 0.9, 1.05, 2.0, 2.2, 1.9, 2.05])
    denominator = np.array([1.0, 1.2, 0.8, 1.1, 1.05, 1.15, 0.95, 1.0])
    reference = fit_cuped_linear(base, arm, control="c", arm_order=("c", "t"))
    scaled = fit_cuped_linear(base * factor, arm, control="c", arm_order=("c", "t"))
    for field in ("estimate", "se", "dof"):
        assert getattr(scaled.relative("t"), field) == pytest.approx(
            getattr(reference.relative("t"), field), rel=1e-10
        )
    reference_ratio = fit_cuped_linear(
        None,
        arm,
        control="c",
        arm_order=("c", "t"),
        numerator=base,
        denominator=denominator,
    )
    scaled_ratio = fit_cuped_linear(
        None,
        arm,
        control="c",
        arm_order=("c", "t"),
        numerator=base * factor,
        denominator=denominator * factor,
    )
    for scale in ("log", "absolute"):
        for field in ("estimate", "se", "dof"):
            assert getattr(scaled_ratio.ratio("t", scale=scale), field) == pytest.approx(
                getattr(reference_ratio.ratio("t", scale=scale), field), rel=1e-10
            ), (scale, field)


def test_overflowing_adjusted_means_keep_representable_contrasts_and_refuse_the_rest():
    arm = np.repeat(["c", "t", "z"], 8)
    x = np.r_[np.tile([-0.25, 0.25], 4), np.tile([-0.25, 0.25], 4), np.tile([5.75, 6.25], 4)]
    y = np.r_[
        np.tile([1e308, 1.2e308], 4),
        np.tile([1e308, 1.2e308], 4),
        np.tile([-1.2e308, -1e308], 4),
    ]
    fit = fit_cuped_linear(
        y,
        arm,
        control="c",
        arm_order=("c", "t", "z"),
        covariates={"x": x},
        kinds={"x": "continuous"},
    )
    assert fit.theta[0] == pytest.approx(4e307, rel=1e-12)
    result = fit.absolute("t")
    assert result.estimate == 0
    assert result.se == pytest.approx(0.0, abs=1e308 * 2**-40)
    assert math.isfinite(result.dof) and result.dof > 0
    assert fit.relative("t").estimate == 0
    with pytest.raises(InvalidRequestError) as raised:
        fit.absolute("z")
    assert raised.value.code == "estimation.cuped_linear.nonfinite_projection"


def test_arm_mean_at_the_float_ceiling_keeps_finite_projections():
    largest = np.finfo(float).max
    unit = 2.0**971
    arm = np.repeat(["c", "t"], 4)
    y = np.tile([largest - 3 * unit, largest - 3 * unit, largest, largest], 2)
    fit = fit_cuped_linear(y, arm, control="c", arm_order=("c", "t"))
    absolute = fit.absolute("t")
    assert absolute.estimate == 0
    assert absolute.se == pytest.approx(unit * math.sqrt(1.5), rel=1e-12)
    assert absolute.dof == pytest.approx(6.0)
    relative = fit.relative("t")
    assert relative.estimate == 0
    assert relative.se == pytest.approx(absolute.se / fit.adjusted_means[0], rel=1e-12)


def test_log_scale_uncertainty_survives_outcomes_at_the_smallest_subnormal():
    tiny = 2.0**-1074
    arm = np.repeat(["c", "t"], 4)
    base = np.tile([1.0, 1.0, 3.0, 3.0], 2)
    reference = fit_cuped_linear(base, arm, control="c", arm_order=("c", "t"))
    fit = fit_cuped_linear(base * tiny, arm, control="c", arm_order=("c", "t"))
    assert reference.relative("t").se == pytest.approx(1 / math.sqrt(6), rel=1e-12)
    for field in ("estimate", "se", "dof"):
        assert getattr(fit.relative("t"), field) == pytest.approx(
            getattr(reference.relative("t"), field), rel=1e-12
        ), field
    reference_ratio = fit_cuped_linear(
        None, arm, control="c", arm_order=("c", "t"), numerator=base, denominator=np.ones(8)
    )
    ratio = fit_cuped_linear(
        None,
        arm,
        control="c",
        arm_order=("c", "t"),
        numerator=base * tiny,
        denominator=np.full(8, tiny),
    )
    for scale in ("log", "absolute"):
        for field in ("estimate", "se", "dof"):
            assert getattr(ratio.ratio("t", scale=scale), field) == pytest.approx(
                getattr(reference_ratio.ratio("t", scale=scale), field), rel=1e-12
            ), (scale, field)


@pytest.mark.parametrize("permutation", [np.arange(6), np.array([5, 0, 1, 2, 3, 4])])
def test_arm_mean_keeps_small_contributions_beside_cancelling_extremes(permutation):
    arm = np.repeat(["c", "t"], 6)
    treatment = np.array([-1e308, 1e308, 0.0, 0.0, 0.0, 1e-20])[permutation]
    fit = fit_cuped_linear(np.r_[np.zeros(6), treatment], arm, control="c", arm_order=("c", "t"))
    assert fit.adjusted_means[1] == pytest.approx(1e-20 / 6, rel=1e-15, abs=0)
    assert fit.absolute("t").estimate == pytest.approx(1e-20 / 6, rel=1e-15, abs=0)
