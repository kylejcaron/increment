from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from scipy import stats

from increment.errors import InvalidRequestError
from increment.estimation.cuped_linear import fit_cuped_linear


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
    fit = fit_cuped_linear(
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
    assert fit.p == 2 and sum(fit.n_by_arm) == 16


def test_numeric_missing_history_is_retained_and_identical_masks_share_indicator():
    arm = np.array(["c"] * 8 + ["t"] * 8)
    history = np.array([1.0, 3.0, 2.0, 5.0, None, None, None, None] * 2, dtype=object)
    second = np.array([4.0, 1.0, 5.0, 2.0, None, None, None, None] * 2, dtype=object)
    fit = fit_cuped_linear(
        np.arange(16.0),
        arm,
        control="c",
        arm_order=("c", "t"),
        covariates={"history": history, "second": second},
        kinds={"history": "continuous", "second": "continuous"},
        missing={"history": "impute-indicator", "second": "impute-indicator"},
    )
    assert sum(fit.n_by_arm) == 16
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
        ("nonfinite", "estimation.cuped_linear.nonfinite"),
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
    elif case == "df":
        rng = np.random.default_rng(88)
        arm = np.array(["c"] * 3 + ["t"] * 3)
        y = np.arange(6, dtype=float)
        x = rng.normal(size=(6, 4))
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


def test_three_arm_absolute_projection_has_independent_hc2_scores():
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
