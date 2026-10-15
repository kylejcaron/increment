from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from increment.errors import InvalidRequestError
from increment.estimation.cuped import fit_cuped_multicovariate
from increment.estimation.cuped.types import FitState, InferenceSummary, ProjectionSpec


def _direct_mean_reference(y, arms, order, x):
    n, p = x.shape
    indices = [np.flatnonzero(arms == arm) for arm in order]
    counts = np.array([len(rows) for rows in indices])
    means_x = np.array([x[rows].mean(axis=0) for rows in indices])
    means_y = np.array([y[rows].mean() for rows in indices])
    anchor = np.average(means_x, axis=0, weights=counts)
    x_centered = x - np.array([means_x[order.index(arm)] for arm in arms])
    y_centered = y - np.array([means_y[order.index(arm)] for arm in arms])
    weight = np.array(
        [1.0 / (counts[order.index(arm)] * (counts[order.index(arm)] - 1)) for arm in arms]
    )
    gram = x_centered.T @ (x_centered * weight[:, None])
    beta = np.linalg.solve(gram, x_centered.T @ (y_centered * weight))
    adjusted = means_y - (means_x - anchor) @ beta
    residual = y_centered - x_centered @ beta
    design = np.zeros((n, len(order) + p))
    for i, arm in enumerate(arms):
        design[i, order.index(arm)] = 1.0
    design[:, len(order) :] = x_centered
    design_gram = design.T @ (design * weight[:, None])
    design_inverse = np.linalg.inv(design_gram)
    hat = design @ design_inverse @ (design * weight[:, None]).T
    leverage = np.diag(hat)

    def contrast_for(j):
        d = j @ (means_x - anchor)
        return np.concatenate((j, -d))

    def project(j, bm_scale=1.0):
        contrast = contrast_for(j)
        influence = design @ (design_inverse @ contrast) * weight
        score = influence * residual / np.sqrt(1.0 - leverage)
        score += (j.sum() / n) * (x_centered @ beta)
        qscore = influence / (bm_scale * np.sqrt(1.0 - leverage))
        projected = (np.eye(n) - hat).T * qscore[None, :]
        singular = np.linalg.svd(projected, compute_uv=False)
        trace = float(singular @ singular)
        trace_square = float((singular * singular) @ (singular * singular))
        return float(j @ adjusted), score, trace, trace_square

    return adjusted, leverage, project, contrast_for


def _input_case(arm_count, covariate_case):
    order = tuple(f"a{i}" for i in range(arm_count))
    arms = np.repeat(order, 24)
    index = np.arange(len(arms), dtype=float)
    arm_index = np.repeat(np.arange(arm_count), 24)
    continuous = np.sin(index * 0.37) + arm_index * 0.1
    y = 8.0 + arm_index * 0.35 + 0.8 * continuous + np.cos(index * 0.61)
    covariates = {}
    kinds = {}
    x_parts = []
    if covariate_case in {"continuous", "mixed", "missing-indicator"}:
        if covariate_case == "missing-indicator":
            continuous = continuous.copy()
            missing = index.astype(int) % 11 == 0
            continuous[missing] = np.nan
            encoded = np.where(missing, 0.0, continuous)
            covariates["x"] = continuous
            kinds["x"] = "continuous"
            x_parts.extend((encoded, missing.astype(float)))
        else:
            covariates["x"] = continuous
            kinds["x"] = "continuous"
            x_parts.append(continuous)
    if covariate_case in {"categorical", "mixed"}:
        categorical = np.where(index.astype(int) % 4 == 0, "B", "A")
        covariates["category"] = categorical
        kinds["category"] = "categorical"
        x_parts.append((categorical == "B").astype(float))
    x = np.column_stack(x_parts)
    missing_options = {"x": "impute-indicator"} if covariate_case == "missing-indicator" else None
    return y, arms, order, covariates, kinds, missing_options, x


@pytest.mark.parametrize(
    ("arm_count", "covariate_case"),
    [(2, "continuous"), (3, "categorical"), (2, "mixed"), (3, "missing-indicator")],
)
def test_mean_all_retained_projections_match_independent_hc2_bm(arm_count, covariate_case):
    y, arms, order, covariates, kinds, missing, x = _input_case(arm_count, covariate_case)
    fit = fit_cuped_multicovariate(
        y,
        arms,
        control=order[0],
        arm_order=order,
        covariates=covariates,
        kinds=kinds,
        missing=missing,
    )
    adjusted, leverage, project, _ = _direct_mean_reference(y, arms, order, x)
    assert fit.adjusted_means == pytest.approx(adjusted, rel=2e-10, abs=1e-11)
    assert fit.max_leverage == pytest.approx(leverage.max(), rel=2e-10, abs=1e-11)
    for treatment_index in range(1, arm_count):
        for control_index in range(treatment_index):
            j = np.zeros(arm_count)
            j[treatment_index], j[control_index] = 1.0, -1.0
            estimate, score, trace, trace_square = project(j)
            se = float(np.linalg.norm(score))
            dof = trace * trace / trace_square if trace_square > 0 else 1.0
            actual = fit.absolute(order[treatment_index], control=order[control_index])
            assert actual.estimate == pytest.approx(estimate, rel=2e-10, abs=1e-11)
            assert actual.se == pytest.approx(se, rel=2e-10, abs=1e-11)
            assert actual.dof == pytest.approx(dof, rel=2e-10, abs=1e-11)
            half = stats.t.ppf(0.975, dof) * se
            assert actual.lower == pytest.approx(estimate - half, rel=2e-10, abs=1e-11)
            assert actual.upper == pytest.approx(estimate + half, rel=2e-10, abs=1e-11)
            relative_j = np.zeros(arm_count)
            relative_j[treatment_index] = 1.0 / adjusted[treatment_index]
            relative_j[control_index] = -1.0 / adjusted[control_index]
            _, relative_score, rel_trace, rel_trace_square = project(relative_j)
            relative_dof = rel_trace * rel_trace / rel_trace_square
            relative_estimate = float(
                np.log(adjusted[treatment_index]) - np.log(adjusted[control_index])
            )
            relative = fit.relative(order[treatment_index], control=order[control_index])
            assert relative.estimate == pytest.approx(relative_estimate, rel=2e-10, abs=1e-11)
            assert relative.se == pytest.approx(
                np.linalg.norm(relative_score), rel=2e-10, abs=1e-11
            )
            assert relative.dof == pytest.approx(relative_dof, rel=2e-10, abs=1e-11)
            rel_half = stats.t.ppf(0.975, relative_dof) * np.linalg.norm(relative_score)
            assert relative.lower == pytest.approx(
                relative_estimate - rel_half, rel=2e-10, abs=1e-11
            )
            assert relative.upper == pytest.approx(
                relative_estimate + rel_half, rel=2e-10, abs=1e-11
            )


@pytest.mark.parametrize("arm_count", [2, 3])
def test_ratio_log_projections_match_independent_component_hc2_bm(arm_count):
    order = tuple(f"a{i}" for i in range(arm_count))
    arms = np.repeat(order, 28)
    row = np.arange(len(arms), dtype=float)
    arm_index = np.repeat(np.arange(arm_count), 28)
    x = np.sin(row * 0.23) + arm_index * 0.12
    numerator = 5.0 + arm_index * 0.6 + 0.3 * x + 0.4 * np.sin(row * 0.71)
    denominator = 2.5 + arm_index * 0.12 + 0.15 * x + 0.2 * np.cos(row * 0.43)
    fit = fit_cuped_multicovariate(
        None,
        arms,
        control=order[0],
        arm_order=order,
        covariates={"x": x},
        kinds={"x": "continuous"},
        numerator=numerator,
        denominator=denominator,
    )
    nfit, leverage, nproject, ncontrast = _direct_mean_reference(numerator, arms, order, x[:, None])
    dfit, _, dproject, dcontrast = _direct_mean_reference(denominator, arms, order, x[:, None])
    for treatment_index in range(1, arm_count):
        for control_index in range(treatment_index):
            nt, nc = nfit[treatment_index], nfit[control_index]
            dt, dc = dfit[treatment_index], dfit[control_index]
            jn = np.zeros(arm_count)
            jd = np.zeros(arm_count)
            jn[treatment_index], jn[control_index] = 1.0 / nt, -1.0 / nc
            jd[treatment_index], jd[control_index] = -1.0 / dt, 1.0 / dc
            bm_scale = max(
                float(np.max(np.abs(ncontrast(jn)))),
                float(np.max(np.abs(dcontrast(jd)))),
            )
            _, nscore, ntrace, ntrace_square = nproject(jn, bm_scale)
            _, dscore, dtrace, dtrace_square = dproject(jd, bm_scale)
            score = nscore + dscore
            dof = (ntrace + dtrace) ** 2 / (ntrace_square + dtrace_square)
            estimate = float(np.log(nt / dt) - np.log(nc / dc))
            se = float(np.linalg.norm(score))
            actual = fit.ratio(order[treatment_index], control=order[control_index], scale="log")
            assert actual.estimate == pytest.approx(estimate, rel=2e-10, abs=1e-11)
            assert actual.se == pytest.approx(se, rel=2e-10, abs=1e-11)
            assert actual.dof == pytest.approx(dof, rel=2e-10, abs=1e-11)
            half = stats.t.ppf(0.975, dof) * se
            assert actual.lower == pytest.approx(estimate - half, rel=2e-10, abs=1e-11)
            assert actual.upper == pytest.approx(estimate + half, rel=2e-10, abs=1e-11)
            jan = np.zeros(arm_count)
            jad = np.zeros(arm_count)
            jan[treatment_index], jan[control_index] = 1.0 / dt, -1.0 / dc
            jad[treatment_index], jad[control_index] = -(nt / (dt * dt)), nc / (dc * dc)
            absolute_scale = max(
                float(np.max(np.abs(ncontrast(jan)))),
                float(np.max(np.abs(dcontrast(jad)))),
            )
            _, ascore_n, atrace_n, atrace_square_n = nproject(jan, absolute_scale)
            _, ascore_d, atrace_d, atrace_square_d = dproject(jad, absolute_scale)
            absolute_estimate = nt / dt - nc / dc
            absolute_score = ascore_n + ascore_d
            absolute_dof = (atrace_n + atrace_d) ** 2 / (atrace_square_n + atrace_square_d)
            absolute = fit.ratio(
                order[treatment_index], control=order[control_index], scale="absolute"
            )
            assert absolute.estimate == pytest.approx(absolute_estimate, rel=2e-10, abs=1e-11)
            assert absolute.se == pytest.approx(
                np.linalg.norm(absolute_score), rel=2e-10, abs=1e-11
            )
            assert absolute.dof == pytest.approx(absolute_dof, rel=2e-10, abs=1e-11)
            absolute_half = stats.t.ppf(0.975, absolute_dof) * np.linalg.norm(absolute_score)
            assert absolute.lower == pytest.approx(
                absolute_estimate - absolute_half, rel=2e-10, abs=1e-11
            )
            assert absolute.upper == pytest.approx(
                absolute_estimate + absolute_half, rel=2e-10, abs=1e-11
            )


def test_fit_finalizes_compact_immutable_evidence_and_alpha_replay():
    arms = np.array(["c"] * 20 + ["t"] * 22)
    x = np.linspace(-1.0, 2.0, len(arms))
    y = 3.0 + 0.5 * (arms == "t") + 0.7 * x + np.sin(np.arange(len(arms)))
    fit = fit_cuped_multicovariate(
        y,
        arms,
        control="c",
        arm_order=("c", "t"),
        covariates={"x": x},
        kinds={"x": "continuous"},
    )
    assert fit.absolute("t", alpha=0.01).estimate == pytest.approx(
        fit.absolute("t", alpha=0.2).estimate, abs=1e-11
    )
    assert fit.absolute("t", alpha=0.01).se == pytest.approx(
        fit.absolute("t", alpha=0.2).se, rel=2e-10, abs=1e-11
    )
    assert fit.max_leverage < 0.45


def test_low_rank_bell_mccaffrey_matches_direct_small_matrix_witness():
    from increment.estimation.cuped.inference import _bm_terms

    rng = np.random.default_rng(718)
    design = np.column_stack((np.ones(19), rng.normal(size=(19, 2))))
    weights = rng.uniform(0.2, 1.0, size=len(design))
    contrast = np.array([0.7, -0.2, 0.4])
    scale = np.max(np.abs(contrast))
    root = np.sqrt(weights)
    q, r = np.linalg.qr(design * root[:, None], mode="reduced")
    leverage = np.sum(q * q, axis=1)
    low_rank = _bm_terms(design, weights, leverage, r, contrast, scale=scale, q_design=q)
    coefficient = np.linalg.solve(r.T, contrast / scale)
    ai = root * (q @ coefficient)
    t = ai / np.sqrt(1.0 - leverage)
    hat_transpose = root[:, None] * (q @ q.T) / root[None, :]
    projected = (np.eye(len(weights)) - hat_transpose) * t[None, :]
    singular = np.linalg.svd(projected, compute_uv=False)
    trace = float(singular @ singular)
    trace_square = float((singular * singular) @ (singular * singular))
    assert low_rank[0] == pytest.approx(trace, rel=2e-10, abs=1e-11)
    assert low_rank[1] == pytest.approx(trace_square, rel=2e-10, abs=1e-11)


def test_equal_second_moments_do_not_identify_leverage_evidence():
    rng = np.random.default_rng(719)
    q, _ = np.linalg.qr(rng.normal(size=(12, 3)), mode="reduced")
    y = rng.normal(size=12)
    rotation, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    rotated_q = rotation @ q
    rotated_y = rotation @ y
    assert q.T @ q == pytest.approx(rotated_q.T @ rotated_q, abs=1e-11)
    assert q.T @ y == pytest.approx(rotated_q.T @ rotated_y, abs=1e-11)
    assert y @ y == pytest.approx(rotated_y @ rotated_y, abs=1e-11)
    assert not np.allclose(np.sum(q * q, axis=1), np.sum(rotated_q * rotated_q, axis=1))


def test_evidence_types_are_frozen_and_reject_mismatched_summary_identity():
    assert FitState.__dataclass_params__.frozen
    assert InferenceSummary.__dataclass_params__.frozen
    assert ProjectionSpec.__dataclass_params__.frozen
    with pytest.raises(ValueError):
        ProjectionSpec(
            identity=("fit-a",), kind="absolute", jacobian=(), scale=1.0, divisor=1.0, exponent=0
        )
    left = InferenceSummary(("fit-a",), ("all",), 0.2, 4.0, (), ((1.0,),), ())
    right = InferenceSummary(("fit-b",), ("all",), 0.2, 4.0, (), ((1.0,),), ())
    with pytest.raises(ValueError, match="incompatible identities"):
        left.merge(right)
    assert left.merge(left) is left


def test_categorical_cell_failure_preserves_joint_cell_context():
    arms = np.array(["c"] * 16 + ["t"] * 16)
    category = np.array(["rare"] + ["common"] * 15 + ["rare"] * 8 + ["common"] * 8)
    with pytest.raises(InvalidRequestError) as raised:
        fit_cuped_multicovariate(
            np.arange(len(arms), dtype=float),
            arms,
            control="c",
            arm_order=("c", "t"),
            covariates={"segment": category},
            kinds={"segment": "categorical"},
        )
    assert raised.value.code == "estimation.cuped_linear.level_support"
    assert raised.value.context["column"] == "segment"
    assert raised.value.context["level"] == "rare"
    assert raised.value.context["counts"] == {"c": 1, "t": 8}
