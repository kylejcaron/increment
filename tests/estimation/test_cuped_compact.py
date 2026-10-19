from __future__ import annotations

import math
import pickle
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy import stats

from increment.errors import InvalidRequestError
from increment.estimation.cuped import fit_cuped_multicovariate
from increment.estimation.cuped.encoding import MulticovariateCupedLayout
from increment.estimation.cuped.inference import _bm_terms, finalize_fit
from increment.estimation.cuped.types import (
    CategoricalCells,
    FitState,
    InferenceSummary,
    ProjectionSpec,
)


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
    with pytest.raises(InvalidRequestError) as raised:
        ProjectionSpec(
            identity=("fit-a",), kind="absolute", jacobian=(), scale=1.0, divisor=1.0, exponent=0
        )
    assert raised.value.code == "estimation.cuped_linear.invalid_evidence"
    left = InferenceSummary(("fit-a",), ("all",), 0.2, 4.0, (), ((1.0,),), ())
    right = InferenceSummary(("fit-b",), ("all",), 0.2, 4.0, (), ((1.0,),), ())
    with pytest.raises(InvalidRequestError) as raised:
        left.merge(right)
    assert raised.value.code == "estimation.cuped_linear.incompatible_evidence"
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


def _minimal_finalizer_evidence():
    identity = ("compact-test-fit",)
    arms = ("c", "t")
    layout = MulticovariateCupedLayout((), (), (), (), (), (), ())
    state = FitState(
        identity=identity,
        eligible_arms=arms,
        withheld_arms=(),
        refusal_contexts=(),
        counts=(3, 3),
        layout=layout,
        offsets=(),
        scales=(),
        centered_means=((), ()),
        anchor=(),
        triangular_factor=((1.0, 0.0), (0.0, 1.0)),
        transformed_rhs=((),),
        slopes=((),),
        adjusted_means=(2.0, -3.0),
        theta=(),
        n_min=3,
        p=0,
        is_ratio=False,
        control="c",
    )
    projection_records = (
        ("absolute", "t", "c", "", 1.0, 0.0, 0.0),
        ("absolute", "c", "t", "", -1.0, 0.0, 0.0),
    )
    matrix = tuple(tuple(0.0 for _ in range(4)) for _ in range(4))
    summary = InferenceSummary(
        fit_identity=identity,
        projection_identity=identity + ("all-pairs-v1",),
        max_leverage=0.2,
        sum_leverage=1.2,
        refusal_witness=(),
        adjusted_component_covariance=((1.0, 0.0), (0.0, 1.0)),
        scaled_projected_score_squares=projection_records,
        s2=((0.0,), (0.0,)),
        s4=((0.0,), (0.0,)),
        g=(((matrix),), ((matrix),)),
        g2=(((matrix),), ((matrix),)),
        u=((0.0, 0.0), (0.0, 0.0)),
        evidence_identity=identity,
        projection_specs=tuple(
            ProjectionSpec(
                identity=identity + row[:4],
                kind=row[0],
                jacobian=(1.0, -1.0) if row[1] == "t" else (-1.0, 1.0),
                scale=1.0,
                divisor=1.0,
                exponent=0,
                arm=row[1],
                control=row[2],
            )
            for row in projection_records
        ),
    )
    return state, summary


def test_finalizer_fit_dof_uses_first_noncontrol_arm_against_designated_control():
    order = ("a", "b", "c")
    arms = np.repeat(order, (20, 25, 30))
    row = np.arange(len(arms), dtype=float)
    x = np.sin(0.37 * row)
    y = 8.0 + 0.4 * x + np.cos(0.61 * row)
    groups = [np.flatnonzero(arms == arm) for arm in order]
    means_x = np.array([x[index].mean() for index in groups])
    centered_x = x - np.array([means_x[order.index(arm)] for arm in arms])
    weights = np.array(
        [
            1.0 / (len(groups[order.index(arm)]) * (len(groups[order.index(arm)]) - 1))
            for arm in arms
        ]
    )
    design = np.zeros((len(arms), len(order) + 1))
    for index, arm in enumerate(arms):
        design[index, order.index(arm)] = 1.0
    design[:, -1] = centered_x
    contrast = np.zeros(len(order) + 1)
    contrast[order.index("a")] = 1.0
    contrast[order.index("c")] = -1.0
    contrast[-1] = -(means_x[order.index("a")] - means_x[order.index("c")])
    inverse = np.linalg.inv(design.T @ (design * weights[:, None]))
    hat = design @ inverse @ (design * weights[:, None]).T
    leverage = np.diag(hat)
    influence = design @ (inverse @ contrast) * weights
    dense = (np.eye(len(arms)) - hat).T * (influence / np.sqrt(1.0 - leverage))[None, :]
    eigenvalues = np.linalg.eigvalsh(dense.T @ dense)
    expected = eigenvalues.sum() ** 2 / (eigenvalues @ eigenvalues)

    fit = fit_cuped_multicovariate(
        y, arms, control="c", arm_order=order, covariates={"x": x}, kinds={"x": "continuous"}
    )
    assert fit.dof == pytest.approx(expected, rel=2e-10, abs=1e-11)


def test_finalizer_rejects_truncated_projection_roster():
    state, summary = _minimal_finalizer_evidence()
    with pytest.raises(InvalidRequestError) as raised:
        finalize_fit(
            state,
            replace(
                summary,
                scaled_projected_score_squares=summary.scaled_projected_score_squares[:1],
                s2=summary.s2[:1],
                s4=summary.s4[:1],
                g=summary.g[:1],
                g2=summary.g2[:1],
                projection_specs=summary.projection_specs[:1],
            ),
        )
    assert raised.value.code == "estimation.cuped_linear.invalid_evidence"


def test_finalizer_rejects_invalid_bm_trace_square():
    state, summary = _minimal_finalizer_evidence()
    with pytest.raises(InvalidRequestError) as raised:
        finalize_fit(state, replace(summary, s4=((-1.0,), (0.0,))))
    assert raised.value.code == "estimation.cuped_linear.invalid_evidence"


def test_finalizer_enforces_m1_support_witness():
    state, summary = _minimal_finalizer_evidence()
    with pytest.raises(InvalidRequestError) as raised:
        finalize_fit(
            state,
            replace(
                summary,
                max_leverage=0.46,
                refusal_witness=(("max_leverage", 0.46, 0.45),),
            ),
        )
    assert raised.value.code == "estimation.cuped_linear.support_region"
    assert raised.value.context["region"] == "M1"


def test_frozen_evidence_recursively_copies_mutable_payloads():
    state, summary = _minimal_finalizer_evidence()
    covariance = np.array([[1.0, 0.0], [0.0, 1.0]])
    frozen_summary = replace(summary, adjusted_component_covariance=covariance)
    covariance[0, 0] = 99.0
    assert frozen_summary.adjusted_component_covariance == ((1.0, 0.0), (0.0, 1.0))
    assert isinstance(frozen_summary.adjusted_component_covariance, tuple)

    counts = np.array([3, 3])
    frozen_state = replace(state, counts=counts)
    counts[0] = 99
    assert frozen_state.counts == (3, 3)

    jacobian: Any = np.array([1.0, -1.0])
    spec = ProjectionSpec(("identity",), "absolute", jacobian, 1.0, 1.0, 0)
    jacobian[0] = 99.0
    assert spec.jacobian == (1.0, -1.0)

    layout_args: Any = (
        ["category"],
        ["categorical"],
        [["a"]],
        [],
        ["category=a"],
        [],
        [],
    )
    layout = MulticovariateCupedLayout(*layout_args)
    frozen_layout_state = replace(state, layout=layout)
    assert frozen_layout_state.layout.names == ("category",)
    assert frozen_layout_state.layout.levels == (("a",),)

    cell_counts: Any = np.array([[2], [3]])
    cell_evidence: Any = {
        "identity": ["cells"],
        "arms": ["c", "t"],
        "patterns": [["a"]],
        "counts": cell_counts,
        "levels": [["a"]],
        "null_metadata": [[False]],
        "outcome_references": [],
        "centered_first_moments": [[], []],
        "symmetric_cross_moments": [[], []],
        "outcome_exponents": [],
    }
    cells = CategoricalCells(**cell_evidence)
    cell_counts[0, 0] = 99
    assert cells.counts == ((2,), (3,))


def test_unavailable_nan_evidence_merges_after_pickle_replay():
    summary = InferenceSummary(("nan-fit",), ("all-pairs",), 0.2, 1.0, (), ((float("nan"),),), ())
    replayed = pickle.loads(pickle.dumps(summary))
    assert summary.merge(replayed) is summary
    with pytest.raises(InvalidRequestError) as raised:
        summary.merge(replace(summary, adjusted_component_covariance=((1.0,),)))
    assert raised.value.code == "estimation.cuped_linear.invalid_evidence"


def test_lowrank_bm_trace_square_is_stable_near_leverage_saturation():
    design = np.array([[1.0, 0.0], [0.0, 1.0], [1e-4, 1e-4]])
    weights = np.ones(3)
    contrast = np.array([1.0, -0.3])
    q, factor = np.linalg.qr(design, mode="reduced")
    leverage = np.einsum("ij,ij->i", q, q)
    trace, trace_square = _bm_terms(
        design,
        weights,
        leverage,
        factor,
        contrast,
        scale=1.0,
        q_design=q,
    )
    influence = q @ np.linalg.solve(factor.T, contrast)
    weighted_score = influence / np.sqrt(1.0 - leverage)
    dense = (np.eye(len(weights)) - q @ q.T).T * weighted_score[None, :]
    eigenvalues = np.linalg.eigvalsh(dense.T @ dense)
    assert trace == pytest.approx(eigenvalues.sum(), rel=2e-8, abs=1e-10)
    assert trace_square == pytest.approx(eigenvalues @ eigenvalues, rel=2e-10, abs=1e-11)


def test_small_n_bm_fallback_matches_weighted_dense_reference():
    design = np.array([[1.0, 0.0], [0.0, 1.0], [1e-4, 1e-4], [1e-4, -1e-4]])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    contrast = np.array([1.0, -0.3])
    roots = np.sqrt(weights)
    q, factor = np.linalg.qr(design * roots[:, None], mode="reduced")
    leverage = np.einsum("ij,ij->i", q, q)
    trace, trace_square = _bm_terms(
        design,
        weights,
        leverage,
        factor,
        contrast,
        scale=1.0,
        q_design=q,
    )
    coefficients = np.linalg.solve(factor.T, contrast)
    score = roots * (q @ coefficients) / np.sqrt(1.0 - leverage)
    weighted_hat_transpose = roots[:, None] * (q @ q.T) / roots[None, :]
    dense = (np.eye(len(weights)) - weighted_hat_transpose) * score[None, :]
    eigenvalues = np.linalg.eigvalsh(dense.T @ dense)
    assert leverage.max() == pytest.approx(0.999999965, abs=1e-9)
    assert trace == pytest.approx(eigenvalues.sum(), rel=2e-10, abs=1e-11)
    assert trace_square == pytest.approx(eigenvalues @ eigenvalues, rel=2e-10, abs=1e-11)


def test_ratio_projection_spec_retains_scaled_jacobian_for_tiny_units(monkeypatch):
    from increment.estimation.cuped import multicovariate

    original_finalize = multicovariate.finalize_fit
    captured = {}

    def capture_finalize(state, summary):
        captured["summary"] = summary
        return original_finalize(state, summary)

    monkeypatch.setattr(multicovariate, "finalize_fit", capture_finalize)
    arms = np.repeat(["c", "t"], 20)
    row = np.arange(len(arms), dtype=float)
    numerator = 1e-200 * (3.0 + 0.1 * np.sin(row))
    denominator = 1e-200 * (2.0 + 0.05 * np.cos(row))
    fit = fit_cuped_multicovariate(
        None,
        arms,
        control="c",
        arm_order=("c", "t"),
        numerator=numerator,
        denominator=denominator,
    )
    projection = next(
        item
        for item in captured["summary"].projection_specs
        if item.kind == "ratio"
        and item.arm == "t"
        and item.control == "c"
        and item.identity[-1] == "absolute"
    )
    assert fit.ratio("t", scale="absolute").estimate == pytest.approx(
        0.0038545262890675876, rel=2e-10, abs=1e-11
    )
    assert fit.ratio("t", scale="absolute").se == pytest.approx(
        0.014151358400968225, rel=2e-10, abs=1e-11
    )
    assert projection.available
    assert np.isfinite(projection.jacobian).all()
    assert projection.divisor != 1.0 or projection.exponent != 0
    component_means = np.asarray(fit.adjusted_means).reshape(2, 2)
    nc, nt = component_means[0]
    dc, dt = component_means[1]
    reconstructed = np.array(
        [
            math.ldexp(value * projection.scale / projection.divisor, projection.exponent)
            for value in projection.jacobian
        ]
    )
    expected_jacobian = np.array([-1.0 / dc, 1.0 / dt, nc / dc / dc, -nt / dt / dt])
    assert reconstructed == pytest.approx(expected_jacobian, rel=2e-10, abs=1e-11)


def _crossed_category_case(is_ratio=False):
    order = ("c", "t", "u")
    patterns = (("A", "desktop"), ("A", "mobile"), ("B", "desktop"), ("B", "mobile"))
    counts = ((17, 8, 11, 6), (7, 19, 9, 13), (12, 6, 21, 10))
    arms, country, device, numerator, denominator = [], [], [], [], []
    for a, arm in enumerate(order):
        for k, (segment, platform) in enumerate(patterns):
            for row in range(counts[a][k]):
                noise = math.sin(0.71 * row + 0.3 * k + a)
                arms.append(arm)
                country.append(segment)
                device.append(platform)
                numerator.append(8.0 + a * 0.6 + k * 0.4 + noise)
                denominator.append(3.0 + a * 0.1 + k * 0.2 + 0.2 * noise + 0.1 * math.cos(row))
    values = {
        "y": None if is_ratio else np.array(numerator),
        "arms": np.array(arms),
        "control": "c",
        "arm_order": order,
        "covariates": {"segment": np.array(country), "device": np.array(device)},
    }
    if is_ratio:
        values.update(numerator=np.array(numerator), denominator=np.array(denominator))
    return values


def _assert_cell_fit_parity(actual, expected):
    for name in (
        "adjusted_means",
        "theta",
        "anchor",
        "covariance",
        "dof",
        "max_leverage",
        "mean_leverage",
        "leverage_trace",
    ):
        assert np.asarray(getattr(actual, name)) == pytest.approx(
            np.asarray(getattr(expected, name)), rel=1e-10, abs=1e-12
        )
    assert actual.layout == expected.layout
    assert actual.n_by_arm == expected.n_by_arm
    assert len(actual._projections) == len(expected._projections)
    for left, right in zip(actual._projections, expected._projections, strict=True):
        assert left[:4] == right[:4]
        assert left[4:] == pytest.approx(right[4:], rel=1e-10, abs=1e-12, nan_ok=True)
        kind, arm, control, scale = right[:4]
        if actual.is_ratio and kind == "relative":
            for fit in (actual, expected):
                with pytest.raises(InvalidRequestError) as raised:
                    fit.relative(arm, control=control)
                assert raised.value.code == "estimation.cuped_linear.projection_kind"
            continue
        for alpha in (0.01, 0.2):
            kwargs = {"control": control, "alpha": alpha}
            if kind == "ratio":
                kwargs["scale"] = scale
            left_interval = getattr(actual, kind)(arm, **kwargs)
            right_interval = getattr(expected, kind)(arm, **kwargs)
            for field in ("estimate", "se", "dof", "lower", "upper"):
                assert getattr(left_interval, field) == pytest.approx(
                    getattr(right_interval, field), rel=1e-10, abs=1e-12
                )


@pytest.mark.parametrize("is_ratio", [False, True])
def test_joint_categorical_cells_match_unit_hc2_bm_and_all_projections(is_ratio):
    from increment.estimation.cuped.cells import categorical_cells, fit_categorical_cells

    values = _crossed_category_case(is_ratio)
    expected = fit_cuped_multicovariate(
        **values, kinds={"segment": "categorical", "device": "categorical"}
    )
    cells = categorical_cells(**values, identity=("snapshot", "metric", "population"))
    assert len(cells.patterns) == 4
    assert cells.counts == ((17, 8, 11, 6), (7, 19, 9, 13), (12, 6, 21, 10))
    actual = fit_categorical_cells(cells, control="c")
    _assert_cell_fit_parity(actual, expected)


@pytest.mark.parametrize("is_ratio", [False, True])
def test_joint_categorical_partition_merge_refits_with_centered_moments(is_ratio):
    import json
    from dataclasses import asdict

    from increment.estimation.cuped.cells import categorical_cells, fit_categorical_cells

    values = _crossed_category_case(is_ratio)
    if not is_ratio:
        values["y"] += 1e8
    parts = []
    for remainder in range(3):
        mask = np.arange(len(values["arms"])) % 3 == remainder
        partition = {
            name: (
                {key: column[mask] for key, column in value.items()}
                if name == "covariates"
                else value[mask]
                if isinstance(value, np.ndarray)
                else value
            )
            for name, value in values.items()
        }
        part = categorical_cells(**partition, identity=("same-snapshot",))
        parts.append(CategoricalCells(**json.loads(json.dumps(asdict(part)))))
    merged = parts[2].merge(parts[0]).merge(parts[1])
    complete = categorical_cells(**values, identity=("same-snapshot",))
    _assert_cell_fit_parity(
        fit_categorical_cells(merged, control="c"), fit_categorical_cells(complete, control="c")
    )
    with pytest.raises(InvalidRequestError) as raised:
        complete.merge(replace(complete, identity=("different-snapshot",)))
    assert raised.value.code == "estimation.cuped_linear.incompatible_evidence"


def test_marginal_categorical_payload_cannot_be_consumed_as_joint_cells():
    from increment.estimation.cuped.cells import categorical_cells, fit_categorical_cells

    complete = categorical_cells(**_crossed_category_case(), identity=("joint",))
    with pytest.raises(InvalidRequestError) as raised:
        fit_categorical_cells(
            replace(complete, patterns=(("A",), ("B",), ("desktop",), ("mobile",))),
            control="c",
        )
    assert raised.value.code == "estimation.cuped_linear.invalid_evidence"


def test_joint_cells_preserve_null_category_and_serialized_failure_context():
    import json
    from dataclasses import asdict

    from increment.estimation.cuped.cells import categorical_cells, fit_categorical_cells

    values = _crossed_category_case()
    segment = values["covariates"]["segment"].astype(object)
    segment[segment == "B"] = None
    values["covariates"]["segment"] = segment
    values["missing"] = {"segment": "impute-indicator"}
    complete = categorical_cells(**values, identity=("nullable",))
    restored = CategoricalCells(**json.loads(json.dumps(asdict(complete))))
    _assert_cell_fit_parity(
        fit_categorical_cells(restored, control="c"),
        fit_cuped_multicovariate(
            **values, kinds={"segment": "categorical", "device": "categorical"}
        ),
    )
    arms = np.array(["c"] * 16 + ["t"] * 16)
    category = np.array(["rare"] + ["common"] * 15 + ["rare"] * 8 + ["common"] * 8)
    evidence = categorical_cells(
        np.arange(len(arms), dtype=float),
        arms,
        control="c",
        arm_order=("c", "t"),
        covariates={"segment": category},
        identity=("rare",),
    )
    for payload in (evidence, CategoricalCells(**json.loads(json.dumps(asdict(evidence))))):
        with pytest.raises(InvalidRequestError) as raised:
            fit_categorical_cells(payload, control="c")
        failure = pickle.loads(pickle.dumps(raised.value))
        assert failure.code == "estimation.cuped_linear.level_support"
        assert failure.context["column"] == "segment"
        assert failure.context["level"] == "rare"
        assert failure.context["counts"] == {"c": 1, "t": 8}


def test_categorical_cells_merge_singleton_partitions_without_changing_support():
    from increment.estimation.cuped.cells import categorical_cells, fit_categorical_cells

    values = _crossed_category_case()
    parts = []
    for mask in (np.arange(len(values["arms"])) == 0, np.arange(len(values["arms"])) != 0):
        partition = {
            **values,
            "y": values["y"][mask],
            "arms": values["arms"][mask],
            "covariates": {name: column[mask] for name, column in values["covariates"].items()},
        }
        parts.append(categorical_cells(**partition, identity=("partitioned",)))
    complete = categorical_cells(**values, identity=("partitioned",))
    _assert_cell_fit_parity(
        fit_categorical_cells(parts[0].merge(parts[1]), control="c"),
        fit_categorical_cells(complete, control="c"),
    )


@pytest.mark.parametrize("unit_scale", [1e-200, 1e155])
@pytest.mark.parametrize("ratio", [False, True])
def test_categorical_component_scales_preserve_unit_projection_scores(unit_scale, ratio):
    from increment.estimation.cuped.cells import categorical_cells, fit_categorical_cells

    arms = np.repeat(["c", "t"], 20)
    row = np.arange(len(arms), dtype=float)
    numerator = unit_scale * (3.0 + 0.1 * np.sin(row))
    denominator = unit_scale * (2.0 + 0.05 * np.cos(row))
    arguments: dict[str, Any] = {
        "y": None if ratio else numerator,
        "arms": arms,
        "control": "c",
        "arm_order": ("c", "t"),
        "covariates": {"category": np.repeat("same", len(arms))},
        **({"numerator": numerator, "denominator": denominator} if ratio else {}),
    }
    cells = categorical_cells(**arguments, identity=("scaled-components",))
    expected = fit_cuped_multicovariate(**arguments, kinds={"category": "categorical"})
    actual = fit_categorical_cells(cells, control="c")
    for actual_row, expected_row in zip(actual._projections, expected._projections, strict=True):
        assert actual_row[:4] == expected_row[:4]
        assert actual_row[4:] == pytest.approx(expected_row[4:], rel=1e-10, abs=0)
    if ratio:
        for scale in ("absolute", "log"):
            assert actual.ratio("t", scale=scale).se == pytest.approx(
                expected.ratio("t", scale=scale).se, rel=1e-10, abs=0
            )
    else:
        assert actual.absolute("t").se / unit_scale == pytest.approx(
            expected.absolute("t").se / unit_scale, rel=1e-10, abs=0
        )


def test_categorical_singleton_merge_preserves_representable_large_scatter():
    import math

    from increment.estimation.cuped.cells import categorical_cells

    arguments = {
        "arms": np.array(["c", "t"]),
        "control": "c",
        "arm_order": ("c", "t"),
        "covariates": {"category": np.array(["same", "same"])},
        "identity": ("large-merge",),
    }
    left = categorical_cells(y=np.array([-8e153, -8e153]), **arguments)
    right = categorical_cells(y=np.array([8e153, 8e153]), **arguments)
    merged = left.merge(right)
    complete = categorical_cells(
        y=np.array([-8e153, -8e153, 8e153, 8e153]),
        **{
            **arguments,
            "arms": np.array(["c", "t", "c", "t"]),
            "covariates": {"category": np.repeat("same", 4)},
        },
    )
    for a in range(2):
        scatter = math.ldexp(
            merged.symmetric_cross_moments[a][0][0][0], 2 * merged.outcome_exponents[0]
        )
        assert scatter == pytest.approx(1.28e308, rel=1e-14)
    assert np.asarray(merged.symmetric_cross_moments) == pytest.approx(
        np.asarray(complete.symmetric_cross_moments), rel=1e-14
    )
