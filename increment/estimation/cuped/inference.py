"""HC2 covariance and Bell–McCaffrey inference for mixed CUPED."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn, cast

import numpy as np
from scipy.linalg import qr as scipy_qr

from increment.estimation.cuped._numeric import _fail
from increment.estimation.cuped.types import (
    FitState,
    InferenceSummary,
    MulticovariateCupedFit,
)


def _invalid_evidence(reason: str) -> NoReturn:
    _fail("estimation.cuped_linear.invalid_evidence", reason=reason)


def finalize_fit(state: FitState, summary: InferenceSummary) -> MulticovariateCupedFit:
    """Finalize a fit from bounded, identity-bound state and inference evidence."""
    _validate_finalization_evidence(state, summary)
    u_gram = np.asarray(summary.u, dtype=float)
    finalized = tuple(
        _finalize_projection(index, record, state, summary, u_gram)
        for index, record in enumerate(summary.scaled_projected_score_squares)
    )
    fit_dof = _fit_degrees_of_freedom(state, finalized)
    return MulticovariateCupedFit(
        state.eligible_arms,
        state.control,
        state.counts,
        state.adjusted_means,
        state.theta,
        state.anchor,
        state.layout,
        summary.adjusted_component_covariance,
        fit_dof,
        state.n_min,
        state.p,
        summary.max_leverage,
        summary.sum_leverage / sum(state.counts),
        summary.sum_leverage,
        state.is_ratio,
        finalized,
    )


def _validate_finalization_evidence(state: FitState, summary: InferenceSummary) -> None:
    if state.identity != summary.fit_identity or (
        summary.evidence_identity and summary.evidence_identity != state.identity
    ):
        _fail(
            "estimation.cuped_linear.incompatible_evidence",
            reason="fit state and inference summary identities differ",
        )
    if not state.eligible_arms or state.control not in state.eligible_arms:
        _invalid_evidence("fit state arms/control are invalid")
    if not np.isfinite((summary.max_leverage, summary.sum_leverage)).all():
        _invalid_evidence("inference summary leverage evidence is not finite")
    _validate_m1_support(state, summary)
    expected_ids = _expected_projection_roster(state)
    if summary.projection_identity != state.identity + ("all-pairs-v1",):
        _invalid_evidence("inference summary projection identity is not the complete all-pair set")
    record_ids = tuple(
        state.identity + (row[0], row[1], row[2], row[3])
        for row in summary.scaled_projected_score_squares
    )
    if record_ids != expected_ids:
        _invalid_evidence(
            "inference summary does not contain the complete all-pair projection roster"
        )
    if tuple(spec.identity for spec in summary.projection_specs) != expected_ids:
        _invalid_evidence("projection specifications do not match the complete all-pair roster")
    if any(
        spec.identity[: len(state.identity)] != state.identity for spec in summary.projection_specs
    ):
        _invalid_evidence("projection specification identity differs from fit state")
    if any(
        spec.available != bool(np.isfinite(record[4:7]).all())
        for spec, record in zip(
            summary.projection_specs, summary.scaled_projected_score_squares, strict=True
        )
    ):
        _invalid_evidence("projection availability differs between derivative and score evidence")
    if len(summary.adjusted_component_covariance) != len(state.adjusted_means) or any(
        len(row) != len(state.adjusted_means) for row in summary.adjusted_component_covariance
    ):
        _invalid_evidence("inference covariance shape does not match fitted components")
    nprojection = len(expected_ids)
    reductions = (summary.s2, summary.s4, summary.g, summary.g2)
    if any(len(rows) != nprojection for rows in reductions):
        _invalid_evidence("BM reductions do not match retained projection set")
    if len(summary.u) != len(state.eligible_arms) + state.p:
        _invalid_evidence("BM U reduction does not match fitted design")


def _validate_m1_support(state: FitState, summary: InferenceSummary) -> None:
    from increment.estimation.cuped.multicovariate import (
        M1_MAX_CATEGORICAL_LEVELS,
        M1_MAX_ENCODED_COVARIATES,
        M1_MAX_LEVERAGE,
        M1_MAX_LEVERAGE_RATIO,
        M1_MAX_RATIO_DENOMINATOR_PRECISION,
    )

    violations: list[tuple[object, ...]] = list(summary.refusal_witness)
    existing = {entry[0] for entry in violations}
    categorical_levels = max(
        (
            len(levels)
            for kind, levels in zip(state.layout.kinds, state.layout.levels, strict=True)
            if kind == "categorical" and levels is not None
        ),
        default=0,
    )
    support = (
        ("p", float(state.p), float(M1_MAX_ENCODED_COVARIATES)),
        ("k_max", float(categorical_levels), float(M1_MAX_CATEGORICAL_LEVELS)),
        ("leverage_ratio", state.p / state.n_min, float(M1_MAX_LEVERAGE_RATIO)),
        ("max_leverage", summary.max_leverage, float(M1_MAX_LEVERAGE)),
    )
    for name, value, bound in support:
        if name not in existing and value > bound:
            violations.append((name, value, bound))
    if state.is_ratio:
        precision = max(
            (float(cast(float | str, row[1])) for row in state.denominator_support), default=0.0
        )
        if (
            "ratio_denominator_precision" not in existing
            and precision > M1_MAX_RATIO_DENOMINATOR_PRECISION
        ):
            violations.append(
                ("ratio_denominator_precision", precision, M1_MAX_RATIO_DENOMINATOR_PRECISION)
            )
    if violations:
        _fail(
            "estimation.cuped_linear.support_region",
            region="M1",
            violations=tuple(violations),
        )


def _finalize_projection(
    index: int,
    record: tuple[str, object, object, str, float, float, float],
    state: FitState,
    summary: InferenceSummary,
    u_gram: np.ndarray,
) -> tuple[str, object, object, str, float, float, float]:
    kind, arm, control, scale, estimate, score_scale, scaled_square_sum = record
    if not np.isfinite((estimate, score_scale, scaled_square_sum)).all():
        return kind, arm, control, scale, np.nan, np.nan, np.nan
    if score_scale < 0 or scaled_square_sum < 0:
        _invalid_evidence("projected score-square evidence is negative")
    component_count = 2 if state.is_ratio and kind == "ratio" else 1
    reductions = (summary.s2, summary.s4, summary.g, summary.g2)
    if any(len(rows[index]) != component_count for rows in reductions):
        _invalid_evidence("BM component reductions do not match projection kind")
    traces: list[float] = []
    trace_squares: list[float] = []
    for component in range(component_count):
        trace, trace_square = _bm_trace_reductions(
            summary.s2[index][component],
            summary.s4[index][component],
            np.asarray(summary.g[index][component], dtype=float),
            np.asarray(summary.g2[index][component], dtype=float),
            u_gram,
        )
        if not np.isfinite((trace, trace_square)).all():
            _invalid_evidence("BM trace evidence is not finite")
        tolerance = 64 * np.finfo(float).eps * max(abs(trace), abs(trace_square), 1.0)
        if trace < -tolerance or trace_square < -tolerance:
            _invalid_evidence("BM trace evidence is negative")
        traces.append(max(0.0, trace))
        trace_squares.append(max(0.0, trace_square))
    trace_sum = float(np.sum(traces))
    trace_square_sum = float(np.sum(trace_squares))
    if trace_square_sum > 0:
        dof = trace_sum * trace_sum / trace_square_sum
    elif state.is_ratio and kind == "ratio":
        return kind, arm, control, scale, np.nan, np.nan, np.nan
    elif (
        trace_sum == 0
        and score_scale == 0
        and scaled_square_sum == 0
        and not any(summary.projection_specs[index].jacobian)
    ):
        dof = 1.0
    else:
        _invalid_evidence("BM trace-square evidence is degenerate")
    se = float(score_scale * np.sqrt(scaled_square_sum))
    return kind, arm, control, scale, estimate, se, dof


def _fit_degrees_of_freedom(
    state: FitState, finalized: tuple[tuple[str, object, object, str, float, float, float], ...]
) -> float:
    first_noncontrol = next(arm for arm in state.eligible_arms if arm != state.control)
    return next(
        (
            row[6]
            for row in finalized
            if row[0] == "absolute" and row[1] == first_noncontrol and row[2] == state.control
        ),
        1.0,
    )


def _expected_projection_roster(state: FitState) -> tuple[tuple[object, ...], ...]:
    arms = state.eligible_arms
    narm = len(arms)
    component_means = np.asarray(state.adjusted_means).reshape((-1, narm))
    roster = []
    for treatment_index in range(1, narm):
        for control_index in range(treatment_index):
            treatment, control = arms[treatment_index], arms[control_index]
            roster.append(state.identity + ("absolute", treatment, control, ""))
            roster.append(state.identity + ("absolute", control, treatment, ""))
            if component_means[0, treatment_index] > 0 and component_means[0, control_index] > 0:
                roster.append(state.identity + ("relative", treatment, control, ""))
                roster.append(state.identity + ("relative", control, treatment, ""))
            if state.is_ratio and np.all(component_means[:, [treatment_index, control_index]] > 0):
                for scale in ("log", "absolute"):
                    roster.append(state.identity + ("ratio", treatment, control, scale))
                    roster.append(state.identity + ("ratio", control, treatment, scale))
    return tuple(roster)


if TYPE_CHECKING:
    from increment.estimation.cuped.multicovariate import _LinearCupedCalculator


def _mean_covariance(fit: _LinearCupedCalculator) -> np.ndarray:
    narm = len(fit.arm_order)
    nfam = len(fit._components)
    covariance = np.zeros((narm * nfam, narm * nfam))
    scores = [fit._arm_score(family, arm) for family in range(nfam) for arm in range(narm)]
    for i, left in enumerate(scores):
        for j, right in enumerate(scores):
            with np.errstate(over="ignore", invalid="ignore"):
                covariance[i, j] = left @ right
    return covariance


def _bm(
    design: np.ndarray,
    weights: np.ndarray,
    leverage: np.ndarray,
    factor: np.ndarray,
    contrast: np.ndarray,
    *,
    q_design: np.ndarray | None = None,
    u_gram: np.ndarray | None = None,
    multiplicities=None,
) -> float:
    scale = float(np.max(np.abs(contrast), initial=0.0))
    if scale == 0:
        return 1.0
    trace, trace_square = _bm_terms(
        design,
        weights,
        leverage,
        factor,
        contrast,
        scale=scale,
        q_design=q_design,
        u_gram=u_gram,
        multiplicities=multiplicities,
    )
    return trace * trace / trace_square if trace_square > 0 else 1.0


def _bm_terms(
    design,
    weights,
    leverage,
    factor,
    contrast,
    *,
    scale,
    q_design=None,
    u_gram=None,
    multiplicities=None,
):
    if multiplicities is not None:
        reductions = _bm_reductions(
            design,
            weights,
            factor,
            contrast,
            scale=scale,
            q_design=q_design,
            u_gram=u_gram,
            multiplicities=multiplicities,
        )
        return _bm_trace_reductions(*reductions)
    sqrt_weights = np.sqrt(weights)
    if q_design is None:
        weighted_design = np.empty_like(design)
        np.multiply(design, sqrt_weights[:, None], out=weighted_design)
        q, r = scipy_qr(weighted_design, mode="economic", overwrite_a=True, check_finite=False)
    else:
        q, r = q_design, factor
    coefficients = np.linalg.solve(r.T, contrast / scale)
    ai = sqrt_weights * (q @ coefficients)
    leverage = np.einsum("ij,ij->i", q, q)
    qscore = ai / np.sqrt(1 - leverage)
    q2 = qscore * qscore
    q4 = q2 * q2
    parameter_count = design.shape[1]
    feature_count = 2 * parameter_count
    gram = np.zeros((feature_count, feature_count))
    weighted_gram = np.zeros_like(gram)
    basis_gram = (
        np.asarray(u_gram, dtype=float)
        if u_gram is not None
        else np.zeros((parameter_count, parameter_count))
    )
    block_size = 65_536
    for start in range(0, len(weights), block_size):
        stop = min(start + block_size, len(weights))
        sqrt_block = sqrt_weights[start:stop]
        q_block = q[start:stop]
        qscore_block = qscore[start:stop]
        v = q_block * (qscore_block / sqrt_block)[:, None]
        u = q_block * sqrt_block[:, None]
        feature = np.empty((stop - start, feature_count))
        feature[:, :parameter_count] = u * qscore_block[:, None]
        feature[:, parameter_count:] = v
        gram += feature.T @ feature
        weighted_gram += feature.T @ (feature * q2[start:stop, None])
        if u_gram is None:
            basis_gram += u.T @ u

    low_rank = np.zeros((feature_count, feature_count))
    low_rank[:parameter_count, parameter_count:] = -np.eye(parameter_count)
    low_rank[parameter_count:, :parameter_count] = -np.eye(parameter_count)
    low_rank[parameter_count:, parameter_count:] = basis_gram
    trace = float(q2.sum() + np.trace(low_rank @ gram))
    trace_square = float(
        q4.sum()
        + 2 * np.trace(low_rank @ weighted_gram)
        + np.trace(low_rank @ gram @ low_rank @ gram)
    )
    if len(weights) <= 1024 and (
        float(leverage.max(initial=0.0)) >= 1.0 - 1e-6 or trace_square < 0.0
    ):
        sqrt_weights = np.sqrt(weights)
        weighted_hat_transpose = sqrt_weights[:, None] * (q @ q.T) / sqrt_weights[None, :]
        bm_scores = (np.eye(len(weights)) - weighted_hat_transpose) * qscore[None, :]
        eigenvalues = np.linalg.eigvalsh(bm_scores.T @ bm_scores)
        trace = float(eigenvalues.sum())
        trace_square = float(eigenvalues @ eigenvalues)
    return trace, trace_square


def _bm_u_gram(q_design, weights, multiplicities=None):
    result = np.zeros((q_design.shape[1], q_design.shape[1]))
    for start in range(0, len(weights), 65_536):
        stop = min(start + 65_536, len(weights))
        weighted_q = q_design[start:stop] * np.sqrt(weights[start:stop, None])
        if multiplicities is not None:
            weighted_q *= np.sqrt(multiplicities[start:stop, None])
        result += weighted_q.T @ weighted_q
    return result


def _bm_reductions(
    design,
    weights,
    factor,
    contrast,
    *,
    scale,
    q_design,
    u_gram,
    multiplicities=None,
):
    """Accumulate the exact compact S2/S4/G/G2 reductions for one component."""
    sqrt_weights = np.sqrt(weights)
    coefficients = np.linalg.solve(factor.T, contrast / scale)
    leverage = np.einsum("ij,ij->i", q_design, q_design)
    t = sqrt_weights * (q_design @ coefficients) / np.sqrt(1.0 - leverage)
    t2_all = t * t
    s2 = float(t @ t if multiplicities is None else t2_all @ multiplicities)
    s4 = float(t2_all @ t2_all if multiplicities is None else (t2_all * t2_all) @ multiplicities)
    parameter_count = design.shape[1]
    gram = np.zeros((2 * parameter_count, 2 * parameter_count))
    weighted_gram = np.zeros_like(gram)
    block_size = 65_536
    for start in range(0, len(weights), block_size):
        stop = min(start + block_size, len(weights))
        q_block = q_design[start:stop]
        root_block = sqrt_weights[start:stop]
        t_block = t[start:stop]
        feature = np.empty((stop - start, 2 * parameter_count))
        feature[:, :parameter_count] = q_block * (root_block * t_block)[:, None]
        feature[:, parameter_count:] = q_block * (t_block / root_block)[:, None]
        t2 = t_block * t_block
        if multiplicities is not None:
            feature *= np.sqrt(multiplicities[start:stop, None])
        gram += feature.T @ feature
        weighted_gram += feature.T @ (feature * t2[:, None])
    return s2, s4, gram, weighted_gram, u_gram


def _bm_trace_reductions(s2, s4, gram, weighted_gram, u_gram):
    p = u_gram.shape[0]
    operator = np.zeros((2 * p, 2 * p))
    operator[:p, p:] = -np.eye(p)
    operator[p:, :p] = -np.eye(p)
    operator[p:, p:] = u_gram
    wide = np.longdouble
    operator_wide = operator.astype(wide)
    gram_wide = gram.astype(wide)
    weighted_wide = weighted_gram.astype(wide)
    trace = float(wide(s2) + np.trace(operator_wide @ gram_wide))
    trace_square = float(
        wide(s4)
        + 2 * np.trace(operator_wide @ weighted_wide)
        + np.trace(operator_wide @ gram_wide @ operator_wide @ gram_wide)
    )
    return trace, trace_square
