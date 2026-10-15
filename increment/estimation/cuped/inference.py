"""HC2 covariance and Bell–McCaffrey inference for mixed CUPED."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import qr as scipy_qr

from increment.estimation.cuped.types import (
    FitState,
    InferenceSummary,
    MulticovariateCupedFit,
)


def finalize_fit(state: FitState, summary: InferenceSummary) -> MulticovariateCupedFit:
    """Finalize a fit from bounded, identity-bound state and inference evidence."""
    if state.identity != summary.fit_identity:
        raise ValueError("fit state and inference summary identities differ")
    if summary.evidence_identity and summary.evidence_identity != state.identity:
        raise ValueError("inference summary evidence identity differs from fit state")
    narm = len(state.eligible_arms)
    if len(set(state.eligible_arms)) != narm or not state.counts:
        raise ValueError("fit state arm identity or count shape is invalid")
    if any(
        spec.identity[: len(state.identity)] != state.identity for spec in summary.projection_specs
    ):
        raise ValueError("projection specification identity differs from fit state")
    expected_projection_ids = tuple(
        state.identity + (row[0], row[1], row[2], row[3])
        for row in summary.scaled_projected_score_squares
    )
    if expected_projection_ids != tuple(spec.identity for spec in summary.projection_specs):
        raise ValueError("projection specifications do not match the summary projection set")
    if len(summary.adjusted_component_covariance) != len(state.adjusted_means):
        raise ValueError("inference covariance does not match fitted outcome components")
    if any(len(row) != len(state.adjusted_means) for row in summary.adjusted_component_covariance):
        raise ValueError("inference covariance shape does not match fitted components")
    nprojection = len(summary.scaled_projected_score_squares)
    evidence = (summary.s2, summary.s4, summary.g, summary.g2)
    if any(len(rows) != nprojection for rows in evidence):
        raise ValueError("BM reductions do not match retained projection set")
    if len(summary.u) != len(state.eligible_arms) + state.p:
        raise ValueError("BM U reduction does not match fitted design")
    u_gram = np.asarray(summary.u, dtype=float)
    finalized = []
    for index, record in enumerate(summary.scaled_projected_score_squares):
        kind, arm, control, scale, estimate, score_scale, scaled_square_sum = record
        if np.isnan(estimate):
            finalized.append((kind, arm, control, scale, estimate, np.nan, np.nan))
            continue
        projection_components = 2 if state.is_ratio and kind == "ratio" else 1
        if any(len(rows[index]) != projection_components for rows in evidence):
            raise ValueError("BM component reductions do not match projection kind")
        traces = []
        trace_squares = []
        for component in range(projection_components):
            trace, trace_square = _bm_trace_reductions(
                summary.s2[index][component],
                summary.s4[index][component],
                np.asarray(summary.g[index][component], dtype=float),
                np.asarray(summary.g2[index][component], dtype=float),
                u_gram,
            )
            traces.append(trace)
            trace_squares.append(trace_square)
        trace_sum = float(np.sum(traces))
        trace_square_sum = float(np.sum(trace_squares))
        dof = trace_sum * trace_sum / trace_square_sum if trace_square_sum > 0 else 1.0
        se = float(score_scale * np.sqrt(scaled_square_sum))
        finalized.append((kind, arm, control, scale, estimate, se, dof))
    absolute = next((row[6] for row in finalized if row[0] == "absolute"), 1.0)
    return MulticovariateCupedFit(
        state.eligible_arms,
        state.control,
        state.counts,
        state.adjusted_means,
        state.theta,
        state.anchor,
        state.layout,
        summary.adjusted_component_covariance,
        float(absolute),
        state.n_min,
        state.p,
        summary.max_leverage,
        summary.sum_leverage / sum(state.counts),
        summary.sum_leverage,
        state.is_ratio,
        tuple(finalized),
    )


if TYPE_CHECKING:
    from increment.estimation.cuped.multicovariate import _LinearCupedCalculator


def _mean_covariance(fit: _LinearCupedCalculator) -> np.ndarray:
    narm = len(fit.arm_order)
    nfam = len(fit._components)
    covariance = np.zeros((narm * nfam, narm * nfam))
    scores = [fit._arm_score(family, arm) for family in range(nfam) for arm in range(narm)]
    for i, left in enumerate(scores):
        for j, right in enumerate(scores):
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
    )
    return trace * trace / trace_square if trace_square > 0 else 1.0


def _bm_terms(design, weights, leverage, factor, contrast, *, scale, q_design=None, u_gram=None):
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
    return trace, trace_square


def _bm_u_gram(q_design, weights):
    result = np.zeros((q_design.shape[1], q_design.shape[1]))
    for start in range(0, len(weights), 65_536):
        stop = min(start + 65_536, len(weights))
        weighted_q = q_design[start:stop] * np.sqrt(weights[start:stop, None])
        result += weighted_q.T @ weighted_q
    return result


def _bm_reductions(design, weights, factor, contrast, *, scale, q_design, u_gram):
    """Accumulate the exact compact S2/S4/G/G2 reductions for one component."""
    sqrt_weights = np.sqrt(weights)
    coefficients = np.linalg.solve(factor.T, contrast / scale)
    leverage = np.einsum("ij,ij->i", q_design, q_design)
    t = sqrt_weights * (q_design @ coefficients) / np.sqrt(1.0 - leverage)
    s2 = float(t @ t)
    s4 = float((t * t) @ (t * t))
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
        gram += feature.T @ feature
        weighted_gram += feature.T @ (feature * t2[:, None])
    return s2, s4, gram, weighted_gram, u_gram


def _bm_trace_reductions(s2, s4, gram, weighted_gram, u_gram):
    p = u_gram.shape[0]
    operator = np.zeros((2 * p, 2 * p))
    operator[:p, p:] = -np.eye(p)
    operator[p:, :p] = -np.eye(p)
    operator[p:, p:] = u_gram
    trace = float(s2 + np.trace(operator @ gram))
    trace_square = float(
        s4 + 2 * np.trace(operator @ weighted_gram) + np.trace(operator @ gram @ operator @ gram)
    )
    return trace, trace_square
