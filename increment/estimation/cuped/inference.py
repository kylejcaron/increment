"""HC2 covariance and Bell–McCaffrey inference for mixed CUPED."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import qr as scipy_qr

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
) -> float:
    scale = float(np.max(np.abs(contrast), initial=0.0))
    if scale == 0:
        return 1.0
    trace, trace_square = _bm_terms(
        design, weights, leverage, factor, contrast, scale=scale, q_design=q_design
    )
    return trace * trace / trace_square if trace_square > 0 else 1.0


def _bm_terms(design, weights, leverage, factor, contrast, *, scale, q_design=None):
    if len(weights) <= 128:
        sqrt_weights = np.sqrt(weights)
        q, r = np.linalg.qr(design * sqrt_weights[:, None], mode="reduced")
        coefficients = np.linalg.solve(r.T, contrast / scale)
        ai = sqrt_weights * (q @ coefficients)
        hat = np.sum(q * q, axis=1)
        qscore = ai / np.sqrt(1 - hat)
        hat_transpose = sqrt_weights[:, None] * (q @ q.T) / sqrt_weights[None, :]
        projected = (np.eye(len(weights)) - hat_transpose) * qscore[None, :]
        singular = np.linalg.svd(projected, compute_uv=False)
        largest = float(singular.max(initial=0.0))
        if largest == 0:
            return 0.0, 0.0
        normalized = singular / largest
        trace = largest**2 * float(normalized @ normalized)
        trace_square = largest**4 * float((normalized**2) @ (normalized**2))
        return trace, trace_square

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
    basis_gram = np.zeros((parameter_count, parameter_count))
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
