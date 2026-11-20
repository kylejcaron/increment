"""Independent occupancy, outcome-support, and raw regression references."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from math import exp, hypot, lgamma, log, log1p, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import t


def planned_se_at_counts(
    weights: Sequence[float], counts: Sequence[int], sigmas: Sequence[float]
) -> float | None:
    """Forecast error from specified population SDs at fixed contributing counts."""
    parts = []
    for c, n, sigma in zip(weights, counts, sigmas, strict=True):
        if c:
            if n == 0:
                return None
            parts.append(c * (sigma / sqrt(n)))
    return hypot(*parts)


def occupancy_probability(n: int, probabilities: Sequence[float], minimum: Sequence[int]) -> float:
    """Sum multinomial mass with the specified lower occupancy in each cell."""
    # Coefficient of z**n in the product of truncated exponential series.
    coefficients: NDArray[np.float64] = np.full(n + 1, -np.inf)
    coefficients[0] = 0.0
    for p, lo in zip(probabilities, minimum, strict=True):
        next_coefficients = np.full(n + 1, -np.inf)
        for k in range(lo, n + 1):
            term = k * log(p) - lgamma(k + 1)
            np.logaddexp(
                next_coefficients[k:], coefficients[: n - k + 1] + term, out=next_coefficients[k:]
            )
        coefficients = next_coefficients
    return min(1.0, exp(lgamma(n + 1) + float(coefficients[n])))


def _occupancies(n: int, slots: int) -> Iterator[tuple[int, ...]]:
    if slots == 1:
        yield (n,)
    else:
        for k in range(n + 1):
            for rest in _occupancies(n - k, slots - 1):
                yield (k, *rest)


def _log_binomial(n: int, k: int, p: float) -> float:
    if p == 0 or p == 1:
        return 0.0 if k == n * p else -np.inf
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log1p(-p)


def two_point_event_probability(
    n: int,
    probabilities: Sequence[float],
    high_probabilities: Sequence[float],
    event: Callable[[tuple[int, ...], tuple[int, ...]], bool],
) -> float:
    """Enumerate the exact small-N event over counts and conditional high counts."""
    terms = []
    for counts in _occupancies(n, len(probabilities)):
        log_counts = lgamma(n + 1) + sum(
            m * log(p) - lgamma(m + 1) for m, p in zip(counts, probabilities, strict=True)
        )
        for highs in product(*(range(m + 1) for m in counts)):
            if event(counts, highs):
                terms.append(
                    log_counts
                    + sum(
                        _log_binomial(m, k, p)
                        for m, k, p in zip(counts, highs, high_probabilities, strict=True)
                    )
                )
    return exp(float(logsumexp(terms))) if terms else 0.0


def contrast_matrix(a: float, b: float) -> NDArray[np.float64]:
    """Build contrasts from the design matrix independently of the readout."""
    # Saturated design columns are intercept, A, B, A*B, not the cell means.
    design = np.asarray(((1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1)))
    questions = np.asarray(
        (
            (0, 1, 0, b),
            (0, 0, 1, a),
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (0, 0, 1, 0),
            (0, 0, 1, 1),
            (0, 1, 1, 1),
            (0, 0, 0, 1),
        )
    )
    return questions @ np.linalg.inv(design)


@dataclass(frozen=True)
class RegressionBatch:
    """Reference arrays by trial, metric, and contrast."""

    counts: NDArray[np.int64]
    estimate: NDArray[np.float64]
    se: NDArray[np.float64]
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    favorable: NDArray[np.bool_]
    ready: NDArray[np.bool_]
    reasons: dict[str, NDArray[np.bool_]]


def regress_raw_trials(
    assignments: NDArray[np.int64],
    outcomes: NDArray[np.float64],
    *,
    contrast: NDArray[np.float64],
    binary_metrics: Iterable[int],
    raw_boundary: NDArray[np.float64],
    direction: NDArray[np.int64],
    alpha: float,
) -> RegressionBatch:
    """Fit saturated group regressions from individual Bernoulli-assigned rows."""
    reps, units, metrics = outcomes.shape
    slots = (4 * np.arange(reps)[:, None] + assignments).ravel()
    counts = np.bincount(slots, minlength=4 * reps).reshape(reps, 4)
    means = np.empty((reps, metrics, 4))
    m2 = np.empty_like(means)
    for metric in range(metrics):
        observations = outcomes[..., metric].reshape(reps * units)
        total = np.bincount(slots, observations, minlength=4 * reps).reshape(reps, 4)
        means[:, metric] = total / np.maximum(1, counts)
        residual = observations - means[:, metric].reshape(-1)[slots]
        m2[:, metric] = np.bincount(slots, residual**2, minlength=4 * reps).reshape(reps, 4)
    support = contrast != 0
    empty = _any_required(counts[:, None, :] == 0, support)
    singleton = _any_required(counts[:, None, :] == 1, support)
    constant = _any_required((m2 == 0) & (counts[:, None, :] >= 2), support)
    point = ~empty
    fit = point & ~singleton & ~constant
    small = _any_required(counts[:, None, :] < 30, support) & fit
    sparse = np.zeros_like(fit)
    for metric in binary_metrics:
        success = means[:, metric] * counts
        events = np.rint(success)
        sparse[:, metric] = (
            _any_required(((events < 10) | (counts - events < 10))[:, None, :], support)[:, 0]
            & fit[:, metric]
        )
    ready = fit & ~small & ~sparse
    estimate = np.where(point, means @ contrast.T, np.nan)
    variance = m2 / (np.maximum(2, counts) * np.maximum(1, counts - 1))[:, None, :]
    contributions = variance[:, :, None, :] * contrast[None, None, :, :] ** 2
    v = contributions.sum(axis=-1)
    sat_den = np.sum(contributions**2 / np.maximum(1, counts[:, None, None, :] - 1), axis=-1)
    df = np.divide(v**2, sat_den, out=np.ones_like(v), where=fit)
    se = np.where(fit, np.sqrt(v), np.nan)
    halfwidth = t.isf(alpha / (2 * raw_boundary.size), df) * se
    lo, hi = estimate - halfwidth, estimate + halfwidth
    favorable = ready & np.where(
        direction == 0,
        (lo > raw_boundary) | (hi < raw_boundary),
        np.where(direction > 0, lo > raw_boundary, hi < raw_boundary),
    )
    return RegressionBatch(
        counts,
        estimate,
        se,
        lo,
        hi,
        favorable,
        ready,
        {
            "empty_cell": np.broadcast_to(empty, fit.shape),
            "singleton_cell": np.broadcast_to(singleton, fit.shape) & point,
            "zero_cell_variance": constant & point & ~singleton,
            "small_cell": small,
            "sparse_binary_events": sparse,
        },
    )


def _any_required(mask: NDArray[np.bool_], support: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return (mask[:, :, None, :] & support[None, None, :, :]).any(axis=-1)
