"""Scale-free summaries of the nonnegative arm weights actually used."""

from __future__ import annotations

import math

import numpy as np


def _weight_summary(
    weights: np.ndarray,
    cluster_index: np.ndarray | None = None,
    n_clusters: int | None = None,
) -> tuple[int, float, float]:
    """Return contributing grain count, Kish ESS and largest weight share.

    Inputs are the estimator's finite nonnegative weights with positive total.
    Scale before summing or squaring, including before cluster aggregation:
    neither a raw square nor a raw cluster total need be representable.
    Cluster summaries describe arm-weight concentration, not score support or
    a residual degree of freedom. Arm-wise normalization cancels in both ratios.
    """
    scaled = np.asarray(weights, dtype=np.float64) / float(np.max(weights))
    if cluster_index is not None:
        assert n_clusters is not None
        scaled = np.asarray(
            np.bincount(cluster_index, weights=scaled, minlength=n_clusters),
            dtype=np.float64,
        )
        scaled /= float(np.max(scaled))
    total = math.fsum(scaled)
    squares = float(np.dot(scaled, scaled))
    n = int(np.count_nonzero(scaled))
    # Compute (sum/sqrt(sum of squares))² without a potentially overflowing sum².
    ess = (total / math.sqrt(squares)) ** 2
    return n, min(float(n), ess), 1.0 / total
