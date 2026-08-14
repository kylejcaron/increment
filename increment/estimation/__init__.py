"""Estimation primitives.

Only the Normal-Normal conjugate pieces are present in this alpha. The
remaining estimators (``infer_lift``, ``infer_ate``, CATE, meta-analysis,
diagnostics) land with the full engine and will be re-exported here.
"""

from increment.estimation.inference import DIFFUSE_SIGMA, Normal

__all__ = ["DIFFUSE_SIGMA", "Normal"]
