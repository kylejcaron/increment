"""Lower confidence limits for a stratified mixture of nonnegative fixed bets."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import elementwise

if TYPE_CHECKING:
    from increment._spillover_registration import SpilloverMethod


class PolicyMixture:
    """Fixed bet grid for one policy's stratum quotas, shared by both tails."""

    def __init__(
        self,
        counts: NDArray[np.int64],
        weights: NDArray[np.float64],
        log_threshold: float,
        method: SpilloverMethod,
    ) -> None:
        self.counts = counts
        rates = weights / counts
        total = sum(map(int, counts))
        scale = int(counts.max()) if method == "two-stage-fixed-bet-v2" else total
        self.terms = 1 + (scale - 1).bit_length()
        bets = np.ldexp(0.95, -np.arange(self.terms))
        eta = bets[:, None] * (rates / rates.max())
        self.log_base = np.log1p(-eta)
        self.log_eta = np.log(eta)
        # Compare the sum, not the mean, of the e-values. Allow relative IEEE
        # error in the accumulated log factors (including cancellation at m=1).
        # This is not an interval-arithmetic certificate for warehouse moments.
        self.level = log_threshold + math.log(self.terms)
        self.level += 32 * np.finfo(float).eps * (total + self.level)

    def negative_log_limit(
        self, mean: NDArray[np.float64], variance: NDArray[np.float64], cap: float
    ) -> NDArray[np.float64]:
        """Return -log(m_lower); +infinity denotes the physical zero endpoint.

        For normalized X in [0,1], replace its empirical distribution by
        masses 1-p,p at 0,b, where b=a+v/a and p=a/b. Quadratic interpolation
        of log(1-eta+eta*X/m) at 0 and twice at b minorizes it (positive third
        derivative). The two distributions agree on that quadratic. These
        ratios need neither a raw second moment subtraction nor a squared mean.
        """
        positive = mean > 0
        spread = np.divide(variance, mean, out=np.zeros_like(mean), where=positive)
        node = mean + spread
        mass = self.counts * np.divide(mean, node, out=np.zeros_like(mean), where=positive)
        zero_mass = self.counts * np.divide(spread, node, out=np.ones_like(mean), where=positive)
        log_node = np.log(node, out=np.full_like(node, -np.inf), where=positive)
        offset = log_node[:, None, :] + self.log_eta
        zero_log = zero_mass[:, None, :] * self.log_base

        def residual(y: np.ndarray, rows: np.ndarray) -> np.ndarray:
            # Sum each complete log factor on the contiguous stratum axis. A
            # difference of two large dot products can lose even the sign of
            # a marginal decision when there are many small strata.
            terms = np.logaddexp(self.log_base, y[:, None, None] + offset[rows])
            terms *= mass[rows, None, :]
            terms += zero_log[rows]
            return np.logaddexp.reduce(terms.sum(axis=2), axis=1) - self.level

        result = np.full(len(mean), np.inf)
        rows = np.flatnonzero(positive.any(axis=1))
        if not rows.size:
            return result
        # Drop the nonnegative base from each positive-node factor of bet 0
        # to obtain an affine lower bound in y and hence an upper bracket.
        # Beyond cap even the rescaled root rounds to zero.
        intercept = zero_log[:, 0, :] + mass * np.where(positive, offset[:, 0, :], 0)
        rhs = self.level - intercept.sum(axis=1)
        slope = mass[rows].sum(axis=1)
        high = np.full(len(rows), cap)
        np.divide(rhs[rows], slope, out=high, where=rhs[rows] < cap * slope)
        high = np.minimum(cap, high + 1)
        at_high = residual(high, rows)
        finite = np.isfinite(at_high)
        inside = finite & (at_high >= 0)
        # Nonconvergence is not a zero endpoint. Leave only an all-zero tail
        # or a root proved smaller than the rescaled float domain at infinity.
        result[rows[~finite | (~inside & (high < cap))]] = np.nan
        rows, high = rows[inside], high[inside]
        if rows.size:
            root = elementwise.find_root(
                residual,
                (np.zeros(len(rows)), high),
                args=(rows,),
                tolerances={"fatol": 0, "frtol": 0},
            )
            # In y the larger endpoint is the outward (smaller m) side. If
            # the residual is exactly zero, the unused side may still be far away.
            outward = np.where(root.f_x == 0, root.x, root.bracket[1])
            result[rows] = np.where(root.success, outward, np.nan)
        return result
