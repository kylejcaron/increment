"""Bayesian Normal-Normal inference.

`Normal` is a frozen pydantic model carrying a conjugate update. Given a
Normal prior and a Normal likelihood with known variance, the posterior is
Normal with precision equal to the sum of the two precisions and mean equal
to their precision-weighted average.

That operation is associative and commutative, so `update` is a pure
binary combine rather than a stateful model object: evidence can be folded
in any order, and `reduce(Normal.update, observations, prior)` is correct
by construction.

This is the shared tail of every lift computation in the full engine --
`infer_lift` and `infer_ate` both reduce two arms to a (point, SE) pair
and run it through the update below. Those functions, and the
`LiftEstimate` they return, land with the rest of the engine.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

# A prior wide enough to be uninformative
DIFFUSE_SIGMA = 1e6


class Normal(BaseModel):
    """Normal distribution."""

    model_config = ConfigDict(frozen=True)

    mu: float
    sigma: float = Field(gt=0)  # positivity via pydantic validation

    @property
    def precision(self) -> float:
        """Inverse variance."""
        return 1.0 / self.sigma**2

    @classmethod
    def diffuse(cls) -> Normal:
        """An approximately flat prior, for callers with no prior information."""
        return cls(mu=0.0, sigma=DIFFUSE_SIGMA)

    def update(self, observation: Normal) -> Normal:
        """Conjugate posterior treating `self` as prior and `observation`
        as a Normal likelihood with known variance.

        Pure: returns a new `Normal` and mutates nothing, so updates chain
        correctly -- `prior.update(a).update(b)` folds in both observations.
        Associative and commutative, so evidence order does not matter.
        """
        precision = self.precision + observation.precision
        return Normal(
            mu=(self.mu * self.precision + observation.mu * observation.precision) / precision,
            sigma=math.sqrt(1.0 / precision),
        )
