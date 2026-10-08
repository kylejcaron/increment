"""Finite normal-mixture priors on the lift, and named heavy-tailed shapes.

A mixture prior updates in closed form against ``infer_lift``'s Normal
likelihood: the posterior is again a finite normal mixture (see
``mixture_posterior``), so every decision statistic stays analytic.
``StudentTPrior`` is a three-field declaration (nu, scale, k); its
Gauss-Laguerre expansion is computed on demand and cached. Result rows keep
the declared prior separate from the already-updated posterior; mixture
posterior components are persisted exactly and replayed without updating them
again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import brentq
from scipy.stats import norm as _norm

from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
)

_MIN_POSITIVE_FLOAT = math.nextafter(0.0, 1.0)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.priors.mixture.prior_needs_least": "a mixture prior needs at least one component",
        "estimation.priors.mixture.weights_means_sigmas": "weights/means/sigmas must have the same length, got {k}/{means}/{sigmas}",
        "estimation.priors.mixture.every_weight_mean": "every mixture weight, mean, and sigma must be finite, got weights={weights}, means={means}, sigmas={sigmas}",
        "estimation.priors.mixture.every_weight": "every mixture weight must be > 0",
        "estimation.priors.mixture.every_sigma": "every mixture sigma must be > 0",
        "estimation.priors.mixture.weights_sum": "mixture weights must sum to 1, got {total}",
        "estimation.priors.student_t.nu_exceeds_200": "nu={nu} exceeds 200, where the mixture expansion is numerically unstable -- and a t with nu > 200 is within 0.6% of a Normal anyway. Use Normal(mu=0, sigma=scale) instead.",
        "estimation.priors.mixture_posterior.probability": "probability must be in (0, 1), got {probability}",
        "estimation.priors.mixture_posterior.tail_probability": "tail_probability must be in (0, 1), got {tail_probability}",
        "estimation.priors.mixture_posterior.expected_negative_part_scale": "scale must be 'linear' or 'log', got {scale!r}",
        "estimation.priors.mixture_posterior.expected_positive_part_scale": "scale must be 'linear' or 'log', got {scale!r}",
        "estimation.priors.standard_error": "standard_error must be > 0, got {standard_error}",
    },
)
_raise = raiser(_REFUSALS)

EXPECTED_NEGATIVE_PART_SCALE = _REFUSALS[
    "estimation.priors.mixture_posterior.expected_negative_part_scale"
]
EXPECTED_POSITIVE_PART_SCALE = _REFUSALS[
    "estimation.priors.mixture_posterior.expected_positive_part_scale"
]


class MixturePrior(CodedModel, BaseModel):
    """Finite normal-mixture prior on the log-relative-risk lift scale."""

    model_config = ConfigDict(frozen=True)

    weights: tuple[float, ...]
    means: tuple[float, ...]
    sigmas: tuple[float, ...]

    @model_validator(mode="after")
    def _consistent(self):
        k = len(self.weights)
        if k == 0:
            _raise("estimation.priors.mixture.prior_needs_least")
        if len(self.means) != k or len(self.sigmas) != k:
            _raise(
                "estimation.priors.mixture.weights_means_sigmas",
                k=k,
                means=len(self.means),
                sigmas=len(self.sigmas),
            )
        # NaN comparisons are always false, so positivity/normalization
        # checks below would silently accept a non-finite component
        # without this: check finiteness first, before any comparison.
        if not (
            all(math.isfinite(w) for w in self.weights)
            and all(math.isfinite(m) for m in self.means)
            and all(math.isfinite(s) for s in self.sigmas)
        ):
            _raise(
                "estimation.priors.mixture.every_weight_mean",
                weights=self.weights,
                means=self.means,
                sigmas=self.sigmas,
            )
        if any(w <= 0.0 for w in self.weights):
            _raise("estimation.priors.mixture.every_weight")
        if any(s <= 0.0 for s in self.sigmas):
            _raise("estimation.priors.mixture.every_sigma")
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-9:
            _raise("estimation.priors.mixture.weights_sum", total=total)
        return self

    def components(self) -> MixturePrior:
        """This mixture, verbatim - the shared seam with ``StudentTPrior``."""
        return self


@lru_cache(maxsize=32)
def _student_t_unit_expansion(nu: float, k: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(weights, unit sigmas) for a scale-1 t_nu - scale multiplies in after."""
    from scipy.special import gamma as _gamma_fn
    from scipy.special import roots_genlaguerre

    half_nu = nu / 2.0
    nodes, quad_weights = roots_genlaguerre(k, half_nu - 1.0)
    weights = quad_weights / _gamma_fn(half_nu)
    weights = weights / weights.sum()
    unit_sigmas = np.sqrt(half_nu / nodes)
    return tuple(float(w) for w in weights), tuple(float(s) for s in unit_sigmas)


class StudentTPrior(CodedModel, BaseModel):
    """Student-t prior on the lift, declared by (nu, scale, k).

    A t is a Gaussian scale mixture: ``theta | lam ~ N(0, scale^2/lam)``
    with ``lam ~ Gamma(nu/2, nu/2)``. ``components()`` discretizes that
    mixing density on generalized Gauss-Laguerre nodes - a fixed,
    data-independent expansion whose error lives entirely in
    representing the prior; the posterior update stays exact. K=80
    holds the posterior mean within 1% of the exact t out to 5-sigma
    surprises (measured).

    ``nu`` is capped at 200: the expansion is numerically unstable for
    large ``nu`` (NaN weights measured at nu=400), and at nu=200 a t
    is within 0.6% of a Normal at the 97.5% quantile anyway.
    """

    model_config = ConfigDict(frozen=True)

    nu: float = Field(gt=0)
    scale: float = Field(gt=0)
    k: int = Field(default=80, ge=2)

    @model_validator(mode="after")
    def _nu_within_stable_range(self):
        if self.nu > 200:
            _raise("estimation.priors.student_t.nu_exceeds_200", nu=self.nu)
        return self

    def components(self) -> MixturePrior:
        """Expand to the K-component normal mixture (cached per (nu, k))."""
        weights, unit_sigmas = _student_t_unit_expansion(self.nu, self.k)
        return MixturePrior(
            weights=weights,
            means=(0.0,) * self.k,
            sigmas=tuple(self.scale * s for s in unit_sigmas),
        )


@dataclass(frozen=True)
class MixturePosterior:
    """K-component normal-mixture posterior on the log-lift scale.

    Every operation is a weight-linear combination of the single-Normal
    closed form, so decision statistics stay analytic at any K.
    """

    weights: np.ndarray
    means: np.ndarray
    sigmas: np.ndarray

    def cdf(self, value: float) -> float:
        return float((self.weights * _norm.cdf((value - self.means) / self.sigmas)).sum())

    def survival(self, value: float) -> float:
        return float((self.weights * _norm.sf((value - self.means) / self.sigmas)).sum())

    def quantile(self, probability: float) -> float:
        if not 0.0 < probability < 1.0:
            _raise("estimation.priors.mixture_posterior.probability", probability=probability)
        lo = float((self.means - 12.0 * self.sigmas).min())
        hi = float((self.means + 12.0 * self.sigmas).max())
        # 12 sigma covers ordinary quantiles; widen for tiny tail probabilities.
        if self.cdf(lo) > probability or self.cdf(hi) < probability:
            span = hi - lo
            lo, hi = lo - 4.0 * span, hi + 4.0 * span
        xtol = max(float(self.sigmas.min()) * 1e-9, _MIN_POSITIVE_FLOAT)
        return float(brentq(lambda x: self.cdf(x) - probability, lo, hi, xtol=xtol))

    def isf(self, tail_probability: float) -> float:
        """Value whose upper-tail probability is ``tail_probability``."""
        if not 0.0 < tail_probability < 1.0:
            _raise(
                "estimation.priors.mixture_posterior.tail_probability",
                tail_probability=tail_probability,
            )
        lo = float((self.means - 12.0 * self.sigmas).min())
        hi = float((self.means + 12.0 * self.sigmas).max())
        if self.survival(lo) < tail_probability or self.survival(hi) > tail_probability:
            span = hi - lo
            lo, hi = lo - 4.0 * span, hi + 4.0 * span
        xtol = max(float(self.sigmas.min()) * 1e-9, _MIN_POSITIVE_FLOAT)
        return float(brentq(lambda x: self.survival(x) - tail_probability, lo, hi, xtol=xtol))

    def probability_between(self, lower: float, upper: float) -> float:
        """Probability mass in ``[lower, upper]``.

        Per component, ``cdf(z_upper) - cdf(z_lower)`` loses all
        precision once both z-scores land deep enough in the upper tail
        that their cdf rounds to exactly 1.0 (e.g. both z > ~8): the
        true mass -- still resolvable via the survival function -- reads
        as exactly 0. Components whose interval sits (at least mostly)
        in the upper tail use ``sf(z_lower) - sf(z_upper)`` instead,
        which is algebraically identical but keeps the small tail
        probabilities themselves, not their near-1.0 complements.
        """
        z_lower = (lower - self.means) / self.sigmas
        z_upper = (upper - self.means) / self.sigmas
        upper_tail = z_lower > 0.0
        cdf_diff = _norm.cdf(z_upper) - _norm.cdf(z_lower)
        sf_diff = _norm.sf(z_lower) - _norm.sf(z_upper)
        per_component = np.where(upper_tail, sf_diff, cdf_diff)
        return float((self.weights * per_component).sum())

    def expected_negative_part(self, *, scale: Literal["linear", "log"]) -> float:
        active = self.weights > 0.0
        weights = self.weights[active]
        means = self.means[active]
        sigmas = self.sigmas[active]
        z = means / sigmas
        if scale == "linear":
            per = sigmas * _norm.pdf(z) - means * _norm.cdf(-z)
        elif scale == "log":
            a = -z
            per = _norm.cdf(a) - np.exp(means + 0.5 * sigmas**2) * _norm.cdf(a - sigmas)
        else:
            _raise("estimation.priors.mixture_posterior.expected_negative_part_scale", scale=scale)
        return float((weights * per).sum())

    def expected_positive_part(self, *, scale: Literal["linear", "log"]) -> float:
        active = self.weights > 0.0
        weights = self.weights[active]
        means = self.means[active]
        sigmas = self.sigmas[active]
        z = means / sigmas
        if scale == "linear":
            per = sigmas * _norm.pdf(z) + means * _norm.cdf(z)
        elif scale == "log":
            a = -z
            per = np.exp(means + 0.5 * sigmas**2) * _norm.cdf(sigmas - a) - _norm.cdf(-a)
        else:
            _raise("estimation.priors.mixture_posterior.expected_positive_part_scale", scale=scale)
        return float((weights * per).sum())


def mixture_posterior(
    estimate: float, standard_error: float, prior: MixturePrior
) -> MixturePosterior:
    """Conjugate update of a normal-mixture prior against a Normal likelihood.

    Per component: precision-weighted mean/variance exactly as
    ``normal_posterior``; the weights re-weight by each component's marginal
    likelihood of the data, computed in log space so a many-sigma surprise
    re-normalizes instead of underflowing to NaN.
    """
    if standard_error <= 0:
        _raise("estimation.priors.standard_error", standard_error=standard_error)
    prior_means = np.asarray(prior.means)
    prior_sigmas = np.asarray(prior.sigmas)
    log_weights = np.log(np.asarray(prior.weights)) + _norm.logpdf(
        estimate, prior_means, np.sqrt(prior_sigmas**2 + standard_error**2)
    )
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()
    variances = 1.0 / (1.0 / prior_sigmas**2 + 1.0 / standard_error**2)
    means = variances * (prior_means / prior_sigmas**2 + estimate / standard_error**2)
    return MixturePosterior(weights=weights, means=means, sigmas=np.sqrt(variances))
