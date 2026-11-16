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
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Annotated, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    FieldSerializationInfo,
    Tag,
    TypeAdapter,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)
from scipy.optimize import brentq

from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
    refuse,
    unwrap_coded,
)
from increment.semantics.models import (
    RelativeLiftPriorSpec,
    _validate_mixture_prior,
    _validate_relative_prior_tag,
)

_MIN_POSITIVE_FLOAT = math.nextafter(0.0, 1.0)
_MAX_FINITE_FLOAT = Fraction(float(np.finfo(float).max))


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.priors.student_t.nu_exceeds_200": "nu={nu} exceeds 200, where the mixture expansion is numerically unstable; and a t with nu > 200 is within 0.6% of a Normal anyway. Use Normal(mu=0, sigma=scale) instead.",
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


def _normal_log_moment(means: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Compute mu + sigma²/2 without losing the low product bits."""
    mantissas, exponents = np.frexp(sigmas)
    split = (2**27 + 1) * mantissas
    high = split - (split - mantissas)
    low = mantissas - high
    square = mantissas * mantissas
    error = ((high * high - square) + 2.0 * high * low) + low * low
    powers = 2 * exponents - 1
    result = np.empty_like(means)
    small = powers <= 0
    result[small] = (means[small] + np.ldexp(square[small], powers[small])) + np.ldexp(
        error[small], powers[small]
    )
    # Scale the sum as well as the square when the variance itself can overflow.
    scaled = np.ldexp(means[~small], -powers[~small])
    total = square[~small] + scaled
    remainder = total - square[~small]
    correction = (square[~small] - (total - remainder)) + (scaled - remainder)
    with np.errstate(over="ignore"):
        result[~small] = np.ldexp(total + (correction + error[~small]), powers[~small])
    return result


def _normal_log_expected_part(
    means: np.ndarray, sigmas: np.ndarray, *, scale: Literal["linear", "log"], positive: bool
) -> np.ndarray:
    """Evaluate log Normal partial expectations before weighting components."""
    from scipy.special import erfcx, log_ndtr

    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    with np.errstate(over="ignore", divide="ignore"):
        z = means / sigmas
        log_density = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
    if scale == "linear":
        directed_means = means if positive else -means
        directed_z = z if positive else -z
        log_scaled_density = np.log(sigmas) + log_density
        result = np.full_like(means, -math.inf)
        central = directed_means >= 0.0
        with np.errstate(divide="ignore"):
            result[central] = np.logaddexp(
                log_scaled_density[central],
                np.log(directed_means[central]) + log_ndtr(directed_z[central]),
            )
        # The loss is bounded above by sigma*phi(z); omit only rounded-zero terms.
        tail = ~central & (log_scaled_density > math.log(_MIN_POSITIVE_FLOAT) - math.log(2.0))
        t = -directed_z[tail]
        mills = math.sqrt(math.pi / 2.0) * erfcx(t / math.sqrt(2.0))
        result[tail] = log_scaled_density[tail] + np.log1p(-t * mills)
        return result

    with np.errstate(over="ignore"):
        tilted_z = z + sigmas
    log_tilted = np.empty_like(means)
    tail = tilted_z < 0.0 if positive else tilted_z >= 0.0
    t = -tilted_z[tail] if positive else tilted_z[tail]
    with np.errstate(divide="ignore"):
        log_tilted[tail] = log_density[tail] + np.log(
            math.sqrt(math.pi / 2.0) * erfcx(t / math.sqrt(2.0))
        )
    log_tilted[~tail] = _normal_log_moment(means[~tail], sigmas[~tail]) + log_ndtr(
        tilted_z[~tail] if positive else -tilted_z[~tail]
    )
    log_mass = log_ndtr(z if positive else -z)
    log_large, log_small = (log_tilted, log_mass) if positive else (log_mass, log_tilted)
    finite = np.isfinite(log_large)
    with np.errstate(divide="ignore"):
        log_large[finite] += np.log(-np.expm1(log_small[finite] - log_large[finite]))
    return log_large


@runtime_checkable
class LiftPosterior(Protocol):
    """Distribution operations consumed by fixed-horizon decision statistics."""

    def cdf(self, value: float) -> float: ...

    def survival(self, value: float) -> float: ...

    def quantile(self, probability: float) -> float: ...

    def isf(self, tail_probability: float) -> float: ...

    def probability_between(self, lower: float, upper: float) -> float: ...

    def expected_negative_part(self, *, scale: Literal["linear", "log"]) -> float: ...

    def expected_positive_part(self, *, scale: Literal["linear", "log"]) -> float: ...


class Normal(CodedModel, BaseModel):
    """Normal distribution with finite scalar parameters."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    mu: float
    sigma: float = Field(gt=0)  # positivity via pydantic validation

    @property
    def precision(self) -> float:
        """Inverse variance."""
        return 1.0 / self.sigma**2

    def cdf(self, value: float) -> float:
        """Probability mass at or below ``value``."""
        from scipy.stats import norm as _norm

        return float(_norm.cdf((value - self.mu) / self.sigma))

    def survival(self, value: float) -> float:
        """Probability mass strictly above ``value``."""
        from scipy.stats import norm as _norm

        return float(_norm.sf((value - self.mu) / self.sigma))

    def quantile(self, probability: float) -> float:
        """Value whose cumulative probability is ``probability``."""
        from scipy.stats import norm as _norm

        return self.mu + self.sigma * float(_norm.ppf(probability))

    def isf(self, tail_probability: float) -> float:
        """Value whose upper-tail probability is ``tail_probability``."""
        from scipy.stats import norm as _norm

        return self.mu + self.sigma * float(_norm.isf(tail_probability))

    def probability_between(self, lower: float, upper: float) -> float:
        """Probability mass between two inclusive bounds."""
        if lower >= self.mu:
            return self.survival(lower) - self.survival(upper)
        return self.cdf(upper) - self.cdf(lower)

    def expected_negative_part(self, *, scale: Literal["linear", "log"]) -> float:
        """Expected magnitude below zero on the requested reporting scale."""
        if scale not in ("linear", "log"):
            refuse(EXPECTED_NEGATIVE_PART_SCALE, scale=scale)
        with np.errstate(over="ignore"):
            return float(
                np.exp(
                    _normal_log_expected_part(
                        np.array([self.mu]), np.array([self.sigma]), scale=scale, positive=False
                    )[0]
                )
            )

    def expected_positive_part(self, *, scale: Literal["linear", "log"]) -> float:
        """Expected magnitude above zero on the requested reporting scale."""
        if scale not in ("linear", "log"):
            refuse(EXPECTED_POSITIVE_PART_SCALE, scale=scale)
        with np.errstate(over="ignore"):
            return float(
                np.exp(
                    _normal_log_expected_part(
                        np.array([self.mu]), np.array([self.sigma]), scale=scale, positive=True
                    )[0]
                )
            )


class MixturePrior(CodedModel, BaseModel):
    """Finite Normal mixture with ordered component weights, means and deviations."""

    model_config = ConfigDict(frozen=True)

    weights: tuple[float, ...]
    means: tuple[float, ...]
    sigmas: tuple[float, ...]

    @model_validator(mode="after")
    def _consistent(self):
        _validate_mixture_prior(self.weights, self.means, self.sigmas)
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
    """Zero-centered Student-t prior with shape nu, scale and legacy size k.

    The precision mixture is ``theta | lam ~ N(0, scale^2/lam)`` with
    ``lam ~ Gamma(nu/2, nu/2)``. Unwrapped priors use ``components()``:
    a fixed, data-independent finite expansion of that mixing density.
    ``RelativeLiftPrior`` uses the continuous target; k does not change it.

    The existing declaration range caps nu at 200. Use a Normal prior
    for a larger, near-Normal shape.
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


def _prior_family(value: object) -> str | None:
    if isinstance(value, Mapping):
        if "type" in value:
            tag = value.get("type")
            return tag if isinstance(tag, str) else None
        if "mu" in value or "sigma" in value:
            return "normal"
        if "nu" in value or "scale" in value:
            return "student_t"
        if "weights" in value or "means" in value or "sigmas" in value:
            return "normal_mixture"
        return None
    if isinstance(value, Normal):
        return "normal"
    if isinstance(value, StudentTPrior):
        return "student_t"
    if isinstance(value, MixturePrior):
        return "normal_mixture"
    if isinstance(value, RelativeLiftPrior):
        return "relative_lift"
    return None


type _RelativeDistribution = Annotated[
    Annotated[Normal, Tag("normal")]
    | Annotated[StudentTPrior, Tag("student_t")]
    | Annotated[MixturePrior, Tag("normal_mixture")],
    Discriminator(_prior_family),
]


class RelativeLiftPrior(CodedModel, BaseModel):
    """Prior directly on fractional relative lift: 0.10 means ten percent."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    type: Literal["relative_lift"] = "relative_lift"
    distribution: _RelativeDistribution

    @field_validator("distribution", mode="before")
    @classmethod
    def _strict_family_payload(cls, value: object) -> object:
        models = {"normal": Normal, "student_t": StudentTPrior, "normal_mixture": MixturePrior}
        payload: object = value
        if isinstance(value, (Normal, StudentTPrior, MixturePrior)):
            payload = {"type": _prior_family(value), **value.model_dump()}
        declared = RelativeLiftPriorSpec.model_validate({"distribution": payload}).distribution
        if isinstance(value, (Normal, StudentTPrior, MixturePrior)):
            return value
        return models[declared.type].model_validate(declared.model_dump(exclude={"type"}))

    @field_serializer("distribution")
    def _distribution_payload(
        self, value: Normal | StudentTPrior | MixturePrior, info: FieldSerializationInfo
    ) -> dict[str, object]:
        return {"type": _prior_family(value), **value.model_dump(mode=info.mode)}


def _tagged_relative_prior(value: object) -> object:
    if isinstance(value, Mapping) and ("type" in value or "distribution" in value):
        _validate_relative_prior_tag(value)
        return RelativeLiftPrior.model_validate(value)
    return value


type Prior = Annotated[
    Annotated[Normal, Tag("normal")]
    | Annotated[StudentTPrior, Tag("student_t")]
    | Annotated[MixturePrior, Tag("normal_mixture")]
    | Annotated[RelativeLiftPrior, Tag("relative_lift")],
    Discriminator(_prior_family),
    BeforeValidator(_tagged_relative_prior),
]

_PRIOR_PAYLOAD = TypeAdapter(Prior)


def prior_from_payload(value: object) -> Prior:
    """Decode prior data without reinterpreting legacy family declarations."""
    try:
        return _PRIOR_PAYLOAD.validate_python(value)
    except ValidationError as error:
        unwrap_coded(error)
        raise


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
        from scipy.stats import norm as _norm

        return float((self.weights * _norm.cdf((value - self.means) / self.sigmas)).sum())

    def survival(self, value: float) -> float:
        from scipy.stats import norm as _norm

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
        from scipy.stats import norm as _norm

        z_lower = (lower - self.means) / self.sigmas
        z_upper = (upper - self.means) / self.sigmas
        upper_tail = z_lower > 0.0
        cdf_diff = _norm.cdf(z_upper) - _norm.cdf(z_lower)
        sf_diff = _norm.sf(z_lower) - _norm.sf(z_upper)
        per_component = np.where(upper_tail, sf_diff, cdf_diff)
        return float((self.weights * per_component).sum())

    def expected_negative_part(self, *, scale: Literal["linear", "log"]) -> float:
        from scipy.special import logsumexp

        if scale not in ("linear", "log"):
            _raise("estimation.priors.mixture_posterior.expected_negative_part_scale", scale=scale)
        active = self.weights > 0.0
        weights = self.weights[active]
        log_per = _normal_log_expected_part(
            self.means[active], self.sigmas[active], scale=scale, positive=False
        )
        with np.errstate(over="ignore"):
            return float(np.exp(logsumexp(np.log(weights) + log_per)))

    def expected_positive_part(self, *, scale: Literal["linear", "log"]) -> float:
        from scipy.special import logsumexp

        if scale not in ("linear", "log"):
            _raise("estimation.priors.mixture_posterior.expected_positive_part_scale", scale=scale)
        active = self.weights > 0.0
        weights = self.weights[active]
        log_per = _normal_log_expected_part(
            self.means[active], self.sigmas[active], scale=scale, positive=True
        )
        with np.errstate(over="ignore"):
            return float(np.exp(logsumexp(np.log(weights) + log_per)))


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
    small = np.minimum(prior_sigmas, standard_error)
    large = np.maximum(prior_sigmas, standard_error)
    ratio = small / large
    unit_marginal = np.hypot(1.0, ratio)
    same_sign = np.signbit(prior_means) == np.signbit(estimate)
    observation = Fraction(estimate)
    observation_variance = Fraction(standard_error) ** 2
    quadratics = [
        (observation - Fraction(mean)) ** 2 / (Fraction(sigma) ** 2 + observation_variance)
        for mean, sigma in zip(prior.means, prior.sigmas, strict=True)
    ]
    closest = min(quadratics)
    # Subtract exact rational quadratics before rounding away component-mean differences.
    penalties = [(quadratic - closest) / 2 for quadratic in quadratics]
    relative_penalty = np.array(
        [float(penalty) if penalty <= _MAX_FINITE_FLOAT else math.inf for penalty in penalties]
    )
    log_weights = (
        np.log(np.asarray(prior.weights)) - np.log(large) - np.log(unit_marginal) - relative_penalty
    )
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()

    prior_smaller = prior_sigmas <= standard_error
    small_means = np.where(prior_smaller, prior_means, estimate)
    large_means = np.where(prior_smaller, estimate, prior_means)
    denominator = 1.0 + ratio * ratio
    means = np.empty_like(prior_means)
    means[same_sign] = small_means[same_sign] + ratio[same_sign] * (
        (ratio[same_sign] * (large_means[same_sign] - small_means[same_sign]))
        / denominator[same_sign]
    )
    means[~same_sign] = small_means[~same_sign] / denominator[~same_sign] + ratio[~same_sign] * (
        (ratio[~same_sign] * large_means[~same_sign]) / denominator[~same_sign]
    )
    return MixturePosterior(weights=weights, means=means, sigmas=small / unit_marginal)
