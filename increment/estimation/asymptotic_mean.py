"""Count-clock shifted-contrast AsympCS inversion with rational geometry."""

from __future__ import annotations

import math
from fractions import Fraction
from math import isqrt

from pydantic import BaseModel, ConfigDict, Field, model_validator

from increment._literals import ALTERNATIVE_VALUES, Alternative
from increment.errors import CapabilityError, CodedModel, RefusalSpec, refuse
from increment.estimation._certified import Interval, log_interval
from increment.estimation._sequential_likelihood import GaussianState
from increment.semantics.sequential import ScalarMeanModel

_INVALID = RefusalSpec(
    "sequential.asymptotic_mean.invalid", CapabilityError, lambda *, reason: reason
)


class MeanSetComponent(CodedModel, BaseModel):
    """Closed ratio-coordinate component; None denotes an infinite endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    lower: Fraction | None
    upper: Fraction | None

    @model_validator(mode="after")
    def _ordered(self):
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            refuse(_INVALID, reason="confidence component endpoints must be ordered")
        return self


class AsymptoticMeanSet(CodedModel, BaseModel):
    """Set geometry is independent of numeric point availability."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    components: tuple[MeanSetComponent, ...]
    alpha: Fraction
    alternative: Alternative
    count: int = Field(ge=0, strict=True)
    k: Fraction | None
    available: bool
    reason: str | None = None
    estimator_contrast: Fraction | None = None
    estimator_variance: Fraction | None = None

    @model_validator(mode="after")
    def _availability(self):
        if not 0 < self.alpha < 1:
            refuse(_INVALID, reason="confidence allocation must be in (0,1)")
        if self.available:
            if (
                self.reason is not None
                or self.k is None
                or self.k <= 0
                or self.estimator_contrast is None
                or self.estimator_variance is None
                or self.estimator_variance <= 0
            ):
                refuse(
                    _INVALID, reason="available confidence geometry requires positive uncertainty"
                )
        elif (
            not self.reason
            or self.k is not None
            or self.estimator_contrast is not None
            or self.estimator_variance is not None
            or self.components != (MeanSetComponent(lower=None, upper=None),)
        ):
            refuse(_INVALID, reason="unavailable confidence geometry must be vacuous with a reason")
        if any(
            left.upper is None or right.lower is None or left.upper >= right.lower
            for left, right in zip(self.components, self.components[1:], strict=False)
        ):
            refuse(_INVALID, reason="confidence components must be ordered and disjoint")
        return self

    @property
    def empty(self) -> bool:
        return not self.components

    @property
    def lower(self) -> Fraction | None:
        return self.components[0].lower if self.components else None

    @property
    def upper(self) -> Fraction | None:
        return self.components[-1].upper if self.components else None

    @property
    def status(self) -> str:
        if not self.available:
            return "unavailable"
        if self.empty:
            return "empty"
        if len(self.components) > 1:
            return "disconnected"
        if self.lower is None and self.upper is None:
            return "full"
        if self.lower is None or self.upper is None:
            return "ray"
        return "bounded"

    def rejects(self) -> bool:
        """Null outside the set: the contrast crosses ``k`` on the tested side."""
        if not self.available or self.empty:
            return False
        d, v, k = self.estimator_contrast, self.estimator_variance, self.k
        assert d is not None and v is not None and k is not None
        crossed = d * d > k * v
        return crossed and (
            self.alternative == "two-sided"
            or (self.alternative == "greater" and d > 0)
            or (self.alternative == "less" and d < 0)
        )


def _sqrt_interval(value: Fraction) -> Interval:
    """Enclose a nonnegative rational square root without binary64 conversion."""
    bits = value.numerator.bit_length() - value.denominator.bit_length()
    scale = 1 << max(0, 100 - bits // 2)
    floor = isqrt(value.numerator * scale * scale // value.denominator)
    lower = Fraction(floor, scale)
    upper = lower if lower * lower == value else Fraction(floor + 1, scale)
    return Interval(lower, upper)


def _component_for_quadratic(a: Fraction, b: Fraction, c: Fraction) -> tuple[MeanSetComponent, ...]:
    """Outward solution of a*r²+b*r+c<=0; no near-zero coefficient tolerance."""
    full = (MeanSetComponent(lower=None, upper=None),)
    if a == 0:
        if b == 0:
            return full if c <= 0 else ()
        root = -c / b
        return (
            (MeanSetComponent(lower=None, upper=root),)
            if b > 0
            else (MeanSetComponent(lower=root, upper=None),)
        )
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return full if a < 0 else ()
    if discriminant == 0:
        root = -b / (2 * a)
        return (MeanSetComponent(lower=root, upper=root),) if a > 0 else full
    root_width = _sqrt_interval(discriminant)
    roots = sorted(
        ((-b - root_width) / (2 * a), (-b + root_width) / (2 * a)),
        key=lambda root: root.midpoint,
    )
    if a > 0:
        return (MeanSetComponent(lower=roots[0].lo, upper=roots[1].hi),)
    if roots[0].hi >= roots[1].lo:
        return full
    return (
        MeanSetComponent(lower=None, upper=roots[0].hi),
        MeanSetComponent(lower=roots[1].lo, upper=None),
    )


def count_boundary(n: int, alpha: Fraction, rho: Fraction) -> Fraction:
    """Conservative K=(1+lambda/N)*(log1p(N/lambda)-2log(alpha))."""
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0 or not 0 < alpha < 1 or rho <= 0:
        refuse(_INVALID, reason="count, alpha or rho is outside the declared domain")
    alpha, rho = Fraction(alpha), Fraction(rho)
    lam = 1 / rho**2
    return (1 + lam / n) * (log_interval(1 + n / lam) - 2 * log_interval(alpha)).hi


def _count_boundary_log_cap(count: int, rho: Fraction, statistic: Fraction) -> Fraction:
    """Strict log score of a squared-precision ``statistic`` against ``count_boundary``.

    ``count_boundary`` is ``K = A * (J.hi + 2 * H(alpha))`` with ``A = 1 + lambda/N``,
    ``J = log_interval(1 + N/lambda)`` and ``H(alpha) = (-log_interval(alpha)).hi``. The
    score ``L = (statistic / A - J.hi) / 2 - J.width`` gives, for every rational alpha,
    ``L >= H(alpha)  =>  statistic >= K + 2 * A * J.width > K``: a non-strict comparison
    of ``L`` with ``H(alpha)`` never admits a statistic the strict test
    ``statistic > K`` rejects, a tie ``statistic == K`` included. ``J.width`` is positive
    for every positive count, so the only loss is a strip of that width just above ``K``
    where ``statistic > K`` holds but ``L`` stays below ``H(alpha)``.
    """
    lam = 1 / Fraction(rho) ** 2
    log1p = log_interval(1 + count / lam)
    return (statistic / (1 + lam / count) - log1p.hi) / 2 - log1p.width


def mixture_r_star(alpha: Fraction) -> Fraction:
    """The always-valid Gaussian mixture's own se-independent optimal
    tuning at the RAW (not directionally doubled) alpha: solves
    r - log1p(r) = -2*log(alpha) by bisection. Both the runtime's
    per-metric rho (sequential_source.py) and the plan-time
    GaussianScoreMixture boundary (power/sequential.py) derive from this
    one root, so they cannot independently drift. Directional (one-sided)
    doubling is applied separately, only to the boundary's own log term --
    see directional_alpha and count_boundary's callers."""
    log_alpha = (
        math.log1p(-float(1 - alpha))
        if alpha > Fraction(1, 2)
        else math.log(alpha.numerator) - math.log(alpha.denominator)
    )
    target = -2 * log_alpha
    lo, hi = 0.0, 2 * target + 2
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if mid == lo or mid == hi:
            break
        if mid - math.log1p(mid) < target:
            lo = mid
        else:
            hi = mid
    return Fraction((lo + hi) / 2)


def count_boundary_log_e(
    *,
    count: int,
    rho: Fraction,
    estimator_contrast: Fraction | float,
    estimator_variance: Fraction | float,
    alternative: Alternative = "two-sided",
) -> Fraction | float:
    """Plug-in normal-mixture value dual to the count boundary.

    Sign gating only decreases this value. Substituting estimated variance
    does not preserve the oracle martingale or its finite-sample expectation
    bound: an exponentiated Gaussian Studentized statistic can have infinite
    expectation. Family selection therefore uses an asymptotic approximation,
    not the finite-sample e-value theorem available to the Bernoulli path.
    The relevant family expectation is truncated at its selection threshold;
    a finite-grid check of that expectation is not a uniform stopping theorem.

    The arguments are the same values retained by ``AsymptoticMeanSet``.
    An available ``e_value_dual`` set rejects when this log value exceeds
    ``-log(alpha)``. Ratio-law family evidence is additionally capped by
    denominator stability (``_count_boundary_log_cap``); the cap decreases
    the value but does not establish a finite-sample e-value guarantee.
    """
    if count <= 0 or rho <= 0 or estimator_variance <= 0:
        return float("-inf")
    if alternative == "greater" and estimator_contrast <= 0:
        return float("-inf")
    if alternative == "less" and estimator_contrast >= 0:
        return float("-inf")
    lam = 1 / Fraction(rho) ** 2
    statistic = estimator_contrast * estimator_contrast / estimator_variance
    scale = 1 + lam / count
    log1p_term = log_interval(1 + count / lam).hi
    return (statistic / scale - log1p_term) / 2


def directional_alpha(alpha: Fraction, alternative: Alternative) -> Fraction:
    """Level the Robbins boundary is built at for the requested tail.

    A one-sided boundary uses the symmetric two-sided boundary at
    ``2 * alpha``. Under a symmetric process, the one-tail crossing
    probability is half the two-sided union probability plus half the
    probability of crossing both tails at different looks. The plug-in
    boundary is asymptotic; neither that approximation error nor the
    both-tails excess is a finite-sample alpha guarantee.
    """
    if alternative == "two-sided":
        return Fraction(alpha)
    doubled = 2 * Fraction(alpha)
    if doubled >= 1:
        refuse(_INVALID, reason="a one-sided asymptotic boundary needs alpha below one half")
    return doubled


def boundary_alpha(alpha: Fraction, alternative: Alternative, *, e_value_dual: bool) -> Fraction:
    """Level the count boundary is built at.

    An available ``e_value_dual`` set inverts the sign-gated contrast's log
    evidence at ``-log(alpha)``, so it is built at ``alpha`` itself on either
    side. A ratio law also requires resolved denominators at that level;
    any other one-sided set uses ``directional_alpha``.
    """
    return Fraction(alpha) if e_value_dual else directional_alpha(alpha, alternative)


def _join_half_line(
    components: tuple[MeanSetComponent, ...], *, point: Fraction, upward: bool
) -> tuple[MeanSetComponent, ...]:
    """Union ordered disjoint closed components with a closed half-line from ``point``."""
    if upward:
        kept = tuple(c for c in components if c.upper is not None and c.upper < point)
        touched = [c for c in components if c.upper is None or c.upper >= point]
        lower: Fraction | None = point
        for c in touched:
            if c.lower is None:
                lower = None
                break
            lower = min(lower, c.lower)
        return (*kept, MeanSetComponent(lower=lower, upper=None))
    kept = tuple(c for c in components if c.lower is not None and c.lower > point)
    touched = [c for c in components if c.lower is None or c.lower <= point]
    upper: Fraction | None = point
    for c in touched:
        if c.upper is None:
            upper = None
            break
        upper = max(upper, c.upper)
    return (MeanSetComponent(lower=None, upper=upper), *kept)


def one_sided_components(
    components: tuple[MeanSetComponent, ...],
    alternative: Alternative,
    *,
    control_value: Fraction,
    treatment_value: Fraction,
) -> tuple[MeanSetComponent, ...]:
    """The one-sided set: the two-sided set joined with the half-line it never rejects.

    With ``d(r) = treatment - r * control``, ``d(r) <= sqrt(K V(r))`` holds
    wherever ``d(r) <= 0`` and, where ``d(r) > 0``, exactly on the two-sided
    set; so the "greater" set is the two-sided set united with
    ``{r : d(r) <= 0}``, and "less" with ``{r : d(r) >= 0}``.
    """
    if alternative == "two-sided" or not components:
        return components
    if control_value == 0:
        everywhere = treatment_value <= 0 if alternative == "greater" else treatment_value >= 0
        return (MeanSetComponent(lower=None, upper=None),) if everywhere else components
    point = treatment_value / control_value
    upward = (alternative == "greater") == (control_value > 0)
    return _join_half_line(components, point=point, upward=upward)


def asymptotic_mean_set(
    control: GaussianState,
    treatment: GaussianState,
    *,
    declaration: ScalarMeanModel,
    alpha: Fraction,
    null_lift: Fraction,
    alternative: Alternative,
    e_value_dual: bool = False,
) -> AsymptoticMeanSet:
    """Use empirical arm variance M2/n and estimator variance M2/n²."""
    if len(control.mean) != 1 or len(treatment.mean) != 1:
        refuse(_INVALID, reason="scalar mean inference requires scalar centered arm moments")
    if not 0 < alpha < 1 or alternative not in ALTERNATIVE_VALUES:
        refuse(_INVALID, reason="alpha or alternative is outside the declared domain")
    level = boundary_alpha(alpha, alternative, e_value_dual=e_value_dual)
    n = control.n + treatment.n
    reason = (
        "missing_arm"
        if not control.n or not treatment.n
        else "insufficient_arm_observations"
        if min(control.n, treatment.n) < 2
        else "before_declared_start"
        if min(control.n, treatment.n) < declaration.start_count
        else "zero_arm_variance"
        if not control.scatter[0][0] or not treatment.scatter[0][0]
        else None
    )
    if reason is not None:
        return AsymptoticMeanSet(
            components=(MeanSetComponent(lower=None, upper=None),),
            alpha=alpha,
            alternative=alternative,
            count=n,
            k=None,
            available=False,
            reason=reason,
        )
    vc = control.scatter[0][0] / control.n**2
    vt = treatment.scatter[0][0] / treatment.n**2
    mc, mt = control.mean[0], treatment.mean[0]
    k = count_boundary(n, level, declaration.rho)
    components = _component_for_quadratic(mc * mc - k * vc, -2 * mt * mc, mt * mt - k * vt)
    return AsymptoticMeanSet(
        components=one_sided_components(
            components, alternative, control_value=mc, treatment_value=mt
        ),
        alpha=alpha,
        alternative=alternative,
        count=n,
        k=k,
        available=True,
        estimator_contrast=mt - (1 + null_lift) * mc,
        estimator_variance=vt + (1 + null_lift) ** 2 * vc,
    )
